"""Omega v2 — the analysis engine.

Original six-model ensemble, each leg wired to a real data source:

    A  Monte-Carlo GBM        risk-neutral + engine-tilted paths, antithetic
    B  BSM Greeks / IV        provider Greeks, IV rank, variance risk premium
    C  Order-flow proxy       VWAP / RVOL / OBV *plus the live option tape*
    D  Technical confluence   EMA stack, RSI+divergence, MACD, ADX/DI, Bollinger,
                              Stoch, ATR, 52-week position, RS vs SPY
    E  NLP catalyst           real newsrooms + SEC EDGAR + Stocktwits
    H  Historical statistics  block bootstrap + fitted fat tails, and the
                              stock's own last-6-month price action

New in v2: a full **strike table** for the coming expiry.  Every strike gets a
probability of profit (with a 95% interval), the probability of finishing ITM,
the probability of *touching* the strike during the week, the same numbers
measured empirically over the last six months, an expectancy, a reliability
score and a plain-Thai explanation of why that strike is (or is not) worth it.
"""
from __future__ import annotations

import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import data, news, quant, sim
from .util import (cache, clean, clip, env_float, env_int, env_list, http, log,
                   now_iso, nz)

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
RF = env_float("RISK_FREE", 0.045)
DEFAULT_WATCHLIST = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "TSLA", "AMD", "PLTR",
                     "SOFI", "INTC", "MU", "COIN", "MARA", "SNAP", "F"]
WATCHLIST = env_list("WATCHLIST", DEFAULT_WATCHLIST)
MIN_VOL = env_int("MIN_VOL", 100)
MIN_OI = env_int("MIN_OI", 200)
MAX_PREMIUM_PCT = env_float("MAX_PREMIUM_PCT", 0.030)
MAX_PREMIUM_ABS = env_float("MAX_PREMIUM_ABS", 5.0)
MAX_SPREAD = env_float("MAX_SPREAD", 0.25)
MIN_DTE = env_int("MIN_DTE", 1)
MAX_DTE = env_int("MAX_DTE", 21)
TARGET_DTE = env_int("TARGET_DTE", 7)
SIMS = env_int("SIMS", 4000)
HIST_WINDOW = env_int("HIST_WINDOW", 126)      # ~6 months of trading days
STRIKE_BAND = env_float("STRIKE_BAND", 0.15)   # +/- 15% of spot
# Cap how much per-strike detail is retained for the whole watchlist: every
# strike keeps its explanation strings, and 15 tickers x 20 strikes is enough
# JSON to matter on a small instance.
MAX_STRIKES = env_int("MAX_STRIKES", 18)
REFRESH_SEC = env_int("REFRESH_SEC", 180)
# Each scan worker runs its own small thread pool, so this number multiplies.
# 2 keeps a 512MB free instance comfortable; the scan is network-bound, so the
# wall-clock cost of parallelism here is small compared with the memory saved.
SCAN_WORKERS = env_int("SCAN_WORKERS", 2)
WEIGHTS = {"A": 0.16, "B": 0.12, "C": 0.16, "D": 0.22, "E": 0.14, "H": 0.20}
TARGET_DELTA = 0.40


def _seed(sym: str) -> int:
    return (abs(hash(sym)) % 100000) + int(time.time() // 900)


# --------------------------------------------------------------------------- #
# signals (models C, D, E)
# --------------------------------------------------------------------------- #
def tech_signal(daily: pd.DataFrame, spy: pd.DataFrame | None = None):
    if daily.empty or len(daily) < 60:
        return 0.0, {}
    c, h, l, v = daily["Close"], daily["High"], daily["Low"], daily["Volume"]

    def stack(s):
        a, b, d = quant.ema(s, 9).iloc[-1], quant.ema(s, 21).iloc[-1], quant.ema(s, 50).iloc[-1]
        return 1.0 if a > b > d else (-1.0 if a < b < d else 0.0)

    wk = c.resample("W").last().dropna()
    wk_sig = stack(wk) if len(wk) > 50 else (1.0 if quant.ema(wk, 9).iloc[-1] > quant.ema(wk, 21).iloc[-1] else -1.0)
    r = quant.rsi(c)
    r_now = float(r.iloc[-1])
    w, rw = c.iloc[-12:], r.iloc[-12:]
    div = 0.0
    if w.iloc[-1] <= w.min() * 1.005 and rw.iloc[-1] > rw.min() + 2:
        div = 1.0
    if w.iloc[-1] >= w.max() * 0.995 and rw.iloc[-1] < rw.max() - 2:
        div = -1.0
    _, _, hist = quant.macd(c)
    m_sig = 0.5 * np.sign(hist.iloc[-1]) + 0.5 * np.sign(hist.iloc[-1] - hist.iloc[-2])
    rsi_sig = float(np.clip((r_now - 50) / 25, -1, 1)) * 0.6 + 0.4 * div
    adx_s, pdi, mdi = quant.adx_di(h, l, c)
    adx = float(adx_s.iloc[-1])
    a = quant.atr(h, l, c, 14)
    atr_pct = float(a.iloc[-1] / c.iloc[-1] * 100)
    ma, sd, up_b, lo_b = quant.bollinger(c, 20)
    pb = float(((c - ma) / (4 * sd.replace(0, np.nan)) + 0.5).iloc[-1])
    bw = (4 * sd / ma).dropna()
    squeeze = bool(len(bw) > 120 and bw.iloc[-1] <= bw.iloc[-120:].quantile(0.2))
    st = float(quant.stoch_k(h, l, c, 14).iloc[-1])
    hi52, lo52 = float(c.iloc[-252:].max()), float(c.iloc[-252:].min())
    pos52 = float((c.iloc[-1] - lo52) / max(hi52 - lo52, 1e-9) * 100)
    mom21 = float(c.iloc[-1] / c.iloc[-22] - 1) if len(c) > 22 else 0.0
    mom63 = float(c.iloc[-1] / c.iloc[-64] - 1) if len(c) > 64 else 0.0
    e9, e21, e50 = float(quant.ema(c, 9).iloc[-1]), float(quant.ema(c, 21).iloc[-1]), float(quant.ema(c, 50).iloc[-1])
    rs = 0.0
    if spy is not None and not spy.empty and len(spy) > 64:
        try:
            sc = spy["Close"].reindex(c.index).ffill()
            rs = float((c.iloc[-1] / c.iloc[-64] - 1) - (sc.iloc[-1] / sc.iloc[-64] - 1))
        except Exception:
            rs = 0.0
    sig = 0.26 * stack(c) + 0.14 * wk_sig + 0.22 * rsi_sig + 0.20 * m_sig
    sig = sig * (0.8 + 0.2 * min(adx / 25, 1)) + 0.10 * float(np.sign(pdi.iloc[-1] - mdi.iloc[-1]))
    sig += 0.06 * float(np.clip(mom21 * 8, -1, 1)) + 0.04 * float(np.clip(mom63 * 4, -1, 1))
    sig += 0.05 * float(np.clip(rs * 6, -1, 1))
    if (pb > 1.05 and sig > 0) or (pb < -0.05 and sig < 0):
        sig *= 0.7
    info = dict(rsi=round(nz(r_now, 50.0), 1), macd_hist=round(nz(float(hist.iloc[-1])), 3), rsi_div=int(div),
                adx=round(nz(adx), 1), pb=round(nz(pb, 0.5), 2), stoch=round(nz(st, 50.0), 1),
                atr_pct=round(nz(atr_pct), 2),
                squeeze=squeeze, pos52=round(nz(pos52, 50.0), 0), mom21=round(nz(mom21) * 100, 1),
                mom63=round(nz(mom63) * 100, 1), rs_spy=round(nz(rs) * 100, 1),
                pdi=round(nz(float(pdi.iloc[-1])), 1), mdi=round(nz(float(mdi.iloc[-1])), 1),
                ema9=round(nz(e9), 2), ema21=round(nz(e21), 2), ema50=round(nz(e50), 2),
                stack=int(stack(c)), bb_upper=round(nz(float(up_b.iloc[-1])), 2),
                bb_lower=round(nz(float(lo_b.iloc[-1])), 2))
    return float(np.clip(nz(sig), -1, 1)), info


def flow_signal(daily: pd.DataFrame, intra: pd.DataFrame, chain: dict | None):
    if daily.empty or len(daily) < 25:
        return 0.0, {}
    c, v = daily["Close"], daily["Volume"]
    vmean = v.iloc[-21:-1].mean()
    rvol = float(v.iloc[-1] / vmean) if vmean and vmean > 0 else 1.0
    vwap, price = float(c.iloc[-1]), float(c.iloc[-1])
    if intra is not None and not intra.empty:
        day = intra[intra.index.date == intra.index[-1].date()]
        if len(day) < 3:
            day = intra
        try:
            tp = (day["High"] + day["Low"] + day["Close"]) / 3
            vwap = float((tp * day["Volume"]).sum() / max(day["Volume"].sum(), 1))
            price = float(day["Close"].iloc[-1])
        except Exception:
            pass
    rvol = nz(rvol, 1.0)
    vwap, price = nz(vwap), nz(price)
    vwap_sig = float(np.clip((price / vwap - 1) * 100, -1, 1)) if vwap else 0.0
    ret = nz(float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 2 else 0.0)
    o = quant.obv(c, v)
    p_slope, o_slope = c.iloc[-1] - c.iloc[-11], o.iloc[-1] - o.iloc[-11]
    p_slope, o_slope = nz(p_slope), nz(o_slope)
    if p_slope > 0 and o_slope < 0:
        div = -1.0
    elif p_slope < 0 and o_slope > 0:
        div = 1.0
    else:
        div = 0.5 * float(np.sign(o_slope))

    pc_vol = pc_oi = None
    intra_pc = None
    intra_trend = None
    unusual = 0
    net_prem_sig = 0.0
    hot = []
    if intra is not None and not intra.empty and "Calls" in intra.columns:
        try:
            day = intra[intra.index.date == intra.index[-1].date()]
            if len(day) < 10:
                day = intra
            cv, pv = float(day["Calls"].sum()), float(day["Puts"].sum())
            if cv + pv > 0:
                intra_pc = pv / cv if cv > 0 else 4.0
                # is the call bid accelerating into the close?
                half = max(len(day) // 2, 1)
                a, b = day.iloc[:half], day.iloc[half:]
                ta = (a["Calls"].sum() - a["Puts"].sum()) / max(a["Calls"].sum() + a["Puts"].sum(), 1)
                tb = (b["Calls"].sum() - b["Puts"].sum()) / max(b["Calls"].sum() + b["Puts"].sum(), 1)
                intra_trend = float(np.clip((tb - ta) * 2.5, -1, 1))
        except Exception:
            pass
    if chain and chain.get("rows") and chain.get("spot"):
        spot = chain["spot"]
        cv = pv = co = po = 0.0
        for r in chain["rows"]:
            if not (0.9 * spot <= r["strike"] <= 1.1 * spot):
                continue
            if r["kind"] == "CALL":
                cv += r["volume"]
                co += r["oi"]
            else:
                pv += r["volume"]
                po += r["oi"]
            if r["oi"] > 0 and r["volume"] > 2.0 * r["oi"] and r["volume"] > 200:
                unusual += 1
                hot.append(dict(strike=r["strike"], kind=r["kind"], volume=int(r["volume"]), oi=int(r["oi"]),
                                expiry=r["expiry"]))
            mid = (r["bid"] + r["ask"]) / 2 if (r["bid"] and r["ask"]) else (r.get("last") or 0.0)
            if mid > 0 and r["volume"] > 0:
                sgn = 1.0 if r["kind"] == "CALL" else -1.0
                net_prem_sig += sgn * mid * r["volume"]
        pc_vol = (pv / cv) if cv > 0 else None
        pc_oi = (po / co) if co > 0 else None
        net_prem_sig = float(np.clip(net_prem_sig / max(cv + pv, 1) / 2.0, -1, 1))
    opt = 0.0
    if pc_vol is not None:
        opt += float(np.clip((1.0 - pc_vol) / 0.5, -1, 1)) * 0.35
    if pc_oi is not None:
        opt += float(np.clip((1.0 - pc_oi) / 0.8, -1, 1)) * 0.15
    if intra_pc is not None:
        opt += float(np.clip((1.0 - intra_pc) / 0.6, -1, 1)) * 0.20
    if intra_trend is not None:
        opt += intra_trend * 0.15
    opt += net_prem_sig * 0.15
    opt += float(np.clip(unusual / 8.0, 0, 1)) * 0.10 * float(np.sign(net_prem_sig or 1))
    sig = 0.30 * vwap_sig + 0.22 * float(np.sign(ret)) * min(rvol / 2, 1) + 0.20 * div + 0.28 * opt
    info = dict(rvol=round(nz(rvol, 1.0), 2), vwap=round(nz(vwap), 2), div=int(div), price=round(nz(price), 2),
                pc_vol=round(nz(pc_vol), 2) if pc_vol is not None else None,
                pc_oi=round(nz(pc_oi), 2) if pc_oi is not None else None,
                intra_pc=round(nz(intra_pc), 2) if intra_pc is not None else None,
                intra_trend=round(nz(intra_trend), 2) if intra_trend is not None else None,
                unusual=int(unusual), net_prem=round(nz(net_prem_sig), 2),
                hot=sorted(hot, key=lambda x: -x["volume"])[:5])
    return float(np.clip(nz(sig), -1, 1)), info


def news_signal(ns: dict):
    s = nz(ns.get("score"), 0.0)
    f = nz(ns.get("filing_score"), 0.0)
    soc = nz((ns.get("social") or {}).get("score"), 0.0)
    checks = ns.get("checks") or []
    verified = sum(1 for c in checks if c["verdict"] in ("ยืนยันแล้ว", "ยืนยันบางส่วน"))
    sig = 0.62 * s + 0.22 * (f * math.copysign(1, s if abs(s) > 0.05 else 1)) + 0.16 * soc
    if verified:
        sig *= 1.0 + min(0.15, 0.05 * verified)
    return float(np.clip(nz(sig), -1, 1)), dict(verified=verified)


# --------------------------------------------------------------------------- #
# earnings calendar (best effort, cached)
# --------------------------------------------------------------------------- #
def earnings_map(days: int = MAX_DTE + 7) -> dict:
    def load():
        out = {}
        dates = []
        d = datetime.now(timezone.utc).date()
        while len(dates) < min(days, 15):
            d += timedelta(days=1)
            if d.weekday() < 5:
                dates.append(d.isoformat())
        for ds in dates:
            j = http.json(f"https://api.nasdaq.com/api/calendar/earnings?date={ds}",
                          headers={"Accept": "application/json"}, timeout=(5, 12))
            if not j:
                continue
            for row in ((j.get("data") or {}).get("rows") or []):
                sym = (row.get("symbol") or "").upper()
                if sym:
                    out[sym] = ds
        return out or None

    return cache.wrap("earn:map", 21600, load) or {}


# --------------------------------------------------------------------------- #
# text: why this strike?
# --------------------------------------------------------------------------- #
def _baht_pct(x):
    return f"{x:+.1f}%" if x is not None else "-"


def _mag_pct(x):
    """Unsigned magnitude, for sentences that already say 'above'/'below'."""
    return f"{abs(x):.1f}%" if x is not None else "-"


def _pct_txt(x):
    return f"{x:+.0f}%" if x is not None else "ไม่มีข้อมูล"


def strike_reasons(row: dict, ctx: dict) -> list:
    """Plain-Thai explanation for one strike."""
    out = []
    up = row["kind"] == "CALL"
    side = "ขึ้น" if up else "ลง"
    d = row["dist_pct"]
    # 1. moneyness + distance vs expected move
    em = ctx["em_pct"]
    itm = (d < 0) if up else (d > 0)
    if itm:
        if up:
            out.append(f"Strike นี้เป็น In-the-Money อยู่แล้ว (ราคาหุ้น ${ctx.get('spot', 0):.2f} สูงกว่า strike "
                       f"{_mag_pct(d)}) จึงมีมูลค่าคงเหลือและโอกาสชนะสูงกว่าสัญญา OTM "
                       f"แต่พรีเมียมก็แพงกว่าตามไปด้วย")
        else:
            out.append(f"Strike นี้เป็น In-the-Money อยู่แล้ว (ราคาหุ้น ${ctx.get('spot', 0):.2f} ต่ำกว่า strike "
                       f"{_mag_pct(d)}) จึงมีมูลค่าคงเหลือและโอกาสชนะสูงกว่าสัญญา OTM "
                       f"แต่พรีเมียมก็แพงกว่าตามไปด้วย")
    elif abs(d) <= em:
        out.append(f"Strike อยู่ห่างจากราคาปัจจุบัน {_mag_pct(d)} ซึ่งอยู่ในระยะการแกว่งที่ตลาดคาดไว้ ±{em:.1f}% "
                   f"(Expected Move) จึงมีโอกาสถูกแตะสูง")
    elif abs(d) <= 2 * em:
        out.append(f"Strike ห่าง {_mag_pct(d)} = ประมาณ {abs(d)/max(em,0.1):.1f} เท่าของ Expected Move ±{em:.1f}% "
                   f"ต้องมีข่าวหรือโมเมนตัมหนุนจึงจะไปถึง")
    else:
        out.append(f"Strike ห่างไกล {_mag_pct(d)} (เกิน 2 เท่าของ Expected Move ±{em:.1f}%) "
                   f"ต้องมีการเคลื่อนไหวรุนแรงผิดปกติ")
    # 2. model vs history (the real edge test)
    # Compare *profit at expiry*, not the chance of merely touching the strike:
    # an in-the-money strike is "touched" almost by definition, so the touch
    # numbers agree trivially and the sentence reads as meaningless agreement.
    # The honest comparison is the historical win rate against the model's
    # probability of profit, which is what decides whether the strike has edge.
    ht, pt = row.get("hist_win"), row.get("pop")
    if ht is not None and pt is not None:
        gap = ht - pt
        if gap >= 5:
            out.append(f"สถิติจริง 6 เดือน: สัญญาแบบนี้ชนะเมื่อหมดอายุ {ht:.0f}% ของรอบ "
                       f"สูงกว่าที่โมเดลประเมิน ({pt:.0f}%) → ฝั่ง {row['kind']} ได้เปรียบเชิงสถิติ "
                       f"เพราะตลาดตั้งราคาเผื่อความเสี่ยงไว้มากกว่าที่เกิดจริง")
        elif gap <= -5:
            out.append(f"สถิติจริง 6 เดือน: สัญญาแบบนี้ชนะเพียง {ht:.0f}% ต่ำกว่าที่โมเดลให้ ({pt:.0f}%) "
                       f"→ ตลาดอาจตั้งราคาเปิดเผื่อไว้มากเกินไป พรีเมียมแพงกว่าความเสี่ยงจริง")
        else:
            out.append(f"สถิติจริง 6 เดือน ({ht:.0f}%) กับที่โมเดลให้ ({pt:.0f}%) สอดคล้องกัน "
                       f"→ ความน่าจะเป็นนี้เชื่อถือได้พอสมควร")
    # 3. probability of finishing beyond breakeven
    hl = row.get("hist_win")
    if hl is not None and row.get("hist_n"):
        out.append(f"ซื้อสัญญานี้แล้วกำไรเมื่อหมดอายุ: สถิติ {ctx['months']} เดือนที่ผ่านมาเทียบเท่า {hl:.0f}% "
                   f"(จาก {row['hist_n']} รอบ) · โมเดลให้ {row['pop']:.0f}% (ช่วง {row['pop_lo']:.0f}–{row['pop_hi']:.0f}%)")
    # 4. IV: skew of this strike + overall IV regime
    vrp = ctx.get("vrp", 0)
    ivx = row.get("iv_vs_atm")
    if ivx is not None:
        if ivx > 3:
            out.append(f"IV ของสัญญานี้ {row['iv']:.0f}% สูงกว่า ATM {ivx:+.0f} จุด — พรีเมียมบวมจากความผันผวน "
                       f"เฉพาะ strike นี้ (ซื้อแพงกว่าเพื่อนร่วมรุ่น)")
        elif ivx < -3:
            out.append(f"IV ของสัญญานี้ {row['iv']:.0f}% ต่ำกว่า ATM {ivx:.0f} จุด — ได้เปรียบฝั่งซื้อเมื่อเทียบทั้งสาย")
    out.append(f"ระดับ IV ของหุ้นทั้งตัวตอนนี้ {row['iv_tag']} (อันดับ {ctx.get('iv_rank', 50):.0f}/100 เทียบ 1 ปีที่ผ่านมา) "
               f"· ความผันผวนจริง 6 เดือน {ctx.get('sigma_hist', 0):.0f}% · ส่วนต่าง {vrp:+.1f} จุด → "
               + ("IV สูงกว่าความผันผวนจริง ออปชันโดยรวมค่อนไปทางแพง ต้องชนะให้มากกว่าพรีเมียมที่จ่าย"
                  if vrp > 4 else
                  "IV ต่ำกว่าความผันผวนจริง ออปชันโดยรวมค่อนไปทางถูก" if vrp < -2 else
                  "IV ใกล้เคียงความผันผวนจริง ราคาออปชันถือว่าเป็นธรรม"))
    # 5. liquidity / execution
    if row.get("liquid"):
        out.append(f"สภาพคล่องผ่านเกณฑ์: ปริมาณ {row['volume']:,} สัญญา · OI {row['oi']:,} · Spread {row['spread_pct']:.1f}%")
    else:
        out.append(f"สภาพคล่องต่ำ: ปริมาณ {row['volume']:,} · OI {row['oi']:,} · Spread {row['spread_pct']:.1f}% "
                   f"— เข้าออกยาก ราคาไถลได้")
    # 6. directional evidence
    t, f = ctx.get("tech") or {}, ctx.get("flow") or {}
    bits = []
    if t:
        bits.append(f"RSI {t.get('rsi')} · MACD hist {t.get('macd_hist')} · ADX {t.get('adx')} · Stoch {t.get('stoch')}")
    if f:
        bits.append(f"ราคาเทียบ VWAP {'เหนือ' if (f.get('price') or 0) >= (f.get('vwap') or 0) else 'ใต้'} "
                    f"· RVOL {f.get('rvol')} · P/C volume {f.get('pc_vol')}")
    if bits:
        lean = "สนับสนุน" if (ctx["signal"] > 0) == up else "สวนทาง"
        out.append(f"อินดิเคเตอร์: {' | '.join(bits)} → {lean}ฝั่ง{side}")
    # 7. greeks reality check
    out.append(f"Greeks: Δ {row['delta']:.2f} · Θ {row['theta']:+.3f}/วัน ({row['theta_bleed']:.1f}% ของพรีเมียม/วัน) "
               f"· Vega {row['vega']:.3f} · จุดคุ้มทุน {row['breakeven']:.2f}")
    return out


def strike_risks(row: dict, ctx: dict) -> list:
    out = []
    if ctx.get("earnings"):
        out.append(f"⚠ งบการเงินออกวันที่ {ctx['earnings']} ก่อนหมดอายุ → IV crush เสี่ยงสูงแม้ทิศทางถูก")
    if row.get("spread_pct", 0) > 12:
        out.append(f"⚠ Spread กว้าง {row['spread_pct']:.1f}% ต้นทุนแฝงสูง")
    if row.get("p_zero") is not None and row["p_zero"] > 70:
        out.append(f"⚠ โอกาสหมดค่าเป็นศูนย์ {row['p_zero']:.0f}% (ถ้าถือถึงหมดอายุ)")
    if row.get("theta_bleed", 0) > 6:
        out.append(f"⚠ Theta ไล่พรีเมียม {row['theta_bleed']:.1f}% ต่อวัน")
    if row.get("reliability", 100) < 55:
        out.append(f"⚠ ความน่าเชื่อถือของข้อมูลต่ำ ({row['reliability']:.0f}/100) — ตัวอย่างสถิติ/สภาพคล่องน้อย")
    if ctx.get("hist_n", 0) and ctx["hist_n"] < 60:
        out.append(f"⚠ ตัวอย่างสถิติ 6 เดือนมีเพียง {ctx['hist_n']} รอบ")
    return out


# --------------------------------------------------------------------------- #
# strike table
# --------------------------------------------------------------------------- #
def build_strike(sym: str, row_c: dict, spot: float, kind: str, dte: float, n_days: int,
                 simset: dict, dist: dict, close: np.ndarray, atm_iv: float, sigma_hist: float,
                 T: float, ctx: dict) -> dict | None:
    spot = nz(spot)
    strike = nz(row_c.get("strike"))
    atm_iv = float(np.clip(nz(atm_iv, 0.35), 0.03, 4.0))
    T = max(nz(T, 7 / 365.0), 1e-5)
    if spot <= 0 or strike <= 0:
        return None
    bid, ask = nz(row_c.get("bid")), nz(row_c.get("ask"))
    last = nz(row_c.get("last"))
    premium = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
    if premium <= 0:
        return None
    iv = nz(row_c.get("iv")) / 100.0 if nz(row_c.get("iv")) > 1 else 0.0
    if not (0.02 <= iv <= 5.0):
        iv = quant.implied_vol(premium, spot, strike, T, kind, RF) or 0.0
    if not (0.02 <= iv <= 5.0):
        iv = atm_iv
    iv = float(np.clip(nz(iv, atm_iv), 0.02, 5.0))
    mu_rn = RF
    r = sim.strike_row(simset, spot, strike, kind, premium, T, iv, mu_rn)
    hq = sim.hist_query(dist, kind, strike, spot, premium, n_days, close)
    r.update(hq)

    spread = (ask - bid) / premium if (bid > 0 and ask > 0) else 1.0
    spread = float(np.clip(nz(spread, 1.0), 0.0, 3.0))
    vol_c, oi_c = max(int(nz(row_c.get("volume"))), 0), max(int(nz(row_c.get("oi"))), 0)
    liq = float(np.clip((math.log1p(vol_c) * 1.1 + math.log1p(oi_c) * 0.9
                         + math.log1p(max(nz(row_c.get("bid_size")), 0) + max(nz(row_c.get("ask_size")), 0)) * 0.4) / 12.0
                        * (1.0 - min(spread, 1.0) * 0.8), 0, 1))
    cheap = premium <= max(spot * MAX_PREMIUM_PCT, 0.01) and premium <= MAX_PREMIUM_ABS
    liquid = (vol_c >= MIN_VOL or oi_c >= MIN_OI) and spread <= MAX_SPREAD
    g = quant.bsm(spot, strike, T, iv, kind, RF)
    for k in ("delta", "gamma", "theta", "vega"):
        try:
            if row_c.get(k) is not None and abs(float(row_c[k])) > 1e-9:
                g[k] = float(row_c[k])
        except Exception:
            pass
    for k in ("delta", "gamma", "theta", "vega"):
        g[k] = nz(g.get(k))
    theta_bleed = abs(g["theta"]) / max(premium, 0.01) * 100
    ivr_ctx = nz(ctx.get("iv_rank"), 50.0)

    # --- score ---------------------------------------------------------------
    # Weights sum to 1.00.  For a deep OTM or deeply ITM contract the "chance"
    # term and the "value" term fight each other, so trades are only ranked
    # inside the band where a human would actually buy (0.10 <= |delta| <= 0.85);
    # outside that band the score is deliberately discounted instead of boosted.
    #
    # Note on EV: buying a long single-leg option has a *negative* expectancy by
    # construction — you pay the spread and rent volatility.  Scoring EV on an
    # absolute scale therefore pins every strike near the same floor and nothing
    # can ever rank well.  What actually matters is which strike gives the least
    # bad (or genuinely positive) expectancy *relative to its siblings*, so EV is
    # scored as a percentile inside the candidate set by ``rank_context`` below.
    liq_score = liq * 100
    cand_ev = r.get("ev_pct")
    if cand_ev is None or not math.isfinite(cand_ev):
        cand_ev = 0.0
        r["ev_pct"] = None
    ev_score = float(np.clip(50 + 60 * math.tanh(cand_ev / 100.0 * 2.5), 0, 100))
    pop_score = r["pop"]
    hist_edge = None
    if r.get("hist_win") is not None and r.get("hist_n", 0) >= 40:
        hist_edge = r["hist_win"] - r["pop"]
    edge_score = float(np.clip(50 + (hist_edge or 0) * 2.0, 0, 100))
    dir_score = float(np.clip(50 + 50 * ctx["signal"] * (1 if kind == "CALL" else -1), 0, 100))
    iv_score = float(np.clip(100 - ivr_ctx - max(0.0, ctx.get("vrp", 0)) * 1.5 - min(theta_bleed * 3, 35), 0, 100))
    omega = (0.35 * pop_score + 0.28 * ev_score + 0.12 * edge_score
             + 0.12 * liq_score + 0.08 * dir_score + 0.05 * iv_score)
    ad = abs(float(g["delta"]))
    if ad < 0.15 or ad > 0.85:
        omega *= 0.90          # outside the tradable delta band
    if not cheap:
        omega *= 0.93
    if not liquid:
        omega *= 0.85
    if ctx.get("earnings"):
        omega *= 0.94
    omega = float(np.clip(nz(omega, 0.0), 0, 100))
    verdict = ("ULTIMATE" if omega >= 80 else "HIGH" if omega >= 68 else
               "WATCH" if omega >= 55 else "NONE")

    rel = 50.0
    rel += 8 if vol_c >= MIN_VOL else -6
    rel += 8 if oi_c >= MIN_OI else -6
    rel += 10 if spread <= 0.10 else (4 if spread <= 0.25 else -10)
    rel += 8 if r.get("hist_n", 0) >= 100 else (2 if r.get("hist_n", 0) >= 60 else -8)
    rel += 6 if nz(ctx.get("news_n")) >= 6 else 0
    rel += 5 if nz(simset.get("meta", {}).get("boot_n")) >= 250 else -6
    rel += 4 if ctx.get("iv_rank") is not None else 0
    reliability = float(np.clip(rel, 0, 100))

    out = dict(
        ticker=sym, kind=kind, strike=strike, dist_pct=round((strike / spot - 1) * 100, 2),
        premium=round(premium, 2), bid=round(bid, 2), ask=round(ask, 2),
        volume=int(vol_c), oi=int(oi_c), spread_pct=round(spread * 100, 1),
        cheap=bool(cheap), liquid=bool(liq >= 0.35 and liquid), premium_pct=round(premium / spot * 100, 2),
        iv=round(iv * 100, 1), iv_vs_atm=round((iv - atm_iv) * 100, 1),
        iv_tag="ถูก" if ivr_ctx < 30 else ("แพง" if ivr_ctx > 70 else "กลาง"),
        delta=round(float(g["delta"]), 4), gamma=round(float(g["gamma"]), 5),
        theta=round(float(g["theta"]), 4), vega=round(float(g["vega"]), 4),
        theta_bleed=round(nz(theta_bleed), 1), liquidity=round(liq, 3),
        omega=round(omega, 1), verdict=verdict, reliability=round(reliability, 1),
        hist_edge=round(hist_edge, 1) if hist_edge is not None else None,
        expiry=row_c.get("expiry"), dte=round(nz(dte), 1),
    )
    out.update(r)
    out["why"] = strike_reasons(out, ctx)
    out["risks"] = strike_risks(out, ctx)
    out["headline"] = (f"{kind} ${out['strike']:g} · หมดอายุ {out.get('expiry')} ({out['dte']:.0f} วัน) · "
                       f"พรีเมียม ${out['premium']} · โอกาสชนะ {nz(out.get('pop')):.0f}% "
                       f"(ช่วง {nz(out.get('pop_lo')):.0f}–{nz(out.get('pop_hi')):.0f}%)")
    return out


# --------------------------------------------------------------------------- #
# per-ticker analysis
# --------------------------------------------------------------------------- #
def analyze(sym: str) -> dict | None:
    sym = re.sub(r"[^A-Z]", "", sym.upper())[:6]
    if not sym:
        return None

    # The four slowest steps are independent network fetches, so run them
    # together instead of one after another.  On a free instance this is the
    # difference between a scan that finishes and one that appears to hang.
    # Every future also gets a hard deadline: a wedged socket must never be
    # able to block the whole scan, which is what kept /api/scan empty.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

    def take(fut, fallback, secs=45.0):
        try:
            return fut.result(timeout=secs)
        except FTimeout:
            log.warning("analyze %s: a provider took longer than %.0fs", sym, secs)
            return fallback
        except Exception:
            return fallback

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        f_daily = pool.submit(data.daily, sym)
        f_intra = pool.submit(data.intraday, sym)
        f_chain = pool.submit(data.chain, sym)
        f_name = pool.submit(data.company_name, sym)
        d = take(f_daily, None)
        intra = take(f_intra, None)
        ch = take(f_chain, None)
        name = take(f_name, "", 15.0)
    finally:
        # do not block on stragglers: the scan must keep moving even if one
        # worker thread is still stuck on a dead connection
        pool.shutdown(wait=False)

    if (d is None or d.empty) and not ch:
        return None
    spot = float(ch["spot"]) if (ch and ch.get("spot")) else (float(d["Close"].iloc[-1]) if not d.empty else 0.0)
    if not spot or not math.isfinite(spot) or spot <= 0:
        return None

    logret = np.log(d["Close"]).diff().dropna().values if (d is not None and len(d) > 30) else np.array([])
    if len(logret) > 504:
        logret = logret[-504:]
    if len(d):
        close = d["Close"].values.astype(float)
        high = d["High"].values.astype(float)
        low = d["Low"].values.astype(float)
    else:
        close = high = low = np.array([])

    hv_close = quant.vol_close(d["Close"], 30) if len(d) > 35 else 0.4
    hv_park = quant.vol_parkinson(d["High"], d["Low"], 30) if len(d) > 35 else hv_close
    hv_gk = quant.vol_garman_klass(d["Open"], d["High"], d["Low"], d["Close"], 30) if len(d) > 35 else hv_close
    sigma_hist = float(np.clip(np.mean([hv_close, hv_park, hv_gk]), 0.03, 3.0))

    # News runs on its own thread with a deadline: it fans out to a dozen
    # publisher feeds, and one unresponsive site must not hold up the ticker.
    from concurrent.futures import ThreadPoolExecutor as _TPE, TimeoutError as _FTO

    _nat = _TPE(max_workers=1)
    try:
        _fut = _nat.submit(news.for_ticker, sym, name)
        ns = _fut.result(timeout=60.0)
    except _FTO:
        log.warning("analyze %s: news took longer than 60s -- continuing without it", sym)
        ns = {}
    except Exception as ex:
        log.info("analyze %s: news failed (%s)", sym, type(ex).__name__)
        ns = {}
    finally:
        _nat.shutdown(wait=False)
    c_sig, c_info = flow_signal(d, intra, ch)
    d_sig, d_info = tech_signal(d, cache.wrap("d:SPY", 900, lambda: data.daily("SPY")))
    e_sig, e_info = news_signal(ns)
    # every signal is forced finite here: one NaN from a feed must not be able to
    # reach the drift (mu) and quietly turn a whole simulation into NaN
    c_sig, d_sig, e_sig = nz(c_sig), nz(d_sig), nz(e_sig)
    net = float(np.clip(0.34 * c_sig + 0.46 * d_sig + 0.20 * e_sig, -1, 1))
    net = nz(net)
    bias = "CALL" if net >= 0 else "PUT"
    emap = earnings_map()
    earn = emap.get(sym)

    atm_iv, iv30 = sigma_hist, None
    if ch and ch.get("rows"):
        pool = [r["iv"] / 100.0 for r in ch["rows"]
                if abs(r["strike"] - spot) <= spot * 0.015 and r.get("iv") and r["iv"] > 1]
        if pool:
            atm_iv = float(np.mean(pool))
        iv30 = ch.get("iv30")
        if iv30 and iv30 > 1:
            atm_iv = 0.5 * atm_iv + 0.5 * iv30 / 100.0
    atm_iv = float(np.clip(atm_iv, 0.03, 4.0))
    if len(logret) > 130:
        rolling = pd.Series(logret).rolling(30).std().dropna() * math.sqrt(252)
        hist = rolling.iloc[-252:].values
        ivr = float(np.clip((atm_iv - hist.min()) / max(hist.max() - hist.min(), 1e-9), 0, 1) * 100)
        ivp = float((hist < atm_iv).mean() * 100)
    else:
        ivr, ivp = 50.0, 50.0
    vrp = float((atm_iv - sigma_hist) * 100)

    base = dict(ticker=sym, name=name, spot=round(spot, 2), spot_source="CBOE (delayed)" if ch and ch.get("spot") else "Yahoo",
                chain_ts=(ch or {}).get("ts"), signal=round(net, 3), bias=bias,
                tech=d_info, flow=c_info,
                tech_score=round(float(np.clip(50 + 50 * d_sig, 0, 100)), 1),
                flow_score=round(float(np.clip(50 + 50 * c_sig, 0, 100)), 1),
                news_score=round(float(np.clip(50 + 50 * e_sig, 0, 100)), 1),
                news={k: v for k, v in e_info.items()},
                news_detail=dict(items=ns.get("items", [])[:12], sources=ns.get("sources", {}),
                                 depth=ns.get("depth", {}), checks=ns.get("checks", []),
                                 n=ns.get("n", 0), engine=ns.get("engine"), social=ns.get("social", {}),
                                 filings=ns.get("filings", [])[:6], note=ns.get("note", "")),
                atm_iv=round(atm_iv * 100, 1), iv30=round(iv30, 1) if iv30 else None,
                iv_rank=round(ivr, 1), iv_pct=round(ivp, 1), vrp=round(vrp, 1),
                sigma_hist=round(sigma_hist * 100, 1),
                earnings=earn, calls=[], puts=[], error=None)

    if not (ch and ch.get("rows")):
        base.update(error="ไม่มีข้อมูลออปชัน (CBOE) สำหรับหุ้นนี้", verdict="NONE", omega=0.0)
        return base

    # ---- pick the expiry ---------------------------------------------------
    today = datetime.now(timezone.utc).date()
    exps = {}
    for r in ch["rows"]:
        try:
            dt = (datetime.strptime(r["expiry"], "%Y-%m-%d").date() - today).days
        except Exception:
            continue
        if MIN_DTE <= dt <= MAX_DTE:
            exps[r["expiry"]] = dt
    base["expiries"] = sorted([dict(expiry=e, dte=v) for e, v in exps.items()], key=lambda x: x["dte"])
    if not exps:
        base.update(error=f"ไม่มีออปชันหมดอายุในช่วง {MIN_DTE}-{MAX_DTE} วัน", verdict="NONE", omega=0.0)
        return base
    exp = min(exps, key=lambda e: abs(exps[e] - TARGET_DTE) + (0 if 2 <= exps[e] <= 12 else 4))
    dte = exps[exp]
    T = max(dte / 365.0, 1e-4)
    n_days = max(int(round(dte * 5 / 7)), 1)

    sigma_mix = float(np.clip(0.65 * atm_iv + 0.35 * sigma_hist, 0.03, 4.0))
    mu = RF + 0.10 * net
    simset = sim.simulate(spot, T, n_days, sigma_mix, mu, logret, sims=SIMS, seed=_seed(sym) % 2_000_000_000)
    dist = sim.hist_distributions(close, high, low, n_days, HIST_WINDOW)
    months = 6
    em = quant.expected_move(spot, sigma_mix, T)

    ctx = dict(signal=net, em_pct=round(em / spot * 100, 2), n_days=n_days, months=months,
               vrp=vrp, iv_rank=ivr, earnings=earn, tech=d_info, flow=c_info, spot=spot,
               sigma_hist=sigma_hist * 100,
               news_n=ns.get("n", 0), hist_n=(dist["n"] if dist else 0), dte=dte, expiry=exp)

    # ---- strike window -----------------------------------------------------
    lo, hi = spot * (1 - STRIKE_BAND), spot * (1 + STRIKE_BAND)
    rows = [r for r in ch["rows"] if r["expiry"] == exp and lo <= r["strike"] <= hi]
    if not rows:
        rows = [r for r in ch["rows"] if r["expiry"] == exp]
    by_strike = {}
    for r in rows:
        r = dict(r)
        r["expiry"] = exp
        by_strike.setdefault(r["strike"], {})[r["kind"]] = r
    all_k = sorted(by_strike, key=lambda k: abs(k - spot))
    keep = set(all_k[:MAX_STRIKES // 2] + sorted([k for k in all_k if k <= spot])[-6:]
               + sorted([k for k in all_k if k > spot])[:6])
    strikes = sorted(keep)
    calls, puts = [], []
    for k in strikes:
        for kind, bucket in (("CALL", calls), ("PUT", puts)):
            r = by_strike[k].get(kind)
            if not r:
                continue
            st = build_strike(sym, r, spot, kind, dte, n_days, simset, dist, close, atm_iv, sigma_hist, T, ctx)
            if st:
                bucket.append(st)
    calls.sort(key=lambda x: x["strike"])
    puts.sort(key=lambda x: x["strike"])

    # ---- selection -----------------------------------------------------------
    # Two different questions must get two different answers:
    #   * which strike has the highest *chance* of winning  -> top_call/top_put
    #   * which strike is the best *trade* (chance + value + liquidity + history)
    #     -> best_call/best_put, chosen only among strikes a human could actually
    #     buy (cheap, liquid, sensible delta)
    def tradable(x):
        return bool(x["cheap"]) and bool(x["liquid"]) and 0.10 <= abs(x["delta"]) <= 0.85

    # ---- relative re-scoring ------------------------------------------------
    # Every strike was scored against absolute yardsticks, which compresses the
    # whole table into a narrow band: long options always have negative EV and
    # POP rarely clears 55, so nothing ever looks worth buying.  What a trader
    # actually asks is "of the strikes on this expiry, which is best?", so here
    # we re-rank each direction by EV and by POP *within the candidate set*,
    # using ranks (0-100) instead of dollar values.  This is calibration, not
    # opinion: it cannot make a bad strike good, it only spreads the field so the
    # genuine standouts surface.
    def _rescore(group: list) -> None:
        if not group:
            return
        evs = [nz(x.get("ev_pct")) for x in group]
        pops = [nz(x.get("pop")) for x in group]
        edges = [nz(x.get("hist_edge")) for x in group if x.get("hist_edge") is not None]
        n = len(group)

        def pct_rank(vals, v):
            if not vals:
                return 50.0
            below = sum(1 for x in vals if x <= v)
            return float(below / len(vals) * 100.0)

        for x in group:
            ev_pct_rank = pct_rank(evs, nz(x.get("ev_pct")))
            pop_pct_rank = pct_rank(pops, nz(x.get("pop")))
            edge_pct_rank = pct_rank(edges, nz(x.get("hist_edge"))) if edges and x.get("hist_edge") is not None else 50.0
            liq_score = float(np.clip(nz(x.get("liquidity")) * 100, 0, 100))
            iv_score = float(np.clip(100 - nz(x.get("iv_rank"), 50.0), 0, 100))
            dir_score = float(np.clip(50 + 50 * nz(net) * (1 if x["kind"] == "CALL" else -1), 0, 100))
            x["omega"] = round(float(np.clip(
                0.34 * pop_pct_rank + 0.26 * ev_pct_rank + 0.14 * edge_pct_rank
                + 0.12 * liq_score + 0.08 * dir_score + 0.06 * iv_score, 0, 100)), 1)
            ad = abs(nz(x.get("delta")))
            if ad < 0.15 or ad > 0.85:
                x["omega"] = round(x["omega"] * 0.90, 1)
            if not x.get("cheap"):
                x["omega"] = round(x["omega"] * 0.93, 1)
            if not x.get("liquid"):
                x["omega"] = round(x["omega"] * 0.85, 1)
            if ctx.get("earnings"):
                x["omega"] = round(x["omega"] * 0.94, 1)
            x["rank_in_group"] = 1 + sum(1 for y in group if y["omega"] > x["omega"])
            x["verdict"] = ("ULTIMATE" if x["omega"] >= 72 else "HIGH" if x["omega"] >= 62 else
                            "WATCH" if x["omega"] >= 52 else "NONE")

    _rescore(calls)
    _rescore(puts)

    t_calls = [x for x in calls if tradable(x)]
    t_puts = [x for x in puts if tradable(x)]

    # Guard against a "best deal" that is cheap only because it barely ever wins.
    # A strike may only be called the best trade if its win chance is also
    # respectable relative to the rest of the table; otherwise we fall back to
    # the tradable set ranked by omega, and finally to the whole side.
    def pick(group, others):
        pool = group or others
        if not pool:
            return None
        # the chance threshold is measured against the *whole* side, not just the
        # tradable subset -- otherwise a table with no cheap-and-liquid strikes
        # falls back to the subset and a 20%-POP lottery ticket gets crowned
        ceiling = max(nz(x.get("pop")) for x in (others or pool))
        floor = max(15.0, ceiling * 0.60)
        strong = [x for x in pool if nz(x.get("pop")) >= floor]
        if not strong:
            strong = [x for x in (others or pool) if nz(x.get("pop")) >= floor]
        return max(strong or pool, key=lambda x: (x["omega"], nz(x.get("pop"))))

    best_call = pick(t_calls, calls) if calls else None
    best_put = pick(t_puts, puts) if puts else None
    top_call = max(calls, key=lambda x: x["pop"]) if calls else None
    top_put = max(puts, key=lambda x: x["pop"]) if puts else None
    primary = best_call if bias == "CALL" else best_put
    alt = best_put if bias == "CALL" else best_call

    # ---- narrative ---------------------------------------------------------
    trend = ("ขาขึ้น" if d_info.get("stack") == 1 else "ขาลง" if d_info.get("stack") == -1 else "ไร้ทิศทาง")
    volw = float(np.mean(np.abs(logret[-5:])) * 100) if len(logret) > 5 else 0.0
    base.update(
        expiry=exp, dte=dte, n_days=n_days, expected_move=round(em, 2),
        em_low=round(spot - em, 2), em_high=round(spot + em, 2),
        sigma_model=round(sigma_mix * 100, 1),
        calls=calls, puts=puts, best_call=best_call, best_put=best_put,
        top_call=top_call, top_put=top_put,
        best=primary, alternate=alt,
        move_stats=sim.move_table(dist),
        hist=dict(n=(dist["n"] if dist else 0),
                  avg_abs_move=round(float(np.mean(np.abs(dist["rel"])) * 100), 2) if dist else None,
                  up_rate=round(float(np.mean(dist["rel"] > 0) * 100), 1) if dist else None,
                  best=round(float(np.max(dist["rel"]) * 100), 1) if dist else None,
                  worst=round(float(np.min(dist["rel"]) * 100), 1) if dist else None,
                  window_days=HIST_WINDOW),
        regime=dict(trend=trend, iv_tag="ถูก" if ivr < 30 else ("แพง" if ivr > 70 else "กลาง"),
                    squeeze=bool(d_info.get("squeeze")), atr_pct=d_info.get("atr_pct")),
    )
    if primary:
        base["omega"] = primary["omega"]
        base["verdict"] = primary["verdict"]
    else:
        base["omega"], base["verdict"] = 0.0, "NONE"

    rel = 50.0
    rel += 10 if ch.get("spot") else -10
    rel += 8 if ns.get("n", 0) >= 6 else (-4 if ns.get("n", 0) == 0 else 0)
    rel += 10 if (dist and dist["n"] >= 100) else (-8 if not dist else 0)
    rel += 6 if primary else 0
    base["reliability"] = round(float(np.clip(rel + 0.3 * (primary["reliability"] if primary else 0), 0, 100)), 1)

    guide = []
    guide.append(f"หุ้น {sym} ราคา ${spot:.2f} · แนวโน้ม {trend} · โมเมนตัม 21 วัน {d_info.get('mom21')}% "
                 f"· RSI {d_info.get('rsi')} · ADX {d_info.get('adx')}")
    guide.append(f"ความผันผวน: IV {atm_iv*100:.0f}% (อันดับ {ivr:.0f}/100 = {base['regime']['iv_tag']}) "
                 f"· ค่าความผันผวนจริง 6 เดือน {sigma_hist*100:.0f}% · ส่วนต่าง {vrp:+.1f} จุด")
    if base["hist"].get("avg_abs_move") is not None:
        guide.append(f"ตลาดคาดการแกว่ง ±{em:.2f} (${spot-em:.2f} – ${spot+em:.2f}) ภายใน {dte} วัน · "
                     f"สถิติ 6 เดือนค่าเฉลี่ยแกว่ง {base['hist']['avg_abs_move']}% ต่อ {n_days} วันทำการ "
                     f"· ใน 6 เดือนหุ้นตัวนี้ปิดบวกในรอบ {n_days} วันทำการ {base['hist'].get('up_rate')}% ของรอบ")
    else:
        guide.append(f"ตลาดคาดการแกว่ง ±{em:.2f} (${spot-em:.2f} – ${spot+em:.2f}) ภายใน {dte} วัน · "
                     f"⚠ ดึงราคาย้อนหลังไม่สำเร็จในรอบนี้ จึงยังไม่มีสถิติ 6 เดือน "
                     f"(ตัวเลขโอกาสชนะยังใช้ได้ แต่ไม่มีการยืนยันด้วยสถิติจริง)")
    if primary:
        guide.append(f"ดีลที่คุ้มค่าที่สุด (และตรงกับทิศทางที่โมเดลเห็น): {bias} — {primary['headline']} · "
                     f"EV {primary['ev']:+.2f}$ ({_pct_txt(primary.get('ev_pct'))} ของพรีเมียม) · "
                     f"ความเชื่อมั่นข้อมูล {primary['reliability']:.0f}/100")
    if top_call and top_put:
        guide.append(f"ถ้าเอา “โอกาสชนะ” เป็นตัวตั้งอย่างเดียว: CALL สูงสุดที่ ${top_call['strike']:g} "
                     f"({top_call['pop']:.0f}%) · PUT สูงสุดที่ ${top_put['strike']:g} ({top_put['pop']:.0f}%) "
                     f"— ดูรายละเอียดและเหตุผลได้ในตาราง Strike ด้านล่าง")
    base["guide"] = guide

    # ---- the direct answer: which strike wins, and why ----------------------
    ans = []
    if top_call:
        ans.append(f"ถ้าถามว่า “CALL strike ไหนโอกาสชนะสูงสุด” → CALL ${top_call['strike']:g} "
                   f"(หมดอายุ {top_call['expiry']}) โอกาสชนะ {top_call['pop']:.0f}% "
                   f"(ช่วง {top_call['pop_lo']:.0f}–{top_call['pop_hi']:.0f}%) พรีเมียม ${top_call['premium']} "
                   f"จุดคุ้มทุน ${top_call['breakeven']} — เหตุผล: {top_call['why'][0] if top_call['why'] else ''}")
    if top_put:
        ans.append(f"ถ้าถามว่า “PUT strike ไหนโอกาสชนะสูงสุด” → PUT ${top_put['strike']:g} "
                   f"(หมดอายุ {top_put['expiry']}) โอกาสชนะ {top_put['pop']:.0f}% "
                   f"(ช่วง {top_put['pop_lo']:.0f}–{top_put['pop_hi']:.0f}%) พรีเมียม ${top_put['premium']} "
                   f"จุดคุ้มทุน ${top_put['breakeven']} — เหตุผล: {top_put['why'][0] if top_put['why'] else ''}")
    if primary:
        ans.append(f"ถ้าถามว่า “สัญญาที่คุ้มค่าที่สุด (โอกาสชนะ + มูลค่า + สภาพคล่อง)” → {primary['kind']} "
                   f"${primary['strike']:g} โอกาสชนะ {primary['pop']:.0f}% EV {primary['ev']:+.2f}$ "
                   f"({_pct_txt(primary.get('ev_pct'))} ของพรีเมียม)"
                   + (f" สถิติ 6 เดือนแบบเดียวกันชนะ {primary['hist_win']:.0f}%"
                      if primary.get("hist_win") is not None else ""))
    if top_call and top_put:
        win = "CALL" if top_call["pop"] >= top_put["pop"] else "PUT"
        ans.append(f"สรุปสัปดาห์นี้: ฝั่ง {win} ให้ความน่าจะเป็นชนะสูงกว่า "
                   f"(CALL สูงสุด {top_call['pop']:.0f}% · PUT สูงสุด {top_put['pop']:.0f}%) "
                   f"แต่คะแนนความคุ้มค่าเฉลี่ยของทั้งสองฝั่งต้องดู EV ประกอบ — "
                   + ("ฝั่ง CALL มี EV เป็นบวก" if (top_call["ev_pct"] or 0) > 0 else
                      "ทั้งสองฝั่ง EV ยังติดลบ ซึ่งเป็นเรื่องปกติของออปชันซื้อเดี่ยว (long option) "
                      "เพราะต้องจ่ายค่าความผันผวนล่วงหน้า"))
    base["answer"] = ans

    cands = [c for c in (calls + puts) if c["cheap"] and c["liquid"]]
    cands.sort(key=lambda x: -x["omega"])
    base["candidates"] = cands[:8]
    base["all_cheap"] = sorted([c for c in (calls + puts) if c["cheap"]], key=lambda x: -x["omega"])[:10]
    return base


# Fields the dashboard's list view actually renders (see index.html).  Anything
# not here is only needed by the detail panel, which fetches /api/analyze/<sym>
# fresh and therefore gets the full, uncompacted result.
#
# This matters for memory, not tidiness: a full analyze() result is ~120 KB
# because every strike carries its "why"/"risks" prose and every ticker carries
# up to 12 news items with sources, checks and filings.  Holding all 15 of them
# in STATE cost ~3.2 MB of live JSON, and on a 512 MB free instance that was
# enough to get the worker killed mid-scan.
_STRIKE_KEEP = (
    "ticker", "kind", "strike", "dist_pct", "premium", "premium_pct", "bid", "ask",
    "volume", "oi", "spread_pct", "cheap", "liquid", "delta", "theta_bleed",
    "liquidity", "omega", "verdict", "reliability", "hist_edge", "hist_n",
    "hist_win", "expiry", "dte", "pop", "pop_lo", "pop_hi", "ev", "ev_pct",
    "breakeven", "iv", "iv_vs_atm", "iv_tag", "headline", "p_itm", "p_touch",
)
_TICKER_KEEP = (
    "ticker", "name", "spot", "spot_source", "chain_ts", "signal", "bias",
    "tech_score", "flow_score", "news_score", "news", "atm_iv", "iv30",
    "iv_rank", "iv_pct", "vrp", "sigma_hist", "sigma_model", "earnings",
    "expected_move", "em_low", "em_high", "expiry", "dte", "n_days",
    "reliability", "omega", "verdict", "error", "hist", "regime", "tech",
    "flow", "iv30", "move_stats",
)


def _slim_strike(s: dict | None) -> dict | None:
    if not s:
        return None
    return {k: s[k] for k in _STRIKE_KEEP if k in s}


def compact_for_scan(r: dict) -> dict:
    """Reduce a full analyze() result to what the list view shows.

    Keeps the per-ticker summary, the winning CALL/PUT and the short candidate
    list; drops the prose, the raw strike table and the news bodies that the
    detail panel reloads on demand.
    """
    out = {k: r[k] for k in _TICKER_KEEP if k in r}
    out["best"] = _slim_strike(r.get("best"))
    out["alternate"] = _slim_strike(r.get("alternate"))
    out["top_call"] = _slim_strike(r.get("top_call"))
    out["top_put"] = _slim_strike(r.get("top_put"))
    out["best_call"] = _slim_strike(r.get("best_call"))
    out["best_put"] = _slim_strike(r.get("best_put"))
    out["calls"] = []
    out["puts"] = []
    # The screener needs the same strike fields as the table rows.
    out["candidates"] = [_slim_strike(c) for c in (r.get("candidates") or [])[:4]]
    # Counts the list view displays, without the underlying bodies.
    nd = r.get("news_detail") or {}
    out["news_detail"] = dict(n=nd.get("n", 0), engine=nd.get("engine"),
                              sources={}, checks=[], items=[], filings=[],
                              social=nd.get("social", {}), note=nd.get("note", ""))
    out["guide"] = (r.get("guide") or [])[:3]
    out["answer"] = (r.get("answer") or [])[:2]
    return out


# --------------------------------------------------------------------------- #
# scan state
# --------------------------------------------------------------------------- #
STATE = dict(items=[], screener=[], updated=None, cycle_sec=0, errors=[], started=None,
             progress=dict(done=0, total=0, current=""), universe=list(WATCHLIST), version="2.0")
_STATE_LOCK = threading.Lock()
# Held for the duration of one full scan. The background loop and any manual
# scan both take it, so two passes can never run at the same time and corrupt
# the progress counter (or double the memory pressure on a free instance).
scan_lock = threading.Lock()


def scan_once() -> None:
    # One scan at a time, whoever asks for it (background loop or /api/scan/now).
    if not scan_lock.acquire(blocking=False):
        log.info("scan_once skipped: another scan is already running")
        return
    try:
        _scan_once_locked()
    finally:
        scan_lock.release()


def _scan_once_locked() -> None:
    t0 = time.time()
    items, errs, cands = [], [], []
    total = len(WATCHLIST)
    with _STATE_LOCK:
        STATE["progress"] = dict(done=0, total=total, current="")
        STATE["universe"] = list(WATCHLIST)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work(sym):
        s = sym.strip().upper()
        try:
            r = analyze(s)
        except Exception as ex:
            log.warning("analyze %s failed: %s", s, ex)
            return s, None, f"{s}: {type(ex).__name__}: {str(ex)[:90]}"
        if r is None:
            return s, None, f"{s}: ไม่มีข้อมูลพอสำหรับวิเคราะห์"
        if r.get("error"):
            return s, None, f"{s}: {r['error']}"
        return s, r, None

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        futs = {ex.submit(work, s): s for s in WATCHLIST}
        # Publish each ticker the moment it lands rather than waiting for the
        # slowest one.  On a free instance the full pass can take a while, and an
        # empty page for minutes looks broken even though work is happening.
        for fut in as_completed(futs):
            sym, r, err = fut.result()
            with _STATE_LOCK:
                STATE["progress"]["done"] += 1
                STATE["progress"]["current"] = sym
            if err:
                errs.append(err)
            if r:
                slim = compact_for_scan(r)
                items.append(slim)
                for cd in (slim.get("candidates") or [])[:4]:
                    cd = dict(cd)
                    cd.pop("why", None)
                    cd.pop("risks", None)
                    cd["signal"] = r.get("signal")
                    cd["tech_score"] = r.get("tech_score")
                    cd["flow_score"] = r.get("flow_score")
                    cd["news_score"] = r.get("news_score")
                    cands.append(cd)

                items.sort(key=lambda x: -(x.get("omega") or 0))
                cands.sort(key=lambda x: -(x.get("omega") or 0))
                with _STATE_LOCK:
                    STATE["items"] = list(items)
                    STATE["screener"] = list(cands[:40])
                    STATE["updated"] = now_iso()
                    STATE["progress"]["total"] = total

    items.sort(key=lambda r: -(r.get("omega") or 0))
    cands.sort(key=lambda c: -(c.get("omega") or 0))
    with _STATE_LOCK:
        STATE.update(items=items, screener=cands[:15], updated=now_iso(),
                     cycle_sec=round(time.time() - t0, 1), errors=errs[:40],
                     progress=dict(done=total, total=total, current=""), universe=list(WATCHLIST))


def loop() -> None:
    while True:
        s = time.time()
        try:
            scan_once()
        except Exception as ex:
            log.exception("scan loop: %s", ex)
        time.sleep(max(REFRESH_SEC - (time.time() - s), 15.0))


_thread = None


def start_scan() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=loop, daemon=True, name="omega-scan")
        _thread.start()
        with _STATE_LOCK:
            STATE["started"] = now_iso()
