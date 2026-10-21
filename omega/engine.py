"""Omega v2 โ€” the analysis engine.

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
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import data, news, quant, sim
from .util import (cache, clean, clip, env_float, env_int, env_list, http, io_pool,
                   log, now_iso, nz, release_memory)

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
# Real expiry dates are used instead of fixed "1/3/7 day" buckets, because every
# contract already states the date it must obey.  A chain can list a dozen
# expiries; these are how many of the nearest ones get their own answer.
MAX_EXPIRY_VIEWS = env_int("MAX_EXPIRY_VIEWS", 4)
_WEEKDAY_TH = ("เธเธฑเธเธ—เธฃเน", "เธญเธฑเธเธเธฒเธฃ", "เธเธธเธ", "เธเธคเธซเธฑเธชเธเธ”เธต", "เธจเธธเธเธฃเน", "เน€เธชเธฒเธฃเน", "เธญเธฒเธ—เธดเธ•เธขเน")
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
# How many full analyses may run simultaneously (scan workers + a detail view or
# a diag probe).  Keeping this at the scan worker count means extra callers wait
# instead of stacking another ~25MB of pandas working set on the heap.
ANALYZE_SLOTS = env_int("ANALYZE_SLOTS", max(1, SCAN_WORKERS))
# The wall-clock ceiling for one ticker.  Everything inside has its own
# timeout; this is the backstop for the case where a provider accepts the
# connection and then never answers.
ANALYZE_TIMEOUT = env_float("ANALYZE_TIMEOUT", 90.0)
WEIGHTS = {"A": 0.16, "B": 0.12, "C": 0.16, "D": 0.22, "E": 0.14, "H": 0.20}
TARGET_DELTA = 0.40


_TRACE: dict = {}
_TRACE_LOCK = threading.Lock()


def trace_snapshot() -> dict:
    """Which stage each ticker is in, and for how long."""
    now = time.time()
    with _TRACE_LOCK:
        return {k: dict(stage=v.get("stage"),
                        secs=round(now - (v.get("t") or now), 1),
                        ms=v.get("ms") or {})
                for k, v in _TRACE.items()}


def _trace(sym: str, stage: str, ms: dict | None = None) -> None:
    with _TRACE_LOCK:
        _TRACE[sym] = dict(stage=stage, t=time.time(), ms=dict(ms or {}))


def _seed(sym: str) -> int:
    return (abs(hash(sym)) % 100000) + int(time.time() // 900)


# The instance this runs on has a hard memory limit, and a worker that is
# killed for using too much of it takes the whole page down with it.  The scan
# therefore stops itself before that point rather than discovering it.
MEMORY_CEILING_MB = env_float("MEMORY_CEILING_MB", 330.0)


def _rss_mb() -> float:
    """The resident size of this process, in megabytes (0 when unknown)."""
    try:
        with open("/proc/self/statm", "r") as f:
            pages = int(f.read().split()[1])
        return pages * (os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
    except Exception:
        return 0.0


_SPY_LOCK = threading.Lock()
_SPY: dict = {"df": None, "ts": 0.0}


def _spy_benchmark():
    """SPY's daily series, used to say whether a ticker is beating the market.

    Fetched at most once an hour and behind a lock.  A benchmark that is an
    hour old is still a benchmark, and when the price provider is refusing us
    every ticker in the scan used to pay the full fallback chain for it -- the
    two slowest tickers in a cycle were exactly the ones that triggered it.
    """
    global _SPY
    if _SPY["df"] is not None and (time.time() - _SPY["ts"]) < 3600:
        return _SPY["df"]
    if not _SPY_LOCK.acquire(blocking=False):
        # A refresh is already running elsewhere; do not make this ticker wait
        # for it, use the series we already have.
        return _SPY["df"]
    try:
        df = cache.wrap("d:spy_bench", 3600, lambda: data.daily("SPY"))
        if df is not None and not df.empty:
            _SPY = {"df": df, "ts": time.time()}
        return df
    except Exception:
        return _SPY["df"]
    finally:
        _SPY_LOCK.release()


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


def tech_votes(info: dict) -> list:
    """One row per technique, so the direction can be audited line by line.

    When there is no news this table *is* the reasoning, so every row has to be
    readable by someone who does not code: the technique, which way it leans,
    and the number that produced that lean.
    """
    if not info:
        return []
    rows = []

    def add(name: str, score: float, why: str):
        score = float(np.clip(nz(score, 0.0), -1, 1))
        rows.append(dict(name=name,
                         dir="CALL" if score > 0.12 else ("PUT" if score < -0.12 else "เธเธฅเธฒเธ"),
                         score=round(score, 2), why=why))

    st = nz(info.get("stack"), 0)
    add("EMA 9/21/50 เน€เธฃเธตเธขเธเธ•เธฑเธง", st,
        "เน€เธฃเธตเธขเธเธเธถเนเธเธเธฃเธ 9>21>50" if st > 0 else
        ("เน€เธฃเธตเธขเธเธฅเธเธเธฃเธ 9<21<50" if st < 0 else "เธขเธฑเธเธชเธฅเธฑเธเธเธฑเธ เนเธกเนเธกเธตเนเธเธงเนเธเนเธกเธเธฑเธ”"))

    r = nz(info.get("rsi"), 50.0)
    div = nz(info.get("rsi_div"), 0)
    add("RSI(14)", (r - 50) / 25 * 0.7 + 0.3 * div,
        f"RSI {r:.0f}"
        + (" + เนเธ”เน€เธงเธญเธฃเนเน€เธเธเธเนเธเธฅเธฑเธเธ•เธฑเธงเธเธถเนเธ" if div > 0 else
           (" + เนเธ”เน€เธงเธญเธฃเนเน€เธเธเธเนเธเธฅเธฑเธเธ•เธฑเธงเธฅเธ" if div < 0 else "")))

    mh = nz(info.get("macd_hist"), 0.0)
    add("MACD histogram", np.clip(mh * 4, -1, 1), f"เนเธ—เนเธ histogram {mh:+.3f}")

    adx, pdi, mdi = nz(info.get("adx"), 0.0), nz(info.get("pdi"), 0.0), nz(info.get("mdi"), 0.0)
    add("ADX / DI เธ—เธดเธจเธ—เธฒเธ", (1.0 if pdi > mdi else -1.0) * min(adx / 25.0, 1.0),
        f"ADX {adx:.0f} ยท DI+ {pdi:.0f} / DI- {mdi:.0f}")

    stoch = nz(info.get("stoch"), 50.0)
    add("Stochastic", np.clip((stoch - 50) / 40, -1, 1), f"Stoch {stoch:.0f}")

    m21, m63 = nz(info.get("mom21"), 0.0), nz(info.get("mom63"), 0.0)
    add("เนเธกเน€เธกเธเธ•เธฑเธก 1โ€“3 เน€เธ”เธทเธญเธ",
        np.clip(m21 * 8, -1, 1) * 0.6 + np.clip(m63 * 4, -1, 1) * 0.4,
        f"1 เน€เธ”เธทเธญเธ {m21:+.1f}% ยท 3 เน€เธ”เธทเธญเธ {m63:+.1f}%")

    rs = nz(info.get("rs_spy"), 0.0)
    add("เนเธเนเธเนเธฃเธเธเธงเนเธฒ SPY", np.clip(rs * 6, -1, 1), f"{rs:+.1f}% เน€เธ—เธตเธขเธ SPY")

    pb = nz(info.get("pb"), 0.5)
    # Bollinger position is mean-reverting: hugging the top band is stretched,
    # not bullish, so the vote points the other way.
    add("เธ•เธณเนเธซเธเนเธ Bollinger", np.clip((0.5 - pb) * 1.6, -1, 1),
        f"เธญเธขเธนเนเธ—เธตเน {pb * 100:.0f}% เธเธญเธเธเธฃเธญเธ"
        + (" (เธเธดเธ”เธเธฃเธญเธเธเธ = เธ•เธถเธเธ•เธฑเธง)" if pb > 0.9 else
           (" (เธเธดเธ”เธเธฃเธญเธเธฅเนเธฒเธ = เธ•เธถเธเธ•เธฑเธง)" if pb < 0.1 else "")))
    return rows


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
    verified = sum(1 for c in checks if c["verdict"] in ("เธขเธทเธเธขเธฑเธเนเธฅเนเธง", "เธขเธทเธเธขเธฑเธเธเธฒเธเธชเนเธงเธ"))
    sig = 0.62 * s + 0.22 * (f * math.copysign(1, s if abs(s) > 0.05 else 1)) + 0.16 * soc
    if verified:
        sig *= 1.0 + min(0.15, 0.05 * verified)
    return float(np.clip(nz(sig), -1, 1)), dict(verified=verified)


# --------------------------------------------------------------------------- #
# earnings calendar (best effort, cached)
# --------------------------------------------------------------------------- #
def _earn_day(ds: str) -> list:
    """The symbols reporting on one calendar day."""
    j = http.json(f"https://api.nasdaq.com/api/calendar/earnings?date={ds}",
                  headers={"Accept": "application/json"}, timeout=(4, 8))
    if not j:
        return []
    out = []
    for row in ((j.get("data") or {}).get("rows") or []):
        sym = (row.get("symbol") or "").upper()
        if sym:
            out.append(sym)
    return out


_EARN_LOCK = threading.Lock()
_EARN: dict = {}
_EARN_TS: float = 0.0


def earnings_map(days: int = MAX_DTE + 7) -> dict:
    global _EARN, _EARN_TS
    if _EARN and (time.time() - _EARN_TS) < 21600:
        return dict(_EARN)
    if not _EARN_LOCK.acquire(blocking=False):
        # Somebody else is already refreshing the calendar.  Waiting for it would
        # put a share price, a technical and a calendar on the same critical
        # path, so the ticker gets what we already know and moves on.
        return dict(_EARN)

    def load():
        out = {}
        dates = []
        d = datetime.now(timezone.utc).date()
        while len(dates) < min(days, 15):
            d += timedelta(days=1)
            if d.weekday() < 5:
                dates.append(d.isoformat())
        # One day at a time cost up to 12s per day, so a calendar that was slow
        # or refused us could hold a single ticker for minutes -- it was the
        # largest single cost in a scan cycle.  The days are independent, so
        # they are asked together and the whole set gets one deadline.
        from concurrent.futures import wait

        ex = io_pool()
        futs = {ex.submit(_earn_day, ds): ds for ds in dates}
        done, _ = wait(list(futs.keys()), timeout=10.0)
        for f in done:
            ds = futs[f]
            try:
                syms = f.result(timeout=0.5)
            except Exception:
                syms = []
            for sym in syms:
                out[sym] = ds
        return out or None

    try:
        val = cache.wrap("earn:map", 21600, load) or {}
        _EARN = dict(val)
        _EARN_TS = time.time()
        return dict(_EARN)
    finally:
        _EARN_LOCK.release()


# --------------------------------------------------------------------------- #
# text: why this strike?
# --------------------------------------------------------------------------- #
def _baht_pct(x):
    return f"{x:+.1f}%" if x is not None else "-"


def _mag_pct(x):
    """Unsigned magnitude, for sentences that already say 'above'/'below'."""
    return f"{abs(x):.1f}%" if x is not None else "-"


def _pct_txt(x):
    return f"{x:+.0f}%" if x is not None else "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅ"


def strike_reasons(row: dict, ctx: dict) -> list:
    """Plain-Thai explanation for one strike."""
    out = []
    up = row["kind"] == "CALL"
    side = "เธเธถเนเธ" if up else "เธฅเธ"
    d = row["dist_pct"]
    # 1. moneyness + distance vs expected move
    em = ctx["em_pct"]
    itm = (d < 0) if up else (d > 0)
    if itm:
        if up:
            out.append(f"Strike เธเธตเนเน€เธเนเธ In-the-Money เธญเธขเธนเนเนเธฅเนเธง (เธฃเธฒเธเธฒเธซเธธเนเธ ${ctx.get('spot', 0):.2f} เธชเธนเธเธเธงเนเธฒ strike "
                       f"{_mag_pct(d)}) เธเธถเธเธกเธตเธกเธนเธฅเธเนเธฒเธเธเน€เธซเธฅเธทเธญเนเธฅเธฐเนเธญเธเธฒเธชเธเธเธฐเธชเธนเธเธเธงเนเธฒเธชเธฑเธเธเธฒ OTM "
                       f"เนเธ•เนเธเธฃเธตเน€เธกเธตเธขเธกเธเนเนเธเธเธเธงเนเธฒเธ•เธฒเธกเนเธเธ”เนเธงเธข")
        else:
            out.append(f"Strike เธเธตเนเน€เธเนเธ In-the-Money เธญเธขเธนเนเนเธฅเนเธง (เธฃเธฒเธเธฒเธซเธธเนเธ ${ctx.get('spot', 0):.2f} เธ•เนเธณเธเธงเนเธฒ strike "
                       f"{_mag_pct(d)}) เธเธถเธเธกเธตเธกเธนเธฅเธเนเธฒเธเธเน€เธซเธฅเธทเธญเนเธฅเธฐเนเธญเธเธฒเธชเธเธเธฐเธชเธนเธเธเธงเนเธฒเธชเธฑเธเธเธฒ OTM "
                       f"เนเธ•เนเธเธฃเธตเน€เธกเธตเธขเธกเธเนเนเธเธเธเธงเนเธฒเธ•เธฒเธกเนเธเธ”เนเธงเธข")
    elif abs(d) <= em:
        out.append(f"Strike เธญเธขเธนเนเธซเนเธฒเธเธเธฒเธเธฃเธฒเธเธฒเธเธฑเธเธเธธเธเธฑเธ {_mag_pct(d)} เธเธถเนเธเธญเธขเธนเนเนเธเธฃเธฐเธขเธฐเธเธฒเธฃเนเธเธงเนเธเธ—เธตเนเธ•เธฅเธฒเธ”เธเธฒเธ”เนเธงเน ยฑ{em:.1f}% "
                   f"(Expected Move) เธเธถเธเธกเธตเนเธญเธเธฒเธชเธ–เธนเธเนเธ•เธฐเธชเธนเธ")
    elif abs(d) <= 2 * em:
        out.append(f"Strike เธซเนเธฒเธ {_mag_pct(d)} = เธเธฃเธฐเธกเธฒเธ“ {abs(d)/max(em,0.1):.1f} เน€เธ—เนเธฒเธเธญเธ Expected Move ยฑ{em:.1f}% "
                   f"เธ•เนเธญเธเธกเธตเธเนเธฒเธงเธซเธฃเธทเธญเนเธกเน€เธกเธเธ•เธฑเธกเธซเธเธธเธเธเธถเธเธเธฐเนเธเธ–เธถเธ")
    else:
        out.append(f"Strike เธซเนเธฒเธเนเธเธฅ {_mag_pct(d)} (เน€เธเธดเธ 2 เน€เธ—เนเธฒเธเธญเธ Expected Move ยฑ{em:.1f}%) "
                   f"เธ•เนเธญเธเธกเธตเธเธฒเธฃเน€เธเธฅเธทเนเธญเธเนเธซเธงเธฃเธธเธเนเธฃเธเธเธดเธ”เธเธเธ•เธด")
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
            out.append(f"เธชเธ–เธดเธ•เธดเธเธฃเธดเธ 6 เน€เธ”เธทเธญเธ: เธชเธฑเธเธเธฒเนเธเธเธเธตเนเธเธเธฐเน€เธกเธทเนเธญเธซเธกเธ”เธญเธฒเธขเธธ {ht:.0f}% เธเธญเธเธฃเธญเธ "
                       f"เธชเธนเธเธเธงเนเธฒเธ—เธตเนเนเธกเน€เธ”เธฅเธเธฃเธฐเน€เธกเธดเธ ({pt:.0f}%) โ’ เธเธฑเนเธ {row['kind']} เนเธ”เนเน€เธเธฃเธตเธขเธเน€เธเธดเธเธชเธ–เธดเธ•เธด "
                       f"เน€เธเธฃเธฒเธฐเธ•เธฅเธฒเธ”เธ•เธฑเนเธเธฃเธฒเธเธฒเน€เธเธทเนเธญเธเธงเธฒเธกเน€เธชเธตเนเธขเธเนเธงเนเธกเธฒเธเธเธงเนเธฒเธ—เธตเนเน€เธเธดเธ”เธเธฃเธดเธ")
        elif gap <= -5:
            out.append(f"เธชเธ–เธดเธ•เธดเธเธฃเธดเธ 6 เน€เธ”เธทเธญเธ: เธชเธฑเธเธเธฒเนเธเธเธเธตเนเธเธเธฐเน€เธเธตเธขเธ {ht:.0f}% เธ•เนเธณเธเธงเนเธฒเธ—เธตเนเนเธกเน€เธ”เธฅเนเธซเน ({pt:.0f}%) "
                       f"โ’ เธ•เธฅเธฒเธ”เธญเธฒเธเธ•เธฑเนเธเธฃเธฒเธเธฒเน€เธเธดเธ”เน€เธเธทเนเธญเนเธงเนเธกเธฒเธเน€เธเธดเธเนเธ เธเธฃเธตเน€เธกเธตเธขเธกเนเธเธเธเธงเนเธฒเธเธงเธฒเธกเน€เธชเธตเนเธขเธเธเธฃเธดเธ")
        else:
            out.append(f"เธชเธ–เธดเธ•เธดเธเธฃเธดเธ 6 เน€เธ”เธทเธญเธ ({ht:.0f}%) เธเธฑเธเธ—เธตเนเนเธกเน€เธ”เธฅเนเธซเน ({pt:.0f}%) เธชเธญเธ”เธเธฅเนเธญเธเธเธฑเธ "
                       f"โ’ เธเธงเธฒเธกเธเนเธฒเธเธฐเน€เธเนเธเธเธตเนเน€เธเธทเนเธญเธ–เธทเธญเนเธ”เนเธเธญเธชเธกเธเธงเธฃ")
    # 3. probability of finishing beyond breakeven
    hl = row.get("hist_win")
    if hl is not None and row.get("hist_n"):
        out.append(f"เธเธทเนเธญเธชเธฑเธเธเธฒเธเธตเนเนเธฅเนเธงเธเธณเนเธฃเน€เธกเธทเนเธญเธซเธกเธ”เธญเธฒเธขเธธ: เธชเธ–เธดเธ•เธด {ctx['months']} เน€เธ”เธทเธญเธเธ—เธตเนเธเนเธฒเธเธกเธฒเน€เธ—เธตเธขเธเน€เธ—เนเธฒ {hl:.0f}% "
                   f"(เธเธฒเธ {row['hist_n']} เธฃเธญเธ) ยท เนเธกเน€เธ”เธฅเนเธซเน {row['pop']:.0f}% (เธเนเธงเธ {row['pop_lo']:.0f}โ€“{row['pop_hi']:.0f}%)")
    # 4. IV: skew of this strike + overall IV regime
    vrp = ctx.get("vrp", 0)
    ivx = row.get("iv_vs_atm")
    if ivx is not None:
        if ivx > 3:
            out.append(f"IV เธเธญเธเธชเธฑเธเธเธฒเธเธตเน {row['iv']:.0f}% เธชเธนเธเธเธงเนเธฒ ATM {ivx:+.0f} เธเธธเธ” โ€” เธเธฃเธตเน€เธกเธตเธขเธกเธเธงเธกเธเธฒเธเธเธงเธฒเธกเธเธฑเธเธเธงเธ "
                       f"เน€เธเธเธฒเธฐ strike เธเธตเน (เธเธทเนเธญเนเธเธเธเธงเนเธฒเน€เธเธทเนเธญเธเธฃเนเธงเธกเธฃเธธเนเธ)")
        elif ivx < -3:
            out.append(f"IV เธเธญเธเธชเธฑเธเธเธฒเธเธตเน {row['iv']:.0f}% เธ•เนเธณเธเธงเนเธฒ ATM {ivx:.0f} เธเธธเธ” โ€” เนเธ”เนเน€เธเธฃเธตเธขเธเธเธฑเนเธเธเธทเนเธญเน€เธกเธทเนเธญเน€เธ—เธตเธขเธเธ—เธฑเนเธเธชเธฒเธข")
    out.append(f"เธฃเธฐเธ”เธฑเธ IV เธเธญเธเธซเธธเนเธเธ—เธฑเนเธเธ•เธฑเธงเธ•เธญเธเธเธตเน {row['iv_tag']} (เธญเธฑเธเธ”เธฑเธ {ctx.get('iv_rank', 50):.0f}/100 เน€เธ—เธตเธขเธ 1 เธเธตเธ—เธตเนเธเนเธฒเธเธกเธฒ) "
               f"ยท เธเธงเธฒเธกเธเธฑเธเธเธงเธเธเธฃเธดเธ 6 เน€เธ”เธทเธญเธ {ctx.get('sigma_hist', 0):.0f}% ยท เธชเนเธงเธเธ•เนเธฒเธ {vrp:+.1f} เธเธธเธ” โ’ "
               + ("IV เธชเธนเธเธเธงเนเธฒเธเธงเธฒเธกเธเธฑเธเธเธงเธเธเธฃเธดเธ เธญเธญเธเธเธฑเธเนเธ”เธขเธฃเธงเธกเธเนเธญเธเนเธเธ—เธฒเธเนเธเธ เธ•เนเธญเธเธเธเธฐเนเธซเนเธกเธฒเธเธเธงเนเธฒเธเธฃเธตเน€เธกเธตเธขเธกเธ—เธตเนเธเนเธฒเธข"
                  if vrp > 4 else
                  "IV เธ•เนเธณเธเธงเนเธฒเธเธงเธฒเธกเธเธฑเธเธเธงเธเธเธฃเธดเธ เธญเธญเธเธเธฑเธเนเธ”เธขเธฃเธงเธกเธเนเธญเธเนเธเธ—เธฒเธเธ–เธนเธ" if vrp < -2 else
                  "IV เนเธเธฅเนเน€เธเธตเธขเธเธเธงเธฒเธกเธเธฑเธเธเธงเธเธเธฃเธดเธ เธฃเธฒเธเธฒเธญเธญเธเธเธฑเธเธ–เธทเธญเธงเนเธฒเน€เธเนเธเธเธฃเธฃเธก"))
    # 5. liquidity / execution
    if row.get("liquid"):
        out.append(f"เธชเธ เธฒเธเธเธฅเนเธญเธเธเนเธฒเธเน€เธเธ“เธ‘เน: เธเธฃเธดเธกเธฒเธ“ {row['volume']:,} เธชเธฑเธเธเธฒ ยท OI {row['oi']:,} ยท Spread {row['spread_pct']:.1f}%")
    else:
        out.append(f"เธชเธ เธฒเธเธเธฅเนเธญเธเธ•เนเธณ: เธเธฃเธดเธกเธฒเธ“ {row['volume']:,} ยท OI {row['oi']:,} ยท Spread {row['spread_pct']:.1f}% "
                   f"โ€” เน€เธเนเธฒเธญเธญเธเธขเธฒเธ เธฃเธฒเธเธฒเนเธ–เธฅเนเธ”เน")
    # 6. directional evidence
    t, f = ctx.get("tech") or {}, ctx.get("flow") or {}
    bits = []
    if t:
        bits.append(f"RSI {t.get('rsi')} ยท MACD hist {t.get('macd_hist')} ยท ADX {t.get('adx')} ยท Stoch {t.get('stoch')}")
    if f:
        bits.append(f"เธฃเธฒเธเธฒเน€เธ—เธตเธขเธ VWAP {'เน€เธซเธเธทเธญ' if (f.get('price') or 0) >= (f.get('vwap') or 0) else 'เนเธ•เน'} "
                    f"ยท RVOL {f.get('rvol')} ยท P/C volume {f.get('pc_vol')}")
    if bits:
        lean = "เธชเธเธฑเธเธชเธเธธเธ" if (ctx["signal"] > 0) == up else "เธชเธงเธเธ—เธฒเธ"
        out.append(f"เธญเธดเธเธ”เธดเน€เธเน€เธ•เธญเธฃเน: {' | '.join(bits)} โ’ {lean}เธเธฑเนเธ{side}")
    # 7. greeks reality check
    out.append(f"Greeks: ฮ” {row['delta']:.2f} ยท ฮ {row['theta']:+.3f}/เธงเธฑเธ ({row['theta_bleed']:.1f}% เธเธญเธเธเธฃเธตเน€เธกเธตเธขเธก/เธงเธฑเธ) "
               f"ยท Vega {row['vega']:.3f} ยท เธเธธเธ”เธเธธเนเธกเธ—เธธเธ {row['breakeven']:.2f}")
    return out


def strike_risks(row: dict, ctx: dict) -> list:
    out = []
    if ctx.get("earnings"):
        out.append(f"โ  เธเธเธเธฒเธฃเน€เธเธดเธเธญเธญเธเธงเธฑเธเธ—เธตเน {ctx['earnings']} เธเนเธญเธเธซเธกเธ”เธญเธฒเธขเธธ โ’ IV crush เน€เธชเธตเนเธขเธเธชเธนเธเนเธกเนเธ—เธดเธจเธ—เธฒเธเธ–เธนเธ")
    if row.get("spread_pct", 0) > 12:
        out.append(f"โ  Spread เธเธงเนเธฒเธ {row['spread_pct']:.1f}% เธ•เนเธเธ—เธธเธเนเธเธเธชเธนเธ")
    if row.get("p_zero") is not None and row["p_zero"] > 70:
        out.append(f"โ  เนเธญเธเธฒเธชเธซเธกเธ”เธเนเธฒเน€เธเนเธเธจเธนเธเธขเน {row['p_zero']:.0f}% (เธ–เนเธฒเธ–เธทเธญเธ–เธถเธเธซเธกเธ”เธญเธฒเธขเธธ)")
    if row.get("theta_bleed", 0) > 6:
        out.append(f"โ  Theta เนเธฅเนเธเธฃเธตเน€เธกเธตเธขเธก {row['theta_bleed']:.1f}% เธ•เนเธญเธงเธฑเธ")
    if row.get("reliability", 100) < 55:
        out.append(f"โ  เธเธงเธฒเธกเธเนเธฒเน€เธเธทเนเธญเธ–เธทเธญเธเธญเธเธเนเธญเธกเธนเธฅเธ•เนเธณ ({row['reliability']:.0f}/100) โ€” เธ•เธฑเธงเธญเธขเนเธฒเธเธชเธ–เธดเธ•เธด/เธชเธ เธฒเธเธเธฅเนเธญเธเธเนเธญเธข")
    if ctx.get("hist_n", 0) and ctx["hist_n"] < 60:
        out.append(f"โ  เธ•เธฑเธงเธญเธขเนเธฒเธเธชเธ–เธดเธ•เธด 6 เน€เธ”เธทเธญเธเธกเธตเน€เธเธตเธขเธ {ctx['hist_n']} เธฃเธญเธ")
    return out


# --------------------------------------------------------------------------- #
# strike table
# --------------------------------------------------------------------------- #
def build_strike(sym: str, row_c: dict, spot: float, kind: str, dte: float, n_days: int,
                 simset: dict, dist: dict, close: np.ndarray, atm_iv: float, sigma_hist: float,
                 T: float, ctx: dict, dist_full: dict | None = None) -> dict | None:
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

    # ---- the number that is counted, not modelled ---------------------------
    # The model probabilities above are kept only as a labelled cross-check.
    # What the page leads with is a count over real prices: for every entry day
    # in the downloaded history, hold this same contract to the same expiry and
    # see how it actually finished.  ``outcome`` therefore replaces ``pop`` as
    # the headline, and carries the sample size and Wilson interval so the
    # reader can judge for themselves how much to trust it.
    outcome = sim.hist_outcome(dist_full, kind, strike, spot, premium) if dist_full else None
    if outcome and outcome.get("hist_win") is not None:
        r["model_pop"] = r.get("pop")          # keep the estimate, clearly named
        r["model_pop_lo"] = r.get("pop_lo")
        r["model_pop_hi"] = r.get("pop_hi")
        r["model_itm"] = r.get("p_itm")
        r["model_touch"] = r.get("p_touch")
        r["pop"] = outcome["hist_win"]         # headline = real, counted outcome
        r["pop_lo"] = outcome["hist_win_lo"]
        r["pop_hi"] = outcome["hist_win_hi"]
        r["pop_source"] = "เธเธฃเธดเธ"               # real history
        r.update(outcome)
        # The same count over the last six and twelve months, so a recent change
        # in behaviour is visible instead of being averaged away by two years of
        # history.  126 and 252 windows are 6 and 12 months of trading days.
        for _lab, _tail in (("6m", 126), ("12m", 252)):
            _w = sim.hist_outcome(dist_full, kind, strike, spot, premium, tail=_tail)
            if _w and _w.get("hist_win") is not None:
                r[f"hist_{_lab}_win"] = _w["hist_win"]
                r[f"hist_{_lab}_n"] = _w["hist_n"]
                r[f"hist_{_lab}_pnl"] = _w["hist_avg_pnl"]
                r[f"hist_{_lab}_count"] = _w["hist_win_count"]
                r[f"hist_{_lab}_pf"] = _w["hist_profit_factor"]
        # ITM and touch are counts too now: the same historical windows already
        # say how often the level was reached and how often the option still had
        # value at expiry, so the page does not need the model for those either.
        if outcome.get("hist_itm") is not None:
            r["p_itm"] = outcome["hist_itm"]
        if outcome.get("hist_touch") is not None:
            r["p_touch"] = outcome["hist_touch"]
    else:
        r["pop_source"] = "เนเธเธเธเธณเธฅเธญเธ"            # fell back to the model

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
    # construction โ€” you pay the spread and rent volatility.  Scoring EV on an
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
    # ``hist_edge`` now means "what really happened versus what the model
    # expected".  Both sides are named explicitly because the headline ``pop``
    # is itself the historical number; comparing it to itself would be a bug.
    hist_edge = None
    if r.get("hist_n", 0) >= 40 and r.get("model_pop") is not None:
        hist_edge = r["hist_win"] - r["model_pop"]
        r["hist_edge"] = round(hist_edge, 1)
        z, pv, _ = edge_significance(r.get("hist_win"), r.get("hist_n_eff"),
                                     r.get("model_pop"))
        r["edge_z"], r["edge_p"] = z, pv
    edge_score = float(np.clip(50 + (hist_edge or 0) * 2.0, 0, 100))
    dir_score = float(np.clip(50 + 50 * ctx["signal"] * (1 if kind == "CALL" else -1), 0, 100))
    iv_score = float(np.clip(100 - ivr_ctx - max(0.0, ctx.get("vrp", 0)) * 1.5 - min(theta_bleed * 3, 35), 0, 100))
    # Expectancy is scored off the *counted* payoff (``hist_avg_pnl``) when it
    # exists, because that is money, not a model output; the modelled EV is the
    # fallback.  Both are expressed per $1 of premium so they are comparable.
    real_ev = r.get("hist_avg_pnl")
    real_ev_pct = (real_ev / premium * 100.0) if (real_ev is not None and premium > 0) else None
    if real_ev_pct is None:
        real_ev_pct = cand_ev
    ev_real_score = float(np.clip(50 + 60 * math.tanh(nz(real_ev_pct, 0.0) / 100.0 * 2.5), 0, 100))
    omega = (0.34 * pop_score + 0.24 * ev_real_score + 0.14 * ev_score + 0.09 * edge_score
             + 0.10 * liq_score + 0.06 * dir_score + 0.03 * iv_score)
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
        iv_tag="เธ–เธนเธ" if ivr_ctx < 30 else ("เนเธเธ" if ivr_ctx > 70 else "เธเธฅเธฒเธ"),
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
    out["headline"] = (f"{kind} ${out['strike']:g} ยท เธซเธกเธ”เธญเธฒเธขเธธ {out.get('expiry')} ({out['dte']:.0f} เธงเธฑเธ) ยท "
                       f"เธเธฃเธตเน€เธกเธตเธขเธก ${out['premium']} ยท เนเธญเธเธฒเธชเธเธเธฐ {nz(out.get('pop')):.0f}% "
                       f"(เธเนเธงเธ {nz(out.get('pop_lo')):.0f}โ€“{nz(out.get('pop_hi')):.0f}%)")
    return out


# --------------------------------------------------------------------------- #
# short/medium horizon answers ("which strike wins in ~3 / ~5 / ~10 days")
# --------------------------------------------------------------------------- #
def _pick_expiry_for(exps: dict, target: int) -> str | None:
    """Chosen expiry for a horizon: closest to the target, ties broken by DTE."""
    if not exps:
        return None
    within = [e for e, d in exps.items() if d <= target + 2] or list(exps)
    return min(within, key=lambda e: (abs(exps[e] - target), exps[e]))


def _pick_expiry_for(exps: dict, target: int) -> str | None:
    """Chosen expiry for a horizon: closest to the target, ties broken by DTE."""
    if not exps:
        return None
    within = [e for e, d in exps.items() if d <= target + 2] or list(exps)
    return min(within, key=lambda e: (abs(exps[e] - target), exps[e]))


# The fields a per-expiry card needs.  Kept in one place so the API payload and
# the compact list view cannot drift apart from what the UI reads.
_STRIKE_SLIM = (
    "kind", "strike", "expiry", "dte", "premium", "breakeven", "pop", "pop_lo",
    "pop_hi", "p_itm", "p_touch", "ev", "ev_pct", "omega", "verdict", "delta",
    "hist_win", "hist_n", "hist_n_eff", "hist_edge", "cheap", "liquid", "volume",
    "oi", "spread_pct", "iv", "iv_vs_atm", "dist_pct", "reliability", "headline",
    "why", "risks", "pop_source", "model_pop", "hist_win_count", "hist_itm",
    "hist_touch", "hist_avg_pnl", "hist_median_pnl", "hist_best_pnl",
    "hist_worst_pnl", "hist_profit_factor", "hist_recent_win", "hist_older_win",
    "hist_recent_n", "p_zero", "avg_win", "edge_z", "edge_p", "edge_signif",
    "hist_6m_win", "hist_6m_n", "hist_6m_pnl", "hist_6m_count", "hist_6m_pf",
    "hist_12m_win", "hist_12m_n", "hist_12m_pnl", "hist_12m_count", "hist_12m_pf",
)


def slim_strike(x: dict | None) -> dict | None:
    if not x:
        return None
    return {k: x.get(k) for k in _STRIKE_SLIM}


def edge_significance(p_hist: float | None, n_eff: float | None,
                      p_ref: float | None) -> tuple:
    """Is the counted win rate distinguishable from the price the market asks?

    ``p_ref`` is the model's implied chance of finishing beyond breakeven, which
    is what the option's premium already charges you for.  If history says 45%
    and the price implies 45%, there is no edge -- you are paying exactly what the
    outcome is worth.  An edge only exists when the counted rate is *reliably*
    above that reference.

    The test is deliberately one-sided (we only care whether the real rate beats
    the price) and uses the effective sample size, so a 7-day horizon with 80
    independent windows cannot claim the same certainty as a 500-window daily one.

    Returns (z, p_value, label).
    """
    if p_hist is None or not n_eff or p_ref is None:
        return (None, None, "")
    p = float(p_hist) / 100.0
    q = float(p_ref) / 100.0
    var = max(p * (1.0 - p), 1e-6) / max(float(n_eff), 1.0)
    if var <= 0:
        return (None, None, "")
    z = (p - q) / math.sqrt(var)
    pv = 1.0 - quant.norm_cdf(z)          # one-sided
    return (round(z, 2), round(pv, 4), "")


def _sl_rank(group: list, net: float, ivr: float) -> None:
    """Rank strikes of one side inside one expiry (same idea as analyze()).

    Scores are computed on a percentile basis within the group, because a long
    single-leg option always has negative EV in absolute terms; ranking inside
    the group is what surfaces the strike that is genuinely best *among its
    peers* rather than the same mid-band value for everything.

    This is also where the multiple-testing correction happens.  We look at
    dozens of strikes per ticker and keep the best one; the best of thirty
    always looks good, so a raw 5% significance threshold would flag noise every
    single cycle.  Each strike's p-value is therefore compared against
    0.05 / (number of strikes examined), which is the Bonferroni bound.
    """
    if not group:
        return

    def pct(vals, v):
        if not vals:
            return 50.0
        return float(sum(1 for x in vals if x <= v) / len(vals) * 100.0)

    evs = [nz(x.get("ev_pct")) for x in group]
    pops = [nz(x.get("pop")) for x in group]
    edges = [nz(x.get("hist_edge")) for x in group if x.get("hist_edge") is not None]
    alpha = 0.05 / max(1, len(group))
    for x in group:
        dir_score = float(np.clip(50 + 50 * nz(net) * (1 if x["kind"] == "CALL" else -1), 0, 100))
        iv_score = float(np.clip(100 - nz(ivr, 50.0) - max(0.0, nz(x.get("iv_vs_atm"))) * 1.2, 0, 100))
        liq = float(np.clip(nz(x.get("liquidity")) * 100, 0, 100))
        edge = pct(edges, nz(x.get("hist_edge"))) if edges and x.get("hist_edge") is not None else 50.0

        # Decide whether this strike's historical edge survives the number of
        # strikes we looked at.  ``edge_p`` is the one-sided p-value for
        # "history beats the price"; for a strike that came out *worse* than the
        # price the interesting tail is the other one, so it is recomputed.
        z, pv = x.get("edge_z"), x.get("edge_p")
        if z is None or pv is None:
            x["edge_signif"] = "เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเธเธญ"
            edge = 50.0 if edge == 50.0 else 50.0 + (edge - 50.0) * 0.4
        elif z > 0 and pv < alpha:
            x["edge_signif"] = "เธกเธตเธเธฑเธขเธชเธณเธเธฑเธ"
        elif z < 0 and quant.norm_cdf(z) < alpha:
            x["edge_signif"] = "เนเธขเนเธเธงเนเธฒเธ•เธฅเธฒเธ”เธเธฑเธ”เน€เธเธ"
            edge = 50.0 + (edge - 50.0) * 0.5
        else:
            x["edge_signif"] = "เธขเธฑเธเนเธกเนเธเธฑเธ”เน€เธเธ"
            # An edge that is not distinguishable from noise should not move the
            # score much in either direction.
            edge = 50.0 + (edge - 50.0) * 0.35

        omega = (0.34 * pct(pops, nz(x.get("pop"))) + 0.28 * pct(evs, nz(x.get("ev_pct")))
                 + 0.14 * edge + 0.12 * liq + 0.07 * dir_score + 0.05 * iv_score)
        ad = abs(nz(x.get("delta")))
        if ad < 0.15 or ad > 0.85:
            omega *= 0.90
        if not x.get("cheap"):
            omega *= 0.93
        if not x.get("liquid"):
            omega *= 0.85
        # A pick whose counted edge is merely noise must not outrank one that is
        # actually evidenced at this sample size.
        if x.get("edge_signif") == "เธขเธฑเธเนเธกเนเธเธฑเธ”เน€เธเธ":
            omega *= 0.95
        x["omega"] = round(float(np.clip(nz(omega), 0, 100)), 1)
        x["verdict"] = ("ULTIMATE" if x["omega"] >= 72 else "HIGH" if x["omega"] >= 62 else
                        "WATCH" if x["omega"] >= 52 else "NONE")


def _horizon_reason(best: dict | None, top: dict | None, kind: str, dte: int,
                    spot: float, em: float, net: float) -> str:
    """One plain-language sentence: why this strike at this expiry.

    Everything here is either a fact about the contract (strike, breakeven, DTE)
    or a count over real prices.  The model appears only as the comparison it
    actually is -- the chance the premium already charges you for.
    """
    pick = best or top
    if not pick:
        return f"เนเธกเนเธเธเธชเธฑเธเธเธฒ {kind} เธ—เธตเนเธกเธตเธชเธ เธฒเธเธเธฅเนเธญเธเธเธญเนเธเธเนเธงเธ {dte} เธงเธฑเธเธเธตเน"
    strike = nz(pick.get("strike"))
    dist = (strike / spot - 1) * 100 if spot else 0.0
    # Where the strike sits relative to spot is a plain fact; the wording must not
    # imply a direction the trade does not need.  A PUT can be struck above spot
    # and still be the pick, so describe the position, not the expectation.
    where = ("เธชเธนเธเธเธงเนเธฒเธฃเธฒเธเธฒเธเธฑเธเธเธธเธเธฑเธ" if dist >= 0 else "เธ•เนเธณเธเธงเนเธฒเธฃเธฒเธเธฒเธเธฑเธเธเธธเธเธฑเธ")
    bits = []
    if kind == "CALL":
        bits.append(f"เธ•เนเธญเธเธเธดเธ”เน€เธซเธเธทเธญ ${pick['breakeven']:g} เธ เธฒเธขเนเธ {dte} เธงเธฑเธ "
                    f"(strike ${strike:g} {where} {abs(dist):.1f}%)")
    else:
        bits.append(f"เธ•เนเธญเธเธเธดเธ”เนเธ•เน ${pick['breakeven']:g} เธ เธฒเธขเนเธ {dte} เธงเธฑเธ "
                    f"(strike ${strike:g} {where} {abs(dist):.1f}%)")

    wc, n = pick.get("hist_win_count"), pick.get("hist_n")
    if wc is not None and n:
        bits.append(f"เธเธฑเธเธเธฃเธดเธเธเธฒเธเธเธฃเธฐเธงเธฑเธ•เธด {n} เธฃเธญเธเธชเธนเธ•เธฃเน€เธ”เธตเธขเธงเธเธฑเธ: เธเธณเนเธฃ {wc} เธฃเธญเธ "
                    f"= {nz(pick.get('pop')):.0f}% "
                    f"(เธเนเธงเธ 95% {nz(pick.get('pop_lo')):.0f}โ€“{nz(pick.get('pop_hi')):.0f}%)")
    neff = pick.get("hist_n_eff")
    if neff and n and neff < n * 0.9:
        # Overlapping windows are not independent evidence, and hiding that
        # would make the far expiries look more certain than they are.
        bits.append(f"เธฃเธญเธเธ—เธตเนเน€เธเนเธเธญเธดเธชเธฃเธฐเธเธฃเธดเธเธเธฃเธฐเธกเธฒเธ“ {neff} เธฃเธญเธ (เธซเธเนเธฒเธ•เนเธฒเธเธเนเธญเธเธเธฑเธ)")

    pnl = pick.get("hist_avg_pnl")
    if pnl is not None:
        pf = pick.get("hist_profit_factor")
        bits.append(f"เธเธณเนเธฃเน€เธเธฅเธตเนเธขเธ•เนเธญเธชเธฑเธเธเธฒ ${pnl:+.2f}"
                    + (f" ยท เธเธณเนเธฃ/เธเธฒเธ”เธ—เธธเธเธฃเธงเธก {pf:.2f} เน€เธ—เนเธฒ" if pf is not None else ""))

    if pick.get("model_pop") is not None:
        bits.append(f"เธฃเธฒเธเธฒเธ—เธตเนเธ•เธฅเธฒเธ”เน€เธฃเธตเธขเธเธญเธขเธนเนเธเธดเธ”เน€เธเนเธเนเธญเธเธฒเธชเธฃเธฒเธง {nz(pick.get('model_pop')):.0f}% "
                    f"โ’ เธ•เนเธฒเธเธเธฒเธเธเธญเธเธเธฃเธดเธ {nz(pick.get('hist_edge')):+.0f} เธเธธเธ”")

    sig = pick.get("edge_signif")
    if sig == "เธกเธตเธเธฑเธขเธชเธณเธเธฑเธ":
        bits.append("เธชเนเธงเธเธ•เนเธฒเธเธเธตเนเธเนเธฒเธเธเธฒเธฃเธ—เธ”เธชเธญเธเธ—เธฒเธเธชเธ–เธดเธ•เธดเธ—เธตเนเธเธณเธเธงเธเธ•เธฑเธงเธญเธขเนเธฒเธเธเธตเน (เธเธฑเธเธฃเธงเธกเธเธฒเธฃเน€เธ—เธตเธขเธเธซเธฅเธฒเธข strike เนเธฅเนเธง)")
    elif sig == "เนเธขเนเธเธงเนเธฒเธ•เธฅเธฒเธ”เธเธฑเธ”เน€เธเธ":
        bits.append("เธฃเธฐเธงเธฑเธ: เธชเธ–เธดเธ•เธดเธเธฃเธดเธเนเธขเนเธเธงเนเธฒเธฃเธฒเธเธฒเธ—เธตเนเธ•เธฅเธฒเธ”เน€เธฃเธตเธขเธเธญเธขเนเธฒเธเธเธฑเธ”เน€เธเธ")
    elif sig == "เธขเธฑเธเนเธกเนเธเธฑเธ”เน€เธเธ":
        bits.append("เธขเธฑเธเธเธญเธเนเธกเนเนเธ”เนเธงเนเธฒเธชเนเธงเธเธ•เนเธฒเธเธเธตเนเน€เธเนเธเธเธญเธเธเธฃเธดเธเธซเธฃเธทเธญเธเธงเธฒเธกเธเธฑเธเน€เธญเธดเธ")

    if pick.get("cheap"):
        bits.append("เธเธฃเธตเน€เธกเธตเธขเธกเธ–เธนเธ (เนเธกเนเน€เธเธดเธ 3% เธเธญเธเธฃเธฒเธเธฒเธซเธธเนเธ)")
    if pick.get("liquid"):
        bits.append(f"เธเธฃเธดเธกเธฒเธ“/เธชเธ–เธฒเธเธฐเธเธเธเนเธฒเธเน€เธเธตเธขเธเธเธญ (spread {nz(pick.get('spread_pct')):.0f}%)")
    # A very wide quoted spread is a real cost even when volume looks fine, so it
    # is worth naming when it is the thing holding the pick back.
    elif nz(pick.get("spread_pct")) > 25:
        bits.append(f"เธฃเธฐเธงเธฑเธ: spread เธเธงเนเธฒเธ {nz(pick.get('spread_pct')):.0f}% (เธเธฒเธ”เธ—เธธเธเธ—เธฑเธเธ—เธตเธ—เธตเนเธเธทเนเธญ)")
    bits.append(f"เธ•เธฅเธฒเธ”เธเธฒเธ”เนเธเธงเนเธ ยฑ${em:.2f} เธเธเธ–เธถเธเธงเธฑเธเธซเธกเธ”เธญเธฒเธขเธธเธเธตเน")
    return " ยท ".join(bits)


def horizon_view(ch: dict, spot: float, close, high, low, atm_iv: float, sigma_hist: float,
                 logret, net: float, ivr: float, earnings: bool,
                 dist_full_fn=None) -> dict:
    """One answer per *real* expiry date in the chain.

    Options do not expire in neat "1 day / 3 day / 7 day" buckets -- every
    contract carries its own date, and that date is the only thing the trade
    actually has to obey.  So instead of forcing three arbitrary windows and then
    apologising when no matching expiry exists, this walks the expiry dates the
    chain really lists, keeps the nearest few that are still worth trading, and
    answers each one on its own time to expiry.

    Keying by the expiry date also removes a subtle lie in the old code: two
    different "horizons" could resolve to the same date and show identical
    contracts twice, once labelled 3 days and once labelled 5.

    Every number for an expiry is computed at *that* expiry's own DTE -- its own
    historical window length, its own simulation, its own expected move -- so
    nothing is rescaled from a neighbouring date.
    """
    today = datetime.now(timezone.utc).date()
    exps: dict = {}
    for r in ch.get("rows") or []:
        try:
            d = (datetime.strptime(r["expiry"], "%Y-%m-%d").date() - today).days
        except Exception:
            continue
        if MIN_DTE <= d <= MAX_DTE:
            exps[r["expiry"]] = d
    if not exps:
        return {}

    # Nearest expiries first; a chain can list a dozen and the far ones are not
    # what a short-dated trader is choosing between.
    ordered = sorted(exps.items(), key=lambda kv: (kv[1], kv[0]))[:MAX_EXPIRY_VIEWS]

    out: dict = {}
    for exp, dte in ordered:
        T = max(dte / 365.0, 1e-4)
        n_days = max(int(round(dte * 5 / 7)), 1)
        sigma_mix = float(np.clip(0.65 * atm_iv + 0.35 * sigma_hist, 0.03, 4.0))
        mu = RF + 0.10 * net
        simset = sim.simulate(spot, T, n_days, sigma_mix, mu, logret,
                              sims=max(1200, SIMS // 2), seed=_seed(exp) % 2_000_000_000)
        dist = sim.hist_distributions(close, high, low, n_days, HIST_WINDOW)
        dist_full = (dist_full_fn(close, high, low, n_days, 504) if dist_full_fn
                     else sim.hist_distributions_full(close, high, low, n_days, 504))
        em = quant.expected_move(spot, sigma_mix, T)
        ctx = dict(signal=net, em_pct=round(em / spot * 100, 2), n_days=n_days, months=6,
                   vrp=0.0, iv_rank=ivr, earnings=earnings, tech={}, flow={}, spot=spot,
                   sigma_hist=sigma_hist * 100, news_n=0,
                   hist_n=(dist_full["n"] if dist_full else 0), dte=dte, expiry=exp)

        lo, hi = spot * (1 - STRIKE_BAND), spot * (1 + STRIKE_BAND)
        rows = [r for r in ch["rows"] if r["expiry"] == exp and lo <= r["strike"] <= hi]
        if not rows:
            rows = [r for r in ch["rows"] if r["expiry"] == exp]
        by_strike: dict = {}
        for r in rows:
            r = dict(r)
            by_strike.setdefault(r["strike"], {})[r["kind"]] = r
        all_k = sorted(by_strike, key=lambda k: abs(k - spot))
        keep = set(all_k[:MAX_STRIKES // 2]
                   + sorted([k for k in all_k if k <= spot])[-5:]
                   + sorted([k for k in all_k if k > spot])[:5])
        calls, puts = [], []
        for k in sorted(keep):
            for kind, bucket in (("CALL", calls), ("PUT", puts)):
                r = by_strike[k].get(kind)
                if not r:
                    continue
                st = build_strike(exp, r, spot, kind, dte, n_days, simset, dist,
                                  close, atm_iv, sigma_hist, T, ctx, dist_full)
                if st:
                    bucket.append(st)
        _sl_rank(calls, net, ivr)
        _sl_rank(puts, net, ivr)

        def pick_side(group):
            if not group:
                return None
            # chance first, then score: the question is explicitly about the
            # highest win probability, not the cheapest ticket
            strong = [x for x in group if x.get("cheap") and x.get("liquid")]
            pool = strong or group
            return max(pool, key=lambda x: (nz(x.get("pop")), x["omega"]))

        tc = pick_side(calls)
        tp = pick_side(puts)
        out[exp] = dict(
            expiry=exp, dte=dte, n_days=n_days,
            weekday=_WEEKDAY_TH[datetime.strptime(exp, "%Y-%m-%d").weekday()],
            label=f"{exp} ยท {_WEEKDAY_TH[datetime.strptime(exp, '%Y-%m-%d').weekday()]} (เธญเธตเธ {dte} เธงเธฑเธ)",
            expected_move=round(em, 2), em_low=round(spot - em, 2), em_high=round(spot + em, 2),
            sigma=round(sigma_mix * 100, 1),
            call=slim_strike(tc), put=slim_strike(tp),
            n_calls=len(calls), n_puts=len(puts),
            call_reason=_horizon_reason(tc, tc, "CALL", dte, spot, em, net),
            put_reason=_horizon_reason(tp, tp, "PUT", dte, spot, em, net),
            winner=("CALL" if nz((tc or {}).get("pop")) >= nz((tp or {}).get("pop")) else "PUT"),
        )
    return out


# --------------------------------------------------------------------------- #
# per-ticker analysis
# --------------------------------------------------------------------------- #
def analyze(sym: str) -> dict | None:
    sym = re.sub(r"[^A-Z]", "", sym.upper())[:6]
    if not sym:
        return None

    # Bound how many full analyses run at once.  A single analyze() peaks around
    # 25MB of pandas/numpy working set, and each one also opens its own fetch
    # pool, so N of them at once is N x that plus N pools of threads.  On a
    # 512MB free instance, letting a scan, a diag probe and a detail view pile up
    # was what pushed the worker into an OOM kill mid-cycle.  Serialising past
    # ANALYZE_SLOTS keeps the ceiling flat no matter who asks for work.
    with _ANALYZE_SLOTS:
        # A hard deadline on the whole analysis.  Every individual fetch has its
        # own timeout, but a name lookup can still hang for minutes on a free
        # instance -- and one hung analysis holds a slot forever, so the rest
        # of the watchlist piles up behind it and the scan never finishes.
        # This bounds the damage: the slot is released, the ticker is reported
        # as failed, and the cycle carries on.
        from concurrent.futures import TimeoutError as FTO

        _trace(sym, "queued")
        fut = _ANALYZE_POOL.submit(_analyze_inner, sym)
        try:
            out = fut.result(timeout=ANALYZE_TIMEOUT)
            _trace(sym, "done")
            return out
        except FTO:
            log.warning("analyze %s: exceeded %ss -- abandoning it", sym, ANALYZE_TIMEOUT)
            _trace(sym, "timeout")
            return None


def _analyze_inner(sym: str) -> dict | None:
    _T0 = time.time()
    _trace(sym, "fetch", _T)
    # The four slowest steps are independent network fetches, so run them
    # together instead of one after another.  On a free instance this is the
    # difference between a scan that finishes and one that appears to hang.
    # Every future also gets a hard deadline: a wedged socket must never be
    # able to block the whole scan, which is what kept /api/scan empty.
    #
    # The deadlines are deliberately tight and the whole group shares one
    # budget, because they are *waited on in sequence*: four separate 45s
    # deadlines would let a single ticker occupy a worker for up to 150s, which
    # across the watchlist is exactly the "scan hangs at 0/15" symptom.  Every
    # provider that actually answers from these hosts does so in under 2s.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

    # The budget for the four price fetches.  Generous enough for a slow
    # instance: a ticker that cannot be priced at all is worth less than one
    # that is priced a little later.
    budget = time.time() + 40.0

    def take(fut, fallback):
        left = max(budget - time.time(), 1.0)
        try:
            return fut.result(timeout=left)
        except FTimeout:
            log.warning("analyze %s: provider deadline hit", sym)
            return fallback
        except Exception:
            return fallback

    pool = io_pool()
    _T = {}
    try:
        _t0 = time.time()
        f_daily = pool.submit(data.daily, sym)
        f_intra = pool.submit(data.intraday, sym)
        f_chain = pool.submit(data.chain, sym)
        f_name = pool.submit(data.company_name, sym)
        d = take(f_daily, None)
        intra = take(f_intra, None)
        ch = take(f_chain, None)
        name = take(f_name, "")
        # The news fan-out starts here, while the price fetches above are still
        # running, instead of strictly after them.  The ticker then waits for
        # the slower of the two instead of for their sum -- on a slow instance
        # that alone is ~20s per ticker, and ~75s over the watchlist.
        f_news = pool.submit(news.for_ticker, sym, name)
        _T["fetch"] = int((time.time() - _t0) * 1000)
        try:
            ns = f_news.result(timeout=20.0)
        except FTimeout:
            log.warning("analyze %s: news deadline hit -- continuing without it", sym)
            ns = {}
        except Exception as ex:
            log.info("analyze %s: news failed (%s)", sym, type(ex).__name__)
            ns = {}
        _T["news"] = int((time.time() - _t0) * 1000) - _T["fetch"]
        _trace(sym, "signals", _T)
    finally:
        # The pool is shared, so it is not shut down here: the stragglers are
        # left to finish on their own, and the scan keeps moving either way.
        pass

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

    _t0 = time.time()
    c_sig, c_info = flow_signal(d, intra, ch)
    d_sig, d_info = tech_signal(d, _spy_benchmark())
    e_sig, e_info = news_signal(ns)
    _T["signals"] = int((time.time() - _t0) * 1000)
    # every signal is forced finite here: one NaN from a feed must not be able to
    # reach the drift (mu) and quietly turn a whole simulation into NaN
    c_sig, d_sig, e_sig = nz(c_sig), nz(d_sig), nz(e_sig)

    # ---- how much of the news is actually new? -----------------------------
    # A dead feed is not evidence of a quiet market.  When nothing arrives the
    # news term contributes 0.0, which silently drags `net` toward zero and makes
    # every ticker look equally undecided.  So freshness is measured first, and
    # the news weight is either used or handed to the signals that do exist.
    _items = ns.get("items") or []
    _fresh = ns.get("fresh")
    if _fresh is None:
        _fresh = 0
        for _it in _items:
            # `age` is published by news.for_ticker; `h` is the rounded form kept
            # for older payloads.  Unknown defaults to one day old, never stale.
            _av = _it.get("age")
            if _av is None:
                _av = _it.get("h")
            try:
                _a = 24.0 if _av is None else float(_av)
            except (TypeError, ValueError):
                _a = 24.0
            if _a <= 72:
                _fresh += 1
    else:
        _fresh = int(_fresh)
    if _fresh >= 3:
        news_state, news_note = "เธกเธตเธเนเธฒเธงเนเธซเธกเน", f"เธเนเธฒเธงเนเธซเธกเน {_fresh} เธเธดเนเธเนเธ 72 เธเธก."
    elif _fresh >= 1:
        news_state, news_note = "เธเนเธฒเธงเธเธฒเธ", f"เธเนเธฒเธงเนเธซเธกเนเน€เธเธตเธขเธ {_fresh} เธเธดเนเธเนเธ 72 เธเธก."
    elif _items:
        news_state, news_note = "เธกเธตเนเธ•เนเธเนเธฒเธงเน€เธเนเธฒ", "เธเนเธฒเธงเธ—เธตเนเธ”เธถเธเนเธ”เนเน€เธเนเธฒเธเธงเนเธฒ 3 เธงเธฑเธเธ—เธฑเนเธเธซเธกเธ”"
    else:
        news_state, news_note = "เนเธกเนเธกเธตเธเนเธฒเธง", "เธ”เธถเธเธเนเธฒเธงเนเธกเนเนเธ”เนเน€เธฅเธขเนเธเธฃเธญเธเธเธตเน"

    if _fresh >= 1:
        weights = dict(flow=0.34, tech=0.46, news=0.20)
        _net = 0.34 * c_sig + 0.46 * d_sig + 0.20 * e_sig
    else:
        # Redistribute the news weight across flow and technicals while keeping
        # their relative importance.  Skipping this is what made an empty feed
        # look like a balanced market instead of an unknown one.
        _w0 = 0.34 + 0.46
        weights = dict(flow=round(0.34 / _w0, 3), tech=round(0.46 / _w0, 3), news=0.0)
        _net = (0.34 * c_sig + 0.46 * d_sig) / _w0
    net = float(np.clip(nz(_net), -1, 1))
    bias = "CALL" if net >= 0 else "PUT"
    _t0 = time.time()
    emap = earnings_map()
    earn = emap.get(sym)
    _T["earn"] = int((time.time() - _t0) * 1000)

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
                news_state=news_state, news_note=news_note, news_fresh=_fresh,
                news_total=ns.get("n", len(_items)), weights=weights,
                tech_votes=tech_votes(d_info),
                news={k: v for k, v in e_info.items()},
                news_detail=dict(items=ns.get("items", [])[:12], sources=ns.get("sources", {}),
                                 depth=ns.get("depth", {}), checks=ns.get("checks", []),
                                 n=ns.get("n", 0), engine=ns.get("engine"), social=ns.get("social", {}),
                                 filings=ns.get("filings", [])[:6], note=ns.get("note", "")),
                atm_iv=round(atm_iv * 100, 1), iv30=round(iv30, 1) if iv30 else None,
                iv_rank=round(ivr, 1), iv_pct=round(ivp, 1), vrp=round(vrp, 1),
                sigma_hist=round(sigma_hist * 100, 1),
                earnings=earn, calls=[], puts=[], error=None)
    _T["total"] = int((time.time() - _T0) * 1000)
    base["timings"] = _T

    if not (ch and ch.get("rows")):
        base.update(error="เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเธญเธญเธเธเธฑเธ (CBOE) เธชเธณเธซเธฃเธฑเธเธซเธธเนเธเธเธตเน", verdict="NONE", omega=0.0)
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
        base.update(error=f"เนเธกเนเธกเธตเธญเธญเธเธเธฑเธเธซเธกเธ”เธญเธฒเธขเธธเนเธเธเนเธงเธ {MIN_DTE}-{MAX_DTE} เธงเธฑเธ", verdict="NONE", omega=0.0)
        return base
    exp = min(exps, key=lambda e: abs(exps[e] - TARGET_DTE) + (0 if 2 <= exps[e] <= 12 else 4))
    dte = exps[exp]
    T = max(dte / 365.0, 1e-4)
    n_days = max(int(round(dte * 5 / 7)), 1)

    sigma_mix = float(np.clip(0.65 * atm_iv + 0.35 * sigma_hist, 0.03, 4.0))
    mu = RF + 0.10 * net
    _t0 = time.time()
    simset = sim.simulate(spot, T, n_days, sigma_mix, mu, logret, sims=SIMS, seed=_seed(sym) % 2_000_000_000)
    _T["sim"] = int((time.time() - _t0) * 1000)
    _trace(sym, "history", _T)
    _t0 = time.time()
    dist = sim.hist_distributions(close, high, low, n_days, HIST_WINDOW)
    # The same rolling study over the *whole* downloaded history (up to two
    # years).  This is the source of the headline probability: it is a count of
    # what actually happened, so a bigger sample is strictly better, while the
    # six-month window is kept for the wording and the recent-vs-older split.
    dist_full = sim.hist_distributions_full(close, high, low, n_days, 504)
    _T["hist"] = int((time.time() - _t0) * 1000)
    _trace(sym, "chain", _T)
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
            st = build_strike(sym, r, spot, kind, dte, n_days, simset, dist, close, atm_iv, sigma_hist, T, ctx, dist_full)
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

    # ---- answer for every real expiry in the chain --------------------------
    # The auto-chosen expiry above is one opinion.  This runs the same machinery
    # once per expiry the chain actually lists, so each date gets a truthful,
    # self-contained best CALL and best PUT with the reason why.
    try:
        base["expiries"] = horizon_view(ch, spot, close, high, low, atm_iv, sigma_hist,
                                        logret, net, ivr, bool(earn),
                                        sim.hist_distributions_full)
    except Exception as ex:
        log.info("expiries %s: %s", sym, ex)
        base["expiries"] = {}

    # ---- narrative ---------------------------------------------------------
    trend = ("เธเธฒเธเธถเนเธ" if d_info.get("stack") == 1 else "เธเธฒเธฅเธ" if d_info.get("stack") == -1 else "เนเธฃเนเธ—เธดเธจเธ—เธฒเธ")
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
        regime=dict(trend=trend, iv_tag="เธ–เธนเธ" if ivr < 30 else ("เนเธเธ" if ivr > 70 else "เธเธฅเธฒเธ"),
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
    _tv = base.get("tech_votes") or []
    _vc = sum(1 for r in _tv if r["dir"] == "CALL")
    _vp = sum(1 for r in _tv if r["dir"] == "PUT")
    base["tech_tally"] = dict(call=_vc, put=_vp, neutral=len(_tv) - _vc - _vp)
    if _fresh >= 1:
        guide.append(f"เธเนเธฒเธง: {news_state} โ€” {news_note} "
                     f"(เธเนเธณเธซเธเธฑเธเธเนเธฒเธง {weights['news'] * 100:.0f}%)")
    else:
        guide.append(f"โ  เนเธกเนเธกเธตเธเนเธฒเธงเนเธซเธกเน ({news_note}) โ€” เธฃเธญเธเธเธตเนเธ•เธฑเธ”เธชเธดเธเธเธฒเธเน€เธ—เธเธเธดเธ {_vc} เธ•เธฑเธงเธเธตเน CALL / "
                     f"{_vp} เธ•เธฑเธงเธเธตเน PUT เนเธฅเธฐเธเธฅเธเธเธฐเธเธฃเธดเธเธขเนเธญเธเธซเธฅเธฑเธ 6โ€“12 เน€เธ”เธทเธญเธ "
                     f"เนเธ”เธขเธขเธเธเนเธณเธซเธเธฑเธเธเนเธฒเธงเนเธเนเธซเนเธเธฃเธฐเนเธชเน€เธเธดเธเนเธฅเธฐเน€เธ—เธเธเธดเธเนเธ—เธ โ€” "
                     f"เนเธกเนเนเธ”เนเธซเธกเธฒเธขเธเธงเธฒเธกเธงเนเธฒเธ•เธฅเธฒเธ”เน€เธเธตเธขเธ เนเธ•เนเธซเธกเธฒเธขเธเธงเธฒเธกเธงเนเธฒเน€เธฃเธฒเนเธกเนเธฃเธนเนเธงเนเธฒเธกเธตเธญเธฐเนเธฃเน€เธเธดเธ”เธเธถเนเธ")
    guide.append(f"เธซเธธเนเธ {sym} เธฃเธฒเธเธฒ ${spot:.2f} ยท เนเธเธงเนเธเนเธก {trend} ยท เนเธกเน€เธกเธเธ•เธฑเธก 21 เธงเธฑเธ {d_info.get('mom21')}% "
                 f"ยท RSI {d_info.get('rsi')} ยท ADX {d_info.get('adx')}")
    guide.append(f"เธเธงเธฒเธกเธเธฑเธเธเธงเธ: IV {atm_iv*100:.0f}% (เธญเธฑเธเธ”เธฑเธ {ivr:.0f}/100 = {base['regime']['iv_tag']}) "
                 f"ยท เธเนเธฒเธเธงเธฒเธกเธเธฑเธเธเธงเธเธเธฃเธดเธ 6 เน€เธ”เธทเธญเธ {sigma_hist*100:.0f}% ยท เธชเนเธงเธเธ•เนเธฒเธ {vrp:+.1f} เธเธธเธ”")
    if base["hist"].get("avg_abs_move") is not None:
        guide.append(f"เธ•เธฅเธฒเธ”เธเธฒเธ”เธเธฒเธฃเนเธเธงเนเธ ยฑ{em:.2f} (${spot-em:.2f} โ€“ ${spot+em:.2f}) เธ เธฒเธขเนเธ {dte} เธงเธฑเธ ยท "
                     f"เธชเธ–เธดเธ•เธด 6 เน€เธ”เธทเธญเธเธเนเธฒเน€เธเธฅเธตเนเธขเนเธเธงเนเธ {base['hist']['avg_abs_move']}% เธ•เนเธญ {n_days} เธงเธฑเธเธ—เธณเธเธฒเธฃ "
                     f"ยท เนเธ 6 เน€เธ”เธทเธญเธเธซเธธเนเธเธ•เธฑเธงเธเธตเนเธเธดเธ”เธเธงเธเนเธเธฃเธญเธ {n_days} เธงเธฑเธเธ—เธณเธเธฒเธฃ {base['hist'].get('up_rate')}% เธเธญเธเธฃเธญเธ")
    else:
        guide.append(f"เธ•เธฅเธฒเธ”เธเธฒเธ”เธเธฒเธฃเนเธเธงเนเธ ยฑ{em:.2f} (${spot-em:.2f} โ€“ ${spot+em:.2f}) เธ เธฒเธขเนเธ {dte} เธงเธฑเธ ยท "
                     f"โ  เธ”เธถเธเธฃเธฒเธเธฒเธขเนเธญเธเธซเธฅเธฑเธเนเธกเนเธชเธณเน€เธฃเนเธเนเธเธฃเธญเธเธเธตเน เธเธถเธเธขเธฑเธเนเธกเนเธกเธตเธชเธ–เธดเธ•เธด 6 เน€เธ”เธทเธญเธ "
                     f"(เธ•เธฑเธงเน€เธฅเธเนเธญเธเธฒเธชเธเธเธฐเธขเธฑเธเนเธเนเนเธ”เน เนเธ•เนเนเธกเนเธกเธตเธเธฒเธฃเธขเธทเธเธขเธฑเธเธ”เนเธงเธขเธชเธ–เธดเธ•เธดเธเธฃเธดเธ)")
    if primary:
        guide.append(f"เธ”เธตเธฅเธ—เธตเนเธเธธเนเธกเธเนเธฒเธ—เธตเนเธชเธธเธ” (เนเธฅเธฐเธ•เธฃเธเธเธฑเธเธ—เธดเธจเธ—เธฒเธเธ—เธตเนเนเธกเน€เธ”เธฅเน€เธซเนเธ): {bias} โ€” {primary['headline']} ยท "
                     f"EV {primary['ev']:+.2f}$ ({_pct_txt(primary.get('ev_pct'))} เธเธญเธเธเธฃเธตเน€เธกเธตเธขเธก) ยท "
                     f"เธเธงเธฒเธกเน€เธเธทเนเธญเธกเธฑเนเธเธเนเธญเธกเธนเธฅ {primary['reliability']:.0f}/100")
    if top_call and top_put:
        guide.append(f"เธ–เนเธฒเน€เธญเธฒ โ€เนเธญเธเธฒเธชเธเธเธฐโ€ เน€เธเนเธเธ•เธฑเธงเธ•เธฑเนเธเธญเธขเนเธฒเธเน€เธ”เธตเธขเธง: CALL เธชเธนเธเธชเธธเธ”เธ—เธตเน ${top_call['strike']:g} "
                     f"({top_call['pop']:.0f}%) ยท PUT เธชเธนเธเธชเธธเธ”เธ—เธตเน ${top_put['strike']:g} ({top_put['pop']:.0f}%) "
                     f"โ€” เธ”เธนเธฃเธฒเธขเธฅเธฐเน€เธญเธตเธขเธ”เนเธฅเธฐเน€เธซเธ•เธธเธเธฅเนเธ”เนเนเธเธ•เธฒเธฃเธฒเธ Strike เธ”เนเธฒเธเธฅเนเธฒเธ")
    base["guide"] = guide

    # ---- the direct answer: which strike wins, and why ----------------------
    ans = []
    if _fresh < 1:
        ans.append(f"เธซเธกเธฒเธขเน€เธซเธ•เธธเธชเธณเธเธฑเธ: {news_note} โ€” เธเธณเธ•เธญเธเธ”เนเธฒเธเธฅเนเธฒเธเธเธถเธเธกเธฒเธเธฒเธเน€เธ—เธเธเธดเธ "
                   f"{_vc} เธ•เธฑเธงเธเธตเน CALL / {_vp} เธ•เธฑเธงเธเธตเน PUT เนเธฅเธฐเธเธฒเธฃเธเธฑเธเธเธฅเธเธฃเธดเธเธขเนเธญเธเธซเธฅเธฑเธ "
                   f"6โ€“12 เน€เธ”เธทเธญเธ เนเธกเนเนเธ”เนเธกเธฒเธเธฒเธเธเนเธฒเธง")
    if top_call:
        ans.append(f"เธ–เนเธฒเธ–เธฒเธกเธงเนเธฒ โ€CALL strike เนเธซเธเนเธญเธเธฒเธชเธเธเธฐเธชเธนเธเธชเธธเธ”โ€ โ’ CALL ${top_call['strike']:g} "
                   f"(เธซเธกเธ”เธญเธฒเธขเธธ {top_call['expiry']}) เนเธญเธเธฒเธชเธเธเธฐ {top_call['pop']:.0f}% "
                   f"(เธเนเธงเธ {top_call['pop_lo']:.0f}โ€“{top_call['pop_hi']:.0f}%) เธเธฃเธตเน€เธกเธตเธขเธก ${top_call['premium']} "
                   f"เธเธธเธ”เธเธธเนเธกเธ—เธธเธ ${top_call['breakeven']} โ€” เน€เธซเธ•เธธเธเธฅ: {top_call['why'][0] if top_call['why'] else ''}")
    if top_put:
        ans.append(f"เธ–เนเธฒเธ–เธฒเธกเธงเนเธฒ โ€PUT strike เนเธซเธเนเธญเธเธฒเธชเธเธเธฐเธชเธนเธเธชเธธเธ”โ€ โ’ PUT ${top_put['strike']:g} "
                   f"(เธซเธกเธ”เธญเธฒเธขเธธ {top_put['expiry']}) เนเธญเธเธฒเธชเธเธเธฐ {top_put['pop']:.0f}% "
                   f"(เธเนเธงเธ {top_put['pop_lo']:.0f}โ€“{top_put['pop_hi']:.0f}%) เธเธฃเธตเน€เธกเธตเธขเธก ${top_put['premium']} "
                   f"เธเธธเธ”เธเธธเนเธกเธ—เธธเธ ${top_put['breakeven']} โ€” เน€เธซเธ•เธธเธเธฅ: {top_put['why'][0] if top_put['why'] else ''}")
    if primary:
        ans.append(f"เธ–เนเธฒเธ–เธฒเธกเธงเนเธฒ โ€เธชเธฑเธเธเธฒเธ—เธตเนเธเธธเนเธกเธเนเธฒเธ—เธตเนเธชเธธเธ” (เนเธญเธเธฒเธชเธเธเธฐ + เธกเธนเธฅเธเนเธฒ + เธชเธ เธฒเธเธเธฅเนเธญเธ)โ€ โ’ {primary['kind']} "
                   f"${primary['strike']:g} เนเธญเธเธฒเธชเธเธเธฐ {primary['pop']:.0f}% EV {primary['ev']:+.2f}$ "
                   f"({_pct_txt(primary.get('ev_pct'))} เธเธญเธเธเธฃเธตเน€เธกเธตเธขเธก)"
                   + (f" เธชเธ–เธดเธ•เธด 6 เน€เธ”เธทเธญเธเนเธเธเน€เธ”เธตเธขเธงเธเธฑเธเธเธเธฐ {primary['hist_win']:.0f}%"
                      if primary.get("hist_win") is not None else ""))
    if top_call and top_put:
        win = "CALL" if top_call["pop"] >= top_put["pop"] else "PUT"
        ans.append(f"เธชเธฃเธธเธเธชเธฑเธเธ”เธฒเธซเนเธเธตเน: เธเธฑเนเธ {win} เนเธซเนเธเธงเธฒเธกเธเนเธฒเธเธฐเน€เธเนเธเธเธเธฐเธชเธนเธเธเธงเนเธฒ "
                   f"(CALL เธชเธนเธเธชเธธเธ” {top_call['pop']:.0f}% ยท PUT เธชเธนเธเธชเธธเธ” {top_put['pop']:.0f}%) "
                   f"เนเธ•เนเธเธฐเนเธเธเธเธงเธฒเธกเธเธธเนเธกเธเนเธฒเน€เธเธฅเธตเนเธขเธเธญเธเธ—เธฑเนเธเธชเธญเธเธเธฑเนเธเธ•เนเธญเธเธ”เธน EV เธเธฃเธฐเธเธญเธ โ€” "
                   + ("เธเธฑเนเธ CALL เธกเธต EV เน€เธเนเธเธเธงเธ" if (top_call["ev_pct"] or 0) > 0 else
                      "เธ—เธฑเนเธเธชเธญเธเธเธฑเนเธ EV เธขเธฑเธเธ•เธดเธ”เธฅเธ เธเธถเนเธเน€เธเนเธเน€เธฃเธทเนเธญเธเธเธเธ•เธดเธเธญเธเธญเธญเธเธเธฑเธเธเธทเนเธญเน€เธ”เธตเนเธขเธง (long option) "
                      "เน€เธเธฃเธฒเธฐเธ•เนเธญเธเธเนเธฒเธขเธเนเธฒเธเธงเธฒเธกเธเธฑเธเธเธงเธเธฅเนเธงเธเธซเธเนเธฒ"))
    base["answer"] = ans

    cands = [c for c in (calls + puts) if c["cheap"] and c["liquid"]]
    cands.sort(key=lambda x: -x["omega"])
    base["candidates"] = cands[:8]
    base["all_cheap"] = sorted([c for c in (calls + puts) if c["cheap"]], key=lambda x: -x["omega"])[:10]
    return base


# --------------------------------------------------------------------------- #
# the position the user already holds
# --------------------------------------------------------------------------- #
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# What each SEC form actually means, so the evidence table is readable instead
# of repeating "4 โ€” FORM 4".
FORM_TH = {
    "3": "เนเธเนเธเธเธฒเธฃเธ–เธทเธญเธซเธธเนเธเธเธฃเธฑเนเธเนเธฃเธเธเธญเธเธเธนเนเธเธฃเธดเธซเธฒเธฃ",
    "4": "เนเธเนเธเธเธฒเธฃเธเธทเนเธญ/เธเธฒเธขเธซเธธเนเธเธเธญเธเธเธนเนเธเธฃเธดเธซเธฒเธฃ (เธเธเธงเธเนเธ)",
    "5": "เธฃเธฒเธขเธเธฒเธเธเธฒเธฃเธ–เธทเธญเธซเธธเนเธเธเธฃเธฐเธเธณเธเธตเธเธญเธเธเธนเนเธเธฃเธดเธซเธฒเธฃ",
    "8-K": "เน€เธซเธ•เธธเธเธฒเธฃเธ“เนเธชเธณเธเธฑเธเธ—เธตเนเธ•เนเธญเธเนเธเนเธเธ—เธฑเธเธ—เธต",
    "10-Q": "เธเธเธเธฒเธฃเน€เธเธดเธเธฃเธฒเธขเนเธ•เธฃเธกเธฒเธช",
    "10-K": "เธเธเธเธฒเธฃเน€เธเธดเธเธฃเธฒเธขเธเธต",
    "SC 13D": "เธกเธตเธเธนเนเธ–เธทเธญเธซเธธเนเธเธฃเธฒเธขเนเธซเธเนเน€เธเธดเธ 5% (เน€เธเธดเธเธฃเธธเธ)",
    "SC 13G": "เธกเธตเธเธนเนเธ–เธทเธญเธซเธธเนเธเธฃเธฒเธขเนเธซเธเนเน€เธเธดเธ 5% (เน€เธเธดเธเธฃเธฑเธ)",
    "S-1": "เธขเธทเนเธเธเธ”เธ—เธฐเน€เธเธตเธขเธเธซเธธเนเธเนเธซเธกเน",
    "424B": "เธเธฒเธขเธซเธธเนเธเน€เธเธดเนเธกเธ—เธธเธ",
    "DEF 14A": "เธซเธเธฑเธเธชเธทเธญเน€เธเธดเธเธเธฃเธฐเธเธธเธกเธเธนเนเธ–เธทเธญเธซเธธเนเธ",
}


def position_check(sym: str, call_strike=None, put_strike=None,
                   call_prem=None, put_prem=None) -> dict:
    """Score the strikes the user already holds against the live tape.

    The user trades daily and knows exactly which strikes are on: what is
    missing is whether those strikes still have a path today.  Every leg
    reports where the stock sits relative to the strike, how far it still has
    to travel, the chance it closes on the winning side today, and the trend
    evidence behind that number -- plus, separately, the counted history of
    that exact contract, which is the part no model can talk its way around.

    Times are treated honestly: an expiry that is *today* is scored on today's
    close, not on some comfortable multi-day horizon.
    """
    sym = re.sub(r"[^A-Z.]", "", (sym or "").upper())[:6]
    if not sym:
        return dict(error="เนเธชเน ticker เธซเธธเนเธเธชเธซเธฃเธฑเธเธเนเธญเธ เน€เธเนเธ SOFI")

    ch = data.chain(sym)
    if not ch or not ch.get("rows"):
        return dict(ticker=sym, error="เธ”เธถเธ chain เธญเธญเธเธเธฑเธเนเธกเนเนเธ”เนเนเธเธเธ“เธฐเธเธตเน เธฅเธญเธเธญเธตเธเธเธฃเธฑเนเธ")

    d = data.daily(sym)
    intra = data.intraday(sym)
    spot = float(ch.get("spot") or 0.0)
    spot_src = "CBOE (เธเนเธญเธกเธนเธฅเธฅเนเธฒเธเนเธฒ)"
    # The newest traded price beats the quote on the chain: the chain is
    # cached for two minutes, the tape is not.  A position check is about
    # "right now", so it is scored on the last price that actually traded.
    live = data.live_chart(sym, minutes=90, ttl=10)
    live_price = float((live or {}).get("price") or 0.0)
    if live_price > 0:
        spot = live_price
        spot_src = (live or {}).get("source") or spot_src
    if spot <= 0 and len(d):
        spot = float(d["Close"].iloc[-1])
        spot_src = "เธฃเธฒเธเธฒเธเธดเธ”เธงเธฑเธเธฅเนเธฒเธชเธธเธ”"
    if spot <= 0:
        return dict(ticker=sym, error="เนเธกเนเธเธเธฃเธฒเธเธฒเธฅเนเธฒเธชเธธเธ”เธเธญเธเธซเธธเนเธเธเธตเน")

    spy = cache.wrap("d:SPY", 900, lambda: data.daily("SPY"))
    d_sig, d_info = tech_signal(d, spy)
    c_sig, c_info = flow_signal(d, intra, ch)
    votes = tech_votes(d_info)
    tally = dict(call=sum(1 for v in votes if v["dir"] == "CALL"),
                 put=sum(1 for v in votes if v["dir"] == "PUT"))
    tally["neutral"] = len(votes) - tally["call"] - tally["put"]
    # Same flow/tech split the full analysis uses when there is no news weight
    # to spend; a position check must not quietly use a different opinion.
    net = float(np.clip(0.425 * nz(c_sig) + 0.575 * nz(d_sig), -1, 1))

    sigma_hist = 0.40
    if len(d) > 35:
        try:
            sigma_hist = float(np.clip(quant.vol_close(d["Close"], 30), 0.03, 3.0))
        except Exception:
            pass

    today = datetime.now(timezone.utc).date()
    exps = set()
    for r in ch["rows"]:
        try:
            dte = (datetime.strptime(r["expiry"], "%Y-%m-%d").date() - today).days
        except Exception:
            continue
        if dte >= 0:
            exps.add((dte, r["expiry"]))
    if not exps:
        return dict(ticker=sym, error="เนเธกเนเธกเธตเธงเธฑเธเธซเธกเธ”เธญเธฒเธขเธธเธ—เธตเนเธขเธฑเธเนเธกเนเธเนเธฒเธเนเธ chain เธเธตเน")
    exps = sorted(exps)
    dte, exp = exps[0]

    def sigma_for(e: str) -> float:
        pool = [r["iv"] / 100.0 for r in ch["rows"]
                if r["expiry"] == e and abs(r["strike"] - spot) <= spot * 0.02
                and r.get("iv") and r["iv"] > 1]
        iv = float(np.mean(pool)) if pool else (ch.get("iv30") or 0) / 100.0
        if not iv or iv <= 0:
            iv = sigma_hist
        return float(np.clip(0.65 * iv + 0.35 * sigma_hist, 0.03, 4.0))

    sig = sigma_for(exp)
    mu = RF + 0.10 * net

    def p_beyond(level: float, T: float, kind: str):
        """Model chance the stock finishes beyond a price, on this expiry's vol."""
        if level <= 0 or T <= 0 or sig <= 0:
            return None
        try:
            z = (math.log(spot / level) + (mu - 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
        except (ValueError, ZeroDivisionError):
            return None
        return _ncdf(z) if kind == "CALL" else _ncdf(-z)

    def build(kind: str, strike_in, prem_in):
        if strike_in in (None, "", 0, "0"):
            return None
        try:
            want = float(strike_in)
        except (TypeError, ValueError):
            return dict(side=kind, error="strike เธ—เธตเนเธเธฃเธญเธเนเธกเนเนเธเนเธ•เธฑเธงเน€เธฅเธ")

        rows = [r for r in ch["rows"] if r["expiry"] == exp and r["kind"] == kind]
        if not rows:
            return dict(side=kind, strike=want,
                        error="เนเธกเนเธเธเธชเธฑเธเธเธฒ %s เนเธเธงเธฑเธเธซเธกเธ”เธญเธฒเธขเธธเธเธตเน" % kind)
        exact = [r for r in rows if abs(r["strike"] - want) < 1e-9]
        r = exact[0] if exact else min(rows, key=lambda x: abs(x["strike"] - want))
        strike = float(r["strike"])
        mid = ((r["bid"] + r["ask"]) / 2.0) if (r["bid"] and r["ask"]) else \
            float(r.get("last") or r.get("theo") or 0.0)
        try:
            prem = float(prem_in) if prem_in not in (None, "", 0, "0") else mid
        except (TypeError, ValueError):
            prem = mid
        be = strike + prem if kind == "CALL" else strike - prem

        T_exp = max(dte, 1) / 365.0
        T_day = 1.0 / 252.0
        p_itm = p_beyond(strike, T_exp, kind)
        p_win_expiry = p_beyond(be, T_exp, kind)
        p_side_today = p_beyond(strike, T_day, kind)
        p_win_today = p_beyond(be, T_day, kind)

        itm = spot > strike if kind == "CALL" else spot < strike
        dist_strike = (strike / spot - 1.0) * 100.0      # + means strike is above spot
        need_strike = abs(dist_strike)
        need_be = abs((be / spot - 1.0) * 100.0)
        pnl = mid - prem
        pnl_pct = (pnl / prem * 100.0) if prem > 0 else None
        # Chance of *touching* the strike at some point today, which is what a
        # day trader actually sells into.  Already beyond it means already met.
        p_touch_today = 1.0 if itm else min(1.0, 2.0 * (p_side_today or 0.0))

        p = p_side_today or 0.0
        same = tally["call"] if kind == "CALL" else tally["put"]
        against = tally["put"] if kind == "CALL" else tally["call"]
        aligned = same >= 5 and same > against

        if itm and p >= 0.5:
            verdict = "เนเธ”เนเน€เธเธฃเธตเธขเธ"
        elif p >= 0.62:
            verdict = "เธกเธตเนเธญเธเธฒเธช"
        elif p >= 0.45:
            verdict = "เธเธญเธกเธตเนเธญเธเธฒเธช"
        elif p >= 0.32:
            verdict = "เน€เธชเธตเนเธขเธงเน€เธ”เธตเธขเธง"
        elif aligned:
            # The direction is right, the distance is the problem.  Saying
            # "เนเธกเนเธเนเธฒ" here would contradict the trend table right above it.
            verdict = "เธ—เธดเธจเธ–เธนเธ เนเธ•เนเนเธเธฅเน€เธเธดเธเธงเธฑเธเธเธตเน"
        else:
            verdict = "เนเธกเนเธเนเธฒ"

        reasons = ["เน€เธ—เธเธเธดเธ %d เธเธฒเธ %d เธ•เธฑเธงเธเธตเนเธ—เธฒเธ %s ยท %d เธ•เธฑเธงเธเธตเนเธชเธงเธเธ—เธฒเธ"
                   % (same, len(votes), kind, against)]
        if itm:
            reasons.append("เธฃเธฒเธเธฒเธญเธขเธนเนเธเธฑเนเธเธ—เธตเนเนเธ”เนเน€เธเธฃเธตเธขเธ (ITM) เนเธฅเนเธง โ€” เน€เธซเธเธทเธญ strike %.1f%%"
                           % need_strike if kind == "CALL" else
                           "เธฃเธฒเธเธฒเธญเธขเธนเนเธเธฑเนเธเธ—เธตเนเนเธ”เนเน€เธเธฃเธตเธขเธ (ITM) เนเธฅเนเธง โ€” เนเธ•เน strike %.1f%%" % need_strike)
        else:
            reasons.append("เธ•เนเธญเธเธเธขเธฑเธเธญเธตเธ %.1f%% เธ–เธถเธ strike $%g" % (need_strike, strike))
        reasons.append("เธเธธเธ”เธเธธเนเธกเธ—เธธเธ $%.2f (เธ•เนเธญเธเธเธขเธฑเธเธญเธตเธ %.1f%%)" % (be, need_be))
        if prem > 0 and mid > 0:
            reasons.append("เธฃเธฒเธเธฒเธชเธฑเธเธเธฒเธ•เธญเธเธเธตเน $%.2f เน€เธ—เธตเธขเธเธ—เธตเนเธเนเธฒเธข $%.2f = %s%.0f%%"
                           % (mid, prem, "+" if pnl >= 0 else "", pnl_pct or 0.0))
        if dte <= 1:
            reasons.append("เธซเธกเธ”เธญเธฒเธขเธธเธงเธฑเธเธเธตเน/เธเธฃเธธเนเธเธเธตเน โ€” เธ•เนเธญเธเธเธดเธ”เธเธฑเนเธเธ—เธตเนเธ–เธนเธเธเนเธญเธเธ•เธฅเธฒเธ”เธเธดเธ” เนเธกเนเธเธฑเนเธเธซเธกเธ”เธเนเธฒ")
        if aligned and p < 0.45:
            reasons.append("เธ—เธดเธจเธ—เธฒเธเน€เธเนเธฒเธเนเธฒเธเธเธฒเธเธตเน เนเธ•เนเธ•เนเธญเธเธงเธดเนเธ %.1f%% เธ เธฒเธขเนเธเธงเธฑเธเน€เธ”เธตเธขเธง โ€” "
                           "เธ—เธดเธจเธ–เธนเธเธญเธขเนเธฒเธเน€เธ”เธตเธขเธงเธขเธฑเธเนเธกเนเธเธญ" % need_strike)

        hist = None
        try:
            n_days = max(int(round(dte * 5 / 7)), 1)
            dist = sim.hist_distributions_full(d["Close"], d["High"], d["Low"], n_days, 504)
            h = sim.hist_outcome(dist, kind, strike, spot, prem)
            if h and h.get("hist_win") is not None:
                hist = dict(win=h.get("hist_win"), n=h.get("hist_n"),
                            count=h.get("hist_win_count"), touch=h.get("hist_touch"),
                            pnl=h.get("hist_avg_pnl"))
        except Exception:
            hist = None

        return dict(side=kind, strike=strike, requested=round(want, 2),
                    strike_moved=not exact,
                    expiry=exp, dte=dte, status="ITM" if itm else "OTM",
                    dist_strike_pct=round(dist_strike, 2),
                    need_strike_pct=round(need_strike, 2),
                    breakeven=round(be, 2), need_be_pct=round(need_be, 2),
                    mid=round(mid, 2), premium=round(prem, 2),
                    pnl_usd=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 1) if pnl_pct is not None else None,
                    p_itm=round(p_itm * 100, 1) if p_itm is not None else None,
                    p_win_expiry=round(p_win_expiry * 100, 1) if p_win_expiry is not None else None,
                    p_side_today=round(p_side_today * 100, 1) if p_side_today is not None else None,
                    p_win_today=round(p_win_today * 100, 1) if p_win_today is not None else None,
                    p_touch_today=round(p_touch_today * 100, 1),
                    delta=r.get("delta"), iv=round(float(r.get("iv") or 0.0), 1),
                    volume=r.get("volume"), oi=r.get("oi"),
                    hist=hist, verdict=verdict, reasons=reasons)

    legs = [x for x in (build("CALL", call_strike, call_prem),
                        build("PUT", put_strike, put_prem)) if x]
    if not legs:
        return dict(ticker=sym, error="เธเธฃเธญเธ strike เธญเธขเนเธฒเธเธเนเธญเธขเธซเธเธถเนเธเธเธฒ (CALL เธซเธฃเธทเธญ PUT)")

    # ---- the evidence behind the direction ---------------------------------
    # "เธเนเธฒเธงเธฅเธฑเธ" does not exist for us and would be illegal to trade on.  What
    # does exist is the paper trail insiders and large holders are *required*
    # to file: Form 4 (officer/director trades), 8-K (material events),
    # 13D/G (5%+ stakes), plus unusual option prints.  That is the closest
    # legal thing to inside information, and it is checkable.
    ns = {}
    try:
        ns = io_pool().submit(news.for_ticker, sym, "").result(timeout=20.0) or {}
    except Exception:
        ns = {}

    ns_items = ns.get("items") or []
    fresh = int(ns.get("fresh") or 0)
    filings = ns.get("filings") or []
    insider = [f for f in filings if str(f.get("form")) in ("4", "3", "5")]
    material = [f for f in filings
                if str(f.get("form")) in ("8-K", "10-Q", "10-K", "SC 13D", "SC 13G",
                                          "S-1", "424B", "DEF 14A")]
    unusual = c_info.get("hot") or []
    npv = c_info.get("net_prem")
    pcv = c_info.get("pc_vol")

    def _form_rows(rows, limit=3):
        """Unique (form, date) rows with a readable meaning, newest first."""
        seen, out = set(), []
        for f in rows:
            key = (str(f.get("form")), f.get("date"))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(form=f.get("form"), date=f.get("date"),
                            title=FORM_TH.get(str(f.get("form")), f.get("title") or ""),
                            url=f.get("url")))
            if len(out) >= limit:
                break
        return out

    insider_rows = _form_rows(insider)
    material_rows = _form_rows(material)

    def why(side: str) -> list:
        out = []
        agree = [v["name"] for v in votes if v["dir"] == side]
        against = [v["name"] for v in votes if v["dir"] in ("CALL", "PUT") and v["dir"] != side]
        if agree:
            out.append("เน€เธ—เธเธเธดเธเธซเธเธธเธ %d เธ•เธฑเธง: %s" % (len(agree), ", ".join(agree)))
        if against:
            out.append("เน€เธ—เธเธเธดเธเธเนเธฒเธ %d เธ•เธฑเธง: %s" % (len(against), ", ".join(against)))

        if side == "CALL":
            if npv is not None and npv > 0.03:
                out.append("เน€เธเธดเธเธเธฃเธตเน€เธกเธตเธขเธกเธญเธญเธเธเธฑเธเนเธซเธฅเธเธฑเนเธ CALL (net %+.2f)" % npv)
            elif npv is not None and npv < -0.03:
                out.append("เน€เธเธดเธเธเธฃเธตเน€เธกเธตเธขเธกเนเธซเธฅเธเธฑเนเธ PUT (net %+.2f) โ€” เธชเธงเธเธ—เธฒเธเธเธฑเธ CALL" % npv)
            if pcv is not None and pcv < 0.85:
                out.append("Put/Call %.2f โ€” เธกเธตเธเธเธเธทเนเธญ CALL เธกเธฒเธเธเธงเนเธฒเธเธเธ•เธด" % pcv)
            elif pcv is not None and pcv > 1.2:
                out.append("Put/Call %.2f โ€” เธ•เธฅเธฒเธ”เน€เธญเธเนเธ PUT" % pcv)
        else:
            if npv is not None and npv < -0.03:
                out.append("เน€เธเธดเธเธเธฃเธตเน€เธกเธตเธขเธกเธญเธญเธเธเธฑเธเนเธซเธฅเธเธฑเนเธ PUT (net %+.2f)" % npv)
            elif npv is not None and npv > 0.03:
                out.append("เน€เธเธดเธเธเธฃเธตเน€เธกเธตเธขเธกเนเธซเธฅเธเธฑเนเธ CALL (net %+.2f) โ€” เธชเธงเธเธ—เธฒเธเธเธฑเธ PUT" % npv)
            if pcv is not None and pcv > 1.2:
                out.append("Put/Call %.2f โ€” เธกเธตเธเธเธเธทเนเธญ PUT เธกเธฒเธเธเธงเนเธฒเธเธเธ•เธด" % pcv)
            elif pcv is not None and pcv < 0.85:
                out.append("Put/Call %.2f โ€” เธ•เธฅเธฒเธ”เน€เธญเธเนเธ CALL" % pcv)

        if unusual:
            hot = ", ".join("%s $%g (vol %s / OI %s)"
                            % (h["kind"], h["strike"], int(h["volume"]), int(h["oi"]))
                            for h in unusual[:2])
            out.append("เธเธฃเธดเธกเธฒเธ“เธญเธญเธเธเธฑเธเธเธดเธ”เธเธเธ•เธดเธ—เธตเนเน€เธซเนเธ: " + hot)
        if insider:
            f = insider[0]
            out.append("เธเธนเนเธเธฃเธดเธซเธฒเธฃ/เธเธเธงเธเนเธเธขเธทเนเธ Form %s เธงเธฑเธเธ—เธตเน %s โ€” %s"
                       % (f["form"], f["date"],
                          FORM_TH.get(str(f.get("form")), "เน€เธญเธเธชเธฒเธฃเธชเธฒเธเธฒเธฃเธ“เธฐ")))
        if material:
            forms = ", ".join(sorted({str(f["form"]) for f in material[:3]}))
            out.append("เน€เธญเธเธชเธฒเธฃเธชเธณเธเธฑเธเธ•เนเธญ เธ.เธฅ.เธ•. เธชเธซเธฃเธฑเธ: %s (เธฅเนเธฒเธชเธธเธ” %s)" % (forms, material[0]["date"]))
        if fresh and ns_items:
            top = ns_items[0]
            out.append("เธเนเธฒเธงเนเธซเธกเน %d เธเธดเนเธ ยท เธฅเนเธฒเธชเธธเธ” โ€%sโ€ (%s, เธเนเธณเน€เธชเธตเธขเธ %+.2f)"
                       % (fresh, (top.get("t") or "")[:90], top.get("s") or "โ€”",
                          float(top.get("c") or 0.0)))
        else:
            out.append("เนเธกเนเธกเธตเธเนเธฒเธงเนเธซเธกเนเธ—เธตเนเนเธเนเนเธ”เน โ€” เน€เธซเธ•เธธเธเธฅเธเธตเนเธกเธฒเธเธฒเธเน€เธ—เธเธเธดเธ เธเธฃเธฐเนเธชเน€เธเธดเธ "
                       "เนเธฅเธฐเธชเธ–เธดเธ•เธดเธเธฃเธดเธเธขเนเธญเธเธซเธฅเธฑเธ เนเธกเนเนเธ”เนเธกเธฒเธเธฒเธเธเนเธฒเธง")
        return out

    why_call = why("CALL")
    why_put = why("PUT")

    badges = []
    if dte == 0:
        badges.append("เธงเธฑเธเธเธตเนเน€เธเนเธเธงเธฑเธเธซเธกเธ”เธญเธฒเธขเธธ โ€” เธ•เธฑเธงเน€เธฅเธเธเธทเธญเนเธญเธเธฒเธชเธเธดเธ”เธ—เนเธฒเธขเธงเธฑเธเธ—เธตเนเธเธฑเนเธเธเธฑเนเธเธเธเธฐ")
    elif dte <= 2:
        badges.append("เน€เธซเธฅเธทเธญเธญเธตเธ %d เธงเธฑเธ โ€” เธเนเธฒเน€เธงเธฅเธฒเธเธฐเธซเธกเธ”เน€เธฃเนเธง เธญเธขเนเธฒเธ–เธทเธญเธเนเธฒเธกเธงเธฑเธเนเธ”เธขเนเธกเนเธกเธตเนเธเธ" % dte)
    for lg in legs:
        if lg.get("strike_moved"):
            badges.append("%s $%g เนเธกเนเธกเธตเนเธเธ•เธฅเธฒเธ” เนเธเน strike เนเธเธฅเนเธชเธธเธ” $%g เนเธ—เธ"
                          % (lg["side"], lg.get("requested", 0), lg.get("strike", 0)))

    trend = "เธเธฒเธเธถเนเธ" if nz(d_info.get("stack"), 0) > 0 else \
        ("เธเธฒเธฅเธ" if nz(d_info.get("stack"), 0) < 0 else "เนเธเธงเนเธเธญเธญเธเธเนเธฒเธ")

    return dict(
        ticker=sym, spot=round(spot, 2),
        spot_source=spot_src,
        chg_pct=ch.get("chg_pct"), asof=ch.get("ts"), fetched=ch.get("fetched"),
        live=dict(price=round(live_price, 4) if live_price else None,
                  change=(live or {}).get("change"),
                  source=(live or {}).get("source"),
                  ts=(live or {}).get("ts"),
                  bars=[(live or {}).get("bars") or []][-90:][0] if live else []),
        expiry=exp, dte=dte,
        expiries=[dict(expiry=e, dte=t) for t, e in exps[:4]],
        trend=trend, signal=round(net, 3),
        tech_tally=tally, tech_votes=votes,
        intraday=dict(vwap=c_info.get("vwap"), rvol=c_info.get("rvol"),
                      pc_vol=c_info.get("pc_vol"), net_prem=c_info.get("net_prem")),
        atm_iv=round(sig * 100, 1), sigma_hist=round(sigma_hist * 100, 1),
        legs=legs, badges=badges,
        why_call=why_call, why_put=why_put,
        evidence=dict(
            news_state=ns.get("news_state") or ("เธกเธตเธเนเธฒเธงเนเธซเธกเน" if fresh else "เนเธกเนเธกเธตเธเนเธฒเธง"),
            fresh=fresh, news_total=int(ns.get("n") or 0),
            headlines=[dict(t=(i.get("t") or "")[:110], s=i.get("s"), c=i.get("c"),
                            h=i.get("h"), url=i.get("url")) for i in ns_items[:3]],
            insider=insider_rows,
            material=material_rows,
            unusual=unusual, pc_vol=pcv, net_prem=npv, rvol=c_info.get("rvol")),
        insider_note=("เธเนเธฒเธงเธฅเธฑเธ/เธเนเธญเธกเธนเธฅเธงเธเนเธ: เน€เธฃเธฒเนเธกเนเธกเธตเนเธฅเธฐเน€เธเนเธฒเธ–เธถเธเนเธกเนเนเธ”เน (เธเธดเธ”เธเธเธซเธกเธฒเธข) โ€” "
                      "เธชเธดเนเธเธ—เธตเนเธ•เธฃเธงเธเนเธ”เนเธเธทเธญเธฃเนเธญเธเธฃเธญเธขเธชเธฒเธเธฒเธฃเธ“เธฐเธ—เธตเนเธเธเธงเธเนเธเธ•เนเธญเธเธขเธทเนเธเธ•เนเธญ เธ.เธฅ.เธ•. เธชเธซเธฃเธฑเธ "
                      "เน€เธเนเธ Form 4 (เธเธฒเธฃเธเธทเนเธญเธเธฒเธขเธเธญเธเธเธนเนเธเธฃเธดเธซเธฒเธฃ), 8-K (เน€เธซเธ•เธธเธเธฒเธฃเธ“เนเธชเธณเธเธฑเธ), "
                      "13D/G (เธเธฒเธฃเธ–เธทเธญเธซเธธเนเธเธ•เธฑเนเธเนเธ•เน 5%) เนเธฅเธฐเธเธฃเธดเธกเธฒเธ“เธญเธญเธเธเธฑเธเธ—เธตเนเธเธดเธ”เธเธเธ•เธด"),
        answer=("เธ•เธฃเธงเธเธ—เธตเนเธฃเธฒเธเธฒ $%.2f ยท %s ยท เธงเธฑเธเธซเธกเธ”เธญเธฒเธขเธธเธ—เธตเนเนเธเน %s (เธญเธตเธ %d เธงเธฑเธ)"
                % (spot, trend, exp, dte)),
        summary=" ยท ".join(
            "%s $%g: %s (เนเธญเธเธฒเธชเธเธดเธ”เธเธฑเนเธเธเธเธฐ %.0f%%, เธ•เนเธญเธเธงเธดเนเธเธญเธตเธ %.1f%%)"
            % (lg["side"], lg["strike"], lg["verdict"],
               lg.get("p_side_today") or 0, lg.get("need_strike_pct") or 0)
            for lg in legs if not lg.get("error")),
        note="เนเธญเธเธฒเธชเธเธเธฐเน€เธเนเธเนเธเธเธเธณเธฅเธญเธเธเธฒเธเธเธงเธฒเธกเธเธฑเธเธเธงเธเธเธญเธเธงเธฑเธเธซเธกเธ”เธญเธฒเธขเธธเธเธตเน + เนเธเธงเนเธเนเธก ยท "
             "เธชเธ–เธดเธ•เธดเธเธฃเธดเธเธเธญเธ strike เน€เธ”เธตเธขเธงเธเธฑเธเนเธชเธ”เธเนเธขเธเนเธงเน ยท เธฃเธฒเธเธฒเธญเธญเธเธเธฑเธเธฅเนเธฒเธเนเธฒเธเธฒเธ CBOE",
    )


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
    "pop_source", "model_pop", "model_pop_lo", "model_pop_hi", "hist_win_count",
    "hist_itm", "hist_touch", "hist_avg_pnl", "hist_median_pnl", "hist_best_pnl",
    "hist_worst_pnl", "hist_profit_factor", "hist_recent_win", "hist_older_win",
    "hist_recent_n", "p_zero", "avg_win", "fair_value", "hist_n_eff",
    "edge_signif", "hist_6m_win", "hist_6m_n", "hist_6m_pnl", "hist_6m_count",
    "hist_6m_pf", "hist_12m_win", "hist_12m_n", "hist_12m_pnl",
    "hist_12m_count", "hist_12m_pf",
)
_TICKER_KEEP = (
    "ticker", "name", "spot", "spot_source", "chain_ts", "signal", "bias",
    "tech_score", "flow_score", "news_score", "news", "atm_iv", "iv30",
    "news_state", "news_note", "news_fresh", "news_total", "weights", "tech_tally",
    "iv_rank", "iv_pct", "vrp", "sigma_hist", "sigma_model", "earnings",
    "expected_move", "em_low", "em_high", "expiry", "dte", "n_days",
    "reliability", "omega", "verdict", "error", "hist", "regime", "tech",
    "flow", "iv30", "move_stats", "expiries", "expiry_list", "timings",
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
    srcs = nd.get("sources") or {}
    out["news_outlets"] = len(srcs)
    out["news_top"] = [k for k, _ in sorted(srcs.items(), key=lambda kv: -kv[1])[:3]]
    out["news_detail"] = dict(n=nd.get("n", 0), engine=nd.get("engine"),
                              sources={}, checks=[], items=[], filings=[],
                              social=nd.get("social", {}), note=nd.get("note", ""))
    out["guide"] = (r.get("guide") or [])[:3]
    out["answer"] = (r.get("answer") or [])[:2]
    # Per-expiry answers.  Keep the pick and its reason, drop the per-strike
    # why/risks prose (the reason sentence already says the same thing) and keep
    # only the fields the expiry cards actually render -- at 15 tickers x 4
    # expiries x 2 sides the extra fields alone would double the stored scan.
    _HZ_STRIKE_KEEP = (
        "kind", "strike", "expiry", "dte", "premium", "breakeven", "pop", "pop_lo",
        "pop_hi", "p_itm", "omega", "verdict", "delta", "hist_win", "hist_edge",
        "hist_n", "hist_n_eff", "cheap", "liquid", "spread_pct", "pop_source",
        "model_pop", "hist_win_count", "hist_avg_pnl", "hist_median_pnl",
        "hist_profit_factor", "hist_recent_win", "hist_older_win", "edge_signif",
        "volume", "oi", "iv_vs_atm",
        "hist_6m_win", "hist_6m_n", "hist_12m_win", "hist_12m_n", "hist_12m_pnl",
    )
    hz = {}
    for k, v in (r.get("expiries") or {}).items():
        if not isinstance(v, dict):
            continue
        v = dict(v)
        for side in ("call", "put"):
            s = v.get(side)
            v[side] = ({kk: s.get(kk) for kk in _HZ_STRIKE_KEEP} if isinstance(s, dict) else None)
        hz[k] = v
    out["expiries"] = hz
    out["expiry_list"] = list(r.get("expiries") or {})
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
# Bounds concurrent analyze() calls from every caller: the scan workers, the
# on-demand detail view and the diagnostics probe all funnel through this.
_ANALYZE_SLOTS = threading.Semaphore(ANALYZE_SLOTS)
# The analyses run on their own pool, never on the network pool: an analysis
# waits for the fetches it submitted, so letting both share one pool could
# deadlock it against itself.  The threads are created at import, while the
# process is quiet -- starting a thread during a pass is what used to stall
# the scan on a busy instance.
_ANALYZE_POOL = ThreadPoolExecutor(max_workers=ANALYZE_SLOTS + 1,
                                   thread_name_prefix="omega-analyze")
for _i in range(ANALYZE_SLOTS + 1):
    _ANALYZE_POOL.submit(lambda: None)
# Timestamp of the last ticker to complete in the running scan.  The scan is
# network-bound and each fetch already has a deadline, so this should always
# move; when it stops moving the pass is wedged (a provider that answers nothing
# and never closes the socket) and the scan lock would otherwise stay held
# forever, leaving the dashboard empty until the next deploy.
_HEARTBEAT = {"t": 0.0, "done": 0}
# Set by the supervisor when a pass has made no progress for a long time.  The
# pass then gives up at the next ticker instead of being torn down from the
# outside: pulling the lock out from under a running pass leaves the lock
# released twice, and the running thread dies holding a semaphore nobody else
# can release.
_SCAN_STOP = threading.Event()


def request_scan_stop() -> None:
    _SCAN_STOP.set()


def scan_stalled(idle_sec: float = 150.0) -> bool:
    """True when a scan is running but has made no progress for a long time.

    A held lock with no heartbeat at all also counts as stalled: that is the
    shape of a pass that wedged before it could begin, which is precisely the
    state that used to leave the dashboard empty until the next deploy.
    """
    if not scan_lock.locked():
        return False
    last = _HEARTBEAT.get("t") or 0.0
    if not last:
        return True
    return (time.time() - last) > idle_sec


def scan_running() -> bool:
    """True when a thread that is supposed to be scanning actually is.

    A lock can be held by a thread that has since died, and nothing then ever
    releases it.  Knowing that the owner is gone is what lets the server take
    the lock back instead of waiting for a pass that will never come.
    """
    if not scan_lock.locked():
        return False
    names = {t.name for t in threading.enumerate()}
    return bool(names & {"omega-scan", "omega-inline-scan", "omega-manual-scan"})


def scan_once() -> None:
    # One scan at a time, whoever asks for it (background loop or /api/scan/now).
    if not scan_lock.acquire(blocking=False):
        log.info("scan_once skipped: another scan is already running")
        return
    _SCAN_STOP.clear()
    # Stamp the heartbeat the instant the pass owns the lock.  Setting it only
    # inside the body would leave a window where a pass is "running" (lock held)
    # with no heartbeat at all, which the watchdog would read as healthy.
    _HEARTBEAT["t"] = time.time()
    _HEARTBEAT["done"] = 0
    try:
        _scan_once_locked()
    finally:
        scan_lock.release()


def scan_now_blocking() -> bool:
    """Run one scan in the calling thread if (and only if) the lock is free.

    This is the safety net behind the dashboard's first paint: a free instance
    can wedge a background pass, and a visitor who arrives then would otherwise
    stare at an empty screen until the watchdog noticed.  Serving the request by
    running the pass right here means the page is populated from the request
    itself, so the site always shows data even when the background loop is stuck.

    It deliberately does *not* wait for the lock: if a pass is already running
    (healthy or not) this returns at once and the caller falls back to whatever
    progress has been reported, rather than queueing the visitor behind it.
    """
    if not scan_lock.acquire(blocking=False):
        return False
    _SCAN_STOP.clear()
    _HEARTBEAT["t"] = time.time()
    _HEARTBEAT["done"] = 0
    try:
        _scan_once_locked()
        return True
    finally:
        scan_lock.release()


def _scan_once_locked() -> None:
    t0 = time.time()
    items, errs, cands = [], [], []
    total = len(WATCHLIST)
    _HEARTBEAT["t"] = time.time()
    # The two shared extras -- the earnings calendar and the market benchmark --
    # are warmed in the background so the tickers do not queue behind them on
    # the critical path of the cycle.
    for _warm in (earnings_map, _spy_benchmark):
        threading.Thread(target=_warm, daemon=True, name="omega-warm").start()
    with _STATE_LOCK:
        STATE["progress"] = dict(done=0, total=total, current="")
        STATE["universe"] = list(WATCHLIST)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work(sym):
        s = sym.strip().upper()
        if _SCAN_STOP.is_set():
            return s, None, None
        try:
            r = analyze(s)
        except Exception as ex:
            log.warning("analyze %s failed: %s", s, ex)
            return s, None, f"{s}: {type(ex).__name__}: {str(ex)[:90]}"
        if r is None:
            return s, None, f"{s}: เนเธกเนเธกเธตเธเนเธญเธกเธนเธฅเธเธญเธชเธณเธซเธฃเธฑเธเธงเธดเน€เธเธฃเธฒเธฐเธซเน"
        if r.get("error"):
            return s, None, f"{s}: {r['error']}"
        return s, r, None

    # Container memory is the binding constraint, and CPython will happily sit on
    # freed pandas frames instead of handing the pages back to the kernel.  Each
    # ticker's working set is therefore released explicitly the moment it is
    # summarised, so the resident size dips between tickers rather than only ever
    # climbing toward the instance limit.
    import gc

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        futs = {ex.submit(work, s): s for s in WATCHLIST}
        # Publish each ticker the moment it lands rather than waiting for the
        # slowest one.  On a free instance the full pass can take a while, and an
        # empty page for minutes looks broken even though work is happening.
        for fut in as_completed(futs):
            if _SCAN_STOP.is_set():
                log.warning("scan: asked to stop -- abandoning the rest of the pass")
                break
            # A pass that is about to be killed for using too much memory is
            # worse than a pass that stops: the whole worker goes down with it,
            # the page empties, and the pass starts again from nothing.  So the
            # pass stops itself before the instance does, keeps what it has,
            # and the next cycle picks up where this one left off.
            _rss = _rss_mb()
            if _rss and _rss > MEMORY_CEILING_MB:
                log.warning("scan: resident size %.0fMB is over the %.0fMB "
                            "ceiling -- stopping the pass here", _rss, MEMORY_CEILING_MB)
                _SCAN_STOP.set()
                break
            sym, r, err = fut.result()
            _HEARTBEAT["t"] = time.time()
            _HEARTBEAT["done"] = _HEARTBEAT.get("done", 0) + 1
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
                # ``r`` is the fat, fully-analysed dict; ``slim`` is the copy that
                # is actually kept.  Drop the original, collect, then ask the
                # allocator to return the pages to the OS so the resident size
                # dips here instead of ratcheting up to the instance limit.
                del r, slim
                gc.collect()
                release_memory()
        # Drop whatever the cache was still holding from earlier passes.  An
        # entry that is never read again is never evicted, so without this the
        # cache -- and with it the resident size -- only ever grew.
        cache.sweep()

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
