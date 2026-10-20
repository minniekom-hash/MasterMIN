"""Omega v2 — simulation & historical-statistics core.

One horizon simulation is shared by every strike, so a 10-strike table costs the
same as a single contract: we draw terminal prices *and* the path extremes
(max/min) once, then every strike is just a vector comparison.

Four views of the same question ("where will this be in n days?") are produced:

  boot  circular block bootstrap of the stock's own 6-24 month returns
  ft    Student-t with the tail index fitted to the stock (fat tails)
  mc    GBM Monte-Carlo with antithetic variates (engine-tilted drift)
  ln    log-normal closed form at the market's implied vol (risk-neutral)

and the historical module answers the same question with *actual* 6-month price
action: how often did this stock move this far, touch this level, or finish
beyond it within the same number of trading days.
"""
from __future__ import annotations

import math

import numpy as np

from . import quant
from .util import nz


# --------------------------------------------------------------------------- #
# forward simulation (one draw, reused for every strike)
# --------------------------------------------------------------------------- #
def simulate(spot: float, T: float, n_days: int, sigma: float, mu: float,
             logret: np.ndarray, sims: int = 12000, seed: int | None = None) -> dict:
    # Hard guard: a single NaN anywhere in here would silently poison every
    # probability on the page, so sanitise the inputs before drawing anything.
    spot = nz(spot, 100.0) or 100.0
    T = max(nz(T, 7 / 365.0), 1e-5)
    sigma = float(np.clip(nz(sigma, 0.35), 0.01, 5.0))
    mu = float(np.clip(nz(mu, 0.04), -2.0, 2.0))
    sims = int(max(200, min(int(nz(sims, 12000)), 200_000)))
    rng = np.random.default_rng(seed)
    n_days = max(int(n_days), 1)
    out: dict = {}

    # ---- 1. block bootstrap of real returns --------------------------------
    rets = np.asarray(logret, dtype=float) if logret is not None else np.empty(0)
    rets = rets[np.isfinite(rets)]
    rets = np.clip(rets, -0.5, 0.5)          # clip split/crash artefacts
    if len(rets) >= 90:
        L = int(max(1, min(5, len(rets) // 4)))
        nb = int(math.ceil(n_days / L))
        idx = (rng.integers(0, len(rets), size=(sims, nb))[:, :, None] + np.arange(L)[None, None, :]) % len(rets)
        r = rets[idx].reshape(sims, nb * L)[:, :n_days]
        lp = np.cumsum(r, axis=1)
        out["boot"] = (spot * np.exp(lp[:, -1]), spot * np.exp(lp.max(axis=1)), spot * np.exp(lp.min(axis=1)))

    # ---- 2. Student-t fat tails -------------------------------------------
    if len(rets) >= 90:
        m, sd = float(rets.mean()), float(rets.std())
        if sd > 0:
            z = (rets - m) / sd
            kurt = float((z ** 4).mean()) - 3.0
            df = float(min(max(6.0 / max(kurt, 0.05) + 4.0 if kurt > 0.02 else 30.0, 3.2), 30.0))
            t = rng.standard_t(df, size=(sims, n_days)) / math.sqrt(df / (df - 2.0))
            lp = np.cumsum(m + sd * t, axis=1)
            out["ft"] = (spot * np.exp(lp[:, -1]), spot * np.exp(lp.max(axis=1)), spot * np.exp(lp.min(axis=1)),
                         round(df, 1))

    # ---- 3. GBM Monte-Carlo (antithetic, chunked) --------------------------
    steps = int(max(10, min(round(n_days * 6), 150)))
    dt = T / steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    chunk, done = 3000, 0
    st_l, mx_l, mn_l = [], [], []
    while done < sims:
        m = min(chunk, sims - done)
        z = rng.standard_normal((m, steps))
        use = np.vstack((z, -z))
        lp = np.cumsum(drift + vol * use, axis=1)
        st_l.append(spot * np.exp(lp[:, -1]))
        mx_l.append(spot * np.exp(lp.max(axis=1)))
        mn_l.append(spot * np.exp(lp.min(axis=1)))
        done += use.shape[0]
    out["mc"] = (np.concatenate(st_l), np.concatenate(mx_l), np.concatenate(mn_l))
    out["meta"] = dict(sims=sims, n_days=n_days, steps=steps, df=out.get("ft", (None, None, None, None))[3],
                       boot_n=len(rets))
    return out


# --------------------------------------------------------------------------- #
# probability of a strike, from every view
# --------------------------------------------------------------------------- #
def _pct(mask) -> float:
    return float(np.mean(mask) * 100.0)


def strike_row(sim: dict, spot: float, strike: float, kind: str, premium: float,
               T: float, iv: float, mu_rn: float, weights=None) -> dict:
    """Probability that a long option struck at `strike` makes money."""
    up = kind == "CALL"
    be = strike + premium if up else strike - premium
    parts, cih = [], 0.0

    ln = quant.lognormal_prob(spot, be, T, iv, mu_rn, above=up)
    parts.append((ln, (weights or {}).get("ln", 0.22), 6.0))
    ln_itm = quant.lognormal_prob(spot, strike, T, iv, mu_rn, above=up)

    if "boot" in sim:
        st, mx, mn = sim["boot"]
        hit = (st > be) if up else (st < be)
        k, n = int(hit.sum()), hit.size
        lo, hi = quant.wilson(k, n)
        parts.append((_pct(hit), (weights or {}).get("boot", 0.34), (hi - lo) / 4.0))
        p_boot, p_boot_lo, p_boot_hi = _pct(hit), lo, hi
        itm_boot = _pct((st > strike) if up else (st < strike))
        touch_boot = _pct((mx >= strike) if up else (mn <= strike))
    else:
        p_boot = p_boot_lo = p_boot_hi = None
        itm_boot = touch_boot = None

    if "ft" in sim:
        st, mx, mn, _df = sim["ft"]
        hit = (st > be) if up else (st < be)
        k, n = int(hit.sum()), hit.size
        lo, hi = quant.wilson(k, n)
        parts.append((_pct(hit), (weights or {}).get("ft", 0.18), (hi - lo) / 4.0))
        p_ft = _pct(hit)
    else:
        p_ft = None

    st, mx, mn = sim["mc"]
    hit = (st > be) if up else (st < be)
    parts.append((_pct(hit), (weights or {}).get("mc", 0.26), 1.2))
    p_mc = _pct(hit)

    blend = quant.blend_prob(parts)
    pop = blend["prob"] if blend["prob"] is not None else _pct(hit)
    cih = blend["half"]

    p_itm = float(np.mean([x for x in (ln_itm, itm_boot, _pct((st > strike) if up else (st < strike))) if x is not None]))
    p_touch = float(np.mean([x for x in (touch_boot, _pct((mx >= strike) if up else (mn <= strike))) if x is not None]))

    # expectancy from the empirical terminal distribution
    pay = np.maximum(st - strike, 0.0) if up else np.maximum(strike - st, 0.0)
    fair = float(pay.mean())
    ev = fair - float(premium)
    ev_pct = (ev / premium * 100.0) if premium > 0 else None
    if ev_pct is not None and not math.isfinite(ev_pct):
        ev_pct = None
    if not math.isfinite(ev):
        ev, fair = 0.0, float(premium)
    win_pay = pay[pay > premium]
    avg_win = float(win_pay.mean()) if win_pay.size else 0.0
    p_zero = _pct(pay <= 0)

    return dict(
        kind=kind, strike=float(strike), premium=round(float(premium), 2), breakeven=round(be, 2),
        pop=round(pop, 1), pop_lo=round(max(0.0, pop - cih), 1), pop_hi=round(min(100.0, pop + cih), 1),
        pop_half=round(cih, 1), p_itm=round(p_itm, 1), p_touch=round(p_touch, 1),
        p_boot=round(p_boot, 1) if p_boot is not None else None,
        p_boot_lo=round(p_boot_lo, 1) if p_boot_lo is not None else None,
        p_boot_hi=round(p_boot_hi, 1) if p_boot_hi is not None else None,
        p_fat=round(p_ft, 1) if p_ft is not None else None, p_mc=round(p_mc, 1), p_market=round(ln, 1),
        ev=round(ev, 3), ev_pct=round(ev_pct, 1) if ev_pct is not None else None,
        avg_win=round(avg_win, 2), p_zero=round(p_zero, 1),
        ll_ratio=round(avg_win / premium, 2) if premium > 0 and avg_win else None,
        fair_value=round(fair, 2),
        edge=round(ev, 3),
    )


# --------------------------------------------------------------------------- #
# 6-month empirical statistics of the stock itself
# --------------------------------------------------------------------------- #
def hist_distributions(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                       n_days: int, window: int = 126) -> dict | None:
    """Rolling n-day outcomes over roughly the last six months (~126 trading days).

    Every rolling window that *starts* inside the last 126 days is used, so the
    number of observations stays near 126 whatever the holding period is — about
    six months of real price action either way.
    """
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(close)
    n_days = max(int(n_days), 1)
    if n < n_days + 40:
        return None
    start = max(0, n - window)
    ends = np.arange(start, n - n_days)
    if len(ends) < 20:
        return None
    entries = close[ends]
    exits = close[ends + n_days]
    fmax = np.array([high[e + 1: e + n_days + 1].max() for e in ends])
    fmin = np.array([low[e + 1: e + n_days + 1].min() for e in ends])
    rel = exits / entries - 1.0
    up_exc = fmax / entries - 1.0
    dn_exc = 1.0 - fmin / entries
    return dict(rel=rel, up_exc=up_exc, dn_exc=dn_exc, n=len(rel),
                entry=entries, exit=exits, fmax=fmax, fmin=fmin,
                window=n_days, span_days=len(ends))


def hist_query(dist: dict, kind: str, strike: float, spot: float, premium: float, n_days: int,
               close: np.ndarray) -> dict:
    """Translate the historical distributions into stats for one strike level."""
    up = kind == "CALL"
    if not dist:
        return dict(hist_n=0)
    # relative distance of this strike from the current spot
    d_req = strike / spot - 1.0
    be = strike + premium if up else strike - premium
    d_be = be / spot - 1.0

    if up:
        hit_move = dist["up_exc"] >= d_req           # touched the strike within n days
        end_be = dist["rel"] >= d_be                  # finished beyond breakeven
        end_k = dist["rel"] >= d_req
    else:
        hit_move = dist["dn_exc"] >= -d_req
        end_be = dist["rel"] <= d_be
        end_k = dist["rel"] <= d_req
    hist_n = int(len(dist["rel"]))
    wins = int(np.sum(end_be))
    lo, hi = quant.wilson(wins, hist_n)
    closes = np.asarray(close, dtype=float)[-126:]
    days_above = float(np.mean(closes > strike) * 100)

    rel = dist["rel"] * 100.0
    return dict(
        hist_n=hist_n,
        hist_window=dist.get("window"),
        hist_span=dist.get("span_days"),
        hist_win=round(100.0 * wins / hist_n, 1),
        hist_win_lo=round(lo, 1), hist_win_hi=round(hi, 1),
        hist_touch=round(float(np.mean(hit_move) * 100.0), 1),
        hist_end=round(float(np.mean(end_k) * 100.0), 1),
        hist_avg_move=round(float(np.mean(np.abs(rel))), 2),
        hist_p90_move=round(float(np.percentile(np.abs(rel), 90)), 2),
        hist_best=round(float(np.max(rel)), 1), hist_worst=round(float(np.min(rel)), 1),
        hist_up_rate=round(float(np.mean(dist["rel"] > 0) * 100.0), 1),
        days_above=round(days_above, 1),
        days_below=round(100.0 - days_above, 1),
    )


def move_table(dist: dict) -> list:
    """A compact 'how often did it move this far' table for the UI."""
    if not dist:
        return []
    rows = []
    for x in (1, 2, 3, 5, 8, 10, 15, 20):
        rows.append(dict(
            move=x,
            up=round(float(np.mean(dist["up_exc"] >= x / 100.0) * 100.0), 1),
            down=round(float(np.mean(dist["dn_exc"] >= x / 100.0) * 100.0), 1),
            finish_up=round(float(np.mean(dist["rel"] >= x / 100.0) * 100.0), 1),
            finish_down=round(float(np.mean(dist["rel"] <= -x / 100.0) * 100.0), 1),
        ))
    return rows
