"""Omega v2 — indicators, Black-Scholes and the probability engines.

Deliberately scipy-free: the normal distribution is evaluated with math.erf, so
the container needs only numpy/pandas.  All estimators are deterministic given a
seed, and every probability comes with a sample size and a confidence interval
so the UI can be honest about how much the number can be trusted.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)


# --------------------------------------------------------------------------- #
# distributions
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT2PI


def wilson(k: float, n: float, z: float = 1.96):
    """Wilson score interval (percent) for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return (max(0.0, (c - half) / d * 100.0), min(100.0, (c + half) / d * 100.0))


# --------------------------------------------------------------------------- #
# indicators  (same family the original engine used, vectorised + guarded)
# --------------------------------------------------------------------------- #
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def true_range(h, l, c) -> pd.Series:
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def atr(h, l, c, n: int = 14) -> pd.Series:
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd(s: pd.Series, fast=12, slow=26, sig=9):
    line = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def adx_di(h, l, c, n: int = 14):
    tr = true_range(h, l, c)
    a = tr.ewm(alpha=1 / n, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index).ewm(alpha=1 / n, adjust=False).mean() / a.replace(0, np.nan)
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index).ewm(alpha=1 / n, adjust=False).mean() / a.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0), pdi.fillna(0.0), mdi.fillna(0.0)


def bollinger(c: pd.Series, n: int = 20, k: float = 2.0):
    ma = c.rolling(n, min_periods=5).mean()
    sd = c.rolling(n, min_periods=5).std()
    return ma, sd, ma + k * sd, ma - k * sd


def stoch_k(h, l, c, n: int = 14) -> pd.Series:
    ll, hh = l.rolling(n, min_periods=3).min(), h.rolling(n, min_periods=3).max()
    return (100 * (c - ll) / (hh - ll).replace(0, np.nan)).fillna(50.0)


def obv(c: pd.Series, v: pd.Series) -> pd.Series:
    return (np.sign(c.diff()).fillna(0.0) * v.fillna(0.0)).cumsum()


# --------------------------------------------------------------------------- #
# volatility estimators  (three of them: agreement between them is informative)
# --------------------------------------------------------------------------- #
def vol_close(c: pd.Series, window: int = 30, ann: int = 252) -> float:
    lr = np.log(c).diff().dropna().iloc[-window:]
    return float(lr.std() * math.sqrt(ann)) if len(lr) > 5 else 0.3


def vol_parkinson(h, l, window: int = 30, ann: int = 252) -> float:
    x = (np.log(h / l) ** 2).dropna().iloc[-window:]
    if len(x) < 5:
        return 0.3
    return float(math.sqrt(x.mean() / (4 * math.log(2))) * math.sqrt(ann))


def vol_garman_klass(o, h, l, c, window: int = 30, ann: int = 252) -> float:
    x = (0.5 * np.log(h / l) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2).dropna().iloc[-window:]
    if len(x) < 5:
        return 0.3
    return float(math.sqrt(max(x.mean(), 1e-9)) * math.sqrt(ann))


# --------------------------------------------------------------------------- #
# Black-Scholes-Merton
# --------------------------------------------------------------------------- #
def bsm(S: float, K: float, T: float, sig: float, kind: str, r: float = 0.045) -> dict:
    T = max(float(T), 1e-6)
    sig = max(float(sig), 1e-4)
    S = max(float(S), 1e-9)
    K = max(float(K), 1e-9)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * sq)
    d2 = d1 - sig * sq
    pdf = norm_pdf(d1)
    disc = math.exp(-r * T)
    if kind == "CALL":
        price = S * norm_cdf(d1) - K * disc * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = (-S * pdf * sig / (2 * sq) - r * K * disc * norm_cdf(d2)) / 365.0
        rho = K * T * disc * norm_cdf(d2) / 100.0
    else:
        price = K * disc * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1.0
        theta = (-S * pdf * sig / (2 * sq) + r * K * disc * norm_cdf(-d2)) / 365.0
        rho = -K * T * disc * norm_cdf(-d2) / 100.0
    return dict(
        price=max(price, 0.0),
        delta=delta,
        gamma=pdf / (S * sig * sq),
        theta=theta,
        vega=S * pdf * sq / 100.0,
        rho=rho,
        d1=d1,
        d2=d2,
        iv=sig,
    )


def implied_vol(px: float, S: float, K: float, T: float, kind: str, r: float = 0.045):
    """Bisection IV solver — used when the data provider does not publish IV."""
    if not px or px <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None
    lo, hi = 0.005, 6.0
    intrinsic = max(0.0, S - K) if kind == "CALL" else max(0.0, K - S)
    if px < intrinsic * 0.98:
        return None
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        p = bsm(S, K, T, mid, kind, r)["price"]
        if p > px:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    mid = 0.5 * (lo + hi)
    return mid if 0.01 <= mid <= 5.5 else None


def expected_move(S: float, sig: float, T: float) -> float:
    return max(S * sig * math.sqrt(max(T, 1e-9)), S * 0.0015)


# --------------------------------------------------------------------------- #
# probability engines
# --------------------------------------------------------------------------- #
def lognormal_prob(S: float, level: float, T: float, sig: float, mu: float, above: bool = True) -> float:
    """P(S_T > level) when S follows GBM with drift mu and vol sig."""
    if S <= 0 or level <= 0 or sig <= 0 or T <= 0:
        return 0.0
    d = (math.log(S / level) + (mu - 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    p = norm_cdf(d)
    return (p if above else 1.0 - p) * 100.0


def mc_gbm(S0: float, level: float, T: float, sigma: float, mu: float, kind: str,
           paths: int = 20000, steps: int | None = None, seed: int | None = None,
           touch: float | None = None) -> dict:
    """Monte-Carlo GBM with antithetic variates + chunking (memory friendly).

    Returns probability (%) of finishing beyond `level`, of ever touching
    `touch`, and the Monte-Carlo standard error in probability points.
    """
    rng = np.random.default_rng(seed)
    if steps is None:
        steps = int(max(8, min(round(T * 252 / 2), 126)))
    dt = T / steps
    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    chunk = 2000
    fin = touch_n = draws = 0
    up = kind == "CALL"
    lvl = touch if touch is not None else level
    while draws < paths:
        m = min(chunk, max(paths - draws, 1))
        z = rng.standard_normal((m, steps))
        use = np.vstack((z, -z))            # antithetic pairs
        logp = np.cumsum(drift + vol * use, axis=1)
        st = S0 * np.exp(logp[:, -1])
        ext = S0 * np.exp(logp.max(axis=1) if up else logp.min(axis=1))
        fin += int((st > level).sum() if up else (st < level).sum())
        touch_n += int((ext >= lvl).sum() if up else (ext <= lvl).sum())
        draws += use.shape[0]
    p_fin = 100.0 * fin / draws
    p_touch = 100.0 * touch_n / draws
    se = 100.0 * math.sqrt(max(p_fin / 100 * (1 - p_fin / 100) / draws, 1e-12))
    return dict(prob=p_fin, touch=p_touch, se=se, paths=draws, itm_n=fin)


def bootstrap_prob(logret: np.ndarray, S0: float, level: float, n_days: int, kind: str,
                   sims: int = 20000, block: int = 5, seed: int | None = None) -> dict | None:
    """Circular block bootstrap of *actual* historical returns.

    This is the estimator that does not assume normality: it resamples the real
    joint behaviour (including fat tails and autocorrelation) of the stock.
    """
    rets = np.asarray(logret, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = len(rets)
    if n < 60 or n_days < 1:
        return None
    rng = np.random.default_rng(seed)
    L = int(max(1, min(block, n // 4)))
    nb = int(math.ceil(n_days / L))
    starts = rng.integers(0, n, size=(sims, nb))
    offs = np.arange(L)
    idx = (starts[:, :, None] + offs[None, None, :]) % n
    tot = rets[idx].reshape(sims, nb * L)[:, :n_days].sum(axis=1)
    st = S0 * np.exp(tot)
    hit = (st > level) if kind == "CALL" else (st < level)
    k = int(hit.sum())
    lo, hi = wilson(k, sims)
    return dict(prob=100.0 * k / sims, lo=lo, hi=hi, n=n, sims=sims, block=L, hits=k)


def fat_tail_prob(logret: np.ndarray, S0: float, level: float, n_days: int, kind: str,
                  sims: int = 20000, seed: int | None = None) -> dict | None:
    """Student-t (fat tailed) simulation with the tail index fitted to history."""
    rets = np.asarray(logret, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 60 or n_days < 1:
        return None
    rng = np.random.default_rng(seed)
    mu, sd = float(rets.mean()), float(rets.std())
    if sd <= 0:
        return None
    z = (rets - mu) / sd
    kurt = float((z ** 4).mean()) - 3.0
    df = 6.0 / max(kurt, 0.05) + 4.0 if kurt > 0.02 else 30.0
    df = float(min(max(df, 3.2), 30.0))
    t = rng.standard_t(df, size=(sims, n_days))
    t = t / math.sqrt(df / (df - 2.0))          # unit variance
    tot = (mu + sd * t).sum(axis=1)
    st = S0 * np.exp(tot)
    hit = (st > level) if kind == "CALL" else (st < level)
    k = int(hit.sum())
    lo, hi = wilson(k, sims)
    return dict(prob=100.0 * k / sims, lo=lo, hi=hi, df=round(df, 1), sims=sims, hits=k)


def block_bootstrap_ci(logret: np.ndarray, S0: float, n_days: int, sims: int = 4000, seed: int | None = None):
    """Distribution of the terminal return — used for percentile scenario bands."""
    rets = np.asarray(logret, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 60 or n_days < 1:
        return None
    rng = np.random.default_rng(seed)
    n = len(rets)
    L = int(max(1, min(5, n // 4)))
    nb = int(math.ceil(n_days / L))
    starts = rng.integers(0, n, size=(sims, nb))
    offs = np.arange(L)
    idx = (starts[:, :, None] + offs[None, None, :]) % n
    tot = rets[idx].reshape(sims, nb * L)[:, :n_days].sum(axis=1)
    st = S0 * np.exp(tot)
    q = np.percentile(st, [5, 25, 50, 75, 95])
    return dict(p5=float(q[0]), p25=float(q[1]), p50=float(q[2]), p75=float(q[3]), p95=float(q[4]))


def blend_prob(parts: list) -> dict:
    """Weighted blend of probability estimates; each part = (prob, weight, ci_half)."""
    num = den = 0.0
    var = 0.0
    used = 0
    for p, w, ci in parts:
        if p is None or w <= 0:
            continue
        num += p * w
        den += w
        var += (w / max(den, 1e-9)) ** 2 * (ci or 0.0) ** 2
        used += 1
    if den <= 0:
        return dict(prob=None, half=0.0, methods=0)
    p = num / den
    half = math.sqrt(var) + max(2.0, 12.0 / max(used, 1))
    return dict(prob=float(p), half=float(min(half, 25.0)), methods=used)
