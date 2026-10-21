"""Omega v2 โ€” market data layer.

Providers, in preference order (each has a hard timeout + circuit breaker):

* prices / history : Yahoo Finance chart API  -> Nasdaq historical API
* option chains    : CBOE delayed quotes CDN  (bid, ask, IV, volume, OI, Greeks)
* last price       : CBOE quote -> Yahoo chart meta -> Nasdaq real-time quote

Nothing here depends on ``yfinance``: the Yahoo *crumb* endpoint that made the
old engine hang and get HTTP 429 is never touched.
"""
from __future__ import annotations

import math
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .util import SEC_UA, breaker, cache, http, io_pool, log, now_iso

YF = "https://query{n}.finance.yahoo.com"
CBOE = "https://cdn.cboe.com/api/global/delayed_quotes"
NASDAQ = "https://api.nasdaq.com/api/quote"

# host keys the breaker uses for Yahoo's chart API (see util.Http.get)
YAHOO_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")

# Yahoo throttles datacentre IPs, and one refused request used to cost every
# ticker in the scan the full dial-and-read time -- twice per host, because
# the breaker only opened after two failures.  The first refusal now switches
# Yahoo off for the whole process for a while, so the tickers that follow go
# straight to the fallback histories instead of queueing behind a provider
# that has already said no.
_YAHOO_LOCK = threading.Lock()
_YAHOO_UNTIL = 0.0
YAHOO_COOLDOWN = 180.0


def _yahoo_off() -> None:
    global _YAHOO_UNTIL
    with _YAHOO_LOCK:
        _YAHOO_UNTIL = time.time() + YAHOO_COOLDOWN


def yahoo_blocked() -> bool:
    with _YAHOO_LOCK:
        return time.time() < _YAHOO_UNTIL


def yahoo_ok() -> bool:
    """True while Yahoo can still be asked.

    Both hosts have to be willing: one refusing host is not a reason to keep
    dialling it, and the other one refusing is the whole provider being
    unavailable.
    """
    if yahoo_blocked():
        return False
    return any(breaker.healthy(h) for h in YAHOO_HOSTS)

HEALTH: dict = {}


def _mark(name: str, ok: bool, note: str = "") -> None:
    d = HEALTH.setdefault(name, {"ok": 0, "fail": 0, "last": None, "note": ""})
    d["ok" if ok else "fail"] += 1
    d["last"] = now_iso()
    if note:
        d["note"] = note


# --------------------------------------------------------------------------- #
# Yahoo chart
# --------------------------------------------------------------------------- #
def _yahoo_chart(sym: str, rng: str, itv: str):
    if yahoo_blocked():
        return None, None
    for host in (1, 2):
        url = f"{YF.format(n=host)}/v8/finance/chart/{sym}?range={rng}&interval={itv}&includePrePost=false&events=div%2Csplit"
        # Check the breaker *before* dialling.  Yahoo throttles datacentre IPs, and
        # without this every ticker pays the host's connect+read timeout before
        # the breaker finally opens -- 15 tickers x 2 hosts is minutes of dead
        # air that makes the whole scan look like it has hung.
        if not breaker.healthy(f"query{host}.finance.yahoo.com"):
            continue
        j = http.json(url, timeout=(3, 6))
        if not j:
            _mark("Yahoo Chart", False, "เน€เธฃเธตเธขเธเนเธกเนเธชเธณเน€เธฃเนเธ (เธ–เธนเธเธเธณเธเธฑเธ”/เน€เธเธฃเธทเธญเธเนเธฒเธข)")
            _yahoo_off()
            continue
        try:
            res = (j.get("chart") or {}).get("result") or []
            if not res:
                continue
            res = res[0]
            ts = res.get("timestamp") or []
            q = ((res.get("indicators") or {}).get("quote") or [{}])[0]
            if not ts or not q.get("close"):
                continue
            df = pd.DataFrame(
                {
                    "Open": q.get("open"),
                    "High": q.get("high"),
                    "Low": q.get("low"),
                    "Close": q.get("close"),
                    "Volume": q.get("volume"),
                },
                index=pd.to_datetime(ts, unit="s", utc=True),
            )
            df = df.dropna(subset=["Close"])
            tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "America/New_York"
            try:
                df.index = df.index.tz_convert(tz).tz_localize(None)
            except Exception:
                df.index = df.index.tz_localize(None)
            df["Volume"] = df["Volume"].fillna(0.0)
            for c in ("Open", "High", "Low"):
                df[c] = df[c].fillna(df["Close"])
            meta = res.get("meta") or {}
            _mark("Yahoo Chart", True)
            return df, meta
        except Exception as ex:
            log.info("yahoo chart parse %s: %s", sym, ex)
            _mark("Yahoo Chart", False, str(ex)[:80])
    return None, None


def _stockanalysis_hist(sym: str):
    """stockanalysis.com unofficial JSON: [{t,o,h,l,c,v}, ...] newest first."""
    j = http.json(f"https://stockanalysis.com/api/symbol/s/{sym.lower()}/history?range=2Y&period=Day",
                  headers={"Accept": "application/json", "Referer": f"https://stockanalysis.com/stocks/{sym.lower()}/history/"},
                  timeout=(3, 8))
    if not isinstance(j, dict) or not isinstance(j.get("data"), list):
        _mark("StockAnalysis History", False, "เธฃเธนเธเนเธเธเธเนเธญเธกเธนเธฅเนเธกเนเธ•เธฃเธ")
        return None
    recs, idx = [], []
    for r in j["data"]:
        try:
            idx.append(pd.to_datetime(r.get("t"), errors="coerce"))
            recs.append(dict(Open=float(r.get("o")), High=float(r.get("h")), Low=float(r.get("l")),
                             Close=float(r.get("c")), Volume=float(r.get("v") or 0.0)))
        except Exception:
            continue
    df = pd.DataFrame(recs, index=pd.DatetimeIndex(idx)).dropna(subset=["Close"]).sort_index()
    if len(df) < 60:
        _mark("StockAnalysis History", False, f"เนเธ”เนเน€เธเธตเธขเธ {len(df)} เนเธ–เธง")
        return None
    _mark("StockAnalysis History", True)
    return df


def _cboe_hist(sym: str):
    """CBOE publishes full daily history on the same CDN as the option chain."""
    j = http.json(f"{CBOE}/charts/historical/{sym}.json", timeout=(3, 7))
    if not j or not isinstance(j.get("data"), list):
        _mark("CBOE History", False)
        return None
    recs, idx = [], []
    for r in j["data"]:
        try:
            idx.append(pd.to_datetime(r.get("date"), errors="coerce"))
            recs.append(dict(Open=float(r.get("open")), High=float(r.get("high")), Low=float(r.get("low")),
                             Close=float(r.get("close")), Volume=float(r.get("volume") or 0.0)))
        except Exception:
            continue
    df = pd.DataFrame(recs, index=pd.DatetimeIndex(idx)).dropna(subset=["Close"]).sort_index()
    df = df[df.index.notna()]
    if len(df) < 60:
        _mark("CBOE History", False, f"เนเธ”เนเน€เธเธตเธขเธ {len(df)} เนเธ–เธง")
        return None
    _mark("CBOE History", True)
    return df


def _nasdaq_hist(sym: str, days: int = 730):
    """Nasdaq historical API โ€” needs an explicit date range, an empty range
    returns no rows (the reason the old build silently produced nothing)."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(days * 1.6))
    url = (f"{NASDAQ}/{sym}/historical?assetclass=stocks&limit={min(days, 5000)}"
           f"&fromdate={start.isoformat()}&todate={end.isoformat()}")
    j = http.json(url, headers={"Accept": "application/json"}, timeout=(3, 8))
    if not j:
        _mark("Nasdaq History", False)
        return None
    try:
        rows = (((j.get("data") or {}).get("tradesTable") or {}).get("rows")) or []
        if not rows:
            _mark("Nasdaq History", False, "เนเธกเนเธกเธตเนเธ–เธงเธเนเธญเธกเธนเธฅ")
            return None
        recs, idx = [], []

        def num(x):
            try:
                return float(str(x).replace("$", "").replace(",", "").replace("%", ""))
            except Exception:
                return np.nan

        for r in rows:
            idx.append(pd.to_datetime(r.get("date"), format="%m/%d/%Y", errors="coerce"))
            recs.append(dict(Open=num(r.get("open")), High=num(r.get("high")), Low=num(r.get("low")),
                             Close=num(r.get("close")), Volume=num(r.get("volume"))))
        df = pd.DataFrame(recs, index=pd.DatetimeIndex(idx)).dropna(subset=["Close"]).sort_index()
        if len(df) < 60:
            _mark("Nasdaq History", False, f"เนเธ”เนเน€เธเธตเธขเธ {len(df)} เนเธ–เธง")
            return None
        _mark("Nasdaq History", True)
        return df
    except Exception as ex:
        _mark("Nasdaq History", False, str(ex)[:80])
        return None


def daily(sym: str, ttl: int = 3600) -> pd.DataFrame:
    """Daily OHLCV.  Four independent providers, tried in freshness order.

    Yahoo is the freshest, but it is also the one that rate-limits hardest; the
    others exist so a single blocked provider can never empty the dashboard.
    When Yahoo's circuit is already open we skip it entirely for this ticker --
    otherwise every name in the scan pays a timeout before reaching the
    alternates, and on a free instance that alone is enough to stall a cycle.

    The daily bars change once a day, so they are kept for an hour: when the
    price provider is refusing us, re-fetching the same two years of history
    every few minutes was the most expensive thing the scan did.
    """
    def load():
        df = None
        if yahoo_ok():
            df, _ = _yahoo_chart(sym, "2y", "1d")
        else:
            _mark("Yahoo Chart", False, "เธเนเธฒเธกเธเธฑเนเธงเธเธฃเธฒเธง (เธ–เธนเธเธเธณเธเธฑเธ”เธญเธฑเธ•เธฃเธฒ)")
        if df is not None and len(df) >= 120:
            _mark("Yahoo Chart", True)
            return df
        if df is not None and len(df) >= 60:
            _mark("Yahoo Chart", True, "เธเธฃเธฐเธงเธฑเธ•เธดเธชเธฑเนเธ")
            return df
        _mark("Yahoo Chart", False, "เธ–เธนเธเธเธณเธเธฑเธ”เธญเธฑเธ•เธฃเธฒ/เธ”เธถเธเนเธกเนเนเธ”เน")
        # The three alternates are independent, so they are asked at the same
        # time and the first usable answer wins.  Tried one after another, a
        # blocked Yahoo cost ~30s of waiting before the ticker had any history
        # at all -- which on a shared instance was most of a scan cycle.
        from concurrent.futures import wait

        alts = {}
        ex = io_pool()
        futs = {ex.submit(fn): name for name, fn in (
            ("StockAnalysis", lambda: _stockanalysis_hist(sym)),
            ("CBOE History", lambda: _cboe_hist(sym)),
            ("Nasdaq History", lambda: _nasdaq_hist(sym)))}
        done, _ = wait(list(futs.keys()), timeout=12.0)
        for f in done:
            name = futs[f]
            try:
                alts[name] = f.result(timeout=0.5)
            except Exception as err:
                log.info("daily %s via %s: %s", sym, name, err)
                alts[name] = None
        for name, alt in alts.items():
            if alt is None or len(alt) < 60:
                continue
            if df is None:
                _mark(name + " (เธชเธณเธฃเธญเธ)", True, f"เนเธเนเนเธ—เธ Yahoo เธชเธณเธซเธฃเธฑเธ {sym}")
                return alt
            # Yahoo came back short: stitch the older series underneath it
            joined = pd.concat([alt[alt.index < df.index[0]], df]).sort_index()
            joined = joined[~joined.index.duplicated(keep="last")]
            if len(joined) > len(df):
                _mark(name + " (เธชเธณเธฃเธญเธ)", True, "เธ•เนเธญเธเธฃเธฐเธงเธฑเธ•เธดเนเธซเน Yahoo")
                return joined
    
    df = cache.wrap(f"d:{sym}", ttl, load)
    return df if df is not None else pd.DataFrame()


def _cboe_intraday(sym: str):
    """CBOE 1-minute candles *including calls/puts volume per minute*.

    This is the closest thing to a real option tape that is available without a
    paid feed, so it powers the intraday part of the order-flow model.
    """
    j = http.json(f"{CBOE}/charts/intraday/{sym}.json", timeout=(3, 7))
    if not j or not isinstance(j.get("data"), list):
        _mark("CBOE Intraday", False)
        return None
    recs, idx = [], []
    for r in j["data"]:
        p = r.get("price") or {}
        v = r.get("volume") or {}
        dt = pd.to_datetime(r.get("datetime"), errors="coerce")
        if dt is None or pd.isna(dt) or p.get("close") is None:
            continue
        idx.append(dt)
        recs.append(dict(Open=p.get("open"), High=p.get("high"), Low=p.get("low"), Close=p.get("close"),
                         Volume=float(v.get("stock_volume") or 0.0),
                         Calls=float(v.get("calls_volume") or 0.0),
                         Puts=float(v.get("puts_volume") or 0.0)))
    if not recs:
        _mark("CBOE Intraday", False, "เนเธกเนเธกเธตเนเธ—เนเธเน€เธ—เธตเธขเธเธฃเธฒเธขเธเธฒเธ—เธต")
        return None
    df = pd.DataFrame(recs, index=pd.DatetimeIndex(idx)).dropna(subset=["Close"]).sort_index()
    if len(df) < 20:
        _mark("CBOE Intraday", False, f"เนเธ”เนเน€เธเธตเธขเธ {len(df)} เนเธ—เนเธ")
        return None
    _mark("CBOE Intraday", True, "เธฃเธงเธกเธเธฃเธดเธกเธฒเธ“ CALL/PUT เธฃเธฒเธขเธเธฒเธ—เธต")
    return df


def intraday(sym: str, ttl: int = 300) -> pd.DataFrame:
    def load():
        df = None
        # CBOE's intraday feed also carries per-minute CALL/PUT volume, so when
        # Yahoo is known to be down it is both the faster and the richer choice.
        if yahoo_ok():
            df, _ = _yahoo_chart(sym, "5d", "5m")
            if df is None or len(df) < 20:
                # A second Yahoo range costs another full dial-and-read cycle
                # and rarely adds anything the first one did not have.
                if yahoo_ok():
                    df, _ = _yahoo_chart(sym, "1d", "5m")
        if df is not None and len(df) > 10:
            return df
        return _cboe_intraday(sym)

    df = cache.wrap(f"i:{sym}", ttl, load)
    return df if df is not None else pd.DataFrame()


def live_chart(sym: str, minutes: int = 90, ttl: int = 10) -> dict | None:
    """Real price candles for the chart, one per minute.

    This is the tape itself, not a drawing: the one-minute closes published by
    the exchange's delayed-quote feed.  The cache is deliberately short, so a
    chart that asks every fifteen seconds gets a genuinely new bar rather than
    the same one dressed up again.
    """
    def load():
        df = None
        src = ""
        if yahoo_ok():
            df, _ = _yahoo_chart(sym, "1d", "5m")
            if df is not None and len(df) > 5:
                src = "Yahoo Chart (5 นาที)"
        if df is None or len(df) < 5:
            df = _cboe_intraday(sym)
            src = "CBOE (ข้อมูลล่าช้า)"
        if df is None or df.empty:
            return None
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        bars = []
        for ts, row in df.iterrows():
            try:
                bars.append([int(pd.Timestamp(ts).timestamp()), float(row["Close"])])
            except Exception:
                continue
        if len(bars) < 2:
            return None
        keep = max(minutes, 5)
        bars = bars[-keep:]
        last = bars[-1][1]
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        chg = (last - prev) / prev * 100 if prev else 0.0
        return dict(symbol=sym, price=round(last, 4), prev=round(prev, 4),
                    change=round(chg, 3), source=src,
                    ts=int(bars[-1][0]), bars=bars)

    return cache.wrap(f"chart:{sym}", ttl, load)


def yahoo_price(sym: str):
    def load():
        _, meta = _yahoo_chart(sym, "5d", "1d")
        if not meta:
            return None
        px = meta.get("regularMarketPrice") or meta.get("fulldayPrice") or meta.get("previousClose")
        return dict(price=px, prev=meta.get("chartPreviousClose") or meta.get("previousClose"),
                    volume=meta.get("regularMarketVolume"), name=meta.get("longName") or meta.get("shortName"),
                    source="Yahoo Chart", ts=now_iso()) if px else None

    return cache.wrap(f"yp:{sym}", 60, load)


# --------------------------------------------------------------------------- #
# Nasdaq real-time quote (fallback for price / company name)
# --------------------------------------------------------------------------- #
def nasdaq_quote(sym: str):
    def load():
        j = http.json(f"{NASDAQ}/{sym}/info?assetclass=stocks", headers={"Accept": "application/json"}, timeout=(3, 8))
        if not j or not (j.get("data") or {}).get("primaryData"):
            _mark("Nasdaq Quote", False)
            return None
        d = j["data"]
        pd_ = d.get("primaryData") or {}
        def num(x):
            try:
                return float(str(x).replace("$", "").replace(",", "").replace("%", ""))
            except Exception:
                return None
        px = num(pd_.get("lastSalePrice"))
        if not px:
            _mark("Nasdaq Quote", False)
            return None
        _mark("Nasdaq Quote", True)
        return dict(price=px, prev=px - (num(pd_.get("netChange")) or 0.0), chg_pct=num(pd_.get("percentageChange")),
                    volume=num(pd_.get("volume")), name=d.get("companyName"),
                    ts=pd_.get("lastTradeTimestamp") or now_iso(), source="Nasdaq (real-time)")

    return cache.wrap(f"nq:{sym}", 60, load)


# --------------------------------------------------------------------------- #
# CBOE delayed quotes โ€” the option chain workhorse
# --------------------------------------------------------------------------- #
def _parse_osym(osym: str):
    """'SOFI261023C00005000' -> ('2026-10-23', 'CALL', 5.0)"""
    if len(osym) < 16:
        return None
    tail = osym[-15:]
    yy, mm, dd, cp = tail[0:2], tail[2:4], tail[4:6], tail[6]
    strike = tail[7:]
    if cp not in ("C", "P") or not strike.isdigit():
        return None
    exp = f"20{yy}-{mm}-{dd}"
    return exp, ("CALL" if cp == "C" else "PUT"), int(strike) / 1000.0


def chain(sym: str, ttl: int = 240):
    """Full delayed option chain with provider Greeks. Returns dict or None."""

    def load():
        j = http.json(f"{CBOE}/options/{sym}.json", timeout=(3, 7))
        if not j or not j.get("data"):
            _mark("CBOE Chain", False)
            return None
        d = j["data"]
        ts = j.get("timestamp")
        opts = d.get("options") or []
        spot = float(d.get("current_price") or 0.0)
        meta = dict(symbol=d.get("symbol") or sym, spot=spot,
                    bid=d.get("bid"), ask=d.get("ask"), volume=d.get("volume"),
                    iv30=d.get("iv30"), open_price=d.get("open"), high=d.get("high"),
                    low=d.get("low"), prev_close=d.get("prev_day_close"),
                    close=d.get("close"), chg_pct=d.get("price_change_percent"))
        # A liquid name has tens of thousands of contracts, and holding them
        # all is what pushed the instance past its memory limit.  The analysis
        # only ever looks at a band around the money and at the nearest
        # expiries, so everything else is dropped while the chain is being read.
        exps = set()
        for o in opts:
            p = _parse_osym(str(o.get("option") or ""))
            if p:
                exps.add(p[0])
        keep_exps = set(sorted(exps)[:8])
        lo = spot * 0.45 if spot > 0 else None
        hi = spot * 1.55 if spot > 0 else None
        rows = []
        for o in opts:
            p = _parse_osym(str(o.get("option") or ""))
            if not p:
                continue
            exp, kind, strike = p
            if exp not in keep_exps:
                continue
            if lo is not None and not (lo <= strike <= hi):
                continue
            rows.append(dict(
                expiry=exp, kind=kind, strike=strike,
                bid=o.get("bid") or 0.0, ask=o.get("ask") or 0.0,
                bid_size=o.get("bid_size") or 0.0, ask_size=o.get("ask_size") or 0.0,
                iv=o.get("iv") or 0.0, oi=o.get("open_interest") or 0.0,
                volume=o.get("volume") or 0.0,
                delta=o.get("delta"), gamma=o.get("gamma"), theta=o.get("theta"),
                vega=o.get("vega"), rho=o.get("rho"), theo=o.get("theo"),
                last=o.get("last_trade_price") or 0.0, prev=o.get("prev_day_close") or 0.0,
                last_ts=o.get("last_trade_time"),
                high=o.get("high") or 0.0, low=o.get("low") or 0.0, open=o.get("open") or 0.0,
            ))
        del j, d, opts, o
        if not rows:
            _mark("CBOE Chain", False, "no contracts")
            return None
        _mark("CBOE Chain", True)
        meta.update(ts=ts, rows=rows, fetched=now_iso())
        return meta

    return cache.wrap(f"c:{sym}", ttl, load)


# --------------------------------------------------------------------------- #
# combined view
# --------------------------------------------------------------------------- #
def snapshot(sym: str) -> dict | None:
    """Spot price from the best available source, with provenance."""
    ch = cache.wrap(f"c:{sym}", 120, lambda: chain(sym))
    if ch and ch.get("spot"):
        return dict(symbol=sym, price=float(ch["spot"]), prev=ch.get("prev_close"),
                    volume=ch.get("volume"), iv30=ch.get("iv30"), chg_pct=ch.get("chg_pct"),
                    ts=ch.get("ts"), source="CBOE (delayed)")
    q = yahoo_price(sym)
    if q and q.get("price"):
        return dict(symbol=sym, price=float(q["price"]), prev=q.get("prev"), volume=q.get("volume"),
                    iv30=None, chg_pct=None, ts=q.get("ts"), source=q["source"], name=q.get("name"))
    q = nasdaq_quote(sym)
    if q and q.get("price"):
        return dict(symbol=sym, price=float(q["price"]), prev=q.get("prev"), volume=q.get("volume"),
                    iv30=None, chg_pct=q.get("chg_pct"), ts=q.get("ts"), source=q["source"], name=q.get("name"))
    return None


def company_name(sym: str) -> str:
    n = cache.wrap(f"nm:{sym}", 86400, lambda: ((nasdaq_quote(sym) or {}).get("name") or None))
    if n:
        return n
    m = cache.wrap(f"nm2:{sym}", 86400, lambda: ((yahoo_price(sym) or {}).get("name") or None))
    return m or sym


def provider_status() -> dict:
    out = {}
    for k, v in HEALTH.items():
        tot = v["ok"] + v["fail"]
        out[k] = dict(v, total=tot, success_pct=round(100.0 * v["ok"] / tot, 1) if tot else None)
    return out
