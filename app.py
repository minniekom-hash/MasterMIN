"""OPTION TRADE: Omega Singularity Engine v2 — Flask backend.

Routes
    /                 dashboard
    /health           liveness probe (used by Render)
    /api/scan         current scan state (per-ticker picks + cheap/high-volume guide)
    /api/sources      news-source health, with per-source status and latency
    /api/analyze/<t>  on-demand analysis of any US ticker
    /api/track        self-scoring prediction log (real hit rate + calibration)
    /api/diag         provider diagnostics: which data source answers from *this*
                      server, with latency — the honest way to see what works
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time

from flask import Flask, jsonify, render_template_string

from omega import engine, news, track
from omega.data import chain, daily, provider_status, yahoo_price
from omega.engine import REFRESH_SEC, STATE
from omega.util import VERSION, cache, clean, http, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("omega.app")

app = Flask(__name__)
_TPL = {"src": None, "mtime": None}


# --------------------------------------------------------------------------- #
# template loading (works whether or not the build step copied index.html)
# --------------------------------------------------------------------------- #
def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, "templates", "index.html"), os.path.join(here, "index.html"),
            os.path.join(os.getcwd(), "templates", "index.html"), os.path.join(os.getcwd(), "index.html")]


def _template_source():
    for p in _candidates():
        if os.path.exists(p):
            mt = os.path.getmtime(p)
            if _TPL["src"] is None or _TPL["mtime"] != mt:
                with open(p, "r", encoding="utf-8") as f:
                    _TPL["src"] = f.read()
                _TPL["mtime"] = mt
                log.info("template loaded from %s (%d bytes)", p, len(_TPL["src"]))
            return _TPL["src"]
    return ("<h1>index.html not found</h1><p>Expected next to app.py or in ./templates/</p>",
            _candidates())


# --------------------------------------------------------------------------- #
# background threads
# --------------------------------------------------------------------------- #
_BOOT = threading.Lock()
_BOOTED = False


def _scan_loop():
    time.sleep(1.5)
    while True:
        t0 = time.time()
        try:
            engine.scan_once()
            for it in STATE.get("items", []):
                try:
                    track.record(it["ticker"], it.get("best"), it.get("spot"), STATE.get("updated"))
                except Exception:
                    pass
            try:
                track.evaluate()
            except Exception as ex:
                log.info("track evaluate: %s", ex)
        except Exception as ex:
            log.exception("scan loop error: %s", ex)
        time.sleep(max(REFRESH_SEC - (time.time() - t0), 15.0))


def _probe_loop():
    time.sleep(20)
    while True:
        try:
            news.probe_all()
        except Exception as ex:
            log.info("probe loop: %s", ex)
        time.sleep(900)


def _boot():
    global _BOOTED
    with _BOOT:
        if _BOOTED:
            return
        if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            return
        _BOOTED = True
        threading.Thread(target=_scan_loop, daemon=True, name="omega-scan").start()
        threading.Thread(target=_probe_loop, daemon=True, name="omega-probe").start()
        log.info("Omega v%s booted · watchlist=%s", VERSION, ",".join(engine.WATCHLIST))


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    src = _template_source()
    if isinstance(src, tuple):
        return src
    return render_template_string(src, refresh=REFRESH_SEC, version=VERSION)


@app.route("/health")
def health():
    return jsonify(ok=True, version=VERSION, updated=STATE.get("updated"),
                   items=len(STATE.get("items", [])), time=now_iso())


@app.route("/api/scan")
def api_scan():
    with engine._STATE_LOCK:
        payload = clean(dict(STATE))
    payload["cache"] = cache.stats()
    return jsonify(payload)


@app.route("/api/sources")
def api_sources():
    return jsonify(clean(news.source_status()))


@app.route("/api/track")
def api_track():
    return jsonify(clean(track.scoreboard()))


@app.route("/api/analyze/<t>")
def api_analyze(t: str):
    sym = re.sub(r"[^A-Za-z]", "", t).upper()[:6]
    if not sym:
        return jsonify(error="ใส่ ticker หุ้นสหรัฐ เช่น AAPL"), 400
    try:
        res = cache.wrap(f"an:{sym}", 60, lambda: engine.analyze(sym))
    except Exception as ex:
        log.warning("analyze %s: %s", sym, ex)
        return jsonify(error=f"{type(ex).__name__}: {str(ex)[:160]}"), 502
    if not res:
        return jsonify(error="ไม่พบข้อมูลหุ้น/ออปชันสำหรับ ticker นี้"), 404
    return jsonify(clean(res))


@app.route("/api/diag")
def api_diag():
    """Live self-test: which providers answer from the server that runs this."""
    checks = []

    def probe(name, fn):
        t0 = time.time()
        ok, note = False, ""
        try:
            r = fn()
            # ``bool(r)`` explodes on a DataFrame ("truth value is ambiguous"),
            # so decide on emptiness explicitly for anything with a length.
            if isinstance(r, dict):
                ok, note = bool(r), str(r.get("source") or r.get("ts") or "")[:60]
            elif hasattr(r, "empty") or isinstance(r, (list, tuple, str)):
                n = len(r)
                ok, note = n > 0, f"{n} records"
                if hasattr(r, "index") and n:
                    note = f"{n} rows · last {str(r.index[-1])[:10]}"
            else:
                ok = r is not None
                note = str(r)[:60]
        except Exception as ex:
            note = f"{type(ex).__name__}: {str(ex)[:70]}"
        checks.append(dict(name=name, ok=ok, ms=int((time.time() - t0) * 1000), note=note))

    probe("CBOE option chain (SPY)", lambda: chain("SPY"))
    probe("Yahoo chart (SPY daily)", lambda: daily("SPY"))
    probe("Yahoo quote (SPY)", lambda: yahoo_price("SPY"))
    probe("CNBC RSS", lambda: news.fetch_feed("https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114", False, 3))
    probe("Yahoo ticker RSS (AAPL)", lambda: news.fetch_feed("https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US", False, 3))
    probe("SEC EDGAR submissions", lambda: news.sec_filings("AAPL"))
    probe("Stocktwits", lambda: news.stocktwits("AAPL"))
    probe("Alpha Vantage news", lambda: news.alphavantage("AAPL"))
    return jsonify(clean(dict(version=VERSION, now=now_iso(), checks=checks,
                              providers=provider_status(), fetchers=news.FETCH,
                              watchlist=engine.WATCHLIST, circuit=__import__("omega.util", fromlist=["breaker"]).breaker.state(),
                              config=dict(sims=engine.SIMS, refresh_sec=REFRESH_SEC,
                                          min_vol=engine.MIN_VOL, min_oi=engine.MIN_OI,
                                          max_premium_pct=engine.MAX_PREMIUM_PCT,
                                          max_premium_abs=engine.MAX_PREMIUM_ABS,
                                          max_spread=engine.MAX_SPREAD, dte=[engine.MIN_DTE, engine.MAX_DTE],
                                          hist_window=engine.HIST_WINDOW, strike_band=engine.STRIKE_BAND,
                                          max_strikes=engine.MAX_STRIKES,
                                          sentiment=news.SENTIMENT_ENGINE))))


_boot()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
