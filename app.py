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
    log.info("scan loop starting (refresh=%ss, watchlist=%d)", REFRESH_SEC, len(engine.WATCHLIST))
    time.sleep(1.5)
    first = True
    while True:
        t0 = time.time()
        try:
            log.info("scan cycle begin")
            engine.scan_once()
            log.info("scan cycle done in %.1fs (%d items)", time.time() - t0,
                     len(STATE.get("items", [])))
            for it in STATE.get("items", []):
                try:
                    track.record(it["ticker"], it.get("best"), it.get("spot"), STATE.get("updated"))
                except Exception:
                    pass
            try:
                track.evaluate()
            except Exception as ex:
                log.info("track evaluate: %s", ex)
            first = False
        except BaseException as ex:            # never let the worker die
            log.exception("scan loop error: %s", ex)
        # If the very first cycle fails we must not sleep for a whole refresh
        # window -- retry sooner so a transient network fault self-heals.
        nap = 30.0 if first else max(REFRESH_SEC - (time.time() - t0), 15.0)
        try:
            time.sleep(nap)
        except BaseException:
            pass


def _watchdog():
    """Restart a background worker if it ever stops.

    A scanner that dies silently leaves the dashboard empty with no clue why,
    which is exactly the failure this project kept hitting; the watchdog makes
    that state self-healing instead of requiring a redeploy.
    """
    while True:
        time.sleep(60)
        try:
            names = {t.name for t in threading.enumerate()}
            for fn, name in ((_scan_loop, "omega-scan"), (_probe_loop, "omega-probe")):
                if name not in names:
                    log.warning("watchdog: %s is gone -- restarting", name)
                    threading.Thread(target=fn, daemon=True, name=name).start()
        except BaseException as ex:
            log.info("watchdog: %s", ex)


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
        log.info("booting background workers")
        names = {t.name for t in threading.enumerate()}
        if "omega-scan" not in names:
            threading.Thread(target=_scan_loop, daemon=True, name="omega-scan").start()
        if "omega-probe" not in names:
            threading.Thread(target=_probe_loop, daemon=True, name="omega-probe").start()
        if "omega-watchdog" not in names:
            threading.Thread(target=_watchdog, daemon=True, name="omega-watchdog").start()
        log.info("background workers started")
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


@app.route("/api/boot")
def api_boot():
    """Show whether the background workers are alive, and start them if not.

    The scan thread is started once at import time.  If it ever dies (an
    unrecoverable error, a platform restart of the worker) the page would sit
    empty forever, so this reports the live thread list and *revives* a missing
    worker instead of trusting the one-shot boot flag.
    """
    global _BOOTED
    alive = any(t.name == "omega-scan" for t in threading.enumerate())
    if not alive:
        log.warning("/api/boot: scan worker missing -- restarting it")
        with _BOOT:
            _BOOTED = False       # force _boot() past its one-shot guard
        _boot()
    threads = [t.name for t in threading.enumerate()]
    return jsonify(booted=_BOOTED, threads=threads,
                   scan_alive=any(t.name == "omega-scan" for t in threading.enumerate()),
                   progress=dict(STATE.get("progress") or {}),
                   items=len(STATE.get("items") or []),
                   time=now_iso())


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
            # ``bool(r)`` explodes on a DataFrame ("truth value is ambiguous"), so
            # decide emptiness explicitly.  Note that ``.index`` is a *method* on
            # list/str, not an attribute -- touching it there raises
            # "builtin_function_or_method object is not subscriptable", which is
            # why the index branch is restricted to objects that really have one.
            if isinstance(r, dict):
                ok, note = bool(r), str(r.get("source") or r.get("ts") or "")[:60]
            elif isinstance(r, (list, tuple, set)):
                n = len(r)
                ok, note = n > 0, f"{n} records"
            elif isinstance(r, str):
                ok, note = bool(r.strip()), f"{len(r)} chars"
            elif hasattr(r, "empty") and hasattr(r, "index"):        # pandas objects
                n = len(r)
                ok, note = n > 0, f"{n} rows"
                if n:
                    try:
                        note = f"{n} rows · last {str(r.index[-1])[:10]}"
                    except Exception:
                        pass
            else:
                ok = r is not None
                note = str(r)[:60]
        except Exception as ex:
            # a failure inside fn() is a real provider problem
            ok, note = False, f"{type(ex).__name__}: {str(ex)[:70]}"
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
