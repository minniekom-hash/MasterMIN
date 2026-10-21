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

import hmac
import logging
import os
import re
import threading
import time

from flask import (Flask, jsonify, redirect, render_template_string, request,
                   session, url_for)

from omega import engine, news, track
from omega.data import chain, daily, provider_status, yahoo_price
from omega.engine import REFRESH_SEC, STATE
from omega.util import VERSION, cache, clean, http, now_iso

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("omega.app")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# single-user access
# --------------------------------------------------------------------------- #
# This dashboard is meant for one person.  Everything except the liveness probe
# is behind a password, because the data behind it (and the ability to burn the
# instance's network budget) is not something to leave open to the internet.
#
# SITE_PASSWORD should be set in the host's environment.  When it is not, the
# site still works but falls back to a known default, and the UI says so --
# silently running without a password is the one outcome worth avoiding.
_DEFAULT_PASSWORD = "omega-min-2026"
SITE_PASSWORD = os.environ.get("SITE_PASSWORD") or _DEFAULT_PASSWORD
USING_DEFAULT_PASSWORD = not os.environ.get("SITE_PASSWORD")

# A signed cookie needs a key; without one Flask invents a new key per process
# and every restart would log the user out.  Render can generate a stable value.
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24).hex()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30)

# Paths reachable without a session.  The probe must stay open so the platform
# can tell a healthy instance from a dead one.
_PUBLIC = {"/health", "/login", "/logout"}

_LOGIN_HTML = """<!doctype html><html lang="th"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>เข้าสู่ระบบ · OMEGA</title>
<style>
  :root{color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;min-height:100vh;display:grid;place-items:center;background:#070a0f;
       color:#e8e6e1;font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif}
  .box{width:min(400px,92vw);background:#0d1218;border:1px solid #1e2733;border-radius:16px;
       padding:28px 26px;box-shadow:0 24px 60px rgba(0,0,0,.55)}
  h1{margin:0 0 4px;font-size:19px;letter-spacing:.02em}
  .sub{color:#7d8a99;font-size:12.5px;margin-bottom:20px;line-height:1.55}
  .gold{color:#d9b166}
  input{width:100%;padding:12px 13px;border-radius:10px;border:1px solid #26303d;
        background:#080c11;color:#e8e6e1;font-size:15px;outline:none}
  input:focus{border-color:#d9b166}
  button{width:100%;margin-top:12px;padding:12px;border:0;border-radius:10px;cursor:pointer;
         background:linear-gradient(180deg,#e3c07a,#c99f4e);color:#0a0d12;font-weight:700;font-size:15px}
  .err{margin-top:12px;color:#ff8b8b;font-size:13px}
  .note{margin-top:16px;color:#66707d;font-size:11.5px;line-height:1.6}
</style></head><body>
<form class="box" method="post">
  <h1>OMEGA <span class="gold">Singularity</span></h1>
  <div class="sub">แดชบอร์ดส่วนตัว · ใส่รหัสผ่านเพื่อเข้าใช้</div>
  <input type="password" name="pw" placeholder="รหัสผ่าน" autofocus autocomplete="current-password">
  <button type="submit">เข้าใช้งาน</button>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <div class="note">ข้อมูลออปชันจาก CBOE เป็นแบบล่าช้า · หน้าเว็บจะแสดงเวลาของข้อมูลทุกครั้ง</div>
</form></body></html>"""


@app.before_request
def _require_login():
    path = request.path
    if path in _PUBLIC or path.startswith("/static"):
        return None
    if session.get("ok"):
        return None
    if path.startswith("/api/"):
        return jsonify(error="ต้องเข้าสู่ระบบก่อน", login=url_for("login")), 401
    return redirect(url_for("login", next=path))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        given = (request.form.get("pw") or "").strip()
        if hmac.compare_digest(given, SITE_PASSWORD):
            session.clear()
            session["ok"] = True
            session.permanent = True
            nxt = request.args.get("next") or "/"
            # Only ever bounce to a local path -- an open redirect here would let
            # a crafted link send the user somewhere else after logging in.
            if not nxt.startswith("/") or nxt.startswith("//"):
                nxt = "/"
            return redirect(nxt)
        time.sleep(0.6)                 # slow down guessing
        error = "รหัสผ่านไม่ถูกต้อง"
        log.warning("failed login from %s", request.headers.get("X-Forwarded-For", "?"))
    elif session.get("ok"):
        return redirect("/")
    return render_template_string(_LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

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

# How long a cold /api/scan will wait for its own pass to produce something
# before answering with what it has.  Kept well under the server's request
# timeout: the scan publishes each ticker as it lands, so a couple of seconds is
# usually enough for the first rows and the rest arrive on the next poll.
INLINE_WAIT_SEC = float(os.environ.get("INLINE_WAIT_SEC", "20"))

# How long a pass may make no progress at all before the supervisor treats it
# as wedged.  It has to sit above the slowest healthy pass: fifteen tickers,
# four at a time, each with its own ceiling of ANALYZE_TIMEOUT.
STALL_SEC = float(os.environ.get("STALL_SEC", "420"))


def _inline_scan():
    """One scan on behalf of a request that found the dashboard empty."""
    try:
        engine.scan_now_blocking()
    except BaseException as ex:
        log.warning("inline scan failed: %s", ex)


# Every worker that ever died, with its traceback.  A background thread that
# dies quietly is indistinguishable from one that was never started, and both
# look the same from outside: an empty page.  Keeping the reason means the
# cause can be read instead of guessed at.
WORKER_DEATHS: list = []


def _record_death(name: str, ex: BaseException) -> None:
    import traceback

    WORKER_DEATHS.append(dict(name=name, error=f"{type(ex).__name__}: {str(ex)[:200]}",
                              trace=traceback.format_exc()[-1500:], at=now_iso()))
    del WORKER_DEATHS[:-10]
    log.exception("worker %s died", name)


def _scan_loop():
    try:
        _scan_loop_body()
    except BaseException as ex:
        _record_death("omega-scan", ex)


def _scan_loop_body():
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
    try:
        _watchdog_body()
    except BaseException as ex:
        _record_death("omega-watchdog", ex)


def _watchdog_body():
    """Restart a background worker if it ever stops, or unstick a wedged scan.

    A scanner that dies silently leaves the dashboard empty with no clue why,
    which is exactly the failure this project kept hitting; the watchdog makes
    that state self-healing instead of requiring a redeploy.

    The second failure mode is subtler: the thread is alive but the scan is
    wedged on a provider that never answers, so the lock stays held and no new
    pass can ever start.  Detecting that by *progress* rather than by thread
    existence is what turns a permanently blank page into a self-healing one.

    The remedy is deliberately gentle.  Releasing the lock out from under a
    running pass leaves the pass holding a lock it no longer owns, and swapping
    the analysis semaphore orphans whoever was waiting on the old one -- both
    turn one slow pass into a permanently broken service.  So the pass is
    *asked* to stop and gives up at the next ticker, and only a service that is
    still stuck a long time after that is replaced outright.
    """
    unstuck_at = 0.0
    while True:
        time.sleep(60)
        try:
            names = {t.name for t in threading.enumerate()}
            for fn, name in ((_scan_loop, "omega-scan"), (_probe_loop, "omega-probe")):
                if name not in names:
                    log.warning("watchdog: %s is gone -- restarting", name)
                    threading.Thread(target=fn, daemon=True, name=name).start()

            # A pass that is slow is not a wedged pass.  The bar has to sit
            # above the slowest healthy pass (fifteen tickers, four at a time,
            # each with its own ceiling), otherwise the watchdog tears down a
            # scan that was working exactly as designed.
            if engine.scan_stalled(STALL_SEC):
                if not unstuck_at:
                    unstuck_at = time.time()
                    log.warning("watchdog: scan made no progress for %ss -- "
                                "asking the pass to stop", STALL_SEC)
                    engine.request_scan_stop()
                elif time.time() - unstuck_at > 600.0:
                    # Still stuck long after being asked to stop: the worker is
                    # in a syscall we cannot interrupt.  A clean process is the
                    # only reliable cure, and the platform restarts us at once.
                    log.error("watchdog: scan still wedged after escalation -- "
                              "exiting so the platform starts a clean worker")
                    logging.shutdown()
                    os._exit(1)
            else:
                unstuck_at = 0.0
        except BaseException as ex:
            log.info("watchdog: %s", ex)


def _probe_loop():
    try:
        _probe_loop_body()
    except BaseException as ex:
        _record_death("omega-probe", ex)


def _probe_loop_body():
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
        if app.debug and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            return
        if not _BOOTED:
            _BOOTED = True
            log.info("booting background workers")
        # Every call checks the workers, not just the first one.  A worker that
        # died -- or was never started because the process was already under
        # pressure when it booted -- left the dashboard empty with nothing to
        # bring it back, because the one-shot flag said "already booted".
        names = {t.name for t in threading.enumerate()}
        for fn, name in ((_scan_loop, "omega-scan"), (_probe_loop, "omega-probe"),
                         (_watchdog, "omega-watchdog")):
            if name not in names:
                log.warning("booting background workers: %s is missing -- starting it", name)
                threading.Thread(target=fn, daemon=True, name=name).start()
        log.info("background workers alive · watchlist=%s", ",".join(engine.WATCHLIST))


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    _boot()          # make sure the workers exist before the first page view
    src = _template_source()
    if isinstance(src, tuple):
        return src
    return render_template_string(src, refresh=REFRESH_SEC, version=VERSION,
                                  using_default_password=USING_DEFAULT_PASSWORD)


@app.route("/health")
def health():
    # The commit the platform deployed, so a version on screen can be tied to
    # the code in the repository that produced it.
    build = (os.environ.get("RENDER_GIT_COMMIT") or "")[:7]
    return jsonify(ok=True, version=VERSION, build=build,
                   updated=STATE.get("updated"),
                   items=len(STATE.get("items", [])), memory=_memory_report(),
                   time=now_iso())


@app.route("/api/scan")
def api_scan():
    # Never serve an empty dashboard while the answer is available.  A free
    # instance can wedge a background pass, and a visitor who arrives then would
    # otherwise stare at nothing until the watchdog noticed.
    #
    # The pass runs on a background thread and this request waits only a short,
    # bounded time for the first results, because the scan publishes each ticker
    # as it lands and a cold request must not sit anywhere near the server's
    # request timeout.  Whatever has arrived is returned; the pass carries on and
    # the next poll picks up the rest.
    if not (STATE.get("items") or []):
        # A lock held by a thread that no longer exists is a lock nobody will
        # ever release.  Take it back rather than waiting for a pass that is
        # not coming.
        if engine.scan_lock.locked() and not engine.scan_running():
            log.warning("api/scan: the scan lock was left behind by a dead worker "
                        "-- taking it back")
            try:
                engine.scan_lock.release()
            except Exception:
                pass
        if engine.scan_stalled(90.0) and engine.scan_running():
            log.warning("api/scan: taking over a wedged scan")
            engine.request_scan_stop()
        if not engine.scan_lock.locked():
            threading.Thread(target=_inline_scan, daemon=True, name="omega-inline-scan").start()
            deadline = time.time() + INLINE_WAIT_SEC
            while time.time() < deadline:
                if STATE.get("items"):
                    break
                if not engine.scan_lock.locked():
                    break
                time.sleep(1.0)
    with engine._STATE_LOCK:
        payload = clean(dict(STATE))
    payload["cache"] = cache.stats()
    return jsonify(payload)


@app.route("/api/scan/now")
def api_scan_now():
    """Run one scan in the background right now, ignoring the refresh timer.

    Useful when the dashboard is empty and you would rather not wait up to a
    whole refresh window (or a cold start) to find out whether the data
    sources answer from this host.  Overlapping scans are refused: two passes
    at once only waste memory and make the progress counter meaningless.
    """
    def run():
        try:
            engine.scan_once()   # takes scan_lock itself; no-op if one is running
            log.info("manual scan finished (%d items)", len(STATE.get("items") or []))
        except BaseException as ex:
            log.exception("manual scan failed: %s", ex)

    if engine.scan_lock.locked():
        return jsonify(started=False, reason="สแกนรอบก่อนยังไม่เสร็จ",
                       progress=dict(STATE.get("progress") or {}),
                       items=len(STATE.get("items") or []), time=now_iso())
    threading.Thread(target=run, daemon=True, name="omega-manual-scan").start()
    return jsonify(started=True, progress=dict(STATE.get("progress") or {}),
                   items=len(STATE.get("items") or []), time=now_iso())


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
    _boot()
    # If nothing has ever been scanned, do not make the first visitor wait for
    # the thread's own schedule -- start a pass immediately.
    prog = STATE.get("progress") or {}
    if not STATE.get("items") and not prog.get("done") and not prog.get("total"):
        if not any(t.name == "omega-manual-scan" for t in threading.enumerate()):
            threading.Thread(target=engine.scan_once, daemon=True,
                             name="omega-manual-scan").start()
    threads = [t.name for t in threading.enumerate()]
    return jsonify(booted=_BOOTED, threads=threads,
                   scan_alive=any(t.name == "omega-scan" for t in threading.enumerate()),
                   progress=dict(STATE.get("progress") or {}),
                   items=len(STATE.get("items") or []),
                   scan_locked=engine.scan_lock.locked(),
                   scan_running=engine.scan_running(),
                   scan_stalled=engine.scan_stalled(STALL_SEC),
                   deaths=list(WORKER_DEATHS),
                   stall_sec=STALL_SEC,
                   analyze_timeout=engine.ANALYZE_TIMEOUT,
                   trace=engine.trace_snapshot(),
                   heartbeat_age=round(time.time() - (engine._HEARTBEAT.get("t") or 0), 1)
                   if engine._HEARTBEAT.get("t") else None,
                   time=now_iso())


@app.route("/api/sources")
def api_sources():
    return jsonify(clean(news.source_status()))


@app.route("/api/threads")
def api_threads():
    """Where every thread in the process is right now.

    A scan that stops making progress on a platform we cannot read the logs
    of is otherwise a mystery.  This dumps the live stack of each thread, so
    the waiting point is visible instead of guessed at.
    """
    import sys
    import traceback

    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for tid, frame in sys._current_frames().items():
        try:
            stack = "".join(traceback.format_stack(frame)[-8:])
        except Exception:
            stack = ""
        out.append(dict(name=names.get(tid, "?"), stack=stack))
    return jsonify(threads=out, trace=engine.trace_snapshot(),
                   progress=dict(engine.STATE.get("progress") or {}),
                   time=now_iso())


@app.route("/api/track")
def api_track():
    return jsonify(clean(track.scoreboard()))


@app.route("/api/position")
def api_position():
    """Check the strikes the user tells us they hold, against the live tape."""
    sym = re.sub(r"[^A-Za-z]", "", request.args.get("ticker", ""))[:6].upper()
    if not sym:
        return jsonify(error="ใส่ ticker หุ้นสหรัฐ เช่น SOFI"), 400

    def _f(name):
        v = (request.args.get(name) or "").strip()
        if not v:
            return None
        try:
            x = float(v)
        except ValueError:
            return None
        return x if x > 0 else None

    call_k = _f("call")
    put_k = _f("put")
    if call_k is None and put_k is None:
        return jsonify(error="กรอก strike อย่างน้อยหนึ่งขา (CALL หรือ PUT)"), 400
    try:
        res = engine.position_check(sym, call_k, put_k,
                                    _f("call_prem"), _f("put_prem"))
    except Exception as ex:
        log.warning("position %s: %s", sym, ex)
        return jsonify(error=f"{type(ex).__name__}: {str(ex)[:160]}"), 502
    return jsonify(clean(res))


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


def _memory_report():
    """Resident size of this process, so the instance can be watched honestly.

    The most useful single number when a free instance keeps restarting: it says
    whether the scan is simply slow (memory flat) or the worker is climbing
    toward the platform limit (memory rising every cycle).
    """
    try:
        import psutil
        p = psutil.Process(os.getpid())
        return dict(rss_mb=round(p.memory_info().rss / 1e6, 1),
                    available_mb=round(psutil.virtual_memory().available / 1e6, 1),
                    total_mb=round(psutil.virtual_memory().total / 1e6, 1))
    except Exception:
        return dict(rss_mb=None, note="psutil ไม่พร้อมใช้งาน")


@app.route("/api/diag")
def api_diag():
    """Live self-test: which providers answer from the server that runs this.

    The probe touches every provider at once.  On a small instance that is the
    one request most likely to overlap a scan and push the process over its
    memory limit, so it politely waits for a scan in progress to finish instead
    of racing it.  (It is a diagnostic, not a hot path -- a few seconds of
    patience is the right trade for not getting the worker killed.)
    """
    if engine.scan_lock.locked():
        got = engine.scan_lock.acquire(timeout=120)
        if got:
            engine.scan_lock.release()
        else:
            return jsonify(error="กำลังสแกนอยู่ กรุณาลอง /api/diag อีกครั้ง", busy=True), 503
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
    probe("SEC EDGAR submissions", lambda: news.sec_filings("AAPL"))
    probe("Stocktwits", lambda: news.stocktwits("AAPL"))
    probe("Alpha Vantage news", lambda: news.alphavantage("AAPL"))
    return jsonify(clean(dict(version=VERSION, now=now_iso(), checks=checks,
                              providers=provider_status(), fetchers=news.FETCH,
                              memory=_memory_report(),
                              watchlist=engine.WATCHLIST, circuit=__import__("omega.util", fromlist=["breaker"]).breaker.state(),
                              config=dict(sims=engine.SIMS, refresh_sec=REFRESH_SEC,
                                          scan_workers=engine.SCAN_WORKERS,
                                          analyze_slots=engine.ANALYZE_SLOTS,
                                          analyze_timeout=engine.ANALYZE_TIMEOUT,
                                          stall_sec=STALL_SEC,
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
