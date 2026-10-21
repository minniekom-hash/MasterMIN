"""Omega Singularity Engine v2 — shared helpers.

Everything here is intentionally dependency-light (numpy + requests only) so the
service stays small enough for a 512 MB free instance, and so an unreachable
provider can never stall a request: every network call has a hard timeout and a
per-host circuit breaker.
"""
from __future__ import annotations

import logging
import math
import os
import queue
import random
import socket
import threading
import time
from datetime import datetime, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("omega")

VERSION = "2.2.0"

CONTACT = os.environ.get("CONTACT_EMAIL", "omega.research@example.com")

BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]
# SEC requires an honest, contactable User-Agent.
SEC_UA = f"OmegaSingularityEngine/{VERSION} (research; contact: {CONTACT})"

# A process-wide floor on socket operations.  requests' own timeout covers the
# connect and the read, but *not* DNS resolution, and a resolver that never
# answers is a classic way for one call to hang a worker forever -- which is
# exactly what parked the scan at 0/15 while the process looked perfectly
# healthy.  setdefaulttimeout() bounds every socket, including getaddrinfo.
import socket as _socket

try:
    _socket.setdefaulttimeout(20.0)
except Exception:
    pass


# --------------------------------------------------------------------------- #
# env helpers
# --------------------------------------------------------------------------- #
def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "").strip() or default))
    except Exception:
        return default


def env_list(name: str, default: list) -> list:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    out = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip().upper()
        if part and part not in out:
            out.append(part)
    return out


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def release_memory() -> None:
    """Ask the allocator to hand freed pages back to the operating system.

    CPython frees objects but glibc keeps the resulting pages in per-thread
    arenas, so a process that has just analysed 15 tickers can sit at its high
    water mark forever.  On a 512MB instance that is the difference between
    finishing a scan and being killed halfway through the next one.

    ``malloc_trim`` is glibc-only and returns those pages; everywhere else this
    is a deliberate no-op.  It is cheap enough to call after every ticker.
    """
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def age_hours(ts) -> float:
    """Age in hours of an epoch number, ISO-8601 string or RFC-822 date string."""
    try:
        if ts is None or ts == "":
            return 24.0
        if isinstance(ts, (int, float)):
            v = float(ts)
            if v > 1e12:  # milliseconds
                v /= 1000.0
            d = datetime.fromtimestamp(v, timezone.utc)
        else:
            s = str(ts).strip()
            if s.isdigit():
                return age_hours(int(s))
            # ISO first, by trying it rather than sniffing the text.  Sniffing for
            # a literal "T" looks reasonable until you meet RSS dates like
            # "Fri, 16 Oct 2026 11:15:36 GMT" -- GMT contains a T, so every feed
            # item was sent to the ISO parser, failed, and was aged at the 24h
            # default.  That made stale news look a day old and hid its date.
            try:
                d = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                from email.utils import parsedate_to_datetime

                d = parsedate_to_datetime(s)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - d).total_seconds() / 3600.0, 0.0)
    except Exception:
        return 24.0


# --------------------------------------------------------------------------- #
# TTL cache
# --------------------------------------------------------------------------- #
_MISS = object()


class TTLCache:
    # An entry that is never read again is never evicted, so the cache used to
    # grow for as long as the process lived: every option chain, every daily
    # history, every article body, for ever.  On a small instance that is what
    # eventually killed the worker.  Expired entries are now swept, and the
    # cache has a ceiling as well.
    MAX_ENTRIES = 600

    def __init__(self) -> None:
        self._data: dict = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return _MISS
            ts, ttl, val = hit
            if time.time() - ts < ttl:
                return val
            self._data.pop(key, None)
            return _MISS

    def set(self, key, val, ttl: float) -> None:
        with self._lock:
            self._data[key] = (time.time(), float(ttl), val)
            if len(self._data) > self.MAX_ENTRIES:
                self._sweep_locked()

    def _sweep_locked(self) -> None:
        """Drop everything that has expired.  Called with the lock held."""
        now = time.time()
        dead = [k for k, (ts, ttl, _v) in self._data.items() if now - ts >= ttl]
        for k in dead:
            self._data.pop(k, None)
        # Still over the ceiling: the oldest entries go, expired or not.
        if len(self._data) > self.MAX_ENTRIES:
            order = sorted(self._data.items(), key=lambda kv: kv[1][0])
            for k, _v in order[:len(self._data) - self.MAX_ENTRIES]:
                self._data.pop(k, None)

    def sweep(self) -> int:
        """Remove expired entries; returns how many were dropped."""
        with self._lock:
            before = len(self._data)
            self._sweep_locked()
            return before - len(self._data)

    def wrap(self, key, ttl: float, fn):
        val = self.get(key)
        if val is not _MISS:
            return val
        val = fn()
        if val is not None:
            self.set(key, val, ttl)
        return val

    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            live = sum(1 for ts, ttl, _v in self._data.values() if now - ts < ttl)
            return {"entries": len(self._data), "live": live}


cache = TTLCache()


# --------------------------------------------------------------------------- #
# HTTP with hard timeouts + circuit breaker
# --------------------------------------------------------------------------- #
class Breaker:
    """Per-host circuit breaker: stop hammering a provider that is refusing us."""

    # Two failures is enough to conclude a host is refusing this instance, and a
    # short cooldown means a host that recovers is picked up again within a
    # couple of scan cycles.  A high threshold here is what let a blocked Yahoo
    # spend minutes of the scan budget on timeouts before it was skipped.
    def __init__(self, threshold: int = 2, cooldown: float = 120.0) -> None:
        self.threshold = threshold
        self.cooldown = cooldown
        self._fail: dict = {}
        self._open_until: dict = {}
        self._lock = threading.Lock()

    def allow(self, host: str) -> bool:
        with self._lock:
            until = self._open_until.get(host, 0.0)
            if until and time.time() < until:
                return False
            if until and time.time() >= until:
                self._open_until.pop(host, None)
                self._fail[host] = 0
            return True

    def healthy(self, host: str) -> bool:
        """True unless this host has tripped the breaker recently.

        Used to *reorder* provider fallbacks: when Yahoo is rate-limiting a
        datacentre IP, trying it first for every ticker burns the whole scan
        budget on timeouts before the working providers are ever reached.
        """
        return self.allow(host)

    def ok(self, host: str) -> None:
        with self._lock:
            self._fail[host] = 0
            self._open_until.pop(host, None)

    def fail(self, host: str) -> None:
        with self._lock:
            n = self._fail.get(host, 0) + 1
            self._fail[host] = n
            if n >= self.threshold:
                self._open_until[host] = time.time() + self.cooldown
                log.warning("circuit open for %s (cooldown %ss)", host, self.cooldown)

    def state(self) -> dict:
        with self._lock:
            return {h: {"fails": self._fail.get(h, 0), "open_until": round(u, 0)}
                    for h, u in self._open_until.items()}


breaker = Breaker()


# --------------------------------------------------------------------------- #
# name resolution with a deadline
# --------------------------------------------------------------------------- #
# A name lookup is the one part of a request that no timeout covers: the
# socket timeout only starts once the address has been resolved.  On the
# platform this runs on, the resolver stops answering from time to time -- and
# a lookup that never returns parks the thread that asked for it, which is how
# a whole scan used to stop making progress with no CPU being used at all.
#
# So every lookup goes through here: it is cached, it has a deadline, and when
# the deadline passes the caller is told the name could not be resolved
# instead of waiting for a resolver that is not answering.  A failed lookup
# then counts as a failed request, the circuit breaker opens, and the fallback
# providers take over.
_REAL_GETADDRINFO = socket.getaddrinfo
_DNS_CACHE: dict = {}
_DNS_INFLIGHT: dict = {}
_DNS_LOCK = threading.Lock()
_DNS_QUEUE: "queue.Queue" = queue.Queue(maxsize=64)
# A small, fixed set of resolver threads.  Every lookup used to get a thread of
# its own, and on a platform where a lookup can hang for ever those threads
# piled up until starting a new one blocked -- which is what stopped the scan
# dead.  Now a lookup that hangs costs one resolver thread and nothing else:
# the callers waiting on it give up and the fallbacks take over.
DNS_WORKERS = 4
DNS_TIMEOUT = 4.0
DNS_TTL = 60.0
_DNS_STARTED = False


def _dns_worker():
    while True:
        job = _DNS_QUEUE.get()
        if job is None:
            return
        key, args, holder = job
        try:
            res = _REAL_GETADDRINFO(*args)
        except Exception as ex:
            res = ex
        holder.append(res)
        with _DNS_LOCK:
            if not isinstance(res, Exception):
                _DNS_CACHE[key] = (time.time(), res)


def _dns_start() -> None:
    global _DNS_STARTED
    with _DNS_LOCK:
        if _DNS_STARTED:
            return
        _DNS_STARTED = True
    for i in range(DNS_WORKERS):
        threading.Thread(target=_dns_worker, daemon=True, name="omega-dns%d" % i).start()


def _getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """socket.getaddrinfo, with a cache, a deadline and a bounded worker pool."""
    _dns_start()
    key = (host, port, family, type, proto, flags)
    with _DNS_LOCK:
        hit = _DNS_CACHE.get(key)
        if hit and (time.time() - hit[0]) < DNS_TTL:
            return hit[1]
        holder = _DNS_INFLIGHT.get(key)
        if holder is None:
            if _DNS_QUEUE.full():
                raise socket.gaierror(socket.EAI_AGAIN, "resolver queue is full")
            holder = []
            _DNS_INFLIGHT[key] = holder
            _DNS_QUEUE.put((key, (host, port, family, type, proto, flags), holder))
    deadline = time.time() + DNS_TIMEOUT
    while time.time() < deadline and not holder:
        time.sleep(0.05)
    with _DNS_LOCK:
        _DNS_INFLIGHT.pop(key, None)
    if not holder:
        # The resolver is not answering.  Say so, do not wait for it.
        raise socket.gaierror(socket.EAI_AGAIN, "name resolution timed out")
    res = holder[0]
    if isinstance(res, Exception):
        raise res
    return res


socket.getaddrinfo = _getaddrinfo


class Http:
    """Tiny requests wrapper: retries, per-host stats, breaker, connection cap."""

    def __init__(self, max_concurrency: int = 24) -> None:
        # The cap must sit comfortably above the number of threads that can ask
        # for a slot at once (scan workers x per-ticker pool + news pools +
        # diagnostics), otherwise the gate itself becomes the bottleneck.  24 is
        # generous but still far below anything that would exhaust sockets, and
        # the timed acquire below means even a mis-set value cannot deadlock.
        self._local = threading.local()
        self._sem = threading.BoundedSemaphore(max_concurrency)
        self.stats: dict = {}
        self._slock = threading.Lock()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            # Keep retries short.  A blocked host (Yahoo throttles datacentre IPs)
            # would otherwise cost several seconds per call, and the scan makes
            # dozens of calls -- enough to outlive the whole request window on a
            # free instance.  One quick connect retry is plenty; the circuit
            # breaker handles a genuinely dead host from then on.
            retry = Retry(total=1, connect=1, read=0, backoff_factor=0.3,
                          status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(["GET"]))
            ad = HTTPAdapter(max_retries=retry, pool_connections=12, pool_maxsize=12)
            s.mount("https://", ad)
            s.mount("http://", ad)
            self._local.s = s
        return s

    def _bump(self, host: str, ok: bool, ms: int) -> None:
        with self._slock:
            d = self.stats.setdefault(host, {"ok": 0, "fail": 0, "ms": 0, "n": 0, "last": None})
            d["ok" if ok else "fail"] += 1
            d["n"] += 1
            d["ms"] = int(0.7 * d["ms"] + 0.3 * ms) if d["n"] > 1 else ms
            d["last"] = now_iso()

    def get(self, url: str, timeout=(5, 15), headers: dict | None = None,
            sec: bool = False, allow_status=(200,)):
        """Return the requests.Response or None. Never raises."""
        host = url.split("/")[2] if "://" in url else url
        if not breaker.allow(host):
            return None
        hdrs = {"User-Agent": SEC_UA if sec else random.choice(BROWSER_UAS),
                "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate"}
        if headers:
            hdrs.update(headers)
        t0 = time.time()
        got = None
        # Never block forever on the concurrency gate.  The scan runs several
        # pools at once (per-ticker fetch, news, per-strike), so the number of
        # threads wanting a slot can exceed the cap; an unbounded acquire() then
        # deadlocks, every worker parks here and the scan sits at 0/15 forever.
        # If no slot frees up in time the request is simply treated as failed,
        # which the circuit breaker and the fallbacks already handle.
        if not self._sem.acquire(timeout=20):
            log.info("HTTP busy, dropping %s", url[:110])
            self._bump(host, False, int((time.time() - t0) * 1000))
            return None
        try:
            r = self._session().get(url, headers=hdrs, timeout=timeout)
            if r.status_code in allow_status:
                got = r
                breaker.ok(host)
            else:
                breaker.fail(host)
                log.info("HTTP %s %s", r.status_code, url[:110])
        except Exception as ex:
            breaker.fail(host)
            log.info("HTTP err %s %s (%s)", type(ex).__name__, url[:110], str(ex)[:80])
        finally:
            self._sem.release()
            self._bump(host, got is not None, int((time.time() - t0) * 1000))
        return got

    def json(self, url: str, **kw):
        r = self.get(url, **kw)
        if r is None:
            return None
        try:
            return r.json()
        except Exception:
            return None

    def text(self, url: str, **kw):
        r = self.get(url, **kw)
        if r is None:
            return None
        try:
            return r.text
        except Exception:
            return None


http = Http()


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def clean(o):
    """Make an object JSON safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def clip(x, lo, hi):
    return float(max(lo, min(hi, x)))


def nz(x, default: float = 0.0) -> float:
    """Return x as a finite float, or `default`.

    NaN is contagious: one NaN price reaches ``np.clip``, survives it (clip
    propagates NaN rather than clamping it), and then poisons an entire
    simulation.  Every value that crosses into the models goes through here so a
    single bad field can only ever cost us one number, never a whole page.
    """
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(f):
        return float(default)
    return f


def safe_div(a, b, default=0.0):
    try:
        return a / b if b else default
    except Exception:
        return default


def pct_rank(sorted_vals, x) -> float:
    """Percentile (0-100) of x inside a list of values."""
    if not sorted_vals:
        return 50.0
    import bisect

    return 100.0 * bisect.bisect_right(sorted_vals, x) / len(sorted_vals)
