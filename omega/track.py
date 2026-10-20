"""Omega v2 — self-scoring prediction log.

Every pick the engine publishes is written down with its predicted probability
of profit.  Once the contract has expired the realised outcome is read back from
the closing price on that date, so the site can show a *real* hit rate and how
well the predicted probabilities were calibrated — instead of only claiming
accuracy.

The log is a small JSON file next to the app (ephemeral on Render's free tier,
so it is honest about "since last deploy").
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from . import data
from .util import log, now_iso

PATH = os.environ.get("TRACK_FILE", os.path.join(os.getcwd(), "omega_track.json"))
MAX_ENTRIES = 1500
_LOCK = threading.Lock()
_DATA = None


def _load() -> list:
    global _DATA
    if _DATA is None:
        try:
            with open(PATH, "r", encoding="utf-8") as f:
                _DATA = json.load(f)
            if not isinstance(_DATA, list):
                _DATA = []
        except Exception:
            _DATA = []
    return _DATA


def _save() -> None:
    try:
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(_DATA[-MAX_ENTRIES:], f, ensure_ascii=False)
    except Exception as ex:
        log.info("track save failed: %s", ex)


def record(sym: str, best: dict, spot: float, cycle: str) -> None:
    if not best or not best.get("expiry"):
        return
    with _LOCK:
        rows = _load()
        key = (sym, best["expiry"], best["kind"], best["strike"])
        if any((r.get("sym"), r.get("expiry"), r.get("kind"), r.get("strike")) == key and not r.get("done")
               for r in rows):
            return
        rows.append(dict(sym=sym, expiry=best["expiry"], kind=best["kind"], strike=best["strike"],
                         premium=best["premium"], spot=round(spot, 2), pop=best["pop"],
                         pop_lo=best.get("pop_lo"), pop_hi=best.get("pop_hi"), omega=best.get("omega"),
                         breakeven=best.get("breakeven"), added=now_iso(), cycle=cycle,
                         done=False, result=None, exit_spot=None))
        _save()


def evaluate() -> int:
    """Score every logged pick whose expiry has passed."""
    today = datetime.now(timezone.utc).date()
    done = 0
    with _LOCK:
        rows = _load()
        for r in rows:
            if r.get("done"):
                continue
            try:
                exp = datetime.strptime(r["expiry"], "%Y-%m-%d").date()
            except Exception:
                r["done"] = True
                continue
            if exp > today:
                continue
            try:
                d = data.daily(r["sym"], ttl=3600)
                if d is None or d.empty:
                    continue
                hist = d[d.index.date <= exp]
                if hist.empty:
                    continue
                st = float(hist["Close"].iloc[-1])
            except Exception:
                continue
            be = r.get("breakeven")
            if be is None:
                be = r["strike"] + r["premium"] if r["kind"] == "CALL" else r["strike"] - r["premium"]
            win = (st > be) if r["kind"] == "CALL" else (st < be)
            r.update(done=True, result="win" if win else "loss", exit_spot=round(st, 2))
            done += 1
        if done:
            _save()
    return done


def scoreboard() -> dict:
    rows = _load()
    fin = [r for r in rows if r.get("done") and r.get("result")]
    wins = sum(1 for r in fin if r["result"] == "win")
    open_rows = [r for r in rows if not r.get("done")]
    buckets = {}
    for r in fin:
        p = r.get("pop")
        if p is None:
            continue
        b = min(int(p // 10) * 10, 90)
        d = buckets.setdefault(b, [0, 0])
        d[1] += 1
        if r["result"] == "win":
            d[0] += 1
    cal = [dict(bucket=b, n=v[1], wins=v[0],
                predicted=round(sum(x["pop"] for x in fin if x.get("pop") is not None and min(int(x["pop"] // 10) * 10, 90) == b) / v[1], 1),
                realized=round(100.0 * v[0] / v[1], 1)) for b, v in sorted(buckets.items())]
    return dict(total=len(rows), open=len(open_rows), closed=len(fin), wins=wins,
                hit_rate=round(100.0 * wins / len(fin), 1) if fin else None,
                avg_predicted=round(sum(r["pop"] for r in fin if r.get("pop") is not None) / max(len([r for r in fin if r.get("pop") is not None]), 1), 1) if fin else None,
                calibration=cal, since=min((r.get("added") for r in rows), default=None))
