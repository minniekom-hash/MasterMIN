"""OPTION TRADE: The Omega Singularity Engine — backend (Flask).
Five-model ensemble: A Monte Carlo GBM, B BSM Greeks + IV rank, C order-flow proxy,
D technical confluence, E NLP catalyst. Educational tool, not financial advice."""
import os, re, time, math, threading, logging
import xml.etree.ElementTree as ET
from urllib import robotparser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np, pandas as pd, pytz, requests, yfinance as yf
from scipy.stats import norm
from flask import Flask, jsonify, render_template

import nltk
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("omega")
app = Flask(__name__)

_wl = os.environ.get("WATCHLIST", "SPY,QQQ,IWM,AAPL,MSFT,NVDA,TSLA,AMD,AMZN,META,GOOGL,NFLX,AVGO,PLTR,COIN,JPM,BAC,INTC,MU,SOFI").split(",")
WATCHLIST = [x.strip().upper() for x in _wl if re.fullmatch(r"[A-Z]{1,5}", x.strip().upper())]  # US tickers only
MIN_VOL, MIN_OI = int(os.environ.get("MIN_VOL", 50)), int(os.environ.get("MIN_OI", 200))
MAX_PREMIUM_PCT = float(os.environ.get("MAX_PREMIUM_PCT", 0.03))  # "cheap" = premium <= 3% of spot
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")
RF = float(os.environ.get("RISK_FREE", "0.045"))
N_PATHS, REFRESH_SEC = 10_000, 15
WEIGHTS = {"A": 0.25, "B": 0.12, "C": 0.15, "D": 0.20, "E": 0.10, "H": 0.18}
TARGET_DELTA = 0.35
NY = pytz.timezone("America/New_York")
sia = SentimentIntensityAnalyzer()
rng = np.random.default_rng()

# ---------------- TTL cache ----------------
_cache, _lock = {}, threading.Lock()
def cached(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _lock:
        _cache[key] = (now, val)
    return val

# ---------------- Model B: Black-Scholes-Merton ----------------
def bsm(S, K, T, sig, kind, r=RF):
    T, sig = max(T, 1e-6), max(sig, 1e-4)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * sq)
    d2 = d1 - sig * sq
    pdf = norm.pdf(d1)
    if kind == "CALL":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = -S * pdf * sig / (2 * sq) - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = -S * pdf * sig / (2 * sq) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    return dict(price=price, delta=delta, gamma=pdf / (S * sig * sq),
                theta=theta / 365, vega=S * pdf * sq / 100)

# ---------------- Model A: Monte Carlo GBM ----------------
def monte_carlo(S0, K, T, sigma, mu, kind, be):
    steps = max(int(T * 365 * 6.5), 8)
    dt = T / steps
    z = rng.standard_normal((N_PATHS, steps))
    logp = np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z, axis=1)
    ST = S0 * np.exp(logp[:, -1])
    call = kind == "CALL"
    itm = (ST > K) if call else (ST < K)
    prof = (ST > be) if call else (ST < be)
    ext = S0 * np.exp(logp.max(axis=1) if call else logp.min(axis=1))
    touch = (ext >= K) if call else (ext <= K)
    return float(itm.mean() * 100), float(touch.mean() * 100), float(prof.mean() * 100)

# ---------------- Data ----------------
def get_daily(t):
    return cached(f"d:{t}", 300, lambda: yf.Ticker(t).history(period="2y", interval="1d", auto_adjust=True))
def get_intraday(t):
    return cached(f"i:{t}", 60, lambda: yf.Ticker(t).history(period="5d", interval="5m", auto_adjust=True))

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

# ---------------- Model C: institutional flow proxy (signed -1..1) ----------------
def model_c(daily, intra):
    c, v = daily["Close"], daily["Volume"]
    rvol = float(v.iloc[-1] / v.iloc[-21:-1].mean()) if v.iloc[-21:-1].mean() else 1.0
    day = intra[intra.index.date == intra.index[-1].date()]
    tp = (day["High"] + day["Low"] + day["Close"]) / 3
    vwap = float((tp * day["Volume"]).sum() / max(day["Volume"].sum(), 1))
    price = float(day["Close"].iloc[-1])
    vwap_sig = float(np.clip((price / vwap - 1) * 100, -1, 1))
    ret = float(c.iloc[-1] / c.iloc[-2] - 1)
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    p_slope, o_slope = c.iloc[-1] - c.iloc[-11], obv.iloc[-1] - obv.iloc[-11]
    if p_slope > 0 and o_slope < 0: div = -1.0      # distribution
    elif p_slope < 0 and o_slope > 0: div = 1.0     # accumulation
    else: div = 0.5 * np.sign(o_slope)
    sig = 0.4 * vwap_sig + 0.3 * np.sign(ret) * min(rvol / 2, 1) + 0.3 * div
    return float(np.clip(sig, -1, 1)), dict(rvol=rvol, vwap=vwap, div=int(div), price=price)

# ---------------- Model D: technical confluence (signed -1..1) ----------------
def model_d(daily):
    c, h, l = daily["Close"], daily["High"], daily["Low"]
    def stack(s):
        a, b, d = ema(s, 9).iloc[-1], ema(s, 21).iloc[-1], ema(s, 50).iloc[-1]
        return 1.0 if a > b > d else (-1.0 if a < b < d else 0.0)
    wk = c.resample("W").last().dropna()
    wk_sig = stack(wk) if len(wk) > 50 else (1.0 if ema(wk, 9).iloc[-1] > ema(wk, 21).iloc[-1] else -1.0)
    r = rsi(c); r_now = float(r.iloc[-1])
    w, rw = c.iloc[-12:], r.iloc[-12:]
    div = 0.0
    if w.iloc[-1] <= w.min() * 1.005 and rw.iloc[-1] > rw.min() + 2: div = 1.0
    if w.iloc[-1] >= w.max() * 0.995 and rw.iloc[-1] < rw.max() - 2: div = -1.0
    macd = ema(c, 12) - ema(c, 26); hist = macd - ema(macd, 9)
    m_sig = 0.5 * np.sign(hist.iloc[-1]) + 0.5 * np.sign(hist.iloc[-1] - hist.iloc[-2])
    rsi_sig = float(np.clip((r_now - 50) / 25, -1, 1)) * 0.6 + 0.4 * div
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    up, dn = h.diff(), -l.diff()
    pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    pb = float(((c - ma) / (4 * sd) + 0.5).iloc[-1])
    bw = 4 * sd / ma
    squeeze = bool(bw.iloc[-1] <= bw.iloc[-120:].quantile(0.2))
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    st = float(((c - ll) / (hh - ll) * 100).iloc[-1])
    sig = 0.30 * stack(c) + 0.20 * wk_sig + 0.25 * rsi_sig + 0.25 * m_sig
    sig = sig * (0.8 + 0.2 * min(adx / 25, 1)) + 0.1 * float(np.sign(pdi.iloc[-1] - mdi.iloc[-1]))
    if (pb > 1.05 and sig > 0) or (pb < -0.05 and sig < 0): sig *= 0.7   # over-extension damping
    return float(np.clip(sig, -1, 1)), dict(rsi=r_now, macd_hist=float(hist.iloc[-1]), rsi_div=int(div), adx=adx, pb=pb,
                                            stoch=st, atr_pct=float(atr.iloc[-1] / c.iloc[-1] * 100), squeeze=squeeze)

# ---------------- Model E: multi-source news catalyst (signed -1..1) ----------------
# Headlines only (RSS / official APIs). Paywalled full text is never scraped.
SOURCES = {"Benzinga": "benzinga.com", "Bloomberg": "bloomberg.com", "CNBC": "cnbc.com", "WSJ": "wsj.com",
           "Reuters": "reuters.com", "Seeking Alpha": "seekingalpha.com", "Yahoo Finance": "finance.yahoo.com",
           "Investing.com": "investing.com", "Morningstar": "morningstar.com", "TradingView": "tradingview.com",
           "MarketWatch": "marketwatch.com", "IBD": "investors.com", "Motley Fool": "fool.com", "Barron's": "barrons.com",
           "Financial Times": "ft.com", "Finviz": "finviz.com", "WhaleWisdom": "whalewisdom.com",
           "GuruFocus": "gurufocus.com", "Investopedia": "investopedia.com"}
UA = {"User-Agent": "Mozilla/5.0 (OmegaEngine personal research)"}
sia.lexicon.update({"upgrade": 2.0, "upgraded": 2.0, "downgrade": -2.2, "downgraded": -2.2, "beat": 1.5, "beats": 1.5,
                    "miss": -1.5, "misses": -1.5, "surge": 2.0, "surges": 2.0, "soar": 2.2, "soars": 2.2, "plunge": -2.5,
                    "plunges": -2.5, "tumble": -2.0, "slump": -2.0, "bullish": 2.0, "bearish": -2.0, "lawsuit": -1.5,
                    "probe": -1.5, "recall": -1.5, "raises": 1.2, "cuts": -1.2, "record": 1.2, "rally": 1.8, "selloff": -2.0})

# ---------- source health monitor ----------
DATA_SITES = {"TradingView", "Finviz", "WhaleWisdom", "GuruFocus"}
PAYWALL = {"Bloomberg", "WSJ", "Barron's", "Financial Times", "Seeking Alpha", "IBD", "Morningstar"}
SRC = {n: dict(status="pending", http=None, count=0, cover=0, ms=None, error="", checked=None, last_ok=None, full=0, summary=0, none=0, why="") for n in SOURCES}
FETCH = {n: dict(status="pending", items=0, error="", checked=None, last_ok=None) for n in ("Google News RSS", "Yahoo Finance News", "NewsAPI")}
FETCHERS = {"_gnews": "Google News RSS", "_ynews": "Yahoo Finance News", "_newsapi": "NewsAPI"}

def _now(): return datetime.now(pytz.utc).isoformat()

def _mark(name, ok, n, err):
    f = FETCH[name]
    f.update(checked=_now(), items=n, error=err, status="disabled" if ok is None else ("ok" if ok else "error"))
    if ok: f["last_ok"] = f["checked"]

def probe_one(name, dom):
    t0 = time.time(); http = None
    try:
        r = requests.get("https://news.google.com/rss/search", headers=UA, timeout=10,
                         params=dict(q=f"stock market site:{dom} when:7d", hl="en-US", gl="US", ceid="US:en"))
        http = r.status_code
        if not r.ok:
            st, n, err = "error", 0, f"HTTP {r.status_code}" + (" (ถูกจำกัด request)" if r.status_code == 429 else "")
        else:
            n = len(ET.fromstring(r.content).findall(".//item")); st, err = ("ok" if n else "empty"), ""
    except Exception as ex:
        st, n, err = "error", 0, f"{type(ex).__name__}: {str(ex)[:120]}"
    d = SRC[name]; d.update(status=st, http=http, count=n, error=err, checked=_now(), ms=int((time.time() - t0) * 1000))
    if st == "ok": d["last_ok"] = d["checked"]

def probe_sources():
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(lambda kv: probe_one(*kv), SOURCES.items()))

def probe_loop():
    time.sleep(8)
    while True:
        try: probe_sources()
        except Exception as ex: log.exception(ex)
        time.sleep(600)

def _age_h(ts):
    try:
        if isinstance(ts, (int, float)): d = datetime.fromtimestamp(ts, pytz.utc)
        elif "T" in str(ts): d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        else: d = parsedate_to_datetime(str(ts))
        return max((datetime.now(pytz.utc) - d).total_seconds() / 3600, 0.0)
    except Exception:
        return 24.0

def _src_name(url, text):
    for name, dom in SOURCES.items():
        if dom in (url or ""): return name
    return text or "อื่นๆ"

def _gnews(t):
    q = f"{t} stock when:3d (" + " OR ".join(f"site:{d}" for d in SOURCES.values()) + ")"
    r = requests.get("https://news.google.com/rss/search", params=dict(q=q, hl="en-US", gl="US", ceid="US:en"), headers=UA, timeout=8)
    r.raise_for_status()
    out = []
    if r.ok:
        for it in ET.fromstring(r.content).iter("item"):
            src = it.find("source")
            out.append(dict(t=(it.findtext("title") or "").rsplit(" - ", 1)[0], url=it.findtext("link") or "", ts=it.findtext("pubDate"),
                            s=_src_name(src.get("url") if src is not None else "", src.text if src is not None else "")))
    return out

def _ynews(t):
    out = []
    for n in (yf.Ticker(t).news or []):
        c = n.get("content") or n
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""
        out.append(dict(t=c.get("title") or "", url=url, ts=c.get("pubDate") or c.get("providerPublishTime"), s="Yahoo Finance", snip=c.get("summary") or c.get("description") or ""))
    return out

def _newsapi(t):
    if not NEWSAPI_KEY: return []
    r = requests.get("https://newsapi.org/v2/everything", timeout=8, params=dict(
        q=f'"{t}" stock', domains=",".join(SOURCES.values()), language="en", pageSize=30, sortBy="publishedAt", apiKey=NEWSAPI_KEY))
    if not r.ok: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:100]}")
    arts = r.json().get("articles", [])
    return [dict(t=a.get("title") or "", url=a.get("url") or "", ts=a.get("publishedAt"), s=_src_name(a.get("url") or "", (a.get("source") or {}).get("name")), snip=((a.get("description") or "") + " " + (a.get("content") or "")).strip()) for a in arts]

# ---------- full-article reader (never bypasses paywalls; honours robots.txt) ----------
ENRICH_N = int(os.environ.get("ENRICH_N", 4))
PAYWALL_RE = re.compile(r"subscribe to (continue|read)|sign in to (read|continue)|already a subscriber|create a free account to continue|premium content", re.I)
TIER = {"Reuters": 1.0, "Bloomberg": 1.0, "WSJ": 1.0, "Financial Times": 1.0, "Barron's": 1.0, "CNBC": 1.0,
        "Yahoo Finance": 0.8, "MarketWatch": 0.8, "Investing.com": 0.8, "Morningstar": 0.8, "IBD": 0.8}  # others 0.6
EVENTS = {"ผลประกอบการ/แนวโน้ม": r"earnings|revenue|\beps\b|quarter|profit|guidance|outlook", "นักวิเคราะห์ปรับเรตติ้ง": r"upgrade|downgrade|price target|rating|initiates",
          "คดีความ/การสอบสวน": r"lawsuit|sues|probe|investigation|antitrust|settle|\bsec\b", "ควบรวม/ดีล": r"acquire|acquisition|merger|buyout|takeover|stake|partnership",
          "ผลิตภัณฑ์/อนุมัติ": r"launch|unveil|recall|\bfda\b|approval|trial", "ปันผล/ซื้อหุ้นคืน/ระดมทุน": r"buyback|dividend|offering|stock split|debt",
          "ผู้บริหาร/ปลดพนักงาน": r"\bceo\b|\bcfo\b|resign|layoff|job cuts", "มหภาค/ภาษี/ดอกเบี้ย": r"tariff|rate cut|inflation|export (ban|curb|control)"}
FIG_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|[BMT])?|\d+(?:\.\d+)?\s?%", re.I)
_robots = {}

def _allowed(url):
    p = urlparse(url); base = f"{p.scheme}://{p.netloc}"
    rp = _robots.get(base)
    if rp is None:
        rp = robotparser.RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", headers=UA, timeout=4); rp.parse(r.text.splitlines() if r.ok else [])
        except Exception:
            rp.parse([])
        _robots[base] = rp
    return rp.can_fetch(UA["User-Agent"], url)

def _article(url):
    # returns (depth, text, reason): depth full | summary | none
    if not url: return "none", "", "ไม่มีลิงก์ข่าว"
    try:
        if "news.google.com" in url:   # Google News links redirect to the publisher
            loc = requests.get(url, headers=UA, timeout=6, allow_redirects=False).headers.get("Location", "")
            if not loc or "google." in loc: return "none", "", "แปลงลิงก์ Google News เป็นลิงก์ต้นทางไม่ได้"
            url = loc
        if not _allowed(url): return "none", "", "robots.txt ของเว็บไม่อนุญาตให้อ่าน"
        r = requests.get(url, headers=UA, timeout=7)
        if r.status_code in (401, 402, 403, 429): return "none", "", f"HTTP {r.status_code}: เว็บบล็อก/ต้องสมัครสมาชิก"
        if not r.ok: return "none", "", f"HTTP {r.status_code}"
        soup = BeautifulSoup(r.text[:400000], "html.parser")
        meta = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        desc = (meta.get("content") if meta else "") or ""
        body = " ".join(x for x in (p.get_text(" ", strip=True) for p in (soup.find("article") or soup).find_all("p")) if len(x) > 60)
        if len(body) >= 600 and not PAYWALL_RE.search(body[:3000]): return "full", body[:6000], ""
        return ("summary", (desc or body)[:800], "เนื้อหาเต็มถูกจำกัด (paywall/ล็อกอิน)") if (desc or body) else ("none", "", "ไม่พบเนื้อหาในหน้าเว็บ")
    except Exception as ex:
        return "none", "", f"{type(ex).__name__}: {str(ex)[:80]}"

def _article_rec(name, url):
    r = _article(url); d = SRC.get(name)
    if d is not None:
        d[r[0]] += 1
        if r[2]: d["why"] = r[2]
    return r

def _enrich(items):
    for it in sorted(items, key=lambda i: i["age"])[:ENRICH_N]:
        d, txt, why = cached("art:" + it["url"], 3600, lambda it=it: _article_rec(it["s"], it["url"]))
        if d == "none" and it.get("snip"): d, txt = "summary", it["snip"]
        it.update(depth=d, body=txt, why=why)
    for it in items:
        it.setdefault("depth", "summary" if it.get("snip") else "headline")
        it.setdefault("body", it.get("snip", "")); it.setdefault("why", "")

def _score_item(it, t):
    hc = sia.polarity_scores(it["t"])["compound"]; body = it.get("body", ""); bc = hc
    if body:
        sents = [x for x in re.split(r"(?<=[.!?])\s+", body[:4000]) if len(x) > 30]
        rel = [x for x in sents if re.search(rf"\b{t}\b|company|shares|stock", x, re.I)] or sents
        cs = [sia.polarity_scores(x)["compound"] for x in rel[:25]]
        if cs: bc = float(np.mean(cs))
    it["cs"] = float(0.4 * hc + 0.6 * bc) if body else float(hc)
    it["mismatch"] = bool(body and abs(hc) > 0.25 and abs(bc) > 0.15 and np.sign(hc) != np.sign(bc))   # headline contradicts body

def fact_check(items):
    # cross-source verification: same event reported by several sources, with consistent direction and figures
    groups = {}
    for it in items:
        txt = (it["t"] + " " + it.get("body", "")[:600]).lower()
        for ev, pat in EVENTS.items():
            if re.search(pat, txt): groups.setdefault(ev, []).append(it)
    out = []
    for ev, g in groups.items():
        srcs = sorted({i["s"] for i in g})
        sg = [np.sign(i["cs"]) for i in g if abs(i["cs"]) > 0.1]
        pos, neg = sg.count(1), sg.count(-1)
        w = sum(TIER.get(x, 0.6) for x in srcs)
        split = bool(pos and neg and min(pos, neg) / (pos + neg) >= 0.2)
        v = "ขัดแย้ง" if split else ("ยืนยันแล้ว" if len(srcs) >= 3 and w >= 2.2 else ("ยืนยันบางส่วน" if len(srcs) >= 2 else "ยังไม่ยืนยัน"))
        fac = {"ยืนยันแล้ว": 1.0, "ยืนยันบางส่วน": 0.8, "ยังไม่ยืนยัน": 0.5, "ขัดแย้ง": 0.3}[v]
        for i in g: i["f"] = max(i.get("f", 0), fac)
        seen = {}
        for i in g:
            for f in set(x.lower().replace(" ", "") for x in FIG_RE.findall(i["t"] + " " + i.get("body", "")[:600])): seen.setdefault(f, set()).add(i["s"])
        out.append(dict(event=ev, n=len(g), sources=srcs, verdict=v, split=split, pos=pos, neg=neg,
                        figures=[f for f, ss in sorted(seen.items(), key=lambda kv: -len(kv[1])) if len(ss) >= 2][:3]))
    for i in items: i.setdefault("f", 0.6)
    return sorted(out, key=lambda x: -x["n"])[:6]

def _news(t):
    items = []
    for fn in (_gnews, _ynews, _newsapi):
        name = FETCHERS[fn.__name__]
        if fn is _newsapi and not NEWSAPI_KEY:
            _mark(name, None, 0, "ไม่ได้ตั้งค่า NEWSAPI_KEY"); continue
        try:
            got = fn(t); items += got; _mark(name, True, len(got), "")
        except Exception as ex:
            log.warning("%s %s: %s", fn.__name__, t, ex)
            _mark(name, False, 0, f"{type(ex).__name__}: {str(ex)[:140]}")
    seen, out = set(), []
    for it in items:
        k = it["t"].lower()[:60]
        if it["t"] and k not in seen:
            seen.add(k); it.setdefault("snip", ""); it["age"] = _age_h(it.get("ts")); out.append(it)
    _enrich(out)
    for it in out: _score_item(it, t); it["c"] = it["cs"]
    return out

def model_e(t):
    items = cached(f"n:{t}", 600, lambda: _news(t))
    src, depth = {}, {"full": 0, "summary": 0, "headline": 0}
    for it in items: src[it["s"]] = src.get(it["s"], 0) + 1; depth[it["depth"]] += 1
    checks = fact_check(items)
    base = dict(n=len(items), sources=src, depth=depth, checks=checks)
    sc = [(it["cs"], 0.5 ** (it["age"] / 24) * TIER.get(it["s"], 0.6) * it["f"]) for it in items if abs(it["cs"]) > 0.05]
    if not sc: return 0.0, dict(note="ไม่พบข่าวที่มีน้ำเสียงชัดเจน", items=[], **base)
    W = sum(w for _, w in sc); m = sum(c * w for c, w in sc) / W
    sig = float(np.clip(m * 1.6, -1, 1) * min(1.0, W / 2.5))
    top = sorted(items, key=lambda i: i["age"])[:6]
    return sig, dict(note="", items=[dict(t=i["t"][:140], s=i["s"], c=round(i["c"], 2), url=i["url"], h=round(i["age"]), depth=i["depth"],
                                          mismatch=i["mismatch"], why=i["why"]) for i in top], **base)

def earnings_in(tk, dte):
    try:
        cal = tk.calendar
        today = datetime.now(NY).date()
        for d in ((cal.get("Earnings Date") if isinstance(cal, dict) else None) or []):
            if 0 <= (d - today).days <= dte + 1: return str(d)
    except Exception:
        pass
    return None

# ---------------- Option selection ----------------
def pick_expiry(tk):
    now = datetime.now(NY)
    best = None
    for e in tk.options:
        exp = NY.localize(datetime.strptime(e, "%Y-%m-%d").replace(hour=16))
        dte = (exp - now).total_seconds() / 86400
        if 2 <= dte <= 10:
            return e, dte
        if dte >= 1 and best is None: best = (e, dte)
    return best or (None, None)

def analyze(t):
    tk = yf.Ticker(t)
    daily, intra = get_daily(t), get_intraday(t)
    if daily.empty or len(daily) < 60 or intra.empty: return None
    spot = float(intra["Close"].iloc[-1])
    exp, dte = cached(f"x:{t}", 600, lambda: pick_expiry(tk))
    if not exp: return None
    chain = cached(f"c:{t}:{exp}", 60, lambda: tk.option_chain(exp))
    T = dte / 365

    c_sig, c_info = model_c(daily, intra)
    d_sig, d_info = model_d(daily)
    e_sig, e_info = model_e(t)
    kind = "CALL" if (c_sig + d_sig + e_sig) >= 0 else "PUT"
    sgn = 1 if kind == "CALL" else -1
    earn = cached(f"e:{t}", 21600, lambda: earnings_in(tk, dte))
    cvol, pvol = chain.calls["volume"].fillna(0).sum(), chain.puts["volume"].fillna(0).sum()
    pc = float(pvol / cvol) if cvol > 0 else None
    band_iv = lambda d, lo, hi: d[d["strike"].between(lo * spot, hi * spot)]["impliedVolatility"].mean()
    skew = float((band_iv(chain.puts, .93, .97) - band_iv(chain.calls, 1.03, 1.07)) * 100)

    df = (chain.calls if kind == "CALL" else chain.puts).copy()
    for col in ("volume", "openInterest", "bid", "ask", "lastPrice"): df[col] = df[col].fillna(0)
    df = df[(df["impliedVolatility"] > 0.01) & ((df["ask"] > 0) | (df["lastPrice"] > 0))]
    df = df[df["strike"] > spot] if kind == "CALL" else df[df["strike"] < spot]
    if df.empty: return None
    atm = pd.concat([chain.calls, chain.puts]); atm = atm[atm["impliedVolatility"] > 0.01]
    atm = atm.iloc[(atm["strike"] - spot).abs().argsort()[:4]]
    atm_iv = float(atm["impliedVolatility"].mean())
    em = max(spot * atm_iv * math.sqrt(T), spot * 0.002)   # 1-sigma expected move
    target = spot + sgn * 0.5 * em

    df["delta"] = [abs(bsm(spot, r.strike, T, r.impliedVolatility, kind)["delta"]) for r in df.itertuples()]
    two = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(two, (df["bid"] + df["ask"]) / 2, np.where(df["lastPrice"] > 0, df["lastPrice"], df["ask"]))
    df["spread"] = np.where(two, (df["ask"] - df["bid"]) / df["mid"], 1.0)
    df["liq"] = np.log1p(df["volume"]) + np.log1p(df["openInterest"])
    band = df[(df["delta"] >= 0.30) & (df["delta"] <= 0.40)]
    pool = band if not band.empty else df
    good = pool[((pool["volume"] >= MIN_VOL) | (pool["openInterest"] >= MIN_OI)) & (pool["spread"] <= 0.25) & (pool["mid"] <= spot * MAX_PREMIUM_PCT)]
    liquid = not good.empty
    if liquid: pool = good
    pool = pool.assign(dist=(pool["strike"] - target).abs() / em + (pool["delta"] - TARGET_DELTA).abs() * 4 + pool["spread"] * 1.5 - 0.08 * pool["liq"])
    row = pool.sort_values("dist").iloc[0]
    K, iv, mid = float(row["strike"]), float(row["impliedVolatility"]), float(row["mid"])
    g = bsm(spot, K, T, iv, kind)
    be = K + sgn * mid

    # IV rank / percentile against 6-month rolling 30d realised-vol distribution (proxy)
    lr = np.log(daily["Close"]).diff()
    hv = (lr.rolling(30).std() * math.sqrt(252)).dropna().iloc[-126:]
    ivr = float(np.clip((atm_iv - hv.min()) / max(hv.max() - hv.min(), 1e-9), 0, 1) * 100)
    ivp = float((hv < atm_iv).mean() * 100)
    hv20 = float(lr.iloc[-20:].std() * math.sqrt(252))

    # Model A: Monte Carlo
    sigma = 0.7 * iv + 0.3 * hv20
    mu = RF + 0.2 * float(np.clip((c_sig + d_sig + e_sig) / 3, -1, 1))
    win_a, touch, prof = monte_carlo(spot, K, T, sigma, mu, kind, be)

    # Model H: historical frequency of this exact move (vol-normalised and raw), 2y of daily data
    n = max(int(round(dte * 5 / 7)), 1)
    cl = daily["Close"]
    fwd = np.log(cl.shift(-n) / cl)
    z = (fwd / (lr.rolling(30).std() * math.sqrt(n))).dropna()
    req = math.log(K / spot) / (sigma * math.sqrt(T))
    raw = fwd.dropna(); raw_req = math.log(K / spot)
    call = kind == "CALL"
    hist_z = float(((z > req) if call else (z < req)).mean() * 100) if len(z) > 40 else win_a
    hist_raw = float(((raw > raw_req) if call else (raw < raw_req)).mean() * 100) if len(raw) > 40 else win_a
    score_h = 0.7 * hist_z + 0.3 * hist_raw

    support = lambda s: float(np.clip(50 + 50 * sgn * s, 0, 100))
    decay = abs(g["theta"]) / max(mid, 0.01)
    score_b = float(np.clip(0.6 * (100 - ivr) + 0.4 * np.clip(100 - decay * 400, 0, 100) - (15 if earn else 0), 0, 100))
    scores = {"A": win_a, "B": score_b, "C": support(c_sig), "D": support(d_sig), "E": support(e_sig), "H": score_h}
    cons = sum(WEIGHTS[k] * v for k, v in scores.items())
    verdict = "ULTIMATE" if cons >= 85 else ("HIGH" if cons >= 70 else "NONE")

    return dict(
        ticker=t, price=round(spot, 2), type=kind, strike=K, expiry=exp, dte=round(dte, 1),
        premium=round(mid, 2), win_rate=round(cons, 1), verdict=verdict, scores={k: round(v, 1) for k, v in scores.items()},
        mc_itm=round(win_a, 1), mc_touch=round(touch, 1), mc_profit=round(prof, 1), hist_z=round(hist_z, 1), hist_raw=round(hist_raw, 1),
        expected_move=round(em, 2), em_low=round(spot - em, 2), em_high=round(spot + em, 2),
        greeks={k: round(v, 4) for k, v in g.items() if k != "price"}, iv=round(iv * 100, 1), atm_iv=round(atm_iv * 100, 1),
        iv_rank=round(ivr, 1), iv_pct=round(ivp, 1), iv_tag="ถูก" if ivr < 30 else ("แพง" if ivr > 70 else "กลาง"),
        volume=int(row["volume"]), oi=int(row["openInterest"]), spread_pct=round(float(row["spread"]) * 100, 1),
        premium_pct=round(mid / spot * 100, 2), liquid=liquid, pc_ratio=pc, skew=skew, earnings=earn,
        flow=c_info, tech=d_info, news=e_info, catalyst=round(50 + 50 * e_sig, 1), breakeven=round(be, 2),
    )

# ---------------- Scanner loop ----------------
STATE = {"items": [], "updated": None, "cycle_sec": 0, "errors": []}
def scan_once():
    t0, items, errs = time.time(), [], []
    def work(t):
        try:
            r = analyze(t.strip().upper())
            if r is None: errs.append(f"{t}: ข้าม — ไม่มีออปชันหมดอายุใน 2-10 วัน หรือข้อมูลไม่พอ")
            return r
        except Exception as ex:
            errs.append(f"{t}: {ex}"); log.warning("scan %s failed: %s", t, ex); return None
    with ThreadPoolExecutor(max_workers=6) as ex:
        items = [r for r in ex.map(work, WATCHLIST) if r]
    items.sort(key=lambda r: r["win_rate"], reverse=True)
    cover = {}
    for it in items:
        for k, v in it["news"].get("sources", {}).items(): cover[k] = cover.get(k, 0) + v
    for k in SRC: SRC[k]["cover"] = cover.get(k, 0)
    STATE.update(items=items, updated=datetime.now(pytz.utc).isoformat(), cycle_sec=round(time.time() - t0, 1), errors=errs[:30])

def loop():
    while True:
        s = time.time()
        try: scan_once()
        except Exception as ex: log.exception(ex)
        time.sleep(max(REFRESH_SEC - (time.time() - s), 1))

_started = False
def start():
    global _started
    if not _started:
        _started = True
        threading.Thread(target=loop, daemon=True).start()
        threading.Thread(target=probe_loop, daemon=True).start()
start()

def clean(o):
    """Make state JSON-safe: NaN/inf -> null, numpy scalars -> Python."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o); return f if math.isfinite(f) else None
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o

@app.route("/")
def index(): return render_template("index.html", refresh=REFRESH_SEC)
@app.route("/api/scan")
def api_scan(): return jsonify(clean(STATE))
@app.route("/health")
def health(): return jsonify(ok=True)

def hint(name, d):
    st = d["status"]
    if st == "ok":
        tot = d["full"] + d["summary"] + d["none"]
        return ("ดึงได้แค่ใจความ/หัวข้อ: " + (d["why"] or "เนื้อหาเต็มถูกจำกัด")) if tot and d["full"] == 0 else ""
    if st == "pending": return "รอตรวจสอบรอบแรก"
    if name in DATA_SITES: return "เป็นเว็บข้อมูล/สกรีนเนอร์ ไม่ค่อยมีหัวข้อข่าวใน RSS: ต้องใช้ API ทางการของเว็บนั้นหากต้องการข้อมูลจริง"
    if st == "error":
        return "Google จำกัด request: รอสักครู่หรือใช้ NewsAPI แทน" if d.get("http") == 429 else "เชื่อมต่อไม่ได้: ตรวจ log ของ Render และการเข้าถึงอินเทอร์เน็ตของเซิร์ฟเวอร์"
    if name in PAYWALL: return "เว็บ paywall: Google ไม่ได้ index หัวข้อล่าสุด แก้ด้วยการใส่ NEWSAPI_KEY หรือใช้ API ทางการ"
    return "ไม่พบข่าวใน 7 วัน: ลองขยายช่วงค้นหา (when:7d เป็น when:30d) ใน probe_one"

@app.route("/api/sources")
def api_sources():
    rows = [dict(name=n, domain=SOURCES[n], kind="data" if n in DATA_SITES else ("paywall" if n in PAYWALL else "news"), hint=hint(n, d), **d) for n, d in SRC.items()]
    return jsonify(clean(dict(sources=rows, fetchers=FETCH, ok=sum(r["status"] == "ok" for r in rows), total=len(rows), newsapi_key=bool(NEWSAPI_KEY))))

@app.route("/api/analyze/<t>")
def api_analyze(t):
    t = t.upper()
    if not re.fullmatch(r"[A-Z]{1,5}", t): return jsonify(error="ใส่ได้เฉพาะ ticker หุ้น US (A-Z 1-5 ตัว)"), 400
    try: res = cached(f"a:{t}", 30, lambda: analyze(t))
    except Exception as ex: return jsonify(error=f"{type(ex).__name__}: {str(ex)[:150]}"), 502
    if not res: return jsonify(error="ไม่พบออปชันรายสัปดาห์หรือข้อมูลไม่พอสำหรับหุ้นนี้"), 404
    return jsonify(clean(res))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
