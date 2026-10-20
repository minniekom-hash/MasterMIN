"""Omega v2 — real news & catalyst layer.

Replaces the Google-News-RSS hop (which rate-limits datacenter IPs and was the
reason the old engine showed "pending" forever) with:

* per-ticker RSS straight from the publisher (Yahoo, Nasdaq, Seeking Alpha)
* headline RSS from the real newsrooms (CNBC, WSJ, MarketWatch, Bloomberg, FT,
  Investing.com, Benzinga, PR Newswire, IBD, Motley Fool, ...)
* SEC EDGAR filings (8-K / 10-Q / 10-K) — primary-source catalysts
* Stocktwits crowd sentiment
* optional keyed APIs (Alpha Vantage NEWS_SENTIMENT, NewsAPI)

Every source is probed and reported to the UI with its own status, latency and
error, so the user can see exactly what the engine could actually read.
"""
from __future__ import annotations

import math
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib import robotparser
from urllib.parse import urlparse

import numpy as np

from .util import SEC_UA, age_hours, cache, http, log, now_iso

# --------------------------------------------------------------------------- #
# sentiment engine (VADER when nltk is present, else a built-in finance lexicon)
# --------------------------------------------------------------------------- #
FIN_LEX = {
    "upgrade": 2.0, "upgraded": 2.0, "upgrades": 2.0, "downgrade": -2.2, "downgraded": -2.2,
    "beat": 1.6, "beats": 1.6, "topped": 1.5, "miss": -1.6, "misses": -1.6, "missed": -1.6,
    "surge": 2.1, "surges": 2.1, "surged": 2.1, "soar": 2.3, "soars": 2.3, "soared": 2.3,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "tumble": -2.1, "tumbles": -2.1,
    "slump": -2.0, "slumps": -2.0, "selloff": -2.1, "sell-off": -2.1, "rout": -2.2,
    "bullish": 2.1, "bearish": -2.1, "rally": 1.9, "rallies": 1.9, "rebound": 1.6,
    "record": 1.3, "high": 1.0, "low": -0.9, "jump": 1.6, "jumps": 1.6, "climb": 1.3,
    "slide": -1.5, "slides": -1.5, "drop": -1.4, "drops": -1.4, "fall": -1.2, "falls": -1.2,
    "gain": 1.3, "gains": 1.3, "loss": -1.3, "losses": -1.3, "profit": 1.2, "profits": 1.2,
    "lawsuit": -1.6, "sues": -1.5, "probe": -1.5, "investigation": -1.6, "antitrust": -1.4,
    "settle": 0.6, "settlement": 0.6, "recall": -1.6, "halt": -1.4, "halted": -1.4,
    "raises": 1.3, "raised": 1.3, "cuts": -1.3, "cut": -1.2, "guidance": 0.4, "outlook": 0.3,
    "approval": 1.6, "approved": 1.6, "fda": 1.0, "launch": 1.0, "launches": 1.0,
    "partnership": 1.2, "acquisition": 1.1, "acquire": 1.0, "merger": 1.0, "buyout": 1.4,
    "takeover": 1.3, "bankruptcy": -3.0, "bankrupt": -3.0, "default": -2.0, "dilution": -1.5,
    "offering": -1.0, "buyback": 1.5, "dividend": 0.8, "split": 0.2, "insider": 0.2,
    "downgrades": -2.2, "initiates": 0.4, "overweight": 1.6, "underweight": -1.6,
    "outperform": 1.6, "underperform": -1.6, "buy": 1.2, "sell": -1.2, "hold": 0.0,
    "warning": -1.5, "warns": -1.6, "cuts outlook": -2.0, "raises outlook": 2.0,
    "beats estimates": 2.2, "misses estimates": -2.2, "all-time high": 2.0, "52-week low": -1.8,
    "short squeeze": 1.8, "bankruptcy risk": -2.5, "sec filing": 0.0, "insider selling": -1.4,
    "insider buying": 1.6, "guidance raised": 2.0, "guidance cut": -2.2,
}
NEGATORS = {"not", "no", "never", "without", "fails", "fail", "cannot", "can't", "won't", "isn't", "aren't"}
BOOSTERS = {"very": 1.3, "extremely": 1.4, "highly": 1.3, "significantly": 1.3, "sharply": 1.35, "slightly": 0.7}


class _LexiconSentiment:
    """Small VADER-like scorer used when NLTK is unavailable."""

    def __init__(self) -> None:
        self.lex = dict(FIN_LEX)

    def polarity_scores(self, text: str) -> dict:
        toks = re.findall(r"[a-zA-Z'\-]+", (text or "").lower())
        if not toks:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        total, hits = 0.0, 0
        for i, t in enumerate(toks):
            v = self.lex.get(t)
            if v is None:
                continue
            for j in range(max(0, i - 3), i):
                if toks[j] in NEGATORS:
                    v *= -0.75
                    break
            if i > 0 and toks[i - 1] in BOOSTERS:
                v *= BOOSTERS[toks[i - 1]]
            total += v
            hits += 1
        if not hits:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        x = total / math.sqrt(total * total + 15.0)
        return {"compound": float(x), "pos": max(x, 0.0), "neg": -min(x, 0.0), "neu": 1 - abs(x)}


def _make_sentiment():
    try:  # preferred: NLTK VADER with the finance lexicon bolted on
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        s = SentimentIntensityAnalyzer()
        s.lexicon.update(FIN_LEX)
        log.info("sentiment engine: NLTK VADER")
        return s, "NLTK VADER + finance lexicon"
    except Exception as ex:
        log.info("sentiment engine: built-in lexicon (%s)", type(ex).__name__)
        return _LexiconSentiment(), "built-in finance lexicon"


SIA, SENTIMENT_ENGINE = _make_sentiment()


def score_text(t: str) -> float:
    try:
        return float(SIA.polarity_scores(t or "")["compound"])
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# source registry
# --------------------------------------------------------------------------- #
# name -> (domain, tier, kind)
PUBLISHERS = {
    "CNBC": ("cnbc.com", 1.0, "news"),
    "WSJ": ("wsj.com", 1.0, "paywall"),
    "MarketWatch": ("marketwatch.com", 0.9, "news"),
    "Bloomberg": ("bloomberg.com", 1.0, "paywall"),
    "Financial Times": ("ft.com", 1.0, "paywall"),
    "Reuters": ("reuters.com", 1.0, "news"),
    "Barron's": ("barrons.com", 1.0, "paywall"),
    "Seeking Alpha": ("seekingalpha.com", 0.75, "paywall"),
    "Investing.com": ("investing.com", 0.8, "news"),
    "Benzinga": ("benzinga.com", 0.8, "news"),
    "IBD": ("investors.com", 0.85, "paywall"),
    "Motley Fool": ("fool.com", 0.7, "news"),
    "Yahoo Finance": ("finance.yahoo.com", 0.8, "news"),
    "Nasdaq": ("nasdaq.com", 0.8, "news"),
    "PR Newswire": ("prnewswire.com", 0.9, "wire"),
    "GlobeNewswire": ("globenewswire.com", 0.9, "wire"),
    "Business Wire": ("businesswire.com", 0.9, "wire"),
    "SEC EDGAR": ("sec.gov", 1.0, "filing"),
    "Stocktwits": ("stocktwits.com", 0.5, "social"),
    "Alpha Vantage": ("alphavantage.co", 0.8, "api"),
    "NewsAPI": ("newsapi.org", 0.9, "api"),
}

# (publisher, url) — market-wide headline feeds, fetched once and filtered per ticker
GENERAL_FEEDS = [
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_marketpulse"),
    ("WSJ", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("WSJ", "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml"),
    ("Bloomberg", "https://feeds.bloomberg.com/markets/news.rss"),
    ("Bloomberg", "https://feeds.bloomberg.com/technology/news.rss"),
    ("Financial Times", "https://www.ft.com/rss/home"),
    ("Investing.com", "https://www.investing.com/rss/news_25.rss"),
    ("Investing.com", "https://www.investing.com/rss/news_14.rss"),
    ("Benzinga", "https://www.benzinga.com/feed"),
    ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US"),
    ("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Markets"),
    ("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?category=Stocks"),
    ("PR Newswire", "https://www.prnewswire.com/rss/news-releases-list.rss"),
    ("IBD", "https://www.investors.com/feed/"),
    ("Motley Fool", "https://www.fool.com/feeds/index.aspx"),
    ("Barron's", "https://feeds.a.dj.com/rss/RSSBarrons.xml"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml"),
]

# per-ticker feeds
TICKER_FEEDS = [
    ("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline?s={sym}&region=US&lang=en-US"),
    ("Nasdaq", "https://www.nasdaq.com/feed/rssoutbound?symbol={sym}"),
    ("Seeking Alpha", "https://seekingalpha.com/api/sa/combined/{sym}.xml"),
]

UA = "Mozilla/5.0 (compatible; OmegaSingularityEngine/2.0)"

SRC: dict = {n: dict(status="pending", kind=k, domain=d, items=0, error="", checked=None,
                    last_ok=None, ms=None, used=0, note="") for n, (d, t, k) in PUBLISHERS.items()}
FETCH = {
    "Per-ticker RSS": dict(status="pending", items=0, error="", checked=None, last_ok=None),
    "Publisher RSS": dict(status="pending", items=0, error="", checked=None, last_ok=None),
    "SEC EDGAR filings": dict(status="pending", items=0, error="", checked=None, last_ok=None),
    "Stocktwits": dict(status="pending", items=0, error="", checked=None, last_ok=None),
    "Alpha Vantage": dict(status="pending", items=0, error="", checked=None, last_ok=None),
    "NewsAPI": dict(status="pending", items=0, error="", checked=None, last_ok=None),
}
_LOCK = threading.Lock()


def _mark_src(name: str, ok, n: int = 0, err: str = "", ms: int | None = None) -> None:
    d = SRC.get(name)
    if d is None:
        return
    with _LOCK:
        d["status"] = "ok" if ok else ("error" if ok is False else "disabled")
        d["items"] = n
        d["error"] = err
        d["checked"] = now_iso()
        if ms is not None:
            d["ms"] = ms
        if ok:
            d["last_ok"] = d["checked"]


def _mark_fetch(name: str, ok, n: int = 0, err: str = "") -> None:
    f = FETCH.get(name)
    if f is None:
        return
    f["status"] = "ok" if ok else ("error" if ok is False else "disabled")
    f["items"] = n
    f["error"] = err
    f["checked"] = now_iso()
    if ok:
        f["last_ok"] = f["checked"]


# --------------------------------------------------------------------------- #
# generic RSS / Atom reader
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    return tag.split("}")[-1].lower()


def parse_feed(text: str, limit: int = 60):
    out = []
    if not text:
        return out
    try:
        root = ET.fromstring(text.encode("utf-8", "replace") if isinstance(text, str) else text)
    except Exception:
        try:
            root = ET.fromstring(re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+);)", "&amp;", text))
        except Exception:
            return out
    nodes = [e for e in root.iter() if _local(e.tag) in ("item", "entry")]
    for e in nodes[:limit]:
        title = link = desc = date = ""
        for ch in e:
            t = _local(ch.tag)
            if t == "title":
                title = (ch.text or "").strip()
            elif t == "link":
                link = (ch.get("href") or ch.text or "").strip()
            elif t in ("description", "summary", "content", "encoded"):
                if not desc:
                    desc = re.sub(r"<[^>]+>", " ", ch.text or "").strip()
            elif t in ("pubdate", "published", "updated", "date"):
                if not date:
                    date = (ch.text or "").strip()
        if not title:
            continue
        out.append(dict(t=title, url=link, ts=date, snip=desc[:400]))
    return out


def fetch_feed(url: str, sec: bool = False, limit: int = 60):
    r = http.get(url, timeout=(5, 14), sec=sec)
    if r is None:
        return None
    try:
        return parse_feed(r.text, limit=limit)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# market-wide headlines
# --------------------------------------------------------------------------- #
def general_items():
    def load():
        out = []
        # Small pool on purpose: this runs *inside* the scan workers, so a wide
        # pool here multiplies with SCAN_WORKERS into dozens of live threads on
        # a 512MB free instance.  The HTTP layer already allows 10 concurrent
        # requests, so a modest thread count still saturates the network.
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(fetch_feed, u): (name, u) for name, u in GENERAL_FEEDS}
            for f in futs:
                name, url = futs[f]
                try:
                    items = f.result()
                except Exception:
                    items = None
                if items is None:
                    _mark_src(name, False, 0, "ดึงฟีดไม่ได้")
                    continue
                _mark_src(name, True, len(items))
                for it in items:
                    it["s"] = name
                    it["k"] = "publisher"
                    it["age"] = age_hours(it.get("ts"))
                    out.append(it)
        _mark_fetch("Publisher RSS", True, len(out))
        return out

    items = cache.wrap("news:general", 600, load)
    return items or []


# --------------------------------------------------------------------------- #
# SEC EDGAR
# --------------------------------------------------------------------------- #
SEC_ITEMS = {
    "1.01": "เข้าทำสัญญาสำคัญ", "1.02": "ยกเลิกสัญญาสำคัญ", "2.01": "ซื้อ/ขายสินทรัพย์",
    "2.02": "ผลประกอบการ", "2.03": "หนี้สินใหม่", "2.04": "ผิดเงื่อนไขหนี้",
    "2.05": "ค่าใช้จ่ายปิดโรงงาน", "2.06": "ตัดจำหน่ายสินทรัพย์", "3.01": "ถูกเพิกถอนจดทะเบียน",
    "3.02": "ขายหุ้นนอกตลาด", "4.01": "เปลี่ยนผู้สอบบัญชี", "5.01": "เปลี่ยนการควบคุม",
    "5.02": "เปลี่ยนผู้บริหาร/กรรมการ", "7.01": "เปิดเผยข้อมูลตามระเบียบ", "8.01": "เหตุการณ์อื่น",
    "9.01": "เอกสารแนบ",
}
SEC_FORMS = {"8-K": 1.0, "10-Q": 0.9, "10-K": 0.9, "SC 13D": 0.9, "SC 13G": 0.7, "S-1": 0.8,
             "424B": 0.7, "4": 0.5, "3": 0.4, "5": 0.4, "DEF 14A": 0.5, "13F-HR": 0.3}


def _cik_map():
    def load():
        j = http.json("https://www.sec.gov/files/company_tickers.json", sec=True, timeout=(6, 25))
        if not j:
            return None
        return {str(v.get("ticker", "")).upper(): int(v.get("cik_str", 0)) for v in j.values()}

    return cache.wrap("sec:cikmap", 86400, load)


def sec_filings(sym: str, days: int = 45, limit: int = 8):
    def load():
        cm = _cik_map()
        if not cm or sym.upper() not in cm:
            _mark_fetch("SEC EDGAR filings", None, 0, "ไม่พบ CIK ของหุ้นนี้")
            _mark_src("SEC EDGAR", None, 0, "ไม่พบ CIK")
            return []
        cik = cm[sym.upper()]
        j = http.json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", sec=True, timeout=(6, 25))
        if not j:
            _mark_fetch("SEC EDGAR filings", False, 0, "ดึงข้อมูลไม่ได้")
            _mark_src("SEC EDGAR", False, 0, "ดึงข้อมูลไม่ได้")
            return []
        rec = (j.get("filings") or {}).get("recent") or {}
        forms = rec.get("form") or []
        dates = rec.get("filingDate") or []
        accs = rec.get("accessionNumber") or []
        docs = rec.get("primaryDocument") or []
        descs = rec.get("primaryDocDescription") or []
        items_arr = rec.get("items") or [""] * len(forms)
        out = []
        for i, f in enumerate(forms):
            if f not in SEC_FORMS:
                continue
            if i >= len(dates) or age_hours(dates[i]) > days * 24:
                continue
            adsh = (accs[i] if i < len(accs) else "").replace("-", "")
            doc = docs[i] if i < len(docs) else ""
            label = SEC_ITEMS.get((items_arr[i] or "").split(",")[0].strip(), "")
            out.append(dict(
                form=f, date=dates[i],
                title=f"{f} — {descs[i] if i < len(descs) and descs[i] else label or 'การยื่นเอกสารต่อ ก.ล.ต. สหรัฐ'}",
                url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/{doc}" if adsh and doc else
                    f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={f}",
                items=[SEC_ITEMS.get(x.strip(), x.strip()) for x in (items_arr[i] or "").split(",") if x.strip()],
                weight=SEC_FORMS.get(f, 0.5),
            ))
            if len(out) >= limit:
                break
        _mark_fetch("SEC EDGAR filings", True, len(out))
        _mark_src("SEC EDGAR", True, len(out))
        return out

    return cache.wrap(f"sec:{sym}", 3600, load) or []


# --------------------------------------------------------------------------- #
# per-ticker fetchers
# --------------------------------------------------------------------------- #
def ticker_feeds(sym: str):
    out = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_feed, tpl.format(sym=sym)): name for name, tpl in TICKER_FEEDS}
        for f in futs:
            name = futs[f]
            try:
                items = f.result()
            except Exception:
                items = None
            if items is None:
                _mark_src(name, False, 0, "ดึงฟีดรายหุ้นไม่ได้")
                continue
            _mark_src(name, True, len(items))
            for it in items:
                it.update(s=name, k="ticker", age=age_hours(it.get("ts")))
                out.append(it)
    _mark_fetch("Per-ticker RSS", True, len(out))
    return out


def stocktwits(sym: str):
    def load():
        j = http.json(f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json",
                      headers={"Accept": "application/json"}, timeout=(5, 12))
        if not j or "messages" not in j:
            _mark_fetch("Stocktwits", False, 0, "ดึงไม่ได้/ถูกจำกัด")
            _mark_src("Stocktwits", False, 0, "ดึงไม่ได้")
            return dict(bull=0, bear=0, n=0, score=0.0, items=[])
        bull = bear = 0
        scores, items = [], []
        for m in j.get("messages", []):
            body = m.get("body") or ""
            sent = (((m.get("entities") or {}).get("sentiment")) or {}).get("basic")
            if sent == "Bullish":
                bull += 1
            elif sent == "Bearish":
                bear += 1
            c = score_text(body)
            if abs(c) > 0.05:
                scores.append(c)
            items.append(dict(t=body[:160], url=f"https://stocktwits.com/symbol/{sym}",
                              ts=m.get("created_at"), c=round(c, 2), tag=sent or "", s="Stocktwits"))
        tagged = bull + bear
        crowd = (bull - bear) / tagged if tagged else (float(np.mean(scores)) if scores else 0.0)
        _mark_fetch("Stocktwits", True, len(items))
        _mark_src("Stocktwits", True, len(items))
        return dict(bull=bull, bear=bear, n=len(items), score=round(float(crowd), 3), items=items[:8])

    return cache.wrap(f"st:{sym}", 300, load) or dict(bull=0, bear=0, n=0, score=0.0, items=[])


def alphavantage(sym: str):
    from .util import env_list  # noqa
    import os

    key = os.environ.get("ALPHAVANTAGE_KEY", "").strip() or "demo"

    def load():
        url = (f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={sym}"
               f"&limit=50&sort=LATEST&apikey={key}")
        j = http.json(url, timeout=(6, 20))
        if not j or "feed" not in j:
            note = ""
            if isinstance(j, dict):
                note = str(j.get("Note") or j.get("Information") or j.get("Error Message") or "")[:90]
            _mark_fetch("Alpha Vantage", False, 0, note or "ไม่มีข้อมูล (โควต้า/คีย์)")
            _mark_src("Alpha Vantage", False, 0, note or "โควต้าเต็ม/ต้องใช้คีย์")
            return None
        out = []
        for a in j.get("feed", []):
            rel, sc = 0.0, float(a.get("overall_sentiment_score") or 0.0)
            for ts_ in a.get("ticker_sentiment", []) or []:
                if str(ts_.get("ticker", "")).upper() == sym.upper():
                    try:
                        if float(ts_.get("relevance_score") or 0) > rel:
                            rel = float(ts_["relevance_score"])
                            sc = float(ts_.get("ticker_sentiment_score") or sc)
                    except Exception:
                        pass
            out.append(dict(t=a.get("title") or "", url=a.get("url") or "",
                            ts=a.get("time_published"), snip=(a.get("summary") or "")[:400],
                            s=a.get("source") or "Alpha Vantage", c=round(sc, 3), rel=round(rel, 2),
                            label=a.get("overall_sentiment_label") or "", k="api"))
        _mark_fetch("Alpha Vantage", True, len(out))
        _mark_src("Alpha Vantage", True, len(out))
        return out

    return cache.wrap(f"av:{sym}", 1800, load)


def newsapi(sym: str, name: str = ""):
    import os

    key = os.environ.get("NEWSAPI_KEY", "").strip()
    if not key:
        _mark_fetch("NewsAPI", None, 0, "ยังไม่ได้ตั้งค่า NEWSAPI_KEY (ไม่บังคับ)")
        _mark_src("NewsAPI", None, 0, "ยังไม่ได้ตั้งค่า NEWSAPI_KEY")
        return []

    def load():
        q = f'"{sym}" OR "{name}" stock' if name else f'"{sym}" stock'
        j = http.json("https://newsapi.org/v2/everything",
                      headers={"X-Api-Key": key},
                      timeout=(6, 20))
        if not j or j.get("status") != "ok":
            err = (j or {}).get("message", "ดึงไม่ได้")
            _mark_fetch("NewsAPI", False, 0, str(err)[:90])
            _mark_src("NewsAPI", False, 0, str(err)[:90])
            return []
        out = []
        for a in j.get("articles", []):
            out.append(dict(t=a.get("title") or "", url=a.get("url") or "", ts=a.get("publishedAt"),
                            snip=(a.get("description") or "")[:400],
                            s=(a.get("source") or {}).get("name") or "NewsAPI", k="api"))
        _mark_fetch("NewsAPI", True, len(out))
        _mark_src("NewsAPI", True, len(out))
        return out

    return cache.wrap(f"na:{sym}", 1800, load)


# --------------------------------------------------------------------------- #
# relevance filtering for market-wide headlines
# --------------------------------------------------------------------------- #
_STOP = {"inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.", "ltd", "limited", "plc",
         "holdings", "holding", "group", "class", "common", "stock", "the", "and", "of", "trust", "fund",
         "technologies", "technology", "international", "global", "systems", "solutions", "industries",
         "american", "depositary", "shares", "new", "capital", "partners", "energy", "financial"}


def _name_tokens(name: str):
    toks = re.findall(r"[A-Za-z]{4,}", (name or "").lower())
    return [t for t in toks if t not in _STOP][:2]


def relevant(it: dict, sym: str, name: str) -> bool:
    text = f"{it.get('t', '')} {it.get('snip', '')}"
    low = text.lower()
    if f"${sym.lower()}" in low or f"({sym.lower()})" in low:
        return True
    for tok in _name_tokens(name):
        if tok in low:
            return True
    if len(sym) >= 3 and re.search(rf"\b{re.escape(sym)}\b", text):
        return True
    return False


# --------------------------------------------------------------------------- #
# optional: read the article body (respects robots.txt, never bypasses paywalls)
# --------------------------------------------------------------------------- #
PAYWALL_RE = re.compile(
    r"subscribe to (continue|read)|sign in to (read|continue)|already a subscriber|"
    r"create a free account to continue|premium content|enable javascript", re.I)
_ROBOTS: dict = {}


def _allowed(url: str) -> bool:
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    rp = _ROBOTS.get(base)
    if rp is None:
        rp = robotparser.RobotFileParser()
        try:
            r = http.get(base + "/robots.txt", timeout=(4, 8))
            rp.parse((r.text or "").splitlines() if r else [])
        except Exception:
            rp.parse([])
        _ROBOTS[base] = rp
    try:
        return rp.can_fetch(UA, url)
    except Exception:
        return False


def read_article(url: str):
    if not url or not url.startswith("http"):
        return "none", "", "ไม่มีลิงก์"
    if not _allowed(url):
        return "none", "", "robots.txt ของเว็บไม่อนุญาตให้อ่าน"
    r = http.get(url, timeout=(5, 12))
    if r is None:
        return "none", "", "ดึงหน้าเว็บไม่ได้"
    if r.status_code in (401, 402, 403, 429):
        return "none", "", f"HTTP {r.status_code}: เว็บบล็อก/ต้องสมัครสมาชิก"
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text[:400000], "html.parser")
        meta = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        desc = (meta.get("content") if meta else "") or ""
        body = " ".join(x for x in (p.get_text(" ", strip=True) for p in (soup.find("article") or soup).find_all("p")) if len(x) > 60)
        if len(body) >= 600 and not PAYWALL_RE.search(body[:3000]):
            return "full", body[:6000], ""
        if desc or body:
            return "summary", (desc or body)[:800], "เนื้อหาเต็มถูกจำกัด (paywall/ล็อกอิน)"
        return "none", "", "ไม่พบเนื้อหาในหน้าเว็บ"
    except Exception as ex:
        return "none", "", f"{type(ex).__name__}: {str(ex)[:70]}"


# --------------------------------------------------------------------------- #
# cross-source verification (same idea as the original engine, tightened)
# --------------------------------------------------------------------------- #
EVENTS = {
    "ผลประกอบการ/แนวโน้ม": r"earnings|revenue|\beps\b|quarter|profit|guidance|outlook",
    "นักวิเคราะห์ปรับเรตติ้ง": r"upgrade|downgrade|price target|rating|initiates|overweight|underweight",
    "คดีความ/การสอบสวน": r"lawsuit|sues|probe|investigation|antitrust|settle|\bsec\b",
    "ควบรวม/ดีล": r"acquire|acquisition|merger|buyout|takeover|stake|partnership",
    "ผลิตภัณฑ์/อนุมัติ": r"launch|unveil|recall|\bfda\b|approval|trial",
    "ปันผล/ซื้อหุ้นคืน/ระดมทุน": r"buyback|dividend|offering|stock split|debt",
    "ผู้บริหาร/ปลดพนักงาน": r"\bceo\b|\bcfo\b|resign|layoff|job cuts",
    "มหภาค/ภาษี/ดอกเบี้ย": r"tariff|rate cut|inflation|export (ban|curb|control)",
}
FIG_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?\s?(?:billion|million|trillion|[BMT])?|\d+(?:\.\d+)?\s?%")


def fact_check(items):
    groups = {}
    for it in items:
        txt = f"{it.get('t','')} {it.get('body','')[:600]}".lower()
        for ev, pat in EVENTS.items():
            if re.search(pat, txt):
                groups.setdefault(ev, []).append(it)
    out = []
    for ev, g in groups.items():
        srcs = sorted({i.get("s", "?") for i in g})
        sg = [np.sign(i.get("c", 0.0)) for i in g if abs(i.get("c", 0.0)) > 0.1]
        pos, neg = sg.count(1), sg.count(-1)
        w = sum(PUBLISHERS.get(x, ("", 0.6, ""))[1] for x in srcs)
        split = bool(pos and neg and min(pos, neg) / (pos + neg) >= 0.2)
        verdict = "ขัดแย้ง" if split else ("ยืนยันแล้ว" if len(srcs) >= 3 and w >= 2.2 else
                                          ("ยืนยันบางส่วน" if len(srcs) >= 2 else "ยังไม่ยืนยัน"))
        fac = {"ยืนยันแล้ว": 1.0, "ยืนยันบางส่วน": 0.8, "ยังไม่ยืนยัน": 0.5, "ขัดแย้ง": 0.3}[verdict]
        for i in g:
            i["f"] = max(i.get("f", 0.0), fac)
        figures = {}
        for i in g:
            for f in {x.lower().replace(" ", "") for x in FIG_RE.findall(f"{i.get('t','')} {i.get('body','')[:600]}")}:
                figures.setdefault(f, set()).add(i.get("s", "?"))
        out.append(dict(event=ev, n=len(g), sources=srcs, verdict=verdict, split=split,
                        pos=pos, neg=neg,
                        figures=[f for f, ss in sorted(figures.items(), key=lambda kv: -len(kv[1])) if len(ss) >= 2][:3]))
    for i in items:
        i.setdefault("f", 0.6)
    return sorted(out, key=lambda x: -x["n"])[:6]


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
ENRICH_N = 2


def for_ticker(sym: str, name: str = "") -> dict:
    def load():
        sym_u = sym.upper()
        items = ticker_feeds(sym_u)
        for it in general_items():
            if relevant(it, sym_u, name):
                it["k"] = "publisher"
                items.append(dict(it))
        av = alphavantage(sym_u) or []
        items += av
        items += newsapi(sym_u, name)
        st = stocktwits(sym_u)
        filings = sec_filings(sym_u)

        seen, dedup = set(), []
        for it in items:
            key = re.sub(r"\W+", "", (it.get("t") or "").lower())[:60]
            if it.get("t") and key and key not in seen:
                seen.add(key)
                it.setdefault("snip", "")
                it.setdefault("age", age_hours(it.get("ts")))
                it.setdefault("c", None)
                dedup.append(it)

        # article bodies for the most recent few only
        head = sorted(dedup, key=lambda i: i["age"])[:max(ENRICH_N, 0)]
        for it in head:
            d, txt, why = cache.wrap("art:" + it.get("url", ""), 3600, lambda it=it: read_article(it.get("url", "")))
            it.update(depth=d, body=txt, why=why)
        for it in dedup:
            it.setdefault("depth", "summary" if it.get("snip") else "headline")
            it.setdefault("body", it.get("snip", ""))
            it.setdefault("why", "")

        for it in dedup:
            hc = score_text(it.get("t", ""))
            body = it.get("body", "")
            bc = hc
            if body:
                sents = [x for x in re.split(r"(?<=[.!?])\s+", body[:4000]) if len(x) > 30]
                rel = [x for x in sents if re.search(rf"\b{re.escape(sym_u)}\b|company|shares|stock", x, re.I)] or sents
                cs = [score_text(x) for x in rel[:20]]
                if cs:
                    bc = float(np.mean(cs))
            api_c = it.get("c")
            if api_c is not None:
                it["c"] = float(np.clip(0.5 * hc + 0.5 * float(api_c), -1, 1))
            else:
                it["c"] = float(0.4 * hc + 0.6 * bc) if body else float(hc)
            if it["c"] is None:
                it["c"] = 0.0
            it["mismatch"] = bool(body and abs(hc) > 0.25 and abs(bc) > 0.15 and np.sign(hc) != np.sign(bc))
            it["c"] = round(it["c"], 3)

        checks = fact_check(dedup)
        depth = {"full": 0, "summary": 0, "headline": 0}
        for it in dedup:
            depth[it.get("depth", "headline")] = depth.get(it.get("depth", "headline"), 0) + 1
        sources = {}
        for it in dedup:
            sources[it.get("s", "?")] = sources.get(it.get("s", "?"), 0) + 1
        for n in sources:
            if n in SRC:
                SRC[n]["used"] = sources[n]

        sc = []
        for it in dedup:
            c = it.get("c")
            if c is not None and abs(c) > 0.05:
                tier = PUBLISHERS.get(it.get("s", ""), ("", 0.6, ""))[1]
                rec = "ticker" if it.get("k") == "ticker" else ("api" if it.get("k") == "api" else "publisher")
                rec_w = {"ticker": 1.25, "api": 1.05, "publisher": 0.9}[rec]
                sc.append((c, (0.5 ** (it["age"] / 24.0)) * tier * it.get("f", 0.6) * rec_w))
        if sc:
            W = sum(w for _, w in sc)
            m = sum(c * w for c, w in sc) / W
            sig = float(np.clip(m * 1.7, -1, 1) * min(1.0, W / 2.5))
            note = ""
        else:
            sig, note = 0.0, "ไม่พบข่าวที่มีน้ำเสียงชัดเจน"

        # primary-source filings get their own weight; 8-K/10-Q move the needle
        fscore = 0.0
        if filings:
            w = sum(f["weight"] for f in filings)
            newest = min(age_hours(f["date"]) for f in filings)
            recency = 0.5 ** (newest / 72.0)
            fscore = float(np.clip(0.55 * recency + 0.25 * min(w / 3, 1) + 0.2, 0, 1))
        social = float(st.get("score") or 0.0)

        return dict(
            n=len(dedup), sources=sources, depth=depth, checks=checks, score=round(sig, 3),
            note=note, filings=filings, filing_score=round(fscore, 3), social=st,
            engine=SENTIMENT_ENGINE,
            items=[dict(t=(i.get("t") or "")[:150], s=i.get("s"), c=i.get("c"), url=i.get("url"),
                        h=round(i.get("age", 24)), depth=i.get("depth"), mismatch=i.get("mismatch"),
                        why=i.get("why", ""), k=i.get("k", "publisher")) for i in
                   sorted(dedup, key=lambda i: i.get("age", 99))[:14]],
        )

    return cache.wrap(f"news:{sym}", 600, load) or dict(n=0, sources={}, depth={}, checks=[], score=0.0,
                                                       note="", filings=[], filing_score=0.0,
                                                       social={}, items=[], engine=SENTIMENT_ENGINE)


def probe_all() -> dict:
    """Health-check every publisher feed (background loop + /api/sources)."""
    fs = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, url in GENERAL_FEEDS:
            fs.append((name, ex.submit(fetch_feed, url, False, 5)))
        for name, url in fs:
            try:
                items = url.result()
            except Exception:
                items = None
            if items is None:
                _mark_src(name, False, 0, "ดึงฟีดไม่ได้/ถูกบล็อกจาก IP เซิร์ฟเวอร์")
            else:
                _mark_src(name, True, len(items))
    for name, url in TICKER_FEEDS:
        items = fetch_feed(url.format(sym="SPY"), False, 5)
        _mark_src(name, items is not None, len(items or []),
                  "" if items is not None else "ดึงฟีดรายหุ้นไม่ได้")
    _mark_src("SEC EDGAR", True, 0, "ใช้งานได้ (ตรวจตอนวิเคราะห์รายหุ้น)")
    _mark_fetch("Publisher RSS", True, 0)
    return source_status()


def source_status() -> dict:
    rows = []
    for n, d in SRC.items():
        dom, tier, _kind = PUBLISHERS.get(n, ("", 0.6, "news"))
        row = dict(d)
        row.update(name=n, domain=d.get("domain") or dom, tier=tier,
                   checked_h=round(age_hours(d.get("checked")), 1) if d.get("checked") else None,
                   last_ok_h=round(age_hours(d.get("last_ok")), 1) if d.get("last_ok") else None)
        rows.append(row)
    rows.sort(key=lambda r: (r.get("status") != "ok", -float(r.get("tier") or 0)))
    ok = sum(1 for r in rows if r["status"] == "ok")
    return dict(sources=rows, fetchers=FETCH, ok=ok, total=len(rows),
                engine=SENTIMENT_ENGINE, checked=now_iso())
