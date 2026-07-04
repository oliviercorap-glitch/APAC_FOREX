#!/usr/bin/env python3
"""
apac_fx_risk_agent.py
======================
Automated intelligence agent: regional FX exchange rate and hedging risk watch.

Business context
-----------------
- Entity      : TLD Group (Alvest subsidiary) - APAC Finance Department (CFO)
- Scope       : CNY, JPY, KRW, THB, PHP, AUD vs EUR and USD
- Objective   : daily monitoring of exchange rate moves against alert
                thresholds, correlation with PBOC / regional central bank
                decisions, and implications for export margins and FX
                hedging strategy.

Architecture (aligned with the GSE / China Tax & Law Watch agent suite)
-------------------------------------------------------------------------
1. Deterministic FX data layer: real reference rates from the ECB via the
   Frankfurter API (free, no API key), daily & weekly % moves computed in
   Python (never left to the LLM, to avoid numeric hallucination) and
   classified against configurable alert thresholds.
2. Static source scraping (BeautifulSoup + exponential retry/backoff) for
   central bank / FX news sources.
3. Complementary Tavily search (bypasses IP-blocking of central bank .gov
   sites from GitHub Actions US runners, covers non-scrapable / JS-heavy
   financial news sites).
4. Weekly structural brief via DeepSeek (training-knowledge only, cached for
   7 days to avoid wasting API calls) covering regional FX regimes and
   central bank mandates.
5. Article enrichment (page body extraction).
6. Keyword filtering (strict \\b word-boundary matching for any ASCII
   acronym of 4 uppercase letters or fewer -- currency codes and central
   bank acronyms are exactly this shape -- to avoid false positives).
7. DeepSeek analysis of qualitative news, with delimiter-based structured
   output + truncation detection, grounded in the deterministic FX figures
   computed in step 1 (DeepSeek explains correlation, it does not invent
   the percentages).
8. HTML report: FX dashboard table + signals with impact levels
   (CRITICAL/IMPORTANT/WATCH/INFO), executive summary, "top risk to watch"
   section, clickable sources.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import markdown as markdown_lib
import requests
from bs4 import BeautifulSoup

try:
    from tavily import TavilyClient
except ImportError:  # pragma: no cover - safeguard if the package isn't installed
    TavilyClient = None


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ORG_NAME = "TLD Group"
ORG_CONTEXT = "APAC Finance Department (CFO) - export margin exposure & FX hedging"
REGION_LABEL = "Asia-Pacific (China, Japan, Korea, Thailand, Philippines, Australia) vs EUR/USD"
REPORT_LANG = "en"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

WEEKLY_BRIEF_CACHE_FILE = DATA_DIR / "weekly_brief_cache.json"
SEEN_URLS_FILE = DATA_DIR / "seen_urls.json"
WEEKLY_BRIEF_MAX_AGE_DAYS = 7
SEEN_URLS_RETENTION_DAYS = 30

TEST_MODE = "--test" in os.sys.argv or os.environ.get("TEST_MODE") == "1"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,fr;q=0.8,zh-CN;q=0.7,ja;q=0.6",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("apac_fx_risk_agent")


# ---------------------------------------------------------------------------
# FX DATA LAYER (deterministic, no LLM involved in the numbers)
# ---------------------------------------------------------------------------
# Frankfurter (https://www.frankfurter.app) republishes ECB reference rates.
# Free, no API key required. Base currency EUR; USD cross-rates are derived.

FRANKFURTER_BASE_URL = "https://api.frankfurter.app"
FX_CURRENCIES = ["CNY", "JPY", "KRW", "THB", "PHP", "AUD"]
FX_SYMBOLS = FX_CURRENCIES + ["USD"]

# Alert thresholds in absolute percentage move. These are deliberately
# currency-specific: CNY trades in a managed float with typically low daily
# volatility, while JPY/KRW/AUD are free-floating and structurally more
# volatile. Tune these constants as needed.
FX_THRESHOLDS = {
    "CNY": {"daily": 0.3, "weekly": 1.0},
    "JPY": {"daily": 1.0, "weekly": 2.5},
    "KRW": {"daily": 1.0, "weekly": 2.5},
    "THB": {"daily": 0.8, "weekly": 2.0},
    "PHP": {"daily": 0.6, "weekly": 1.5},
    "AUD": {"daily": 1.0, "weekly": 2.5},
}


def classify_move(pct_move: Optional[float], threshold: float) -> Optional[str]:
    """Classify an absolute percentage move against a base threshold."""
    if pct_move is None or threshold <= 0:
        return None
    abs_move = abs(pct_move)
    if abs_move >= 2 * threshold:
        return "CRITICAL"
    if abs_move >= threshold:
        return "IMPORTANT"
    if abs_move >= 0.5 * threshold:
        return "WATCH"
    return None


IMPACT_RANK = {"CRITICAL": 3, "IMPORTANT": 2, "WATCH": 1, "INFO": 0, None: -1}


def worse_impact(a: Optional[str], b: Optional[str]) -> Optional[str]:
    return a if IMPACT_RANK.get(a, -1) >= IMPACT_RANK.get(b, -1) else b


# ---------------------------------------------------------------------------
# SOURCES (qualitative news / central bank communications)
# ---------------------------------------------------------------------------
# Central bank websites are frequently anti-bot / IP-blocked from US-based
# GitHub Actions runners (same issue previously solved via Tavily on the GSE
# aviation agent), and most financial news outlets (Reuters, Bloomberg, WSJ)
# are JS-heavy or paywalled and are intentionally NOT scraped directly here.
# Tavily is the primary channel for those; static scraping covers the more
# stable, less protected sources.

STATIC_SOURCES = [
    {
        "name": "Reserve Bank of Australia - Media Releases",
        "url": "https://www.rba.gov.au/media-releases/",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "Bank of Japan - Announcements",
        "url": "https://www.boj.or.jp/en/announcements/index.htm",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "Bank of Thailand - News",
        "url": "https://www.bot.or.th/en/news-and-media/news.html",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "Bank of Korea - Press Releases",
        "url": "https://www.bok.or.kr/eng/bbs/E0000634/list.do",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "People's Bank of China - English News",
        "url": "http://www.pbc.gov.cn/en/3688006/index.html",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "Bangko Sentral ng Pilipinas - Media Releases",
        "url": "https://www.bsp.gov.ph/SitePages/MediaAndResearch/MediaDisp.aspx",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "FXStreet - News",
        "url": "https://www.fxstreet.com/news",
        "max_items": 15,
        "min_text_len": 20,
    },
    {
        "name": "China Briefing (Dezan Shira) - Finance",
        "url": "https://www.china-briefing.com/news/category/finance/",
        "max_items": 15,
        "min_text_len": 20,
    },
]

TAVILY_QUERIES = [
    "PBOC yuan reference rate decision",
    "Bank of Japan yen intervention monetary policy",
    "Bank of Korea won interest rate decision",
    "Bank of Thailand baht monetary policy",
    "Bangko Sentral Philippines peso policy rate",
    "Reserve Bank of Australia AUD interest rate decision",
    "China capital controls foreign exchange yuan",
    "Asia currency depreciation export competitiveness",
    "dollar index DXY Asia FX impact",
    "China yuan corporate hedging forward points",
]

ENTITIES_TO_WATCH = [
    "People's Bank of China (PBOC) / 中国人民银行",
    "Bank of Japan (BOJ) / 日本銀行",
    "Bank of Korea (BOK)",
    "Bank of Thailand (BOT)",
    "Bangko Sentral ng Pilipinas (BSP)",
    "Reserve Bank of Australia (RBA)",
    "US Federal Reserve (Fed / FOMC) - as a driver of USD strength",
    "European Central Bank (ECB) - as a driver of EUR strength",
]

# ---------------------------------------------------------------------------
# KEYWORDS
# ---------------------------------------------------------------------------
# Anti-false-positive rule: any ASCII acronym of 4 uppercase letters or fewer
# (currency codes CNY/JPY/KRW/THB/PHP/AUD/USD/EUR, central bank acronyms
# PBOC/BOJ/BOK/BOT/BSP/RBA/FED/ECB/FOMC(5, treated as phrase), DXY, CNH) is
# matched with a strict word boundary \b...\b to avoid e.g. "AUD" matching
# inside an unrelated word, or "BOT" matching inside "robot"/"bottle".

KEYWORDS = [
    # Currencies
    "CNY", "CNH", "JPY", "KRW", "THB", "PHP", "AUD", "USD", "EUR", "RMB",
    "yuan", "renminbi", "yen", "won", "baht", "peso", "Australian dollar",
    # Central banks / policy institutions
    "PBOC", "BOJ", "BOK", "BOT", "BSP", "RBA", "FOMC", "Federal Reserve",
    "European Central Bank", "ECB", "DXY", "dollar index",
    # FX mechanics / policy actions
    "exchange rate", "reference rate", "midpoint rate", "currency intervention",
    "capital controls", "managed float", "devaluation", "depreciation",
    "appreciation", "interest rate decision", "policy rate", "rate hike",
    "rate cut", "quantitative easing", "foreign exchange reserves",
    "cross-border capital flow", "carry trade",
    # Corporate FX risk management
    "FX hedging", "forward contract", "forward points", "currency swap",
    "hedging strategy", "export margin", "currency risk", "translation risk",
    "transaction exposure",
    # Chinese
    "人民币汇率", "汇率中间价", "中国人民银行", "贬值", "升值",
    "外汇储备", "跨境资金流动", "购汇", "结汇", "稳汇率",
    # Japanese
    "日本銀行", "為替介入", "円安", "円高",
]

ACRONYM_PATTERN = re.compile(r"^[A-Z]{1,4}$")


def _build_keyword_patterns():
    patterns = []
    for kw in KEYWORDS:
        if ACRONYM_PATTERN.match(kw):
            patterns.append(re.compile(r"\b" + re.escape(kw) + r"\b"))
        else:
            patterns.append(re.compile(re.escape(kw), re.IGNORECASE))
    return patterns


KEYWORD_PATTERNS = _build_keyword_patterns()


def keyword_match(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in KEYWORD_PATTERNS)


# ---------------------------------------------------------------------------
# RETRY / BACKOFF
# ---------------------------------------------------------------------------

def retry_with_backoff(max_retries=3, base_delay=1.5, max_delay=20.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, ValueError) as exc:
                    last_exc = exc
                    logger.warning(
                        "%s failed (attempt %d/%d): %s",
                        func.__name__, attempt, max_retries, exc,
                    )
                    if attempt < max_retries:
                        sleep_time = min(delay, max_delay) + random.uniform(0, 0.75)
                        time.sleep(sleep_time)
                        delay *= 2
            logger.error("%s gave up after %d attempts: %s", func.__name__, max_retries, last_exc)
            return None
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=1.5)
def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


@retry_with_backoff(max_retries=3, base_delay=1.5)
def fetch_json(url: str, timeout: int = 15) -> Optional[dict]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


@retry_with_backoff(max_retries=3, base_delay=2.0)
def post_json(url: str, headers: dict, payload: dict, timeout: int = 90) -> Optional[dict]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# FX SNAPSHOT (rates + daily/weekly % moves + threshold classification)
# ---------------------------------------------------------------------------

def get_rates_for_date(date_token: str) -> Optional[dict]:
    """date_token is 'latest' or an ISO date string 'YYYY-MM-DD'.
    Frankfurter automatically falls back to the closest earlier available
    date if the requested date has no published rate (weekends/holidays)."""
    symbols = ",".join(FX_SYMBOLS)
    url = f"{FRANKFURTER_BASE_URL}/{date_token}?from=EUR&to={symbols}"
    return fetch_json(url)


def _cross_rates_vs_usd(eur_based_rates: dict) -> dict:
    """Given {'CNY': x, 'USD': y, ...} amounts of currency per 1 EUR,
    derive amount of each currency per 1 USD."""
    usd_per_eur = eur_based_rates.get("USD")
    if not usd_per_eur:
        return {}
    return {ccy: (rate / usd_per_eur) for ccy, rate in eur_based_rates.items() if ccy != "USD"}


def pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def build_fx_snapshot() -> dict:
    today = datetime.now(timezone.utc).date()
    latest_resp = get_rates_for_date("latest")
    d1_resp = get_rates_for_date((today - timedelta(days=1)).isoformat())
    d7_resp = get_rates_for_date((today - timedelta(days=7)).isoformat())

    if not latest_resp:
        logger.error("Could not fetch latest FX rates from Frankfurter API")
        return {"available": False, "rows": [], "as_of": None}

    latest_eur = latest_resp.get("rates", {})
    d1_eur = (d1_resp or {}).get("rates", {})
    d7_eur = (d7_resp or {}).get("rates", {})

    latest_usd = _cross_rates_vs_usd(latest_eur)
    d1_usd = _cross_rates_vs_usd(d1_eur) if d1_eur else {}
    d7_usd = _cross_rates_vs_usd(d7_eur) if d7_eur else {}

    rows = []
    for ccy in FX_CURRENCIES:
        th = FX_THRESHOLDS.get(ccy, {"daily": 1.0, "weekly": 2.5})

        eur_now, eur_d1, eur_d7 = latest_eur.get(ccy), d1_eur.get(ccy), d7_eur.get(ccy)
        usd_now, usd_d1, usd_d7 = latest_usd.get(ccy), d1_usd.get(ccy), d7_usd.get(ccy)

        eur_daily_pct = pct_change(eur_now, eur_d1)
        eur_weekly_pct = pct_change(eur_now, eur_d7)
        usd_daily_pct = pct_change(usd_now, usd_d1)
        usd_weekly_pct = pct_change(usd_now, usd_d7)

        impact = None
        for pct, horizon in ((eur_daily_pct, "daily"), (eur_weekly_pct, "weekly"),
                              (usd_daily_pct, "daily"), (usd_weekly_pct, "weekly")):
            impact = worse_impact(impact, classify_move(pct, th[horizon]))

        rows.append({
            "currency": ccy,
            "eur_rate": eur_now,
            "eur_daily_pct": eur_daily_pct,
            "eur_weekly_pct": eur_weekly_pct,
            "usd_rate": usd_now,
            "usd_daily_pct": usd_daily_pct,
            "usd_weekly_pct": usd_weekly_pct,
            "threshold_daily": th["daily"],
            "threshold_weekly": th["weekly"],
            "impact": impact,
        })

    return {
        "available": True,
        "as_of": latest_resp.get("date"),
        "rows": rows,
    }


def fx_snapshot_to_signals(snapshot: dict) -> list[dict]:
    """Turn threshold breaches into report-ready signal dicts. Deterministic:
    numbers come straight from the API, never from the LLM."""
    if not snapshot.get("available"):
        return [{
            "TITLE": "FX rate data unavailable",
            "IMPACT": "WATCH",
            "CATEGORY": "FX Rate Alert",
            "SUMMARY": "The Frankfurter/ECB reference rate API could not be reached this run.",
            "IMPLICATIONS": "FX threshold monitoring is degraded for this run; verify manually via a market data source.",
            "SOURCE": "Frankfurter API (ECB reference rates)",
            "URL": "https://www.frankfurter.app",
            "DATE": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }]

    signals = []
    as_of = snapshot["as_of"]
    for row in snapshot["rows"]:
        if not row["impact"]:
            continue

        def fmt_pct(x):
            return f"{x:+.2f}%" if x is not None else "n/a"

        summary = (
            f"{row['currency']}/USD: {row['usd_rate']:.4f} "
            f"(daily {fmt_pct(row['usd_daily_pct'])}, weekly {fmt_pct(row['usd_weekly_pct'])}). "
            f"{row['currency']}/EUR: {row['eur_rate']:.4f} "
            f"(daily {fmt_pct(row['eur_daily_pct'])}, weekly {fmt_pct(row['eur_weekly_pct'])}). "
            f"Alert thresholds for {row['currency']}: daily {row['threshold_daily']}%, "
            f"weekly {row['threshold_weekly']}%."
        )
        implications = (
            f"A move of this magnitude in {row['currency']} directly affects the EUR/USD-denominated "
            "value of export sales invoiced in this currency and the cost of any existing forward hedges. "
            "Review open hedging positions and consider adjusting forward cover ratios for upcoming "
            "shipments if the trend persists."
        )
        signals.append({
            "TITLE": f"{row['currency']} threshold breach ({row['impact']})",
            "IMPACT": row["impact"],
            "CATEGORY": "FX Rate Alert",
            "SUMMARY": summary,
            "IMPLICATIONS": implications,
            "SOURCE": "Frankfurter API (ECB reference rates)",
            "URL": "https://www.frankfurter.app",
            "DATE": as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })
    return signals


# ---------------------------------------------------------------------------
# STATIC SCRAPING
# ---------------------------------------------------------------------------

def scrape_source(source_cfg: dict) -> list[dict]:
    html = fetch_url(source_cfg["url"])
    if not html:
        logger.warning("Source unreachable (scrape): %s", source_cfg["name"])
        return []

    soup = BeautifulSoup(html, "lxml")
    articles, seen_local = [], set()

    for a_tag in soup.find_all("a", href=True):
        text = a_tag.get_text(strip=True)
        href = a_tag["href"]
        if not text or len(text) < source_cfg.get("min_text_len", 20):
            continue
        if not keyword_match(text):
            continue
        full_url = urljoin(source_cfg["url"], href)
        if full_url in seen_local or full_url == source_cfg["url"]:
            continue
        seen_local.add(full_url)
        articles.append({
            "title": text,
            "url": full_url,
            "source": source_cfg["name"],
            "date": None,
            "snippet": "",
        })

    max_items = source_cfg.get("max_items", 15)
    logger.info("Scrape [%s]: %d relevant article(s)", source_cfg["name"], len(articles[:max_items]))
    return articles[:max_items]


def scrape_all_static_sources() -> list[dict]:
    all_articles = []
    sources = STATIC_SOURCES[:3] if TEST_MODE else STATIC_SOURCES
    for src in sources:
        all_articles.extend(scrape_source(src))
    return all_articles


# ---------------------------------------------------------------------------
# TAVILY SEARCH
# ---------------------------------------------------------------------------

def tavily_search_all() -> list[dict]:
    if not TAVILY_API_KEY or TavilyClient is None:
        logger.warning("TAVILY_API_KEY missing or tavily package not installed: skipping Tavily search")
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = []
    queries = TAVILY_QUERIES[:3] if TEST_MODE else TAVILY_QUERIES

    for query in queries:
        try:
            resp = client.search(
                query=query,
                search_depth="advanced",
                topic="news",
                days=3,
                max_results=6,
                include_answer=False,
            )
        except Exception as exc:  # noqa: BLE001 - log and continue
            logger.warning("Tavily failed for query '%s': %s", query, exc)
            continue

        for item in resp.get("results", []):
            results.append({
                "title": item.get("title", "").strip(),
                "url": item.get("url", ""),
                "source": f"Tavily ({query})",
                "date": item.get("published_date"),
                "snippet": (item.get("content") or "")[:600],
            })
        logger.info("Tavily [%s]: %d result(s)", query, len(resp.get("results", [])))

    return results


# ---------------------------------------------------------------------------
# DEDUPLICATION / PERSISTENT STATE
# ---------------------------------------------------------------------------

def load_seen_urls() -> dict:
    if not SEEN_URLS_FILE.exists():
        return {}
    try:
        return json.loads(SEEN_URLS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_seen_urls(seen: dict) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=SEEN_URLS_RETENTION_DAYS)
    pruned = {}
    for url, date_str in seen.items():
        try:
            if datetime.fromisoformat(date_str) >= cutoff:
                pruned[url] = date_str
        except ValueError:
            continue
    SEEN_URLS_FILE.write_text(json.dumps(pruned, ensure_ascii=False, indent=2), encoding="utf-8")


def dedupe_and_filter(articles: list[dict], seen_urls: dict) -> list[dict]:
    deduped, seen_local = [], set()
    for art in articles:
        url = art.get("url")
        if not url or url in seen_local or url in seen_urls:
            continue
        if not keyword_match(art.get("title", "") + " " + art.get("snippet", "")):
            continue
        seen_local.add(url)
        deduped.append(art)
    return deduped


# ---------------------------------------------------------------------------
# ENRICHMENT
# ---------------------------------------------------------------------------

def enrich_article(article: dict) -> dict:
    html = fetch_url(article["url"])
    if not html:
        article["body"] = article.get("snippet", "")
        return article
    try:
        soup = BeautifulSoup(html, "lxml")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        article["body"] = text[:2000] if text else article.get("snippet", "")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Enrichment failed for %s: %s", article["url"], exc)
        article["body"] = article.get("snippet", "")
    return article


def enrich_all(articles: list[dict]) -> list[dict]:
    enriched = []
    for art in articles:
        enriched.append(enrich_article(art))
        time.sleep(0.5)  # courtesy delay towards target servers
    return enriched


# ---------------------------------------------------------------------------
# DEEPSEEK - API CALL
# ---------------------------------------------------------------------------

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"


def call_deepseek(messages: list[dict], max_tokens: int = 4000, temperature: float = 0.3):
    """Returns (text, truncated: bool). text is None on total failure."""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing: cannot call DeepSeek")
        return None, False

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = post_json(DEEPSEEK_URL, headers, payload)
    if not data:
        return None, False

    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        truncated = choice.get("finish_reason") == "length"
        return text, truncated
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected DeepSeek response shape: %s", exc)
        return None, False


# ---------------------------------------------------------------------------
# WEEKLY STRUCTURAL BRIEF (DeepSeek training knowledge)
# ---------------------------------------------------------------------------

WEEKLY_BRIEF_PROMPT = """You are a macro FX strategist covering Asia-Pacific currencies.
Write, in English and in Markdown, a structural background note (not recent news,
only your background knowledge) for the APAC CFO of a Western manufacturing group
invoicing exports in CNY, JPY, KRW, THB, PHP and AUD, reporting in EUR, with USD as
a secondary reference currency. The note should cover, in 500-700 words:

1. The FX regime of each currency (CNY: PBOC-managed float with daily reference
   rate and trading band; JPY, KRW, THB, PHP, AUD: free-floating with varying
   degrees of central bank intervention).
2. The mandate and typical policy tools of each central bank (PBOC, BOJ, BOK,
   BOT, BSP, RBA) relevant to currency stability.
3. Structural drivers of these currencies versus USD/EUR (rate differentials,
   capital flows, commodity exposure for AUD, export competitiveness dynamics).
4. Practical implications for a manufacturing exporter: margin exposure from
   invoicing currency mismatches, typical corporate hedging instruments (forward
   contracts, currency swaps, natural hedging), and timing considerations.

This note serves as a background reading grid; it will be combined with recent
FX rate data and central bank news analyzed separately. Do not fabricate any
recent news or precise current exchange rate figures you are not sure of."""


def get_weekly_brief() -> str:
    if WEEKLY_BRIEF_CACHE_FILE.exists():
        try:
            cache = json.loads(WEEKLY_BRIEF_CACHE_FILE.read_text(encoding="utf-8"))
            cached_date = datetime.fromisoformat(cache["date"])
            if datetime.now(timezone.utc) - cached_date < timedelta(days=WEEKLY_BRIEF_MAX_AGE_DAYS):
                logger.info("Weekly structural brief: valid cache (generated on %s)", cache["date"])
                return cache["brief"]
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Unreadable weekly brief cache, regenerating: %s", exc)

    logger.info("Weekly structural brief: calling DeepSeek (cache expired or missing)")
    text, truncated = call_deepseek(
        messages=[{"role": "user", "content": WEEKLY_BRIEF_PROMPT}],
        max_tokens=1600,
        temperature=0.2,
    )
    if not text:
        logger.warning("Could not generate weekly brief, using a fallback text")
        text = ("Structural brief unavailable for this run (API failure). "
                "Refer to the last known version or a primary FX research source.")
    if truncated:
        logger.warning("The weekly structural brief appears to be truncated")

    WEEKLY_BRIEF_CACHE_FILE.write_text(
        json.dumps({"date": datetime.now(timezone.utc).isoformat(), "brief": text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return text


# ---------------------------------------------------------------------------
# SIGNAL ANALYSIS (delimiter-based structured output)
# ---------------------------------------------------------------------------

SIGNAL_BLOCK_RE = re.compile(r"===SIGNAL_START===(.*?)===SIGNAL_END===", re.DOTALL)
EXEC_SUMMARY_RE = re.compile(r"===EXEC_SUMMARY_START===(.*?)===EXEC_SUMMARY_END===", re.DOTALL)
TOP_RISK_RE = re.compile(r"===TOP_RISK_START===(.*?)===TOP_RISK_END===", re.DOTALL)

SIGNAL_FIELD_RE = re.compile(r"^(TITLE|IMPACT|CATEGORY|SUMMARY|IMPLICATIONS|SOURCE|URL|DATE):\s*(.*)$")

VALID_IMPACTS = {"CRITICAL", "IMPORTANT", "WATCH", "INFO"}


def format_fx_snapshot_for_prompt(snapshot: dict) -> str:
    if not snapshot.get("available"):
        return "FX rate data unavailable for this run."
    lines = [f"Reference date: {snapshot['as_of']}"]
    for row in snapshot["rows"]:
        lines.append(
            f"- {row['currency']}/USD={row['usd_rate']:.4f} "
            f"(daily {row['usd_daily_pct']:+.2f}%, weekly {row['usd_weekly_pct']:+.2f}%); "
            f"{row['currency']}/EUR={row['eur_rate']:.4f} "
            f"(daily {row['eur_daily_pct']:+.2f}%, weekly {row['eur_weekly_pct']:+.2f}%); "
            f"threshold status: {row['impact'] or 'within normal range'}"
            if row["usd_daily_pct"] is not None and row["eur_daily_pct"] is not None
            else f"- {row['currency']}: incomplete historical data this run"
        )
    return "\n".join(lines)


def build_analysis_prompt(weekly_brief: str, fx_snapshot: dict, articles: list[dict]) -> str:
    articles_block = []
    for i, art in enumerate(articles, start=1):
        articles_block.append(
            f"[Article {i}]\n"
            f"Title: {art['title']}\n"
            f"Source: {art['source']}\n"
            f"Date: {art.get('date') or 'unknown'}\n"
            f"URL: {art['url']}\n"
            f"Excerpt: {art.get('body', art.get('snippet',''))[:1200]}\n"
        )
    articles_text = "\n".join(articles_block) if articles_block else "No qualitative articles matched the keyword filter this run."

    return f"""You are an FX risk analyst for {ORG_NAME} ({ORG_CONTEXT}), scope: {REGION_LABEL}.

STRUCTURAL CONTEXT (background reading grid):
{weekly_brief}

CENTRAL BANKS / ENTITIES UNDER WATCH:
{chr(10).join('- ' + e for e in ENTITIES_TO_WATCH)}

VERIFIED FX RATE DATA FOR THIS RUN (ground truth - do not alter these figures):
{format_fx_snapshot_for_prompt(fx_snapshot)}

RECENT CENTRAL BANK / FX NEWS ARTICLES TO ANALYZE:
{articles_text}

INSTRUCTIONS:
Your job is to explain WHY the FX rate data above may be moving and what it means
-- do not invent alternative percentages or rates. For each news article that is
GENUINELY relevant (ignore off-topic or overly generic articles), and wherever
you can draw a credible link between a central bank action/statement and the FX
rate data above, produce a block in the EXACT following format (nothing before
or after the delimiters):

===SIGNAL_START===
TITLE: <short, clear title in English>
IMPACT: <CRITICAL|IMPORTANT|WATCH|INFO>
CATEGORY: <Central Bank Policy|Currency Intervention|Capital Flows|Hedging Strategy|Other>
SUMMARY: <2-3 factual sentences in English, referencing the verified FX data above where relevant>
IMPLICATIONS: <concrete impact on export margins and FX hedging for an APAC CFO invoicing in these currencies, in English>
SOURCE: <source name>
URL: <original url>
DATE: <date if known, otherwise "unknown">
===SIGNAL_END===

Impact scale:
- CRITICAL: central bank action or FX move requiring immediate hedging review or cashflow impact assessment.
- IMPORTANT: significant policy signal or trend requiring action or close monitoring in the coming weeks.
- WATCH: weak signal or early-stage policy discussion to monitor, no immediate action required.
- INFO: useful contextual information, no action required.

Then, add EXACTLY one executive summary block:
===EXEC_SUMMARY_START===
<4-6 sentences in English summarizing the FX situation and central bank activity for this run>
===EXEC_SUMMARY_END===

Then EXACTLY one top risk block:
===TOP_RISK_START===
<2-4 sentences identifying the single most important FX risk or watch point right now>
===TOP_RISK_END===

If no article is relevant, produce only the EXEC_SUMMARY and TOP_RISK blocks, basing
them on the verified FX rate data alone."""


def parse_signals(raw_text: str) -> tuple[list[dict], str, str, bool]:
    """Returns (signals, executive_summary, top_risk, truncation_detected)."""
    signals = []
    for block in SIGNAL_BLOCK_RE.findall(raw_text):
        fields = {}
        for line in block.strip().splitlines():
            match = SIGNAL_FIELD_RE.match(line.strip())
            if match:
                fields[match.group(1)] = match.group(2).strip()
        if fields.get("TITLE") and fields.get("IMPACT") in VALID_IMPACTS:
            signals.append(fields)
        elif fields.get("TITLE"):
            fields["IMPACT"] = "INFO"
            signals.append(fields)

    exec_summary_match = EXEC_SUMMARY_RE.search(raw_text)
    top_risk_match = TOP_RISK_RE.search(raw_text)
    exec_summary = exec_summary_match.group(1).strip() if exec_summary_match else ""
    top_risk = top_risk_match.group(1).strip() if top_risk_match else ""

    # Truncation detection: a START delimiter without a matching END, or a
    # last block clearly cut off mid-field.
    truncation_suspected = raw_text.count("===SIGNAL_START===") > raw_text.count("===SIGNAL_END===")
    if not exec_summary_match and "===EXEC_SUMMARY_START===" in raw_text:
        truncation_suspected = True
    if not top_risk_match and "===TOP_RISK_START===" in raw_text:
        truncation_suspected = True

    return signals, exec_summary, top_risk, truncation_suspected


def analyze_articles(weekly_brief: str, fx_snapshot: dict, articles: list[dict]):
    prompt = build_analysis_prompt(weekly_brief, fx_snapshot, articles)
    raw_text, api_truncated = call_deepseek(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        temperature=0.3,
    )
    if not raw_text:
        logger.error("DeepSeek analysis unavailable: generating a degraded report from raw articles")
        fallback_signals = [{
            "TITLE": art["title"],
            "IMPACT": "WATCH",
            "CATEGORY": "Unclassified (AI analysis unavailable)",
            "SUMMARY": (art.get("body") or art.get("snippet") or "")[:300],
            "IMPLICATIONS": "To be assessed manually (DeepSeek call failed).",
            "SOURCE": art["source"],
            "URL": art["url"],
            "DATE": art.get("date") or "unknown",
        } for art in articles]
        fallback_exec = "AI analysis unavailable for this run; see FX dashboard for verified rate data."
        return fallback_signals, fallback_exec, "N/A", False

    signals, exec_summary, top_risk, parse_truncated = parse_signals(raw_text)
    truncated = api_truncated or parse_truncated
    if truncated:
        logger.warning("Truncation detected in DeepSeek response: some signals may be missing")
    return signals, exec_summary, top_risk, truncated


# ---------------------------------------------------------------------------
# HTML REPORT
# ---------------------------------------------------------------------------

IMPACT_ORDER = ["CRITICAL", "IMPORTANT", "WATCH", "INFO"]
IMPACT_STYLE = {
    "CRITICAL": {"color": "#b91c1c", "bg": "#fee2e2", "label": "CRITICAL"},
    "IMPORTANT": {"color": "#c2410c", "bg": "#ffedd5", "label": "IMPORTANT"},
    "WATCH": {"color": "#a16207", "bg": "#fef9c3", "label": "WATCH"},
    "INFO": {"color": "#1d4ed8", "bg": "#dbeafe", "label": "INFO"},
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>APAC FX Risk Watch - {run_date}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; background:#f3f4f6; color:#111827; margin:0; padding:0; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 24px; }}
  header {{ background:#111827; color:#fff; padding: 28px 24px; }}
  header h1 {{ margin:0 0 6px 0; font-size: 22px; }}
  header p {{ margin:0; color:#d1d5db; font-size: 14px; }}
  .card {{ background:#fff; border-radius: 10px; padding: 20px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight:600; letter-spacing:.03em; }}
  .signal {{ border-left: 5px solid #9ca3af; padding: 14px 18px; margin-bottom: 14px; border-radius: 6px; }}
  .signal h3 {{ margin: 4px 0 8px 0; font-size: 16px; }}
  .signal .meta {{ font-size: 12px; color:#6b7280; margin-bottom: 8px; }}
  .signal p {{ margin: 6px 0; font-size: 14px; line-height: 1.5; }}
  .signal a {{ color:#1d4ed8; text-decoration:none; }}
  .signal a:hover {{ text-decoration:underline; }}
  h2.section-title {{ font-size: 15px; text-transform: uppercase; letter-spacing:.04em; color:#374151; margin: 28px 0 12px 0; }}
  .top-risk {{ background:#fff7ed; border:1px solid #fdba74; border-radius: 10px; padding: 18px; }}
  .brief {{ font-size: 13px; color:#374151; }}
  .footer-note {{ font-size: 12px; color:#9ca3af; margin-top: 24px; }}
  .truncation-warning {{ background:#fee2e2; color:#991b1b; padding: 10px 16px; border-radius: 8px; font-size: 13px; margin-bottom: 16px; }}
  table.fx-table {{ width:100%; border-collapse: collapse; font-size: 13px; }}
  table.fx-table th, table.fx-table td {{ padding: 8px 10px; text-align: right; border-bottom: 1px solid #e5e7eb; }}
  table.fx-table th:first-child, table.fx-table td:first-child {{ text-align: left; }}
  table.fx-table th {{ color:#6b7280; font-weight:600; text-transform: uppercase; font-size: 11px; }}
  .pos {{ color:#15803d; }}
  .neg {{ color:#b91c1c; }}
</style>
</head>
<body>
<header>
  <h1>APAC FX Risk Watch</h1>
  <p>{org_name} - {org_context}</p>
  <p>Scope: {region_label}</p>
  <p>Report generated on {run_date} UTC &middot; FX reference date: {fx_as_of}</p>
</header>
<div class="container">

  {truncation_html}

  <div class="card">
    <h2 class="section-title">Executive summary</h2>
    <p>{exec_summary}</p>
  </div>

  <div class="top-risk">
    <h2 class="section-title" style="margin-top:0;">Top risk to watch</h2>
    <p>{top_risk}</p>
  </div>

  <div class="card">
    <h2 class="section-title" style="margin-top:0;">FX dashboard (ECB reference rates)</h2>
    {fx_table_html}
  </div>

  {signals_html}

  <div class="card">
    <h2 class="section-title">Structural brief (background context, updated weekly)</h2>
    <div class="brief">{weekly_brief_html}</div>
  </div>

  <p class="footer-note">
    Automatically generated by apac_fx_risk_agent.py &middot;
    {nb_signals} signal(s) detected across {nb_articles} article(s) analyzed &middot;
    FX data: Frankfurter API (ECB reference rates) &middot; News: direct scraping + Tavily search.
  </p>
</div>
</body>
</html>"""

SIGNAL_TEMPLATE = """<div class="signal" style="border-left-color:{color}; background:{bg}22;">
  <span class="badge" style="background:{bg}; color:{color};">{impact_label}</span>
  <span class="badge" style="background:#e5e7eb; color:#374151;">{category}</span>
  <h3>{title}</h3>
  <div class="meta">{source} &middot; {date}</div>
  <p>{summary}</p>
  <p><strong>Implication:</strong> {implications}</p>
  <p><a href="{url}" target="_blank" rel="noopener">View original source &rarr;</a></p>
</div>"""


def html_escape(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _pct_cell(pct: Optional[float]) -> str:
    if pct is None:
        return "n/a"
    css_class = "pos" if pct >= 0 else "neg"
    return f'<span class="{css_class}">{pct:+.2f}%</span>'


def render_fx_table(snapshot: dict) -> str:
    if not snapshot.get("available"):
        return "<p>FX rate data unavailable for this run.</p>"

    rows_html = []
    for row in snapshot["rows"]:
        impact = row["impact"]
        flag = f'<span class="badge" style="background:{IMPACT_STYLE[impact]["bg"]};color:{IMPACT_STYLE[impact]["color"]};">{impact}</span>' if impact else ""
        rows_html.append(
            "<tr>"
            f"<td>{row['currency']} {flag}</td>"
            f"<td>{row['usd_rate']:.4f}</td>"
            f"<td>{_pct_cell(row['usd_daily_pct'])}</td>"
            f"<td>{_pct_cell(row['usd_weekly_pct'])}</td>"
            f"<td>{row['eur_rate']:.4f}</td>"
            f"<td>{_pct_cell(row['eur_daily_pct'])}</td>"
            f"<td>{_pct_cell(row['eur_weekly_pct'])}</td>"
            "</tr>"
        )

    return f"""<table class="fx-table">
      <thead>
        <tr>
          <th>Currency</th>
          <th>vs USD</th><th>Daily %</th><th>Weekly %</th>
          <th>vs EUR</th><th>Daily %</th><th>Weekly %</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>"""


def generate_html_report(signals: list[dict], exec_summary: str, top_risk: str,
                          weekly_brief: str, fx_snapshot: dict, nb_articles: int,
                          truncated: bool) -> str:
    grouped = {impact: [] for impact in IMPACT_ORDER}
    for sig in signals:
        grouped.setdefault(sig.get("IMPACT", "INFO"), []).append(sig)

    signals_html_parts = []
    for impact in IMPACT_ORDER:
        items = grouped.get(impact, [])
        if not items:
            continue
        style = IMPACT_STYLE[impact]
        signals_html_parts.append(f'<h2 class="section-title">{style["label"]} ({len(items)})</h2>')
        for sig in items:
            signals_html_parts.append(SIGNAL_TEMPLATE.format(
                color=style["color"],
                bg=style["bg"],
                impact_label=style["label"],
                category=html_escape(sig.get("CATEGORY", "Other")),
                title=html_escape(sig.get("TITLE", "")),
                source=html_escape(sig.get("SOURCE", "")),
                date=html_escape(sig.get("DATE", "unknown")),
                summary=html_escape(sig.get("SUMMARY", "")),
                implications=html_escape(sig.get("IMPLICATIONS", "")),
                url=sig.get("URL", "#"),
            ))

    if not signals_html_parts:
        signals_html_parts.append('<div class="card"><p>No significant signal detected this run.</p></div>')

    truncation_html = ""
    if truncated:
        truncation_html = (
            '<div class="truncation-warning">Warning: the AI analysis response appears to have '
            "been truncated. Some signals or sections may be incomplete.</div>"
        )

    return HTML_TEMPLATE.format(
        run_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        org_name=ORG_NAME,
        org_context=ORG_CONTEXT,
        region_label=REGION_LABEL,
        fx_as_of=fx_snapshot.get("as_of") or "unavailable",
        truncation_html=truncation_html,
        exec_summary=html_escape(exec_summary) or "No summary available.",
        top_risk=html_escape(top_risk) or "No particular risk identified.",
        fx_table_html=render_fx_table(fx_snapshot),
        signals_html="\n".join(signals_html_parts),
        weekly_brief_html=markdown_lib.markdown(weekly_brief),
        nb_signals=len(signals),
        nb_articles=nb_articles,
    )


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Starting apac_fx_risk_agent.py (TEST_MODE=%s) ===", TEST_MODE)

    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY missing: the report will be degraded (no AI analysis).")
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY missing: Tavily search will be skipped.")

    weekly_brief = get_weekly_brief()

    fx_snapshot = build_fx_snapshot()
    fx_alert_signals = fx_snapshot_to_signals(fx_snapshot)
    logger.info("FX dashboard: %d threshold alert(s) out of %d currencies",
                len(fx_alert_signals) if fx_snapshot.get("available") else 0, len(FX_CURRENCIES))

    scraped = scrape_all_static_sources()
    tavily_results = tavily_search_all()
    logger.info("Raw total: %d scraped article(s), %d Tavily result(s)", len(scraped), len(tavily_results))

    seen_urls = load_seen_urls()
    candidates = dedupe_and_filter(scraped + tavily_results, seen_urls)
    logger.info("%d article(s) retained after keyword filtering and deduplication", len(candidates))

    if TEST_MODE:
        candidates = candidates[:8]

    enriched = enrich_all(candidates)

    news_signals, exec_summary, top_risk, truncated = analyze_articles(weekly_brief, fx_snapshot, enriched)

    all_signals = fx_alert_signals + news_signals

    html_report = generate_html_report(
        signals=all_signals,
        exec_summary=exec_summary,
        top_risk=top_risk,
        weekly_brief=weekly_brief,
        fx_snapshot=fx_snapshot,
        nb_articles=len(enriched),
        truncated=truncated,
    )

    run_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = REPORTS_DIR / f"apac_fx_risk_report_{run_date_str}.html"
    report_path.write_text(html_report, encoding="utf-8")
    (REPORTS_DIR / "latest.html").write_text(html_report, encoding="utf-8")
    logger.info("Report written: %s", report_path)

    now_iso = datetime.now(timezone.utc).isoformat()
    for art in enriched:
        seen_urls[art["url"]] = now_iso
    save_seen_urls(seen_urls)

    logger.info("=== Done: %d signal(s) (%d FX alerts + %d news) across %d article(s) analyzed ===",
                len(all_signals), len(fx_alert_signals), len(news_signals), len(enriched))


if __name__ == "__main__":
    main()
