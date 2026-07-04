import os
import json
import logging
import hashlib
import requests
import time
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import markdown

# =============================================================================
#  CONFIGURATION
# =============================================================================
load_dotenv()
LOG_FILE   = Path("logs/agent_apac.log")
SEEN_FILE  = Path("seen_apac_articles.json")
Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# Max articles to enrich with body excerpt (costs extra HTTP requests)
ENRICH_MAX = 20
# Max articles sent to DeepSeek in one call (keep prompt manageable)
DEEPSEEK_MAX_ARTICLES = 50

# =============================================================================
#  KEYWORDS — APAC ex-mainland China
# =============================================================================
KEYWORDS_GSE = [
    # Core GSE equipment
    "ground support", "gse", "tug", "tractor", "loader", "de-icer", "gpu",
    "towbar", "baggage", "passenger boarding bridge", "air start unit",
    "belt loader", "conveyor belt", "staircase", "dolly", "catering truck",
    "lavatory truck", "water truck", "apron", "ramp", "electric ground support",
    "hybrid gse", "lithium battery gse", "autonomous gse", "maintenance gse",
    "mro ground",
    # APAC airports (demand drivers)
    "Tokyo Haneda", "Tokyo Narita", "Kansai International", "Chubu Centrair",
    "Seoul Incheon", "Gimpo International", "Hong Kong International",
    "Taipei Taoyuan", "Kaohsiung International", "Singapore Changi",
    "Kuala Lumpur International", "Kota Kinabalu", "Penang International",
    "Bangkok Suvarnabhumi", "Don Mueang", "Phuket International",
    "Jakarta Soekarno-Hatta", "Bali Ngurah Rai", "Surabaya Juanda",
    "Manila Ninoy Aquino", "Cebu Mactan", "Clark International",
    "Ho Chi Minh Tan Son Nhat", "Hanoi Noi Bai", "Da Nang International",
    "Phnom Penh International", "Yangon International",
    "Dhaka Hazrat Shahjalal", "Kathmandu Tribhuvan", "Colombo Bandaranaike",
    "Ulaanbaatar Chinggis Khaan",
    "Sydney Kingsford Smith", "Melbourne Tullamarine", "Brisbane Airport",
    "Perth Airport", "Auckland International", "Fiji Nadi",
    "airport opening", "new runway", "terminal expansion", "airport expansion",
    "passenger record", "traffic record", "cargo volume", "load factor",
    "inauguration", "infrastructure investment",
    # Chinese airlines operating internationally (fleet = GSE demand)
    "Air China", "China Eastern", "China Southern", "Hainan Airlines",
    "XiamenAir", "Shenzhen Airlines", "Sichuan Airlines", "Shandong Airlines",
    "Juneyao Airlines", "Spring Airlines",
    "中国国航", "国航", "中国东方航空", "东方航空", "东航",
    "中国南方航空", "南方航空", "南航", "海南航空", "海航",
    "厦门航空", "厦航", "深圳航空", "深航", "春秋航空", "春秋",
    "吉祥航空", "吉祥", "四川航空", "川航", "山东航空", "山航",
    # Other APAC airlines (fleet = GSE demand)
    "Cathay Pacific", "Hong Kong Airlines", "Greater Bay Airlines",
    "Korean Air", "Asiana Airlines", "Jeju Air", "T'way Air",
    "Japan Airlines", "ANA", "All Nippon Airways", "Peach Aviation",
    "China Airlines", "EVA Air", "StarLux",
    "Singapore Airlines", "Scoot", "Malaysia Airlines", "AirAsia",
    "Thai Airways", "Thai AirAsia", "Vietnam Airlines", "VietJet",
    "Garuda Indonesia", "Lion Air", "Philippine Airlines", "Cebu Pacific",
    "Qantas", "Jetstar", "Virgin Australia", "Air New Zealand", "Fiji Airways",
    "airline order", "fleet delivery", "fleet expansion", "airline profit",
    "airline loss", "bankruptcy", "revenue", "EBIT",
    "订购", "交付", "机队", "盈利", "亏损", "营收", "净利润",
    "复航", "停飞", "新开航线", "重组",
    # Ground handlers (M&A signals)
    "SATS", "dnata", "Swissport", "Menzies Aviation", "Worldwide Flight Services",
    "WFS", "Celebi", "Havas Ground Handling", "Gapura Angkasa", "Aviapartner",
    "Alliance Ground International", "斯威斯波特",
    # Chinese GSE competitors — international / export angle (primary lens)
    "Weihai Guangtai", "Guangtai", "威海广泰", "广泰", "广泰航空",
    "威海广泰航空", "GT系列", "广泰电动", "广泰牵引", "广泰出口",
    "CIMC Tianda", "中集天达", "天达", "CIMC",
    "Jiangsu Tianyi", "Tianyi", "江苏天一",
    "Shenzhen TECHKING", "TECHKING", "深圳达航",
    "Hangfu", "航福",
    "Shanghai Jiajie", "上海嘉捷",
    "Guangzhou Jinhaoyang", "广州金浩阳",
    "Shenyang Tianhua", "沈阳天华",
    "Shandong Tianhe", "山东天河",
    "Zhejiang Goodsense", "浙江中力",
    "Alha GSE", "Shanghai Ifly", "Ifly GSE",
    # Western / global GSE competitors
    "TLD Group", "TLD", "Alvest", "JBT Corporation", "JBT",
    "Oshkosh AeroTech", "Oshkosh", "Textron GSE", "Textron",
    "Tug Technologies", "Tronair", "ITW GSE", "Fast Global Solutions",
    "Fast Global", "WASP GSE", "Mallaghan", "Mallaghan Engineering",
    "Goldhofer", "MULAG", "HYDRO", "Guinault", "Cavotec",
    "AERO Specialties", "Aero Specialties", "Global Ground Support",
    "DOLL", "Nepean", "Gate GSE", "Clyde Machines", "Douglas Equipment",
    "TCR Group", "TCR", "Mercury GSE",
    "Toyota Industries", "JCB", "Jungheinrich", "Komatsu", "Cobus",
    "Vestergaard", "Trepel", "Aviapartner",
    # Electrification / regulation / supply chain
    "emission regulation", "electric ramp", "diesel ban",
    "steel price", "aluminium", "lithium", "battery cost",
    "semiconductor", "chip shortage", "supply chain disruption",
    "carbon peak", "electrification", "net zero airport",
    "电动化", "碳中和机场",
    # Trade / geopolitics
    "Belt and Road", "BRI", "tariff", "trade war", "EU tariffs",
    "export control", "China export", "一带一路", "关税",
    "APAC", "Asia Pacific", "Southeast Asia", "ASEAN", "South Asia",
    # Tenders / procurement
    "tender", "procurement", "contract award", "RFP", "RFQ",
]

# =============================================================================
#  SOURCES
# =============================================================================
SOURCES = [
    {
        "nom": "International Airport Review - News",
        "url": "https://www.internationalairportreview.com/news/",
        "type": "scrape_generic",
        "selector": "article h3 a, .post-title a, .entry-title a, a",
        "base_url": "https://www.internationalairportreview.com",
    },
    {
        "nom": "Airport Technology - News",
        "url": "https://www.airport-technology.com/news/",
        "type": "scrape_generic",
        "selector": "article h3 a, .card-title a, .post-title a, a",
        "base_url": "https://www.airport-technology.com",
    },
    {
        "nom": "Future Airport",
        "url": "https://www.futureairport.com/",
        "type": "scrape_generic",
        "selector": "article h2 a, .post-title a, .entry-title a, a",
        "base_url": "https://www.futureairport.com",
    },
    {
        "nom": "Airport World - Asia Pacific",
        "url": "https://www.airport-world.com/category/regions/asia-pacific/",
        "type": "scrape_generic",
        "selector": "article h3 a, .entry-title a, a",
        "base_url": "https://www.airport-world.com",
    },
    {
        "nom": "Ground Handling International",
        "url": "https://www.groundhandling.com/",
        "type": "scrape_generic",
        "selector": "article h3 a, .post-title a, h2.entry-title a, a",
        "base_url": "https://www.groundhandling.com",
    },
    {
        "nom": "Aviation Pros - Ground Handling",
        "url": "https://www.aviationpros.com/ground-handling",
        "type": "scrape_generic",
        "selector": "div.article-listing a, h2.article-title a, .listing-title a, a",
        "base_url": "https://www.aviationpros.com",
    },
    {
        "nom": "Simple Flying - Asia",
        "url": "https://simpleflying.com/category/asia/",
        "type": "scrape_generic",
        "selector": "article h2 a, .post-title a, a",
        "base_url": "https://simpleflying.com",
    },
    {
        "nom": "Aviation Week - Asia",
        "url": "https://aviationweek.com/regions/asia",
        "type": "scrape_generic",
        "selector": "h3.article-title a, .node-title a, .headline a, a",
        "base_url": "https://aviationweek.com",
    },
    {
        "nom": "Flight Global - Asia",
        "url": "https://www.flightglobal.com/asia/",
        "type": "scrape_generic",
        "selector": "article h3 a, .teaser-title a, a",
        "base_url": "https://www.flightglobal.com",
    },
    {
        "nom": "Reuters - Aerospace & Defense",
        "url": "https://www.reuters.com/business/aerospace-defense/",
        "type": "scrape_generic",
        "selector": "article h3 a, .story-title a, a",
        "base_url": "https://www.reuters.com",
    },
]

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def normaliser_url(url, base=None):
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    parsed = urlparse(url)
    url_propre = parsed._replace(query="", fragment="").geturl()
    if url_propre.endswith("/"):
        url_propre = url_propre[:-1]
    return url_propre


def charger_vus():
    if SEEN_FILE.exists():
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            try:
                return set(json.load(f))
            except Exception:
                return set()
    return set()


def sauvegarder_vus(vus):
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(list(vus), f, ensure_ascii=False, indent=2)


def requeter_avec_retry(url, retries=3, timeout=30, **kwargs):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    for i in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            log.warning(f"Attempt {i+1}/{retries} failed for {url}: {e}")
            time.sleep(2 ** i)
    return None


# =============================================================================
#  SCRAPERS
# =============================================================================

def scrape_generic(source):
    articles = []
    resp = requeter_avec_retry(source["url"])
    if not resp:
        return articles
    try:
        encoding = source.get("encoding", "utf-8")
        soup = BeautifulSoup(resp.content, "html.parser", from_encoding=encoding)
        unique_links = {}
        for link in soup.select(source["selector"]):
            href = link.get("href")
            titre = link.get_text(strip=True)
            if not href or not titre or len(titre) < 10:
                continue
            href = normaliser_url(href, source.get("base_url"))
            if href:
                unique_links[href] = titre
        for href, titre in list(unique_links.items())[:30]:
            articles.append({
                "source": source["nom"],
                "titre": titre[:150],
                "lien": href,
                "desc": "",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "id": hashlib.md5((titre + href).encode()).hexdigest(),
            })
    except Exception as e:
        log.warning(f"Error scraping {source['nom']}: {e}")
    log.info(f"  Scraped {source['nom']}: {len(articles)} articles")
    return articles


def collecter_tous_articles():
    tous = []
    for source in SOURCES:
        log.info(f"Collecting from: {source['nom']}")
        articles = scrape_generic(source)
        tous.extend(articles)
        time.sleep(1.5)
    log.info(f"Total raw articles collected: {len(tous)}")
    return tous


# =============================================================================
#  FILTERING — with verbose match logging
# =============================================================================
#
#  Short, all-caps ASCII acronyms (ANA, BRI, WFS, TLD...) are prone to
#  false-positive substring matches inside unrelated words (e.g. "BRI"
#  inside "Bristol", "Gabriel", "debris"; "ANA" inside "China", "Analysis",
#  "Annual"). For those, require word boundaries. Longer terms, terms with
#  spaces, and Chinese keywords keep simple substring matching.

def _est_acronyme_ambigu(kw):
    return kw.isascii() and kw.isalpha() and kw.isupper() and len(kw) <= 4


def _compiler_motifs_keywords():
    motifs = []
    for kw in KEYWORDS_GSE:
        if _est_acronyme_ambigu(kw):
            motifs.append((kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)))
        else:
            motifs.append((kw, None))
    return motifs


KEYWORD_PATTERNS = _compiler_motifs_keywords()


def filtrer_pertinents(articles, vus):
    nouveaux = []
    for a in articles:
        if a["id"] in vus:
            continue
        texte = (a["titre"] + " " + a.get("desc", "")).lower()
        matched = [
            kw for kw, motif in KEYWORD_PATTERNS
            if (motif.search(texte) if motif else kw.lower() in texte)
        ]
        if matched:
            log.info(
                f"  KEPT [{a['source']}] {a['titre'][:70]} "
                f"— matched: {matched[:3]}"
            )
            nouveaux.append(a)
        else:
            log.debug(f"  SKIP [{a['source']}] {a['titre'][:70]}")
    log.info(f"Relevant articles after filtering: {len(nouveaux)}")
    return nouveaux


# =============================================================================
#  ARTICLE ENRICHMENT — fetch body excerpt for better DeepSeek context
# =============================================================================

def enrichir_article(article):
    """Fetch first ~400 chars of article body to give DeepSeek more context."""
    try:
        resp = requests.get(
            article["lien"],
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True,
        )
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        article["desc"] = text[:400]
    except Exception as e:
        log.debug(f"Could not enrich {article['lien']}: {e}")
    return article


def enrichir_articles(articles):
    """Enrich up to ENRICH_MAX articles with body excerpts."""
    log.info(f"Enriching up to {ENRICH_MAX} articles with body excerpts...")
    enriched = []
    for i, a in enumerate(articles):
        if i < ENRICH_MAX:
            enriched.append(enrichir_article(a))
            time.sleep(0.4)
        else:
            enriched.append(a)
    log.info("Enrichment complete.")
    return enriched


# =============================================================================
#  TAVILY SEARCH — real-time web search for APAC GSE / competitor news
#
#  Tavily searches from its own servers, bypassing the IP-blocking that
#  prevents GitHub Actions US runners from reaching some regional sources,
#  and surfaces coverage the static scraper selectors miss entirely.
#
#  8 targeted queries per run, focused on Chinese competitors' export/
#  international moves in APAC and on regional GSE demand drivers.
# =============================================================================

TAVILY_QUERIES = [
    # Guangtai — primary Chinese rival, export/international angle
    "Weihai Guangtai export Southeast Asia airport 2026",
    "威海广泰 海外市场 出口 2026",
    # CIMC Tianda international
    "CIMC Tianda international ground support equipment 2026",
    # Western competitors active in APAC
    "JBT Corporation GSE Asia Pacific contract 2026",
    "Oshkosh AeroTech Asia Pacific airport 2026",
    # Airport tenders / infrastructure across APAC
    "Southeast Asia airport ground support equipment tender 2026",
    "Japan Korea airport GSE procurement 2026",
    # Ground handler M&A / contract wins
    "Swissport Menzies dnata Asia Pacific contract 2026",
]


def rechercher_tavily():
    """Search for APAC GSE competitor and market news using Tavily API.

    Tavily searches from its own servers, giving coverage the static
    scraper selectors above cannot reach. Returns articles in the same
    format as the scraper pipeline.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        log.warning("TAVILY_API_KEY not set — skipping Tavily search.")
        return []

    found     = []
    seen_urls = set()

    for query in TAVILY_QUERIES:
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":        api_key,
                    "query":          query,
                    "search_depth":   "basic",
                    "max_results":    5,
                    "include_answer": False,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            batch   = 0
            for r in results:
                url     = str(r.get("url", "")).strip()
                title   = str(r.get("title", "")).strip()[:150]
                content = str(r.get("content", "")).strip()[:300]

                if not url or not title or url in seen_urls:
                    continue
                seen_urls.add(url)

                found.append({
                    "source": "Tavily Search",
                    "titre":  title,
                    "lien":   url,
                    "desc":   content,
                    "date":   datetime.now().strftime("%Y-%m-%d"),
                    "id":     hashlib.md5((title + url).encode()).hexdigest(),
                })
                batch += 1

            log.info(f"  Tavily '{query[:45]}...': {batch} results")

        except Exception as e:
            log.warning(f"Tavily search failed for '{query[:40]}': {e}")

        time.sleep(0.5)

    log.info(f"Tavily total: {len(found)} articles found")
    return found


# =============================================================================
#  DEEPSEEK COMPETITOR BRIEF — no web search tool needed
#
#  Asks DeepSeek to report on Chinese GSE manufacturers' international /
#  export expansion into APAC markets from its training knowledge. Not
#  real-time, but covers structural intelligence (export contracts,
#  regional footprint, partnerships) that changes slowly and is largely
#  invisible to the scraper pipeline above.
#
#  Runs once per week (Monday) to avoid redundant daily API calls.
# =============================================================================

COMPETITOR_BRIEF_PROMPT = """You are a GSE (Ground Support Equipment) market intelligence analyst
specializing in the Asia-Pacific market (excluding mainland China).

Based on your training knowledge, provide a competitive intelligence brief on the
international / export expansion of the following Chinese manufacturers into APAC
markets (Southeast Asia, South Asia, Japan, Korea, Australia, New Zealand, Pacific).
Focus on information relevant to TLD Group (Alvest subsidiary), a Western GSE
manufacturer competing against them in the region.

For each company below, report what you know about:
- Their export footprint and installed base outside mainland China within APAC
- Any recent international contract wins, distributor partnerships, or market entries
- Their pricing strategy and market positioning vs TLD in APAC markets
- Which APAC countries/airports appear to be their priority targets

Companies to cover:
1. 威海广泰航空科技 (Weihai Guangtai Aviation Technology) — primary rival
2. 中集天达控股 (CIMC Tianda Holdings)
3. 江苏天一航空工业 (Jiangsu Tianyi Aviation Industry)
4. Any other Chinese GSE manufacturers with a notable APAC export presence

Return ONLY a JSON array. Each element must have exactly these fields:
  "company_cn": Chinese company name
  "company_en": English name
  "footprint": export footprint / installed base in APAC (one sentence)
  "recent": most notable recent international activity or development you know about
  "positioning": how they position vs TLD on price/quality/service in APAC
  "threat": "HIGH", "MEDIUM", or "LOW" for TLD's APAC business
  "confidence": "HIGH", "MEDIUM", or "LOW" — your confidence in this information

Return ONLY the JSON array. No markdown fences, no preamble, no explanation."""


def synthese_concurrents_deepseek():
    """Ask DeepSeek for an APAC competitor export brief using its training knowledge.

    Only runs on Mondays to avoid redundant daily calls — this kind of
    structural intelligence changes slowly. On other days returns an
    empty list.
    """
    if datetime.now().weekday() != 0:
        log.info("Competitor brief: skipping (runs Mondays only)")
        return []

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set — skipping competitor brief.")
        return []

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    log.info("Requesting APAC competitor intelligence brief from DeepSeek...")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": COMPETITOR_BRIEF_PROMPT}],
            max_tokens=2000,
            temperature=0.3,
        )

        text = response.choices[0].message.content or ""
        text = re.sub(r"```(?:json)?|```", "", text).strip()

        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not array_match:
            log.warning("Competitor brief: no JSON array found in response")
            log.debug(f"Raw response: {text[:500]}")
            return []

        competitors = json.loads(array_match.group(0))
        if not isinstance(competitors, list):
            log.warning("Competitor brief: response is not a JSON array")
            return []

        articles = []
        for c in competitors:
            company     = c.get("company_en") or c.get("company_cn", "Unknown")
            footprint   = c.get("footprint", "")
            recent      = c.get("recent", "")
            positioning = c.get("positioning", "")
            threat      = c.get("threat", "MEDIUM")
            confidence  = c.get("confidence", "MEDIUM")

            # Skip low-confidence entries to avoid hallucinated signal noise
            if confidence == "LOW":
                log.debug(f"Competitor brief: skipping {company} (low confidence)")
                continue

            desc = (
                f"APAC footprint: {footprint}. "
                f"Recent activity: {recent}. "
                f"Positioning vs TLD: {positioning}. "
                f"Threat level for TLD APAC: {threat}."
            )[:400]

            title = f"{company} — APAC competitor brief [{threat} threat to TLD]"

            articles.append({
                "source": "DeepSeek Competitor Intelligence",
                "titre":  title[:150],
                "lien":   "#competitor-brief",
                "desc":   desc,
                "date":   datetime.now().strftime("%Y-%m-%d"),
                "id": hashlib.md5(
                    (company + datetime.now().strftime("%Y-W%W")).encode()
                ).hexdigest(),
            })

        log.info(
            f"Competitor brief: {len(articles)} companies "
            f"(skipped low-confidence entries)"
        )
        return articles

    except json.JSONDecodeError as e:
        log.warning(f"Competitor brief: JSON parse error — {e}")
        return []
    except Exception as e:
        log.warning(f"Competitor brief failed: {e}")
        return []


# =============================================================================
#  DEEPSEEK — STRUCTURED PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a senior strategy analyst advising the CEO of TLD Group, a global GSE (Ground Support Equipment) manufacturer and lessor, on its Asia-Pacific business excluding mainland China: Southeast Asia, South Asia, Japan, Korea, Australia, New Zealand, and the Pacific.

Your role: translate raw news signals into actionable commercial and industrial intelligence for the APAC region.

ANALYSIS SCOPE — do not limit yourself to articles explicitly mentioning equipment:
- Airport openings, capacity expansions, traffic records in APAC → leading demand indicators (quantify where possible: +5% traffic ≈ +10 aircraft tractors per hub)
- Airline fleet orders, deliveries, financial results (regional carriers and Chinese carriers operating into APAC) → fleet-driven GSE procurement cycles
- Chinese GSE manufacturers' export and international expansion moves (Guangtai, CIMC Tianda, Jiangsu Tianyi, etc.) into APAC → direct competitive threats or market gaps for TLD
- Western competitor moves (JBT, Textron, Oshkosh, etc.) in APAC → threats or market gaps
- Raw material costs (steel, aluminium, lithium, semiconductors) → margin pressure
- Ground handler M&A (Swissport, Menzies, dnata, SATS) in APAC → contract consolidation risk
- Trade policy (tariffs, BRI, export controls) → supply chain and pricing implications in the region
- Environmental regulations at APAC airports → electrification timeline and diesel phase-out

IMPACT LEVELS:
- CRITICAL: Act within 48h (major competitor move, urgent tender, direct threat/opportunity)
- IMPORTANT: Act this week (significant market shift, pricing signal, client development)
- WATCH: Monitor closely, no immediate action (emerging trend, early-stage signal)
- INFO: Background context only

OUTPUT FORMAT — use EXACTLY this structure. Do not add any text outside the delimited blocks.

For each meaningful signal:
===SIGNAL_START===
SIGNAL_ID: [number]
IMPACT: [CRITICAL | IMPORTANT | WATCH | INFO]
HEADLINE: [One sharp sentence — max 15 words]
READING: [2-3 sentences: what happened and why it matters for the APAC GSE market]
BUSINESS_IMPACT: [2-3 sentences: concrete commercial/financial consequences for TLD — volumes, margins, contracts, competition]
ACTION: [1-2 sentences: specific recommended action, time-bound if possible]
===SIGNAL_END===

After ALL signals, output this closing block:
===SUMMARY_START===
EXECUTIVE_SUMMARY: [4-5 sentences for an executive committee. What happened, what it means, what we do.]
WATCH_1: [Key indicator #1 to monitor this week]
WATCH_2: [Key indicator #2 to monitor this week]
WATCH_3: [Key indicator #3 to monitor this week]
MAIN_RISK: [Single biggest risk for TLD GSE business in APAC this week — one sentence]
===SUMMARY_END===

Rules:
- English only
- Specific and quantitative when possible (volumes, %, timelines, EUR values)
- No bullet points inside field values — plain prose only
- Skip articles with no connection to the GSE market or its demand drivers
- Always output the SUMMARY block even if there are few signals
"""


def construire_prompt_user(articles):
    date_str = datetime.now().strftime("%d %B %Y")
    lines = [
        "GSE STRATEGIC WATCH — Asia-Pacific (excluding mainland China)",
        f"Date: {date_str}",
        f"Articles to analyze: {len(articles)}",
        "",
    ]
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] SOURCE: {a['source']}")
        lines.append(f"    TITLE: {a['titre']}")
        lines.append(f"    URL: {a['lien']}")
        if a.get("desc"):
            lines.append(f"    EXCERPT: {a['desc'][:300]}")
        lines.append("")

    lines.append(
        "Analyze each article for signals relevant to TLD Group's APAC GSE business. "
        "Output ONLY the structured blocks defined in your instructions."
    )
    lines.append("")
    lines.append(
        "CRITICAL RULE: If ANY article mentions a GSE manufacturer or competitor "
        "by name — including Guangtai (广泰/威海广泰), JBT, Textron, Oshkosh, "
        "CIMC Tianda (中集天达), Alvest, TLD, or any other GSE brand — you MUST "
        "generate a signal for it, even if the mention is brief. Never omit a "
        "competitor signal. Rate it CRITICAL if it involves a product launch, "
        "contract win, or pricing move; IMPORTANT for M&A or strategic announcements; "
        "WATCH for general company news."
    )
    return "\n".join(lines)


def analyser_avec_deepseek(articles):
    if not articles:
        log.info("No articles to analyze.")
        return ""

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    batch = articles[:DEEPSEEK_MAX_ARTICLES]
    if len(articles) > DEEPSEEK_MAX_ARTICLES:
        log.warning(
            f"Capped input at {DEEPSEEK_MAX_ARTICLES} articles "
            f"(had {len(articles)})."
        )

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    log.info(f"Sending {len(batch)} articles to DeepSeek...")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": construire_prompt_user(batch)},
            ],
            max_tokens=8192,
            temperature=0.2,
        )
        raw = response.choices[0].message.content
        log.info(f"DeepSeek response: {len(raw)} chars")
        return raw
    except Exception as e:
        log.error(f"DeepSeek API error: {e}")
        return ""


# =============================================================================
#  PARSER — delimiter-based, with truncation detection
# =============================================================================

def extract_field(block, field):
    """Extract a named field value from a delimited block."""
    pattern = rf"^{field}:\s*(.+?)(?=\n[A-Z_]{{2,}}:|$)"
    match = re.search(pattern, block, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parser_analyse(raw_text):
    signals = []
    summary = {"executive_summary": "", "watch": [], "main_risk": ""}

    if not raw_text:
        log.warning("Empty DeepSeek response — nothing to parse.")
        return signals, summary

    n_starts = raw_text.count("===SIGNAL_START===")
    n_ends   = raw_text.count("===SIGNAL_END===")
    has_summary = "===SUMMARY_START===" in raw_text

    if n_starts != n_ends:
        log.warning(
            f"TRUNCATION DETECTED: {n_starts} signal starts but only "
            f"{n_ends} ends. Raise max_tokens or reduce input."
        )
    if n_starts > 0 and not has_summary:
        log.warning(
            "TRUNCATION DETECTED: signals found but no SUMMARY block. "
            "Output was cut before the end."
        )

    for block in re.findall(
        r"===SIGNAL_START===(.*?)===SIGNAL_END===", raw_text, re.DOTALL
    ):
        impact = extract_field(block, "IMPACT").upper() or "INFO"
        if impact not in ("CRITICAL", "IMPORTANT", "WATCH", "INFO"):
            impact = "INFO"
        signals.append({
            "id":              extract_field(block, "SIGNAL_ID"),
            "impact":          impact,
            "headline":        extract_field(block, "HEADLINE"),
            "reading":         extract_field(block, "READING"),
            "business_impact": extract_field(block, "BUSINESS_IMPACT"),
            "action":          extract_field(block, "ACTION"),
        })

    sm = re.search(
        r"===SUMMARY_START===(.*?)===SUMMARY_END===", raw_text, re.DOTALL
    )
    if sm:
        b = sm.group(1)
        summary["executive_summary"] = extract_field(b, "EXECUTIVE_SUMMARY")
        summary["main_risk"]         = extract_field(b, "MAIN_RISK")
        summary["watch"] = [
            extract_field(b, f"WATCH_{i}")
            for i in range(1, 4)
            if extract_field(b, f"WATCH_{i}")
        ]

    log.info(
        f"Parsed: {len(signals)} signals, "
        f"summary={'yes' if summary['executive_summary'] else 'NO'}"
    )
    return signals, summary


# =============================================================================
#  HTML REPORT
# =============================================================================

IMPACT_CONFIG = {
    "CRITICAL": {
        "label": "Critical",
        "color": "#dc2626",
        "bg": "#fef2f2",
        "border": "#fecaca",
        "text": "#991b1b",
    },
    "IMPORTANT": {
        "label": "Important",
        "color": "#d97706",
        "bg": "#fffbeb",
        "border": "#fde68a",
        "text": "#92400e",
    },
    "WATCH": {
        "label": "Watch",
        "color": "#0369a1",
        "bg": "#f0f9ff",
        "border": "#bae6fd",
        "text": "#0c4a6e",
    },
    "INFO": {
        "label": "Info",
        "color": "#6b7280",
        "bg": "#f9fafb",
        "border": "#e5e7eb",
        "text": "#374151",
    },
}


def md(text):
    """Convert markdown to HTML; strip single wrapping <p> for inline use."""
    if not text:
        return ""
    html = markdown.markdown(text.strip(), extensions=["nl2br"])
    if html.count("<p>") == 1:
        html = re.sub(r"^<p>(.*)</p>$", r"\1", html, flags=re.DOTALL)
    return html


def trouver_article(sig, articles):
    """Match a signal to its most likely source article by keyword overlap."""
    haystack = (
        sig.get("headline", "") + " " +
        sig.get("reading", "")
    ).lower()

    best_article = None
    best_score   = 0

    for a in articles:
        candidate = (a["titre"] + " " + a.get("desc", "")).lower()
        words = [w for w in re.split(r"[\s\W]+", candidate) if len(w) >= 3]
        score = sum(1 for w in words if w in haystack)
        if score > best_score:
            best_score   = score
            best_article = a

    return best_article if best_score >= 1 else None


def render_signal_card(sig, articles):
    """Build HTML for one signal card."""
    cfg = IMPACT_CONFIG.get(sig["impact"], IMPACT_CONFIG["INFO"])

    article      = trouver_article(sig, articles)
    source_block = ""
    if article:
        titre_esc = (
            article["titre"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        source_block = (
            f'<div class="signal-source">'
            f'<span class="source-label">Source</span>'
            f'<a href="{article.get("lien","#")}" target="_blank" rel="noopener">'
            f"{titre_esc}</a>"
            f'<span class="source-name"> — {article["source"]}</span>'
            f"</div>"
        )

    return f"""
<div class="signal-card impact-{sig['impact'].lower()}">
  <div class="signal-card-header" style="border-left:4px solid {cfg['color']};">
    <span class="signal-badge"
          style="background:{cfg['bg']};color:{cfg['text']};border:1px solid {cfg['border']};">
      {cfg['label']}
    </span>
    <h3 class="signal-headline">{md(sig['headline'])}</h3>
  </div>
  <div class="signal-body">
    <div class="signal-section">
      <div class="signal-section-label">Reading</div>
      <div class="signal-section-text">{md(sig['reading'])}</div>
    </div>
    <div class="signal-section">
      <div class="signal-section-label">Business impact</div>
      <div class="signal-section-text">{md(sig['business_impact'])}</div>
    </div>
    <div class="signal-section signal-action">
      <div class="signal-section-label">Recommended action</div>
      <div class="signal-section-text">{md(sig['action'])}</div>
    </div>
    {source_block}
  </div>
</div>"""


def generer_rapport(articles, signals, summary, raw_text="", truncated=False):
    now_full = datetime.now().strftime("%B %d, %Y")
    now_time = datetime.now().strftime("%H:%M")

    counts = {"CRITICAL": 0, "IMPORTANT": 0, "WATCH": 0, "INFO": 0}
    for s in signals:
        counts[s["impact"]] = counts.get(s["impact"], 0) + 1

    actionable = [s for s in signals if s["impact"] in ("CRITICAL", "IMPORTANT", "WATCH")]
    background = [s for s in signals if s["impact"] == "INFO"]

    signals_html = ""
    if not actionable and not background:
        signals_html = (
            '<p style="color:#6b7280;font-style:italic;padding:24px 0;">'
            "No significant signals identified today.</p>"
        )
    else:
        for sig in actionable:
            signals_html += render_signal_card(sig, articles)

        if background:
            info_items = "".join(
                f'<li style="font-size:13px;color:#64748b;padding:3px 0;'
                f'line-height:1.5;">{sig["headline"]}</li>'
                for sig in background
            )
            signals_html += f"""
<details style="margin-top:12px;">
  <summary style="font-size:12px;color:#94a3b8;cursor:pointer;
                  padding:8px 4px;user-select:none;list-style:none;
                  display:flex;align-items:center;gap:6px;">
    <span style="font-size:10px;background:#f1f5f9;border:1px solid #e2e8f0;
                 border-radius:20px;padding:2px 8px;color:#64748b;font-weight:600;
                 letter-spacing:.04em;">
      + {len(background)} background item{"s" if len(background) != 1 else ""}
    </span>
    <span style="color:#94a3b8;">— no action required, click to expand</span>
  </summary>
  <ul style="list-style:none;padding:12px 16px;margin-top:8px;
             background:#f8fafc;border:1px solid #e2e8f0;
             border-radius:8px;">
    {info_items}
  </ul>
</details>"""

    counter_html = "".join(
        f'<span class="counter-pill" '
        f'style="background:{IMPACT_CONFIG[lvl]["bg"]};'
        f'color:{IMPACT_CONFIG[lvl]["text"]};'
        f'border:1px solid {IMPACT_CONFIG[lvl]["border"]};">'
        f'{counts[lvl]} {IMPACT_CONFIG[lvl]["label"]}</span>'
        for lvl in ("CRITICAL", "IMPORTANT", "WATCH", "INFO")
        if counts[lvl] > 0
    )

    watch_html = "".join(f"<li>{md(w)}</li>" for w in summary.get("watch", []))
    exec_html  = md(summary.get("executive_summary", ""))
    risk_html  = md(summary.get("main_risk", ""))

    trunc_banner = ""
    if truncated:
        trunc_banner = """
<div style="background:#fef9c3;border:1px solid #fde047;border-radius:8px;
            padding:12px 16px;margin-bottom:24px;font-size:13px;color:#713f12;">
  <strong>Warning:</strong> DeepSeek output was truncated — some signals or the
  summary block may be missing. Raise <code>max_tokens</code> or reduce the
  number of input articles.
</div>"""

    sources_list = "".join(f"<li>{s['nom']}</li>" for s in SOURCES)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APAC GSE Intelligence Brief — {now_full}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --ink:#0f172a;--ink-2:#334155;--ink-3:#64748b;--ink-4:#94a3b8;
  --surface:#ffffff;--surface-1:#f8fafc;--border:#e2e8f0;
  --radius:8px;--radius-lg:12px;
}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
      background:#f0f2f5;color:var(--ink);line-height:1.6;padding:32px 16px 64px}}
.wrapper{{max-width:900px;margin:0 auto}}

/* ── MASTHEAD ── */
.masthead{{background:var(--ink);border-radius:var(--radius-lg) var(--radius-lg) 0 0;
           padding:28px 36px 24px}}
.masthead-eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:10px;
                   letter-spacing:.12em;text-transform:uppercase;
                   color:#64748b;margin-bottom:8px}}
.masthead-title{{font-size:22px;font-weight:600;letter-spacing:-.02em;
                 color:#fff;margin-bottom:12px}}
.masthead-meta{{display:flex;align-items:center;gap:20px;flex-wrap:wrap}}
.meta-item{{font-size:13px;color:#94a3b8;display:flex;align-items:center;gap:6px}}
.meta-item strong{{color:#e2e8f0;font-weight:500}}
.masthead-counters{{display:flex;gap:8px;flex-wrap:wrap;
                    margin-top:16px;padding-top:16px;border-top:1px solid #1e293b}}
.counter-pill{{font-size:11px;font-weight:500;padding:3px 10px;
               border-radius:20px;letter-spacing:.02em}}

/* ── CARD BODY ── */
.card-body{{background:var(--surface);border:1px solid var(--border);
            border-top:none;border-radius:0 0 var(--radius-lg) var(--radius-lg);
            padding:36px}}
.section-header{{display:flex;align-items:center;gap:10px;
                 margin-bottom:20px;padding-bottom:12px;
                 border-bottom:1px solid var(--border)}}
.section-header h2{{font-size:13px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.08em;color:var(--ink-3)}}
.section-divider{{margin:36px 0;border:none;border-top:1px solid var(--border)}}

/* ── EXEC SUMMARY ── */
.exec-panel{{background:var(--ink);border-radius:var(--radius-lg);
             padding:24px 28px;margin-bottom:32px;
             color:#e2e8f0;font-size:15px;line-height:1.75}}
.exec-panel-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;
                   letter-spacing:.1em;text-transform:uppercase;
                   color:#475569;margin-bottom:10px}}
.exec-panel p{{margin:0}}

/* ── SIGNAL CARDS ── */
.signal-card{{border:1px solid var(--border);border-radius:var(--radius-lg);
              margin-bottom:16px;overflow:hidden;
              transition:box-shadow .15s}}
.signal-card:hover{{box-shadow:0 4px 12px rgba(0,0,0,.06)}}
.signal-card-header{{padding:16px 20px;background:var(--surface-1);
                     display:flex;align-items:flex-start;gap:12px}}
.signal-badge{{font-size:11px;font-weight:600;padding:3px 9px;
               border-radius:20px;white-space:nowrap;
               letter-spacing:.03em;margin-top:2px;flex-shrink:0}}
.signal-headline{{font-size:15px;font-weight:600;
                  color:var(--ink);line-height:1.4}}
.signal-headline p{{margin:0}}
.signal-body{{padding:20px;display:grid;gap:16px}}
.signal-section-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                        letter-spacing:.1em;color:var(--ink-4);margin-bottom:4px}}
.signal-section-text{{font-size:14px;color:var(--ink-2);line-height:1.65}}
.signal-section-text p{{margin:0}}
.signal-action .signal-section-text{{color:var(--ink);font-weight:500}}
.signal-source{{padding-top:12px;border-top:1px dashed var(--border);
                font-size:12px;color:var(--ink-4);
                display:flex;flex-wrap:wrap;gap:4px;align-items:center}}
.source-label{{font-weight:600;text-transform:uppercase;letter-spacing:.06em;
               font-size:10px;color:var(--ink-4);margin-right:4px}}
.signal-source a{{color:#2563eb;text-decoration:none;font-weight:500}}
.signal-source a:hover{{text-decoration:underline}}
.source-name{{color:var(--ink-4)}}

/* ── WATCH / RISK ── */
.watch-panel{{background:#fffbeb;border:1px solid #fde68a;
              border-radius:var(--radius-lg);padding:20px 24px;margin-bottom:16px}}
.watch-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                    letter-spacing:.08em;color:#92400e;margin-bottom:12px}}
.watch-panel ol{{padding-left:20px;display:grid;gap:6px}}
.watch-panel li{{font-size:14px;color:#78350f;line-height:1.5}}
.watch-panel li p{{margin:0}}
.risk-panel{{background:#fef2f2;border:1px solid #fecaca;
             border-radius:var(--radius-lg);padding:20px 24px}}
.risk-panel-label{{font-size:11px;font-weight:600;text-transform:uppercase;
                   letter-spacing:.08em;color:#991b1b;margin-bottom:8px}}
.risk-panel-text{{font-size:14px;color:#7f1d1d;line-height:1.6;font-weight:500}}
.risk-panel-text p{{margin:0}}

/* ── SOURCES ── */
.sources-panel{{background:var(--surface-1);border:1px solid var(--border);
                border-radius:var(--radius);padding:16px 20px;margin-top:36px}}
.sources-panel-label{{font-size:10px;font-weight:600;text-transform:uppercase;
                       letter-spacing:.1em;color:var(--ink-4);margin-bottom:10px}}
.sources-panel ul{{list-style:none;display:flex;flex-wrap:wrap;
                   gap:6px 0;column-gap:24px;columns:2}}
.sources-panel li{{font-size:12px;color:var(--ink-3);break-inside:avoid}}
.sources-panel li::before{{content:"·";margin-right:6px;color:var(--ink-4)}}

/* ── FOOTER ── */
.page-footer{{text-align:center;font-size:11px;color:var(--ink-4);
              margin-top:24px;font-family:'IBM Plex Mono',monospace;
              letter-spacing:.04em}}

@media(max-width:600px){{
  body{{padding:12px 8px 48px}}
  .masthead,.card-body{{padding:20px 16px}}
  .sources-panel ul{{columns:1}}
}}
</style>
</head>
<body>
<div class="wrapper">

<div class="masthead">
  <div class="masthead-eyebrow">TLD Group · Market Intelligence</div>
  <div class="masthead-title">APAC GSE Intelligence Brief</div>
  <div class="masthead-meta">
    <div class="meta-item"><span>Date</span><strong>{now_full}</strong></div>
    <div class="meta-item"><span>Generated</span><strong>{now_time}</strong></div>
    <div class="meta-item"><span>Articles analyzed</span><strong>{len(articles)}</strong></div>
    <div class="meta-item"><span>Signals</span><strong>{len(signals)}</strong></div>
  </div>
  {f'<div class="masthead-counters">{counter_html}</div>' if counter_html else ''}
</div>

<div class="card-body">

  {trunc_banner}

  {f'<div class="exec-panel"><div class="exec-panel-label">Executive summary</div>{exec_html}</div>' if exec_html else ''}

  <div class="section-header"><h2>Signals</h2></div>
  {signals_html}

  {'<hr class="section-divider">' if watch_html or risk_html else ''}

  {f'<div class="section-header"><h2>To watch this week</h2></div><div class="watch-panel"><div class="watch-panel-label">Key indicators</div><ol>{watch_html}</ol></div>' if watch_html else ''}

  {f'<div class="risk-panel"><div class="risk-panel-label">Main risk</div><div class="risk-panel-text">{risk_html}</div></div>' if risk_html else ''}

  <div class="sources-panel">
    <div class="sources-panel-label">Monitored sources</div>
    <ul>{sources_list}</ul>
  </div>

</div>

<div class="page-footer">
  APAC GSE Intelligence Agent v2.0 · Powered by DeepSeek + Tavily · {now_full}
</div>

</div>
</body>
</html>"""


# =============================================================================
#  SAVE
# =============================================================================

def sauvegarder_rapport(rapport_html):
    dossier = Path("rapports_apac")
    dossier.mkdir(exist_ok=True, parents=True)
    fichier = dossier / f"apac_veille_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    with open(fichier, "w", encoding="utf-8") as f:
        f.write(rapport_html)
    log.info(f"Report saved: {fichier.absolute()}")
    return fichier


# =============================================================================
#  MAIN
# =============================================================================

def executer_agent():
    log.info("=" * 60)
    log.info("Starting APAC GSE Intelligence Agent v2.0")
    log.info("=" * 60)
    try:
        vus = charger_vus()

        # 1. Collect from scrapers
        tous_articles = collecter_tous_articles()

        # 2. Tavily search — real-time web search for APAC GSE/competitor news
        tavily_articles = rechercher_tavily()
        if tavily_articles:
            tous_articles.extend(tavily_articles)
            log.info(f"Total after Tavily: {len(tous_articles)} articles")

        # 3. DeepSeek competitor export brief (runs Mondays only)
        competitor_articles = synthese_concurrents_deepseek()
        if competitor_articles:
            tous_articles.extend(competitor_articles)
            log.info(f"Total after competitor brief: {len(tous_articles)} articles")

        # 4. Filter
        articles_pertinents = filtrer_pertinents(tous_articles, vus)

        # 5. Prioritize Tavily/competitor articles so they survive the cap
        def source_priority(a):
            if a["source"] == "Tavily Search":
                return 0
            if a["source"] == "DeepSeek Competitor Intelligence":
                return 1
            return 2

        articles_pertinents.sort(key=source_priority)
        log.info(
            f"After prioritization: "
            f"{sum(1 for a in articles_pertinents if a['source'] == 'Tavily Search')} Tavily, "
            f"{sum(1 for a in articles_pertinents if a['source'] == 'DeepSeek Competitor Intelligence')} competitor, "
            f"{sum(1 for a in articles_pertinents if a['source'] not in ('Tavily Search', 'DeepSeek Competitor Intelligence'))} scraped"
        )

        # 6. Enrich scraped articles with body excerpts
        if articles_pertinents:
            articles_pertinents = enrichir_articles(articles_pertinents)

        # 7. Analyze with DeepSeek
        raw_analyse = (
            analyser_avec_deepseek(articles_pertinents)
            if articles_pertinents
            else ""
        )

        # 8. Save raw output for debugging
        Path("rapports_apac").mkdir(exist_ok=True, parents=True)
        Path("rapports_apac/debug_raw.txt").write_text(
            raw_analyse or "", encoding="utf-8"
        )
        log.info("Raw DeepSeek output saved to rapports_apac/debug_raw.txt")

        # 9. Detect truncation
        n_starts  = raw_analyse.count("===SIGNAL_START===")
        n_ends    = raw_analyse.count("===SIGNAL_END===")
        has_sum   = "===SUMMARY_START===" in raw_analyse
        truncated = (n_starts != n_ends) or (n_starts > 0 and not has_sum)

        # 10. Parse
        signals, summary = parser_analyse(raw_analyse)

        # 11. Generate report
        rapport_html = generer_rapport(
            articles_pertinents, signals, summary,
            raw_text=raw_analyse, truncated=truncated
        )

        # 12. Save
        fichier = sauvegarder_rapport(rapport_html)
        print(f"✅ Report generated: {fichier}")

        # 13. Mark articles as seen
        for a in articles_pertinents:
            vus.add(a["id"])
        sauvegarder_vus(vus)
        log.info("Done.")

    except Exception as e:
        log.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    executer_agent()
