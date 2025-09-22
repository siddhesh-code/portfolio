import os
import re
import html
import logging
from typing import List, Dict, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed


# --- CONFIG ---
WATCH_SOURCES = {
    "Defence Tenders": {
        "url": "https://defproc.gov.in/nicgep/app?page=FrontEndLatestActiveTenders&service=page",
        "type": "table",
        "selector": "table.list_table, table#resultTable, table#activeTenders",
        "limit": 5,
        "keywords": ["tender", "bid", "procurement", "rfp"],
        "description": "Fresh procurement notices from MoD e-procurement portal",
        "fallback_message": "Automated scrape blocked by portal captcha; check the Active Tenders page manually.",
    },
    "DRDO News": {
        "url": "https://www.drdo.gov.in/whats-new",
        "type": "list",
        "selector": "ul.commonListing li",
        "limit": 5,
        "description": "Latest press releases and project updates from DRDO",
    },
    "MNRE": {
        "url": "https://mnre.gov.in/notifications",
        "type": "table",
        "selector": "table",
        "limit": 5,
        "keywords": ["tender", "notice", "order", "policy", "guidelines"],
        "description": "Recent policy circulars and notices from MNRE",
    },
    "SECI": {
        "url": "https://seci.co.in/",
        "type": "cards",
        "selector": "div#latest-annoucements div.latest-item",
        "limit": 5,
        "keywords": ["tender", "rfp", "notice", "auction", "result"],
        "description": "SECI tenders, auction results, and corporate announcements",
        "other_limit": 3,
    },
    "Moneycontrol Companies": {
        "url": "https://www.moneycontrol.com/news/business/companies/",
        "type": "cards",
        "selector": "li.clearfix",
        "title_selector": "h2 a",
        "extra_selector": "p",
        "limit": 8,
        "description": "Top corporate stories from Moneycontrol (filtered via focus keywords)",
        "other_limit": 2,
    },
    "SolarQuarter India": {
        "url": "https://solarquarter.com/category/india/",
        "type": "cards",
        "selector": "div.td_module_trending_now, div.td-block-span6",
        "title_selector": "h3.entry-title a",
        "limit": 6,
        "description": "Renewable energy project announcements curated by SolarQuarter",
        "other_limit": 2,
        "allow_404": True,
    },
}


NOISE_EXACT = {
    "screen reader access",
    "feedback",
    "sitemap",
    "faqs",
    "login",
    "skip to main content",
    "statutory information",
    "form iv & form v",
    "license information",
    "link to old website",
    "vigilance",
    "accessibility tools",
    "high contrast",
    "normal contrast",
    "highlight links",
    "invert",
    "saturation",
    "font size increase",
    "font size decrease",
    "normal font",
    "text spacing",
    "line height",
    "last updated",
    "visitors",
    "screen reader",
    "facebook",
    "twitter",
    "instagram",
    "youtube",
    "linkedin",
    "search",
    "tender id",
    "tender title",
    "enter captcha",
    "refresh",
    "clear",
}

NOISE_CONTAINS = [
    "screen reader",
    "select",
    "accessibility",
    "form iv",
    "form v",
    "statutory",
    "license information",
    "link to old website",
    "vigilance",
    "visitors",
    "last updated",
    "toggle",
    "menu",
    "login",
    "feedback",
    "sitemap",
    "tender id",
    "tender title",
    "enter captcha",
    "latest announcements",
    "provide captcha",
    "active tenders",
    "tenders by closing date",
    "results of tenders",
    "tenders by location",
    "tenders in archive",
    "tenders status",
    "cancelled/retendered",
]


DEFAULT_FOCUS_KEYWORDS = [
    # Listed defence & aerospace primes
    "Hindustan Aeronautics",
    "HAL",
    "Bharat Electronics",
    "BEL",
    "Bharat Dynamics",
    "BDL",
    "BEML",
    "Mazagon Dock",
    "MDL",
    "Garden Reach Shipbuilders",
    "GRSE",
    "Cochin Shipyard",
    "CSL",
    "Paras Defence",
    "Data Patterns",
    "Astra Microwave",
    "Zen Technologies",
    "Bharat Forge",
    "Larsen & Toubro",
    "L&T",

    # Supply-chain & electronics names you track
    "DCX India",
    "DCXINDIA",
    "MTAR Technologies",
    "MTAR",
    "Shivalik Bimetal",
    "SBCL",
    "TD Power Systems",
    "TD Power",
    "IdeaForge",
    "Paras Aerospace",

    # Energy & utilities benefiting from MNRE / SECI policy
    "NTPC",
    "NTPC Green",
    "NHPC",
    "Power Grid",
    "Adani Green",
    "Adani Energy",
    "Reliance",
    "Torrent Power",
    "JSW Energy",
    "Tata Power",
    "Tata Projects",
    "ReNew",

    # Water / infra (matches WABAG etc.)
    "VA Tech Wabag",
    "WABAG",
    "IRCON",
    "RVNL",
    "NCC",

    # Technology & drone ecosystem
    "Cyient",
    "Tata Advanced Systems",
    "Mahindra Defence",
    "Samteck",

    # Frontier themes
    "Green Hydrogen",
    "Hydrogen",
    "Solar",
    "Wind",
    "Offshore Wind",
]


def _split_keywords(value: str) -> List[str]:
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def _normalize_keyword(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _build_keyword_map(keywords: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for kw in keywords:
        normalized = _normalize_keyword(kw)
        if not normalized:
            continue
        mapping.setdefault(normalized, kw)
    return mapping


def _load_focus_keywords() -> List[str]:
    env = os.getenv("DEFENCE_FOCUS_KEYWORDS", "")
    if env:
        return _split_keywords(env)
    return DEFAULT_FOCUS_KEYWORDS


FOCUS_KEYWORDS = _load_focus_keywords()
FOCUS_KEYWORD_MAP = _build_keyword_map(FOCUS_KEYWORDS)


ENTITY_PATTERN = re.compile(
    r"\b([A-Z][\w&./-]*(?:\s+(?:[A-Z][\w&./-]*|&|of|and))*\s+(?:Limited|Ltd|Corporation|Company|Enterprises|Industries|Systems|Technologies|Engineering|Shipyard|Shipbuilders))\b",
    re.IGNORECASE,
)
ABBR_PATTERN = re.compile(r"\b[A-Z]{3,6}\b")
STOP_ABBREVIATIONS = {
    "RFP",
    "TDF",
    "DRDO",
    "MNRE",
    "SECI",
    "RTSPV",
    "RESCO",
    "NGHM",
    "GOV",
    "GOVT",
    "UT",
    "PSU",
    "MTS",
    "SSPL",
    "NPOL",
    "ITR",
    "PDF",
    "III",
    "II",
    "IV",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
}


def _env_flag(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


EMAIL_ALERTS = _env_flag("DEFENCE_EMAIL_ALERTS", "1")
EMAIL_FROM = os.getenv("DEFENCE_EMAIL_FROM") or os.getenv("MAIL_FROM") or os.getenv("SMTP_USER", "")
EMAIL_TO = (
    os.getenv("DEFENCE_EMAIL_TO")
    or os.getenv("MAIL_TO")
    or EMAIL_FROM
)
EMAIL_PASS = os.getenv("DEFENCE_EMAIL_PASS") or os.getenv("SMTP_PASS", "")
SMTP_SERVER = os.getenv("DEFENCE_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("DEFENCE_SMTP_PORT", "465"))
USE_SSL = _env_flag("DEFENCE_SMTP_SSL", str(SMTP_PORT == 465))
USE_TLS = _env_flag("DEFENCE_SMTP_TLS", str(SMTP_PORT in {587, 25})) if not USE_SSL else False
EMAIL_SUBJECT = os.getenv("DEFENCE_EMAIL_SUBJECT", "Defence & Policy Feed Update")


LOG_LEVEL = os.getenv("DEFENCE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        read=3,
        connect=3,
        backoff_factor=1.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; DefenceWatch/1.0; +https://github.com/siddheshdhoot)",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


SESSION = _build_session()


def clean_text(text: str) -> str:
    return " ".join(text.split()) if text else ""


def is_noise_text(text: str) -> bool:
    normalized = clean_text(text).lower()
    if not normalized:
        return True
    if normalized in NOISE_EXACT:
        return True
    for phrase in NOISE_CONTAINS:
        if phrase in normalized:
            return True
    return False


def find_focus_matches(text: str, keyword_map: Dict[str, str]) -> List[str]:
    normalized_text = _normalize_keyword(text)
    if not normalized_text:
        return []
    matches: List[str] = []
    for norm_kw, original in keyword_map.items():
        if norm_kw and norm_kw in normalized_text:
            matches.append(original)
    return matches


def extract_entities(text: str) -> List[str]:
    entities = set()
    for match in ENTITY_PATTERN.findall(text):
        cleaned = clean_text(match)
        if cleaned:
            entities.add(cleaned)
    for abbr in ABBR_PATTERN.findall(text):
        upper = abbr.upper()
        if upper in STOP_ABBREVIATIONS:
            continue
        entities.add(upper)
    return sorted(entities, key=lambda s: s.lower())


def build_items_table(soup: BeautifulSoup, meta: Dict[str, str], limit: int, base_url: str) -> List[Dict[str, str]]:
    selector = meta.get("selector", "table")
    table = soup.select_one(selector)
    if not table:
        return []

    items: List[Dict[str, str]] = []
    headers: List[str] = []
    for row in table.find_all("tr"):
        header_cells = row.find_all("th")
        if header_cells:
            headers = [clean_text(th.get_text(" ", strip=True)) for th in header_cells]
            continue

        cells = row.find_all("td")
        if not cells:
            continue
        texts = []
        link: Optional[str] = None
        title: Optional[str] = None
        extra_parts: List[str] = []
        for cell in cells:
            text = clean_text(cell.get_text(" ", strip=True))
            if text:
                texts.append(text)
            if not link:
                anchor = cell.find("a", href=True)
                if anchor:
                    link = urljoin(base_url, anchor["href"])
        if not texts:
            continue
        values = texts
        if headers and len(headers) == len(values):
            for key, value in zip(headers, values):
                if not value:
                    continue
                key_lower = key.lower()
                if title is None and any(term in key_lower for term in ["title", "subject", "tender", "description", "name"]):
                    title = value
                elif any(term in key_lower for term in ["closing", "deadline", "submission", "bid", "opening"]):
                    extra_parts.append(f"{key}: {value}")
                elif len(extra_parts) < 2:
                    extra_parts.append(f"{key}: {value}")
        if not title:
            title = values[0] if values else "(no title)"
            extra_parts = [val for val in values[1:3] if val]
        if is_noise_text(title):
            continue
        item: Dict[str, str] = {"title": title, "link": link or base_url}
        if extra_parts:
            item["extra"] = " | ".join(extra_parts)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def build_items_cards(soup: BeautifulSoup, meta: Dict[str, str], limit: int, base_url: str) -> List[Dict[str, str]]:
    selector = meta.get("selector", "div")
    nodes = soup.select(selector)
    title_selector = meta.get("title_selector")
    extra_selector = meta.get("extra_selector")
    date_selector = meta.get("date_selector")
    items: List[Dict[str, str]] = []
    for node in nodes:
        anchor = None
        title_tag = None
        if title_selector:
            title_tag = node.select_one(title_selector)
            if title_tag and title_tag.name == "a" and title_tag.has_attr("href"):
                anchor = title_tag
        if anchor is None:
            anchor = node.find("a", href=True)
        if title_tag is None:
            title_tag = node.find(["h2", "h3", "h4", "h5"]) or node.find(class_=lambda x: x and "title" in x.lower())

        if title_tag:
            title = clean_text(title_tag.get_text(" ", strip=True))
        elif anchor:
            title = clean_text(anchor.get_text(" ", strip=True))
        else:
            title = clean_text(node.get_text(" ", strip=True))
        if not title:
            continue
        if is_noise_text(title):
            continue
        link = urljoin(base_url, anchor["href"]) if anchor else base_url
        summary_node = None
        if extra_selector:
            summary_node = node.select_one(extra_selector)
        if summary_node is None:
            summary_node = node.find("p") or node.find("div", class_="field-content")
        summary = clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""
        date_node = None
        if date_selector:
            date_node = node.select_one(date_selector)
        if date_node is None:
            date_node = node.find("time") or node.find("span", class_=lambda x: x and "date" in x.lower())
        date_text = clean_text(date_node.get_text(" ", strip=True)) if date_node else ""
        item: Dict[str, str] = {"title": title, "link": link}
        extra_parts: List[str] = []
        if date_text:
            extra_parts.append(date_text)
        if summary and summary.lower() not in title.lower():
            extra_parts.append(summary)
        if extra_parts:
            item["extra"] = " | ".join(extra_parts)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def build_items_list(soup: BeautifulSoup, meta: Dict[str, str], limit: int, base_url: str) -> List[Dict[str, str]]:
    selector = meta.get("selector", "li")
    nodes = soup.select(selector)
    title_selector = meta.get("title_selector")
    extra_selector = meta.get("extra_selector")
    date_selector = meta.get("date_selector")
    items: List[Dict[str, str]] = []
    for node in nodes:
        anchor = node.find("a", href=True)
        if title_selector:
            title_node = node.select_one(title_selector)
            if title_node:
                text = clean_text(title_node.get_text(" ", strip=True))
                if title_node.name == "a" and title_node.has_attr("href"):
                    anchor = title_node
            else:
                text = clean_text(anchor.get_text(" ", strip=True) if anchor else node.get_text(" ", strip=True))
        else:
            text = clean_text(anchor.get_text(" ", strip=True) if anchor else node.get_text(" ", strip=True))
        if not text:
            continue
        if is_noise_text(text):
            continue
        link = urljoin(base_url, anchor["href"]) if anchor else base_url
        date_node = None
        if date_selector:
            date_node = node.select_one(date_selector)
        if date_node is None:
            date_node = node.find("time") or node.find("span", class_=lambda x: x and "date" in x.lower())
        extra_node = None
        if extra_selector:
            extra_node = node.select_one(extra_selector)
        if extra_node is None:
            extra_node = node.find("span") or node.find("small")
        extras: List[str] = []
        if date_node:
            date_text = clean_text(date_node.get_text(" ", strip=True))
            if date_text:
                extras.append(date_text)
        if extra_node:
            extra_text = clean_text(extra_node.get_text(" ", strip=True))
            if extra_text and extra_text.lower() not in text.lower():
                extras.append(extra_text)
        item: Dict[str, str] = {"title": text, "link": link}
        if extras:
            item["extra"] = " | ".join(extras)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def build_items_generic(soup: BeautifulSoup, limit: int, base_url: str, keywords: Optional[List[str]] = None) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        text = clean_text(anchor.get_text(" ", strip=True))
        if len(text) < 5:
            continue
        if is_noise_text(text):
            continue
        if keywords and not any(kw.lower() in text.lower() for kw in keywords):
            continue
        if not keywords:
            nav_words = {"home", "login", "sitemap", "accessibility", "faq", "contact", "screen reader"}
            if text.lower() in nav_words:
                continue
        slug = text.lower()
        if slug in seen:
            continue
        seen.add(slug)
        items.append({"title": text, "link": urljoin(base_url, anchor["href"])})
        if len(items) >= limit:
            break
    return items


def format_items(
    name: str,
    base_url: str,
    items: List[Dict[str, str]],
    description: Optional[str] = None,
    focus_map: Optional[Dict[str, str]] = None,
    other_limit: int = 2,
) -> (str, List[Dict[str, str]], str):
    if not items:
        plain = f"[{name}] {base_url}\n(no highlights parsed)"
        html_fragment = (
            f"<h3><a href='{html.escape(base_url)}'>{html.escape(name)}</a></h3>"
            "<p>(no highlights parsed)</p>"
        )
        return plain, [], html_fragment

    lines = [f"[{name}] {base_url}"]
    if description:
        lines.append(f"  {description}")
    focus_map = focus_map or FOCUS_KEYWORD_MAP

    matched_lines: List[str] = []
    other_lines: List[str] = []
    matched_meta: List[Dict[str, str]] = []
    matched_html: List[str] = []
    other_html: List[str] = []

    for item in items:
        title = item.get("title", "(no title)")
        extra = item.get("extra")
        link = item.get("link")
        composite_text = f"{title} {extra}" if extra else title
        keyword_hits = find_focus_matches(composite_text, focus_map)
        entity_hits = extract_entities(composite_text)
        highlights: List[str] = []
        seen = set()
        for label in keyword_hits + entity_hits:
            norm = clean_text(label).lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            highlights.append(label)

        prefix = ""
        if highlights:
            prefix = f"[{', '.join(highlights)}] "

        bullet = f"• {prefix}{title}"
        if extra:
            bullet += f" — {extra}"
        if link:
            bullet += f" ({link})"

        if highlights:
            matched_lines.append(bullet)
            matched_meta.append(
                {
                    "source": name,
                    "title": title,
                    "extra": extra,
                    "link": link,
                    "highlights": highlights,
                }
            )
            highlight_label = html.escape(", ".join(highlights)) if highlights else ""
            title_html = html.escape(title)
            extra_html = html.escape(extra) if extra else ""
            if link:
                link_html = f"<a href='{html.escape(link)}'>{title_html}</a>"
            else:
                link_html = title_html
            content = f"<strong>[{highlight_label}]</strong> {link_html}"
            if extra_html:
                content += f" — {extra_html}"
            matched_html.append(f"<li>{content}</li>")
        else:
            other_lines.append(bullet)
            title_html = html.escape(title)
            extra_html = html.escape(extra) if extra else ""
            if link:
                link_html = f"<a href='{html.escape(link)}'>{title_html}</a>"
            else:
                link_html = title_html
            content = link_html
            if extra_html:
                content += f" — {extra_html}"
            other_html.append(f"<li>{content}</li>")

    if matched_lines:
        lines.append("  Focus matches:")
        lines.extend(matched_lines)

    if other_lines:
        lines.append("  Other updates:")
        limit = max(0, other_limit)
        if limit:
            lines.extend(other_lines[:limit])
        else:
            lines.extend(other_lines)

    html_lines = [
        f"<h3><a href='{html.escape(base_url)}'>{html.escape(name)}</a></h3>"
    ]
    if description:
        html_lines.append(f"<p>{html.escape(description)}</p>")

    if matched_html:
        html_lines.append("<p><strong>Focus matches:</strong></p><ul>" + "".join(matched_html) + "</ul>")

    if other_html:
        limit = max(0, other_limit)
        pruned = other_html if not limit else other_html[:limit]
        html_lines.append("<p><strong>Other updates:</strong></p><ul>" + "".join(pruned) + "</ul>")

    return "\n".join(lines), matched_meta, "\n".join(html_lines)


def fetch_updates(name: str, meta: Dict[str, str]) -> (str, List[Dict[str, str]], str):
    url = meta["url"]
    limit = int(meta.get("limit", 5))
    parsing_type = meta.get("type", "generic")
    keywords = meta.get("keywords")

    logging.info("Fetching %s", name)
    try:
        response = SESSION.get(url, timeout=20)
        allow_404 = bool(meta.get("allow_404"))
        if response.status_code == 404 and allow_404 and response.text:
            logging.debug("%s returned 404 but allow_404 enabled; continuing", name)
        else:
            response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        logging.error("Timeout while fetching %s: %s", name, exc)
        error_text = f"[{name}] ERROR: Request timed out"
        error_html = f"<h3>{html.escape(name)}</h3><p><strong>Error:</strong> Request timed out</p>"
        return error_text, [], error_html
    except requests.RequestException as exc:
        logging.error("HTTP error for %s: %s", name, exc)
        error_text = f"[{name}] ERROR: {exc}"
        error_html = f"<h3>{html.escape(name)}</h3><p><strong>Error:</strong> {html.escape(str(exc))}</p>"
        return error_text, [], error_html

    soup = BeautifulSoup(response.text, "html.parser")

    if parsing_type == "table":
        items = build_items_table(soup, meta, limit, url)
    elif parsing_type == "cards":
        items = build_items_cards(soup, meta, limit, url)
    elif parsing_type == "list":
        items = build_items_list(soup, meta, limit, url)
    else:
        items = build_items_generic(soup, limit, url, keywords)

    if not items:
        logging.warning("Falling back to generic parsing for %s", name)
        items = build_items_generic(soup, limit, url, keywords)

    if not items:
        fallback_msg = meta.get("fallback_message")
        if fallback_msg:
            plain = f"[{name}] {url}\n• {fallback_msg}"
            html_fragment = (
                f"<h3><a href='{html.escape(url)}'>{html.escape(name)}</a></h3>"
                f"<p>{html.escape(fallback_msg)}</p>"
            )
            return plain, [], html_fragment

    focus_spec = meta.get("focus_keywords")
    if isinstance(focus_spec, str):
        focus_list = _split_keywords(focus_spec)
    elif isinstance(focus_spec, list):
        focus_list = [kw for kw in focus_spec if isinstance(kw, str)]
    else:
        focus_list = None

    focus_map = _build_keyword_map(focus_list) if focus_list else FOCUS_KEYWORD_MAP
    other_limit = int(meta.get("other_limit", 2))

    formatted_text, matched_meta, html_fragment = format_items(
        name,
        url,
        items,
        description=meta.get("description"),
        focus_map=focus_map,
        other_limit=other_limit,
    )

    return formatted_text, matched_meta, html_fragment


def send_email(subject: str, body_html: str, body_plain: Optional[str] = None) -> None:
    if not EMAIL_FROM or not EMAIL_TO or not EMAIL_PASS:
        logging.warning("Email credentials incomplete; skipping email send")
        return

    if body_plain is None:
        body_plain = re.sub("<[^>]+>", " ", body_html)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body_plain, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    logging.info("Preparing email from %s to %s via %s:%s", EMAIL_FROM, EMAIL_TO, SMTP_SERVER, SMTP_PORT)

    try:
        if USE_SSL:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
                server.login(EMAIL_FROM, EMAIL_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
                if USE_TLS:
                    server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASS)
                server.send_message(msg)
        logging.info("Email sent successfully")
    except Exception as exc:
        logging.error("Failed to send email: %s", exc)


def main():
    sections_plain: Dict[str, str] = {}
    sections_html: Dict[str, str] = {}
    highlights: List[Dict[str, str]] = []

    ordered_names = list(WATCH_SOURCES.keys())
    source_order = {name: idx for idx, name in enumerate(ordered_names)}

    with ThreadPoolExecutor(max_workers=min(len(WATCH_SOURCES), 6)) as executor:
        future_map = {
            executor.submit(fetch_updates, name, meta): name
            for name, meta in WATCH_SOURCES.items()
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                plain_text, matched, html_fragment = future.result()
            except Exception as exc:
                logging.error("Worker failure for %s: %s", name, exc)
                plain_text = f"[{name}] ERROR: {exc}"
                html_fragment = f"<h3>{html.escape(name)}</h3><p><strong>Error:</strong> {html.escape(str(exc))}</p>"
                matched = []

            sections_plain[name] = plain_text
            sections_html[name] = html_fragment
            for item in matched:
                item_copy = dict(item)
                item_copy["source"] = name
                highlights.append(item_copy)

    highlights.sort(key=lambda item: source_order.get(item.get("source", ""), 1_000))

    ordered_plain = [sections_plain[name] for name in ordered_names if name in sections_plain]
    ordered_html = [sections_html[name] for name in ordered_names if name in sections_html]

    summary_lines = ["Actionable Highlights"]
    summary_html_lines = ["<h2>Actionable Highlights</h2>"]

    if highlights:
        for item in highlights:
            tags = ", ".join(item.get("highlights", []))
            title = item.get("title", "(no title)")
            source = item.get("source", "")
            extra = item.get("extra")
            link = item.get("link")

            summary_lines.append(f"- [{tags}] {title} ({source})")
            title_html = html.escape(title)
            tags_html = html.escape(tags)
            source_html = html.escape(source)
            if link:
                link_html = f"<a href='{html.escape(link)}'>{title_html}</a>"
            else:
                link_html = title_html
            bullet_html = f"<strong>[{tags_html}]</strong> {link_html} ({source_html})"
            if extra:
                summary_lines.append(f"  {extra}")
                bullet_html += f"<br/><em>{html.escape(extra)}</em>"
            if link:
                summary_lines.append(f"  {link}")
                bullet_html += f"<br/><small><a href='{html.escape(link)}'>Open link</a></small>"

            summary_html_lines.append(f"<p>{bullet_html}</p>")
    else:
        summary_lines.append("- No tracked companies flagged today.")
        summary_html_lines.append("<p>No tracked companies flagged today.</p>")

    summary_text = "\n".join(summary_lines)
    body_plain = summary_text + "\n\n" + "\n\n".join(ordered_plain)

    body_html = (
        "<html><body>"
        + "\n".join(summary_html_lines)
        + "<hr/>"
        + "".join(ordered_html)
        + "</body></html>"
    )

    print(summary_text)
    print()
    print("\n\n".join(ordered_plain))

    if EMAIL_ALERTS:
        send_email(EMAIL_SUBJECT, body_html, body_plain)


if __name__ == "__main__":
    main()
