"""Land acquisition notice scraper and analyser.

This script fetches notice listings from configured district sites, parses PDF/HTML
notices for key fields, stores them in SQLite, and surfaces potentially actionable
items (small-area acquisitions, early-stage sections, etc.).
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import logging
import os
import re
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

try:  # Optional PDF parsers
    import pdfplumber  # type: ignore
    HAVE_PDFPLUMBER = True
except Exception:  # noqa: BLE001
    HAVE_PDFPLUMBER = False

try:  # Fallback
    from PyPDF2 import PdfReader  # type: ignore
    HAVE_PYPDF2 = True
except Exception:  # noqa: BLE001
    HAVE_PYPDF2 = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
)

DB_PATH = Path(os.environ.get("LAND_ACQ_DB", "land_acq.db")).resolve()
CSV_DEFAULT_PATH = Path(os.environ.get("LAND_ACQ_SITES_CSV", "~/Downloads/maharashtra_land_acquisition_sources.csv")).expanduser()
REQUEST_TIMEOUT = int(os.environ.get("LAND_ACQ_REQUEST_TIMEOUT", "10"))
MAX_WORKERS = int(os.environ.get("LAND_ACQ_MAX_WORKERS", "30"))

# (url, district)
def load_sites() -> List[SiteConfig]:
    sites: List[SiteConfig] = []

    csv_path = CSV_DEFAULT_PATH
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                for row in reader:
                    url = (row.get("la_page") or row.get("primary_site") or "").strip()
                    if not url:
                        continue
                    district = (row.get("district") or row.get("division") or "Unknown").strip()
                    example = (row.get("example_notice") or "").strip() or None
                    sites.append(SiteConfig(url, district or "Unknown", example))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to read %s: %s", csv_path, exc)

    # Allow environment overrides / additional entries
    raw = os.environ.get("LAND_ACQ_SITES", "").strip()
    if raw:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "|" in entry:
                url, district = entry.split("|", 1)
                sites.append(SiteConfig(url.strip(), district.strip() or "Unknown"))
            else:
                sites.append(SiteConfig(entry, "Unknown"))

    if not sites:
        LOGGER.warning(
            "No land acquisition sites configured. Set LAND_ACQ_SITES_CSV or LAND_ACQ_SITES."
        )
        sites = [SiteConfig("https://www.example.com/", "Example")]

    return sites

NOTICE_KEYWORDS = ("acquisition", "land acquisition", "section")
SECTION_PRIORITY = ("11", "3A", "19", "17")
MAX_ACTIONABLE_AREA_SQM = 10000.0
MAX_SITES = int(os.environ.get("LAND_ACQ_MAX_SITES", "30"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("land_acq")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SiteConfig:
    listing_url: str
    district: str
    example_notice: Optional[str] = None


@dataclasses.dataclass
class Notice:
    notice_id: str
    district: str
    title: str
    url: str
    notice_date: datetime
    survey_no: Optional[str] = None
    mouje: Optional[str] = None
    taluka: Optional[str] = None
    section: Optional[str] = None
    purpose: Optional[str] = None
    area_text: Optional[str] = None
    area_sqm: Optional[float] = None

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "id": "TEXT PRIMARY KEY",
    "district": "TEXT",
    "title": "TEXT",
    "url": "TEXT",
    "notice_date": "TEXT",
    "survey_no": "TEXT",
    "mouje": "TEXT",
    "taluka": "TEXT",
    "section": "TEXT",
    "purpose": "TEXT",
    "area_text": "TEXT",
    "area_sqm": "REAL",
    "created_at": "TEXT",
}


def init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notices (
            id TEXT PRIMARY KEY,
            district TEXT,
            title TEXT,
            url TEXT,
            notice_date TEXT,
            survey_no TEXT,
            mouje TEXT,
            taluka TEXT,
            section TEXT,
            purpose TEXT,
            area_text TEXT,
            area_sqm REAL,
            created_at TEXT
        )
        """
    )
    # Ensure schema compatibility with older deployments (add missing columns lazily)
    existing = {
        row[1]: (row[2], row[4])  # name -> (type, default)
        for row in conn.execute("PRAGMA table_info(notices)").fetchall()
    }
    for column, declaration in EXPECTED_COLUMNS.items():
        if column not in existing:
            LOGGER.info("Adding missing column '%s' to notices", column)
            conn.execute(f"ALTER TABLE notices ADD COLUMN {column} {declaration}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_notices_section ON notices(section)")
    conn.commit()
    return conn


def fetch_html(url: str, retries: int = 3, backoff: float = 1.5) -> Optional[str]:
    delay = 1.0
    for attempt in range(max(retries, 1)):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                LOGGER.warning("GET %s -> %s", url, resp.status_code)
                return None
            return resp.text
        except requests.exceptions.SSLError as exc:
            LOGGER.warning("SSL error for %s: %s — retrying without verification", url, exc)
            with contextlib.suppress(Exception):
                from requests.packages.urllib3.exceptions import InsecureRequestWarning  # type: ignore

                requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore
            try:
                resp = SESSION.get(url, timeout=REQUEST_TIMEOUT, verify=False)
                if resp.status_code >= 400:
                    LOGGER.warning("GET %s -> %s", url, resp.status_code)
                    return None
                return resp.text
            except requests.RequestException as exc2:  # noqa: PERF203
                LOGGER.error("Failed to fetch %s after SSL retry: %s", url, exc2)
        except requests.RequestException as exc:  # noqa: PERF203
            LOGGER.error("Failed to fetch %s (attempt %s/%s): %s", url, attempt + 1, retries, exc)
        if attempt < retries - 1:
            time.sleep(delay)
            delay *= backoff
    return None


def iter_notice_links(html: str, base_url: str) -> Iterable[Tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True)
        if not text:
            continue
        if any(keyword.lower() in text.lower() for keyword in NOTICE_KEYWORDS):
            href = anchor["href"].strip()
            if not href:
                continue
            if href.startswith("http"):
                yield text, href
            else:
                base = base_url.rstrip("/")
                yield text, f"{base}/{href.lstrip('/') }"


AREA_PATTERN = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*(sq\.?\.m|sqm|square metre|hectare|ha)", re.IGNORECASE)
SURVEY_PATTERN = re.compile(r"Survey\s*No\.?\s*([0-9A-Za-z/ -]+)")
MOUJE_PATTERN = re.compile(r"Mouje\s*[:\-]?\s*([A-Za-z0-9 /-]+)")
SECTION_PATTERN = re.compile(r"Section\s*([0-9A-Za-z]+)")


UNIT_TO_SQM = {
    "sq.m": 1.0,
    "sqm": 1.0,
    "square metre": 1.0,
    "hectare": 10000.0,
    "ha": 10000.0,
}


def _area_to_sqm(match: re.Match) -> Tuple[str, Optional[float]]:
    raw = match.group(0)
    value = float(match.group(1).replace(",", ""))
    unit = match.group(2).lower().replace(" ", "")
    for key, factor in UNIT_TO_SQM.items():
        if key.replace(" ", "") in unit:
            return raw, value * factor
    return raw, None


def extract_fields_from_text(text: str) -> Dict[str, Optional[str]]:
    fields: Dict[str, Optional[str]] = {
        "survey_no": None,
        "mouje": None,
        "section": None,
        "area_text": None,
        "area_sqm": None,
    }
    if survey := SURVEY_PATTERN.search(text):
        fields["survey_no"] = survey.group(1).strip()
    if mouje := MOUJE_PATTERN.search(text):
        fields["mouje"] = mouje.group(1).strip()
    if section := SECTION_PATTERN.search(text):
        fields["section"] = section.group(1).strip()
    if area := AREA_PATTERN.search(text):
        area_text, area_sqm = _area_to_sqm(area)
        fields["area_text"] = area_text
        fields["area_sqm"] = area_sqm
    return fields


def read_pdf_text(content: bytes) -> str:
    if HAVE_PDFPLUMBER:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            with pdfplumber.open(tmp.name) as pdf:  # type: ignore[arg-type]
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if HAVE_PYPDF2:
        reader = PdfReader(BytesIO(content))  # type: ignore[name-defined]
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    LOGGER.warning("No PDF parser available; skipping text extraction")
    return ""


def download_and_parse_notice(url: str) -> Dict[str, Optional[str]]:
    try:
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.error("Failed to download %s: %s", url, exc)
        return {}

    content_type = resp.headers.get("Content-Type", "").lower()
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        text = read_pdf_text(resp.content)
        return extract_fields_from_text(text)

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    return extract_fields_from_text(text)


def upsert_notice(conn: sqlite3.Connection, notice: Notice) -> None:
    payload = (
        notice.notice_id,
        notice.district,
        notice.title,
        notice.url,
        notice.notice_date.isoformat(),
        notice.survey_no,
        notice.mouje,
        notice.taluka,
        notice.section,
        notice.purpose,
        notice.area_text,
        notice.area_sqm,
        datetime.utcnow().isoformat(),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO notices (
            id, district, title, url, notice_date, survey_no, mouje, taluka,
            section, purpose, area_text, area_sqm, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    conn.commit()


def collect_site_notices(site: SiteConfig, seen_ids: set[str]) -> List[Notice]:
    LOGGER.info("Processing %s", site.listing_url)
    html = fetch_html(site.listing_url)
    notices: List[Notice] = []
    if not html and site.example_notice:
        fields = download_and_parse_notice(site.example_notice)
        if fields:
            notice_id = f"{site.district}||example||{site.example_notice}"
            if notice_id not in seen_ids:
                notices.append(
                    Notice(
                        notice_id=notice_id,
                        district=site.district,
                        title="Example Notice",
                        url=site.example_notice,
                        notice_date=datetime.utcnow(),
                        survey_no=fields.get("survey_no"),
                        mouje=fields.get("mouje"),
                        section=fields.get("section"),
                        purpose="Example notice fallback",
                        area_text=fields.get("area_text"),
                        area_sqm=fields.get("area_sqm"),
                    )
                )
        return notices
    if not html:
        return []
    notices: List[Notice] = []
    for title, link in iter_notice_links(html, site.listing_url):
        notice_id = f"{site.district}||{title}||{link}"
        if notice_id in seen_ids:
            continue
        fields = download_and_parse_notice(link)
        notices.append(
            Notice(
                notice_id=notice_id,
                district=site.district,
                title=title,
                url=link,
                notice_date=datetime.utcnow(),
                survey_no=fields.get("survey_no"),
                mouje=fields.get("mouje"),
                section=fields.get("section"),
                purpose=title,
                area_text=fields.get("area_text"),
                area_sqm=fields.get("area_sqm"),
            )
        )
    return notices


def actionable_insights(conn: sqlite3.Connection, limit: int = 5) -> List[Notice]:
    rows = conn.execute(
        """
        SELECT id, district, title, url, notice_date, survey_no, mouje,
               taluka, section, purpose, area_text, area_sqm
        FROM notices
        WHERE (section IS NULL OR section IN (?,?,?,?))
        ORDER BY notice_date DESC
        LIMIT ?
        """,
        (*SECTION_PRIORITY, limit),
    ).fetchall()
    insights: List[Notice] = []
    for row in rows:
        area_sqm = row[11]
        if area_sqm is not None and area_sqm > MAX_ACTIONABLE_AREA_SQM:
            continue
        insights.append(
            Notice(
                notice_id=row[0],
                district=row[1],
                title=row[2],
                url=row[3],
                notice_date=datetime.fromisoformat(row[4]),
                survey_no=row[5],
                mouje=row[6],
                taluka=row[7],
                section=row[8],
                purpose=row[9],
                area_text=row[10],
                area_sqm=area_sqm,
            )
        )
    return insights


def print_insights(insights: Iterable[Notice]) -> None:
    for notice in insights:
        area_display = notice.area_text or (f"{notice.area_sqm:.0f} sqm" if notice.area_sqm else "n/a")
        LOGGER.info(
            "Hot Prospect | District: %s | Mouje: %s | Survey: %s | Section: %s | Area: %s | URL: %s",
            notice.district,
            notice.mouje,
            notice.survey_no,
            notice.section,
            area_display,
            notice.url,
        )


def main() -> None:
    conn = init_db(DB_PATH)
    try:
        sites = load_sites()
        if MAX_SITES and len(sites) > MAX_SITES:
            LOGGER.info(
                "Processing first %s of %s configured sites (override with LAND_ACQ_MAX_SITES)",
                MAX_SITES,
                len(sites),
            )
            sites = sites[:MAX_SITES]

        existing_ids = {
            row[0] for row in conn.execute("SELECT id FROM notices").fetchall()
        }

        all_notices: List[Notice] = []
        worker_count = max(1, min(MAX_WORKERS, len(sites)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(collect_site_notices, site, existing_ids.copy()): site
                for site in sites
            }
            for future in as_completed(future_map):
                site = future_map[future]
                try:
                    notices = future.result()
                    all_notices.extend(notices)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.error(
                        "Worker failed for %s (%s): %s",
                        site.district,
                        site.listing_url,
                        exc,
                    )

        inserted = 0
        for notice in all_notices:
            if notice.notice_id in existing_ids:
                continue
            upsert_notice(conn, notice)
            existing_ids.add(notice.notice_id)
            inserted += 1
        LOGGER.info("Inserted/updated %s new notices", inserted)

        insights = actionable_insights(conn)
        print_insights(insights)
    finally:
        with contextlib.suppress(Exception):
            conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
