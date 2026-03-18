"""Wikidata SPARQL helpers for Fact-State Layer extraction.

Strategy
--------
For each target property we issue a paginated SPARQL query against
the Wikidata Query Service (WDQS) that returns every time-qualified
statement for that property:

    SELECT ?item ?itemLabel ?value ?valueLabel ?start ?end
    WHERE {
        ?item p:P6 ?stmt .
        ?stmt ps:P6 ?value .
        OPTIONAL { ?stmt pq:P580 ?start . }
        OPTIONAL { ?stmt pq:P582 ?end   . }
        SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
    }
    LIMIT {page_size} OFFSET {offset}

Each raw row is one "statement interval" for an entity.  builder.py
assembles these into year-by-year FactTimeline objects.

Caching
-------
Results are cached as JSON files in *cache_dir* keyed by a hash of the
query text.  On re-runs only missing pages are fetched.

Rate-limiting
-------------
WDQS enforces a 60-second timeout and discourages more than ~1 query/s
from a single IP.  We use exponential back-off on 429/503 responses.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

WDQS_URL = "https://query.wikidata.org/sparql"
DEFAULT_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "TemporalKCResearch/0.1 (research; temporal-knowledge-conflict)",
}

# WDQS occasionally leaks ISO control characters into JSON responses.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# ---------------------------------------------------------------------------
# Domain catalogue – extend freely
# ---------------------------------------------------------------------------

PROPERTY_META: dict[str, dict] = {
    # --- Politics / Government ---
    "P6":    {"label": "head of government",        "domain": "politics"},
    "P35":   {"label": "head of state",             "domain": "politics"},
    "P488":  {"label": "chairperson",               "domain": "politics"},
    "P39":   {"label": "position held",             "domain": "politics"},
    "P102":  {"label": "member of political party", "domain": "politics"},
    "P1308": {"label": "officeholder",              "domain": "politics"},
    # --- Organisations ---
    "P463":  {"label": "member of organisation",    "domain": "org"},
    # --- Business / Corporate ---
    "P169":  {"label": "chief executive officer",   "domain": "corporate"},
    "P108":  {"label": "employer",                  "domain": "corporate"},
    "P1037": {"label": "director / manager",        "domain": "corporate"},
    "P127":  {"label": "owned by",                  "domain": "corporate"},
    "P749":  {"label": "parent organisation",       "domain": "corporate"},
    # --- Entertainment ---
    "P264":  {"label": "record label",              "domain": "entertainment"},
    # --- Academia ---
    "P69":   {"label": "educated at",               "domain": "academia"},
    # --- Sports ---
    "P54":   {"label": "member of sports team",     "domain": "sports"},
    "P286":  {"label": "head coach",                "domain": "sports"},
    "P413":  {"label": "position played",           "domain": "sports"},
}


def property_label(pid: str) -> str:
    return PROPERTY_META.get(pid, {}).get("label", pid)


def property_domain(pid: str) -> str:
    return PROPERTY_META.get(pid, {}).get("domain", "general")


# ---------------------------------------------------------------------------
# SPARQL query templates
# ---------------------------------------------------------------------------

def _make_query(pid: str, page_size: int, offset: int) -> str:
    """Build a paginated SPARQL query for time-qualified statements.

    Joins the English Wikipedia sitelink so we get the article title in the
    same request.  Only entities with an enwiki page are returned — this
    ensures that Wikipedia evidence can always be fetched.
    """
    return f"""
SELECT ?item ?itemLabel ?title ?value ?valueLabel ?start ?end WHERE {{
  ?item p:{pid} ?stmt .
  ?stmt ps:{pid} ?value .
  OPTIONAL {{ ?stmt pq:P580 ?start . }}
  OPTIONAL {{ ?stmt pq:P582 ?end   . }}
  ?sitelink schema:about ?item ;
            schema:isPartOf <https://en.wikipedia.org/> ;
            schema:name ?title .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
LIMIT {page_size}
OFFSET {offset}
""".strip()


def _cache_key(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


# ---------------------------------------------------------------------------
# HTTP fetch with retry
# ---------------------------------------------------------------------------

def _fetch_sparql(
    query: str,
    retries: int = 6,
    backoff: float = 2.0,
) -> list[dict]:
    """Execute *query* against WDQS and return list of result bindings.

    Robustness layers (matching production WDQS behaviour):
    1. Transient HTTP codes (429/500/502/503/504) → exponential back-off.
    2. Non-JSON content-type → WDQS returned an HTML error page; log and retry.
    3. JSON with control characters → strip them and retry the parse.
    4. All retries exhausted → return [] so the caller can skip this page.
    """
    params = {"query": query, "format": "json"}
    url    = f"{WDQS_URL}?{urlencode(params)}"
    delay  = 1.5

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=60)

            # ── transient server errors ───────────────────────────────────────
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) \
                          else min(30.0, delay)
                logger.warning("WDQS HTTP %s on attempt %d/%d, sleeping %.1fs",
                               resp.status_code, attempt + 1, retries, sleep_s)
                time.sleep(sleep_s)
                delay *= backoff
                continue

            resp.raise_for_status()

            # ── content-type guard (HTML error pages look like 200 OK) ────────
            ctype = resp.headers.get("Content-Type", "").lower()
            if "json" not in ctype and "sparql-results" not in ctype:
                snippet = resp.text[:300]
                logger.warning("WDQS returned non-JSON (attempt %d/%d): ctype=%r head=%r",
                               attempt + 1, retries, ctype, snippet)
                time.sleep(min(30.0, delay))
                delay *= backoff
                continue

            # ── parse with strict=False; fall back to control-char strip ──────
            raw_text = resp.text
            try:
                data = json.loads(raw_text, strict=False)
            except (json.JSONDecodeError, ValueError):
                sanitized = _CTRL_RE.sub(" ", raw_text)
                try:
                    data = json.loads(sanitized, strict=False)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning("WDQS JSON parse failed (attempt %d/%d): %s; head=%r",
                                   attempt + 1, retries, exc, raw_text[:200])
                    time.sleep(min(30.0, delay))
                    delay *= backoff
                    continue

            if not isinstance(data, dict) or "results" not in data:
                logger.warning("WDQS response missing 'results' key (attempt %d/%d)",
                               attempt + 1, retries)
                time.sleep(min(30.0, delay))
                delay *= backoff
                continue

            return data["results"]["bindings"]

        except requests.RequestException as exc:
            logger.warning("Network error (attempt %d/%d): %s", attempt + 1, retries, exc)
            time.sleep(min(30.0, delay))
            delay *= backoff

    logger.error("All %d retries exhausted for query: %s…", retries, query[:80])
    return []


# ---------------------------------------------------------------------------
# Cached paginated fetch
# ---------------------------------------------------------------------------

def fetch_statements_for_property(
    pid: str,
    *,
    page_size: int = 200,
    max_pages: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    sleep_between_pages: float = 1.2,
    cache_only: bool = False,
) -> Iterator[dict]:
    """Yield raw SPARQL result rows for property *pid*.

    Each row is a dict with keys: item, itemLabel, title, value, valueLabel,
    start (optional), end (optional).  Values are Wikidata binding dicts
    with "type" and "value" sub-keys.

    Parameters
    ----------
    pid                  Wikidata property ID, e.g. "P6"
    page_size            SPARQL LIMIT per query
    max_pages            Stop after this many pages (None = fetch all)
    cache_dir            If set, cache raw JSON results here
    sleep_between_pages  Courtesy delay between requests
    cache_only           If True, only serve from cache; raise on any cache miss.
                         Use for reproducible offline re-runs (e.g. SLURM).
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    offset   = 0
    page_idx = 0

    while True:
        if max_pages is not None and page_idx >= max_pages:
            logger.info("Reached max_pages=%d for %s, stopping.", max_pages, pid)
            break

        query      = _make_query(pid, page_size, offset)
        cache_path: Optional[Path] = (
            cache_dir / f"{_cache_key(query)}.json" if cache_dir else None
        )

        # ── serve from cache if available ────────────────────────────────────
        if cache_path and cache_path.exists():
            bindings = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.debug("Cache hit  %s page %d (offset %d)", pid, page_idx, offset)

        elif cache_only:
            # Strict offline mode: a cache miss is a hard error.
            raise RuntimeError(
                f"cache_only=True but no cached page for {pid} "
                f"page {page_idx} (offset {offset}).  "
                f"Run without --cache-only first to warm the cache."
            )

        else:
            # ── fetch from network; skip page on failure (don't abort run) ───
            bindings = _fetch_sparql(query)
            if not bindings and page_idx > 0:
                # An empty mid-run page most likely means we hit the end or a
                # transient failure.  Log and stop rather than skipping blindly
                # to avoid an infinite offset loop.
                logger.info("Empty/failed page for %s at offset %d — stopping.", pid, offset)
                break
            if cache_path is not None and bindings:
                cache_path.write_text(
                    json.dumps(bindings, ensure_ascii=False), encoding="utf-8"
                )
            logger.info("Fetched %s page %d: %d rows", pid, page_idx, len(bindings))
            time.sleep(sleep_between_pages)

        if not bindings:
            logger.info("Empty page for %s at offset %d, done.", pid, offset)
            break

        yield from bindings

        if len(bindings) < page_size:
            break   # last page

        offset   += page_size
        page_idx += 1


# ---------------------------------------------------------------------------
# Binding parsers
# ---------------------------------------------------------------------------

def _uri_to_qid(uri: str) -> str:
    """Extract QID from a Wikidata entity URI, e.g. Q9461."""
    return uri.rsplit("/", 1)[-1]


def _parse_year(dt_str: Optional[str]) -> Optional[int]:
    """Parse an ISO 8601 datetime string to year integer, or None."""
    if not dt_str:
        return None
    try:
        return int(dt_str[:4])
    except (ValueError, TypeError):
        return None


def parse_binding(row: dict) -> Optional[dict]:
    """Convert a raw SPARQL binding row to a clean Python dict.

    Returns None if the row is missing required fields or the value
    is not an entity (we skip literal-valued properties for now).
    """
    try:
        item_uri    = row["item"]["value"]
        item_label  = row.get("itemLabel", {}).get("value", "")
        value_uri   = row["value"]["value"]
        value_label = row.get("valueLabel", {}).get("value", "")
    except KeyError:
        return None

    # Only keep entity-valued statements (QIDs)
    if not value_uri.startswith("http://www.wikidata.org/entity/Q"):
        return None

    start_str = row.get("start", {}).get("value")
    end_str   = row.get("end",   {}).get("value")
    wiki_title = row.get("title", {}).get("value", "")

    return {
        "subject_qid":      _uri_to_qid(item_uri),
        "subject_label":    item_label,
        "wikipedia_title":  wiki_title,
        "object_qid":       _uri_to_qid(value_uri),
        "object_label":     value_label,
        "year_start":       _parse_year(start_str),
        "year_end":         _parse_year(end_str),
    }
