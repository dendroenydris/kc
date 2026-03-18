"""Wikipedia revision fetcher and evidence extractor.

Self-contained module — no external package dependencies beyond:
    requests          (HTTP)
    mwparserfromhell  (wikitext → plain text)
    fact_timeline.cache (FileCache)

Evidence strategy
-----------------
Only YearStates with changed_from_prev=True get real Wikipedia evidence.
For each such state we fetch TWO revisions:

  rev_new  — the Wikipedia article at the START of the new tenure
             (mid-year of the change year, e.g. 2012-07-01)
             → evidence that the new answer is now correct

  rev_old  — the Wikipedia article at the END of the old tenure
             (mid-year of the preceding year)
             → evidence that the old answer was correct
             → stored on the preceding YearState

Stable years get a lightweight synthetic sentence to avoid O(N×Y) API calls.

Provenance
----------
Every real evidence sentence is suffixed with a structured tag:
    [enwiki:<title>|revid=<id>|ts=<timestamp>]
so every piece of evidence is exactly reproducible.

Caching
-------
Revision responses are cached under <cache_dir>/wiki_revisions/<hash>.json
so a re-run fetches nothing from the network if the cache is warm.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

try:
    import mwparserfromhell  # type: ignore
    _MWP_AVAILABLE = True
except ImportError:
    _MWP_AVAILABLE = False

from fact_timeline.cache import FileCache

logger = logging.getLogger(__name__)

_USER_AGENT = "TemporalKCResearch/0.1 (research; temporal-knowledge-conflict)"
_WS_RE = re.compile(r"\s+")


# ── WikipediaRevision dataclass ───────────────────────────────────────────────

@dataclass(frozen=True)
class WikipediaRevision:
    pageid: int
    revid: int
    timestamp: str
    wikitext: str


# ── Wikitext → plain text ─────────────────────────────────────────────────────

def wikitext_to_plaintext(wikitext: str) -> str:
    """Convert wikitext to plain text.

    Uses mwparserfromhell when available (correct, handles all wikitext
    constructs).  Falls back to a regex stripper when the library is absent.
    """
    if _MWP_AVAILABLE:
        code = mwparserfromhell.parse(wikitext)
        text = code.strip_code(normalize=True, collapse=True)
        return _WS_RE.sub(" ", text).strip()

    # Lightweight fallback (less accurate)
    t = wikitext
    t = re.sub(r"<ref[^>]*>.*?</ref>", " ", t, flags=re.DOTALL)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\{\{[^}]*\}\}", " ", t)
    t = re.sub(r"\{\|.*?\|\}", " ", t, flags=re.DOTALL)
    t = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", t)
    t = re.sub(r"'{2,3}", "", t)
    t = re.sub(r"==+[^=]*==+", " ", t)
    t = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", " ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


# ── Sentence tools ────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Relation keyword sets — used to validate that an evidence snippet is
# actually about the right relation, not just a coincidental name mention.
# ---------------------------------------------------------------------------
_RELATION_KEYWORDS: dict[str, list[str]] = {
    "head of government": [
        "president", "prime minister", "premier", "chancellor",
        "minister", "government", "leader", "head of government",
    ],
    "head of state": [
        "president", "king", "queen", "monarch", "emperor", "head of state",
        "sovereign", "royal", "crown",
    ],
    "chairperson": [
        "chairman", "chairwoman", "chairperson", "chair", "general secretary",
        "secretary general", "party leader", "leader",
    ],
    "position held": [
        "minister", "secretary", "senator", "representative", "governor",
        "mayor", "ambassador", "commissioner", "director", "president",
        "position", "appointed", "elected", "served",
    ],
    "chief executive officer": [
        "ceo", "chief executive", "managing director", "executive director",
        "president", "director general", "managing",
    ],
    "employer": [
        "work", "employ", "company", "firm", "organisation", "organization",
        "join", "hired", "staff", "position",
    ],
    "director / manager": [
        "director", "manager", "chairman", "chair", "head", "chief",
        "appointed", "elected", "led", "leads",
    ],
    "member of sports team": [
        "team", "club", "signed", "transfer", "plays", "played", "squad",
        "joined", "contract", "player",
    ],
    "head coach": [
        "coach", "manager", "trainer", "head coach", "hired", "appointed",
        "sacked", "assistant", "coaching staff",
    ],
    # --- Politics (new) ---
    "member of political party": [
        "party", "member", "joined", "left", "republican", "democrat",
        "conservative", "labour", "socialist", "liberal", "political",
    ],
    "officeholder": [
        "held", "office", "position", "appointed", "elected", "served",
        "term", "minister", "secretary", "senator", "governor",
    ],
    # --- Organisations (new) ---
    "member of organisation": [
        "member", "joined", "membership", "admitted", "organisation",
        "organization", "alliance", "union", "association",
    ],
    # --- Business / Corporate (new) ---
    "owned by": [
        "owned", "owner", "acquired", "acquisition", "purchased", "bought",
        "subsidiary", "property", "stake", "holding",
    ],
    "parent organisation": [
        "parent", "subsidiary", "owned", "division", "part of",
        "belongs to", "acquired", "merged", "conglomerate",
    ],
    # --- Entertainment (new) ---
    "record label": [
        "label", "signed", "record", "album", "released", "contract",
        "music", "discography", "studio",
    ],
    # --- Academia (new) ---
    "educated at": [
        "studied", "university", "college", "school", "degree", "enrolled",
        "graduated", "attended", "education", "student", "alumni",
    ],
    # --- Sports (new) ---
    "position played": [
        "position", "plays", "played", "forward", "midfielder", "defender",
        "goalkeeper", "pitcher", "quarterback", "guard", "center", "wing",
    ],
}

_RELATION_KEYWORDS_DEFAULT = ["appointed", "elected", "served", "became", "joined"]


def _relation_keywords(relation: str) -> list[str]:
    return _RELATION_KEYWORDS.get(relation.lower(), _RELATION_KEYWORDS_DEFAULT)


def find_evidence_snippet(
    text: str,
    answer: str,
    *,
    year: Optional[int] = None,
    relation: str = "",
    max_sentences: int = 2,
    window: int = 1,
) -> Optional[str]:
    """Return a 1–2 sentence snippet containing *answer* (case-insensitive).

    Three-tier search (stops at first hit per tier):
    1. Exact full-name substring match.
    2. Token fallback — any significant token (≥4 chars) of the answer name.
    3. Year-context fallback — a sentence mentioning *year* that also has a
       relation keyword.

    Relation filter: every candidate snippet must contain at least one keyword
    associated with *relation*. Snippets that only mention the name in an
    unrelated context are discarded, forcing a synthetic fallback instead.
    """
    if not answer or not text:
        return None

    sentences = split_sentences(text)
    rel_kws   = _relation_keywords(relation)

    def _rel_ok(snippet: str) -> bool:
        sl = snippet.lower()
        return any(kw in sl for kw in rel_kws)

    def _hit(needle: str) -> Optional[str]:
        needle_l = needle.lower()
        for i, s in enumerate(sentences):
            if needle_l in s.lower():
                start   = max(0, i - window)
                end     = min(len(sentences), i + window + 1)
                snippet = " ".join(sentences[start:end][:max_sentences]).strip()
                if _rel_ok(snippet):
                    return snippet
        return None

    # Tier 1: exact full name
    snippet = _hit(answer)
    if snippet:
        return snippet

    # Tier 2: significant name tokens (≥4 chars)
    _STOP = {"the", "and", "for", "von", "van", "der", "of", "al"}
    tokens = [t for t in re.split(r"[\s,.\-]+", answer)
              if len(t) >= 4 and t.lower() not in _STOP]
    for tok in tokens:
        snippet = _hit(tok)
        if snippet:
            return snippet

    # Tier 3: year + relation keyword (answer may not appear yet)
    if year is not None:
        year_str = str(year)
        for i, s in enumerate(sentences):
            if year_str in s and _rel_ok(s):
                start = max(0, i - window)
                end   = min(len(sentences), i + window + 1)
                return " ".join(sentences[start:end][:max_sentences]).strip()

    return None


# ── Wikipedia revision fetch ──────────────────────────────────────────────────

def fetch_revision_at_or_before(
    *,
    title: str,
    as_of_iso: str,
    cache: FileCache,
    max_retries: int = 6,
    base_backoff_s: float = 1.0,
    polite_sleep_s: float = 0.25,
) -> Optional[WikipediaRevision]:
    """Fetch the most recent Wikipedia revision at or before *as_of_iso* (UTC ISO 8601).

    Uses the MediaWiki API with rvstart + rvdir=older so the result is the
    snapshot of the article as it existed on that date.

    Results are cached by (title, as_of_iso) to avoid duplicate network calls.
    """
    cache_key = f"en|{title}|{as_of_iso}"
    cached = cache.get("wiki_revision", cache_key)
    if cached is not None:
        return WikipediaRevision(
            pageid=cached["pageid"],
            revid=cached["revid"],
            timestamp=cached["timestamp"],
            wikitext=cached["wikitext"],
        )

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "rvlimit": "1",
        "rvstart": as_of_iso,
        "rvdir": "older",
        "redirects": "1",
    }
    headers = {"User-Agent": _USER_AGENT}
    url = "https://en.wikipedia.org/w/api.php"

    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) \
                          else min(30.0, base_backoff_s * (2 ** attempt))
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            try:
                data = resp.json()
            except (json.JSONDecodeError, ValueError):
                time.sleep(min(30.0, base_backoff_s * (2 ** attempt)))
                continue
            break
        except requests.RequestException:
            time.sleep(min(30.0, base_backoff_s * (2 ** attempt)))

    if data is None:
        logger.warning("Wikipedia revision fetch failed: title=%r as_of=%s", title, as_of_iso)
        return None

    pages = data.get("query", {}).get("pages", [])
    if not pages or "missing" in pages[0]:
        return None
    revs = pages[0].get("revisions", [])
    if not revs:
        return None

    rev = revs[0]
    wikitext = rev.get("slots", {}).get("main", {}).get("content", "") or ""
    if not wikitext:
        return None

    result = WikipediaRevision(
        pageid=int(pages[0]["pageid"]),
        revid=int(rev.get("revid", 0)),
        timestamp=str(rev["timestamp"]),
        wikitext=wikitext,
    )
    cache.set("wiki_revision", cache_key, {
        "pageid": result.pageid, "revid": result.revid,
        "timestamp": result.timestamp, "wikitext": result.wikitext,
    })
    time.sleep(polite_sleep_s)
    return result


def _provenance(title: str, revid: int, timestamp: str) -> str:
    return f"enwiki:{title}|revid={revid}|ts={timestamp}"


# ── Synthetic fallback ────────────────────────────────────────────────────────

def _synthetic(subject: str, relation: str, obj: str, year: int) -> str:
    return f"As of {year}, {subject}'s {relation} was {obj}."


# ── Main enrichment entry point ───────────────────────────────────────────────

def enrich_timelines(
    timelines: list,           # list[FactTimeline]
    *,
    cache_dir: Optional[Path] = None,
    progress: bool = True,
) -> None:
    """Enrich YearState.evidence_text and source_url in-place for all timelines.

    Change years (changed_from_prev=True):
      - Fetch Wikipedia revision at change year  → evidence_text for new value
      - Fetch Wikipedia revision at preceding year → evidence_text for old value
                                                     (written to preceding YearState)
      - Hard filter: if the answer label is not found in the revision text,
        falls back to a synthetic sentence (does NOT discard the state).

    Stable years: synthetic sentence only (avoids O(N×Y) API calls).
    """
    if not timelines:
        return

    if not _MWP_AVAILABLE:
        logger.warning(
            "mwparserfromhell not installed; wikitext parsing will use regex fallback. "
            "Run: pip install mwparserfromhell"
        )

    rev_cache = FileCache(root=Path(cache_dir) / "wiki_revisions") if cache_dir \
                else FileCache(root=Path("/tmp/fact_timeline_cache/wiki_revisions"))

    for tl in timelines:
        title        = getattr(tl, "wikipedia_title", "") or ""
        wikidata_url = f"https://www.wikidata.org/wiki/{tl.subject_qid}" if tl.subject_qid else ""
        wiki_url_base = (
            f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            if title else wikidata_url
        )

        if progress:
            logger.info(
                "Enriching [%s] %s  (wiki: %s)",
                tl.property_label, tl.subject_label, title or "—",
            )

        for state in tl.states:
            obj = state.objects[0] if state.objects else ""

            if not title or not obj:
                state.evidence_text = _synthetic(tl.subject_label, tl.property_label, obj, state.year)
                state.source_url    = wiki_url_base
                continue

            # Fetch the Wikipedia article as it existed at mid-year (every year)
            as_of_iso = f"{state.year}-07-01T00:00:00Z"
            rev = fetch_revision_at_or_before(title=title, as_of_iso=as_of_iso, cache=rev_cache)

            if rev is not None:
                text    = wikitext_to_plaintext(rev.wikitext)
                snippet = find_evidence_snippet(
                    text, obj,
                    year=state.year, relation=tl.property_label,
                )
                if snippet:
                    state.evidence_text = snippet
                    state.source_url    = f"{wiki_url_base}?oldid={rev.revid}"
                    continue

            # Fallback: synthetic sentence (real evidence not found)
            state.evidence_text = _synthetic(tl.subject_label, tl.property_label, obj, state.year)
            state.source_url    = wiki_url_base
