from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from dateutil.parser import isoparse

from .cache import FileCache
from .schema import TemporalConflictExample
from .templates import render_question
from .text import find_evidence_snippet, wikitext_to_plaintext
from .wikidata import WikidataStatement, fetch_time_scoped_statements
from .wikipedia import fetch_revision_at_or_before, provenance_string


def _to_date_str(iso_dt: str) -> str:
    # iso_dt may include time; we render only date for prompt readability
    dt = _safe_isoparse(iso_dt)
    return dt.date().isoformat()


def _days_between(a_iso: str, b_iso: str) -> int:
    a = _safe_isoparse(a_iso)
    b = _safe_isoparse(b_iso)
    return abs((b - a).days)


_WIKIDATA_PARTIAL_DATE_RE = re.compile(r"^([+-]?\d{4})-(\d{2})-(\d{2})")


def _normalize_wikidata_time(raw: str) -> str:
    """
    Wikidata qualifier times can include partial dates like YYYY-00-00.
    Convert those to parseable placeholders (month/day -> 01).
    """
    s = raw.strip()
    m = _WIKIDATA_PARTIAL_DATE_RE.match(s)
    if not m:
        return s
    year, month, day = m.group(1), m.group(2), m.group(3)
    if month == "00":
        month = "01"
    if day == "00":
        day = "01"
    return _WIKIDATA_PARTIAL_DATE_RE.sub(f"{year}-{month}-{day}", s, count=1)


def _safe_isoparse(raw: str) -> datetime:
    """
    Robust timestamp parse for mixed ISO / Wikidata partial dates.
    Falls back to a very old sentinel on parse failure.
    """
    try:
        return isoparse(raw)
    except Exception:
        try:
            return isoparse(_normalize_wikidata_time(raw))
        except Exception:
            return datetime(1900, 1, 1, tzinfo=timezone.utc)


def _pick_time_within_interval(start: Optional[str], end: Optional[str], fallback: str) -> str:
    """
    Pick a timepoint that is likely to be inside an interval.
    If both start and end exist, choose midpoint. If only one exists, use it.
    Otherwise use fallback.
    """
    if start and end:
        s = _safe_isoparse(start)
        e = _safe_isoparse(end)
        if e <= s:
            return s.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        mid = s + (e - s) / 2
        return mid.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if start:
        return _safe_isoparse(start).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if end:
        return _safe_isoparse(end).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return fallback


@dataclass(frozen=True)
class ChangeEvent:
    subject_qid: str
    subject_label: str
    wikipedia_title: str

    property_pid: str
    property_label: str

    old_value_qid: str
    old_value_label: str
    old_start: Optional[str]
    old_end: Optional[str]

    new_value_qid: str
    new_value_label: str
    new_start: Optional[str]
    new_end: Optional[str]


def derive_change_events(statements: Iterable[WikidataStatement]) -> List[ChangeEvent]:
    """
    Convert time-scoped statements into change events: (old value) -> (new value)

    Heuristic:
    - group by (subject, property)
    - sort statements by start_time (or end_time) if available
    - pick consecutive statements with different values and a non-trivial time gap
    """
    groups: Dict[Tuple[str, str], List[WikidataStatement]] = defaultdict(list)
    for s in statements:
        groups[(s.subject_qid, s.property_pid)].append(s)

    events: List[ChangeEvent] = []
    for (subj_qid, pid), stmts in groups.items():
        def sort_key(x: WikidataStatement) -> datetime:
            t = x.start_time or x.end_time or "1900-01-01T00:00:00Z"
            return _safe_isoparse(t)

        stmts_sorted = sorted(stmts, key=sort_key)
        for a, b in zip(stmts_sorted, stmts_sorted[1:]):
            if a.value_qid == b.value_qid:
                continue
            events.append(
                ChangeEvent(
                    subject_qid=a.subject_qid,
                    subject_label=a.subject_label,
                    wikipedia_title=a.wikipedia_title,
                    property_pid=a.property_pid,
                    property_label=a.property_label,
                    old_value_qid=a.value_qid,
                    old_value_label=a.value_label,
                    old_start=a.start_time,
                    old_end=a.end_time,
                    new_value_qid=b.value_qid,
                    new_value_label=b.value_label,
                    new_start=b.start_time,
                    new_end=b.end_time,
                )
            )
    return events


def _stable_shard_key(ev: ChangeEvent) -> str:
    return f"{ev.subject_qid}|{ev.property_pid}|{ev.old_value_qid}->{ev.new_value_qid}"


def _stable_shard_id(key: str, num_shards: int) -> int:
    # Stable, cross-run shard assignment.
    # We avoid Python's built-in hash() because it is randomized per process.
    import hashlib

    h = hashlib.sha256(key.encode("utf-8")).digest()
    # use first 8 bytes as unsigned int
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    return v % num_shards


def build_examples(
    *,
    limit: int,
    property_pids: List[str],
    min_gap_days: int,
    wiki_lang: str,
    user_agent: str,
    cache_dir: str,
    shard_id: int = 0,
    num_shards: int = 1,
    max_events_to_scan: int = 200000,
    fetch_multiplier: int = 20,
    min_fetch_per_property: int = 100,
    max_fetch_per_property: int = 6000,
    page_size: int = 200,
    pages_per_property: Optional[int] = None,
    pages: Optional[int] = None,
    include_time_in_question: bool = False,
    progress_callback: Optional[Callable[[str, Dict], None]] = None,
    cache_only: bool = False,
) -> Iterator[TemporalConflictExample]:
    cache = FileCache(root=__import__("pathlib").Path(cache_dir))

    # Overfetch to compensate for filtering (missing evidence in revisions, etc.)
    if pages_per_property is not None:
        fetch_limit = max(1, pages_per_property) * page_size
    else:
        fetch_limit = max(min_fetch_per_property, min(limit * fetch_multiplier, max_fetch_per_property))
    if progress_callback is not None:
        progress_callback(
            "fetch_plan",
            {
                "fetch_limit_per_property": fetch_limit,
                "num_properties": len(property_pids),
                "page_size": page_size,
                "pages_per_property": pages_per_property,
                "pages": pages,
            },
        )
    stmts = fetch_time_scoped_statements(
        property_pids=property_pids,
        limit_per_property=fetch_limit,
        page_size=page_size,
        max_pages_per_property=pages_per_property,
        max_total_pages=pages,
        user_agent=user_agent,
        cache=cache,
        wiki_lang=wiki_lang,
        progress_callback=progress_callback,
        cache_only=cache_only,
    )
    events = derive_change_events(stmts)
    # Deterministic traversal for SLURM sharding/reproducibility:
    # sort by stable key, then each shard takes its own slice by stable hash mod num_shards.
    events = sorted(events, key=_stable_shard_key)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    written = 0
    scanned = 0
    for idx, ev in enumerate(events):
        if written >= limit:
            break
        if scanned >= max_events_to_scan:
            break

        if num_shards > 1:
            sid = _stable_shard_id(_stable_shard_key(ev), num_shards)
            if sid != shard_id:
                continue
        scanned += 1
        if progress_callback is not None and scanned % 100 == 0:
            progress_callback("scan_progress", {"scanned": scanned, "written": written})

        # Choose timepoints. Prefer within each value's interval.
        t_old = _pick_time_within_interval(ev.old_start, ev.old_end, fallback=now_iso)
        t_new = _pick_time_within_interval(ev.new_start, ev.new_end, fallback=now_iso)

        gap = _days_between(t_old, t_new)
        if gap < min_gap_days:
            continue

        rev_old = fetch_revision_at_or_before(
            title=ev.wikipedia_title,
            as_of_iso=t_old,
            lang=wiki_lang,
            user_agent=user_agent,
            cache=cache,
            cache_only=cache_only,
        )
        if progress_callback is not None:
            progress_callback("revision_fetch", {"which": "old", "ok": rev_old is not None})
        rev_new = fetch_revision_at_or_before(
            title=ev.wikipedia_title,
            as_of_iso=t_new,
            lang=wiki_lang,
            user_agent=user_agent,
            cache=cache,
            cache_only=cache_only,
        )
        if progress_callback is not None:
            progress_callback("revision_fetch", {"which": "new", "ok": rev_new is not None})
        if rev_old is None or rev_new is None:
            continue

        text_old = wikitext_to_plaintext(rev_old.wikitext)
        text_new = wikitext_to_plaintext(rev_new.wikitext)

        evidence_old = find_evidence_snippet(text=text_old, answer=ev.old_value_label)
        evidence_new = find_evidence_snippet(text=text_new, answer=ev.new_value_label)

        # Filter out cases where we can't locate evidence sentences.
        if not evidence_old or not evidence_new:
            continue

        question = render_question(
            pid=ev.property_pid,
            subject_label=ev.subject_label,
            as_of_date=_to_date_str(t_new),
            property_label=ev.property_label,
            include_time_in_question=include_time_in_question,
        )

        ex = TemporalConflictExample(
            id=f"{ev.subject_qid}_{ev.property_pid}_{idx}",
            question=question,
            as_of_mode="timestamp",
            t_old=t_old,
            t_new=t_new,
            time_gap_days=gap,
            answer_old=ev.old_value_label,
            answer_new=ev.new_value_label,
            evidence_old=evidence_old,
            evidence_new=evidence_new,
            subject_qid=ev.subject_qid,
            property_pid=ev.property_pid,
            subject_label=ev.subject_label,
            property_label=ev.property_label,
            value_old_qid=ev.old_value_qid,
            value_new_qid=ev.new_value_qid,
            provenance_old=provenance_string(
                lang=wiki_lang, title=ev.wikipedia_title, revid=rev_old.revid, timestamp=rev_old.timestamp
            ),
            provenance_new=provenance_string(
                lang=wiki_lang, title=ev.wikipedia_title, revid=rev_new.revid, timestamp=rev_new.timestamp
            ),
            wikipedia_title=ev.wikipedia_title,
        )

        written += 1
        if progress_callback is not None:
            progress_callback("example_written", {"written": written, "scanned": scanned})
        yield ex

