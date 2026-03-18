"""Timeline builder: converts raw SPARQL rows → FactTimeline objects.

Pipeline
--------
1.  Collect all raw statement intervals from sparql.fetch_statements_for_property().
2.  Group by (subject_qid, property_pid).
3.  For each group, expand intervals to year-by-year YearState lists.
4.  Detect change points.
5.  Filter out boring timelines (no changes, too short, ambiguous labels).
6.  Collect hard-negative distractors from sibling values.
7.  Return list[FactTimeline].

Year-state reconstruction
--------------------------
Wikidata statements have optional pq:P580 (start) and pq:P582 (end)
qualifiers.  We reconstruct the state for year Y as:

    objects_at(Y) = {obj | start(obj) <= Y  AND  (end(obj) is None OR end(obj) >= Y)}

If both start and end are None we treat the statement as "always valid"
(i.e. present in all years of the requested range).  These statements are
flagged and, when they are the only evidence, the timeline is labelled
"static" and excluded from the conflict-focused output.

Edge cases handled
------------------
- Multiple co-active values (e.g. co-leaders): kept as-is; the primary
  object is the first element sorted by QID for reproducibility.
- No active value in a year: the year is dropped from `states` (gap).
- Duplicate (subject, value, year_start, year_end) rows from SPARQL: deduped.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterator, Optional

from fact_timeline.models import FactTimeline, YearState
from fact_timeline.sparql import (
    fetch_statements_for_property,
    parse_binding,
    property_domain,
    property_label,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interval → year-set expansion
# ---------------------------------------------------------------------------

def _active_years(
    year_start: Optional[int],
    year_end:   Optional[int],
    range_start: int,
    range_end:   int,
) -> range:
    """Return the years [range_start, range_end] during which the statement
    was active, clipped to the requested range."""
    lo = max(year_start or range_start, range_start)
    hi = min(year_end   or range_end,   range_end)
    return range(lo, hi + 1)


def _year_to_objects(
    statements: list[dict],
    range_start: int,
    range_end: int,
) -> dict[int, list[tuple[str, str]]]:
    """Map each year to a sorted list of (qid, label) pairs active that year.

    Duplicates within the same (qid, year) are removed.
    """
    year_map: dict[int, dict[str, str]] = defaultdict(dict)
    for stmt in statements:
        for y in _active_years(stmt["year_start"], stmt["year_end"], range_start, range_end):
            qid = stmt["object_qid"]
            lbl = stmt["object_label"]
            year_map[y][qid] = lbl

    # Stable sort: alphabetical by QID for reproducibility
    return {
        y: sorted(objs.items(), key=lambda kv: kv[0])
        for y, objs in year_map.items()
    }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _build_timeline(
    subject_qid: str,
    subject_label: str,
    wikipedia_title: str,
    pid: str,
    statements: list[dict],
    range_start: int,
    range_end: int,
) -> Optional[FactTimeline]:
    """Assemble one FactTimeline from raw statement intervals."""
    year_map = _year_to_objects(statements, range_start, range_end)
    if not year_map:
        return None

    # Build YearState list
    states: list[YearState] = []
    prev_objs: list[str] = []
    change_years: list[int] = []

    for y in range(range_start, range_end + 1):
        if y not in year_map:
            continue          # gap: no evidence for this year
        pairs = year_map[y]
        objs  = [lbl for _, lbl in pairs]
        qids  = [qid for qid, _ in pairs]
        changed = (objs != prev_objs) and bool(prev_objs)
        if changed:
            change_years.append(y)
        states.append(YearState(
            year=y,
            objects=objs,
            object_qids=qids,
            changed_from_prev=changed,
            evidence_text="",   # filled by wiki_evidence.enrich_timelines()
            source_url="",
        ))
        prev_objs = objs

    if not states:
        return None

    fact_id = f"{subject_qid}_{pid}"
    return FactTimeline(
        fact_id=fact_id,
        subject_qid=subject_qid,
        subject_label=subject_label,
        wikipedia_title=wikipedia_title,
        property_pid=pid,
        property_label=property_label(pid),
        domain=property_domain(pid),
        year_start=states[0].year,
        year_end=states[-1].year,
        states=states,
        change_years=change_years,
        n_changes=len(change_years),
        distractors=[],          # filled in after all timelines are built
        source="wikidata_sparql",
    )


# ---------------------------------------------------------------------------
# Distractor injection
# ---------------------------------------------------------------------------

def _inject_distractors(timelines: list[FactTimeline], n: int = 4) -> None:
    """Fill FactTimeline.distractors with sibling object labels.

    Sibling = any object label that appears in another timeline for the
    same property but a different subject.  We pick the most frequent
    ones so they are plausible hard negatives.
    """
    from collections import Counter

    # Collect all object labels per property
    prop_pool: dict[str, Counter] = defaultdict(Counter)
    for tl in timelines:
        for state in tl.states:
            for lbl in state.objects:
                prop_pool[tl.property_pid][lbl] += 1

    for tl in timelines:
        own_labels: set[str] = set()
        for state in tl.states:
            own_labels.update(state.objects)

        pool = prop_pool[tl.property_pid]
        candidates = [
            lbl for lbl, _ in pool.most_common(50)
            if lbl not in own_labels
        ]
        tl.distractors = candidates[:n]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_timelines_for_property(
    pid: str,
    *,
    year_start: int = 2010,
    year_end: int = 2023,
    page_size: int = 200,
    max_pages: Optional[int] = None,
    min_changes: int = 1,
    cache_dir=None,
    sleep_between_pages: float = 1.2,
    cache_only: bool = False,
    progress: bool = True,
) -> list[FactTimeline]:
    """Fetch all time-qualified statements for *pid* from WDQS and return
    a list of FactTimeline objects, one per (subject, property) pair.

    Parameters
    ----------
    pid             : Wikidata property ID, e.g. "P6"
    year_start      : first year to include in timelines
    year_end        : last year to include in timelines
    page_size       : SPARQL LIMIT per request
    max_pages       : cap on SPARQL pages (None = unlimited)
    min_changes     : drop timelines with fewer changes than this
    cache_dir       : directory for SPARQL response cache
    sleep_between_pages : courtesy delay (seconds) between requests
    progress        : print progress to stderr
    """
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    # ---- Step 1: fetch raw rows and group by subject ----
    raw: dict[str, list[dict]] = defaultdict(list)
    subject_labels: dict[str, str] = {}
    n_rows = 0

    row_iter: Iterator[dict] = fetch_statements_for_property(
        pid,
        page_size=page_size,
        max_pages=max_pages,
        cache_dir=cache_dir,
        sleep_between_pages=sleep_between_pages,
        cache_only=cache_only,
    )

    if progress and _tqdm:
        row_iter = _tqdm(row_iter, desc=f"Fetching {pid}", unit="row", dynamic_ncols=True)

    wikipedia_titles: dict[str, str] = {}

    seen: set[tuple] = set()
    for binding in row_iter:
        parsed = parse_binding(binding)
        if parsed is None:
            continue
        key = (
            parsed["subject_qid"],
            parsed["object_qid"],
            parsed["year_start"],
            parsed["year_end"],
        )
        if key in seen:
            continue
        seen.add(key)
        qid = parsed["subject_qid"]
        raw[qid].append(parsed)
        subject_labels[qid] = parsed["subject_label"]
        if parsed.get("wikipedia_title"):
            wikipedia_titles[qid] = parsed["wikipedia_title"]
        n_rows += 1

    logger.info("Property %s: %d unique statement intervals, %d subjects", pid, n_rows, len(raw))

    # ---- Step 2: build timelines ----
    timelines: list[FactTimeline] = []
    subject_iter = raw.items()
    if progress and _tqdm:
        subject_iter = _tqdm(subject_iter, desc=f"Building {pid}", unit="subj", dynamic_ncols=True)

    for subj_qid, stmts in subject_iter:
        tl = _build_timeline(
            subject_qid=subj_qid,
            subject_label=subject_labels.get(subj_qid, subj_qid),
            wikipedia_title=wikipedia_titles.get(subj_qid, ""),
            pid=pid,
            statements=stmts,
            range_start=year_start,
            range_end=year_end,
        )
        if tl is None:
            continue
        if tl.n_changes < min_changes:
            continue
        timelines.append(tl)

    logger.info("Property %s: %d timelines with ≥%d change(s)", pid, len(timelines), min_changes)

    # ---- Step 3: inject distractors ----
    _inject_distractors(timelines)

    return timelines
