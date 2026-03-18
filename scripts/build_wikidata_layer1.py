#!/usr/bin/env python3
"""Build Layer-1 (Fact-State Layer) from Wikidata SPARQL + Wikipedia evidence.

Pipeline
--------
1.  Query WDQS for each property in PROPERTY_META; only entities that have
    an English Wikipedia sitelink are returned (joined in SPARQL).
2.  Reconstruct year-by-year YearState timelines (builder.py).
3.  Rank timelines and keep N per property.
4.  For each change year, fetch the Wikipedia revision at that timestamp and
    extract a 1-2 sentence evidence snippet (wiki_evidence.py, mwparserfromhell).
5.  Write Layer-1 JSONL.

Usage
-----
    # Default: 10 timelines per property, 2010-2023, all 9 properties
    python code/scripts/build_wikidata_layer1.py

    # Politics only, 20 per property
    python code/scripts/build_wikidata_layer1.py --n 20 --pids P6 P35 P39

    # Skip Wikipedia (fast smoke-test; synthetic evidence only)
    python code/scripts/build_wikidata_layer1.py --no-wiki

Cache
-----
    code/data/cache/sparql/          SPARQL response pages
    code/data/cache/wiki_revisions/  Wikipedia revision JSON
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "source"))

from fact_timeline.sparql import PROPERTY_META
from fact_timeline.builder import build_timelines_for_property
from fact_timeline.wiki_evidence import enrich_timelines, _synthetic
from fact_timeline.models import FactTimeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OUT  = REPO_ROOT / "data/processed/wikidata_layer1.jsonl"
CACHE_ROOT   = REPO_ROOT / "data/cache"
YEAR_START   = 2010
YEAR_END     = 2023


def _score(tl: FactTimeline) -> float:
    name_penalty = max(0.0, (len(tl.subject_label) - 20) / 100)
    return len(tl.states) * 0.3 + tl.n_changes * 2.0 + (tl.year_end - tl.year_start) * 0.2 - name_penalty


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n",           type=int, default=1,  help="Timelines per property (default: 1)")
    p.add_argument("--pids",        nargs="+", default=list(PROPERTY_META.keys()), metavar="PID")
    p.add_argument("--year-start",  type=int, default=YEAR_START)
    p.add_argument("--year-end",    type=int, default=YEAR_END)
    p.add_argument("--max-pages",   type=int, default=5,
                   help="SPARQL pages per property, 200 rows/page (default: 5)")
    p.add_argument("--no-wiki",     action="store_true",
                   help="Skip Wikipedia; fill evidence_text with synthetic sentences")
    p.add_argument("--cache-only",  action="store_true",
                   help="Only use cached SPARQL pages; error on any cache miss (for offline re-runs)")
    p.add_argument("--out",         default=str(DEFAULT_OUT))
    args = p.parse_args()

    out_path   = Path(args.out)
    stats_path = out_path.with_suffix("").with_name(out_path.stem + "_stats.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pids = [pid for pid in args.pids if pid in PROPERTY_META]
    if not pids:
        sys.exit(f"[ERROR] None of {args.pids} are in PROPERTY_META. "
                 f"Valid: {list(PROPERTY_META.keys())}")

    logger.info("Properties : %s", pids)
    logger.info("Year range : %d–%d  |  N per property: %d", args.year_start, args.year_end, args.n)
    logger.info("Evidence   : %s", "synthetic (--no-wiki)" if args.no_wiki else "Wikipedia revisions (mwparserfromhell)")

    all_timelines: list[FactTimeline] = []
    stat_rows: list[str] = []

    for pid in pids:
        label = PROPERTY_META[pid]["label"]
        logger.info("── %s (%s)", label, pid)

        timelines = build_timelines_for_property(
            pid,
            year_start=args.year_start,
            year_end=args.year_end,
            max_pages=args.max_pages,
            min_changes=1,
            cache_dir=CACHE_ROOT / "sparql",
            sleep_between_pages=1.2,
            cache_only=args.cache_only,
            progress=True,
        )
        timelines.sort(key=_score, reverse=True)
        chosen = timelines[: args.n]
        logger.info("   raw=%d  selected=%d", len(timelines), len(chosen))

        if args.no_wiki:
            for tl in chosen:
                for state in tl.states:
                    obj = state.objects[0] if state.objects else ""
                    state.evidence_text = _synthetic(tl.subject_label, tl.property_label, obj, state.year)
                    state.source_url    = (
                        f"https://www.wikidata.org/wiki/{tl.subject_qid}" if tl.subject_qid else ""
                    )
        elif chosen:
            logger.info("   Enriching %d timelines with Wikipedia evidence …", len(chosen))
            enrich_timelines(chosen, cache_dir=CACHE_ROOT, progress=False)

        all_timelines.extend(chosen)
        stat_rows.append(f"  {label:35s} ({pid})  raw={len(timelines):4d}  selected={len(chosen):3d}")

    # Write Layer-1
    with open(out_path, "w", encoding="utf-8") as fh:
        for tl in all_timelines:
            fh.write(tl.to_json() + "\n")

    total_changes = sum(tl.n_changes for tl in all_timelines)
    stats = "\n".join([
        "=== Wikidata Layer-1 Stats ===",
        f"Properties   : {len(pids)}",
        f"FactTimelines: {len(all_timelines)}",
        f"Change events: {total_changes}",
        f"Year range   : {args.year_start}–{args.year_end}",
        f"Evidence     : {'synthetic' if args.no_wiki else 'Wikipedia revisions'}",
        "", *stat_rows,
    ])
    stats_path.write_text(stats + "\n", encoding="utf-8")
    print(f"\n{stats}\n\n[OK] {out_path}")

    # Sample
    if all_timelines:
        tl = all_timelines[0]
        print(f"\n── Sample: {tl.subject_label} / {tl.property_label}  (wiki: {tl.wikipedia_title})")
        for s in tl.states[:3]:
            obj = s.objects[0] if s.objects else "?"
            print(f"  {s.year}: {obj}{'  ← CHANGE' if s.changed_from_prev else ''}")
            if s.evidence_text:
                print(f"       {s.evidence_text[:110].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
