from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from temporal_kc_dataset_builder.cli import run  # type: ignore[reportMissingImports]  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=500, help="Target number of examples to write (default: 500)")
    parser.add_argument(
        "--properties",
        default="P6,P35,P169",
        help="Comma-separated Wikidata properties (entity-valued) to mine, e.g. P6,P35",
    )
    parser.add_argument(
        "--min_gap_days",
        type=int,
        default=30,
        help="Minimum days between old and new timepoints",
    )
    parser.add_argument("--shard_id", type=int, default=0, help="Shard id (for SLURM arrays)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--cache_dir", default="", help="Cache dir (default: next to output)")
    parser.add_argument(
        "--max_events_to_scan",
        type=int,
        default=200000,
        help="Hard cap on scanned candidate change events (prevents endless scanning)",
    )
    parser.add_argument(
        "--fetch_multiplier",
        type=int,
        default=20,
        help="How many raw statements to fetch per target example",
    )
    parser.add_argument(
        "--min_fetch_per_property",
        type=int,
        default=100,
        help="Minimum raw statements to fetch per property",
    )
    parser.add_argument(
        "--max_fetch_per_property",
        type=int,
        default=6000,
        help="Maximum raw statements to fetch per property",
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=200,
        help="WDQS page size (smaller is more robust)",
    )
    parser.add_argument(
        "--pages_per_property",
        type=int,
        default=None,
        help="Fetch exactly this many WDQS pages per property (preferred).",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Global WDQS page cap across all properties, e.g. --pages 7.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar (useful for SLURM logs)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Wikipedia language edition (default: en)",
    )
    parser.add_argument(
        "--include_time_in_question",
        action="store_true",
        help="If set, question explicitly contains 'as of {date}' (default: false)",
    )
    parser.add_argument(
        "--cache_only",
        action="store_true",
        help="Use only existing cache entries; skip all network calls",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run(
        out_path=str(out_path),
        limit=args.limit,
        properties=[p.strip() for p in args.properties.split(",") if p.strip()],
        min_gap_days=args.min_gap_days,
        wiki_lang=args.lang,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        cache_dir=(args.cache_dir or None),
        max_events_to_scan=args.max_events_to_scan,
        progress=(not args.no_progress),
        fetch_multiplier=args.fetch_multiplier,
        min_fetch_per_property=args.min_fetch_per_property,
        max_fetch_per_property=args.max_fetch_per_property,
        page_size=args.page_size,
        pages_per_property=args.pages_per_property,
        pages=args.pages,
        include_time_in_question=args.include_time_in_question,
        cache_only=args.cache_only,
    )


if __name__ == "__main__":
    main()

