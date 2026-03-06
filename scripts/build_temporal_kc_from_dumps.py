from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from temporal_kc_dataset_builder.offline_dumps import build_dataset_from_dumps  # type: ignore[reportMissingImports]  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Build temporal conflict dataset from offline dumps only.")
    p.add_argument("--wikidata_dump", required=True, help="Path to Wikidata JSON dump (.json/.gz/.bz2)")
    p.add_argument("--wikipedia_dump", required=True, help="Path to Wikipedia XML revision dump (.xml/.gz/.bz2)")
    p.add_argument("--out", required=True, help="Output JSONL")
    p.add_argument("--pages", type=int, default=7, help="Global page budget proxy (default: 7)")
    p.add_argument("--page_size", type=int, default=200, help="Page size proxy (default: 200)")
    p.add_argument("--limit", type=int, default=500, help="Max output examples (default: 500)")
    p.add_argument("--min_gap_days", type=int, default=30, help="Minimum old/new day gap")
    p.add_argument("--properties", default="P6,P35,P169", help="Comma-separated property IDs")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    p.add_argument(
        "--include_time_in_question",
        action="store_true",
        help="Include 'as of {date}' in question text",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    props = [x.strip() for x in args.properties.split(",") if x.strip()]

    wrote = build_dataset_from_dumps(
        wikidata_dump_path=args.wikidata_dump,
        wikipedia_dump_path=args.wikipedia_dump,
        out_path=str(out_path),
        properties=props,
        pages=args.pages,
        page_size=args.page_size,
        limit=args.limit,
        min_gap_days=args.min_gap_days,
        include_time_in_question=args.include_time_in_question,
        progress=(not args.no_progress),
    )
    print(f"[temporal-kc-offline] done. wrote={wrote}")


if __name__ == "__main__":
    main()

