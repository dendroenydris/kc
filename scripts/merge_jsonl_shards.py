from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="Directory with shard JSONL files")
    p.add_argument("--out", required=True, help="Output merged JSONL file")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shard_files = sorted([f for f in in_dir.glob("*.jsonl") if f.is_file()])
    if not shard_files:
        raise SystemExit(f"No .jsonl shards found in {in_dir}")

    with out_path.open("w", encoding="utf-8") as out_f:
        for sf in shard_files:
            with sf.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)


if __name__ == "__main__":
    main()

