from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Make JSONL 'id' consistently a string.")
    parser.add_argument("--path", type=Path, required=True, help="Path to JSONL file to rewrite in-place")
    args = parser.parse_args()

    in_path: Path = args.path
    tmp_path = in_path.with_suffix(in_path.suffix + ".tmp")

    num_in = 0
    num_out = 0
    num_fixed = 0

    with in_path.open("r", encoding="utf-8") as fin, tmp_path.open("w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, start=1):
            s = raw.strip()
            if not s:
                continue
            num_in += 1
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {line_no}: {e}") from e

            if "id" in obj and not isinstance(obj["id"], str):
                obj["id"] = str(obj["id"])
                num_fixed += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            num_out += 1

    tmp_path.replace(in_path)
    print(f"Rewrote {in_path} ({num_in} lines -> {num_out} lines). Fixed id type in {num_fixed} rows.")


if __name__ == "__main__":
    main()

