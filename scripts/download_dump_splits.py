from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    tqdm = None  # type: ignore


def _pick_enwiki_split(
    *,
    date: str,
    min_size_mb: int,
    max_size_mb: int,
    seed: int,
) -> Tuple[str, int]:
    u = f"https://dumps.wikimedia.org/enwiki/{date}/dumpstatus.json"
    js = requests.get(u, timeout=60).json()
    files = js["jobs"]["metahistorybz2dump"]["files"]
    cands: List[Tuple[str, int]] = []
    lo = min_size_mb * 1024 * 1024
    hi = max_size_mb * 1024 * 1024
    for fn, meta in files.items():
        if not fn.endswith(".bz2"):
            continue
        sz = int(meta.get("size", 0))
        if lo <= sz <= hi:
            cands.append((fn, sz))
    if not cands:
        raise RuntimeError("No enwiki metahistory split matches size filter.")
    rng = random.Random(seed)
    return rng.choice(cands)


def _download_file(url: str, out_path: Path, *, progress: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        bar = None
        if tqdm is not None:
            bar = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc=f"download:{out_path.name}",
                disable=not progress,
            )
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
        if bar is not None:
            bar.close()


def _stream_sample_wikidata(
    *,
    out_jsonl: Path,
    max_entities: int,
    bucket_mod: int,
    seed: int,
    progress: bool = True,
) -> Dict:
    url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
    rng = random.Random(seed)
    bucket = rng.randrange(bucket_mod)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    decomp = bz2.BZ2Decompressor()
    buf = b""
    kept = 0
    seen = 0

    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0") or 0)
        dl_bar = None
        keep_bar = None
        if tqdm is not None:
            dl_bar = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                desc="wikidata_download",
                disable=not progress,
            )
            keep_bar = tqdm(
                total=max_entities,
                unit="item",
                desc="wikidata_sample_kept",
                disable=not progress,
            )
        with out_jsonl.open("w", encoding="utf-8") as out_f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if dl_bar is not None:
                    dl_bar.update(len(chunk))
                data = decomp.decompress(chunk)
                if not data:
                    continue
                buf += data
                lines = buf.split(b"\n")
                buf = lines[-1]
                for raw in lines[:-1]:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line or line in {"[", "]"}:
                        continue
                    if line.endswith(","):
                        line = line[:-1]
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict) or obj.get("type") != "item":
                        continue
                    qid = obj.get("id", "")
                    if not qid:
                        continue
                    seen += 1
                    hv = int(hashlib.sha256(qid.encode("utf-8")).hexdigest(), 16)
                    if hv % bucket_mod != bucket:
                        continue
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    if keep_bar is not None:
                        keep_bar.update(1)
                        if kept % 500 == 0:
                            keep_bar.set_postfix_str(f"seen={seen}")
                    if kept >= max_entities:
                        if dl_bar is not None:
                            dl_bar.close()
                        if keep_bar is not None:
                            keep_bar.close()
                        return {
                            "source_url": url,
                            "bucket_mod": bucket_mod,
                            "bucket": bucket,
                            "seen_items": seen,
                            "kept_items": kept,
                        }
    if 'dl_bar' in locals() and dl_bar is not None:
        dl_bar.close()
    if 'keep_bar' in locals() and keep_bar is not None:
        keep_bar.close()
    return {
        "source_url": url,
        "bucket_mod": bucket_mod,
        "bucket": bucket,
        "seen_items": seen,
        "kept_items": kept,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Download random dump splits for offline pipeline.")
    p.add_argument("--out_dir", default="data", help="Output directory")
    p.add_argument("--enwiki_date", default="20260201", help="enwiki dump date folder")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--enwiki_min_mb", type=int, default=80, help="Min enwiki split size in MB")
    p.add_argument("--enwiki_max_mb", type=int, default=400, help="Max enwiki split size in MB")
    p.add_argument("--wikidata_max_entities", type=int, default=20000, help="Max sampled wikidata entities")
    p.add_argument("--wikidata_bucket_mod", type=int, default=20, help="Hash bucket mod for random split")
    p.add_argument("--no_progress", action="store_true", help="Disable progress bars")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) enwiki random history split
    en_fn, en_size = _pick_enwiki_split(
        date=args.enwiki_date,
        min_size_mb=args.enwiki_min_mb,
        max_size_mb=args.enwiki_max_mb,
        seed=args.seed,
    )
    en_url = f"https://dumps.wikimedia.org/enwiki/{args.enwiki_date}/{en_fn}"
    en_out = out_dir / en_fn
    print(f"[download] enwiki split: {en_fn} ({en_size} bytes)")
    _download_file(en_url, en_out, progress=(not args.no_progress))

    # 2) wikidata sampled split from latest-all
    wd_out = out_dir / f"wikidata-latest-all.sample-{args.wikidata_max_entities}.jsonl"
    print(f"[download] wikidata sampled split -> {wd_out.name}")
    wd_meta = _stream_sample_wikidata(
        out_jsonl=wd_out,
        max_entities=args.wikidata_max_entities,
        bucket_mod=args.wikidata_bucket_mod,
        seed=args.seed,
        progress=(not args.no_progress),
    )

    meta = {
        "enwiki_split_file": en_fn,
        "enwiki_url": en_url,
        "enwiki_size_bytes": en_size,
        "wikidata_sample_file": wd_out.name,
        "wikidata_sampling": wd_meta,
    }
    meta_path = out_dir / "dump_splits_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()

