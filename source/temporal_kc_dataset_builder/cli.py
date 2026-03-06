from __future__ import annotations

import json
import sys
from pathlib import Path

from .builder import build_examples


def run(
    *,
    out_path: str,
    limit: int,
    properties: list[str],
    min_gap_days: int,
    wiki_lang: str,
    shard_id: int = 0,
    num_shards: int = 1,
    cache_dir: str | None = None,
    max_events_to_scan: int = 200000,
    progress: bool | None = None,
    fetch_multiplier: int = 20,
    min_fetch_per_property: int = 100,
    max_fetch_per_property: int = 6000,
    page_size: int = 200,
    pages_per_property: int | None = None,
    pages: int | None = None,
    include_time_in_question: bool = False,
    cache_only: bool = False,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = cache_dir or str(out.parent / ".cache_temporal_kc")
    user_agent = "knowledge-temporal-kc-dataset-builder/0.1 (contact: local)"

    if progress is None:
        progress = sys.stderr.isatty()

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # noqa: BLE001
        tqdm = None  # type: ignore

    print(
        f"[temporal-kc] writing up to {limit} examples → {out_path}",
        file=sys.stderr,
    )
    print(
        f"[temporal-kc] properties={properties} min_gap_days={min_gap_days} "
        f"shard={shard_id}/{num_shards} cache_dir={cache_dir} cache_only={cache_only} "
        f"pages_per_property={pages_per_property} pages={pages}",
        file=sys.stderr,
    )

    fetch_bar = None
    rev_bar = None
    write_bar = None

    with out.open("w", encoding="utf-8") as f:
        expected_pages = None
        if pages is not None:
            expected_pages = pages
        elif pages_per_property is not None:
            expected_pages = len(properties) * pages_per_property

        if tqdm is not None:
            fetch_bar = tqdm(
                total=expected_pages,
                desc="wdqs_pages",
                unit="page",
                disable=not progress,
                position=0,
            )
            rev_bar = tqdm(
                total=limit * 2,
                desc="wiki_revisions",
                unit="req",
                disable=not progress,
                position=1,
            )

            write_bar = tqdm(
                total=limit,
                desc="examples",
                unit="ex",
                disable=not progress,
                position=2,
            )

        def on_progress(event: str, payload: dict) -> None:
            if fetch_bar is not None and event == "page_done":
                fetch_bar.update(1)
                rows = payload.get("rows", 0)
                pid = payload.get("pid", "?")
                fetch_bar.set_postfix_str(f"pid={pid} rows={rows}")

            if rev_bar is not None and event == "revision_fetch":
                rev_bar.update(1)
                ok = payload.get("ok", False)
                rev_bar.set_postfix_str(f"ok={ok}")

            if write_bar is not None and event == "example_written":
                target = int(payload.get("written", 0))
                delta = target - int(write_bar.n)
                if delta > 0:
                    write_bar.update(delta)

        it = build_examples(
            limit=limit,
            property_pids=properties,
            min_gap_days=min_gap_days,
            wiki_lang=wiki_lang,
            user_agent=user_agent,
            cache_dir=cache_dir,
            shard_id=shard_id,
            num_shards=num_shards,
            max_events_to_scan=max_events_to_scan,
            fetch_multiplier=fetch_multiplier,
            min_fetch_per_property=min_fetch_per_property,
            max_fetch_per_property=max_fetch_per_property,
            page_size=page_size,
            pages_per_property=pages_per_property,
            pages=pages,
            include_time_in_question=include_time_in_question,
            progress_callback=on_progress,
            cache_only=cache_only,
        )

        wrote = 0
        for ex in it:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
            wrote += 1

        if fetch_bar is not None:
            fetch_bar.close()
        if rev_bar is not None:
            rev_bar.close()
        if write_bar is not None:
            # ensure bar reaches wrote (in case callback updates were suppressed)
            delta = wrote - int(write_bar.n)
            if delta > 0:
                write_bar.update(delta)
            write_bar.close()

        print(f"[temporal-kc] done. wrote={wrote}", file=sys.stderr)

