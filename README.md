# Code

This folder is intended to become a standalone GitHub repository for experiments and dataset tooling.

## Setup

```bash
conda env create -f environment.yml
conda activate knowledge-temporal-kc
```

## Build a temporal conflict dataset (Wikidata + Wikipedia revisions)

This builder mines **time-scoped Wikidata statements** (with start/end time qualifiers), then fetches
Wikipedia revisions near two timepoints to extract `evidence_old` and `evidence_new`.

### Quick start (small sample)

```bash
python scripts/build_temporal_kc_dataset.py \
  --out data/temporal_kc_sample.jsonl \
  --pages_per_property 5 \
  --limit 500
```

Output is JSONL with fields like:
`question`, `t_old`, `t_new`, `answer_old`, `answer_new`, `evidence_old`, `evidence_new`, `provenance_*`.

By default, `question` is **time-implicit** (no date in text).  
Temporal condition is provided via metadata (`t_old`/`t_new`).

If you want time-explicit question text, add:

```bash
--include_time_in_question
```

Progress display uses two bars by default:
- `wdqs_pages`: Wikidata/SPARQL page fetching progress
- `wiki_revisions`: Wikipedia revision fetch progress
- `examples`: written dataset examples

`wdqs_pages` total is driven by `--pages_per_property` (user-specified).  
If `--pages_per_property` is not set, the page bar runs with unknown total.

If you want a hard global cap across all properties, use `--pages`:

```bash
python scripts/build_temporal_kc_dataset.py \
  --out data/temporal_kc_capped.jsonl \
  --pages 7
```

### Cache-only mode (no network)

If upstream APIs are unstable or rate-limited, you can force cache-only execution:

```bash
python scripts/build_temporal_kc_dataset.py \
  --out data/temporal_kc_cached_only.jsonl \
  --limit 100 \
  --pages_per_property 5 \
  --cache_only
```

In this mode, cache misses are skipped (so final count may be below `--limit`).

## SLURM-friendly usage (job arrays / sharding)

This builder supports sharding so you can run it as a SLURM job array:

```bash
sbatch scripts/slurm/build_temporal_kc_dataset.sbatch
```

Each task writes a shard file:
`data/shards/temporal_kc_shard_${SLURM_ARRAY_TASK_ID}.jsonl`

## Offline dump pipeline (no online API)

Use this when you want zero dependency on WDQS / Wikipedia API limits.

```bash
python scripts/build_temporal_kc_from_dumps.py \
  --wikidata_dump "/path/to/wikidata-latest-all.json.bz2" \
  --wikipedia_dump "/path/to/enwiki-latest-pages-meta-history.xml.bz2" \
  --out data/temporal_kc_offline.jsonl \
  --pages 7 \
  --limit 500
```

Notes:
- `--pages` is a global sampling budget proxy for candidate extraction.
- This script is fully offline once dumps are local.
- Detailed progress bars are enabled by default for long stages:
  - `wikidata_entities`
  - `wikidata_label_scan`
  - `wikipedia_pages`
  - `build_examples`
- Disable with `--no_progress`.

