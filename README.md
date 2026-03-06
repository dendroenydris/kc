# Temporal Knowledge Conflict (Code)

This repository contains code for temporal knowledge conflict dataset construction and TATM diagnosis experiments.

## Paper

Paper link: `paperlink`

## Dataset

Main processed sample used in local diagnosis:

- `data/processed/temporal_sample_20.jsonl`

Dataset records include:
`question`, `t_old`, `t_new`, `answer_old`, `answer_new`, `evidence_old`, `evidence_new`, `provenance_old`, `provenance_new`.

## TATM Diagnosis

Main script:

- `source/tatm_diagnosis/tatm_experiments.py`

Recommended low-memory run:

```bash
conda activate ml
python source/tatm_diagnosis/tatm_experiments.py --device cpu --max-samples 2 --skip-f2f3
```

## Dataset Builder

Quick build command:

```bash
python scripts/build_temporal_kc_dataset.py --out data/temporal_kc_sample.jsonl --pages_per_property 5 --limit 500
```

Offline (dump-based) build:

```bash
python scripts/build_temporal_kc_from_dumps.py --wikidata_dump "/path/to/wikidata.json.bz2" --wikipedia_dump "/path/to/enwiki.xml.bz2" --out data/temporal_kc_offline.jsonl --pages 7 --limit 500
```

## Environment

Use Python 3.11 with:

```bash
conda env create -f environment.yml
conda activate knowledge-temporal-kc
```

## References


