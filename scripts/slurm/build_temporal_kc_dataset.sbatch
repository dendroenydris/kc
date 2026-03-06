#!/bin/bash
#SBATCH --job-name=temporal_kc_build
#SBATCH --output=logs/temporal_kc_build_%A_%a.out
#SBATCH --error=logs/temporal_kc_build_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --array=0-9

set -euo pipefail

# Adjust these:
OUT_DIR="data/shards"
N_SHARDS=10
LIMIT_PER_SHARD=500
MIN_GAP_DAYS=90
PROPS="P6,P35,P169"
WIKI_LANG="en"

mkdir -p "${OUT_DIR}" logs

# Conda activation (cluster-specific; change if needed)
source ~/.bashrc || true
conda activate knowledge-temporal-kc

# Make source importable
export PYTHONPATH="$(pwd)/source:${PYTHONPATH:-}"

SHARD_ID="${SLURM_ARRAY_TASK_ID}"
OUT_PATH="${OUT_DIR}/temporal_kc_shard_${SHARD_ID}.jsonl"

python scripts/build_temporal_kc_dataset.py \
  --out "${OUT_PATH}" \
  --limit "${LIMIT_PER_SHARD}" \
  --properties "${PROPS}" \
  --min_gap_days "${MIN_GAP_DAYS}" \
  --lang "${WIKI_LANG}" \
  --shard_id "${SHARD_ID}" \
  --num_shards "${N_SHARDS}" \
  --cache_dir "data/.cache_temporal_kc"

