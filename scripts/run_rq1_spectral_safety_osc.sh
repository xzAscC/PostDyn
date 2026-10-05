#!/usr/bin/env bash
#SBATCH --job-name=rq1-safety
#SBATCH --account=PAS2324
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/rq1_spectral_safety/slurm-%j.out
#SBATCH --error=logs/rq1_spectral_safety/slurm-%j.err

set -euo pipefail
cd "$HOME/Documents/PostDyn"

export HF_HOME="${POSTDYN_HF_HOME:-$PWD/hf_cache}"
mkdir -p "$HF_HOME" logs/rq1_spectral_safety

module load cuda/12.4.1

uv sync --group dev

uv run python scripts/run_rq1_spectral_safety.py \
    --batch-size 8 \
    --max-length 2048 \
    --token-budget 4096 \
    --attention-budget 8388608 \
    --dtype bfloat16 \
    --device cuda \
    --output logs/rq1_spectral_safety

echo "=== RQ1 spectral safety complete $(date -Is) ==="
