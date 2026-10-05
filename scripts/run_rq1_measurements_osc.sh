#!/usr/bin/env bash
#SBATCH --job-name=rq1-meas
#SBATCH --account=PAS2324
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=96G
#SBATCH --output=logs/rq1_measurements/slurm-%j.out
#SBATCH --error=logs/rq1_measurements/slurm-%j.err

set -euo pipefail
cd "$HOME/Documents/PostDyn"

export HF_HOME="${POSTDYN_HF_HOME:-$PWD/hf_cache}"
mkdir -p "$HF_HOME" logs/rq1_measurements

module load cuda/12.4.1

uv sync --group dev

if [ ! -f data/domain_prompts/math.json ]; then
    SNAP=$(uv run python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("allenai/Dolci-Think-SFT-7B", repo_type="dataset"))')
    uv run python scripts/materialize_pools.py --snapshot-dir "$SNAP"
fi

uv run python scripts/run_rq1_measurements.py --output logs/rq1_measurements

echo "=== RQ1 measurements complete $(date -Is) ==="
