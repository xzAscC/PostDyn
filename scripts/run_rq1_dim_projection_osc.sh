#!/usr/bin/env bash
#SBATCH --job-name=rq1-dim-proj
#SBATCH --account=PAS2324
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/rq1_dim_projection/slurm-%j.out
#SBATCH --error=logs/rq1_dim_projection/slurm-%j.err

set -euo pipefail
cd "$HOME/Documents/PostDyn"

export HF_HOME="${POSTDYN_HF_HOME:-$PWD/hf_cache}"
mkdir -p "$HF_HOME" logs/rq1_dim_projection

module load cuda/12.4.1

uv sync --group dev

# Materialize domain pools if missing
if [ ! -f data/domain_prompts/math.json ]; then
    SNAP=$(uv run python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("allenai/Dolci-Think-SFT-7B", repo_type="dataset"))')
    uv run python scripts/materialize_pools.py --snapshot-dir "$SNAP"
fi

uv run python scripts/run_rq1_dim_projection.py \
    --n 5000 \
    --batch-size 8 \
    --max-length 2048 \
    --token-budget 4096 \
    --attention-budget 8388608 \
    --dtype bfloat16 \
    --device cuda \
    --output logs/rq1_dim_projection

echo "=== RQ1 DiM projection stats complete $(date -Is) ==="
