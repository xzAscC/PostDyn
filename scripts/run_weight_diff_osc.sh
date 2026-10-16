#!/usr/bin/env bash
#SBATCH --job-name=weight-diff
#SBATCH --account=PAS2324
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/weights/slurm-%j.out
#SBATCH --error=logs/weights/slurm-%j.err

# Usage: sbatch [--time ...] scripts/run_weight_diff_osc.sh --which final --runs olmo3-7b_sft ...
# Checkpoints come from the scratch Hugging Face cache filled by scripts/download_checkpoints.py.
set -euo pipefail
cd "$HOME/Documents/PostDyn"

SCRATCH=/fs/scratch/PAS2324/zhu.3944
export HF_HOME="${POSTDYN_HF_HOME:-$SCRATCH/hf}"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export HF_HUB_OFFLINE=1
mkdir -p logs/weights

# The torch wheel bundles its CUDA runtime; the module is only loaded where it exists (not on Cardinal).
module load cuda/12.4.1 2>/dev/null || true

uv run python scripts/run_weight_diff.py --output logs/weights --basis-root "$SCRATCH/postdyn_basis" "$@"

echo "=== weight diff complete $(date -Is) ==="
