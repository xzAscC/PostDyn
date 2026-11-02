#!/usr/bin/env bash
#SBATCH --job-name=spectrum
#SBATCH --account=PAS2324
#SBATCH --partition=debug-nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/ess/PAS2324/zhu.3944/postdyn_passk/logs/spectrum-%j.out
#SBATCH --error=/fs/ess/PAS2324/zhu.3944/postdyn_passk/logs/spectrum-%j.err

# One step of the spectrum ablation (see scripts/run_spectrum.py). The same command resumes a job
# cut off by its limit. Moments and maps go to scratch: 4 GB per model, text and fold set.
# Usage: sbatch scripts/run_spectrum_osc.sh moments --problems 200
#        sbatch scripts/run_spectrum_osc.sh ablate --kind reorder
set -euo pipefail

STORE=/fs/ess/PAS2324/zhu.3944
ROOT="$STORE/postdyn_passk"
REPO="${REPO:-$ROOT/code-spectrum}"
HUB=/fs/scratch/PAS2324/zhu.3944/hf/hub
OUT="${OUT:-/fs/scratch/PAS2324/zhu.3944/postdyn_spectrum/toy200}"
# Scores are written next to the answers, so point this at links to them, not at the shared run.
PASS_ROOT="${PASS_ROOT:-$OUT/pass}"
snapshot() { echo "$HUB/models--allenai--$1/snapshots/$(cat "$HUB/models--allenai--$1/refs/main")"; }

COMMAND=$1
shift
export HF_HOME=/fs/scratch/PAS2324/zhu.3944/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# flashinfer's sampler is compiled at run time and needs cuRAND headers that OSC's CUDA lacks.
export VLLM_USE_FLASHINFER_SAMPLER=0 CC=gcc CXX=g++
unset ROCR_VISIBLE_DEVICES
set +u  # the venv activate script reads unset variables
# shellcheck disable=SC1091
# transformers 4.57.1, as run_passk_osc.sh: 5.x breaks Olmo 3.
source "$ROOT/venv-vllm/bin/activate"
set -u

case "$COMMAND" in
    moments | ablate) MODELS=(--pass-root "$PASS_ROOT" --base-model "$(snapshot Olmo-3-1025-7B)"
        --post-model "$(snapshot Olmo-3-7B-Think-SFT)") ;;
    select) MODELS=(--pass-root "$PASS_ROOT") ;;
    report) MODELS=() ;;
    *) echo "unknown command: $COMMAND" >&2; exit 2 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$REPO/scripts/run_spectrum.py" "$COMMAND" "${MODELS[@]}" --out "$OUT" "$@"
echo "=== spectrum $COMMAND done $(date -Is) ==="
