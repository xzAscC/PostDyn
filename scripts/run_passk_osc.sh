#!/usr/bin/env bash
#SBATCH --job-name=passk
#SBATCH --account=PAS2324
#SBATCH --partition=debug-nextgen
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/ess/PAS2324/zhu.3944/postdyn_passk/logs/passk-%j.out
#SBATCH --error=/fs/ess/PAS2324/zhu.3944/postdyn_passk/logs/passk-%j.err

# Sample one condition of the Olmo 3 7B pass@k comparison; a job cut off by its limit resumes
# when submitted again with the same settings.
# Conditions: sft-chat (Think-SFT on its chat template), base-simple (base on the RL-Zero prompt).
# The base has no chat template and writes noise under the Think one, so it never gets it.
# Usage: CONDITION=sft-chat RUN=passk64 SCORE=1 sbatch scripts/run_passk_osc.sh --limit 4 --samples 16
set -euo pipefail

STORE=/fs/ess/PAS2324/zhu.3944
ROOT="$STORE/postdyn_passk"
REPO="${REPO:-$ROOT/code}"
HUB=/fs/scratch/PAS2324/zhu.3944/hf/hub
snapshot() { echo "$HUB/models--allenai--$1/snapshots/$(cat "$HUB/models--allenai--$1/refs/main")"; }

TOKENIZER="$(snapshot Olmo-3-7B-Think-SFT)"
case "$CONDITION" in
    sft-chat) MODEL="$TOKENIZER" FORMAT=chat ;;
    base-simple) MODEL="$(snapshot Olmo-3-1025-7B)" FORMAT=simple ;;
    *) echo "unknown CONDITION: $CONDITION" >&2; exit 2 ;;
esac
LCB="$(echo /fs/scratch/PAS2324/zhu.3944/hf/hub/datasets--livecodebench--code_generation_lite/snapshots/*)"
OUT="$ROOT/${RUN:?set RUN}/$CONDITION"

export HF_HOME=/fs/scratch/PAS2324/zhu.3944/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# flashinfer's sampler is compiled at run time and needs cuRAND headers that OSC's CUDA lacks.
export VLLM_USE_FLASHINFER_SAMPLER=0 CC=gcc CXX=g++
unset ROCR_VISIBLE_DEVICES
set +u  # the venv activate script reads unset variables
# shellcheck disable=SC1091
source "$STORE/verl/.venv/bin/activate"
set -u

echo "job ${SLURM_JOB_ID:-local} on $(hostname): $CONDITION -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$REPO/scripts/run_passk.py" generate --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt-format "$FORMAT" --lcb-source "$LCB" --out-dir "$OUT" "$@"
if [ "${SCORE:-0}" = 1 ]; then
    python "$REPO/scripts/run_passk.py" score --out-dir "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}"
fi
echo "=== passk done $(date -Is) ==="
