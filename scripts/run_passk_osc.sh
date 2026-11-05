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
# Conditions: sft-chat (Think-SFT on its chat template), base-plain (base on the same user text,
# no template; the base has no chat template and writes noise under the Think one), and six
# replacement models on the SFT prompt: {attn,mlp}-{front,middle,back} put that component of that
# third of the layers back at the base weights (src/postdyn/hybrid.py).
# RL-Zero (Olmo-3-7B-RL-Zero-Math, RL straight from the base): rlzero-plain, and
# rlzero-{attn,mlp}-{front,middle,back} with that part back at the base weights. All of them get
# base-plain's prompt, so they differ from base-plain only in the weights.
# CONDITION=auto picks when the job starts, for jobs that queue for hours, from AUTO_CONDITIONS
# (default: the SFT conditions).
# Usage: CONDITION=sft-chat RUN=passk64 SCORE=1 sbatch scripts/run_passk_osc.sh --limit 4 --samples 16
set -euo pipefail

STORE=/fs/ess/PAS2324/zhu.3944
ROOT="$STORE/postdyn_passk"
REPO="${REPO:-$ROOT/code}"
HUB=/fs/scratch/PAS2324/zhu.3944/hf/hub
snapshot() { echo "$HUB/models--allenai--$1/snapshots/$(cat "$HUB/models--allenai--$1/refs/main")"; }
# The SFT replacement models were written to ess; the RL-Zero ones go to scratch.
hybrid_dir() {
    case "$1" in
        rlzero-*) echo "/fs/scratch/PAS2324/zhu.3944/postdyn_passk/hybrids/$1" ;;
        *) echo "$ROOT/hybrids/$1" ;;
    esac
}

# The condition furthest behind: a claim is a job at work, worth about 20 problems of its run.
pick_condition() {
    local best="" best_score=-1 condition done_count score
    for condition in ${AUTO_CONDITIONS:-sft-chat base-plain attn-front attn-middle attn-back mlp-front mlp-middle mlp-back}; do
        case "$condition" in *-chat | *-plain) ;; *) [ -d "$(hybrid_dir "$condition")" ] || continue ;; esac
        done_count=$(ls "$ROOT/$RUN/$condition/$AUTO_TASK" 2>/dev/null | grep -c 'gz$' || true)
        [ "$done_count" -lt "$AUTO_TARGET" ] || continue
        score=$((done_count + 20 * $(ls "$ROOT/$RUN/$condition/$AUTO_TASK" 2>/dev/null | grep -c 'claim$' || true)))
        if [ "$best_score" -lt 0 ] || [ "$score" -lt "$best_score" ]; then
            best=$condition best_score=$score
        fi
    done
    echo "$best"
}

# Every condition uses the SFT tokenizer: plain prompts take only its stop tokens from it.
TOKENIZER="$(snapshot Olmo-3-7B-Think-SFT)"
BASE="$(snapshot Olmo-3-1025-7B)"
RLZERO="$(snapshot Olmo-3-7B-RL-Zero-Math)"
if [ "$CONDITION" = auto ]; then
    CONDITION=$(pick_condition)
    [ -n "$CONDITION" ] || { echo "nothing left to do"; exit 0; }
fi
case "$CONDITION" in
    sft-chat) MODEL="$TOKENIZER" FORMAT=chat ;;
    base-plain) MODEL="$BASE" FORMAT=plain ;;
    rlzero-plain) MODEL="$RLZERO" FORMAT=plain ;;
    attn-front|attn-middle|attn-back|mlp-front|mlp-middle|mlp-back)
        TRAINED="$TOKENIZER" MODEL="$(hybrid_dir "$CONDITION")" FORMAT=chat ;;
    rlzero-attn-front|rlzero-attn-middle|rlzero-attn-back|rlzero-mlp-front|rlzero-mlp-middle|rlzero-mlp-back)
        TRAINED="$RLZERO" MODEL="$(hybrid_dir "$CONDITION")" FORMAT=plain ;;
    *) echo "unknown CONDITION: $CONDITION" >&2; exit 2 ;;
esac
LCB="$(echo /fs/scratch/PAS2324/zhu.3944/hf/hub/datasets--livecodebench--code_generation_lite/snapshots/*)"
OUT="$ROOT/${RUN:?set RUN}/$CONDITION"

export HF_HOME=/fs/scratch/PAS2324/zhu.3944/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# flashinfer's sampler is compiled at run time and needs cuRAND headers that OSC's CUDA lacks.
export VLLM_USE_FLASHINFER_SAMPLER=0 CC=gcc CXX=g++
unset ROCR_VISIBLE_DEVICES
set +u  # the venv activate script reads unset variables
# vllm==0.11.2 with transformers==4.57.1: later vLLM runs Olmo 3 through transformers 5, which
# breaks it (see check_transformers in run_passk.py).
# shellcheck disable=SC1091
source "$ROOT/venv-vllm/bin/activate"
set -u

echo "job ${SLURM_JOB_ID:-local} on $(hostname): $CONDITION -> $OUT"
case "$CONDITION" in
    *-chat | *-plain) ;;
    *) [ -d "$MODEL" ] || python "$REPO/scripts/make_passk_hybrid.py" "$TRAINED" "$BASE" "$MODEL" \
        "${CONDITION#rlzero-}" ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$REPO/scripts/run_passk.py" generate --model "$MODEL" --tokenizer "$TOKENIZER" \
    --prompt-format "$FORMAT" --lcb-source "$LCB" --out-dir "$OUT" "$@"
if [ "${SCORE:-0}" = 1 ]; then
    python "$REPO/scripts/run_passk.py" score --out-dir "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}"
fi
echo "=== passk done $(date -Is) ==="
