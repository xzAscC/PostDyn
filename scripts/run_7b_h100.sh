#!/usr/bin/env bash
# Intended for the REMOTE H100 box. On the local RTX 4090, use
# scripts/run_7b_overnight.sh via gpu-queue instead.
# Q1 streams checkpoints with one-deep prefetch and pruning; Q2 finals download
# on demand into HF_HOME. HF_HOME keeps all downloads in the local cache.
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${POSTDYN_HF_HOME:-$PWD/hf_cache}"
RUN=""
if [ "${DRY_RUN:-0}" = "1" ]; then
    RUN="echo"
else
    mkdir -p "$HF_HOME" logs
fi

LOGFILE="logs/h100_7b.log"
REPEATS="${REPEATS:-5}"
SHORT_POOL_FLAG=()
if [ "${ALLOW_SHORT_POOL:-1}" = "1" ]; then
    SHORT_POOL_FLAG=(--allow-short-pool)
fi
UPLOAD_FLAG=()
if [ -n "${POSTDYN_UPLOAD_TO:-}" ]; then
    UPLOAD_FLAG=(--upload-to "$POSTDYN_UPLOAD_TO")
fi

run_stage() {
    local name="$1"
    shift
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] stage: ${name}"
        printf '%s ' "$@"
        printf '\n'
        return 0
    fi
    echo "=== [${name}] start $(date -Is) ===" | tee -a "$LOGFILE"
    if "$@" 2>&1 | tee -a "$LOGFILE"; then
        echo "=== [${name}] done $(date -Is) ===" | tee -a "$LOGFILE"
    else
        echo "!!! [${name}] FAILED $(date -Is) — see $LOGFILE" | tee -a "$LOGFILE"
        exit 1
    fi
}

run_stage setup $RUN uv sync --group dev
echo "HF_HOME=$HF_HOME"

DATASET_STAGE='
if [ "${DRY_RUN:-0}" = "1" ]; then
    SNAP="__DOLCI_SNAPSHOT__"
else
    SNAP=$(uv run python -c '\''from huggingface_hub import snapshot_download; print(snapshot_download("allenai/Dolci-Think-SFT-7B", repo_type="dataset"))'\'')
fi
if [ ! -f data/domain_prompts/math.json ] || [ ! -f data/domain_prompts/code.json ] || [ ! -f data/domain_prompts/instruction_following.json ] || [ ! -f data/domain_prompts/general_reasoning.json ]; then
    uv run python scripts/materialize_pools.py --snapshot-dir "$SNAP"
fi
'
run_stage dataset_and_pools bash -c "$DATASET_STAGE"

run_stage q1_full $RUN uv run python scripts/run_q1.py --family 7b --scale full \
    --dtype bfloat16 --device cuda --output logs/q1/7b \
    "${SHORT_POOL_FLAG[@]}" "${UPLOAD_FLAG[@]}"
run_stage q1_robustness $RUN uv run python scripts/run_q1_robustness.py --family 7b \
    --repeats "$REPEATS" --dtype bfloat16 --device cuda \
    --output logs/q1_robustness/7b "${UPLOAD_FLAG[@]}"
run_stage q2_rlvr $RUN uv run python scripts/run_q2_model.py --family 7b \
    --q1-root logs/q1/7b --model rlvr --output-root logs/q2/7b \
    "${UPLOAD_FLAG[@]}"
run_stage q2_sft $RUN uv run python scripts/run_q2_model.py --family 7b \
    --q1-root logs/q1/7b --model sft --output-root logs/q2/7b \
    "${UPLOAD_FLAG[@]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "=== H100 7B pipeline complete (dry-run) ==="
else
    echo "=== H100 7B pipeline complete $(date -Is) ===" | tee -a "$LOGFILE"
fi
