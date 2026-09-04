#!/usr/bin/env bash
# Sequential 7B pipeline on the local GPU queue host (RTX 4090, bf16).
# Stages: Q1 full extraction -> Q1 robustness -> Q2 exp1 -> exp2 -> exp3.
# Every stage resumes from its incremental artifacts, so re-running the
# wrapper after an interruption continues where it stopped.
#
# Usage:
#   gpu-queue add postdyn-7b-overnight bash scripts/run_7b_overnight.sh
#   REPEATS=5 gpu-queue add postdyn-7b-overnight bash scripts/run_7b_overnight.sh
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."

REPEATS="${REPEATS:-5}"
LOGDIR=logs
mkdir -p "$LOGDIR"

run_stage() {
    local name="$1"
    shift
    echo "=== [${name}] start $(date -Is) ===" | tee -a "${LOGDIR}/overnight_7b.log"
    if "$@" 2>&1 | tee -a "${LOGDIR}/overnight_7b.log"; then
        echo "=== [${name}] done $(date -Is) ===" | tee -a "${LOGDIR}/overnight_7b.log"
    else
        echo "!!! [${name}] FAILED $(date -Is) — see ${LOGDIR}/overnight_7b.log" | tee -a "${LOGDIR}/overnight_7b.log"
        exit 1
    fi
}

# The GR domain has only 6,210 unique prompts (< n=3d). Q1 therefore fails
# closed unless the operator explicitly accepts the reduced GR sample:
#   ALLOW_SHORT_POOL=1 gpu-queue add postdyn-7b-overnight bash scripts/run_7b_overnight.sh
SHORT_POOL_FLAG=()
if [ "${ALLOW_SHORT_POOL:-0}" = "1" ]; then
    SHORT_POOL_FLAG=(--allow-short-pool)
fi

run_stage q1_full \
    uv run python scripts/run_q1.py --family 7b --scale full --output logs/q1/7b \
    "${SHORT_POOL_FLAG[@]}"

run_stage q1_robustness \
    uv run python scripts/run_q1_robustness.py --family 7b --repeats "${REPEATS}" \
    --output logs/q1_robustness/7b

run_stage q2_exp1 \
    uv run python scripts/run_q2_exp1.py --family 7b --q1-root logs/q1/7b \
    --output logs/q2/7b/exp1

run_stage q2_exp2 \
    uv run python scripts/run_q2_exp2.py --family 7b --q1-root logs/q1/7b \
    --exp1-output logs/q2/7b/exp1 --output logs/q2/7b/exp2

run_stage q2_exp3 \
    uv run python scripts/run_q2_exp3.py --family 7b --q1-root logs/q1/7b \
    --output logs/q2/7b/exp3

echo "=== overnight pipeline complete $(date -Is) ===" | tee -a "${LOGDIR}/overnight_7b.log"
