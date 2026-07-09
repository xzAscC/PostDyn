#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-results/concept_dynamics}"

log() { echo -e "\033[1;34m[$(date +%H:%M:%S)]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

usage() {
    cat << 'USAGE'
Usage: experiments/run_concept_dynamics.sh [MODE]

Runs the Concept Dynamics experiment: DiM concept extraction across
Olmo-3-7B post-training variants, then stability + gram analysis.

Modes:
  full     Full experiment: 7 models × 4 concepts × 10 layers × 50 samples (default)
  quick    Smoke test:      1 model  × 2 concepts × 2 layers × 5 samples

Options (override defaults):
  --models M1,M2,...        Comma-separated OLMO3_VARIANTS keys
  --concepts C1,C2,...      Comma-separated concept names
  --layers L1,L2,...        Comma-separated layer indices
  --n-samples N             Samples per concept per class
  --output DIR              Output directory
  --max-seq-len N           Max tokenization length

Environment:
  OUTPUT_DIR  Output directory (default: results/concept_dynamics)

Examples:
  experiments/run_concept_dynamics.sh quick
  experiments/run_concept_dynamics.sh full --models olmo3-think-sft,olmo3-rl-zero-math
  experiments/run_concept_dynamics.sh --concepts math,code --n-samples 100

USAGE
}

main() {
    local mode="${1:-full}"
    shift || true

    case "$mode" in
        full)
            log "Running FULL concept dynamics experiment"
            uv run python experiments/run_concept_dynamics.py \
                --output "$OUTPUT_DIR" "$@"
            ;;
        quick)
            log "Running QUICK concept dynamics (smoke test)"
            uv run python experiments/run_concept_dynamics.py \
                --quick --output "$OUTPUT_DIR" "$@"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            err "Unknown mode: ${mode}"
            usage
            exit 1
            ;;
    esac

    log "Results saved to ${OUTPUT_DIR}/"
    log "  Concept vectors: ${OUTPUT_DIR}/vectors/"
    log "  Stability:       ${OUTPUT_DIR}/stability/stability.json"
    log "  Gram:            ${OUTPUT_DIR}/gram/gram.json"
    log "  Summary:         ${OUTPUT_DIR}/extraction_results.json"
}

main "$@"
