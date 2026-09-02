#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

log() { echo -e "\033[1;34m[$(date +%H:%M:%S)]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

usage() {
    cat << 'USAGE'
Usage: experiments/run_concept_dynamics.sh [MODE]

Runs the Concept Dynamics experiment: DiM concept extraction across
Olmo-3-7B post-training variants, then stability + gram analysis.

Modes:
  full     Full experiment: 6 trajectories x N concepts x 10 layers x 50 samples (default)
  quick    Smoke test:      1 model  x 2 concepts x 2 layers x 5 samples

Options (override defaults):
  --models M1,M2,...        Comma-separated OLMO3_VARIANTS keys
  --concepts C1,C2,...      Comma-separated concept names
  --layers L1,L2,...        Comma-separated layer indices
  --n-samples N             Samples per concept per class
  --output DIR              Output directory
  --max-seq-len N           Max tokenization length
  --[no-]chat-template      Toggle tokenizer.apply_chat_template wrapping

Environment:
  OUTPUT_DIR  Output directory (full default: results/concept_dynamics_multi;
              quick default: results/concept_dynamics_multi_quick)

Examples:
  experiments/run_concept_dynamics.sh quick
  experiments/run_concept_dynamics.sh full --models olmo3-think-sft,olmo3-rl-zero-math
  experiments/run_concept_dynamics.sh --concepts code_python_vs_cpp,gender_she_vs_he

USAGE
}

main() {
    local mode="full"
    local output_dir
    local -a passthrough=()

    if [[ $# -gt 0 ]]; then
        case "$1" in
            full|quick|help|--help|-h)
                mode="$1"
                shift
                ;;
        esac
    fi
    passthrough=("$@")

    case "$mode" in
        full)
            output_dir="${OUTPUT_DIR:-results/concept_dynamics_multi}"
            log "Running FULL concept dynamics experiment"
            uv run python experiments/run_concept_dynamics.py \
                --output "$output_dir" "${passthrough[@]}"
            ;;
        quick)
            output_dir="${OUTPUT_DIR:-results/concept_dynamics_multi_quick}"
            log "Running QUICK concept dynamics (smoke test)"
            uv run python experiments/run_concept_dynamics.py \
                --quick --output "$output_dir" "${passthrough[@]}"
            ;;
        help|--help|-h)
            usage
            return 0
            ;;
        *)
            err "Unknown mode: ${mode}"
            usage
            exit 1
            ;;
    esac

    log "Results saved to ${output_dir}/"
    log "  Concept vectors: ${output_dir}/vectors/"
    log "  Stability:       ${output_dir}/stability/stability.json"
    log "  Gram:            ${output_dir}/gram/gram.json"
    log "  Summary:         ${output_dir}/extraction_results.json"
}

main "$@"
