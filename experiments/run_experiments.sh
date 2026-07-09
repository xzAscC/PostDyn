#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "$LOG_DIR"

log() { echo -e "\033[1;34m[$(date +%H:%M:%S)]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

run_step() {
    local name="$1"; shift
    local logfile="${LOG_DIR}/${name}.log"
    log "START: ${name}"
    if "$@" > "$logfile" 2>&1; then
        log "  OK:   ${name} (log: ${logfile})"
    else
        err "FAIL:  ${name} (log: ${logfile})"
        tail -20 "$logfile" >&2
        return 1
    fi
}

# =============================================================================
# Experiment groups
# =============================================================================

exp_tests() {
    log "Running test suite"
    uv run pytest tests/ -q
}

exp_weight_rank() {
    log "Weight-level effective rank analysis (quick mode)"
    uv run python main.py --analysis all --quick
}

exp_activation_rank() {
    log "Activation-level effective rank analysis"
    uv run python main.py --analysis activation-all --quick --num-samples 500
}

exp_concept_steering() {
    log "Concept steering: select 100 concepts + DIM pipeline"
    uv run python -c "
from src.concept_steering import load_concept_index, select_concepts
concepts = load_concept_index()
selected = select_concepts(concepts, n=100, strategy='first')
print(f'Selected {len(selected)} concepts (first-100 by frequency)')
print(f'First 5: {selected[:5]}')
print(f'Last 5:  {selected[-5:]}')
"
}

exp_concept_analysis() {
    log "Concept analysis: 4 metrics on synthetic steering vectors"
    uv run python -c "
import torch
from src.concept_steering import compute_steering_vector
from src.concept_analysis import (
    directional_stability,
    separability_margin,
    concept_gram_matrix,
    anisotropy_spectrum,
)

torch.manual_seed(42)
concepts = [f'concept_{i}' for i in range(10)]
d_model = 64

# Build mock steering vectors for 3 checkpoints
trajectory = {}
for ckpt in ['base', 'sft', 'dpo']:
    trajectory[ckpt] = {}
    for c in concepts:
        pos = torch.randn(20, d_model)
        neg = torch.randn(20, d_model)
        trajectory[ckpt][c] = compute_steering_vector(pos, neg, c)

# Metric 1: Directional stability
stability = directional_stability(trajectory)
print(f'Directional stability: {len(stability)} concepts, matrix shape {stability[concepts[0]].shape}')

# Metric 2: Separability margin
margin = separability_margin(trajectory['sft'][concepts[0]])
print(f'Separability margin (concept_0 @ sft): scalar={margin.scalar_summary:.4f}')

# Metric 3: Concept Gram matrix
gram = concept_gram_matrix(trajectory['sft'])
print(f'Gram matrix shape: {gram.shape}, off-diag mean: {gram[~torch.eye(gram.shape[0], dtype=bool)].mean():.4f}')

# Metric 4: Anisotropy spectrum
spectrum = anisotropy_spectrum(trajectory['sft'])
print(f'Anisotropy spectrum: {spectrum.eigenvalues.shape[0]} eigenvalues, top-5 ratio: {spectrum.explained_variance_ratio[:5]}')
print(f'Top explained variance: {spectrum.explained_variance_ratio[0]:.4f}')
"
}

exp_dry_run() {
    log "Dry run: validate configs"
    uv run python main.py --dry-run
}

# =============================================================================
# CLI
# =============================================================================

usage() {
    cat << 'USAGE'
Usage: ./run_experiments.sh [COMMAND]

Commands:
  all              Run all experiments (tests + weight + activation + concept)
  tests            Run pytest test suite only
  weight           Weight-level effective rank analysis
  activation       Activation-level effective rank analysis
  concept-steering Concept selection + DIM pipeline check
  concept-analysis 4 concept analysis metrics (synthetic demo)
  dry-run          Validate configs without downloading models
  help             Show this help message

Environment:
  RESULTS_DIR      Output directory (default: results/)
  LOG_DIR          Log directory (default: results/logs/)

USAGE
}

main() {
    local cmd="${1:-all}"
    case "$cmd" in
        all)
            log "========== RUNNING ALL EXPERIMENTS =========="
            run_step "tests"             exp_tests
            run_step "dry-run"           exp_dry_run
            run_step "weight-rank"       exp_weight_rank
            run_step "activation-rank"   exp_activation_rank
            run_step "concept-steering"  exp_concept_steering
            run_step "concept-analysis"  exp_concept_analysis
            log "========== ALL EXPERIMENTS COMPLETE =========="
            log "Results: ${RESULTS_DIR}/"
            log "Logs:    ${LOG_DIR}/"
            ;;
        tests)              exp_tests ;;
        weight)             exp_weight_rank ;;
        activation)         exp_activation_rank ;;
        concept-steering)   exp_concept_steering ;;
        concept-analysis)   exp_concept_analysis ;;
        dry-run)            exp_dry_run ;;
        help|--help|-h)     usage ;;
        *)
            err "Unknown command: ${cmd}"
            usage
            exit 1
            ;;
    esac
}

main "$@"
