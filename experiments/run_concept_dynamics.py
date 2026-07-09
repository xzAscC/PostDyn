#!/usr/bin/env python3
"""CLI entry point for the Concept Dynamics experiment.

Runs DiM concept extraction across Olmo-3-7B post-training variants,
then computes cross-model stability and per-model gram matrices.

Usage:
    uv run python experiments/run_concept_dynamics.py [OPTIONS]

Options:
    --quick          Smoke test: 1 model, 2 concepts, 2 layers, 5 samples
    --models M1,M2   Comma-separated model names (default: all 6)
    --concepts C1,C2 Comma-separated concepts (default: math,code,if,general)
    --layers L1,L2   Comma-separated layer indices (default: 10 uniform)
    --n-samples N    Samples per concept per class (default: 50)
    --output DIR     Output directory (default: results/concept_dynamics)
    --max-seq-len N  Max tokenization length (default: 2048)
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OLMO3_VARIANTS, EXPERIMENT_LAYERS_7B
from src.concept_dynamics import run_full_experiment, select_uniform_layers


DEFAULT_MODELS = [
    "olmo3-think-sft",
    "olmo3-instruct-sft",
    "olmo3-rl-zero-math",
    "olmo3-rl-zero-code",
    "olmo3-rl-zero-if",
    "olmo3-rl-zero-general",
    "olmo3-rl-zero-mix",
]

DEFAULT_CONCEPTS = ["math", "code", "if", "general"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concept Dynamics across Olmo-3-7B post-training variants",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 model, 2 concepts, 2 layers, 5 samples",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated model names (default: {','.join(DEFAULT_MODELS)})",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help=f"Comma-separated concepts (default: {','.join(DEFAULT_CONCEPTS)})",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: 10 uniform from 32)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Samples per concept per class (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/concept_dynamics",
        help="Output directory (default: results/concept_dynamics)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max tokenization length (default: 2048)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        models = ["olmo3-think-sft"]
        concepts = ["math", "code"]
        layers = [3, 16]
        n_samples = 5
        max_seq_len = 512
        print("=" * 60)
        print("QUICK MODE (smoke test)")
        print("=" * 60)
    else:
        models = args.models.split(",") if args.models else DEFAULT_MODELS
        concepts = args.concepts.split(",") if args.concepts else DEFAULT_CONCEPTS
        layers = (
            [int(x) for x in args.layers.split(",")]
            if args.layers
            else EXPERIMENT_LAYERS_7B
        )
        n_samples = args.n_samples
        max_seq_len = args.max_seq_len

    valid_models = [m.strip() for m in models if m.strip() in OLMO3_VARIANTS]
    invalid = [m for m in models if m.strip() not in OLMO3_VARIANTS]
    if invalid:
        print(f"WARNING: Unknown models (skipped): {invalid}")
        print(f"Available: {sorted(OLMO3_VARIANTS.keys())}")

    if not valid_models:
        print("ERROR: No valid models selected")
        sys.exit(1)

    print(f"\nConcept Dynamics Experiment")
    print(f"  Models ({len(valid_models)}):    {valid_models}")
    print(f"  Concepts ({len(concepts)}): {concepts}")
    print(f"  Layers ({len(layers)}):     {layers}")
    print(f"  Samples/concept:  {n_samples}")
    print(f"  Max seq len:      {max_seq_len}")
    print(f"  Output:           {args.output}")
    print()

    results = run_full_experiment(
        model_names=valid_models,
        concepts=concepts,
        layers=layers,
        n_samples=n_samples,
        output_dir=args.output,
        max_seq_len=max_seq_len,
    )

    n_errors = sum(1 for v in results.get("extraction", {}).values() if "error" in v)
    n_ok = len(results.get("models_done", [])) - n_errors
    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {n_ok} OK, {n_errors} errors")
    print(f"Results: {args.output}")
    print(f"{'=' * 60}")

    if n_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
