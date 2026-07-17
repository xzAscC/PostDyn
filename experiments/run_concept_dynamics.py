#!/usr/bin/env python3
"""CLI entry point for the Concept Dynamics experiment.

Runs DiM concept extraction across Olmo-3-7B post-training variants,
then computes cross-model stability and per-model gram matrices.

Usage:
    uv run python experiments/run_concept_dynamics.py [OPTIONS]

Options:
    --quick          Smoke test: 1 model, 2 concepts, 2 layers, 5 samples
    --models M1,M2   Comma-separated model names (default: all 7)
    --concepts C1,C2 Comma-separated paired concept directions
    --layers L1,L2   Comma-separated layer indices (default: 10 uniform)
    --n-samples N    Samples per concept per class (default: 50)
    --output DIR     Output directory (mode-specific default)
    --max-seq-len N  Max tokenization length (default: 2048)
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Callable

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

DEFAULT_CONCEPTS = [
    "python_vs_cpp",
    "concise_math_reasoning_vs_verbose_math_reasoning",
    "french_vs_english_language",
    "female_vs_male_gender",
]

CONCEPT_DIRECTIONS = [
    "Python - C++",
    "Concise - Verbose",
    "French - English",
    "Female - Male",
]

DEFAULT_OUTPUT = "results/concept_dynamics_paired"
DEFAULT_QUICK_OUTPUT = "results/concept_dynamics_paired_quick"

DEFAULT_HUMANEVAL_REPORT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "artifacts",
    "humaneval-x-validation.jsonl",
)
HUMANEVAL_X_CONCEPT_KEY = "python_vs_cpp"


def run_humaneval_preflight(
    report_path: str,
    n_samples: int,
    *,
    preflight_fn: "Callable[[str, int], None] | None" = None,
) -> None:
    """Verify a HumanEval-X validation report covers the requested sample count.

    Imports the validator lazily so ``--help`` stays offline. Raises on the
    first failed check (missing report, wrong revision, hash mismatch,
    duplicate task ids, insufficient successful rows). Re-exports a
    callable seam for tests that want to assert the wiring without
    loading the real validator.
    """
    if preflight_fn is not None:
        preflight_fn(report_path, n_samples)
        return

    from pathlib import Path

    from src.humaneval_x_validator import (
        PreflightOptions,
        load_humaneval_x_raw_pairs,
        preflight_validation,
    )

    current_pairs = load_humaneval_x_raw_pairs(n_samples)
    preflight_validation(
        Path(report_path),
        current_pairs,
        PreflightOptions(n_required=n_samples),
    )


def resolve_output_directory(*, quick: bool, output: str | None) -> str:
    if output is not None:
        return output
    return DEFAULT_QUICK_OUTPUT if quick else DEFAULT_OUTPUT


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Concept Dynamics across Olmo-3-7B post-training variants",
        formatter_class=argparse.RawTextHelpFormatter,
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
        help=(
            "Comma-separated concepts. Defaults and directions:\n"
            + "\n".join(
                f"  {concept}: {direction}"
                for concept, direction in zip(DEFAULT_CONCEPTS, CONCEPT_DIRECTIONS)
            )
        ),
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
        default=None,
        help=(
            "Output directory "
            f"(default: {DEFAULT_OUTPUT}; quick: {DEFAULT_QUICK_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Max tokenization length (default: 2048)",
    )
    parser.add_argument(
        "--humaneval-report-path",
        type=str,
        default=DEFAULT_HUMANEVAL_REPORT,
        help=(
            "HumanEval-X validation report required before extracting the "
            "python_vs_cpp concept. Generate it with "
            "`experiments/validate_humaneval_x.py`. "
            f"(default: {DEFAULT_HUMANEVAL_REPORT})"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    output_dir = resolve_output_directory(quick=args.quick, output=args.output)

    if args.quick:
        models = ["olmo3-think-sft"]
        concepts = DEFAULT_CONCEPTS[:2]
        layers = [3, 16]
        n_samples = 5
        max_seq_len = 512
        print("=" * 60)
        print("QUICK MODE (smoke test)")
        print("=" * 60)
    else:
        models = _split_csv(args.models) if args.models else list(DEFAULT_MODELS)
        concepts = (
            _split_csv(args.concepts) if args.concepts else list(DEFAULT_CONCEPTS)
        )
        if args.layers:
            layers = [int(x) for x in _split_csv(args.layers)]
        else:
            layers = list(EXPERIMENT_LAYERS_7B)
        n_samples = args.n_samples
        max_seq_len = args.max_seq_len

    if n_samples <= 0:
        print("ERROR: --n-samples must be positive", file=sys.stderr)
        sys.exit(2)
    if "female_vs_male_gender" in concepts and n_samples % 2 != 0:
        print(
            "ERROR: --n-samples must be even when female_vs_male_gender is selected",
            file=sys.stderr,
        )
        sys.exit(2)
    if any(layer < 0 for layer in layers):
        print("ERROR: --layers must be non-negative integers", file=sys.stderr)
        sys.exit(2)

    valid_models = [m for m in models if m in OLMO3_VARIANTS]
    invalid = [m for m in models if m not in OLMO3_VARIANTS]
    if invalid:
        print(f"WARNING: Unknown models (skipped): {invalid}")
        print(f"Available: {sorted(OLMO3_VARIANTS.keys())}")

    if not valid_models:
        print("ERROR: No valid models selected")
        sys.exit(1)

    if HUMANEVAL_X_CONCEPT_KEY in concepts:
        print(
            f"Preflight: HumanEval-X validation report at "
            f"{args.humaneval_report_path} (n_required={n_samples})"
        )
        try:
            run_humaneval_preflight(args.humaneval_report_path, n_samples)
        except (ValueError, FileNotFoundError) as exc:
            print(
                f"ERROR: HumanEval-X preflight failed: {exc}\n"
                "Run `uv run python experiments/validate_humaneval_x.py` "
                "first.",
                file=sys.stderr,
            )
            sys.exit(2)
        print("Preflight OK: HumanEval-X report validated.")
        print()

    print(f"\nConcept Dynamics Experiment")
    print(f"  Models ({len(valid_models)}):    {valid_models}")
    print(f"  Concepts ({len(concepts)}): {concepts}")
    print(f"  Layers ({len(layers)}):     {layers}")
    print(f"  Samples/concept:  {n_samples}")
    print(f"  Max seq len:      {max_seq_len}")
    print(f"  Output:           {output_dir}")
    print()

    results = run_full_experiment(
        model_names=valid_models,
        concepts=concepts,
        layers=layers,
        n_samples=n_samples,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
    )

    selected_prefixes = tuple(f"{m}/" for m in valid_models)
    selected_done = [
        key
        for key in results.get("checkpoints_done", [])
        if key.startswith(selected_prefixes)
    ]
    selected_errors = sum(
        1
        for key, value in results.get("extraction", {}).items()
        if key.startswith(selected_prefixes)
        and isinstance(value, dict)
        and "error" in value
    )
    n_ok = len(selected_done)
    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {n_ok} OK, {selected_errors} errors")
    print(f"Results: {output_dir}")
    print(f"{'=' * 60}")

    if selected_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
