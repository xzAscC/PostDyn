#!/usr/bin/env python3
"""CLI entry point for the Concept Dynamics experiment.

Runs DiM concept extraction across Olmo-3-7B post-training variants,
then computes cross-model stability and per-model gram matrices.

Usage:
    uv run python experiments/run_concept_dynamics.py [OPTIONS]

Options:
    --quick             Smoke test: 1 model, 2 concepts, 2 layers, 5 samples
    --models M1,M2      Comma-separated model names (default: 6 trajectories)
    --concepts C1,C2    Comma-separated paired concept directions
    --layers L1,L2      Comma-separated layer indices (default: 10 uniform)
    --n-samples N       Samples per concept per class (default: 50)
    --output DIR        Output directory (mode-specific default)
    --max-seq-len N     Max tokenization length (default: 2048)
    --[no-]chat-template  Wrap inputs with tokenizer.apply_chat_template (default: on)
"""

from __future__ import annotations

import argparse
import sys
import os
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OLMO3_VARIANTS, EXPERIMENT_LAYERS_7B
from src.concept_dynamics import run_full_experiment
from src.contrastive_datasets import (
    PAIRED_CONCEPTS,
    _resolve_concept,
    all_concept_keys,
    list_concepts,
)


DEFAULT_MODELS = [
    "olmo3-think-sft",
    "olmo3-rl-zero-math",
    "olmo3-rl-zero-code",
    "olmo3-rl-zero-if",
    "olmo3-rl-zero-general",
    "olmo3-rl-zero-mix",
]

DEFAULT_CONCEPTS = all_concept_keys()

DEFAULT_OUTPUT = "results/concept_dynamics_multi"
DEFAULT_QUICK_OUTPUT = "results/concept_dynamics_multi_quick"

DEFAULT_HUMANEVAL_REPORT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "artifacts",
    "humaneval-x-validation.jsonl",
)
HUMANEVAL_X_CONCEPT_KEY = "code_python_vs_cpp"
HUMANEVAL_X_LEGACY_KEYS = {"code_python_vs_cpp", "python_vs_cpp"}
_HUMANEVAL_X_DATASETS_FALLBACK = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets",
    "humaneval_x.json",
)


def _is_code_concept(concept: str) -> bool:
    return concept in HUMANEVAL_X_LEGACY_KEYS or concept.startswith("code_")


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
            "Comma-separated concepts (default: all "
            f"{len(DEFAULT_CONCEPTS)} canonical keys from "
            "src.contrastive_datasets.all_concept_keys)"
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
    parser.add_argument(
        "--keep-hf-cache",
        action="store_true",
        help="Do not delete Hugging Face cache entries after each model finishes.",
    )
    parser.add_argument(
        "--chat-template",
        dest="chat_template",
        action="store_true",
        default=True,
        help=(
            "Wrap each input with tokenizer.apply_chat_template before the "
            "forward pass (default)."
        ),
    )
    parser.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        help=(
            "Disable chat-template wrapping and feed the raw paired text "
            "directly to the model."
        ),
    )
    return parser.parse_args(argv)


def _preflight_concepts_needing_strict_check(concepts: list[str]) -> list[str]:
    return [c for c in concepts if c in HUMANEVAL_X_LEGACY_KEYS]


def _preflight_concepts_needing_warning(concepts: list[str]) -> list[str]:
    strict = set(_preflight_concepts_needing_strict_check(concepts))
    return [c for c in concepts if _is_code_concept(c) and c not in strict]


def _run_strict_humaneval_preflight(
    report_path: str,
    n_samples: int,
) -> None:
    print(
        f"Preflight: HumanEval-X validation report at "
        f"{report_path} (n_required={n_samples})"
    )
    try:
        run_humaneval_preflight(report_path, n_samples)
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


def _run_relaxed_code_warning(
    concepts: list[str],
    report_path: str,
    datasets_fallback: str,
) -> None:
    if not concepts:
        return
    if os.path.exists(datasets_fallback):
        print(
            f"Preflight (relaxed): {concepts} will load from "
            f"{datasets_fallback}; strict HumanEval-X sandbox preflight "
            f"skipped (multi-language report at {report_path} only covers "
            "python/cpp)."
        )
        return
    print(
        f"WARNING: HumanEval-X preflight report at {report_path} only "
        f"validates python/cpp. Concepts {concepts} are code-related but "
        "no multi-language preflight is available yet. Proceeding without "
        "strict validation; place a snapshot at "
        f"{datasets_fallback} to silence this warning.",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    output_dir = resolve_output_directory(quick=args.quick, output=args.output)

    max_checkpoints_per_model = None
    if args.quick:
        models = ["olmo3-think-sft"]
        concepts = [
            "code_python_vs_cpp",
            "math_cot_vs_direct",
        ]
        layers = [3, 14]
        n_samples = 5
        max_seq_len = 512
        max_checkpoints_per_model = 1
        print("=" * 60)
        print("QUICK MODE (smoke test: 1 model x 1 ckpt x 2 concepts)")
        print("=" * 60)
    else:
        models = _split_csv(args.models) if args.models else list(DEFAULT_MODELS)
        concepts = (
            _split_csv(args.concepts) if args.concepts else list(DEFAULT_CONCEPTS)
        )
        if args.layers:
            try:
                layers = [int(x) for x in _split_csv(args.layers)]
            except ValueError:
                print(
                    "ERROR: --layers must be comma-separated integers",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            layers = list(EXPERIMENT_LAYERS_7B)
        n_samples = args.n_samples
        max_seq_len = args.max_seq_len

    if n_samples <= 0:
        print("ERROR: --n-samples must be positive", file=sys.stderr)
        sys.exit(2)
    resolved_concepts: list[str] = []
    for c in concepts:
        try:
            resolved_concepts.append(_resolve_concept(c))
        except ValueError:
            resolved_concepts.append(c)
    if any(rc == "gender_she_vs_he" for rc in resolved_concepts) and n_samples % 2 != 0:
        print(
            "ERROR: --n-samples must be even when gender_she_vs_he "
            "(a.k.a. female_vs_male_gender) is selected",
            file=sys.stderr,
        )
        sys.exit(2)
    if any(layer < 0 or layer > 31 for layer in layers):
        print("ERROR: --layers must be integers in [0, 31]", file=sys.stderr)
        sys.exit(2)
    unknown_concepts = [c for c in concepts if c not in list_concepts()]
    if unknown_concepts:
        print(
            f"ERROR: Unknown concepts: {unknown_concepts}. "
            f"Supported canonical concepts ({len(PAIRED_CONCEPTS)}): "
            f"{sorted(PAIRED_CONCEPTS)}\n"
            f"  aliases also accepted: see src.contrastive_datasets.list_concepts()",
            file=sys.stderr,
        )
        sys.exit(2)

    valid_models = [m for m in models if m in OLMO3_VARIANTS]
    invalid = [m for m in models if m not in OLMO3_VARIANTS]
    if invalid:
        print(f"WARNING: Unknown models (skipped): {invalid}")
        print(f"Available: {sorted(OLMO3_VARIANTS.keys())}")

    if not valid_models:
        print("ERROR: No valid models selected")
        sys.exit(1)

    strict_code = _preflight_concepts_needing_strict_check(concepts)
    if strict_code:
        _run_strict_humaneval_preflight(args.humaneval_report_path, n_samples)

    relaxed_code = _preflight_concepts_needing_warning(concepts)
    if relaxed_code:
        _run_relaxed_code_warning(
            relaxed_code, args.humaneval_report_path, _HUMANEVAL_X_DATASETS_FALLBACK
        )

    print(f"\nConcept Dynamics Experiment")
    print(f"  Models ({len(valid_models)}):    {valid_models}")
    print(f"  Concepts ({len(concepts)}): {concepts}")
    print(f"  Layers ({len(layers)}):     {layers}")
    print(f"  Samples/concept:  {n_samples}")
    print(f"  Max seq len:      {max_seq_len}")
    print(f"  Chat template:    {'on' if args.chat_template else 'off'}")
    print(f"  Output:           {output_dir}")
    print()

    results = run_full_experiment(
        model_names=valid_models,
        concepts=concepts,
        layers=layers,
        n_samples=n_samples,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        clean_hf_cache=not args.keep_hf_cache,
        use_chat_template=args.chat_template,
        max_checkpoints_per_model=max_checkpoints_per_model,
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
