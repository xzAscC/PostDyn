#!/usr/bin/env python3
"""CLI entry point for plotting Concept Dynamics results.

Renders Gram and stability heatmaps from a concept-dynamics output
directory produced by ``experiments/run_concept_dynamics.py``.

Modes (selected via flags):

* ``--model`` + ``--checkpoint`` + ``--layer``:
    one Gram heatmap for that triple.
* ``--model`` + ``--concept`` + ``--layer``:
    one checkpoint-stability heatmap for that triple.
* no selector flags (default):
    per-model summary: mid-layer Gram at the last checkpoint plus a
    handful of stability heatmaps across representative concepts.

Examples
--------
Default summary plot::

    uv run python experiments/plot_concept_dynamics.py

Single Gram heatmap::

    uv run python experiments/plot_concept_dynamics.py \\
        --input results/concept_dynamics_multi \\
        --model olmo3-think-sft --checkpoint step_500 --layer 14

Single stability heatmap::

    uv run python experiments/plot_concept_dynamics.py \\
        --input results/concept_dynamics_multi \\
        --model olmo3-think-sft --concept python_vs_cpp --layer 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_INPUT = "results/concept_dynamics_multi"


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path} to exist. Run experiments/run_concept_dynamics.py first."
        )
    with open(path) as f:
        return json.load(f)


def _list_models(gram: dict, stability: dict) -> list[str]:
    seen: dict[str, int] = {}
    for source in (gram, stability):
        for name in source:
            seen[name] = seen.get(name, 0) + 1
    return sorted(seen)


def _list_checkpoints(gram: dict, model: str) -> list[str]:
    block = gram.get(model, {}) if isinstance(gram, dict) else {}
    return sorted(k for k, v in block.items() if isinstance(v, dict) and v)


def _list_concepts(stability: dict, model: str) -> list[str]:
    block = stability.get(model, {}) if isinstance(stability, dict) else {}
    return sorted(k for k, v in block.items() if isinstance(v, dict) and v)


def _list_layers(block: dict) -> list[int]:
    out: list[int] = []
    for key in block:
        if isinstance(key, str) and key.lstrip("-").isdigit():
            out.append(int(key))
    return sorted(out)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Render Gram + stability heatmaps for concept-dynamics results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=(f"Concept-dynamics output directory (default: {DEFAULT_INPUT})."),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Restrict the plot to this model. With --checkpoint/--layer,\n"
            "renders one Gram heatmap; with --concept/--layer, renders one\n"
            "stability heatmap. Without any selector, runs the summary plot\n"
            "for just this model."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint key for a single Gram heatmap (requires --model, --layer).",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default=None,
        help="Concept key for a single stability heatmap (requires --model, --layer).",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index (integer) for a single heatmap.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model list to include in the summary plot.",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Comma-separated concept list to include in the summary plot.",
    )
    return parser.parse_args(argv)


def _plot_single_gram(args, gram_path: str) -> int:
    from src.visualization import plot_gram_heatmap

    if args.layer is None or args.checkpoint is None:
        print(
            "ERROR: --checkpoint and --layer are required for a single Gram heatmap.",
            file=sys.stderr,
        )
        return 2
    try:
        plot_gram_heatmap(
            gram_path,
            model=args.model,
            checkpoint=args.checkpoint,
            layer=args.layer,
        )
    except (KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _plot_single_stability(args, stab_path: str) -> int:
    from src.visualization import plot_stability_heatmap

    if args.layer is None or args.concept is None:
        print(
            "ERROR: --concept and --layer are required for a single stability heatmap.",
            file=sys.stderr,
        )
        return 2
    try:
        plot_stability_heatmap(
            stab_path,
            model=args.model,
            concept=args.concept,
            layer=args.layer,
        )
    except (KeyError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def _plot_summary(args, gram: dict, stability: dict) -> int:
    from src.visualization import plot_concept_dynamics_summary

    models = _split_csv(args.models) if args.models else None
    concepts = _split_csv(args.concepts) if args.concepts else None
    if args.model and models is None:
        models = [args.model]
    if models is None:
        models = _list_models(gram, stability)
    try:
        plot_concept_dynamics_summary(
            args.input,
            models=models,
            concepts=concepts,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    gram_path = os.path.join(args.input, "gram", "gram.json")
    stab_path = os.path.join(args.input, "stability", "stability.json")
    gram = _load_json(gram_path) if os.path.exists(gram_path) else {}
    stability = _load_json(stab_path) if os.path.exists(stab_path) else {}
    if not gram and not stability:
        print(
            f"ERROR: no gram.json/stability.json under {args.input}. "
            "Run experiments/run_concept_dynamics.py first.",
            file=sys.stderr,
        )
        return 1

    if args.model and args.model not in _list_models(gram, stability):
        print(
            f"ERROR: model {args.model!r} not found in gram/stability.",
            file=sys.stderr,
        )
        return 1

    if args.concept:
        return _plot_single_stability(args, stab_path)
    if args.checkpoint:
        return _plot_single_gram(args, gram_path)
    return _plot_summary(args, gram, stability)


if __name__ == "__main__":
    sys.exit(main())
