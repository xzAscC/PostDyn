#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.concept_dynamics import load_concept_vectors, load_model_and_tokenizer
from src.config import EXPERIMENT_LAYERS_7B, MODEL_CHECKPOINTS, OLMO3_VARIANTS
from src.gender_surface_analysis import (
    compare_gender_surface_vectors,
    compute_surface_pronoun_vectors,
    save_surface_analysis,
)


GENDER_CONCEPT = "female_vs_male_gender"
DEFAULT_MODEL = "olmo3-rl-zero-math"
DEFAULT_VECTORS_DIR = "results/concept_dynamics_paired/vectors"
DEFAULT_OUTPUT = "results/concept_dynamics_paired/gender_surface_control.json"


def default_checkpoint_for(model: str) -> str:
    checkpoints = MODEL_CHECKPOINTS.get(model) or ["main"]
    return checkpoints[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved Female-Male steering vectors with a weighted "
            "pronoun-only surface-token control."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint/revision (default: last trajectory step for --model)",
    )
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in EXPERIMENT_LAYERS_7B),
        help="Comma-separated layer indices",
    )
    parser.add_argument("--vectors-dir", default=DEFAULT_VECTORS_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-len", type=int, default=32)
    args = parser.parse_args(argv)
    if args.checkpoint is None:
        args.checkpoint = default_checkpoint_for(args.model)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.model not in OLMO3_VARIANTS:
        print(f"ERROR: Unknown model {args.model!r}", file=sys.stderr)
        return 2
    try:
        layers = [int(value) for value in args.layers.split(",") if value.strip()]
    except ValueError:
        print("ERROR: --layers must be comma-separated integers", file=sys.stderr)
        return 2
    if not layers or any(layer < 0 or layer > 31 for layer in layers):
        print("ERROR: --layers must be non-empty integers in [0, 31]", file=sys.stderr)
        return 2
    gender_vectors = {}
    for layer in layers:
        vectors = load_concept_vectors(
            args.vectors_dir, args.model, layer, args.checkpoint
        )
        if GENDER_CONCEPT not in vectors:
            raise ValueError(
                f"Missing {GENDER_CONCEPT!r} at layer {layer} for "
                f"{args.model}/{args.checkpoint}"
            )
        gender_vectors[layer] = vectors[GENDER_CONCEPT]

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            OLMO3_VARIANTS[args.model], args.checkpoint
        )
        surface_vectors = compute_surface_pronoun_vectors(
            model, tokenizer, layers, max_seq_len=args.max_seq_len
        )
        comparisons = compare_gender_surface_vectors(gender_vectors, surface_vectors)
        save_surface_analysis(
            Path(args.output),
            model_name=args.model,
            checkpoint=args.checkpoint,
            comparisons=comparisons,
        )
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved gender surface-control analysis to {args.output}")
    for layer in layers:
        print(f"  layer {layer}: cosine={comparisons[layer]['cosine']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
