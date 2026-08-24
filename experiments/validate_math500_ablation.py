#!/usr/bin/env python3
"""Validate MATH-500 ablation outputs without loading a model."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.math500_eval import DEFAULT_DTYPE, DEFAULT_MAX_NEW_TOKENS, DEFAULT_QUANTIZATION
from src.think_sft_differential_experiment import root_for_trajectory
from src.think_32b_differential_validator import validate_full_canonical_publication

CANONICAL_32B_MAX_NEW_TOKENS = 2048

validate_result_tree = importlib.import_module(
    "src.math500_ablation_validator"
).validate_result_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--scale", choices=("7b", "32b"), default="7b")
    parser.add_argument(
        "--trajectory",
        choices=("sft", "rlvr", "sft_lr_1e-4", "sft_lr_5e-5"),
        default="sft",
    )
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/math500.json"))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.quantization is None:
        args.quantization = "nf4" if args.scale == "32b" else DEFAULT_QUANTIZATION
    if args.scale == "32b" and args.max_new_tokens != CANONICAL_32B_MAX_NEW_TOKENS:
        print("ERROR: 32b requires max_new_tokens=2048")
        return 2
    if args.scale == "32b" and (args.dtype != "bfloat16" or args.quantization != "nf4"):
        print("ERROR: 32b requires dtype=bfloat16 and quantization=nf4")
        return 2
    if args.scale == "32b":
        require = importlib.import_module(
            "src.cross_pipeline_integrity"
        ).require_canonical_7b
        try:
            if args.project_root is None:
                require()
            else:
                require(project_root=args.project_root)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 2
    if args.artifact_root is None:
        args.artifact_root = root_for_trajectory(
            "think", args.scale, args.trajectory, project_root=args.project_root
        )
    if args.project_root is not None and args.dataset == Path("datasets/math500.json"):
        args.dataset = args.project_root / args.dataset
    if args.scale == "32b":
        publication = validate_full_canonical_publication(
            args.artifact_root, args.trajectory
        )
        if not publication.ok:
            for error in publication.errors:
                print(f"ERROR: {error}")
            return 1
    report = validate_result_tree(
        args.root,
        trajectory=args.trajectory,
        dataset_path=args.dataset,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        quantization=args.quantization,
        artifact_root=args.artifact_root,
        scale=args.scale,
        selected_checkpoints=None
        if args.checkpoints is None
        else tuple(args.checkpoints),
        selected_layers=None if args.layers is None else tuple(args.layers),
        project_root=args.project_root,
    )
    if report.ok:
        print(f"VALID: {len(report.conditions)} condition summaries")
        return 0
    for error in report.errors:
        print(f"ERROR: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
