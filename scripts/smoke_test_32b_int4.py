#!/usr/bin/env python3
"""Small, explicit smoke CLI for Olmo-3 32B Think NF4 loading.

This command does not run unless invoked, and it never silently changes to a
BF16-only load. The checkpoint is downloaded only when ``--load`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


from postdyn.quantized_model_loader import (
    check_quantization_dependencies,
    load_olmo3_32b_think,
    validate_canonical_32b_request,
)
from postdyn.cross_pipeline_integrity import require_canonical_7b
from postdyn.think_sft_differential_experiment import (
    FAMILY_THINK,
    SCALE_32B,
    available_trajectories,
    model_config,
    trajectory_config,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check or load Olmo-3 32B Think with int4 NF4."
    )
    parser.add_argument("--model-id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--revision",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trajectory",
        choices=available_trajectories(FAMILY_THINK, SCALE_32B),
        default="rlvr",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Actually load the checkpoint (may download ~32B weights).",
    )
    args = parser.parse_args(argv)
    try:
        model_id = revision = None
        if args.load:
            config = trajectory_config(FAMILY_THINK, SCALE_32B, args.trajectory)
            if args.checkpoint is None:
                raise ValueError("--checkpoint is required with --load")
            if args.checkpoint not in config.checkpoints:
                raise ValueError(
                    f"unknown {args.trajectory} checkpoint {args.checkpoint!r}"
                )
            model_id = model_config(config.model_key).hf_id
            revision = config.revisions[args.checkpoint]
            if args.model_id is not None and args.model_id != model_id:
                raise ValueError("--model-id does not match the configured trajectory")
            if args.revision is not None and args.revision != revision:
                raise ValueError("--revision does not match the configured checkpoint")
            validate_canonical_32b_request(model_id, revision)
            if args.project_root is None:
                require_canonical_7b()
            else:
                require_canonical_7b(project_root=args.project_root)
        versions = check_quantization_dependencies()
        print(json.dumps({"dependencies": versions}, sort_keys=True))
        if not args.load:
            print(
                "Dependency check passed; use --load to perform the model smoke load."
            )
            return 0
        assert model_id is not None and revision is not None
        loaded = load_olmo3_32b_think(
            model_id,
            revision=revision,
        )
        print(json.dumps(loaded.diagnostics.as_dict(), sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
