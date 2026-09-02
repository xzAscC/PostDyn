#!/usr/bin/env python3
"""CLI for validating pinned HumanEval-X canonical solutions.

Assembles official CodeGeeX Python and C++ programs for the first
``--n`` aligned task ids from ``zai-org/humaneval-x`` at the pinned
revision, executes each inside a bubblewrap sandbox, and writes a
machine-readable JSONL report that ``run_concept_dynamics.py`` uses as
its preflight gate for ``python_vs_cpp``.

Usage:
    uv run python experiments/validate_humaneval_x.py [OPTIONS]

Options:
    --n N             Aligned pairs to validate (default: 50)
    --report-path P   Output JSONL report
                      (default: experiments/artifacts/humaneval-x-validation.jsonl)
    --timeout SECS    Per-program subprocess timeout in seconds (default: 10)
    --skip-tool-check Skip the bwrap/g++ presence check (for testing only)
    --help            Show this message and exit

Run this before ``experiments/run_concept_dynamics.sh`` whenever the
``python_vs_cpp`` concept is enabled. See
``docs/humaneval_x_validation.md`` for the full workflow.

This script never modifies the host filesystem outside ``--report-path``
and a per-task scratch directory under the system temp dir.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.contrastive_datasets import HUMANEVAL_X_DATASET, HUMANEVAL_X_REVISION
from src.humaneval_x_validator import (
    BwrapRunner,
    ValidationFailure,
    check_sandbox_tools_available,
    validate_first_n_pairs,
)


DEFAULT_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "artifacts",
    "humaneval-x-validation.jsonl",
)
DEFAULT_N_SAMPLES = 50
DEFAULT_TIMEOUT_SECONDS = 10.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pinned HumanEval-X canonical solutions in a "
            "bubblewrap sandbox and write a JSONL report."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=(
            "Aligned (python, cpp) pairs to validate. Must be positive. "
            "The validator raises if fewer than N shared task ids exist "
            f"upstream. (default: {DEFAULT_N_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Output JSONL report path. Created atomically; only written "
            "when every requested pair passes. "
            f"(default: {DEFAULT_REPORT_PATH})"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Per-program subprocess timeout in seconds. Programs that "
            f"exceed it are recorded as timeouts. (default: {DEFAULT_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--skip-tool-check",
        action="store_true",
        help="Do not verify bwrap/g++ presence before running (testing only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.n <= 0:
        print("ERROR: --n must be positive", file=sys.stderr)
        return 2
    if not (args.timeout > 0) or args.timeout != args.timeout:
        print("ERROR: --timeout must be a finite positive number", file=sys.stderr)
        return 2

    if not args.skip_tool_check:
        try:
            check_sandbox_tools_available()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    print("=" * 60)
    print("HumanEval-X canonical-solution validation")
    print("=" * 60)
    print(f"  Dataset:    {HUMANEVAL_X_DATASET}")
    print(f"  Revision:   {HUMANEVAL_X_REVISION}")
    print(f"  Pairs (N):  {args.n}")
    print(f"  Timeout:    {args.timeout}s per program")
    print(f"  Report:     {args.report_path}")
    print()

    try:
        summary = validate_first_n_pairs(
            n_samples=args.n,
            report_path=Path(args.report_path),
            runner=BwrapRunner(),
            timeout=args.timeout,
            check_tools=False,
        )
    except ValidationFailure as exc:
        print(f"FAIL: task {exc.task_id}", file=sys.stderr)
        print(
            f"  python_outcome={exc.row.python_outcome} "
            f"cpp_outcome={exc.row.cpp_outcome}",
            file=sys.stderr,
        )
        print(
            f"  python_diagnostics: {exc.row.python_diagnostics}",
            file=sys.stderr,
        )
        print(f"  cpp_diagnostics: {exc.row.cpp_diagnostics}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"OK: {summary.n_validated} aligned pairs validated")
    print(f"Report written to: {summary.report_path}")
    print(
        "Run `experiments/run_concept_dynamics.sh` next; this report "
        "is required whenever python_vs_cpp is enabled."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
