#!/usr/bin/env python3
"""Deterministic producer + validator for ``analysis_summary.json``.

Regenerates the canonical RL-Zero-Code syntax analysis summary from the
authoritative raw-protocol ``metrics.json`` (v2 logistic) and the downstream
``aggregate_summary.json`` (11 checkpoints), binds the exact source SHA-256
values and a config-coordinate fingerprint, computes a canonical build
fingerprint, and writes the file atomically (``mkstemp`` + ``fsync`` +
``os.replace``). The summary is then re-validated in place to guarantee the
on-disk bytes match the recorded fingerprints.

The analysis semantics are unchanged from the legacy ad-hoc producer:

* 11 deterministic checkpoint rows, one per ``main`` + ten RL steps;
* each row is an unweighted mean across the ten experiment layers (M2-related
  also averages over the four related code-language concepts; M4 averages over
  ``python_valid`` and ``python_syntax_error``);
* Pearson + Spearman correlations of the six aggregate metrics against Python
  pass@1 and MMLU accuracy across all 11 checkpoints;
* explicit disclosure of the duplicate ``step_100`` / ``step_1000`` weights;
* honest scientific limitations (n=11, duplicate weights, constant-zero C++,
  sparse sensitivity target) preserved verbatim.

No model is loaded, no extraction runs, no source artifact is mutated, and no
wall-clock or RNG non-determinism is introduced. The output is byte-stable on
a fixed SciPy/Python version.

Usage::

    uv run python experiments/build_analysis_summary.py [OPTIONS]

Options:
    --metrics PATH           Override metrics.json path
    --aggregate PATH         Override aggregate_summary.json path
    --output PATH            Override output analysis_summary.json path
    --no-validate            Skip post-write re-validation (NOT RECOMMENDED)
    --check-only             Do not write; only validate the existing summary

Exit codes:
    0  success (summary written or already up to date)
    1  validation or source error (see stderr)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``src`` importable when run directly via ``python experiments/...``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_summary import (  # noqa: E402
    DEFAULT_AGGREGATE_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_SUMMARY_PATH,
    AnalysisSummaryError,
    build_summary,
    load_json,
    validate_summary,
    validate_summary_file,
    write_summary_atomically,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically rebuild and validate the RL-Zero-Code syntax "
            "analysis_summary.json from metrics.json + aggregate_summary.json."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--metrics",
        "-m",
        default=DEFAULT_METRICS_PATH,
        help="Input metrics.json path (default: %(default)s)",
    )
    parser.add_argument(
        "--aggregate",
        "-a",
        default=DEFAULT_AGGREGATE_PATH,
        help="Input aggregate_summary.json path (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_SUMMARY_PATH,
        help="Output analysis_summary.json path (default: %(default)s)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help=(
            "Skip the post-write re-validation step. NOT RECOMMENDED: the "
            "validator is what guarantees the on-disk bytes match the "
            "recorded source/build fingerprints."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Do not write a new summary; validate the existing file at "
            "--output against the live --metrics and --aggregate artifacts."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns 0 on success (summary written and re-validated, or ``--check-only``
    validation passed), 1 on any source/validation error.
    """
    args = parse_args(argv)
    metrics_path = Path(args.metrics)
    aggregate_path = Path(args.aggregate)
    output_path = Path(args.output)

    if args.check_only:
        try:
            validate_summary_file(
                output_path,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )
        except AnalysisSummaryError as e:
            print(f"VALIDATION FAILED: {e}", file=sys.stderr)
            return 1
        print(f"Validated: {output_path}", flush=True)
        return 0

    previous: bytes | None = None
    if output_path.is_file():
        previous = output_path.read_bytes()

    try:
        metrics = load_json(metrics_path)
        aggregate = load_json(aggregate_path)
        if int(metrics.get("version", -1)) != 2:
            raise AnalysisSummaryError(
                f"metrics version must be 2, got {metrics.get('version')!r}"
            )
        summary = build_summary(
            metrics,
            aggregate,
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        # Validate the in-memory summary before replacing any on-disk artifact.
        validate_summary(
            summary,
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        path = write_summary_atomically(output_path, summary)
    except AnalysisSummaryError as e:
        print(f"BUILD FAILED: {e}", file=sys.stderr)
        return 1

    if not args.no_validate:
        try:
            validate_summary_file(
                output_path,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )
        except AnalysisSummaryError as e:
            if previous is not None:
                output_path.write_bytes(previous)
            print(
                f"POST-WRITE VALIDATION FAILED: {e}",
                file=sys.stderr,
            )
            return 1

    build_fp = summary.get("build_fingerprint", "<missing>")
    src_hashes = summary.get("source_hashes", {})
    metrics_sha = (
        src_hashes.get("metrics/metrics.json", "<missing>")
        if isinstance(src_hashes, dict)
        else "<missing>"
    )
    aggregate_sha = (
        src_hashes.get("downstream/aggregate_summary.json", "<missing>")
        if isinstance(src_hashes, dict)
        else "<missing>"
    )
    print(
        f"Wrote {path}\n"
        f"  build_fingerprint={build_fp}\n"
        f"  metrics_sha256={metrics_sha}\n"
        f"  aggregate_sha256={aggregate_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
