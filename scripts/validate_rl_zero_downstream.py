#!/usr/bin/env python3
"""Preflight CLI for the 50 downstream HumanEval-X task ids.

This is a hard gate that runs *before* any model-side pass@1 work on the
RL-Zero-Code syntax experiment. It validates the exact 50 downstream
HumanEval-X ids pinned in ``data/allenai/Dolci-RL-Zero-Code-7B/downstream.json``
with the existing bubblewrap canonical validator
(:func:`postdyn.humaneval_x_validator.validate_pairs_by_ids`) and writes an
atomic JSONL report under the isolated experiment results root.

Why a separate CLI?
    The legacy ``scripts/validate_humaneval_x.py`` validates the first
    ``N`` aligned task ids (sorted ``0..49``). The downstream set is an
    explicit, disjoint selection (``[1, 5, 6, ..., 161, 163]``) that the
    first-N report does not cover. Downstream pass@1 must only run against
    these exact ids, so a dedicated preflight report is required.

Idempotency.
    On every invocation the CLI first checks whether an existing report at
    ``--report-path`` already covers the exact ordered downstream id set
    with the pinned revision and all-pass canonical outcomes (delegated to
    :func:`postdyn.humaneval_x_validator.preflight_validation`, which also
    re-derives the SHA-256 of every assembled program from the current
    dataset rows). When that check passes, the CLI skips all sandbox work
    and exits 0. Use ``--force`` to ignore a valid report and regenerate.

Usage:
    uv run python scripts/validate_rl_zero_downstream.py [OPTIONS]

Options:
    --report-path P    Output JSONL report
                       (default: logs/rl_zero_code_syntax/preflight/
                        humaneval-x-downstream.jsonl)
    --timeout SECS     Per-program subprocess timeout in seconds (default: 10)
    --skip-tool-check  Skip the bwrap/g++ presence check (testing only)
    --force            Regenerate the report even if it is already valid
    --help             Show this message and exit

This script never modifies existing validation reports, datasets, or
config; it only writes ``--report-path`` (atomically) and a per-task
scratch directory under the system temp dir.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# Make ``src`` importable when run directly via ``python scripts/...``.

from postdyn.contrastive_datasets import HUMANEVAL_X_DATASET, HUMANEVAL_X_REVISION
from postdyn.humaneval_x_validator import (
    BwrapRunner,
    PreflightOptions,
    SandboxRunner,
    ValidationFailure,
    check_sandbox_tools_available,
    load_humaneval_x_pairs_by_ids,
    preflight_validation,
    read_validation_report,
    validate_pairs_by_ids,
)
from postdyn.rl_zero_experiment import (
    N_SAMPLES,
    RL_ZERO_CODE_RESULTS_ROOT,
    load_downstream,
)


#: Default report location, isolated under this experiment's results root.
DEFAULT_REPORT_PATH: str = os.path.join(
    RL_ZERO_CODE_RESULTS_ROOT, "preflight", "humaneval-x-downstream.jsonl"
)
DEFAULT_TIMEOUT_SECONDS: float = 10.0


# =============================================================================
# Downstream id loading
# =============================================================================


def load_downstream_humaneval_ids(
    *,
    downstream_loader: Callable[[], dict[str, Any]] = load_downstream,
) -> list[int]:
    """Return the pinned 50 downstream HumanEval-X numeric task ids, in order.

    Reads ``humaneval_x.task_ids`` straight from ``downstream.json`` (via
    :func:`postdyn.rl_zero_experiment.load_downstream`) so the **caller's
    manifest order** is preserved end-to-end -- ``validate_pairs_by_ids``
    writes rows in exactly this order, and the idempotency check compares
    against it line-for-line.

    A ``downstream_loader`` seam lets unit tests inject a fake manifest
    without touching disk.

    Raises:
        ValueError: the id list is missing, not a list, contains
            non-integers, has duplicates, or does not carry exactly
            ``N_SAMPLES`` (50) ids.
    """
    downstream = downstream_loader()
    block = downstream.get("humaneval_x")
    if not isinstance(block, dict):
        raise ValueError("downstream.json missing 'humaneval_x' block")
    raw_ids = block.get("task_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("downstream 'humaneval_x.task_ids' is not a list")
    ids: list[int] = []
    for value in raw_ids:
        # bool is a subclass of int; refuse it so a corrupted manifest is loud.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"downstream humaneval_x task_ids must be ints; got {value!r}"
            )
        ids.append(int(value))
    if len(ids) != len(set(ids)):
        raise ValueError(f"downstream humaneval_x task_ids has duplicates: {ids}")
    if len(ids) != N_SAMPLES:
        raise ValueError(
            f"downstream humaneval_x task_ids has {len(ids)} ids, expected {N_SAMPLES}"
        )
    return ids


# =============================================================================
# Idempotency check
# =============================================================================


def report_matches_ids(
    report_path: Path,
    ids: Sequence[int],
    *,
    dataset_loader: Callable[[str], Any] | None = None,
    n_required: int | None = None,
) -> bool:
    """Return ``True`` iff ``report_path`` is a valid gate for ``ids``.

    Verified conditions (all must hold):

    1. The report parses (delegates to :func:`read_validation_report`).
    2. The report rows' task ids equal ``ids`` **in the exact same order**.
       A report that covers the right set in the wrong order, or with
       surplus/missing rows, is rejected -- downstream consumers pair
       rows back to the manifest line-for-line.
    3. :func:`preflight_validation` succeeds: every row carries the pinned
       revision and dataset, both python/cpp outcomes are ``pass``, task
       ids are unique, and the SHA-256 of every freshly assembled program
       matches the stored hash.

    Any :class:`ValueError` (missing file, parse error, mismatch, stale
    revision, non-pass outcome, hash drift) collapses to ``False`` so the
    caller can decide to regenerate without inspecting exception types.
    """
    expected = list(ids)
    if n_required is None:
        n_required = len(expected)

    try:
        rows = read_validation_report(report_path)
    except ValueError:
        return False

    # Exact ordered coverage: the report must be the manifest, line-for-line.
    if [row.task_id for row in rows] != expected:
        return False

    pairs = load_humaneval_x_pairs_by_ids(expected, dataset_loader=dataset_loader)
    try:
        preflight_validation(
            report_path,
            pairs,
            PreflightOptions(n_required=n_required),
        )
    except ValueError:
        return False
    return True


# =============================================================================
# Preflight runner
# =============================================================================


@dataclass(frozen=True)
class PreflightOutcome:
    """Outcome of a downstream preflight invocation.

    ``skipped`` is ``True`` when an already-valid report was reused and no
    sandbox work ran. ``n_validated`` is the number of rows covered by the
    report (0 when skipped without a report, otherwise the id count).
    """

    skipped: bool
    n_validated: int
    report_path: Path


def run_downstream_preflight(
    ids: Sequence[int],
    report_path: Path,
    *,
    runner: SandboxRunner | None = None,
    dataset_loader: Callable[[str], Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    check_tools: bool = True,
    force: bool = False,
) -> PreflightOutcome:
    """Validate the downstream ids, reusing a valid report when possible.

    Idempotent contract:

    * If ``force`` is ``False`` and ``report_path`` already satisfies
      :func:`report_matches_ids` for ``ids``, no sandbox work runs and the
      function returns with ``skipped=True``.
    * Otherwise the report is (re)generated via
      :func:`validate_pairs_by_ids`, which writes atomically -- a failure
      midway through leaves any pre-existing report untouched.

    Raises:
        ValidationFailure: a canonical program failed (propagated from
            :func:`validate_pairs_by_ids`).
        ValueError: alignment could not deliver the requested ids.
    """
    expected = list(ids)

    if not force and report_matches_ids(
        report_path, expected, dataset_loader=dataset_loader
    ):
        return PreflightOutcome(
            skipped=True,
            n_validated=len(expected),
            report_path=report_path,
        )

    summary = validate_pairs_by_ids(
        expected,
        report_path,
        runner=runner,
        timeout=timeout,
        dataset_loader=dataset_loader,
        check_tools=check_tools,
    )
    return PreflightOutcome(
        skipped=False,
        n_validated=summary.n_validated,
        report_path=report_path,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 50 downstream HumanEval-X canonical solutions "
            "in a bubblewrap sandbox and write an atomic JSONL preflight "
            "report. Skips sandbox work when an existing report already "
            "covers the exact ordered downstream id set."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Output JSONL report path. Written atomically; only replaced "
            "when every requested pair passes. Reused as-is when it "
            f"already validates the downstream ids. (default: {DEFAULT_REPORT_PATH})"
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the report even if it already validates the downstream ids.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if (
        not math.isfinite(args.timeout)
        or args.timeout != args.timeout
        or args.timeout <= 0
    ):
        print("ERROR: --timeout must be a finite positive number", file=sys.stderr)
        return 2

    if not args.skip_tool_check:
        try:
            check_sandbox_tools_available()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    report_path = Path(args.report_path)

    print("=" * 60)
    print("RL-Zero-Code downstream HumanEval-X preflight")
    print("=" * 60)
    print(f"  Dataset:       {HUMANEVAL_X_DATASET}")
    print(f"  Revision:      {HUMANEVAL_X_REVISION}")
    print(f"  Downstream ids:{N_SAMPLES} (from downstream.json)")
    print(f"  Timeout:       {args.timeout}s per program")
    print(f"  Report:        {report_path}")
    print(f"  Force rerun:   {bool(args.force)}")
    print()

    try:
        ids = load_downstream_humaneval_ids()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Idempotent skip probe (no sandbox work). Tool check already passed.
    try:
        if not args.force and report_matches_ids(report_path, ids):
            print(
                f"SKIP: report already valid for {len(ids)} downstream ids; "
                f"no sandbox work needed."
            )
            print(f"Report: {report_path}")
            return 0
    except ValueError as exc:
        # A stale or corrupt report is not fatal here -- we regenerate.
        print(f"NOTE: existing report rejected ({exc}); regenerating.")

    try:
        outcome = run_downstream_preflight(
            ids,
            report_path,
            runner=BwrapRunner(),
            timeout=args.timeout,
            check_tools=False,
            force=args.force,
        )
    except ValidationFailure as exc:
        print(f"FAIL: task {exc.task_id}", file=sys.stderr)
        print(
            f"  python_outcome={exc.row.python_outcome} "
            f"cpp_outcome={exc.row.cpp_outcome}",
            file=sys.stderr,
        )
        print(f"  python_diagnostics: {exc.row.python_diagnostics}", file=sys.stderr)
        print(f"  cpp_diagnostics: {exc.row.cpp_diagnostics}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if outcome.skipped:
        print(f"SKIP: report already valid for {outcome.n_validated} downstream ids.")
    else:
        print(
            f"OK: validated {outcome.n_validated} downstream ids; "
            f"report written atomically."
        )
    print(f"Report: {outcome.report_path}")
    print("This report is the hard gate before any downstream model pass@1 run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
