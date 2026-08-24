"""Tests for experiments/validate_rl_zero_downstream.py.

Covers the downstream preflight CLI without ever running real benchmark
code: a fake dataset loader feeds canned HumanEval-X rows and a fake
sandbox runner returns canned ``CompletedProcess`` outcomes, so no
bubblewrap / g++ / model is invoked.

Test matrix:
  * ``load_downstream_humaneval_ids`` -- ordered ints, order preservation,
    and rejection of missing / non-list / non-int / duplicate / wrong-count
    manifests.
  * ``report_matches_ids`` -- True for a freshly produced valid report;
    False for missing, stale revision, partial (subset), surplus, wrong
    order, non-pass outcome, and hash-mismatch reports.
  * ``run_downstream_preflight`` -- idempotent skip when the report is
    valid (no runner calls); regeneration when missing, stale, partial,
    wrong-order; and forced regeneration even when valid.
  * Real-manifest guard -- when ``downstream.json`` is materialized, the
    loader returns exactly 50 ids in the pinned order (skipped otherwise).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest

import experiments.validate_rl_zero_downstream as cli
from experiments.validate_rl_zero_downstream import (
    DEFAULT_REPORT_PATH,
    DEFAULT_TIMEOUT_SECONDS,
    PreflightOutcome,
    load_downstream_humaneval_ids,
    main,
    parse_args,
    report_matches_ids,
    run_downstream_preflight,
)
from src.contrastive_datasets import HUMANEVAL_X_REVISION
from src.humaneval_x_validator import (
    OUTCOME_FAIL,
    OUTCOME_PASS,
    PYTHON_PATH,
    ValidationFailure,
    ValidationRow,
    read_validation_report,
    validate_pairs_by_ids,
)


# =============================================================================
# Fakes & helpers
# =============================================================================


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _raw_row(language: str, task_id: int) -> dict[str, Any]:
    prefix = "Python" if language == "python" else "CPP"
    return {
        "task_id": f"{prefix}/{task_id}",
        "prompt": f"{language} prompt {task_id}\n",
        "canonical_solution": f"{language} solution {task_id}\n",
        "test": f"def test_{task_id}(): assert True\n",
    }


def _raw_rows(language: str, ids: Sequence[int]) -> list[dict[str, Any]]:
    return [_raw_row(language, i) for i in ids]


class _FakeDatasetLoader:
    """Returns canned rows for python/cpp keyed by the requested ids."""

    def __init__(self, ids: Sequence[int]):
        self.ids = list(ids)

    def __call__(self, language: str):
        return iter(_raw_rows(language, self.ids))


def _passing_runner() -> MagicMock:
    runner = MagicMock()
    runner.run_in_sandbox.return_value = _completed(0)
    return runner


def _downstream_manifest(ids: Sequence[Any]) -> dict[str, Any]:
    # Accepts ``Any`` (not just int) so tests can feed malformed manifests
    # (strings, bools) to exercise the loader's rejection paths.
    return {
        "humaneval_x": {
            "task_ids": list(ids),
            "n_items": len(ids),
        },
        "mmlu": {"n_questions": 0, "items": []},
    }


def _write_valid_report(
    path: Path, ids: Sequence[int], loader: _FakeDatasetLoader
) -> list[ValidationRow]:
    """Produce a report whose SHA round-trips through ``loader``."""
    validate_pairs_by_ids(
        list(ids),
        path,
        runner=_passing_runner(),
        dataset_loader=loader,
        check_tools=False,
    )
    return read_validation_report(path)


def _rewrite_report(path: Path, rows: Sequence[ValidationRow]) -> None:
    """Overwrite the report with the given rows (non-atomic, test only)."""
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True))
            handle.write("\n")


# =============================================================================
# load_downstream_humaneval_ids
# =============================================================================


class TestLoadDownstreamHumanevalIds:
    def test_returns_ordered_ints_from_manifest(self, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = lambda: _downstream_manifest(ids)  # noqa: E731
        assert load_downstream_humaneval_ids(downstream_loader=loader) == [1, 5, 6]

    def test_preserves_unsorted_manifest_order(self, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [6, 1, 5]
        loader = lambda: _downstream_manifest(ids)  # noqa: E731
        # Order is preserved verbatim -- no internal sorting.
        assert load_downstream_humaneval_ids(downstream_loader=loader) == [6, 1, 5]

    def test_real_manifest_yields_fifty_pinned_ids_when_present(self):
        # Guards the actual downstream contract end-to-end. Skipped when
        # the builder artifact has not been materialized in this checkout.
        try:
            ids = load_downstream_humaneval_ids()
        except FileNotFoundError:
            pytest.skip("downstream.json not built in this checkout")
        assert len(ids) == 50
        assert len(set(ids)) == 50
        # The pinned first three and last two anchor the manifest order.
        assert ids[:3] == [1, 5, 6]
        assert ids[-2:] == [161, 163]

    def test_raises_when_humaneval_x_block_missing(self):
        loader = lambda: {"humaneval_x": None}  # noqa: E731
        with pytest.raises(ValueError, match="missing 'humaneval_x'"):
            load_downstream_humaneval_ids(downstream_loader=loader)

    def test_raises_when_task_ids_not_a_list(self):
        loader = lambda: {"humaneval_x": {"task_ids": "1,5,6"}}  # noqa: E731
        with pytest.raises(ValueError, match="is not a list"):
            load_downstream_humaneval_ids(downstream_loader=loader)

    def test_raises_on_non_int_id(self):
        loader = lambda: _downstream_manifest([1, "5", 6])  # noqa: E731
        with pytest.raises(ValueError, match="must be ints"):
            load_downstream_humaneval_ids(downstream_loader=loader)

    def test_raises_on_boolean_id(self):
        # bool is an int subclass; it must be refused loudly.
        loader = lambda: _downstream_manifest([1, 5, True])  # noqa: E731
        with pytest.raises(ValueError, match="must be ints"):
            load_downstream_humaneval_ids(downstream_loader=loader)

    def test_raises_on_duplicate_ids(self):
        loader = lambda: _downstream_manifest([1, 5, 1])  # noqa: E731
        with pytest.raises(ValueError, match="duplicates"):
            load_downstream_humaneval_ids(downstream_loader=loader)

    def test_raises_on_wrong_count(self, monkeypatch):
        # N_SAMPLES is 50 by default; a 3-id manifest must be rejected
        # unless the test shrinks the expected count.
        loader = lambda: _downstream_manifest([1, 5, 6])  # noqa: E731
        with pytest.raises(ValueError, match="expected 50"):
            load_downstream_humaneval_ids(downstream_loader=loader)

    def test_wrong_count_passes_when_n_samples_is_shrunk(self, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        loader = lambda: _downstream_manifest([1, 5, 6])  # noqa: E731
        assert load_downstream_humaneval_ids(downstream_loader=loader) == [1, 5, 6]


# =============================================================================
# report_matches_ids
# =============================================================================


class TestReportMatchesIds:
    def test_true_for_freshly_produced_valid_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        _write_valid_report(path, ids, loader)
        assert report_matches_ids(path, ids, dataset_loader=loader) is True

    def test_false_when_report_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        assert (
            report_matches_ids(tmp_path / "absent.jsonl", ids, dataset_loader=loader)
            is False
        )

    def test_false_when_report_has_wrong_revision(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        # Corrupt the revision so the report is stale relative to the pin.
        stale = [replace(row, revision="deadbeef") for row in rows]
        _rewrite_report(path, stale)
        assert report_matches_ids(path, ids, dataset_loader=loader) is False

    def test_false_when_report_has_non_pass_outcome(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        bad = [replace(row, python_outcome=OUTCOME_FAIL) for row in rows]
        _rewrite_report(path, bad)
        assert report_matches_ids(path, ids, dataset_loader=loader) is False

    def test_false_when_report_is_partial_subset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        # Drop the last row -> only ids [1, 5] remain.
        _rewrite_report(path, rows[:2])
        assert report_matches_ids(path, ids, dataset_loader=loader) is False

    def test_false_when_report_has_surplus_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        # Append a duplicate of an existing id -> wrong shape.
        _rewrite_report(path, list(rows) + [rows[0]])
        assert report_matches_ids(path, ids, dataset_loader=loader) is False

    def test_false_when_report_has_right_set_but_wrong_order(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        # Same set, swapped order -- downstream pairs rows line-for-line,
        # so order mismatch must be treated as invalid.
        swapped = [rows[1], rows[0], rows[2]]
        _rewrite_report(path, swapped)
        assert report_matches_ids(path, ids, dataset_loader=loader) is False

    def test_false_when_report_hashes_drift(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        drifted = [replace(row, python_code_sha256="0" * 64) for row in rows]
        _rewrite_report(path, drifted)
        assert report_matches_ids(path, ids, dataset_loader=loader) is False


# =============================================================================
# run_downstream_preflight: idempotent skip vs regenerate
# =============================================================================


class TestRunDownstreamPreflight:
    def test_skips_without_runner_calls_when_report_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        _write_valid_report(path, ids, loader)

        runner = _passing_runner()
        outcome = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert outcome.skipped is True
        assert outcome.n_validated == 3
        assert outcome.report_path == path
        # Idempotency: no sandbox work ran.
        runner.run_in_sandbox.assert_not_called()

    def test_regenerates_when_report_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"

        runner = _passing_runner()
        outcome = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert outcome.skipped is False
        assert outcome.n_validated == 3
        assert path.exists()
        rows = read_validation_report(path)
        assert [r.task_id for r in rows] == [1, 5, 6]
        for row in rows:
            assert row.python_outcome == OUTCOME_PASS
            assert row.cpp_outcome == OUTCOME_PASS
        # 3 pairs * (1 python + 2 cpp calls) == 9 sandbox invocations.
        assert runner.run_in_sandbox.call_count == 9

    def test_regenerates_when_report_is_stale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        stale = [replace(row, revision="deadbeef") for row in rows]
        _rewrite_report(path, stale)

        runner = _passing_runner()
        outcome = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert outcome.skipped is False
        # The stale revision must have been overwritten with the pinned one.
        rows_after = read_validation_report(path)
        assert rows_after[0].revision == HUMANEVAL_X_REVISION

    def test_regenerates_when_report_is_partial(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        _rewrite_report(path, rows[:2])  # drop id 6

        runner = _passing_runner()
        outcome = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert outcome.skipped is False
        assert outcome.n_validated == 3
        rows_after = read_validation_report(path)
        assert [r.task_id for r in rows_after] == [1, 5, 6]

    def test_regenerates_when_report_has_wrong_order(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        rows = _write_valid_report(path, ids, loader)
        _rewrite_report(path, [rows[1], rows[0], rows[2]])  # swap 1<->5

        runner = _passing_runner()
        outcome = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert outcome.skipped is False
        rows_after = read_validation_report(path)
        # Order is restored to the manifest order.
        assert [r.task_id for r in rows_after] == [1, 5, 6]

    def test_force_regenerates_even_when_valid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"
        _write_valid_report(path, ids, loader)
        original_mtime = path.stat().st_mtime_ns

        runner = _passing_runner()
        outcome = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
            force=True,
        )
        assert outcome.skipped is False
        assert runner.run_in_sandbox.call_count == 9
        # The report was rewritten (atomic replace -> new inode/mtime).
        assert path.stat().st_mtime_ns >= original_mtime

    def test_failure_propagates_and_leaves_report_untouched(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"

        def fail_python(command, scratch_dir, timeout, **_kwargs):
            if PYTHON_PATH in command or any(
                str(part).endswith(".py") for part in command
            ):
                return _completed(1, stderr="boom")
            return _completed(0)

        runner = MagicMock()
        runner.run_in_sandbox.side_effect = fail_python

        with pytest.raises(ValidationFailure):
            run_downstream_preflight(
                ids,
                path,
                runner=runner,
                dataset_loader=loader,
                check_tools=False,
            )
        # Atomicity: no partial report lands on disk on failure.
        assert not path.exists()

    def test_second_call_skips_after_first_regenerates(self, tmp_path, monkeypatch):
        # The real "rerun without work when the report is valid" contract.
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        ids = [1, 5, 6]
        loader = _FakeDatasetLoader(ids)
        path = tmp_path / "report.jsonl"

        runner = _passing_runner()
        first = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert first.skipped is False
        calls_after_first = runner.run_in_sandbox.call_count

        second = run_downstream_preflight(
            ids,
            path,
            runner=runner,
            dataset_loader=loader,
            check_tools=False,
        )
        assert second.skipped is True
        # No additional sandbox work on the second pass.
        assert runner.run_in_sandbox.call_count == calls_after_first


# =============================================================================
# parse_args + main CLI wiring (no sandbox, no model)
# =============================================================================


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.report_path == DEFAULT_REPORT_PATH
        assert args.timeout == DEFAULT_TIMEOUT_SECONDS
        assert args.skip_tool_check is False
        assert args.force is False

    def test_explicit_report_and_timeout(self):
        args = parse_args(["--report-path", "/tmp/x.jsonl", "--timeout", "7.5"])
        assert args.report_path == "/tmp/x.jsonl"
        assert args.timeout == 7.5

    def test_force_and_skip_tool_check_flags(self):
        args = parse_args(["--force", "--skip-tool-check"])
        assert args.force is True
        assert args.skip_tool_check is True


class TestMainWiring:
    def test_rejects_non_positive_timeout(self, capsys):
        # argparse coerces "0" to 0.0; main() must refuse it.
        rc = main(["--timeout", "0", "--skip-tool-check"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "timeout" in err.lower()

    def test_skip_tool_check_avoids_runtime_error(self, tmp_path, monkeypatch, capsys):
        # With --skip-tool-check, missing bwrap/g++ must not abort main.
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        path = tmp_path / "report.jsonl"

        def fake_ids():
            return [1, 5, 6]

        monkeypatch.setattr(cli, "load_downstream_humaneval_ids", fake_ids)

        # Force validate_pairs_by_ids to succeed without any real sandbox.
        monkeypatch.setattr(
            cli,
            "run_downstream_preflight",
            lambda *a, **k: PreflightOutcome(
                skipped=False, n_validated=3, report_path=path
            ),
        )
        rc = main(["--report-path", str(path), "--skip-tool-check", "--timeout", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "OK" in out
        assert "validated 3" in out

    def test_skip_path_when_report_already_valid(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "N_SAMPLES", 3)
        path = tmp_path / "report.jsonl"

        monkeypatch.setattr(cli, "load_downstream_humaneval_ids", lambda: [1, 5, 6])
        monkeypatch.setattr(
            cli,
            "report_matches_ids",
            lambda report_path, ids, dataset_loader=None: True,
        )

        # If the skip probe fires, run_downstream_preflight must NOT run.
        def fail_if_called(*a, **k):
            pytest.fail("run_downstream_preflight must not run on the skip path")

        monkeypatch.setattr(cli, "run_downstream_preflight", fail_if_called)

        rc = main(["--report-path", str(path), "--skip-tool-check"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "SKIP" in out
