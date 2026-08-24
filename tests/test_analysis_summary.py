"""Tests for ``src/analysis_summary.py`` and its CLI
``experiments/build_analysis_summary.py``.

The tests are model-free and split into three layers:

1. **Pure-math unit tests** -- hand-built metrics/aggregate fixtures assert
   that the unweighted-mean aggregations and the Pearson/Spearman correlations
   match known values exactly (no SciPy beyond what the producer itself uses).
2. **Determinism + integrity tests** -- the producer is a pure function of
   its sources, the build fingerprint is stable, and the atomic write leaves
   no ``.tmp`` residue.
3. **Strict validator tests** -- tampering with any field, swapping a source
   artifact for one with a different SHA-256, removing a source, dropping a
   limitation, breaking the aggregation, or forging the build fingerprint each
   surface a typed exception.
4. **Real-artifact regression test** (skipped if the on-disk artifacts are
   absent) -- the deterministic producer reproduces the canonical 11-row
   checkpoint table, the canonical correlation block, and the canonical
   limitations/integrity text from the actual
   ``results/rl_zero_code_syntax`` artifacts.

Only ``numpy`` / ``scipy`` (already required) and ``pytest`` are used.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experiments.build_analysis_summary as cli
from src.analysis_summary import (
    AGGREGATE_METRIC_KEYS,
    AGGREGATION_DESCRIPTION,
    BASE_CHECKPOINT,
    CONTROL_CONCEPT,
    DEFAULT_AGGREGATE_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_SUMMARY_PATH,
    DOWNSTREAM_TARGET_KEYS,
    DUPLICATE_CHECKPOINTS,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_LAYERS,
    INTEGRITY_NOTES,
    LIMITATIONS,
    METRICS_RELPATH,
    AGGREGATE_RELPATH,
    MissingSource,
    PROBE_CLASSES,
    RELATED_CONCEPTS,
    ROW_ONLY_FIELDS,
    SCHEMA,
    SYNTAX_LABELS,
    TARGET_CONCEPT,
    VERSION,
    AnalysisSummaryError,
    SourceHashMismatch,
    SummaryTamper,
    aggregate_m1_cos_vs_main,
    aggregate_m2_control_delta_cos,
    aggregate_m2_related_delta_cos,
    aggregate_m3_norm_delta,
    aggregate_m4_syntax_metric,
    build_summary,
    canonical_json_bytes,
    compute_build_fingerprint,
    config_fingerprint,
    correlate,
    downstream_value,
    extract_config_coordinate,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_summary,
    validate_summary_file,
    write_summary_atomically,
)

# =============================================================================
# Constants
# =============================================================================

#: Two checkpoints / two layers / 8 probe classes -- small grid for the math
#: unit tests that exercise aggregation in isolation (the build helper still
#: needs the canonical 11-checkpoint schedule, see TEST_BUILD_CHECKPOINTS).
TEST_CHECKPOINTS: list[str] = [BASE_CHECKPOINT, "step_100"]
TEST_LAYERS: list[int] = [3, 6]
TEST_RELATED: tuple[str, ...] = RELATED_CONCEPTS

#: Canonical 11-checkpoint schedule for tests that drive ``build_summary`` /
#: ``validate_summary``. These functions intentionally enforce the canonical
#: experiment grid (see :data:`src.rl_zero_experiment.EXPERIMENT_CHECKPOINTS`).
TEST_BUILD_CHECKPOINTS: list[str] = list(EXPERIMENT_CHECKPOINTS)
#: Canonical 10-layer schedule (same reasoning as TEST_BUILD_CHECKPOINTS).
TEST_BUILD_LAYERS: list[int] = list(EXPERIMENT_LAYERS)

#: Pinned source SHA-256 values of the canonical on-disk artifacts (recorded
#: from a known-good producer run). The real-artifact regression test asserts
#: the producer reproduces these exactly; if the artifacts legitimately change
#: (re-extraction, re-scoring), this constant must be updated alongside the
#: regeneration.
REAL_METRICS_SHA256: str = (
    "9aaa26100081d3cd5592af4e8554d20adfc18ece351bbc42828e0f392d834bda"
)
REAL_AGGREGATE_SHA256: str = (
    "66df2db7854a7af238000fa9c6ff410950cdf063850910424a7028efc8e7288c"
)


# =============================================================================
# Synthetic metrics + aggregate fixture builders
# =============================================================================


def _metric_block_m1(
    checkpoints: list[str],
    layers: list[int],
    cos_value: float,
) -> dict[str, Any]:
    """Build a minimal M1 block where every cos_vs_main equals ``cos_value``.

    ``main`` vs ``main`` is always exactly 1.0 (cosine of itself); other
    checkpoints carry the supplied ``cos_value`` so the unit test can compute
    the expected unweighted mean by hand.
    """
    by_layer: dict[str, Any] = {}
    for layer in layers:
        by_ckpt: dict[str, dict[str, float]] = {}
        for ckpt in checkpoints:
            cos_main = 1.0 if ckpt == BASE_CHECKPOINT else cos_value
            by_ckpt[ckpt] = {"cos_vs_main": cos_main, "cos_vs_step_100": cos_value}
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}
    return {
        "description": "test m1",
        "target_concept": TARGET_CONCEPT,
        "references": [BASE_CHECKPOINT, "step_100"],
        "by_layer": by_layer,
    }


def _metric_block_m2(
    checkpoints: list[str],
    layers: list[int],
    related_delta: float,
    control_delta: float,
) -> dict[str, Any]:
    """Build an M2 block with per-checkpoint delta variation (non-constant)."""
    by_layer: dict[str, Any] = {}
    for layer in layers:
        by_ckpt: dict[str, Any] = {}
        for i, ckpt in enumerate(checkpoints):
            # Base checkpoint keeps the zero-delta identity; RL steps fan out.
            rd = 0.0 if ckpt == BASE_CHECKPOINT else related_delta + 0.001 * i
            cd = 0.0 if ckpt == BASE_CHECKPOINT else control_delta + 0.0005 * i
            related = {
                concept: {
                    "cos_current": -0.1,
                    "cos_base": -0.1 + rd,
                    "delta_cos": rd,
                }
                for concept in TEST_RELATED
            }
            by_ckpt[ckpt] = {
                "related": related,
                "control_diagnostic": {
                    "cos_current": 0.0,
                    "cos_base": -cd,
                    "delta_cos": cd,
                },
            }
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}
    return {
        "description": "test m2",
        "target_concept": TARGET_CONCEPT,
        "related_concepts": list(TEST_RELATED),
        "control_concept_diagnostic": CONTROL_CONCEPT,
        "base_checkpoint": BASE_CHECKPOINT,
        "by_layer": by_layer,
    }


def _metric_block_m3(
    checkpoints: list[str],
    layers: list[int],
    delta: float,
) -> dict[str, Any]:
    """Build an M3 block with per-checkpoint magnitude deltas."""
    by_layer: dict[str, Any] = {}
    for layer in layers:
        by_ckpt: dict[str, dict[str, float]] = {}
        for i, ckpt in enumerate(checkpoints):
            d = 0.0 if ckpt == BASE_CHECKPOINT else delta + 0.01 * i
            by_ckpt[ckpt] = {
                "norm_current": 1.0 + d,
                "norm_base": 1.0,
                "delta": d,
            }
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}
    return {
        "description": "test m3",
        "target_concept": TARGET_CONCEPT,
        "base_checkpoint": BASE_CHECKPOINT,
        "by_layer": by_layer,
    }


def _metric_block_m4(
    checkpoints: list[str],
    layers: list[int],
    balanced_accuracy: float,
    auroc: float,
) -> dict[str, Any]:
    """Build an M4 block with per-checkpoint probe scores (non-constant)."""
    by_layer: dict[str, Any] = {}
    for layer in layers:
        by_ckpt: dict[str, Any] = {}
        for i, ckpt in enumerate(checkpoints):
            ba = balanced_accuracy + 0.001 * i
            ar = auroc + 0.0005 * i
            by_label = {
                label: {
                    "balanced_accuracy": ba,
                    "auroc": ar,
                    "n_positive": 50,
                    "n_negative": 350,
                    "n_folds_used": 5,
                    "n_folds_skipped": 0,
                }
                for label in PROBE_CLASSES
            }
            by_ckpt[ckpt] = {"by_label": by_label, "n_samples": 400, "n_groups": 50}
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}
    return {
        "description": "test m4",
        "method": "logistic",
        "label_names": list(PROBE_CLASSES),
        "linear_probe_params": {
            "n_folds": 5,
            "alpha": 1.0,
            "seed": 42,
            "standardize": True,
        },
        "by_layer": by_layer,
    }


def _synthetic_metrics(
    checkpoints: list[str] = TEST_BUILD_CHECKPOINTS,
    layers: list[int] = TEST_BUILD_LAYERS,
    *,
    cos_value: float = 0.95,
    related_delta: float = -0.01,
    control_delta: float = 0.005,
    m3_delta: float = 0.1,
    m4_balanced: float = 0.85,
    m4_auroc: float = 0.92,
    n_samples: int = 400,
) -> dict[str, Any]:
    """Build a minimal but schema-valid metrics.json dict for unit tests.

    Defaults are picked so that, across the 11 canonical checkpoints, every
    aggregate correlation is well-defined (no constant series) -- ``main``
    carries the identity M1 cosine and the base reference deltas, while the
    ten RL steps carry the supplied non-trivial values.
    """
    return {
        "schema": "rl_zero_code_syntax_metrics",
        "version": 2,
        "metadata": {
            "checkpoints": list(checkpoints),
            "layers": list(layers),
            "concepts": {
                "target": TARGET_CONCEPT,
                "related": list(TEST_RELATED),
                "control": CONTROL_CONCEPT,
            },
            "probe_classes": list(PROBE_CLASSES),
            "protocol": "raw",
            "n_samples": n_samples,
            "base_checkpoint": BASE_CHECKPOINT,
            "optional_references": ["step_100"],
            "source_models": {
                "olmo3-base": "allenai/Olmo-3-1025-7B",
                "olmo3-rl-zero-code": "allenai/Olmo-3-7B-RL-Zero-Code",
            },
            "humaneval_x_revision": "62c78627f3072a1454fa0cb0184737cafe5e4198",
            "linear_probe": {
                "method": "logistic",
                "n_folds": 5,
                "alpha": 1.0,
                "seed": 42,
                "standardize": True,
            },
        },
        "m1_checkpoint_cosine": _metric_block_m1(checkpoints, layers, cos_value),
        "m2_target_related_delta_cos": _metric_block_m2(
            checkpoints, layers, related_delta, control_delta
        ),
        "m3_target_raw_direction_magnitude_delta": _metric_block_m3(
            checkpoints, layers, m3_delta
        ),
        "m4_eight_class_grouped_logistic_separability": _metric_block_m4(
            checkpoints, layers, m4_balanced, m4_auroc
        ),
    }


def _synthetic_aggregate(
    checkpoints: list[str] = TEST_BUILD_CHECKPOINTS,
    *,
    python_p1: tuple[float, ...] | None = None,
    cpp_p1: float = 0.0,
    mmlu: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Build a minimal aggregate_summary.json dict for unit tests.

    When ``python_p1`` / ``mmlu`` are omitted, varied default values are
    spread across the 11 canonical checkpoints so the build-step correlation
    math is well-defined.
    """
    if python_p1 is None:
        python_p1 = tuple(round(0.20 + 0.02 * i, 2) for i in range(len(checkpoints)))
    if mmlu is None:
        mmlu = tuple(round(0.55 + 0.01 * (i % 5), 2) for i in range(len(checkpoints)))
    assert len(python_p1) == len(checkpoints)
    assert len(mmlu) == len(checkpoints)
    cells: dict[str, Any] = {}
    for ckpt, py, mm in zip(checkpoints, python_p1, mmlu, strict=True):
        cells[ckpt] = {
            "python_pass_at_1": py,
            "cpp_pass_at_1": cpp_p1,
            "mmlu_accuracy": mm,
            "model": "allenai/Olmo-3-7B-RL-Zero-Code",
            "revision": ckpt,
        }
    return {
        "expected_checkpoints": list(checkpoints),
        "n_expected": len(checkpoints),
        "n_present": len(checkpoints),
        "selected_checkpoints": list(checkpoints),
        "checkpoints": cells,
    }


def _write_metrics_and_aggregate(
    tmp_path: Path,
    metrics: dict[str, Any] | None = None,
    aggregate: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write the synthetic metrics + aggregate to ``tmp_path`` and return paths."""
    metrics = metrics if metrics is not None else _synthetic_metrics()
    aggregate = aggregate if aggregate is not None else _synthetic_aggregate()
    metrics_path = tmp_path / "metrics.json"
    aggregate_path = tmp_path / "aggregate_summary.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    return metrics_path, aggregate_path


def _build(
    metrics: dict[str, Any],
    aggregate: dict[str, Any],
    *,
    metrics_path: str | Path | None = None,
    aggregate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Wrap build_summary to return dict[str, Any] for ergonomic test access."""
    return build_summary(
        metrics,
        aggregate,
        metrics_path=metrics_path,
        aggregate_path=aggregate_path,
    )


# =============================================================================
# 1. Pure-math aggregation unit tests
# =============================================================================


class TestAggregationMath:
    """The six aggregation helpers compute the expected unweighted means."""

    def test_m1_mean_cos_vs_main(self) -> None:
        metrics = _synthetic_metrics(cos_value=0.9)
        # ``main`` cos_vs_main is 1.0 across both layers; mean = 1.0.
        assert aggregate_m1_cos_vs_main(metrics, BASE_CHECKPOINT, TEST_LAYERS) == 1.0
        # ``step_100`` cos_vs_main is 0.9 across both layers; mean = 0.9.
        assert aggregate_m1_cos_vs_main(metrics, "step_100", TEST_LAYERS) == 0.9

    def test_m2_related_delta_is_mean_over_layers_and_concepts(self) -> None:
        metrics = _synthetic_metrics(related_delta=-0.0123)
        result = aggregate_m2_related_delta_cos(
            metrics, "step_100", TEST_LAYERS, TEST_RELATED
        )
        # step_100 is checkpoint index 1 → delta = -0.0123 + 0.001*1
        assert result == pytest.approx(-0.0113, abs=1e-15)

    def test_m2_control_delta_is_mean_over_layers(self) -> None:
        metrics = _synthetic_metrics(control_delta=0.007)
        assert aggregate_m2_control_delta_cos(
            metrics, "step_100", TEST_LAYERS
        ) == pytest.approx(0.0075, abs=1e-15)

    def test_m3_norm_delta_is_mean_over_layers(self) -> None:
        metrics = _synthetic_metrics(m3_delta=0.42)
        assert aggregate_m3_norm_delta(
            metrics, "step_100", TEST_LAYERS
        ) == pytest.approx(0.43, abs=1e-15)

    def test_m4_balanced_accuracy_averages_two_syntax_labels(self) -> None:
        metrics = _synthetic_metrics(m4_balanced=0.79)
        assert aggregate_m4_syntax_metric(
            metrics, "step_100", TEST_LAYERS, field="balanced_accuracy"
        ) == pytest.approx(0.791, abs=1e-15)

    def test_m4_auroc_averages_two_syntax_labels(self) -> None:
        metrics = _synthetic_metrics(m4_auroc=0.93)
        # main is index 0 → auroc = 0.93 + 0.0005*0
        assert aggregate_m4_syntax_metric(
            metrics, BASE_CHECKPOINT, TEST_LAYERS, field="auroc"
        ) == pytest.approx(0.93, abs=1e-15)

    def test_m4_invalid_field_raises(self) -> None:
        metrics = _synthetic_metrics()
        with pytest.raises(ValueError, match="M4 aggregate field"):
            aggregate_m4_syntax_metric(
                metrics, BASE_CHECKPOINT, TEST_LAYERS, field="n_positive"
            )


class TestDownstreamValue:
    def test_reads_float(self) -> None:
        agg = _synthetic_aggregate()
        # Default spread: python_p1 = 0.20 + 0.02*i, mmlu = 0.55 + 0.01*(i%5).
        assert downstream_value(agg, BASE_CHECKPOINT, "python_pass_at_1") == 0.20
        assert downstream_value(agg, "step_100", "mmlu_accuracy") == 0.56

    def test_missing_checkpoint_raises(self) -> None:
        agg = _synthetic_aggregate()
        with pytest.raises(AnalysisSummaryError, match="missing checkpoint"):
            downstream_value(agg, "step_9999", "python_pass_at_1")

    def test_non_number_raises(self) -> None:
        agg = _synthetic_aggregate()
        agg["checkpoints"][BASE_CHECKPOINT]["python_pass_at_1"] = "0.2"
        with pytest.raises(AnalysisSummaryError, match="must be a number"):
            downstream_value(agg, BASE_CHECKPOINT, "python_pass_at_1")


# =============================================================================
# 2. Correlation unit tests
# =============================================================================


class TestCorrelate:
    def test_known_pearson_perfect_positive(self) -> None:
        result = correlate([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-12)
        assert result["spearman_rho"] == pytest.approx(1.0, abs=1e-12)

    def test_known_pearson_perfect_negative(self) -> None:
        result = correlate([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        assert result["pearson_r"] == pytest.approx(-1.0, abs=1e-12)
        assert result["spearman_rho"] == pytest.approx(-1.0, abs=1e-12)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            correlate([1.0, 2.0], [1.0])

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 2 points"):
            correlate([1.0], [2.0])

    def test_constant_metric_raises(self) -> None:
        with pytest.raises(AnalysisSummaryError, match="metric series is constant"):
            correlate([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_constant_target_raises(self) -> None:
        with pytest.raises(AnalysisSummaryError, match="target series is constant"):
            correlate([1.0, 2.0, 3.0], [0.5, 0.5, 0.5])

    def test_non_finite_raises(self) -> None:
        with pytest.raises(AnalysisSummaryError, match="non-finite"):
            correlate([1.0, math.inf, 3.0], [1.0, 2.0, 3.0])

    def test_spearman_handles_ties(self) -> None:
        # Tied metric values; SciPy's average-rank Spearman is well-defined.
        result = correlate([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
        assert -1.0 <= result["spearman_rho"] <= 1.0
        assert 0.0 <= result["spearman_p"] <= 1.0


# =============================================================================
# 3. Hashing + fingerprint determinism
# =============================================================================


class TestHashing:
    def test_sha256_bytes_is_hex_64(self) -> None:
        h = sha256_bytes(b"hello")
        assert len(h) == 64
        int(h, 16)  # parses as hex.

    def test_sha256_file_streams_large_file(self, tmp_path: Path) -> None:
        path = tmp_path / "big.bin"
        # 3 MiB so the streaming loop runs more than once.
        data = b"abcdef" * (1024 * 1024 // 2)
        path.write_bytes(data)
        assert sha256_file(path) == sha256_bytes(data)

    def test_canonical_json_is_sorted_compact_with_newline(self) -> None:
        payload = {"b": 1, "a": 2, "c": [3, 2, 1]}
        raw = canonical_json_bytes(payload)
        assert raw.endswith(b"\n")
        # sort_keys=True -> 'a' before 'b' before 'c'.
        assert raw.index(b'"a"') < raw.index(b'"b"')
        # separators -> no spaces after separators.
        assert b": " not in raw
        assert b", " not in raw

    def test_canonical_json_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="Out of range"):
            canonical_json_bytes({"x": math.nan})

    def test_compute_build_fingerprint_is_stable(self) -> None:
        summary_a = {
            "schema": SCHEMA,
            "version": VERSION,
            "checkpoints": [{"checkpoint": "main"}],
            "limitations": list(LIMITATIONS),
        }
        summary_b = {
            "limitations": list(LIMITATIONS),
            "checkpoints": [{"checkpoint": "main"}],
            "version": VERSION,
            "schema": SCHEMA,
        }
        # Key order does not affect the canonical fingerprint.
        assert compute_build_fingerprint(summary_a) == compute_build_fingerprint(
            summary_b
        )

    def test_compute_build_fingerprint_excludes_self(self) -> None:
        base = {"schema": SCHEMA, "version": VERSION, "checkpoints": []}
        without_fp = dict(base)
        with_different_fp = dict(base)
        with_different_fp["build_fingerprint"] = "deadbeef"
        assert compute_build_fingerprint(without_fp) == compute_build_fingerprint(
            with_different_fp
        )

    def test_config_fingerprint_binds_coordinate(self) -> None:
        concepts = {
            "target": TARGET_CONCEPT,
            "related": list(TEST_RELATED),
            "control": CONTROL_CONCEPT,
        }
        kwargs: dict[str, Any] = dict(
            checkpoints=TEST_CHECKPOINTS,
            layers=TEST_LAYERS,
            concepts=concepts,
            probe_classes=list(PROBE_CLASSES),
            base_checkpoint=BASE_CHECKPOINT,
            optional_references=["step_100"],
            n_samples=400,
            protocol="raw",
            humaneval_x_revision="rev123",
            source_models={"a": "b"},
            linear_probe={"method": "logistic"},
        )
        fp1 = config_fingerprint(**kwargs)
        # Same inputs -> same fingerprint.
        assert config_fingerprint(**kwargs) == fp1
        # Any coordinate change -> different fingerprint.
        kwargs_diff = copy.deepcopy(kwargs)
        kwargs_diff["n_samples"] = 401
        assert config_fingerprint(**kwargs_diff) != fp1
        kwargs_diff = copy.deepcopy(kwargs)
        kwargs_diff["layers"] = [3, 6, 9]
        assert config_fingerprint(**kwargs_diff) != fp1
        kwargs_diff = copy.deepcopy(kwargs)
        kwargs_diff["protocol"] = "chat"
        assert config_fingerprint(**kwargs_diff) != fp1


# =============================================================================
# 4. build_summary: structure + determinism
# =============================================================================


class TestBuildSummary:
    def test_build_returns_schema_and_version(self, tmp_path: Path) -> None:
        metrics = _synthetic_metrics()
        aggregate = _synthetic_aggregate()
        summary = _build(metrics, aggregate)
        assert summary["schema"] == SCHEMA
        assert summary["version"] == VERSION
        assert summary["source_metrics_version"] == 2

    def test_build_records_source_hashes_when_paths_given(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        metrics = load_json(metrics_path)
        aggregate = load_json(aggregate_path)
        summary = _build(
            metrics,
            aggregate,
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        assert summary["source_hashes"][METRICS_RELPATH] == sha256_file(metrics_path)
        assert summary["source_hashes"][AGGREGATE_RELPATH] == sha256_file(
            aggregate_path
        )

    def test_build_omits_source_hash_when_path_is_none(self) -> None:
        metrics = _synthetic_metrics()
        aggregate = _synthetic_aggregate()
        summary = _build(metrics, aggregate)
        assert summary["source_hashes"][METRICS_RELPATH] == ""
        assert summary["source_hashes"][AGGREGATE_RELPATH] == ""

    def test_build_includes_aggregation_description(self) -> None:
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        assert summary["aggregation"] == AGGREGATION_DESCRIPTION

    def test_build_includes_known_duplicate(self) -> None:
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        assert tuple(summary["known_duplicate"]["checkpoints"]) == DUPLICATE_CHECKPOINTS

    def test_build_includes_limitations_verbatim(self) -> None:
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        assert tuple(summary["limitations"]) == LIMITATIONS

    def test_build_includes_integrity_notes(self) -> None:
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        assert summary["integrity"] == INTEGRITY_NOTES

    def test_build_row_count_matches_checkpoint_count(self) -> None:
        metrics = _synthetic_metrics()
        summary = _build(metrics, _synthetic_aggregate())
        rows = summary["checkpoints"]
        assert isinstance(rows, list) and len(rows) == len(TEST_BUILD_CHECKPOINTS)
        ckpt_names = [r["checkpoint"] for r in rows]
        assert ckpt_names == TEST_BUILD_CHECKPOINTS

    def test_build_correlations_cover_all_metrics_and_targets(self) -> None:
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        correlations = summary["correlations_all_11"]
        assert isinstance(correlations, dict)
        for metric_key in AGGREGATE_METRIC_KEYS:
            block = correlations.get(metric_key, {})
            assert isinstance(block, dict)
            for tgt in DOWNSTREAM_TARGET_KEYS:
                entry = block.get(tgt, {})
                assert isinstance(entry, dict)
                for stat in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p"):
                    v = entry.get(stat)
                    assert isinstance(v, float)
                    assert math.isfinite(v)

    def test_build_build_fingerprint_matches_recompute(self) -> None:
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        recomputed = compute_build_fingerprint(summary)
        assert summary["build_fingerprint"] == recomputed

    def test_build_is_pure_function_of_inputs(self, tmp_path: Path) -> None:
        metrics = _synthetic_metrics()
        aggregate = _synthetic_aggregate()
        s1 = _build(metrics, aggregate)
        s2 = _build(metrics, aggregate)
        assert s1 == s2
        # Different downstream values -> different summary (and different fp).
        py = tuple(
            round(0.30 + 0.02 * i, 2) for i in range(len(TEST_BUILD_CHECKPOINTS))
        )
        mm = tuple(
            round(0.50 + 0.01 * i, 2) for i in range(len(TEST_BUILD_CHECKPOINTS))
        )
        s3 = _build(metrics, _synthetic_aggregate(python_p1=py, mmlu=mm))
        assert s3["build_fingerprint"] != s1["build_fingerprint"]

    def test_build_rejects_wrong_metrics_schema(self) -> None:
        metrics = _synthetic_metrics()
        metrics["schema"] = "something_else"
        with pytest.raises(AnalysisSummaryError, match="metrics schema"):
            _build(metrics, _synthetic_aggregate())

    def test_build_rejects_wrong_checkpoint_count(self) -> None:
        # The producer is hard-coded to the 11-checkpoint canonical schedule.
        ckpts = [BASE_CHECKPOINT] + [f"step_{i}" for i in range(100, 1000, 100)]
        metrics = _synthetic_metrics(checkpoints=ckpts, layers=TEST_LAYERS)
        with pytest.raises(AnalysisSummaryError, match="expected exactly 11"):
            _build(metrics, _synthetic_aggregate(checkpoints=ckpts))


# =============================================================================
# 5. Atomic write
# =============================================================================


class TestAtomicWrite:
    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "analysis_summary.json"
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        written = write_summary_atomically(path, summary)
        assert Path(written).exists()
        assert path.exists()

    def test_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        write_summary_atomically(path, summary)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        # All fields are preserved.
        assert loaded["schema"] == SCHEMA
        assert loaded["version"] == VERSION
        assert len(loaded["checkpoints"]) == len(TEST_BUILD_CHECKPOINTS)

    def test_no_tmp_residue(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.json"
        summary = _build(_synthetic_metrics(), _synthetic_aggregate())
        write_summary_atomically(path, summary)
        # No leftover temp files in the destination directory.
        leftovers = [p for p in path.parent.iterdir() if p.name.startswith(".")]
        leftovers += [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []

    def test_overwrite_preserves_old_on_partial_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "summary.json"
        original = _build(_synthetic_metrics(), _synthetic_aggregate())
        write_summary_atomically(path, original)
        original_bytes = path.read_bytes()

        # Force the second write's fsync to fail mid-stream; the original
        # bytes must survive untouched.
        new_summary = _build(
            _synthetic_metrics(m4_auroc=0.99),
            _synthetic_aggregate(),
        )

        def boom(_fd: int) -> None:
            raise OSError("simulated fsync failure")

        monkeypatch.setattr("os.fsync", boom)
        with pytest.raises(OSError, match="simulated fsync"):
            write_summary_atomically(path, new_summary)

        # The destination is unchanged (atomic replace was never reached).
        assert path.read_bytes() == original_bytes
        # No temp residue.
        leftovers = [p for p in path.parent.iterdir() if p.name != path.name]
        assert leftovers == []


# =============================================================================
# 6. Strict validator: positive + tamper + missing-source + drift
# =============================================================================


class TestValidateSummaryPositive:
    def test_synthetic_summary_validates(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        validate_summary(
            summary,
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )

    def test_validate_summary_file_round_trip(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        out = tmp_path / "analysis_summary.json"
        write_summary_atomically(out, summary)
        validate_summary_file(
            out, metrics_path=metrics_path, aggregate_path=aggregate_path
        )


class TestValidateSummaryTamper:
    """Each tampering vector raises a typed AnalysisSummaryError."""

    def _valid(self, tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        return metrics_path, aggregate_path, summary

    def test_tampered_schema_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["schema"] = "tampered"
        with pytest.raises(AnalysisSummaryError, match="schema"):
            validate_summary(summary)

    def test_tampered_version_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["version"] = 999
        with pytest.raises(AnalysisSummaryError, match="version"):
            validate_summary(summary)

    def test_tampered_source_metrics_version_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["source_metrics_version"] = 1
        with pytest.raises(AnalysisSummaryError, match="source_metrics_version"):
            validate_summary(summary)

    def test_tampered_aggregation_text_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["aggregation"]["layers"] = "weighted mean"
        with pytest.raises(AnalysisSummaryError, match="aggregation.layers"):
            validate_summary(summary)

    def test_tampered_row_value_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["checkpoints"][0]["m1_mean_cos_vs_main"] = 0.123
        with pytest.raises(SummaryTamper, match="build fingerprint"):
            validate_summary(summary)

    def test_tampered_row_value_nan_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["checkpoints"][0]["m4_syntax_auroc"] = math.nan
        with pytest.raises((SummaryTamper, AnalysisSummaryError)):
            # Either the validator catches the NaN early (non-finite) or the
            # fingerprint recompute catches it. Both raise an AnalysisSummaryError.
            validate_summary(summary)

    def test_tampered_correlation_value_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["correlations_all_11"]["m1_mean_cos_vs_main"]["python_pass_at_1"][
            "pearson_r"
        ] = 0.999
        with pytest.raises(SummaryTamper, match="build fingerprint"):
            validate_summary(summary)

    def test_tampered_limitation_text_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["limitations"][0] = "everything is causal and great"
        with pytest.raises(AnalysisSummaryError, match=r"limitations\[0\]"):
            validate_summary(summary)

    def test_dropped_limitation_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["limitations"] = summary["limitations"][:2]
        with pytest.raises(AnalysisSummaryError, match="limitations"):
            validate_summary(summary)

    def test_tampered_integrity_note_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["integrity"]["downstream"] = "no rescoring"
        with pytest.raises(AnalysisSummaryError, match="integrity.downstream"):
            validate_summary(summary)

    def test_tampered_known_duplicate_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["known_duplicate"]["checkpoints"] = ["step_100", "step_700"]
        with pytest.raises(AnalysisSummaryError, match="known_duplicate"):
            validate_summary(summary)

    def test_tampered_build_fingerprint_raises(self, tmp_path: Path) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["build_fingerprint"] = "0" * 64
        with pytest.raises(SummaryTamper, match="build fingerprint"):
            validate_summary(summary)

    def test_tampered_config_coordinate_fingerprint_raises(
        self, tmp_path: Path
    ) -> None:
        _, _, summary = self._valid(tmp_path)
        summary["config_coordinate"]["fingerprint_sha256"] = "a" * 64
        # Build fingerprint will also be off, but the coordinate check fires
        # before the build fingerprint check (so the error message points at
        # the real culprit).
        with pytest.raises(AnalysisSummaryError, match="config_coordinate"):
            validate_summary(summary)

    def test_tampered_config_coordinate_checkpoint_drift_raises(
        self, tmp_path: Path
    ) -> None:
        # Mutate checkpoints *and* recompute the coordinate fingerprint *and*
        # the build fingerprint, so only the live-experiment drift check
        # catches the discrepancy.
        _, _, summary = self._valid(tmp_path)
        summary["config_coordinate"]["checkpoints"] = [
            "main",
            "step_100",
            "step_200",
            "step_300",
            "step_400",
            "step_500",
            "step_600",
            "step_700",
            "step_800",
            "step_900",
            "step_1000",
        ]
        # Recompute the coordinate fingerprint to match the tampered list.
        cc = summary["config_coordinate"]
        cc["fingerprint_sha256"] = config_fingerprint(
            cc["checkpoints"],
            cc["layers"],
            cc["concepts"],
            cc["probe_classes"],
            cc["base_checkpoint"],
            cc["optional_references"],
            n_samples=cc["n_samples"],
            protocol=cc["protocol"],
            humaneval_x_revision=cc["humaneval_x_revision"],
            source_models=cc["source_models"],
            linear_probe=cc["linear_probe"],
        )
        summary["build_fingerprint"] = compute_build_fingerprint(summary)
        with pytest.raises(AnalysisSummaryError, match="config_coordinate.checkpoints"):
            validate_summary(summary)


class TestValidateSummarySourceHash:
    """Validator catches a changed source artifact."""

    def test_metrics_hash_mismatch_raises(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        # Rewrite the metrics file with a slightly different value.
        metrics = load_json(metrics_path)
        metrics["m1_checkpoint_cosine"]["by_layer"]["3"]["by_checkpoint"]["step_100"][
            "cos_vs_main"
        ] = 0.91
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        with pytest.raises(SourceHashMismatch, match="metrics.json hash changed"):
            validate_summary(
                summary,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )

    def test_aggregate_hash_mismatch_raises(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        # Rewrite the aggregate file with a different downstream value.
        agg = load_json(aggregate_path)
        agg["checkpoints"][BASE_CHECKPOINT]["mmlu_accuracy"] = 0.99
        aggregate_path.write_text(json.dumps(agg), encoding="utf-8")
        with pytest.raises(SourceHashMismatch, match="aggregate_summary.json hash"):
            validate_summary(
                summary,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )

    def test_metrics_file_missing_raises(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        metrics_path.unlink()
        with pytest.raises(MissingSource, match="metrics source missing"):
            validate_summary(
                summary,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )

    def test_aggregate_file_missing_raises(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        aggregate_path.unlink()
        with pytest.raises(MissingSource, match="aggregate source missing"):
            validate_summary(
                summary,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )

    def test_no_paths_skips_live_hash_check(self, tmp_path: Path) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        # Passing no paths: validation succeeds even after both files are
        # mutated; the on-disk re-hash is skipped by design.
        metrics = load_json(metrics_path)
        metrics["m1_checkpoint_cosine"]["by_layer"]["3"]["by_checkpoint"]["step_100"][
            "cos_vs_main"
        ] = 0.55
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        validate_summary(summary)  # no paths -> no live hash check.


class TestValidateSummaryStructure:
    """Validator catches structural problems unrelated to tampering."""

    def _valid(self, tmp_path: Path) -> dict[str, Any]:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        return build_summary(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )

    def test_missing_correlation_metric_raises(self, tmp_path: Path) -> None:
        summary = self._valid(tmp_path)
        del summary["correlations_all_11"]["m4_syntax_auroc"]
        with pytest.raises(AnalysisSummaryError, match="correlations missing metric"):
            validate_summary(summary)

    def test_missing_correlation_target_raises(self, tmp_path: Path) -> None:
        summary = self._valid(tmp_path)
        del summary["correlations_all_11"]["m1_mean_cos_vs_main"]["mmlu_accuracy"]
        with pytest.raises(AnalysisSummaryError, match="missing target"):
            validate_summary(summary)

    def test_correlation_pearson_r_out_of_range_raises(self, tmp_path: Path) -> None:
        summary = self._valid(tmp_path)
        entry = summary["correlations_all_11"]["m1_mean_cos_vs_main"][
            "python_pass_at_1"
        ]
        entry["pearson_r"] = 1.5
        entry["pearson_p"] = -0.1
        # Re-stamp build fingerprint so the structural check (not the tamper
        # check) surfaces.
        summary["build_fingerprint"] = compute_build_fingerprint(summary)
        with pytest.raises(AnalysisSummaryError, match="pearson_r out of"):
            validate_summary(summary)

    def test_row_missing_aggregate_key_raises(self, tmp_path: Path) -> None:
        summary = self._valid(tmp_path)
        del summary["checkpoints"][0]["m3_mean_norm_delta"]
        summary["build_fingerprint"] = compute_build_fingerprint(summary)
        with pytest.raises(AnalysisSummaryError, match="missing keys"):
            validate_summary(summary)

    def test_invalid_source_hash_hex_raises(self, tmp_path: Path) -> None:
        summary = self._valid(tmp_path)
        summary["source_hashes"][METRICS_RELPATH] = "z" * 64
        summary["build_fingerprint"] = compute_build_fingerprint(summary)
        with pytest.raises(AnalysisSummaryError, match="not a valid hex"):
            validate_summary(summary)

    def test_wrong_length_source_hash_raises(self, tmp_path: Path) -> None:
        summary = self._valid(tmp_path)
        summary["source_hashes"][METRICS_RELPATH] = "abc"
        summary["build_fingerprint"] = compute_build_fingerprint(summary)
        with pytest.raises(AnalysisSummaryError, match="64-char hex"):
            validate_summary(summary)


# =============================================================================
# 7. CLI smoke tests
# =============================================================================


class TestCLI:
    def test_help_exits_cleanly(self) -> None:
        with pytest.raises(SystemExit) as e:
            cli.parse_args(["--help"])
        assert e.value.code == 0

    def test_default_paths(self) -> None:
        args = cli.parse_args([])
        assert args.metrics == DEFAULT_METRICS_PATH
        assert args.aggregate == DEFAULT_AGGREGATE_PATH
        assert args.output == DEFAULT_SUMMARY_PATH
        assert args.no_validate is False
        assert args.check_only is False

    def test_build_and_check_only_round_trip(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        out = tmp_path / "analysis_summary.json"
        # Build.
        rc = cli.main(
            [
                "--metrics",
                str(metrics_path),
                "--aggregate",
                str(aggregate_path),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        build_stdout = capsys.readouterr().out
        assert "build_fingerprint=" in build_stdout
        # Check-only against the same artifacts.
        rc = cli.main(
            [
                "--metrics",
                str(metrics_path),
                "--aggregate",
                str(aggregate_path),
                "--output",
                str(out),
                "--check-only",
            ]
        )
        assert rc == 0
        check_stdout = capsys.readouterr().out
        assert "Validated" in check_stdout

    def test_check_only_returns_1_on_drift(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        metrics_path, aggregate_path = _write_metrics_and_aggregate(tmp_path)
        out = tmp_path / "analysis_summary.json"
        assert (
            cli.main(
                [
                    "--metrics",
                    str(metrics_path),
                    "--aggregate",
                    str(aggregate_path),
                    "--output",
                    str(out),
                ]
            )
            == 0
        )
        # Mutate the metrics file.
        m = load_json(metrics_path)
        m["m1_checkpoint_cosine"]["by_layer"]["3"]["by_checkpoint"]["step_100"][
            "cos_vs_main"
        ] = 0.42
        metrics_path.write_text(json.dumps(m), encoding="utf-8")
        # check-only now must fail with rc=1.
        rc = cli.main(
            [
                "--metrics",
                str(metrics_path),
                "--aggregate",
                str(aggregate_path),
                "--output",
                str(out),
                "--check-only",
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "VALIDATION FAILED" in err

    def test_build_fails_on_missing_source(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = cli.main(
            [
                "--metrics",
                str(tmp_path / "absent_metrics.json"),
                "--aggregate",
                str(tmp_path / "absent_agg.json"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "BUILD FAILED" in err


# =============================================================================
# 8. Real-artifact regression test (skipped if artifacts absent)
# =============================================================================


@pytest.fixture(scope="module")
def real_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the canonical on-disk metrics + aggregate (skip if absent)."""
    if not Path(DEFAULT_METRICS_PATH).exists():
        pytest.skip(f"real metrics not present at {DEFAULT_METRICS_PATH}")
    if not Path(DEFAULT_AGGREGATE_PATH).exists():
        pytest.skip(f"real aggregate not present at {DEFAULT_AGGREGATE_PATH}")
    return load_json(DEFAULT_METRICS_PATH), load_json(DEFAULT_AGGREGATE_PATH)


class TestRealArtifactRegression:
    """The deterministic producer reproduces the canonical summary."""

    def test_reproduces_pinned_source_hashes(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        metrics, aggregate = real_artifacts
        summary = _build(
            metrics,
            aggregate,
            metrics_path=DEFAULT_METRICS_PATH,
            aggregate_path=DEFAULT_AGGREGATE_PATH,
        )
        assert summary["source_hashes"][METRICS_RELPATH] == REAL_METRICS_SHA256, (
            "metrics.json SHA-256 drift detected; update REAL_METRICS_SHA256"
        )
        assert summary["source_hashes"][AGGREGATE_RELPATH] == REAL_AGGREGATE_SHA256, (
            "aggregate_summary.json SHA-256 drift detected; update REAL_AGGREGATE_SHA256"
        )

    def test_produces_eleven_canonical_checkpoints(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        metrics, aggregate = real_artifacts
        summary = _build(metrics, aggregate)
        ckpts = [r["checkpoint"] for r in summary["checkpoints"]]
        assert ckpts == list(EXPERIMENT_CHECKPOINTS)
        assert len(ckpts) == 11

    def test_first_row_main_has_identity_m1(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        metrics, aggregate = real_artifacts
        summary = _build(metrics, aggregate)
        assert summary["checkpoints"][0]["checkpoint"] == "main"
        assert summary["checkpoints"][0]["m1_mean_cos_vs_main"] == 1.0

    def test_step_100_and_step_1000_are_duplicates(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """The two duplicate-weight checkpoints must produce identical rows."""
        metrics, aggregate = real_artifacts
        summary = _build(metrics, aggregate)
        rows = {r["checkpoint"]: r for r in summary["checkpoints"]}
        a = rows["step_100"]
        b = rows["step_1000"]
        for key in AGGREGATE_METRIC_KEYS:
            assert a[key] == b[key], (
                f"duplicate-weight checkpoints disagree on {key}: {a[key]} vs {b[key]}"
            )
        for key in DOWNSTREAM_TARGET_KEYS:
            assert a[key] == b[key], (
                f"duplicate-weight checkpoints disagree on {key}: {a[key]} vs {b[key]}"
            )

    def test_cpp_pass_at_1_is_constant_zero(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Under the strict raw protocol, C++ pass@1 is 0 across all 11 rows."""
        metrics, aggregate = real_artifacts
        summary = _build(metrics, aggregate)
        for row in summary["checkpoints"]:
            assert row["cpp_pass_at_1"] == 0.0

    def test_correlations_match_legacy_ad_hoc_values(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """The deterministic producer reproduces the original ad-hoc values.

        These constants are the canonical correlations the legacy ad-hoc
        one-liner produced. Any drift here means the producer's math diverged
        from the legacy semantics -- which would be a regression even if the
        new value is "more correct".
        """
        metrics, aggregate = real_artifacts
        summary = _build(metrics, aggregate)
        expected = {
            "m1_mean_cos_vs_main": {
                "python_pass_at_1": {
                    "pearson_r": -0.732437358152346,
                    "pearson_p": 0.010366051535435716,
                    "spearman_rho": -0.7110166549574457,
                    "spearman_p": 0.014167220861190788,
                },
                "mmlu_accuracy": {
                    "pearson_r": 0.4918682875113296,
                    "pearson_p": 0.1243555222985842,
                    "spearman_rho": 0.3739521127743522,
                    "spearman_p": 0.25723956552008365,
                },
            },
            "m4_syntax_auroc": {
                "python_pass_at_1": {
                    "pearson_r": -0.8925007512205,
                    "pearson_p": 0.00021903639681761513,
                    "spearman_rho": -0.9312024577829773,
                    "spearman_p": 3.112815148412833e-05,
                },
                "mmlu_accuracy": {
                    "pearson_r": 0.5948059262284704,
                    "pearson_p": 0.05358277121661782,
                    "spearman_rho": 0.4701112274877571,
                    "spearman_p": 0.14452569332197937,
                },
            },
        }
        for metric_key, by_tgt in expected.items():
            for tgt, exp in by_tgt.items():
                got = summary["correlations_all_11"][metric_key][tgt]
                for stat, v in exp.items():
                    assert got[stat] == pytest.approx(v, abs=1e-10), (
                        f"correlations[{metric_key!r}][{tgt!r}].{stat} drifted: "
                        f"got {got[stat]} expected {v}"
                    )

    def test_validation_passes_against_real_sources(
        self, real_artifacts: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        metrics, aggregate = real_artifacts
        summary = _build(
            metrics,
            aggregate,
            metrics_path=DEFAULT_METRICS_PATH,
            aggregate_path=DEFAULT_AGGREGATE_PATH,
        )
        validate_summary(
            summary,
            metrics_path=DEFAULT_METRICS_PATH,
            aggregate_path=DEFAULT_AGGREGATE_PATH,
        )

    def test_validation_detects_real_source_drift(
        self,
        real_artifacts: tuple[dict[str, Any], dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Rebuild against pristine sources, then mutate metrics; validator fires."""
        metrics, aggregate = real_artifacts
        # Copy the real sources into tmp so we can mutate without disturbing
        # the canonical on-disk files.
        metrics_path = tmp_path / "metrics.json"
        aggregate_path = tmp_path / "aggregate_summary.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        summary = _build(
            load_json(metrics_path),
            load_json(aggregate_path),
            metrics_path=metrics_path,
            aggregate_path=aggregate_path,
        )
        # Mutate one M1 cosine by 0.001 -- the producer's hash must catch it.
        mutated = load_json(metrics_path)
        mutated["m1_checkpoint_cosine"]["by_layer"]["3"]["by_checkpoint"]["step_2900"][
            "cos_vs_main"
        ] += 0.001
        metrics_path.write_text(json.dumps(mutated), encoding="utf-8")
        with pytest.raises(SourceHashMismatch, match="metrics.json hash changed"):
            validate_summary(
                summary,
                metrics_path=metrics_path,
                aggregate_path=aggregate_path,
            )

    def test_two_runs_produce_identical_bytes(
        self,
        real_artifacts: tuple[dict[str, Any], dict[str, Any]],
        tmp_path: Path,
    ) -> None:
        """Determinism: two independent runs write byte-identical files."""
        metrics, aggregate = real_artifacts
        metrics_path = tmp_path / "metrics.json"
        aggregate_path = tmp_path / "aggregate_summary.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        out1 = tmp_path / "out1.json"
        out2 = tmp_path / "out2.json"
        rc1 = cli.main(
            [
                "--metrics",
                str(metrics_path),
                "--aggregate",
                str(aggregate_path),
                "--output",
                str(out1),
            ]
        )
        rc2 = cli.main(
            [
                "--metrics",
                str(metrics_path),
                "--aggregate",
                str(aggregate_path),
                "--output",
                str(out2),
            ]
        )
        assert rc1 == 0 and rc2 == 0
        assert out1.read_bytes() == out2.read_bytes()
