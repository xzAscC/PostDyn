"""Tests for the four-metric driver ``experiments/run_rl_zero_syntax_metrics.py``.

These tests are fully model-independent. They exercise:

  * **Pure metric math** (M1-M4) with known numpy vectors -- verifies exact
    cosine / delta / norm values without touching disk.
  * **On-disk integration** -- synthetic concept-vector safetensors files and
    synthetic probe-activation safetensors files are written under a tmp
    directory, the full ``compute_all_metrics`` pipeline is run, and the
    output structure and values are verified.
  * **Validation** -- finite values, checkpoint/layer coverage, ``raw``
    protocol, and the exactly-four-metric-keys invariant.
  * **Atomic write** -- no ``.tmp`` residue, valid JSON round-trip.
  * **CLI** -- argument parsing and the ``main`` entry point.

No GPU, no network, no model -- only ``numpy``, ``torch`` (for writing
synthetic safetensors), and ``pytest`` are used.
"""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import experiments.run_rl_zero_syntax_metrics as cli
from experiments.run_rl_zero_syntax_metrics import (
    DEFAULT_ALPHA,
    DEFAULT_METRICS_PATH,
    DEFAULT_SEED,
    DEFAULT_STANDARDIZE,
    HUMANEVAL_X_REVISION,
    METRIC_KEYS,
    SCHEMA,
    SOURCE_MODELS,
    VERSION,
    _cv_raw_dir_to_np,
    _resolve_n_samples,
    checkpoint_model_map,
    compute_all_metrics,
    default_concept_vectors_root,
    main,
    make_activation_loader,
    metric_m1_checkpoint_cosine,
    metric_m2_target_related_delta_cos,
    metric_m3_raw_direction_magnitude_delta,
    metric_m4_eight_class_grouped_logistic_separability,
    parse_args,
    safe_cosine,
    validate_metrics,
    write_metrics_json,
)
from experiments.run_rl_zero_syntax_metrics import (
    PROBE_CLASSES as CLI_PROBE_CLASSES,
)
from src.concept_dynamics import ConceptVector, save_concept_vectors
from src.probe_activations import (
    PROTOCOL,
    ProbeRecord,
    save_layer_activations,
    save_records_json,
)
from src.rl_zero_experiment import (
    BASE_CHECKPOINT,
    CONTROL_CONCEPT,
    PROBE_CLASSES,
    RELATED_CONCEPTS,
    TARGET_CONCEPT,
)

# =============================================================================
# Constants reused across tests
# =============================================================================

#: Small dimensionality for synthetic concept vectors (6 concepts fit easily).
D_CONCEPT: int = 8

#: Dimensionality for synthetic probe activations (8 classes -> 8 dims).
D_ACTIVATION: int = 8

#: Two-layer / two-checkpoint mini grid for fast tests.
TEST_LAYERS: list[int] = [3, 6]
TEST_CHECKPOINTS: list[str] = [BASE_CHECKPOINT, "step_100"]


# =============================================================================
# Vector helpers
# =============================================================================


def _unit(dim: int, idx: int) -> np.ndarray:
    """Unit vector along dimension ``idx`` in ``dim`` dimensions."""
    v = np.zeros(dim, dtype=np.float64)
    v[idx] = 1.0
    return v


def _scaled(dim: int, idx: int, scale: float) -> np.ndarray:
    """Scaled unit vector ``scale * e_idx``."""
    v = np.zeros(dim, dtype=np.float64)
    v[idx] = scale
    return v


# =============================================================================
# Concept-direction builders (pure numpy)
# =============================================================================


def _build_concept_dirs_known() -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Build a ConceptDirs dict with known orthogonal directions.

    Layout (per checkpoint, per layer -- identical across layers)::

        base (main):
            target  = e0,  cpp = e1,  js = e2,  java = e3,  go = e4,  ctrl = e5
        step_100:
            target  = e0 + e1  (rotated toward cpp)
            others  unchanged

    This gives non-trivial cosines for M1 and deltas for M2/M3.
    """
    base_dirs = {
        TARGET_CONCEPT: _unit(D_CONCEPT, 0),
        "code_python_vs_cpp": _unit(D_CONCEPT, 1),
        "code_python_vs_js": _unit(D_CONCEPT, 2),
        "code_python_vs_java": _unit(D_CONCEPT, 3),
        "code_python_vs_go": _unit(D_CONCEPT, 4),
        CONTROL_CONCEPT: _unit(D_CONCEPT, 5),
    }
    step_dirs = dict(base_dirs)
    step_dirs[TARGET_CONCEPT] = _unit(D_CONCEPT, 0) + _unit(D_CONCEPT, 1)

    result: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for ckpt, dirs in [(BASE_CHECKPOINT, base_dirs), ("step_100", step_dirs)]:
        result[ckpt] = {layer: dict(dirs) for layer in TEST_LAYERS}
    return result


def _build_target_dirs_known() -> dict[str, dict[int, np.ndarray]]:
    """Extract target-only directions from ``_build_concept_dirs_known``."""
    all_dirs = _build_concept_dirs_known()
    return {
        ckpt: {layer: all_dirs[ckpt][layer][TARGET_CONCEPT] for layer in TEST_LAYERS}
        for ckpt in TEST_CHECKPOINTS
    }


def _build_concept_dirs_scaled() -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """Directions with non-unit norms for M3 magnitude tests.

    base:  target = 2*e0   (norm = 2.0)
    step:  target = 3*e0   (norm = 3.0)  ->  delta = 1.0
    """
    base_dirs = {
        TARGET_CONCEPT: _scaled(D_CONCEPT, 0, 2.0),
        "code_python_vs_cpp": _unit(D_CONCEPT, 1),
        "code_python_vs_js": _unit(D_CONCEPT, 2),
        "code_python_vs_java": _unit(D_CONCEPT, 3),
        "code_python_vs_go": _unit(D_CONCEPT, 4),
        CONTROL_CONCEPT: _unit(D_CONCEPT, 5),
    }
    step_dirs = dict(base_dirs)
    step_dirs[TARGET_CONCEPT] = _scaled(D_CONCEPT, 0, 3.0)
    result: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for ckpt, dirs in [(BASE_CHECKPOINT, base_dirs), ("step_100", step_dirs)]:
        result[ckpt] = {layer: dict(dirs) for layer in TEST_LAYERS}
    return result


# =============================================================================
# Synthetic activation data
# =============================================================================


def _make_separable_data(
    n_per_class: int = 15,
    n_classes: int = 8,
    n_features: int = 8,
    seed: int = 0,
    scale: float = 10.0,
    noise: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build perfectly linearly-separable multi-class data.

    Each class ``c`` occupies a distinct cluster centered at ``scale * e_c``
    so the one-vs-rest logistic probe recovers near-perfect separation.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * n_classes
    X = np.zeros((n, n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    groups: list[str] = []
    for c in range(n_classes):
        start = c * n_per_class
        X[start : start + n_per_class, c % n_features] = scale
        X[start : start + n_per_class] += rng.normal(
            0.0, noise, (n_per_class, n_features)
        )
        y[start : start + n_per_class] = c
        groups.extend(f"g_{c}_{i}" for i in range(n_per_class))
    return X, y, groups


def _separable_loader(
    n_per_class: int = 15,
) -> cli.ActivationLoader:
    """Return a loader that always yields separable data (ignores ckpt/layer)."""
    features, labels, groups = _make_separable_data(n_per_class=n_per_class)

    def load(checkpoint: str, layer: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        return features, labels, groups

    return load


def _build_synthetic_probe_records(n_per_class: int = 10) -> list[ProbeRecord]:
    """Build ``n_per_class`` synthetic probe records for each of the 8 classes."""
    records: list[ProbeRecord] = []
    for label in PROBE_CLASSES:
        for i in range(n_per_class):
            records.append(
                ProbeRecord(
                    sample_id=f"{label}:{i}",
                    label=label,
                    text=f"synthetic {label} sample {i}",
                    group_id=f"grp:{label}:{i}",
                    source_id=f"{label}_{i}",
                )
            )
    return records


def _build_synthetic_concept_vector(
    concept_name: str,
    model_name: str,
    layer_idx: int,
    direction: np.ndarray,
) -> ConceptVector:
    """Build a :class:`ConceptVector` from a numpy direction."""
    raw = torch.from_numpy(direction).to(torch.float32)
    norm = raw.norm()
    steering = raw / norm if norm > 1e-10 else raw.clone()
    return ConceptVector(
        concept_name=concept_name,
        model_name=model_name,
        layer_idx=layer_idx,
        steering_vector=steering,
        raw_direction=raw,
        positive_mean=raw.clone(),
        negative_mean=torch.zeros_like(raw),
        positive_std=torch.ones_like(raw),
        negative_std=torch.ones_like(raw),
        n_positive=50,
        n_negative=50,
        d_model=int(raw.shape[0]),
    )


def _write_synthetic_concept_vectors(
    root: str,
    ckpt_model_map: dict[str, tuple[str, str]],
    layers: list[int],
    checkpoints: list[str],
    dirs_by_ckpt: dict[str, dict[str, np.ndarray]],
) -> None:
    """Write synthetic concept-vector safetensors files for every triple."""
    from src.contrastive_datasets import load_contrastive_texts

    for ckpt in checkpoints:
        model_name, ckpt_name = ckpt_model_map[ckpt]
        dirs = dirs_by_ckpt[ckpt]
        concept_sources = {name: load_contrastive_texts(name, 50) for name in dirs}
        for layer in layers:
            vectors = {
                name: _build_synthetic_concept_vector(
                    name, model_name, layer, direction
                )
                for name, direction in dirs.items()
            }
            save_concept_vectors(
                vectors,
                root,
                model_name,
                layer,
                checkpoint=ckpt_name,
                protocol="raw",
                revision=ckpt_name,
                hf_id=model_name,
                max_seq_len=2048,
                use_chat_template=False,
                concept_sources=concept_sources,
            )


def _write_synthetic_activations(
    root: str,
    ckpt_model_map: dict[str, tuple[str, str]],
    layers: list[int],
    checkpoints: list[str],
    records: list[ProbeRecord],
    d: int = D_ACTIVATION,
) -> None:
    """Write synthetic probe-activation safetensors files for every triple.

    Also writes the global ``records.json`` once so the hardened activation
    loader (which loads it a single time and binds every layer sidecar to it
    via ``validate_sidecar_record_identity``) can verify text/source provenance.
    """
    save_records_json(root, records)
    n = len(records)
    rng = np.random.default_rng(42)
    for ckpt in checkpoints:
        model_name, ckpt_name = ckpt_model_map[ckpt]
        for layer in layers:
            # Deterministic but layer/checkpoint-varying separable features.
            features = np.zeros((n, d), dtype=np.float32)
            label_to_idx = {name: i for i, name in enumerate(PROBE_CLASSES)}
            for i, r in enumerate(records):
                c = label_to_idx[r.label]
                features[i, c % d] = 10.0
                features[i] += rng.normal(0.0, 0.01, d)
            activations = torch.from_numpy(features).to(torch.float32)
            save_layer_activations(
                root,
                model_name,
                ckpt_name,
                layer,
                activations,
                records,
            )


# =============================================================================
# 1. safe_cosine
# =============================================================================


class TestSafeCosine:
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 2.0, 3.0])
        assert safe_cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert safe_cosine(_unit(4, 0), _unit(4, 1)) == pytest.approx(0.0)

    def test_anti_parallel(self) -> None:
        v = np.array([1.0, 1.0])
        assert safe_cosine(v, -v) == pytest.approx(-1.0)

    def test_known_cosine(self) -> None:
        # cos(45 degrees) = 1/sqrt(2)
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 1.0])
        assert safe_cosine(a, b) == pytest.approx(1.0 / math.sqrt(2))

    def test_zero_vector_returns_zero(self) -> None:
        zero = np.zeros(4)
        assert safe_cosine(zero, _unit(4, 0)) == 0.0
        assert safe_cosine(_unit(4, 0), zero) == 0.0

    def test_clamping(self) -> None:
        # Floating-point drift beyond [-1, 1] is clamped.
        a = np.array([3.0, 4.0])  # norm = 5
        b = np.array([6.0, 8.0])  # norm = 10, dot = 50
        assert safe_cosine(a, b) == 1.0  # exactly parallel


# =============================================================================
# 2. M1 -- checkpoint cosine
# =============================================================================


class TestM1CheckpointCosine:
    def test_self_reference_is_one(self) -> None:
        target_dirs = _build_target_dirs_known()
        m1 = metric_m1_checkpoint_cosine(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            ("step_100",),
        )
        for layer in TEST_LAYERS:
            blk = m1["by_layer"][str(layer)]["by_checkpoint"]
            assert blk[BASE_CHECKPOINT]["cos_vs_main"] == pytest.approx(1.0)
            assert blk["step_100"]["cos_vs_step_100"] == pytest.approx(1.0)

    def test_known_cosine_step_vs_base(self) -> None:
        target_dirs = _build_target_dirs_known()
        m1 = metric_m1_checkpoint_cosine(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            ("step_100",),
        )
        for layer in TEST_LAYERS:
            blk = m1["by_layer"][str(layer)]["by_checkpoint"]
            # step_100 target = e0+e1, base target = e0 -> cos = 1/sqrt(2)
            assert blk["step_100"]["cos_vs_main"] == pytest.approx(1.0 / math.sqrt(2))
            # base target = e0, step target = e0+e1 -> cos = 1/sqrt(2)
            assert blk[BASE_CHECKPOINT]["cos_vs_step_100"] == pytest.approx(
                1.0 / math.sqrt(2)
            )

    def test_references_list(self) -> None:
        target_dirs = _build_target_dirs_known()
        m1 = metric_m1_checkpoint_cosine(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            ("step_100",),
        )
        assert m1["references"] == ["main", "step_100"]

    def test_no_optional_references(self) -> None:
        target_dirs = _build_target_dirs_known()
        m1 = metric_m1_checkpoint_cosine(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            (),
        )
        assert m1["references"] == ["main"]
        for layer in TEST_LAYERS:
            blk = m1["by_layer"][str(layer)]["by_checkpoint"]
            for ckpt in TEST_CHECKPOINTS:
                assert "cos_vs_step_100" not in blk[ckpt]
                assert "cos_vs_main" in blk[ckpt]

    def test_coverage(self) -> None:
        target_dirs = _build_target_dirs_known()
        m1 = metric_m1_checkpoint_cosine(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            ("step_100",),
        )
        assert set(m1["by_layer"].keys()) == {str(l) for l in TEST_LAYERS}
        for layer in TEST_LAYERS:
            ckpts = m1["by_layer"][str(layer)]["by_checkpoint"]
            assert set(ckpts.keys()) == set(TEST_CHECKPOINTS)

    def test_missing_reference_raises(self) -> None:
        target_dirs = _build_target_dirs_known()
        with pytest.raises(ValueError, match="missing from target_dirs"):
            metric_m1_checkpoint_cosine(
                target_dirs,
                TEST_LAYERS,
                TEST_CHECKPOINTS,
                "nonexistent",
                (),
            )


# =============================================================================
# 3. M2 -- target-vs-related Delta-cos + control diagnostic
# =============================================================================


class TestM2TargetRelatedDeltaCos:
    def test_delta_zero_at_base(self) -> None:
        all_dirs = _build_concept_dirs_known()
        m2 = metric_m2_target_related_delta_cos(
            all_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
            RELATED_CONCEPTS,
            CONTROL_CONCEPT,
        )
        for layer in TEST_LAYERS:
            base_block = m2["by_layer"][str(layer)]["by_checkpoint"][BASE_CHECKPOINT]
            for concept in RELATED_CONCEPTS:
                assert base_block["related"][concept]["delta_cos"] == pytest.approx(0.0)
            assert base_block["control_diagnostic"]["delta_cos"] == pytest.approx(0.0)

    def test_known_delta_at_step(self) -> None:
        all_dirs = _build_concept_dirs_known()
        m2 = metric_m2_target_related_delta_cos(
            all_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
            RELATED_CONCEPTS,
            CONTROL_CONCEPT,
        )
        for layer in TEST_LAYERS:
            step_block = m2["by_layer"][str(layer)]["by_checkpoint"]["step_100"]
            # step target = e0+e1, cpp = e1 -> cos = 1/sqrt(2)
            # base target = e0,  cpp = e1 -> cos = 0
            # delta = 1/sqrt(2)
            cpp = step_block["related"]["code_python_vs_cpp"]
            assert cpp["cos_current"] == pytest.approx(1.0 / math.sqrt(2))
            assert cpp["cos_base"] == pytest.approx(0.0)
            assert cpp["delta_cos"] == pytest.approx(1.0 / math.sqrt(2))

            # js, java, go remain orthogonal to the step target (e0+e1).
            for concept in (
                "code_python_vs_js",
                "code_python_vs_java",
                "code_python_vs_go",
            ):
                entry = step_block["related"][concept]
                assert entry["cos_current"] == pytest.approx(0.0)
                assert entry["delta_cos"] == pytest.approx(0.0)

    def test_exactly_four_related_concepts(self) -> None:
        all_dirs = _build_concept_dirs_known()
        m2 = metric_m2_target_related_delta_cos(
            all_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
            RELATED_CONCEPTS,
            CONTROL_CONCEPT,
        )
        for layer in TEST_LAYERS:
            for ckpt in TEST_CHECKPOINTS:
                related = m2["by_layer"][str(layer)]["by_checkpoint"][ckpt]["related"]
                assert set(related.keys()) == set(RELATED_CONCEPTS)
                assert len(related) == 4

    def test_control_diagnostic_present(self) -> None:
        all_dirs = _build_concept_dirs_known()
        m2 = metric_m2_target_related_delta_cos(
            all_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
            RELATED_CONCEPTS,
            CONTROL_CONCEPT,
        )
        for layer in TEST_LAYERS:
            for ckpt in TEST_CHECKPOINTS:
                block = m2["by_layer"][str(layer)]["by_checkpoint"][ckpt]
                assert "control_diagnostic" in block
                ctrl = block["control_diagnostic"]
                assert set(ctrl.keys()) == {"cos_current", "cos_base", "delta_cos"}

    def test_control_is_not_a_fifth_metric_key(self) -> None:
        all_dirs = _build_concept_dirs_known()
        m2 = metric_m2_target_related_delta_cos(
            all_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
            RELATED_CONCEPTS,
            CONTROL_CONCEPT,
        )
        # The M2 block is the only place the control appears; it is not a
        # top-level metric key.
        assert "control_diagnostic" not in METRIC_KEYS


# =============================================================================
# 4. M3 -- target raw-direction magnitude Delta
# =============================================================================


class TestM3RawDirectionMagnitudeDelta:
    def test_delta_zero_at_base(self) -> None:
        target_dirs = {
            ckpt: {layer: _scaled(D_CONCEPT, 0, 2.0) for layer in TEST_LAYERS}
            for ckpt in TEST_CHECKPOINTS
        }
        m3 = metric_m3_raw_direction_magnitude_delta(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
        )
        for layer in TEST_LAYERS:
            base_entry = m3["by_layer"][str(layer)]["by_checkpoint"][BASE_CHECKPOINT]
            assert base_entry["delta"] == pytest.approx(0.0)

    def test_known_delta(self) -> None:
        all_dirs = _build_concept_dirs_scaled()
        target_dirs = {
            ckpt: {
                layer: all_dirs[ckpt][layer][TARGET_CONCEPT] for layer in TEST_LAYERS
            }
            for ckpt in TEST_CHECKPOINTS
        }
        m3 = metric_m3_raw_direction_magnitude_delta(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
        )
        for layer in TEST_LAYERS:
            blk = m3["by_layer"][str(layer)]["by_checkpoint"]
            assert blk[BASE_CHECKPOINT]["norm_current"] == pytest.approx(2.0)
            assert blk[BASE_CHECKPOINT]["norm_base"] == pytest.approx(2.0)
            assert blk[BASE_CHECKPOINT]["delta"] == pytest.approx(0.0)
            assert blk["step_100"]["norm_current"] == pytest.approx(3.0)
            assert blk["step_100"]["norm_base"] == pytest.approx(2.0)
            assert blk["step_100"]["delta"] == pytest.approx(1.0)

    def test_uses_raw_not_normalized(self) -> None:
        """M3 must use the un-normalized raw direction, not the unit vector."""
        # If both directions have the same norm, delta is zero even though
        # their orientations differ.
        target_dirs = {
            BASE_CHECKPOINT: {l: _unit(D_CONCEPT, 0) for l in TEST_LAYERS},
            "step_100": {l: _unit(D_CONCEPT, 1) for l in TEST_LAYERS},
        }
        m3 = metric_m3_raw_direction_magnitude_delta(
            target_dirs,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            BASE_CHECKPOINT,
            TARGET_CONCEPT,
        )
        for layer in TEST_LAYERS:
            for ckpt in TEST_CHECKPOINTS:
                assert m3["by_layer"][str(layer)]["by_checkpoint"][ckpt][
                    "delta"
                ] == pytest.approx(0.0)


# =============================================================================
# 5. M4 -- eight-class grouped logistic separability
# =============================================================================


class TestM4EightClassGroupedLogisticSeparability:
    def test_separable_data_high_accuracy(self) -> None:
        loader = _separable_loader(n_per_class=15)
        m4 = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            PROBE_CLASSES,
            n_folds=3,
        )
        for layer in TEST_LAYERS:
            for ckpt in TEST_CHECKPOINTS:
                by_label = m4["by_layer"][str(layer)]["by_checkpoint"][ckpt]["by_label"]
                for name in PROBE_CLASSES:
                    assert by_label[name]["balanced_accuracy"] > 0.95, name
                    assert by_label[name]["auroc"] > 0.95, name

    def test_all_eight_classes_present(self) -> None:
        loader = _separable_loader()
        m4 = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            [TEST_LAYERS[0]],
            [TEST_CHECKPOINTS[0]],
            PROBE_CLASSES,
            n_folds=3,
        )
        by_label = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][
            TEST_CHECKPOINTS[0]
        ]["by_label"]
        assert set(by_label.keys()) == set(PROBE_CLASSES)
        assert len(by_label) == 8

    def test_deterministic_same_seed(self) -> None:
        loader = _separable_loader(n_per_class=12)
        m4_a = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            [TEST_LAYERS[0]],
            [TEST_CHECKPOINTS[0]],
            PROBE_CLASSES,
            n_folds=3,
            seed=42,
        )
        m4_b = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            [TEST_LAYERS[0]],
            [TEST_CHECKPOINTS[0]],
            PROBE_CLASSES,
            n_folds=3,
            seed=42,
        )
        a = m4_a["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]][
            "by_label"
        ]
        b = m4_b["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]][
            "by_label"
        ]
        for name in PROBE_CLASSES:
            assert a[name]["auroc"] == b[name]["auroc"]
            assert a[name]["balanced_accuracy"] == b[name]["balanced_accuracy"]

    def test_label_names_match_probe_classes(self) -> None:
        loader = _separable_loader()
        m4 = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            [TEST_LAYERS[0]],
            [TEST_CHECKPOINTS[0]],
            PROBE_CLASSES,
            n_folds=3,
        )
        assert m4["label_names"] == list(PROBE_CLASSES)

    def test_linear_probe_params_recorded(self) -> None:
        loader = _separable_loader()
        m4 = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            [TEST_LAYERS[0]],
            [TEST_CHECKPOINTS[0]],
            PROBE_CLASSES,
            n_folds=4,
            alpha=2.5,
            seed=7,
            standardize=False,
        )
        params = m4["linear_probe_params"]
        assert params == {"n_folds": 4, "alpha": 2.5, "seed": 7, "standardize": False}

    def test_method_is_logistic(self) -> None:
        """The M4 block mirrors ``LinearProbeResult.method`` == 'logistic'."""
        loader = _separable_loader()
        m4 = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            [TEST_LAYERS[0]],
            [TEST_CHECKPOINTS[0]],
            PROBE_CLASSES,
            n_folds=3,
        )
        assert m4["method"] == "logistic"

    def test_method_logistic_across_whole_grid(self) -> None:
        loader = _separable_loader()
        m4 = metric_m4_eight_class_grouped_logistic_separability(
            loader,
            TEST_LAYERS,
            TEST_CHECKPOINTS,
            PROBE_CLASSES,
            n_folds=3,
        )
        assert m4["method"] == "logistic"


# =============================================================================
# 6. Validation
# =============================================================================


class TestValidateMetrics:
    """Validate the strict v2 schema, provenance, and per-cell field contract.

    The valid base fixture is built **once** from the real metric functions on
    in-memory numpy directions (M1/M2/M3) and a small separable activation
    loader (M4), then deep-copied per mutation test. This exercises the exact
    field/type/range/count/label structure validate_metrics enforces, instead
    of arbitrary ``{value: 0.5}`` placeholder cells.
    """

    _N_FOLDS: int = 3
    _valid_cache: dict[str, object] | None = None

    @classmethod
    def _build_valid_metrics(cls) -> dict[str, object]:
        """Compute a fully-structured valid metrics dict (cached on the class)."""
        if cls._valid_cache is None:
            all_dirs = _build_concept_dirs_known()
            target_dirs = _build_target_dirs_known()
            related = list(RELATED_CONCEPTS)
            m1 = metric_m1_checkpoint_cosine(
                target_dirs,
                TEST_LAYERS,
                TEST_CHECKPOINTS,
                BASE_CHECKPOINT,
                ("step_100",),
            )
            m2 = metric_m2_target_related_delta_cos(
                all_dirs,
                TEST_LAYERS,
                TEST_CHECKPOINTS,
                BASE_CHECKPOINT,
                TARGET_CONCEPT,
                related,
                CONTROL_CONCEPT,
            )
            m3 = metric_m3_raw_direction_magnitude_delta(
                target_dirs,
                TEST_LAYERS,
                TEST_CHECKPOINTS,
                BASE_CHECKPOINT,
                TARGET_CONCEPT,
            )
            loader = _separable_loader(n_per_class=10)
            m4 = metric_m4_eight_class_grouped_logistic_separability(
                loader,
                TEST_LAYERS,
                TEST_CHECKPOINTS,
                PROBE_CLASSES,
                n_folds=cls._N_FOLDS,
            )
            n_samples = _resolve_n_samples(m4, TEST_LAYERS, TEST_CHECKPOINTS)
            cls._valid_cache = {
                "schema": SCHEMA,
                "version": VERSION,
                "metadata": {
                    "checkpoints": list(TEST_CHECKPOINTS),
                    "layers": list(TEST_LAYERS),
                    "concepts": {
                        "target": TARGET_CONCEPT,
                        "related": related,
                        "control": CONTROL_CONCEPT,
                    },
                    "probe_classes": list(PROBE_CLASSES),
                    "protocol": PROTOCOL,
                    "n_samples": n_samples,
                    "base_checkpoint": BASE_CHECKPOINT,
                    "optional_references": ["step_100"],
                    "source_models": dict(SOURCE_MODELS),
                    "humaneval_x_revision": HUMANEVAL_X_REVISION,
                    "linear_probe": {
                        "method": "logistic",
                        "n_folds": cls._N_FOLDS,
                        "alpha": DEFAULT_ALPHA,
                        "seed": DEFAULT_SEED,
                        "standardize": DEFAULT_STANDARDIZE,
                    },
                },
                METRIC_KEYS[0]: m1,
                METRIC_KEYS[1]: m2,
                METRIC_KEYS[2]: m3,
                METRIC_KEYS[3]: m4,
            }
        return cls._valid_cache

    def _valid_metrics(self) -> dict[str, object]:
        """Return a fresh deep copy of the cached valid metrics dict."""
        return copy.deepcopy(self._build_valid_metrics())

    def test_valid_passes(self) -> None:
        validate_metrics(
            self._valid_metrics(),
            expected_checkpoints=TEST_CHECKPOINTS,
            expected_layers=TEST_LAYERS,
        )

    # --- top-level structure ---------------------------------------------

    def test_missing_metric_key_raises(self) -> None:
        metrics = self._valid_metrics()
        del metrics["m4_eight_class_grouped_logistic_separability"]
        with pytest.raises(ValueError, match="metric key mismatch"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_extra_metric_key_raises(self) -> None:
        metrics = self._valid_metrics()
        metrics["m5_extra"] = {"by_layer": {}}
        with pytest.raises(ValueError, match="metric key mismatch"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_schema_raises(self) -> None:
        metrics = self._valid_metrics()
        metrics["schema"] = "wrong"
        with pytest.raises(ValueError, match="schema"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_version_raises(self) -> None:
        metrics = self._valid_metrics()
        metrics["version"] = 99
        with pytest.raises(ValueError, match="version"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_missing_metadata_raises(self) -> None:
        metrics = self._valid_metrics()
        del metrics["metadata"]
        with pytest.raises(ValueError, match="metadata must be a dict"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_non_dict_metadata_raises(self) -> None:
        metrics = self._valid_metrics()
        metrics["metadata"] = "not-a-dict"
        with pytest.raises(ValueError, match="metadata must be a dict"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- protocol --------------------------------------------------------

    def test_wrong_protocol_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["protocol"] = "chat"
        with pytest.raises(ValueError, match="protocol"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- metadata n_samples ----------------------------------------------

    def test_missing_metadata_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        del metadata["n_samples"]
        with pytest.raises(ValueError, match="non-boolean int"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_bool_metadata_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["n_samples"] = True
        with pytest.raises(ValueError, match="non-boolean int"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_zero_metadata_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["n_samples"] = 0
        with pytest.raises(ValueError, match="must be positive"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_negative_metadata_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["n_samples"] = -1
        with pytest.raises(ValueError, match="must be positive"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- source model / dataset provenance -------------------------------

    def test_missing_source_models_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        del metadata["source_models"]
        with pytest.raises(ValueError, match="source_models"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_source_models_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["source_models"] = {"olmo3-base": "wrong/repo"}
        with pytest.raises(ValueError, match="source_models"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_missing_humaneval_x_revision_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        del metadata["humaneval_x_revision"]
        with pytest.raises(ValueError, match="humaneval_x_revision"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_humaneval_x_revision_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["humaneval_x_revision"] = "deadbeef"
        with pytest.raises(ValueError, match="humaneval_x_revision"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_metadata_concepts_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        concepts = metadata["concepts"]
        assert isinstance(concepts, dict)
        concepts["target"] = "something_else"
        with pytest.raises(ValueError, match="concepts.target"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_metadata_checkpoints_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["checkpoints"] = ["main", "step_9999"]
        with pytest.raises(ValueError, match="checkpoints"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_metadata_layers_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["layers"] = [3, 6, 99]
        with pytest.raises(ValueError, match="layers"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_wrong_probe_classes_raises(self) -> None:
        metrics = self._valid_metrics()
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        metadata["probe_classes"] = ["a", "b"]
        with pytest.raises(ValueError, match="probe_classes"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- coverage --------------------------------------------------------

    def test_missing_checkpoint_raises(self) -> None:
        metrics = self._valid_metrics()
        m1 = metrics[METRIC_KEYS[0]]
        assert isinstance(m1, dict)
        ckpt_blk = m1["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"]
        assert isinstance(ckpt_blk, dict)
        del ckpt_blk["step_100"]
        with pytest.raises(ValueError, match="checkpoint mismatch"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_layer_coverage_mismatch(self) -> None:
        metrics = self._valid_metrics()
        m1 = metrics[METRIC_KEYS[0]]
        assert isinstance(m1, dict)
        # Removing a layer key from one metric block trips the coverage loop
        # even though metadata.layers still matches the expected schedule.
        del m1["by_layer"][str(TEST_LAYERS[0])]
        with pytest.raises(ValueError, match="layer mismatch"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- finiteness ------------------------------------------------------

    def test_non_finite_raises(self) -> None:
        metrics = self._valid_metrics()
        m1 = metrics[METRIC_KEYS[0]]
        assert isinstance(m1, dict)
        ckpt_blk = m1["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"]
        assert isinstance(ckpt_blk, dict)
        entry = ckpt_blk[TEST_CHECKPOINTS[0]]
        assert isinstance(entry, dict)
        cos_key = next(iter(entry.keys()))
        entry[cos_key] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- M1 field structure ----------------------------------------------

    def test_m1_cosine_out_of_range_raises(self) -> None:
        metrics = self._valid_metrics()
        m1 = metrics[METRIC_KEYS[0]]
        assert isinstance(m1, dict)
        entry = m1["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][
            TEST_CHECKPOINTS[0]
        ]
        assert isinstance(entry, dict)
        cos_key = next(iter(entry.keys()))
        entry[cos_key] = 1.5
        with pytest.raises(ValueError, match="out of range"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m1_wrong_reference_keys_raises(self) -> None:
        metrics = self._valid_metrics()
        m1 = metrics[METRIC_KEYS[0]]
        assert isinstance(m1, dict)
        entry = m1["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][
            TEST_CHECKPOINTS[0]
        ]
        assert isinstance(entry, dict)
        entry["cos_vs_bogus"] = 0.5
        with pytest.raises(ValueError, match="keys"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- M2 field structure ----------------------------------------------

    def test_m2_wrong_related_count_raises(self) -> None:
        metrics = self._valid_metrics()
        m2 = metrics[METRIC_KEYS[1]]
        assert isinstance(m2, dict)
        m2["related_concepts"] = list(RELATED_CONCEPTS)[:2]
        with pytest.raises(ValueError, match="related_concepts"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m2_missing_control_raises(self) -> None:
        metrics = self._valid_metrics()
        m2 = metrics[METRIC_KEYS[1]]
        assert isinstance(m2, dict)
        cell = m2["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        del cell["control_diagnostic"]
        with pytest.raises(ValueError, match="cell keys"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m2_delta_inconsistency_raises(self) -> None:
        metrics = self._valid_metrics()
        m2 = metrics[METRIC_KEYS[1]]
        assert isinstance(m2, dict)
        cell = m2["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        concept = RELATED_CONCEPTS[0]
        cell["related"][concept]["delta_cos"] = 0.99
        with pytest.raises(ValueError, match="delta_cos"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- M3 field structure ----------------------------------------------

    def test_m3_negative_norm_raises(self) -> None:
        metrics = self._valid_metrics()
        m3 = metrics[METRIC_KEYS[2]]
        assert isinstance(m3, dict)
        cell = m3["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        cell["norm_current"] = -0.5
        with pytest.raises(ValueError, match="out of range"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m3_delta_inconsistency_raises(self) -> None:
        metrics = self._valid_metrics()
        m3 = metrics[METRIC_KEYS[2]]
        assert isinstance(m3, dict)
        cell = m3["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        cell["delta"] = 99.0
        with pytest.raises(ValueError, match="delta"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- M4 field structure ----------------------------------------------

    def test_m4_wrong_method_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        m4["method"] = "ridge"
        with pytest.raises(ValueError, match="method"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_wrong_label_count_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        m4["label_names"] = list(PROBE_CLASSES)[:4]
        with pytest.raises(ValueError, match="label_names"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_balanced_accuracy_out_of_range_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        cell["by_label"][PROBE_CLASSES[0]]["balanced_accuracy"] = 1.5
        with pytest.raises(ValueError, match="balanced_accuracy"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_auroc_out_of_range_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        cell["by_label"][PROBE_CLASSES[0]]["auroc"] = -0.1
        with pytest.raises(ValueError, match="auroc"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_count_exceeds_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        rec = cell["by_label"][PROBE_CLASSES[0]]
        n_samples = cell["n_samples"]
        assert isinstance(n_samples, int)
        rec["n_positive"] = n_samples + 10
        with pytest.raises(ValueError, match="exceeds n_samples"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_fold_count_inconsistent_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        rec = cell["by_label"][PROBE_CLASSES[0]]
        rec["n_folds_skipped"] = rec["n_folds_used"]
        with pytest.raises(ValueError, match="n_folds"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_bool_n_positive_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][TEST_CHECKPOINTS[0]]
        assert isinstance(cell, dict)
        cell["by_label"][PROBE_CLASSES[0]]["n_positive"] = True
        with pytest.raises(ValueError, match="non-boolean int"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    # --- M4 n_samples uniformity / consistency ---------------------------

    def test_m4_cell_bool_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][BASE_CHECKPOINT]
        assert isinstance(cell, dict)
        cell["n_samples"] = True
        with pytest.raises(ValueError, match="non-boolean int"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_cell_zero_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"][BASE_CHECKPOINT]
        assert isinstance(cell, dict)
        cell["n_samples"] = 0
        with pytest.raises(ValueError, match="must be positive"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_m4_cell_inconsistent_n_samples_raises(self) -> None:
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        base_meta = self._build_valid_metrics()["metadata"]
        assert isinstance(base_meta, dict)
        n_samples = base_meta["n_samples"]
        assert isinstance(n_samples, int)
        cell = m4["by_layer"][str(TEST_LAYERS[0])]["by_checkpoint"]["step_100"]
        assert isinstance(cell, dict)
        # Keep counts consistent with the new (smaller) n_samples so only the
        # cross-cell uniformity check fires.
        cell["n_samples"] = n_samples - 2
        cell["by_label"][PROBE_CLASSES[0]]["n_positive"] = max(
            0, cell["by_label"][PROBE_CLASSES[0]]["n_positive"] - 2
        )
        with pytest.raises(ValueError, match="n_samples differs across"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )

    def test_metadata_m4_n_samples_mismatch_raises(self) -> None:
        """Grid is internally uniform but disagrees with metadata.n_samples."""
        metrics = self._valid_metrics()
        m4 = metrics[METRIC_KEYS[3]]
        assert isinstance(m4, dict)
        base_meta = self._build_valid_metrics()["metadata"]
        assert isinstance(base_meta, dict)
        n_samples = base_meta["n_samples"]
        assert isinstance(n_samples, int)
        shifted = max(1, n_samples - 2)
        for layer in TEST_LAYERS:
            for ckpt in TEST_CHECKPOINTS:
                cell = m4["by_layer"][str(layer)]["by_checkpoint"][ckpt]
                assert isinstance(cell, dict)
                cell["n_samples"] = shifted
        with pytest.raises(ValueError, match="does not match M4 grid"):
            validate_metrics(
                metrics,
                expected_checkpoints=TEST_CHECKPOINTS,
                expected_layers=TEST_LAYERS,
            )


# =============================================================================
# 7. Atomic write
# =============================================================================


class TestWriteMetricsJson:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        path = str(tmp_path / "out" / "metrics.json")
        payload: dict[str, object] = {"schema": SCHEMA, "value": 3.14}
        result = write_metrics_json(path, payload)
        assert result == path
        assert os.path.exists(path)

    def test_no_tmp_residue(self, tmp_path: Path) -> None:
        path = str(tmp_path / "metrics.json")
        write_metrics_json(path, {"v": 1})
        assert not os.path.exists(path + ".tmp")

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = str(tmp_path / "metrics.json")
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "version": VERSION,
            "values": [1.0, 2.0],
        }
        write_metrics_json(path, payload)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_overwrite(self, tmp_path: Path) -> None:
        path = str(tmp_path / "metrics.json")
        write_metrics_json(path, {"v": 1})
        write_metrics_json(path, {"v": 2})
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["v"] == 2


# =============================================================================
# 8. checkpoint_model_map
# =============================================================================


class TestCheckpointModelMap:
    def test_base_maps_to_base_model(self) -> None:
        m = checkpoint_model_map()
        assert m[BASE_CHECKPOINT] == ("olmo3-base", "main")

    def test_rl_checkpoints_map_to_target_model(self) -> None:
        m = checkpoint_model_map()
        for ckpt in ("step_100", "step_2900"):
            assert m[ckpt] == ("olmo3-rl-zero-code", ckpt)

    def test_covers_all_experiment_checkpoints(self) -> None:
        from src.rl_zero_experiment import EXPERIMENT_CHECKPOINTS

        m = checkpoint_model_map()
        assert set(m.keys()) >= set(EXPERIMENT_CHECKPOINTS)


# =============================================================================
# 9. On-disk integration test
# =============================================================================


class TestOnDiskIntegration:
    """Full pipeline: write synthetic files -> compute_all_metrics -> verify."""

    @pytest.fixture
    def synthetic_roots(self, tmp_path: Path) -> tuple[str, str]:
        """Create synthetic concept-vector and activation files under tmp_path."""
        cv_root = str(tmp_path / "concept_vectors")
        act_root = str(tmp_path / "activations")
        ckpt_map = checkpoint_model_map()
        all_dirs = _build_concept_dirs_known()
        records = _build_synthetic_probe_records(n_per_class=10)

        # Write concept vectors: one file per (model, ckpt, layer), 6 concepts each.
        dirs_by_ckpt: dict[str, dict[str, np.ndarray]] = {}
        for ckpt in TEST_CHECKPOINTS:
            dirs_by_ckpt[ckpt] = all_dirs[ckpt][TEST_LAYERS[0]]
        _write_synthetic_concept_vectors(
            cv_root, ckpt_map, TEST_LAYERS, TEST_CHECKPOINTS, dirs_by_ckpt
        )

        # Write activations: one file per (model, ckpt, layer), 80 records each.
        _write_synthetic_activations(
            act_root, ckpt_map, TEST_LAYERS, TEST_CHECKPOINTS, records
        )
        return cv_root, act_root

    def test_compute_all_metrics_returns_valid_dict(
        self, synthetic_roots: tuple[str, str]
    ) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        # Schema / version.
        assert metrics["schema"] == SCHEMA
        assert metrics["version"] == VERSION
        # Exactly four metric keys.
        assert set(METRIC_KEYS) <= set(metrics.keys())
        for key in METRIC_KEYS:
            assert key in metrics

    def test_m1_values_match_known_cosines(
        self, synthetic_roots: tuple[str, str]
    ) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        m1 = metrics["m1_checkpoint_cosine"]
        assert isinstance(m1, dict)
        for layer in TEST_LAYERS:
            blk = m1["by_layer"][str(layer)]["by_checkpoint"]
            assert blk[BASE_CHECKPOINT]["cos_vs_main"] == pytest.approx(1.0)
            assert blk["step_100"]["cos_vs_main"] == pytest.approx(1.0 / math.sqrt(2))

    def test_m2_delta_cos_values(self, synthetic_roots: tuple[str, str]) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        m2 = metrics["m2_target_related_delta_cos"]
        assert isinstance(m2, dict)
        for layer in TEST_LAYERS:
            step = m2["by_layer"][str(layer)]["by_checkpoint"]["step_100"]
            cpp = step["related"]["code_python_vs_cpp"]
            assert cpp["delta_cos"] == pytest.approx(1.0 / math.sqrt(2))

    def test_m3_norm_values(self, synthetic_roots: tuple[str, str]) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        m3 = metrics["m3_target_raw_direction_magnitude_delta"]
        assert isinstance(m3, dict)
        for layer in TEST_LAYERS:
            blk = m3["by_layer"][str(layer)]["by_checkpoint"]
            # base target = e0 -> norm 1; step target = e0+e1 -> norm sqrt(2)
            assert blk[BASE_CHECKPOINT]["norm_current"] == pytest.approx(1.0)
            assert blk["step_100"]["norm_current"] == pytest.approx(math.sqrt(2))
            assert blk["step_100"]["delta"] == pytest.approx(math.sqrt(2) - 1.0)

    def test_m4_separable_high_accuracy(self, synthetic_roots: tuple[str, str]) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        m4 = metrics["m4_eight_class_grouped_logistic_separability"]
        assert isinstance(m4, dict)
        for layer in TEST_LAYERS:
            for ckpt in TEST_CHECKPOINTS:
                by_label = m4["by_layer"][str(layer)]["by_checkpoint"][ckpt]["by_label"]
                for name in PROBE_CLASSES:
                    assert by_label[name]["balanced_accuracy"] > 0.9, (
                        f"{ckpt}/{layer}/{name}"
                    )
                    assert by_label[name]["auroc"] > 0.9, f"{ckpt}/{layer}/{name}"

    def test_metadata_protocol_raw(self, synthetic_roots: tuple[str, str]) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["protocol"] == "raw"
        assert metadata["n_samples"] == 80  # 8 classes x 10

    def test_method_logistic_recorded(self, synthetic_roots: tuple[str, str]) -> None:
        """M4 block and metadata both report method == 'logistic'."""
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        m4 = metrics["m4_eight_class_grouped_logistic_separability"]
        assert isinstance(m4, dict)
        assert m4["method"] == "logistic"
        metadata = metrics["metadata"]
        assert isinstance(metadata, dict)
        linear_probe = metadata["linear_probe"]
        assert isinstance(linear_probe, dict)
        assert linear_probe["method"] == "logistic"

    def test_full_json_roundtrip(
        self, synthetic_roots: tuple[str, str], tmp_path: Path
    ) -> None:
        cv_root, act_root = synthetic_roots
        metrics = compute_all_metrics(
            concept_vectors_root=cv_root,
            activations_root=act_root,
            layers=TEST_LAYERS,
            checkpoints=TEST_CHECKPOINTS,
            n_folds=3,
        )
        path = str(tmp_path / "metrics.json")
        write_metrics_json(path, metrics)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        # Four metric keys survive the JSON round-trip.
        for key in METRIC_KEYS:
            assert key in loaded


# =============================================================================
# 9b. Metadata n_samples consistency (derived from M4 cells)
# =============================================================================


class TestResolveNSamples:
    """Lock the singular-metadata invariant for ``metadata.n_samples``."""

    @staticmethod
    def _m4_grid(
        layers: list[int],
        checkpoints: list[str],
        n_by_cell: dict[tuple[str, int], int],
    ) -> dict[str, object]:
        by_layer: dict[str, object] = {}
        for layer in layers:
            by_ckpt: dict[str, dict[str, int]] = {}
            for ckpt in checkpoints:
                by_ckpt[ckpt] = {"n_samples": n_by_cell[(ckpt, layer)]}
            by_layer[str(layer)] = {"by_checkpoint": by_ckpt}
        return {"by_layer": by_layer}

    def test_uniform_grid_returns_common_count(self) -> None:
        layers = [3, 6]
        ckpts = [BASE_CHECKPOINT, "step_100"]
        n_by_cell = {(ckpt, ly): 400 for ckpt in ckpts for ly in layers}
        m4 = self._m4_grid(layers, ckpts, n_by_cell)
        assert _resolve_n_samples(m4, layers, ckpts) == 400

    def test_disagreeing_cells_raises(self) -> None:
        layers = [3, 6]
        ckpts = [BASE_CHECKPOINT, "step_100"]
        n_by_cell = {(ckpt, ly): 400 for ckpt in ckpts for ly in layers}
        n_by_cell[("step_100", 6)] = 398
        m4 = self._m4_grid(layers, ckpts, n_by_cell)
        with pytest.raises(ValueError, match="n_samples differs across"):
            _resolve_n_samples(m4, layers, ckpts)

    def test_missing_by_layer_raises(self) -> None:
        with pytest.raises(ValueError, match="missing 'by_layer'"):
            _resolve_n_samples({}, [3], [BASE_CHECKPOINT])

    def test_bool_n_samples_rejected(self) -> None:
        """``True``/``False`` are int subclasses and must not pass as counts."""
        layers = [3]
        ckpts = [BASE_CHECKPOINT]
        n_by_cell: dict[tuple[str, int], object] = {(BASE_CHECKPOINT, 3): True}
        m4 = self._m4_grid_typed(layers, ckpts, n_by_cell)
        with pytest.raises(ValueError, match="non-boolean int"):
            _resolve_n_samples(m4, layers, ckpts)

    def test_zero_n_samples_rejected(self) -> None:
        layers = [3]
        ckpts = [BASE_CHECKPOINT]
        n_by_cell: dict[tuple[str, int], object] = {(BASE_CHECKPOINT, 3): 0}
        m4 = self._m4_grid_typed(layers, ckpts, n_by_cell)
        with pytest.raises(ValueError, match="must be positive"):
            _resolve_n_samples(m4, layers, ckpts)

    def test_negative_n_samples_rejected(self) -> None:
        layers = [3]
        ckpts = [BASE_CHECKPOINT]
        n_by_cell: dict[tuple[str, int], object] = {(BASE_CHECKPOINT, 3): -5}
        m4 = self._m4_grid_typed(layers, ckpts, n_by_cell)
        with pytest.raises(ValueError, match="must be positive"):
            _resolve_n_samples(m4, layers, ckpts)

    @staticmethod
    def _m4_grid_typed(
        layers: list[int],
        checkpoints: list[str],
        n_by_cell: dict[tuple[str, int], object],
    ) -> dict[str, object]:
        """Like ``_m4_grid`` but permits non-int ``n_samples`` values."""
        by_layer: dict[str, object] = {}
        for layer in layers:
            by_ckpt: dict[str, object] = {}
            for ckpt in checkpoints:
                by_ckpt[ckpt] = {"n_samples": n_by_cell[(ckpt, layer)]}
            by_layer[str(layer)] = {"by_checkpoint": by_ckpt}
        return {"by_layer": by_layer}


# =============================================================================
# 10. Protocol validation in the activation loader
# =============================================================================


class TestActivationLoaderProtocol:
    def test_rejects_chat_protocol(self, tmp_path: Path) -> None:
        act_root = str(tmp_path / "activations")
        records = _build_synthetic_probe_records(n_per_class=5)
        ckpt_map = checkpoint_model_map()
        _write_synthetic_activations(
            act_root, ckpt_map, [TEST_LAYERS[0]], [BASE_CHECKPOINT], records
        )
        # Corrupt the sidecar protocol.
        from src.probe_activations import _layer_base

        base = _layer_base(act_root, "olmo3-base", "main", TEST_LAYERS[0])
        with open(base + ".json", encoding="utf-8") as f:
            sidecar = json.load(f)
        sidecar["protocol"] = "chat"
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        loader = make_activation_loader(act_root, PROBE_CLASSES)
        with pytest.raises(ValueError, match="protocol"):
            loader(BASE_CHECKPOINT, TEST_LAYERS[0])

    def test_missing_protocol_key_raises(self, tmp_path: Path) -> None:
        """A sidecar with no ``protocol`` key must not default to raw."""
        act_root = str(tmp_path / "activations")
        records = _build_synthetic_probe_records(n_per_class=5)
        ckpt_map = checkpoint_model_map()
        _write_synthetic_activations(
            act_root, ckpt_map, [TEST_LAYERS[0]], [BASE_CHECKPOINT], records
        )
        from src.probe_activations import _layer_base

        base = _layer_base(act_root, "olmo3-base", "main", TEST_LAYERS[0])
        with open(base + ".json", encoding="utf-8") as f:
            sidecar = json.load(f)
        del sidecar["protocol"]
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        loader = make_activation_loader(act_root, PROBE_CLASSES)
        with pytest.raises(ValueError, match="missing 'protocol'"):
            loader(BASE_CHECKPOINT, TEST_LAYERS[0])

    def test_identity_mismatch_raises(self, tmp_path: Path) -> None:
        """A sidecar whose text provenance drifted from records.json is rejected."""
        act_root = str(tmp_path / "activations")
        records = _build_synthetic_probe_records(n_per_class=5)
        ckpt_map = checkpoint_model_map()
        _write_synthetic_activations(
            act_root, ckpt_map, [TEST_LAYERS[0]], [BASE_CHECKPOINT], records
        )
        from src.probe_activations import _layer_base

        base = _layer_base(act_root, "olmo3-base", "main", TEST_LAYERS[0])
        with open(base + ".json", encoding="utf-8") as f:
            sidecar = json.load(f)
        # Corrupt one per-record text hash so identity re-derivation fails.
        sidecar["text_sha256"][0] = "0" * 64
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f)

        loader = make_activation_loader(act_root, PROBE_CLASSES)
        with pytest.raises(ValueError, match="record identity"):
            loader(BASE_CHECKPOINT, TEST_LAYERS[0])

    def test_rank1_tensor_raises(self, tmp_path: Path) -> None:
        """A flattened (rank-1) activation tensor must be rejected, not reshaped."""
        from safetensors.torch import save_file

        from src.probe_activations import _ACTIVATIONS_KEY, _layer_base

        act_root = str(tmp_path / "activations")
        records = _build_synthetic_probe_records(n_per_class=5)
        ckpt_map = checkpoint_model_map()
        _write_synthetic_activations(
            act_root, ckpt_map, [TEST_LAYERS[0]], [BASE_CHECKPOINT], records
        )
        # Overwrite the safetensors with a rank-1 tensor (same row count) so the
        # sidecar identity still matches but the tensor shape is wrong.
        base = _layer_base(act_root, "olmo3-base", "main", TEST_LAYERS[0])
        flat = torch.zeros(len(records), dtype=torch.float32)
        save_file({_ACTIVATIONS_KEY: flat}, base + ".safetensors")

        loader = make_activation_loader(act_root, PROBE_CLASSES)
        with pytest.raises(ValueError, match="rank-2"):
            loader(BASE_CHECKPOINT, TEST_LAYERS[0])

    def test_missing_records_json_raises(self, tmp_path: Path) -> None:
        """Without ``records.json`` the loader cannot bind identity at all."""
        act_root = str(tmp_path / "activations")
        records = _build_synthetic_probe_records(n_per_class=5)
        ckpt_map = checkpoint_model_map()
        _write_synthetic_activations(
            act_root, ckpt_map, [TEST_LAYERS[0]], [BASE_CHECKPOINT], records
        )
        os.remove(os.path.join(act_root, "records.json"))
        with pytest.raises(FileNotFoundError, match="records.json"):
            make_activation_loader(act_root, PROBE_CLASSES)


# =============================================================================
# 10b. Concept raw-direction projection (non-1D rejection)
# =============================================================================


class TestConceptDirectionProjection:
    def test_non_1d_raw_direction_rejected(self) -> None:
        """A rank-2 raw_direction must be rejected before flattening."""
        cv = _build_synthetic_concept_vector(
            TARGET_CONCEPT, "olmo3-base", TEST_LAYERS[0], np.zeros((4, 8))
        )
        with pytest.raises(ValueError, match="must be 1-D"):
            _cv_raw_dir_to_np(cv)

    def test_1d_raw_direction_passes(self) -> None:
        cv = _build_synthetic_concept_vector(
            TARGET_CONCEPT, "olmo3-base", TEST_LAYERS[0], np.linspace(-1.0, 1.0, 8)
        )
        arr = _cv_raw_dir_to_np(cv)
        assert arr.ndim == 1
        assert arr.shape == (8,)


# =============================================================================
# 11. CLI
# =============================================================================


class TestCLI:
    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        assert args.output == DEFAULT_METRICS_PATH
        assert args.n_folds == 5
        assert args.alpha == 1.0
        assert args.seed == 42
        assert args.no_standardize is False
        assert args.no_step_100_ref is False

    def test_parse_args_custom(self) -> None:
        args = parse_args(
            [
                "--output",
                "/tmp/x.json",
                "--n-folds",
                "3",
                "--alpha",
                "0.5",
                "--seed",
                "7",
                "--no-standardize",
                "--no-step-100-ref",
            ]
        )
        assert args.output == "/tmp/x.json"
        assert args.n_folds == 3
        assert args.alpha == 0.5
        assert args.seed == 7
        assert args.no_standardize is True
        assert args.no_step_100_ref is True

    def test_main_writes_valid_file(self, tmp_path: Path) -> None:
        cv_root = str(tmp_path / "concept_vectors")
        act_root = str(tmp_path / "activations")
        out_path = str(tmp_path / "out" / "metrics.json")
        ckpt_map = checkpoint_model_map()
        all_dirs = _build_concept_dirs_known()
        records = _build_synthetic_probe_records(n_per_class=10)
        dirs_by_ckpt = {
            ckpt: all_dirs[ckpt][TEST_LAYERS[0]] for ckpt in TEST_CHECKPOINTS
        }
        _write_synthetic_concept_vectors(
            cv_root, ckpt_map, TEST_LAYERS, TEST_CHECKPOINTS, dirs_by_ckpt
        )
        _write_synthetic_activations(
            act_root, ckpt_map, TEST_LAYERS, TEST_CHECKPOINTS, records
        )

        # main() uses the full EXPERIMENT_CHECKPOINTS by default; override
        # via a monkeypatched compute_all_metrics is fragile, so instead
        # we call main with our mini grid by patching the defaults.
        import experiments.run_rl_zero_syntax_metrics as mod

        original_compute = mod.compute_all_metrics

        def patched_compute(**kwargs: Any) -> Any:
            kwargs.setdefault("layers", TEST_LAYERS)
            kwargs.setdefault("checkpoints", TEST_CHECKPOINTS)
            kwargs.setdefault("n_folds", 3)
            return original_compute(**kwargs)

        mod.compute_all_metrics = patched_compute
        try:
            rc = main(
                [
                    "--concept-vectors-root",
                    cv_root,
                    "--activations-root",
                    act_root,
                    "--output",
                    out_path,
                    "--n-folds",
                    "3",
                ]
            )
        finally:
            mod.compute_all_metrics = original_compute

        assert rc == 0
        assert os.path.exists(out_path)
        with open(out_path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["schema"] == SCHEMA
        assert loaded["version"] == VERSION
        for key in METRIC_KEYS:
            assert key in loaded


# =============================================================================
# 12. Structural constants
# =============================================================================


class TestStructuralConstants:
    def test_exactly_four_metric_keys(self) -> None:
        assert len(METRIC_KEYS) == 4
        assert len(set(METRIC_KEYS)) == 4

    def test_probe_classes_match_experiment(self) -> None:
        assert tuple(CLI_PROBE_CLASSES) == PROBE_CLASSES

    def test_schema_and_version(self) -> None:
        assert SCHEMA == "rl_zero_code_syntax_metrics"
        assert VERSION == 2

    def test_default_concept_vectors_root_isolated(self) -> None:
        root = default_concept_vectors_root()
        assert "rl_zero_code_syntax" in root
        assert "concept_dynamics_multi" not in root
        assert root.endswith("concept_vectors")
