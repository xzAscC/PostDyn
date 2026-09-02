#!/usr/bin/env python3
"""Four-metric computation driver for the RL-Zero-Code Python syntax concept.

Reads isolated **raw** concept vectors (DiM directions) and 400 x d **raw**
probe activations already extracted under the experiment's isolated results
root, and computes exactly four representation/readout metrics per
(checkpoint, layer):

  M1 -- checkpoint cosine (directional stability vs base ``main``, with an
        optional ``step_100`` second reference).
  M2 -- target-vs-related Delta-cos (``cos(target, related)`` now minus the
        same cosine at base) for the four related code-language concepts,
        plus a ``she -> he`` control cosine as a **diagnostic only** (never a
        fifth metric).
  M3 -- target raw-direction magnitude Delta (``||r||`` now minus ``||r||``
        at base, using the un-normalized DiM mean-difference).
  M4 -- eight-class one-vs-rest grouped logistic separability (balanced
        accuracy and AUROC per class) from the stored 400 x d probe
        activations.

**No model is loaded or run**; this driver is pure post-extraction analysis.

The single output is one atomic ``metrics.json`` carrying ``schema``,
``version``, ``metadata``, and exactly four top-level metric keys.  Finite
values, full checkpoint/layer coverage, and the ``raw`` protocol are
validated before the file is written.

Usage::

    uv run python scripts/run_rl_zero_syntax_metrics.py [OPTIONS]

Options:
    --output PATH                 Output metrics.json path
    --concept-vectors-root DIR    Override concept-vectors directory
    --activations-root DIR        Override probe-activations directory
    --n-folds N                   Grouped CV folds (default 5)
    --alpha FLOAT                 Logistic L2 penalty on weights (default 1.0)
    --seed INT                    Grouped-fold seed (default 42)
    --no-standardize              Disable per-fold train standardization
    --no-step-100-ref             Disable step_100 as an M1 optional reference

This script never changes extraction, data, or config; it only reads
on-disk artifacts and writes ``--output`` (atomically).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch

# Make ``src`` importable when run directly via ``python scripts/...``.

from postdyn.concept_dynamics import (  # noqa: E402
    EXPECTED_D_MODEL,
    ConceptVector,
    load_concept_sidecar,
    load_concept_vectors,
    validate_concept_sidecar,
)
from postdyn.linear_probe import LinearProbeResult, linear_probe_score  # noqa: E402
from postdyn.probe_activations import (  # noqa: E402
    PROTOCOL,
    default_activations_root,
    load_layer_activations,
    load_records_json,
    validate_sidecar_record_identity,
)
from postdyn.rl_zero_experiment import (  # noqa: E402
    BASE_CHECKPOINT,
    BASE_MODEL,
    BASE_MODEL_KEY,
    CONTROL_CONCEPT,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_CONCEPTS,
    EXPERIMENT_LAYERS,
    PROBE_CLASSES,
    RELATED_CONCEPTS,
    RL_CHECKPOINTS,
    RL_ZERO_CODE_RESULTS_ROOT,
    TARGET_CONCEPT,
    TARGET_MODEL,
    TARGET_MODEL_KEY,
    results_root,
)

# =============================================================================
# Schema constants
# =============================================================================

SCHEMA: str = "rl_zero_code_syntax_metrics"
VERSION: int = 2

#: The exact four top-level metric keys (no fifth metric).
METRIC_KEYS: tuple[str, ...] = (
    "m1_checkpoint_cosine",
    "m2_target_related_delta_cos",
    "m3_target_raw_direction_magnitude_delta",
    "m4_eight_class_grouped_logistic_separability",
)

#: Optional M1 reference checkpoints in addition to the base ``main``.
DEFAULT_OPTIONAL_REFERENCES: tuple[str, ...] = ("step_100",)

#: Source model provenance: each experiment model key pinned to its Hugging
#: Face repository id. Recorded in ``metadata.source_models`` and validated
#: exactly so a metrics file cannot silently describe a different model pair.
SOURCE_MODELS: dict[str, str] = {
    BASE_MODEL_KEY: BASE_MODEL.hf_id,
    TARGET_MODEL_KEY: TARGET_MODEL.hf_id,
}

#: Pinned HumanEval-X dataset revision (git SHA of ``zai-org/humaneval-x``)
#: from which the four non-python code probe classes are materialized. Mirrors
#: the pin in ``scripts/download_datasets.py``; recorded in metadata and
#: validated exactly so regenerated data cannot silently shift the probe texts.
HUMANEVAL_X_REVISION: str = "62c78627f3072a1454fa0cb0184737cafe5e4198"

#: Default linear-probe parameters (matching ``postdyn.linear_probe`` defaults).
DEFAULT_N_FOLDS: int = 5
DEFAULT_ALPHA: float = 1.0
DEFAULT_SEED: int = 42
DEFAULT_STANDARDIZE: bool = True

#: Subdirectory names under the isolated results root.
CONCEPT_VECTORS_SUBDIR: str = "concept_vectors"
METRICS_SUBDIR: str = "metrics"
METRICS_FILENAME: str = "metrics.json"

#: Default metrics output path.
DEFAULT_METRICS_PATH: str = os.path.join(
    RL_ZERO_CODE_RESULTS_ROOT, METRICS_SUBDIR, METRICS_FILENAME
)


# =============================================================================
# Path & checkpoint helpers
# =============================================================================


def default_concept_vectors_root(
    *, quick: bool = False, override: str | None = None
) -> str:
    """Default concept-vectors root: ``{results_root}/concept_vectors``."""
    return os.path.join(
        results_root(quick=quick, override=override), CONCEPT_VECTORS_SUBDIR
    )


def checkpoint_model_map() -> dict[str, tuple[str, str]]:
    """Map experiment checkpoint name -> ``(model_name, checkpoint_name)``.

    The base checkpoint ``main`` lives under the ``olmo3-base`` model; every
    RL step lives under ``olmo3-rl-zero-code``.  This mapping mirrors the
    directory layout used by ``probe_activations`` and ``concept_dynamics``
    persistence so the same ``(model_name, checkpoint)`` pair addresses both
    concept-vector and activation files.
    """
    mapping: dict[str, tuple[str, str]] = {
        BASE_CHECKPOINT: (BASE_MODEL_KEY, BASE_CHECKPOINT)
    }
    for ckpt in RL_CHECKPOINTS:
        mapping[ckpt] = (TARGET_MODEL_KEY, ckpt)
    return mapping


# =============================================================================
# Type aliases
# =============================================================================

#: ``{checkpoint: {layer: {concept_name: raw_direction_1d_np}}}``
ConceptDirs = dict[str, dict[int, dict[str, np.ndarray]]]

#: ``{checkpoint: {layer: target_raw_direction_1d_np}}``
TargetDirs = dict[str, dict[int, np.ndarray]]

#: Activation loader: ``(checkpoint, layer) -> (features, labels, groups)``.
ActivationLoader = Callable[[str, int], "tuple[np.ndarray, np.ndarray, list[str]]"]


# =============================================================================
# Numerical helpers (pure, no I/O)
# =============================================================================


def safe_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity with a zero-norm guard.

    Returns ``0.0`` when either vector has a norm below ``eps`` (this covers
    the degenerate zero-direction case where the raw DiM direction vanishes).
    The result is clamped to ``[-1, 1]`` to absorb floating-point drift.
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return max(-1.0, min(1.0, cos))


def _assert_finite(value: float, context: str) -> None:
    """Raise ``ValueError`` if ``value`` is not finite."""
    if not math.isfinite(value):
        raise ValueError(f"non-finite value at {context}: {value}")


def _assert_all_finite(obj: object, context: str) -> None:
    """Recursively assert every ``float`` in a nested structure is finite."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite(v, f"{context}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_finite(v, f"{context}[{i}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite value at {context}: {obj}")
    # int, str, bool, None: integers and the rest are always "finite".


# =============================================================================
# M1 -- checkpoint cosine (directional stability)
# =============================================================================


def metric_m1_checkpoint_cosine(
    target_dirs: TargetDirs,
    layers: Sequence[int],
    checkpoints: Sequence[str],
    base_checkpoint: str,
    optional_references: Sequence[str],
) -> dict[str, Any]:
    """M1: cosine of the target raw direction vs each reference checkpoint.

    For every ``(checkpoint, layer)`` the cosine between the target concept's
    raw DiM direction and the same direction at each reference checkpoint is
    computed.  The base ``main`` checkpoint is always a reference; the
    optional references (default: ``step_100``) are appended when present in
    ``checkpoints``.

    Args:
        target_dirs: ``{checkpoint: {layer: raw_direction}}`` for the target.
        layers: Layer indices to cover.
        checkpoints: Checkpoint names to cover.
        base_checkpoint: Primary reference (always ``main``).
        optional_references: Additional references (e.g. ``step_100``).

    Returns:
        M1 metric block (see module docstring for schema).

    Raises:
        ValueError: If a reference is missing from ``target_dirs`` or a
            computed cosine is non-finite.
    """
    references: list[str] = [base_checkpoint]
    for ref in optional_references:
        if ref in checkpoints and ref not in references:
            references.append(ref)

    for ref in references:
        if ref not in target_dirs:
            raise ValueError(f"M1 reference {ref!r} missing from target_dirs")

    by_layer: dict[str, object] = {}
    for layer in layers:
        by_ckpt: dict[str, dict[str, float]] = {}
        for ckpt in checkpoints:
            current = target_dirs[ckpt][layer]
            entry: dict[str, float] = {}
            for ref in references:
                ref_vec = target_dirs[ref][layer]
                cos = safe_cosine(current, ref_vec)
                _assert_finite(cos, f"M1/{ckpt}/layer_{layer}/cos_vs_{ref}")
                entry[f"cos_vs_{ref}"] = cos
            by_ckpt[ckpt] = entry
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}

    return {
        "description": (
            "Cosine similarity of the target raw DiM direction vs each "
            "reference checkpoint. High intra-trajectory cosine means the "
            "syntactic-validity direction is stable across RL."
        ),
        "target_concept": TARGET_CONCEPT,
        "references": list(references),
        "by_layer": by_layer,
    }


# =============================================================================
# M2 -- target-vs-related Delta-cos (+ she->he diagnostic)
# =============================================================================


def metric_m2_target_related_delta_cos(
    all_dirs: ConceptDirs,
    layers: Sequence[int],
    checkpoints: Sequence[str],
    base_checkpoint: str,
    target_concept: str,
    related_concepts: Sequence[str],
    control_concept: str,
) -> dict[str, Any]:
    """M2: ``Delta-cos(target, related)`` vs base, plus a control diagnostic.

    For each related concept ``c`` and each ``(checkpoint, layer)``::

        delta_cos = cos(target^t, c^t) - cos(target^base, c^base)

    The ``she -> he`` control cosine is reported under
    ``control_diagnostic`` (current cosine, base cosine, delta) but is **not**
    a fifth metric -- it only signals whether a generic post-training drift
    has occurred.

    Args:
        all_dirs: ``{checkpoint: {layer: {concept: raw_direction}}}``.
        layers, checkpoints: Coverage grid.
        base_checkpoint: Reference checkpoint for the delta.
        target_concept: Target concept key.
        related_concepts: The four related code-language concept keys.
        control_concept: The ``gender_she_vs_he`` control key.

    Returns:
        M2 metric block.
    """
    by_layer: dict[str, object] = {}
    for layer in layers:
        by_ckpt: dict[str, object] = {}
        for ckpt in checkpoints:
            target_cur = all_dirs[ckpt][layer][target_concept]
            target_base = all_dirs[base_checkpoint][layer][target_concept]

            related: dict[str, dict[str, float]] = {}
            for concept in related_concepts:
                cos_cur = safe_cosine(target_cur, all_dirs[ckpt][layer][concept])
                cos_base = safe_cosine(
                    target_base, all_dirs[base_checkpoint][layer][concept]
                )
                delta = cos_cur - cos_base
                _assert_finite(
                    cos_cur, f"M2/{ckpt}/layer_{layer}/{concept}/cos_current"
                )
                _assert_finite(cos_base, f"M2/{ckpt}/layer_{layer}/{concept}/cos_base")
                _assert_finite(delta, f"M2/{ckpt}/layer_{layer}/{concept}/delta_cos")
                related[concept] = {
                    "cos_current": cos_cur,
                    "cos_base": cos_base,
                    "delta_cos": delta,
                }

            # Control diagnostic (NOT a fifth metric).
            ctrl_cur = safe_cosine(target_cur, all_dirs[ckpt][layer][control_concept])
            ctrl_base = safe_cosine(
                target_base, all_dirs[base_checkpoint][layer][control_concept]
            )
            ctrl_delta = ctrl_cur - ctrl_base
            _assert_finite(ctrl_cur, f"M2/{ckpt}/layer_{layer}/control/cos_current")
            _assert_finite(ctrl_base, f"M2/{ckpt}/layer_{layer}/control/cos_base")
            _assert_finite(ctrl_delta, f"M2/{ckpt}/layer_{layer}/control/delta_cos")
            control_diagnostic: dict[str, float] = {
                "cos_current": ctrl_cur,
                "cos_base": ctrl_base,
                "delta_cos": ctrl_delta,
            }

            by_ckpt[ckpt] = {
                "related": related,
                "control_diagnostic": control_diagnostic,
            }
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}

    return {
        "description": (
            "Change in cosine between the target direction and each related "
            "code-language direction, relative to base. The she->he control "
            "cosine is a diagnostic only (not a fifth metric)."
        ),
        "target_concept": target_concept,
        "related_concepts": list(related_concepts),
        "control_concept_diagnostic": control_concept,
        "base_checkpoint": base_checkpoint,
        "by_layer": by_layer,
    }


# =============================================================================
# M3 -- target raw-direction magnitude Delta
# =============================================================================


def metric_m3_raw_direction_magnitude_delta(
    target_dirs: TargetDirs,
    layers: Sequence[int],
    checkpoints: Sequence[str],
    base_checkpoint: str,
    target_concept: str,
) -> dict[str, Any]:
    """M3: ``||r_target^t|| - ||r_target^base||`` (un-normalized DiM norm).

    Uses the **raw** DiM direction (``positive_mean - negative_mean``), not
    the unit-normalized steering vector, so the magnitude change reflects
    how strongly the model separates valid from invalid Python independent
    of orientation.

    Args:
        target_dirs: ``{checkpoint: {layer: raw_direction}}`` for the target.
        layers, checkpoints: Coverage grid.
        base_checkpoint: Reference checkpoint.
        target_concept: Target concept key (for metadata).

    Returns:
        M3 metric block.
    """
    by_layer: dict[str, object] = {}
    for layer in layers:
        base_norm = float(np.linalg.norm(target_dirs[base_checkpoint][layer]))
        _assert_finite(base_norm, f"M3/base/layer_{layer}/norm_base")
        by_ckpt: dict[str, dict[str, float]] = {}
        for ckpt in checkpoints:
            norm_cur = float(np.linalg.norm(target_dirs[ckpt][layer]))
            _assert_finite(norm_cur, f"M3/{ckpt}/layer_{layer}/norm_current")
            by_ckpt[ckpt] = {
                "norm_current": norm_cur,
                "norm_base": base_norm,
                "delta": norm_cur - base_norm,
            }
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}

    return {
        "description": (
            "Change in the L2 norm of the target raw DiM direction relative "
            "to base. A growing norm means the model separates valid from "
            "invalid Python more strongly, independent of orientation."
        ),
        "target_concept": target_concept,
        "base_checkpoint": base_checkpoint,
        "by_layer": by_layer,
    }


# =============================================================================
# M4 -- eight-class one-vs-rest grouped logistic separability
# =============================================================================


def metric_m4_eight_class_grouped_logistic_separability(
    load_activations: ActivationLoader,
    layers: Sequence[int],
    checkpoints: Sequence[str],
    label_names: Sequence[str],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    standardize: bool = DEFAULT_STANDARDIZE,
) -> dict[str, Any]:
    """M4: eight-class one-vs-rest grouped logistic probe separability.

    For each ``(checkpoint, layer)`` the stored ``400 x d`` raw-text probe
    activations are loaded together with their labels and group IDs, and a
    leakage-safe grouped one-vs-rest logistic probe is run via
    :func:`postdyn.linear_probe.linear_probe_score`.  Balanced accuracy and
    AUROC are reported for all eight classes.

    Args:
        load_activations: ``(checkpoint, layer) -> (features, labels, groups)``.
        layers, checkpoints: Coverage grid.
        label_names: The eight canonical class names.
        n_folds, alpha, seed, standardize: Logistic-probe hyper-parameters.

    Returns:
        M4 metric block. The block records ``method`` (mirroring
        :attr:`LinearProbeResult.method`, always ``"logistic"``).
    """
    by_layer: dict[str, object] = {}
    method: str = "logistic"
    for layer in layers:
        by_ckpt: dict[str, object] = {}
        for ckpt in checkpoints:
            features, labels, groups = load_activations(ckpt, layer)
            result: LinearProbeResult = linear_probe_score(
                features,
                labels,
                groups,
                label_names=tuple(label_names),
                n_folds=n_folds,
                alpha=alpha,
                seed=seed,
                standardize=standardize,
            )
            # ``method`` is constant across cells (always "logistic"); capture
            # it from each result so the output reflects the probe's own
            # self-identification rather than a hardcoded string.
            method = result.method
            by_label: dict[str, dict[str, float | int]] = {}
            for name in result.label_names:
                r = result.by_label[name]
                _assert_finite(
                    r.balanced_accuracy,
                    f"M4/{ckpt}/layer_{layer}/{name}/balanced_accuracy",
                )
                _assert_finite(r.auroc, f"M4/{ckpt}/layer_{layer}/{name}/auroc")
                by_label[name] = {
                    "balanced_accuracy": r.balanced_accuracy,
                    "auroc": r.auroc,
                    "n_positive": r.n_positive,
                    "n_negative": r.n_negative,
                    "n_folds_used": r.n_folds_used,
                    "n_folds_skipped": r.n_folds_skipped,
                }
            by_ckpt[ckpt] = {
                "by_label": by_label,
                "n_samples": result.n_samples,
                "n_groups": result.n_groups,
            }
        by_layer[str(layer)] = {"by_checkpoint": by_ckpt}

    return {
        "description": (
            "One-vs-rest logistic linear-probe separability of all eight "
            "probe classes under leakage-safe task/template-grouped K-fold "
            "CV. Balanced accuracy and AUROC are pooled across "
            "non-degenerate folds."
        ),
        "method": method,
        "label_names": list(label_names),
        "linear_probe_params": {
            "n_folds": n_folds,
            "alpha": alpha,
            "seed": seed,
            "standardize": standardize,
        },
        "by_layer": by_layer,
    }


# =============================================================================
# Disk loaders
# =============================================================================


def _cv_raw_dir_to_np(cv: ConceptVector) -> np.ndarray:
    """Extract the raw DiM direction as a 1-D ``float64`` numpy array.

    Raises:
        ValueError: If ``raw_direction`` is not rank-1. A non-1D direction would
            otherwise be silently flattened by ``ravel()``, hiding a corrupt or
            mis-extracted concept vector.
    """
    if cv.raw_direction.dim() != 1:
        raise ValueError(
            f"concept {cv.concept_name!r} raw_direction must be 1-D, got "
            f"{cv.raw_direction.dim()}D shape {tuple(cv.raw_direction.shape)}"
        )
    return cv.raw_direction.detach().cpu().to(torch.float64).numpy()


def load_concept_directions_from_disk(
    concept_vectors_root: str,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
    *,
    expected_concept_sources: dict[str, tuple[list[str], list[str]]] | None = None,
    expected_d_model: int | None = None,
    expected_max_seq_len: int | None = 2048,
    expected_protocol: str = "raw",
    expected_use_chat_template: bool = False,
) -> dict[str, np.ndarray]:
    """Load all concept ``raw_direction`` vectors for one triple.

    Requires a provenance-bound v1 sidecar. Legacy v0 files are rejected so
    callers must migrate or re-extract before metrics can be computed.
    """
    sidecar = load_concept_sidecar(
        concept_vectors_root, model_name, layer_idx, checkpoint
    )
    if not validate_concept_sidecar(
        sidecar,
        expected_model_name=model_name,
        expected_checkpoint=checkpoint,
        expected_layer_idx=layer_idx,
        expected_d_model=expected_d_model,
        expected_max_seq_len=expected_max_seq_len,
        expected_protocol=expected_protocol,
        expected_use_chat_template=expected_use_chat_template,
        expected_concept_sources=expected_concept_sources,
    ):
        raise ValueError(
            f"concept vectors at {model_name}/{checkpoint}/layer_{layer_idx} "
            "failed v1 provenance validation; run "
            "scripts/migrate_concept_sidecars.py or re-extract"
        )
    vectors = load_concept_vectors(
        concept_vectors_root, model_name, layer_idx, checkpoint
    )
    return {name: _cv_raw_dir_to_np(cv) for name, cv in vectors.items()}


def load_all_concept_directions(
    concept_vectors_root: str,
    layers: Sequence[int],
    checkpoints: Sequence[str],
    concepts: Sequence[str],
    *,
    expected_concept_sources: dict[str, tuple[list[str], list[str]]] | None = None,
) -> ConceptDirs:
    """Load every concept raw direction for every ``(checkpoint, layer)``.

    Validates that every required concept is present and that no vector
    contains NaN/Inf. When ``expected_concept_sources`` is omitted, the
    canonical contrastive loaders are used so metrics refuse unmigrated v0
    sidecars.
    """
    if expected_concept_sources is None:
        from postdyn.contrastive_datasets import load_contrastive_texts

        expected_concept_sources = {
            name: load_contrastive_texts(name, 50) for name in concepts
        }
    ckpt_map = checkpoint_model_map()
    result: ConceptDirs = {}
    for ckpt in checkpoints:
        model_name, ckpt_name = ckpt_map[ckpt]
        result[ckpt] = {}
        for layer in layers:
            dirs = load_concept_directions_from_disk(
                concept_vectors_root,
                model_name,
                ckpt_name,
                layer,
                expected_concept_sources=expected_concept_sources,
            )
            missing = set(concepts) - set(dirs)
            if missing:
                raise ValueError(
                    f"concept vectors at {ckpt}/{layer} missing concepts: "
                    f"{sorted(missing)}"
                )
            for name, vec in dirs.items():
                if vec.size == 0:
                    raise ValueError(
                        f"concept vector {name!r} at {ckpt}/{layer} is empty"
                    )
                if not np.isfinite(vec).all():
                    raise ValueError(
                        f"concept vector {name!r} at {ckpt}/{layer} contains NaN/Inf"
                    )
            result[ckpt][layer] = dirs
    return result


def make_activation_loader(
    activations_root: str,
    label_names: Sequence[str],
) -> ActivationLoader:
    """Create a closure that loads ``(features, labels, groups)`` from disk.

    The returned callable takes ``(checkpoint, layer)`` and reads the
    corresponding safetensors + JSON sidecar via
    :func:`postdyn.probe_activations.load_layer_activations`.     The global
    ``records.json`` is loaded **once** (as typed ProbeRecord objects) and
    every layer's sidecar is bound to it via
    :func:`validate_sidecar_record_identity`, so a stored activation is only
    accepted when its ordered text/source provenance matches the records the
    caller intends to score. The sidecar ``protocol`` key must be **present**
    and equal ``"raw"`` (no silent default), and the activation tensor must be
    exactly rank-2 with ``rows == n_records`` and ``cols == sidecar.d_model``.
    """
    ckpt_map = checkpoint_model_map()
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    records = load_records_json(activations_root)
    n_records = len(records)

    def load(checkpoint: str, layer: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        model_name, ckpt_name = ckpt_map[checkpoint]
        tensor, sidecar = load_layer_activations(
            activations_root, model_name, ckpt_name, layer
        )
        if sidecar.get("model_name") != model_name:
            raise ValueError(
                f"activations {checkpoint}/{layer} model_name="
                f"{sidecar.get('model_name')!r}, expected {model_name!r}"
            )
        if sidecar.get("checkpoint") != ckpt_name:
            raise ValueError(
                f"activations {checkpoint}/{layer} checkpoint="
                f"{sidecar.get('checkpoint')!r}, expected {ckpt_name!r}"
            )
        if int(sidecar.get("layer_idx", -1)) != int(layer):
            raise ValueError(
                f"activations {checkpoint}/{layer} layer_idx="
                f"{sidecar.get('layer_idx')!r}, expected {layer}"
            )
        # Require the protocol key to be present (no default): a missing key
        # means the sidecar predates the strict protocol contract and must be
        # re-extracted rather than silently treated as raw.
        if "protocol" not in sidecar:
            raise ValueError(
                f"activations {checkpoint}/{layer} sidecar missing 'protocol' key"
            )
        protocol = sidecar["protocol"]
        if protocol != PROTOCOL:
            raise ValueError(
                f"activations {checkpoint}/{layer} protocol={protocol!r}, "
                f"expected {PROTOCOL!r}"
            )
        # Strict text/source identity: the stored activation rows must correspond
        # to the exact records (by sample_id/label/group/source/text hash) the
        # caller intends to score. Old sidecars without this provenance fail here
        # and force a re-extraction.
        if not validate_sidecar_record_identity(sidecar, records):
            raise ValueError(
                f"activations {checkpoint}/{layer} sidecar record identity does "
                f"not match records.json; re-extraction required"
            )
        if tensor.dim() != 2:
            raise ValueError(
                f"activations {checkpoint}/{layer} must be rank-2, got "
                f"{tensor.dim()}D shape {tuple(tensor.shape)}"
            )
        if tensor.shape[0] != n_records:
            raise ValueError(
                f"activations {checkpoint}/{layer} rows ({tensor.shape[0]}) != "
                f"records ({n_records})"
            )
        sidecar_d_model = sidecar.get("d_model")
        if sidecar_d_model is None:
            raise ValueError(
                f"activations {checkpoint}/{layer} sidecar missing d_model"
            )
        if int(sidecar_d_model) != tensor.shape[1]:
            raise ValueError(
                f"activations {checkpoint}/{layer} d_model mismatch: tensor "
                f"{tensor.shape[1]} vs sidecar {int(sidecar_d_model)}"
            )
        features = tensor.detach().cpu().to(torch.float32).numpy().astype(np.float64)
        labels_str = sidecar["labels"]
        groups = sidecar["group_ids"]
        if len(labels_str) != features.shape[0]:
            raise ValueError(
                f"activations {checkpoint}/{layer} labels length "
                f"{len(labels_str)} != features rows {features.shape[0]}"
            )
        labels = np.array([label_to_idx[lbl] for lbl in labels_str], dtype=np.int64)
        return features, labels, list(groups)

    return load


# =============================================================================
# Validation
# =============================================================================


def _validate_m4_n_samples(
    metrics: dict[str, object],
    expected_metric_keys: Sequence[str],
    expected_layers: Sequence[int],
    expected_ckpt_set: set[str],
    n_samples_meta: int,
) -> None:
    """Enforce positive, non-boolean, uniform ``n_samples`` across the M4 grid.

    Walks every ``(layer, checkpoint)`` cell of the M4 block (the last entry of
    ``expected_metric_keys``) and requires each cell's ``n_samples`` to be a
    positive, non-boolean int; all cells must agree; and the common value must
    equal ``n_samples_meta`` (the metadata-level count). Structural problems
    (missing block / ``by_layer``) are silently skipped here because the main
    coverage loop in :func:`validate_metrics` already reports them.

    Raises:
        ValueError: On a non-int / boolean / non-positive cell count, on
            disagreeing cell counts, or on a metadata-vs-grid mismatch.
    """
    if not expected_metric_keys:
        return
    m4_key = expected_metric_keys[-1]
    m4_block = metrics.get(m4_key)
    if not isinstance(m4_block, dict):
        return
    by_layer = m4_block.get("by_layer")
    if not isinstance(by_layer, dict):
        return
    grid_counts: set[int] = set()
    for layer in expected_layers:
        layer_block = by_layer.get(str(layer))
        if not isinstance(layer_block, dict):
            continue
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            continue
        for ckpt in expected_ckpt_set:
            cell = by_ckpt.get(ckpt)
            if not isinstance(cell, dict):
                continue
            n = cell.get("n_samples")
            # bool is an int subclass; reject it so True/False can't pose as
            # a valid sample count.
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError(
                    f"M4 {ckpt}/layer_{layer} 'n_samples' must be a "
                    f"non-boolean int, got {n!r}"
                )
            if n <= 0:
                raise ValueError(
                    f"M4 {ckpt}/layer_{layer} 'n_samples' must be positive, got {n}"
                )
            grid_counts.add(n)
    if len(grid_counts) > 1:
        raise ValueError(
            f"M4 n_samples differs across (checkpoint, layer) grid: "
            f"{sorted(grid_counts)}"
        )
    if len(grid_counts) == 1:
        grid_n = next(iter(grid_counts))
        if grid_n != n_samples_meta:
            raise ValueError(
                f"metadata n_samples ({n_samples_meta}) does not match M4 grid "
                f"n_samples ({grid_n})"
            )


def _require_number(value: object, lo: float, hi: float, context: str) -> float:
    """Require ``value`` to be a finite number in ``[lo, hi]``; return as float.

    ``bool`` is rejected even though it subclasses int. ``lo``/``hi`` may be
    ``-math.inf``/``math.inf`` for one-sided bounds; non-finite values are always
    rejected regardless of bounds.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{context} must be a number, got {type(value).__name__}: {value!r}"
        )
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{context} is non-finite: {f}")
    if f < lo or f > hi:
        raise ValueError(f"{context} out of range [{lo}, {hi}]: {f}")
    return f


def _require_int(value: object, context: str, *, minimum: int | None = None) -> int:
    """Require ``value`` to be a non-boolean int, optionally ``>= minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{context} must be a non-boolean int, got {type(value).__name__}: {value!r}"
        )
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be >= {minimum}, got {value}")
    return value


def _validate_m1_fields(block: dict[str, object], base_checkpoint: str) -> None:
    """M1: exact reference keys + cosine values clamped to ``[-1, 1]``."""
    references = block.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("M1 missing non-empty 'references' list")
    if references[0] != base_checkpoint:
        raise ValueError(
            f"M1 first reference must be {base_checkpoint!r}, got {references[0]!r}"
        )
    expected_cell_keys = {f"cos_vs_{ref}" for ref in references}
    by_layer = block["by_layer"]
    assert isinstance(by_layer, dict)
    for layer_key, layer_block in by_layer.items():
        assert isinstance(layer_block, dict)
        by_ckpt = layer_block["by_checkpoint"]
        assert isinstance(by_ckpt, dict)
        for ckpt, cell in by_ckpt.items():
            assert isinstance(cell, dict)
            if set(cell.keys()) != expected_cell_keys:
                raise ValueError(
                    f"M1 {ckpt}/layer_{layer_key} keys {sorted(cell.keys())} != "
                    f"{sorted(expected_cell_keys)}"
                )
            for k, v in cell.items():
                _require_number(v, -1.0, 1.0, f"M1 {ckpt}/layer_{layer_key}/{k}")


def _validate_m2_fields(block: dict[str, object]) -> None:
    """M2: exactly four related concepts + one control, finite cosines/deltas."""
    related = block.get("related_concepts")
    if not isinstance(related, list) or set(related) != set(RELATED_CONCEPTS):
        raise ValueError(
            f"M2 related_concepts must be exactly {list(RELATED_CONCEPTS)}, got {related}"
        )
    if block.get("control_concept_diagnostic") != CONTROL_CONCEPT:
        raise ValueError(
            f"M2 control_concept_diagnostic must be {CONTROL_CONCEPT!r}, got "
            f"{block.get('control_concept_diagnostic')!r}"
        )
    entry_keys = {"cos_current", "cos_base", "delta_cos"}
    by_layer = block["by_layer"]
    assert isinstance(by_layer, dict)
    for layer_key, layer_block in by_layer.items():
        assert isinstance(layer_block, dict)
        by_ckpt = layer_block["by_checkpoint"]
        assert isinstance(by_ckpt, dict)
        for ckpt, cell in by_ckpt.items():
            assert isinstance(cell, dict)
            if set(cell.keys()) != {"related", "control_diagnostic"}:
                raise ValueError(
                    f"M2 {ckpt}/layer_{layer_key} cell keys must be "
                    f"{{'related','control_diagnostic'}}, got {sorted(cell.keys())}"
                )
            related_cells = cell["related"]
            assert isinstance(related_cells, dict)
            if set(related_cells.keys()) != set(RELATED_CONCEPTS):
                raise ValueError(
                    f"M2 {ckpt}/layer_{layer_key} related keys must be "
                    f"{sorted(RELATED_CONCEPTS)}, got {sorted(related_cells.keys())}"
                )
            for concept, entry in related_cells.items():
                assert isinstance(entry, dict)
                _validate_delta_entry(
                    entry, entry_keys, f"M2 {ckpt}/layer_{layer_key}/{concept}"
                )
            ctrl = cell["control_diagnostic"]
            assert isinstance(ctrl, dict)
            _validate_delta_entry(
                ctrl, entry_keys, f"M2 {ckpt}/layer_{layer_key}/control"
            )


def _validate_delta_entry(entry: dict[str, object], keys: set[str], ctx: str) -> None:
    """Check a ``{cos_current, cos_base, delta_cos}`` triple for range + consistency."""
    if set(entry.keys()) != keys:
        raise ValueError(
            f"{ctx} keys must be {sorted(keys)}, got {sorted(entry.keys())}"
        )
    cc = _require_number(entry["cos_current"], -1.0, 1.0, f"{ctx}/cos_current")
    cb = _require_number(entry["cos_base"], -1.0, 1.0, f"{ctx}/cos_base")
    dc = _require_number(entry["delta_cos"], -2.0, 2.0, f"{ctx}/delta_cos")
    if not math.isclose(dc, cc - cb, abs_tol=1e-9, rel_tol=0.0):
        raise ValueError(
            f"{ctx} delta_cos ({dc}) != cos_current - cos_base ({cc - cb})"
        )


def _validate_m3_fields(block: dict[str, object]) -> None:
    """M3: nonnegative norms + finite delta consistent with ``current - base``."""
    expected_keys = {"norm_current", "norm_base", "delta"}
    by_layer = block["by_layer"]
    assert isinstance(by_layer, dict)
    for layer_key, layer_block in by_layer.items():
        assert isinstance(layer_block, dict)
        by_ckpt = layer_block["by_checkpoint"]
        assert isinstance(by_ckpt, dict)
        for ckpt, cell in by_ckpt.items():
            assert isinstance(cell, dict)
            if set(cell.keys()) != expected_keys:
                raise ValueError(
                    f"M3 {ckpt}/layer_{layer_key} keys must be {sorted(expected_keys)}, "
                    f"got {sorted(cell.keys())}"
                )
            nc = _require_number(
                cell["norm_current"],
                0.0,
                math.inf,
                f"M3 {ckpt}/layer_{layer_key}/norm_current",
            )
            nb = _require_number(
                cell["norm_base"],
                0.0,
                math.inf,
                f"M3 {ckpt}/layer_{layer_key}/norm_base",
            )
            d = _require_number(
                cell["delta"], -math.inf, math.inf, f"M3 {ckpt}/layer_{layer_key}/delta"
            )
            if not math.isclose(d, nc - nb, abs_tol=1e-9, rel_tol=0.0):
                raise ValueError(
                    f"M3 {ckpt}/layer_{layer_key} delta ({d}) != norm_current - norm_base ({nc - nb})"
                )


def _validate_m4_fields(block: dict[str, object]) -> None:
    """M4: logistic method, exact 8 labels, bounded metrics, consistent counts."""
    if block.get("method") != "logistic":
        raise ValueError(f"M4 method must be 'logistic', got {block.get('method')!r}")
    label_names = block.get("label_names")
    if not isinstance(label_names, list) or list(label_names) != list(PROBE_CLASSES):
        raise ValueError(
            f"M4 label_names must be exactly {list(PROBE_CLASSES)}, got {label_names}"
        )
    params = block.get("linear_probe_params")
    if not isinstance(params, dict):
        raise ValueError("M4 missing 'linear_probe_params' dict")
    n_folds = params.get("n_folds")
    n_folds = _require_int(n_folds, "M4 linear_probe_params.n_folds", minimum=2)
    expected_cell_keys = {"by_label", "n_samples", "n_groups"}
    expected_label_keys = {
        "balanced_accuracy",
        "auroc",
        "n_positive",
        "n_negative",
        "n_folds_used",
        "n_folds_skipped",
    }
    by_layer = block["by_layer"]
    assert isinstance(by_layer, dict)
    for layer_key, layer_block in by_layer.items():
        assert isinstance(layer_block, dict)
        by_ckpt = layer_block["by_checkpoint"]
        assert isinstance(by_ckpt, dict)
        for ckpt, cell in by_ckpt.items():
            assert isinstance(cell, dict)
            if set(cell.keys()) != expected_cell_keys:
                raise ValueError(
                    f"M4 {ckpt}/layer_{layer_key} cell keys must be "
                    f"{sorted(expected_cell_keys)}, got {sorted(cell.keys())}"
                )
            n_samples = _require_int(
                cell["n_samples"], f"M4 {ckpt}/layer_{layer_key}/n_samples", minimum=1
            )
            _require_int(
                cell["n_groups"], f"M4 {ckpt}/layer_{layer_key}/n_groups", minimum=1
            )
            by_label = cell["by_label"]
            assert isinstance(by_label, dict)
            if set(by_label.keys()) != set(PROBE_CLASSES):
                raise ValueError(
                    f"M4 {ckpt}/layer_{layer_key} by_label must have exactly "
                    f"{sorted(PROBE_CLASSES)}, got {sorted(by_label.keys())}"
                )
            for name, r in by_label.items():
                assert isinstance(r, dict)
                if set(r.keys()) != expected_label_keys:
                    raise ValueError(
                        f"M4 {ckpt}/layer_{layer_key}/{name} keys must be "
                        f"{sorted(expected_label_keys)}, got {sorted(r.keys())}"
                    )
                _require_number(
                    r["balanced_accuracy"],
                    0.0,
                    1.0,
                    f"M4 {ckpt}/layer_{layer_key}/{name}/balanced_accuracy",
                )
                _require_number(
                    r["auroc"], 0.0, 1.0, f"M4 {ckpt}/layer_{layer_key}/{name}/auroc"
                )
                n_pos = _require_int(
                    r["n_positive"],
                    f"M4 {ckpt}/layer_{layer_key}/{name}/n_positive",
                    minimum=0,
                )
                n_neg = _require_int(
                    r["n_negative"],
                    f"M4 {ckpt}/layer_{layer_key}/{name}/n_negative",
                    minimum=0,
                )
                n_used = _require_int(
                    r["n_folds_used"],
                    f"M4 {ckpt}/layer_{layer_key}/{name}/n_folds_used",
                    minimum=1,
                )
                n_skip = _require_int(
                    r["n_folds_skipped"],
                    f"M4 {ckpt}/layer_{layer_key}/{name}/n_folds_skipped",
                    minimum=0,
                )
                # Pooled test counts cannot exceed the total scored samples
                # (degenerate-fold skips reduce them, so ``<=`` not ``==``).
                if n_pos + n_neg > n_samples:
                    raise ValueError(
                        f"M4 {ckpt}/layer_{layer_key}/{name} n_positive+n_negative "
                        f"({n_pos + n_neg}) exceeds n_samples ({n_samples})"
                    )
                if n_used + n_skip != n_folds:
                    raise ValueError(
                        f"M4 {ckpt}/layer_{layer_key}/{name} n_folds_used+n_folds_skipped "
                        f"({n_used + n_skip}) != n_folds ({n_folds})"
                    )


def _validate_metadata_consistency(
    metadata: dict[str, object],
    expected_checkpoints: Sequence[str],
    expected_layers: Sequence[int],
) -> None:
    """Validate metadata provenance fields against the expected grid exactly.

    Checks ``source_models``, ``humaneval_x_revision``, ``checkpoints``,
    ``layers``, ``base_checkpoint``, ``concepts`` (target/related/control), and
    ``probe_classes`` so a metrics file cannot silently describe a different
    model pair, dataset pin, schedule, layer set, or concept catalogue.
    """
    source_models = metadata.get("source_models")
    if not isinstance(source_models, dict) or source_models != SOURCE_MODELS:
        raise ValueError(
            f"metadata source_models must be exactly {SOURCE_MODELS!r}, got {source_models!r}"
        )
    revision = metadata.get("humaneval_x_revision")
    if revision != HUMANEVAL_X_REVISION:
        raise ValueError(
            f"metadata humaneval_x_revision must be {HUMANEVAL_X_REVISION!r}, got {revision!r}"
        )
    md_ckpts = metadata.get("checkpoints")
    if not isinstance(md_ckpts, list) or set(md_ckpts) != set(expected_checkpoints):
        raise ValueError(
            f"metadata checkpoints must cover {list(expected_checkpoints)!r}, got {md_ckpts!r}"
        )
    md_layers = metadata.get("layers")
    if not isinstance(md_layers, list) or set(md_layers) != set(expected_layers):
        raise ValueError(
            f"metadata layers must cover {list(expected_layers)!r}, got {md_layers!r}"
        )
    base_ckpt = metadata.get("base_checkpoint")
    if not isinstance(base_ckpt, str) or base_ckpt not in expected_checkpoints:
        raise ValueError(
            f"metadata base_checkpoint must be in the schedule, got {base_ckpt!r}"
        )
    concepts = metadata.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError("metadata concepts must be a dict")
    if concepts.get("target") != TARGET_CONCEPT:
        raise ValueError(
            f"metadata concepts.target must be {TARGET_CONCEPT!r}, got {concepts.get('target')!r}"
        )
    md_related = concepts.get("related")
    if not isinstance(md_related, list) or set(md_related) != set(RELATED_CONCEPTS):
        raise ValueError(
            f"metadata concepts.related must be {list(RELATED_CONCEPTS)!r}, got {md_related!r}"
        )
    if concepts.get("control") != CONTROL_CONCEPT:
        raise ValueError(
            f"metadata concepts.control must be {CONTROL_CONCEPT!r}, got {concepts.get('control')!r}"
        )
    md_probe = metadata.get("probe_classes")
    if not isinstance(md_probe, list) or list(md_probe) != list(PROBE_CLASSES):
        raise ValueError(
            f"metadata probe_classes must be {list(PROBE_CLASSES)!r}, got {md_probe!r}"
        )
    linear_probe = metadata.get("linear_probe")
    if not isinstance(linear_probe, dict):
        raise ValueError("metadata linear_probe must be a dict")
    if linear_probe.get("method") != "logistic":
        raise ValueError(
            f"metadata linear_probe.method must be 'logistic', got {linear_probe.get('method')!r}"
        )


def validate_metrics(
    metrics: dict[str, object],
    *,
    expected_checkpoints: Sequence[str],
    expected_layers: Sequence[int],
    expected_metric_keys: Sequence[str] = METRIC_KEYS,
) -> None:
    """Validate schema, metadata provenance, coverage, finiteness, and the exact
    field/type/range/count/label structure of every M1-M4 cell.

    Beyond the prior coverage + finiteness + n_samples checks, this verifies:
    * metadata provenance exactly (``source_models`` model-key -> HF id map,
      pinned ``humaneval_x_revision``, schedule, layers, concepts, probe
      classes, base checkpoint, linear-probe method);
    * M1 reference keys and ``[-1, 1]`` cosines;
    * M2 exactly four related concepts + one control, finite cosines/deltas with
      ``delta == current - base`` consistency;
    * M3 nonnegative norms with ``delta == current - base`` consistency;
    * M4 ``method == "logistic"``, exactly eight canonical labels, balanced
      accuracy / AUROC in ``[0, 1]``, nonnegative counts, fold-count identity
      (``used + skipped == n_folds``), and pooled counts ``<= n_samples``.

    Raises:
        ValueError: On any structural, coverage, finiteness, provenance,
            protocol, or per-metric field violation.
    """
    # --- Schema / version -------------------------------------------------
    if metrics.get("schema") != SCHEMA:
        raise ValueError(f"schema mismatch: {metrics.get('schema')!r}")
    if metrics.get("version") != VERSION:
        raise ValueError(f"version mismatch: {metrics.get('version')!r}")

    # --- Exactly four metric keys -----------------------------------------
    # ``schema``/``version``/``metadata`` are the only allowed non-metric
    # top-level keys; every other top-level key is treated as a metric payload
    # that must match ``expected_metric_keys`` exactly (no missing, no extras).
    allowed_non_metric = {"schema", "version", "metadata"}
    actual_metric_keys = set(metrics.keys()) - allowed_non_metric
    expected = set(expected_metric_keys)
    missing_keys = expected - actual_metric_keys
    extra_keys = actual_metric_keys - expected
    if missing_keys or extra_keys:
        raise ValueError(
            f"metric key mismatch: missing={sorted(missing_keys)}, "
            f"extra={sorted(extra_keys)}"
        )

    # --- Metadata required (dict + raw protocol + positive n_samples) -----
    metadata = metrics.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"metadata must be a dict, got {type(metrics.get('metadata')).__name__}"
        )
    proto = metadata.get("protocol")
    if proto != PROTOCOL:
        raise ValueError(f"metadata protocol is {proto!r}, expected {PROTOCOL!r}")
    n_samples_meta = metadata.get("n_samples")
    # bool is an int subclass; reject it explicitly.
    if isinstance(n_samples_meta, bool) or not isinstance(n_samples_meta, int):
        raise ValueError(
            f"metadata n_samples must be a non-boolean int, got {n_samples_meta!r}"
        )
    if n_samples_meta <= 0:
        raise ValueError(f"metadata n_samples must be positive, got {n_samples_meta}")

    # --- Metadata provenance consistency (models, dataset pin, schedule) --
    _validate_metadata_consistency(metadata, expected_checkpoints, expected_layers)

    # --- Coverage + finiteness -------------------------------------------
    expected_layer_keys = {str(l) for l in expected_layers}
    expected_ckpt_set = set(expected_checkpoints)

    for key in expected_metric_keys:
        block = metrics[key]
        if not isinstance(block, dict):
            raise ValueError(f"metric {key!r} is not a dict")
        by_layer = block.get("by_layer")
        if not isinstance(by_layer, dict):
            raise ValueError(f"metric {key!r} missing 'by_layer'")

        actual_layers = set(by_layer.keys())
        if actual_layers != expected_layer_keys:
            miss = expected_layer_keys - actual_layers
            extra = actual_layers - expected_layer_keys
            raise ValueError(
                f"metric {key!r} layer mismatch: missing={sorted(miss)}, "
                f"extra={sorted(extra)}"
            )

        for layer_key, layer_block in by_layer.items():
            if not isinstance(layer_block, dict):
                raise ValueError(f"metric {key!r} layer {layer_key} is not a dict")
            by_ckpt = layer_block.get("by_checkpoint")
            if not isinstance(by_ckpt, dict):
                raise ValueError(
                    f"metric {key!r} layer {layer_key} missing 'by_checkpoint'"
                )
            actual_ckpt = set(by_ckpt.keys())
            if actual_ckpt != expected_ckpt_set:
                miss = expected_ckpt_set - actual_ckpt
                extra = actual_ckpt - expected_ckpt_set
                raise ValueError(
                    f"metric {key!r} layer {layer_key} checkpoint mismatch: "
                    f"missing={sorted(miss)}, extra={sorted(extra)}"
                )
            _assert_all_finite(by_ckpt, f"{key}/{layer_key}")

    # --- M4 n_samples uniformity + positivity -----------------------------
    # Validate the ``n_samples`` field itself (positive non-boolean int, uniform
    # across the grid, matching ``metadata.n_samples``) BEFORE the per-cell count
    # checks, so a grid-level ``n_samples`` violation surfaces ahead of the
    # count-vs-n_samples cell check.
    _validate_m4_n_samples(
        metrics,
        expected_metric_keys,
        expected_layers,
        expected_ckpt_set,
        n_samples_meta,
    )

    # --- Per-metric field structure (types / ranges / counts / labels) ----
    base_checkpoint = metadata["base_checkpoint"]
    assert isinstance(base_checkpoint, str)
    m1_block = metrics[METRIC_KEYS[0]]
    m2_block = metrics[METRIC_KEYS[1]]
    m3_block = metrics[METRIC_KEYS[2]]
    m4_block = metrics[METRIC_KEYS[3]]
    assert isinstance(m1_block, dict) and isinstance(m2_block, dict)
    assert isinstance(m3_block, dict) and isinstance(m4_block, dict)
    _validate_m1_fields(m1_block, base_checkpoint)
    _validate_m2_fields(m2_block)
    _validate_m3_fields(m3_block)
    _validate_m4_fields(m4_block)


# =============================================================================
# Atomic output
# =============================================================================


def write_metrics_json(path: str, metrics: dict[str, object]) -> str:
    """Write ``metrics.json`` atomically (temp file + ``os.replace``)."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


# =============================================================================
# Orchestration
# =============================================================================


def _resolve_n_samples(
    m4: dict[str, object],
    layers: Sequence[int],
    checkpoints: Sequence[str],
) -> int:
    """Return the singular per-cell ``n_samples`` recorded by M4.

    M4 records ``result.n_samples`` for every ``(checkpoint, layer)`` cell.
    The metadata reports a single ``n_samples`` value, so this helper verifies
    the count is identical across the whole grid and returns it.

    Raises:
        ValueError: If the grid is empty, a cell is malformed, a cell's
            ``n_samples`` is a bool / not an int / non-positive, or cells
            disagree on ``n_samples``.
    """
    by_layer_obj = m4.get("by_layer")
    if not isinstance(by_layer_obj, dict):
        raise ValueError("M4 block missing 'by_layer'")
    counts: set[int] = set()
    for layer in layers:
        layer_block = by_layer_obj.get(str(layer))
        if not isinstance(layer_block, dict):
            raise ValueError(f"M4 missing layer {layer}")
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            raise ValueError(f"M4 layer {layer} missing 'by_checkpoint'")
        for ckpt in checkpoints:
            cell = by_ckpt.get(ckpt)
            if not isinstance(cell, dict):
                raise ValueError(f"M4 {ckpt}/{layer} missing cell")
            n = cell.get("n_samples")
            # bool is an int subclass; reject it so True/False can't pose as
            # a valid sample count.
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError(
                    f"M4 {ckpt}/{layer} 'n_samples' must be a non-boolean int, "
                    f"got {n!r}"
                )
            if n <= 0:
                raise ValueError(
                    f"M4 {ckpt}/{layer} 'n_samples' must be positive, got {n}"
                )
            counts.add(n)
    if not counts:
        raise ValueError("M4 grid is empty; cannot resolve n_samples")
    if len(counts) > 1:
        raise ValueError(
            f"M4 n_samples differs across (checkpoint, layer) grid: {sorted(counts)}"
        )
    return next(iter(counts))


def compute_all_metrics(
    *,
    concept_vectors_root: str | None = None,
    activations_root: str | None = None,
    layers: Sequence[int] | None = None,
    checkpoints: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
    base_checkpoint: str = BASE_CHECKPOINT,
    optional_references: Sequence[str] = DEFAULT_OPTIONAL_REFERENCES,
    target_concept: str = TARGET_CONCEPT,
    related_concepts: Sequence[str] | None = None,
    control_concept: str = CONTROL_CONCEPT,
    probe_classes: Sequence[str] = PROBE_CLASSES,
    n_folds: int = DEFAULT_N_FOLDS,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
    standardize: bool = DEFAULT_STANDARDIZE,
) -> dict[str, object]:
    """Compute all four metrics from on-disk artifacts and validate them.

    Loads raw concept directions and raw probe activations, computes M1-M4,
    assembles the metrics dict, and runs :func:`validate_metrics` before
    returning.

    Args:
        concept_vectors_root: Override for ``{results_root}/concept_vectors``.
        activations_root: Override for ``{results_root}/activations``.
        layers, checkpoints, concepts: Defaults to the experiment constants.
        base_checkpoint: Primary M1/M2/M3 reference (always ``main``).
        optional_references: Additional M1 references (default ``step_100``).
        target_concept, related_concepts, control_concept: Concept keys.
        probe_classes: Eight canonical class names.
        n_folds, alpha, seed, standardize: Logistic-probe hyper-parameters.

    Returns:
        The validated metrics dict (schema/version/metadata + four keys).

    Raises:
        ValueError: If an optional reference is not in ``checkpoints`` or
            validation fails.
        FileNotFoundError: If an on-disk artifact is missing.
    """
    layers_resolved = list(EXPERIMENT_LAYERS if layers is None else layers)
    checkpoints_resolved = list(
        EXPERIMENT_CHECKPOINTS if checkpoints is None else checkpoints
    )
    concepts_resolved = tuple(EXPERIMENT_CONCEPTS if concepts is None else concepts)
    related_resolved = tuple(
        RELATED_CONCEPTS if related_concepts is None else related_concepts
    )
    cv_root = concept_vectors_root or default_concept_vectors_root()
    act_root = activations_root or default_activations_root()

    # Validate optional references are available.
    for ref in optional_references:
        if ref not in checkpoints_resolved:
            raise ValueError(
                f"optional reference {ref!r} is not in the checkpoint schedule"
            )

    # --- Load all concept raw directions ----------------------------------
    all_dirs = load_all_concept_directions(
        cv_root, layers_resolved, checkpoints_resolved, concepts_resolved
    )
    target_dirs: TargetDirs = {
        ckpt: {
            layer: all_dirs[ckpt][layer][target_concept] for layer in layers_resolved
        }
        for ckpt in checkpoints_resolved
    }

    # --- M1 / M2 / M3 (from concept vectors) ------------------------------
    m1 = metric_m1_checkpoint_cosine(
        target_dirs,
        layers_resolved,
        checkpoints_resolved,
        base_checkpoint,
        optional_references,
    )
    m2 = metric_m2_target_related_delta_cos(
        all_dirs,
        layers_resolved,
        checkpoints_resolved,
        base_checkpoint,
        target_concept,
        related_resolved,
        control_concept,
    )
    m3 = metric_m3_raw_direction_magnitude_delta(
        target_dirs,
        layers_resolved,
        checkpoints_resolved,
        base_checkpoint,
        target_concept,
    )

    # --- M4 (from probe activations) --------------------------------------
    activation_loader = make_activation_loader(act_root, probe_classes)
    m4 = metric_m4_eight_class_grouped_logistic_separability(
        activation_loader,
        layers_resolved,
        checkpoints_resolved,
        probe_classes,
        n_folds=n_folds,
        alpha=alpha,
        seed=seed,
        standardize=standardize,
    )

    # --- Assemble ---------------------------------------------------------
    metrics: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "metadata": {
            "checkpoints": list(checkpoints_resolved),
            "layers": list(layers_resolved),
            "concepts": {
                "target": target_concept,
                "related": list(related_resolved),
                "control": control_concept,
            },
            "probe_classes": list(probe_classes),
            "protocol": PROTOCOL,
            "n_samples": _resolve_n_samples(m4, layers_resolved, checkpoints_resolved),
            "base_checkpoint": base_checkpoint,
            "optional_references": list(optional_references),
            "source_models": dict(SOURCE_MODELS),
            "humaneval_x_revision": HUMANEVAL_X_REVISION,
            "linear_probe": {
                "method": m4.get("method", "logistic"),
                "n_folds": n_folds,
                "alpha": alpha,
                "seed": seed,
                "standardize": standardize,
            },
        },
        "m1_checkpoint_cosine": m1,
        "m2_target_related_delta_cos": m2,
        "m3_target_raw_direction_magnitude_delta": m3,
        "m4_eight_class_grouped_logistic_separability": m4,
    }

    # --- Validate ---------------------------------------------------------
    validate_metrics(
        metrics,
        expected_checkpoints=checkpoints_resolved,
        expected_layers=layers_resolved,
    )
    return metrics


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute the four RL-Zero-Code syntax representation/readout "
            "metrics from on-disk raw concept vectors and probe activations."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_METRICS_PATH,
        help="Output metrics.json path (default: %(default)s)",
    )
    parser.add_argument(
        "--concept-vectors-root",
        default=None,
        help="Override concept-vectors directory",
    )
    parser.add_argument(
        "--activations-root",
        default=None,
        help="Override probe-activations directory",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help="Grouped CV folds for the logistic probe (default: %(default)s)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Logistic L2 penalty on the weights (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Grouped-fold shuffle seed (default: %(default)s)",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-fold train standardization",
    )
    parser.add_argument(
        "--no-step-100-ref",
        action="store_true",
        help="Disable step_100 as an optional M1 reference",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: compute metrics and write atomic JSON."""
    args = parse_args(argv)
    optional_refs: tuple[str, ...] = (
        () if args.no_step_100_ref else DEFAULT_OPTIONAL_REFERENCES
    )
    metrics = compute_all_metrics(
        concept_vectors_root=args.concept_vectors_root,
        activations_root=args.activations_root,
        optional_references=optional_refs,
        n_folds=args.n_folds,
        alpha=args.alpha,
        seed=args.seed,
        standardize=not args.no_standardize,
    )
    path = write_metrics_json(args.output, metrics)
    print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
