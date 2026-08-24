"""Deterministic, atomic, reproducible producer + validator for
``analysis_summary.json`` (RL-Zero-Code syntax experiment).

The original ``results/rl_zero_code_syntax/analysis_summary.json`` was produced
ad hoc with a Python one-liner and was therefore neither reproducible nor
integrity-bound to its sources. This module replaces it with a deterministic
pipeline that:

* reads the authoritative raw-protocol ``metrics.json`` (v2 logistic) and the
  downstream ``aggregate_summary.json`` (11 checkpoints);
* rebuilds the **exact** 11-checkpoint table, where each row is the unweighted
  mean across the ten experiment layers (and across the four related concepts
  for M2-related, and across the two target syntax labels for M4);
* recomputes Pearson and Spearman correlations of the six aggregated
  representation/readout metrics against Python pass@1 and MMLU accuracy, using
  :func:`scipy.stats.pearsonr` and :func:`scipy.stats.spearmanr` so the values
  are bit-for-bit reproducible on the same SciPy version;
* discloses the ``step_100`` / ``step_1000`` duplicate-weight observation and
  the scientific limitations honestly;
* binds the exact source artifacts by SHA-256 (``metrics.json`` and
  ``aggregate_summary.json`` at minimum) plus a config-coordinate fingerprint
  (checkpoints, layers, concepts, probe classes, base checkpoint, optional
  references, N samples, raw protocol, dataset pin, model pair, linear-probe
  parameters);
* emits a canonical JSON build fingerprint (SHA-256 over the canonical serialized
  summary) so any downstream consumer can detect tampering with the summary
  itself;
* writes the file **atomically** via ``tempfile.mkstemp`` + ``fsync`` +
  ``os.replace`` so a crash mid-write cannot leave a half-written summary;
* provides a strict validator (:func:`validate_summary`) that detects changed
  sources (hash mismatch), missing sources, schema/version drift, tampered
  build fingerprints, broken aggregation invariants, missing limitations, and
  correlation/value-shape violations.

This module is **pure post-processing**. It runs no model, performs no
extraction, never mutates ``metrics.json`` / ``aggregate_summary.json``, and
introduces no wall-clock or RNG nondeterminism. The summary it produces is a
pure function of its two source artifacts and the constants in
:mod:`src.rl_zero_experiment`.

Authoritative docs:

* :mod:`src.rl_zero_experiment` -- experiment constants (checkpoints, layers,
  concepts, probe classes, base checkpoint, optional references, sample count,
  primary protocol).
* ``experiments/run_rl_zero_syntax_metrics.py`` -- metrics.json producer.
* ``experiments/run_rl_zero_downstream.py`` -- aggregate_summary.json producer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from scipy.stats import pearsonr, spearmanr

from src.rl_zero_experiment import (
    BASE_CHECKPOINT,
    CONTROL_CONCEPT,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_LAYERS,
    PROBE_CLASSES,
    RELATED_CONCEPTS,
    RL_ZERO_CODE_RESULTS_ROOT,
    TARGET_CONCEPT,
)

# =============================================================================
# Schema constants
# =============================================================================

#: Summary schema identifier recorded in every produced file.
SCHEMA: str = "rl_zero_code_syntax_analysis_summary"

#: Summary schema version. The legacy ad-hoc file was ``v2``; the new
#: deterministic producer keeps the same major version (the analysis semantics
#: are unchanged) and is bit-compatible with the ad-hoc output when computed
#: against the same source artifacts.
VERSION: int = 2

#: The two target-syntax labels whose M4 balanced accuracy / AUROC are averaged
#: (one-vs-rest logistic probes) to produce the ``m4_syntax_*`` aggregates.
SYNTAX_LABELS: tuple[str, ...] = ("python_valid", "python_syntax_error")

#: The six aggregated metric keys, in canonical order. Each is a row field in
#: the per-checkpoint table and a correlation source.
AGGREGATE_METRIC_KEYS: tuple[str, ...] = (
    "m1_mean_cos_vs_main",
    "m2_mean_related_delta_cos",
    "m2_mean_control_delta_cos",
    "m3_mean_norm_delta",
    "m4_syntax_balanced_accuracy",
    "m4_syntax_auroc",
)

#: The two downstream targets correlated against every aggregate metric.
DOWNSTREAM_TARGET_KEYS: tuple[str, ...] = (
    "python_pass_at_1",
    "mmlu_accuracy",
)

#: ``cpp_pass_at_1`` is also surfaced per row but, because it is constant zero
#: under the strict raw ``prompt + completion + official test`` protocol, it is
#: explicitly excluded from correlation (the correlation is undefined).
ROW_ONLY_FIELDS: tuple[str, ...] = ("cpp_pass_at_1",)

#: Default source paths inside the isolated results root.
DEFAULT_METRICS_PATH: str = os.path.join(
    RL_ZERO_CODE_RESULTS_ROOT, "metrics", "metrics.json"
)
DEFAULT_AGGREGATE_PATH: str = os.path.join(
    RL_ZERO_CODE_RESULTS_ROOT, "downstream", "aggregate_summary.json"
)
DEFAULT_SUMMARY_PATH: str = os.path.join(
    RL_ZERO_CODE_RESULTS_ROOT, "analysis_summary.json"
)

#: Filenames recorded in the ``source_hashes`` block (relative to the results
#: root) so a reader can re-resolve them. The full paths are also captured in
#: ``source_paths`` to remove any ambiguity.
METRICS_RELPATH: str = os.path.relpath(DEFAULT_METRICS_PATH, RL_ZERO_CODE_RESULTS_ROOT)
AGGREGATE_RELPATH: str = os.path.relpath(
    DEFAULT_AGGREGATE_PATH, RL_ZERO_CODE_RESULTS_ROOT
)

#: The two checkpoints that ship with identical weights and must never be
#: treated as independent evidence. Disclosed under ``known_duplicate``.
DUPLICATE_CHECKPOINTS: tuple[str, ...] = ("step_100", "step_1000")

#: Aggregation description (mirrors the legacy ``aggregation`` block).
AGGREGATION_DESCRIPTION: dict[str, str] = {
    "layers": "unweighted mean over 10 layers",
    "m2": "also unweighted mean over four related concepts",
    "m4": "mean over python_valid and python_syntax_error one-vs-rest logistic probes",
}

#: Scientific limitations disclosed verbatim by every produced summary. These
#: are factual properties of the underlying experiment (n=11, duplicate
#: weights, zero C++, sparse sensitivity) and must never be stripped from the
#: output by a refactor.
LIMITATIONS: tuple[str, ...] = (
    "n=11 sequential checkpoints; correlations are exploratory and not causal",
    "step_100 and step_1000 are duplicate-weight observations",
    "C++ pass@1 is constant zero because strict raw completions commonly add main(); correlation is undefined",
    "sensitivity target chat direction is unavailable; 30 target entries remain chat_missing",
)

#: Integrity provenance notes mirrored from the durable run notes; they
#: describe the cache integrity posture, not new claims.
INTEGRITY_NOTES: dict[str, str] = {
    "probe_sidecars": "re-extracted with per-record text_sha256/source_ids",
    "downstream": "1100 cached code completions re-scored in bubblewrap with zero outcome drift",
    "tests": "results/rl_zero_code_syntax/test-results.xml",
}


# =============================================================================
# Errors
# =============================================================================


class AnalysisSummaryError(ValueError):
    """Base class for analysis-summary errors."""


class SourceHashMismatch(AnalysisSummaryError):
    """A source artifact's on-disk SHA-256 differs from the recorded value."""


class MissingSource(AnalysisSummaryError):
    """A required source artifact is absent."""


class SummaryTamper(AnalysisSummaryError):
    """The summary's recorded build fingerprint disagrees with its content."""


# =============================================================================
# Hashing helpers (deterministic, no wall-clock)
# =============================================================================


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the hex SHA-256 digest of ``path`` read as raw bytes.

    The file is streamed in 1 MiB chunks so multi-megabyte metrics files do
    not need to fit in memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Mapping[str, object]) -> bytes:
    """Serialize ``obj`` to canonical JSON bytes used for fingerprinting.

    The fingerprint covers a *content* projection of the summary -- i.e. the
    schema, version, aggregation semantics, source identity, config
    fingerprint, source hashes, checkpoint table, correlations, duplicate
    disclosure, limitations, and integrity notes -- but **excludes** the
    recorded ``build_fingerprint`` field itself (which would be circular) and
    any source path strings (which are environment-specific).

    The serialization is fully deterministic:

    * ``sort_keys=True`` -- key order does not depend on insertion order;
    * ``ensure_ascii=False`` -- the UTF-8 bytes are stable across locales;
    * ``allow_nan=False`` -- a NaN would silently serialize to ``NaN`` which is
      invalid JSON; we refuse it instead so a corrupt float cannot forge a
      fingerprint over non-standard tokens;
    * ``separators=(",", ":")`` -- no incidental whitespace.

    A trailing newline is appended so the canonical form matches the on-disk
    file format produced by :func:`write_summary_atomically`.
    """
    return (
        json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def compute_build_fingerprint(summary_content: Mapping[str, object]) -> str:
    """Compute the canonical SHA-256 build fingerprint of the summary content.

    ``summary_content`` MUST already have the ``build_fingerprint`` field
    removed (or set to ``None``); this function injects no field, it only
    hashes the canonical serialization of what it is given.
    """
    payload = {k: v for k, v in summary_content.items() if k != "build_fingerprint"}
    return sha256_bytes(canonical_json_bytes(payload))


def config_fingerprint(
    checkpoints: Sequence[str],
    layers: Sequence[int],
    concepts: Mapping[str, object],
    probe_classes: Sequence[str],
    base_checkpoint: str,
    optional_references: Sequence[str],
    *,
    n_samples: int,
    protocol: str,
    humaneval_x_revision: str,
    source_models: Mapping[str, str],
    linear_probe: Mapping[str, object],
) -> str:
    """Compute a deterministic SHA-256 fingerprint over the experiment config.

    Binds every coordinate that could plausibly shift the analysis if changed:
    checkpoint schedule, layer indices, concept catalogue, probe-class
    ordering, base/reference checkpoints, sample count, extraction protocol,
    the HumanEval-X dataset pin, the source model HF ids, and the linear-probe
    hyper-parameters. The fingerprint is recorded in the summary so a reader
    can detect that a regenerated summary used the same experiment coordinate
    even if the source artifacts themselves drifted.
    """
    related = concepts["related"]
    if not isinstance(related, list):
        raise AnalysisSummaryError(
            f"concepts.related must be a list, got {type(related).__name__}"
        )
    payload: dict[str, object] = {
        "checkpoints": list(checkpoints),
        "layers": list(layers),
        "concepts": {
            "target": concepts["target"],
            "related": list(related),
            "control": concepts["control"],
        },
        "probe_classes": list(probe_classes),
        "base_checkpoint": base_checkpoint,
        "optional_references": list(optional_references),
        "n_samples": int(n_samples),
        "protocol": str(protocol),
        "humaneval_x_revision": str(humaneval_x_revision),
        "source_models": dict(source_models),
        "linear_probe": dict(linear_probe),
    }
    return sha256_bytes(canonical_json_bytes(payload))


# =============================================================================
# Source loaders (no mutation, no wall-clock)
# =============================================================================


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON object from ``path``; raise :class:`MissingSource` if absent.

    Accepts only JSON objects (dicts) at the top level; a non-dict payload is a
    structural corruption and is rejected explicitly rather than silently
    coerced.
    """
    p = Path(path)
    if not p.exists():
        raise MissingSource(f"required source not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AnalysisSummaryError(f"invalid JSON in {p}: {e}") from e
    if not isinstance(raw, dict):
        raise AnalysisSummaryError(
            f"{p}: expected a JSON object, got {type(raw).__name__}"
        )
    return raw


# =============================================================================
# Per-checkpoint aggregation (unweighted means; preserves ad-hoc semantics)
# =============================================================================


def _mean(values: Sequence[float]) -> float:
    """Unweighted arithmetic mean. Raises ``ValueError`` on an empty sequence."""
    if not values:
        raise ValueError("cannot take the mean of an empty sequence")
    return float(sum(values)) / len(values)


def aggregate_m1_cos_vs_main(
    metrics: Mapping[str, object],
    checkpoint: str,
    layers: Sequence[int],
) -> float:
    """Mean over layers of ``cos_vs_main`` for one checkpoint (M1)."""
    m1 = metrics["m1_checkpoint_cosine"]
    if not isinstance(m1, dict):
        raise AnalysisSummaryError("metrics: m1 block must be a dict")
    by_layer = m1.get("by_layer")
    if not isinstance(by_layer, dict):
        raise AnalysisSummaryError("metrics: m1 block missing 'by_layer'")
    values: list[float] = []
    for layer in layers:
        layer_block = by_layer.get(str(layer))
        if not isinstance(layer_block, dict):
            raise AnalysisSummaryError(f"metrics: m1 missing layer {layer}")
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            raise AnalysisSummaryError(
                f"metrics: m1 layer {layer} missing 'by_checkpoint'"
            )
        cell = by_ckpt.get(checkpoint)
        if not isinstance(cell, dict):
            raise AnalysisSummaryError(
                f"metrics: m1 missing checkpoint {checkpoint!r} at layer {layer}"
            )
        v = cell.get("cos_vs_main")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnalysisSummaryError(
                f"metrics: m1 {checkpoint}/layer_{layer}/cos_vs_main must be a "
                f"number, got {v!r}"
            )
        f = float(v)
        if not math.isfinite(f):
            raise AnalysisSummaryError(
                f"metrics: m1 {checkpoint}/layer_{layer}/cos_vs_main is non-finite: {f}"
            )
        values.append(f)
    return _mean(values)


def aggregate_m2_related_delta_cos(
    metrics: Mapping[str, object],
    checkpoint: str,
    layers: Sequence[int],
    related_concepts: Sequence[str],
) -> float:
    """Mean over (layers x related concepts) of M2 ``related.delta_cos``."""
    m2 = metrics["m2_target_related_delta_cos"]
    if not isinstance(m2, dict):
        raise AnalysisSummaryError("metrics: m2 block must be a dict")
    by_layer = m2.get("by_layer")
    if not isinstance(by_layer, dict):
        raise AnalysisSummaryError("metrics: m2 block missing 'by_layer'")
    values: list[float] = []
    for layer in layers:
        layer_block = by_layer.get(str(layer))
        if not isinstance(layer_block, dict):
            raise AnalysisSummaryError(f"metrics: m2 missing layer {layer}")
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            raise AnalysisSummaryError(
                f"metrics: m2 layer {layer} missing 'by_checkpoint'"
            )
        cell = by_ckpt.get(checkpoint)
        if not isinstance(cell, dict):
            raise AnalysisSummaryError(
                f"metrics: m2 missing checkpoint {checkpoint!r} at layer {layer}"
            )
        related = cell.get("related")
        if not isinstance(related, dict):
            raise AnalysisSummaryError(
                f"metrics: m2 {checkpoint}/layer_{layer} missing 'related' block"
            )
        per_concept_means: list[float] = []
        for concept in related_concepts:
            entry = related.get(concept)
            if not isinstance(entry, dict):
                raise AnalysisSummaryError(
                    f"metrics: m2 {checkpoint}/layer_{layer}/{concept} missing"
                )
            v = entry.get("delta_cos")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise AnalysisSummaryError(
                    f"metrics: m2 {checkpoint}/layer_{layer}/{concept}/delta_cos "
                    f"must be a number, got {v!r}"
                )
            f = float(v)
            if not math.isfinite(f):
                raise AnalysisSummaryError(
                    f"metrics: m2 {checkpoint}/layer_{layer}/{concept}/delta_cos "
                    f"is non-finite: {f}"
                )
            per_concept_means.append(f)
        values.append(_mean(per_concept_means))
    return _mean(values)


def aggregate_m2_control_delta_cos(
    metrics: Mapping[str, object],
    checkpoint: str,
    layers: Sequence[int],
) -> float:
    """Mean over layers of the ``she -> he`` control diagnostic ``delta_cos``.

    The control is reported as a diagnostic only -- never a fifth metric. Its
    aggregate is included in the table so a reader can sanity-check that the
    post-training drift is dominated by code, not gender.
    """
    m2 = metrics["m2_target_related_delta_cos"]
    if not isinstance(m2, dict):
        raise AnalysisSummaryError("metrics: m2 block must be a dict")
    by_layer = m2.get("by_layer")
    if not isinstance(by_layer, dict):
        raise AnalysisSummaryError("metrics: m2 block missing 'by_layer'")
    values: list[float] = []
    for layer in layers:
        layer_block = by_layer.get(str(layer))
        if not isinstance(layer_block, dict):
            raise AnalysisSummaryError(f"metrics: m2 missing layer {layer}")
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            raise AnalysisSummaryError(
                f"metrics: m2 layer {layer} missing 'by_checkpoint'"
            )
        cell = by_ckpt.get(checkpoint)
        if not isinstance(cell, dict):
            raise AnalysisSummaryError(
                f"metrics: m2 missing checkpoint {checkpoint!r} at layer {layer}"
            )
        ctrl = cell.get("control_diagnostic")
        if not isinstance(ctrl, dict):
            raise AnalysisSummaryError(
                f"metrics: m2 {checkpoint}/layer_{layer} missing control_diagnostic"
            )
        v = ctrl.get("delta_cos")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnalysisSummaryError(
                f"metrics: m2 {checkpoint}/layer_{layer}/control delta_cos "
                f"must be a number, got {v!r}"
            )
        f = float(v)
        if not math.isfinite(f):
            raise AnalysisSummaryError(
                f"metrics: m2 {checkpoint}/layer_{layer}/control delta_cos "
                f"is non-finite: {f}"
            )
        values.append(f)
    return _mean(values)


def aggregate_m3_norm_delta(
    metrics: Mapping[str, object],
    checkpoint: str,
    layers: Sequence[int],
) -> float:
    """Mean over layers of the M3 raw-direction magnitude ``delta``."""
    m3 = metrics["m3_target_raw_direction_magnitude_delta"]
    if not isinstance(m3, dict):
        raise AnalysisSummaryError("metrics: m3 block must be a dict")
    by_layer = m3.get("by_layer")
    if not isinstance(by_layer, dict):
        raise AnalysisSummaryError("metrics: m3 block missing 'by_layer'")
    values: list[float] = []
    for layer in layers:
        layer_block = by_layer.get(str(layer))
        if not isinstance(layer_block, dict):
            raise AnalysisSummaryError(f"metrics: m3 missing layer {layer}")
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            raise AnalysisSummaryError(
                f"metrics: m3 layer {layer} missing 'by_checkpoint'"
            )
        cell = by_ckpt.get(checkpoint)
        if not isinstance(cell, dict):
            raise AnalysisSummaryError(
                f"metrics: m3 missing checkpoint {checkpoint!r} at layer {layer}"
            )
        v = cell.get("delta")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnalysisSummaryError(
                f"metrics: m3 {checkpoint}/layer_{layer}/delta must be a number, "
                f"got {v!r}"
            )
        f = float(v)
        if not math.isfinite(f):
            raise AnalysisSummaryError(
                f"metrics: m3 {checkpoint}/layer_{layer}/delta is non-finite: {f}"
            )
        values.append(f)
    return _mean(values)


def aggregate_m4_syntax_metric(
    metrics: Mapping[str, object],
    checkpoint: str,
    layers: Sequence[int],
    *,
    field: str,
) -> float:
    """Mean over (layers x {python_valid, python_syntax_error}) of an M4 field.

    ``field`` is one of ``"balanced_accuracy"`` or ``"auroc"``. The two
    target-syntax labels are averaged into a single M4 syntax score per the
    legacy semantics.
    """
    if field not in ("balanced_accuracy", "auroc"):
        raise ValueError(
            f"M4 aggregate field must be 'balanced_accuracy' or 'auroc', got {field!r}"
        )
    m4 = metrics["m4_eight_class_grouped_logistic_separability"]
    if not isinstance(m4, dict):
        raise AnalysisSummaryError("metrics: m4 block must be a dict")
    by_layer = m4.get("by_layer")
    if not isinstance(by_layer, dict):
        raise AnalysisSummaryError("metrics: m4 block missing 'by_layer'")
    values: list[float] = []
    for layer in layers:
        layer_block = by_layer.get(str(layer))
        if not isinstance(layer_block, dict):
            raise AnalysisSummaryError(f"metrics: m4 missing layer {layer}")
        by_ckpt = layer_block.get("by_checkpoint")
        if not isinstance(by_ckpt, dict):
            raise AnalysisSummaryError(
                f"metrics: m4 layer {layer} missing 'by_checkpoint'"
            )
        cell = by_ckpt.get(checkpoint)
        if not isinstance(cell, dict):
            raise AnalysisSummaryError(
                f"metrics: m4 missing checkpoint {checkpoint!r} at layer {layer}"
            )
        by_label = cell.get("by_label")
        if not isinstance(by_label, dict):
            raise AnalysisSummaryError(
                f"metrics: m4 {checkpoint}/layer_{layer} missing 'by_label'"
            )
        for label in SYNTAX_LABELS:
            entry = by_label.get(label)
            if not isinstance(entry, dict):
                raise AnalysisSummaryError(
                    f"metrics: m4 {checkpoint}/layer_{layer}/{label} missing"
                )
            v = entry.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise AnalysisSummaryError(
                    f"metrics: m4 {checkpoint}/layer_{layer}/{label}/{field} "
                    f"must be a number, got {v!r}"
                )
            f = float(v)
            if not math.isfinite(f):
                raise AnalysisSummaryError(
                    f"metrics: m4 {checkpoint}/layer_{layer}/{label}/{field} "
                    f"is non-finite: {f}"
                )
            values.append(f)
    return _mean(values)


# =============================================================================
# Downstream extraction
# =============================================================================


def downstream_value(
    aggregate: Mapping[str, object],
    checkpoint: str,
    field: str,
) -> float:
    """Read a numeric downstream field for ``checkpoint`` from the aggregate."""
    checkpoints = aggregate.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise AnalysisSummaryError("aggregate_summary.json: missing 'checkpoints' dict")
    cell = checkpoints.get(checkpoint)
    if not isinstance(cell, dict):
        raise AnalysisSummaryError(
            f"aggregate_summary.json: missing checkpoint {checkpoint!r}"
        )
    v = cell.get(field)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise AnalysisSummaryError(
            f"aggregate_summary.json: {checkpoint}/{field} must be a number, got {v!r}"
        )
    f = float(v)
    if not math.isfinite(f):
        raise AnalysisSummaryError(
            f"aggregate_summary.json: {checkpoint}/{field} is non-finite: {f}"
        )
    return f


# =============================================================================
# Correlation (deterministic SciPy)
# =============================================================================


def correlate(
    metric_values: Sequence[float],
    target_values: Sequence[float],
) -> dict[str, float]:
    """Pearson r and Spearman rho (plus p-values) of two equal-length series.

    Uses :func:`scipy.stats.pearsonr` and :func:`scipy.stats.spearmanr` so the
    output is bit-for-bit reproducible on a fixed SciPy version. The result
    dict mirrors the per-target shape already shipped by the legacy ad-hoc
    summary, so the new producer is byte-compatible.

    A constant input series raises :class:`AnalysisSummaryError` -- the
    correlation is mathematically undefined (zero variance) and silently
    emitting ``NaN`` would propagate corruption into the summary. The strict
    raw-protocol ``cpp_pass@1`` is therefore explicitly excluded from the
    correlation block (it is constant zero across all 11 checkpoints).
    """
    n = len(metric_values)
    if n != len(target_values):
        raise ValueError(f"correlate: length mismatch ({n} vs {len(target_values)})")
    if n < 2:
        raise ValueError(f"correlate: need >= 2 points, got {n}")
    a = [float(x) for x in metric_values]
    b = [float(x) for x in target_values]
    for label, series in (("metric", a), ("target", b)):
        if any(not math.isfinite(x) for x in series):
            raise AnalysisSummaryError(
                f"correlate: {label} series contains non-finite value"
            )
    if len(set(a)) == 1:
        raise AnalysisSummaryError(
            "correlate: metric series is constant; correlation undefined"
        )
    if len(set(b)) == 1:
        raise AnalysisSummaryError(
            "correlate: target series is constant; correlation undefined"
        )
    pearson_res = pearsonr(a, b)
    spearman_res = spearmanr(a, b)
    pearson_r = float(cast(float, pearson_res[0]))
    pearson_p = float(cast(float, pearson_res[1]))
    spearman_rho = float(cast(float, getattr(spearman_res, "statistic")))
    spearman_p = float(cast(float, getattr(spearman_res, "pvalue")))
    for label, value in (
        ("pearson_r", pearson_r),
        ("pearson_p", pearson_p),
        ("spearman_rho", spearman_rho),
        ("spearman_p", spearman_p),
    ):
        if not math.isfinite(value):
            raise AnalysisSummaryError(
                f"correlate: {label} is non-finite ({value}); "
                f"this indicates a degenerate input"
            )
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
    }


# =============================================================================
# Provenance extraction (from metrics + aggregate)
# =============================================================================


def _required(metadata: Mapping[str, object], key: str, what: str) -> object:
    """Fetch ``key`` from ``metadata`` or raise :class:`AnalysisSummaryError`."""
    if key not in metadata:
        raise AnalysisSummaryError(f"{what}: metadata missing {key!r}")
    return metadata[key]


def extract_config_coordinate(
    metrics: Mapping[str, object],
) -> dict[str, object]:
    """Pull the experiment-coordinate fingerprint inputs from metrics metadata.

    The coordinate is recorded in two equivalent forms in the summary:

    * the raw metadata fields (``checkpoints``, ``layers``, ``concepts``,
      ``probe_classes``, ``base_checkpoint``, ``optional_references``,
      ``protocol``, ``n_samples``, ``source_models``,
      ``humaneval_x_revision``, ``linear_probe``);
    * the SHA-256 fingerprint over their canonical serialization.

    A reader that wants to know "did this summary come from the same
    experiment coordinate as another summary" can compare fingerprints without
    having to recreate the canonical-JSON encoder.
    """
    metadata = metrics.get("metadata")
    if not isinstance(metadata, dict):
        raise AnalysisSummaryError("metrics: missing 'metadata' dict")
    checkpoints = _required(metadata, "checkpoints", "metrics")
    if not isinstance(checkpoints, list):
        raise AnalysisSummaryError("metrics: metadata.checkpoints must be a list")
    layers = _required(metadata, "layers", "metrics")
    if not isinstance(layers, list):
        raise AnalysisSummaryError("metrics: metadata.layers must be a list")
    concepts = _required(metadata, "concepts", "metrics")
    if not isinstance(concepts, dict):
        raise AnalysisSummaryError("metrics: metadata.concepts must be a dict")
    probe_classes = _required(metadata, "probe_classes", "metrics")
    if not isinstance(probe_classes, list):
        raise AnalysisSummaryError("metrics: metadata.probe_classes must be a list")
    base_checkpoint = _required(metadata, "base_checkpoint", "metrics")
    if not isinstance(base_checkpoint, str):
        raise AnalysisSummaryError("metrics: metadata.base_checkpoint must be a string")
    optional_references = _required(metadata, "optional_references", "metrics")
    if not isinstance(optional_references, list):
        raise AnalysisSummaryError(
            "metrics: metadata.optional_references must be a list"
        )
    n_samples = _required(metadata, "n_samples", "metrics")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise AnalysisSummaryError(
            f"metrics: metadata.n_samples must be a non-boolean int, got {n_samples!r}"
        )
    protocol = _required(metadata, "protocol", "metrics")
    if not isinstance(protocol, str):
        raise AnalysisSummaryError("metrics: metadata.protocol must be a string")
    source_models = _required(metadata, "source_models", "metrics")
    if not isinstance(source_models, dict):
        raise AnalysisSummaryError("metrics: metadata.source_models must be a dict")
    humaneval_x_revision = _required(metadata, "humaneval_x_revision", "metrics")
    if not isinstance(humaneval_x_revision, str):
        raise AnalysisSummaryError(
            "metrics: metadata.humaneval_x_revision must be a string"
        )
    linear_probe = _required(metadata, "linear_probe", "metrics")
    if not isinstance(linear_probe, dict):
        raise AnalysisSummaryError("metrics: metadata.linear_probe must be a dict")
    fp = config_fingerprint(
        checkpoints,
        layers,
        concepts,
        probe_classes,
        base_checkpoint,
        optional_references,
        n_samples=n_samples,
        protocol=protocol,
        humaneval_x_revision=humaneval_x_revision,
        source_models=source_models,
        linear_probe=linear_probe,
    )
    return {
        "checkpoints": list(checkpoints),
        "layers": list(layers),
        "concepts": {
            "target": concepts["target"],
            "related": list(concepts["related"]),
            "control": concepts["control"],
        },
        "probe_classes": list(probe_classes),
        "base_checkpoint": base_checkpoint,
        "optional_references": list(optional_references),
        "n_samples": n_samples,
        "protocol": protocol,
        "source_models": dict(source_models),
        "humaneval_x_revision": humaneval_x_revision,
        "linear_probe": dict(linear_probe),
        "fingerprint_sha256": fp,
    }


# =============================================================================
# Summary construction (pure function of metrics + aggregate)
# =============================================================================


def build_summary(
    metrics: Mapping[str, object],
    aggregate: Mapping[str, object],
    *,
    metrics_path: str | os.PathLike[str] | None = None,
    aggregate_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Build the full ``analysis_summary.json`` dict from its two sources.

    The returned dict is fully deterministic: same inputs in, byte-identical
    JSON out (after :func:`canonical_json_bytes` serialization). It is a pure
    function of ``metrics`` and ``aggregate`` plus the
    :mod:`src.rl_zero_experiment` constants.

    Args:
        metrics: parsed ``metrics.json`` object.
        aggregate: parsed ``aggregate_summary.json`` object.
        metrics_path: filesystem path of ``metrics.json``. Used only to compute
            the on-disk SHA-256; pass ``None`` to record an empty hash (the
            validator will then refuse the summary, which is the right
            behaviour for a deserialized/in-memory summary).
        aggregate_path: filesystem path of ``aggregate_summary.json``.

    Returns:
        The summary dict, including a freshly-computed ``build_fingerprint``.

    Raises:
        AnalysisSummaryError: on any structural, coverage, or finiteness
            violation.
    """
    # --- Schema + version ------------------------------------------------
    if metrics.get("schema") != "rl_zero_code_syntax_metrics":
        raise AnalysisSummaryError(
            f"metrics schema is {metrics.get('schema')!r}, "
            "expected 'rl_zero_code_syntax_metrics'"
        )
    src_v = metrics.get("version")
    if isinstance(src_v, bool) or not isinstance(src_v, int):
        raise AnalysisSummaryError(
            f"metrics version must be a non-boolean int, got {src_v!r}"
        )

    # --- Config coordinate (also validates metadata) ---------------------
    coordinate = extract_config_coordinate(metrics)
    coord_ckpts_raw = coordinate["checkpoints"]
    coord_layers_raw = coordinate["layers"]
    coord_concepts_raw = coordinate["concepts"]
    if not isinstance(coord_ckpts_raw, list):
        raise AnalysisSummaryError("config_coordinate.checkpoints must be a list")
    if not isinstance(coord_layers_raw, list):
        raise AnalysisSummaryError("config_coordinate.layers must be a list")
    if not isinstance(coord_concepts_raw, dict):
        raise AnalysisSummaryError("config_coordinate.concepts must be a dict")
    coord_related_raw = coord_concepts_raw.get("related")
    if not isinstance(coord_related_raw, list):
        raise AnalysisSummaryError("config_coordinate.concepts.related must be a list")
    checkpoints: list[str] = [str(c) for c in coord_ckpts_raw]
    layers: list[int] = [int(cast(int, l)) for l in coord_layers_raw]
    related_concepts: list[str] = [str(c) for c in coord_related_raw]
    if len(checkpoints) != 11:
        raise AnalysisSummaryError(
            f"expected exactly 11 checkpoints, got {len(checkpoints)}"
        )
    if len(layers) != 10:
        raise AnalysisSummaryError(f"expected exactly 10 layers, got {len(layers)}")

    # --- Per-checkpoint row aggregation ----------------------------------
    rows: list[dict[str, object]] = []
    for ckpt in checkpoints:
        row: dict[str, object] = {
            "checkpoint": ckpt,
            "m1_mean_cos_vs_main": aggregate_m1_cos_vs_main(metrics, ckpt, layers),
            "m2_mean_related_delta_cos": aggregate_m2_related_delta_cos(
                metrics, ckpt, layers, related_concepts
            ),
            "m2_mean_control_delta_cos": aggregate_m2_control_delta_cos(
                metrics, ckpt, layers
            ),
            "m3_mean_norm_delta": aggregate_m3_norm_delta(metrics, ckpt, layers),
            "m4_syntax_balanced_accuracy": aggregate_m4_syntax_metric(
                metrics, ckpt, layers, field="balanced_accuracy"
            ),
            "m4_syntax_auroc": aggregate_m4_syntax_metric(
                metrics, ckpt, layers, field="auroc"
            ),
            "python_pass_at_1": downstream_value(aggregate, ckpt, "python_pass_at_1"),
            "cpp_pass_at_1": downstream_value(aggregate, ckpt, "cpp_pass_at_1"),
            "mmlu_accuracy": downstream_value(aggregate, ckpt, "mmlu_accuracy"),
        }
        rows.append(row)

    # --- Correlations (Pearson + Spearman) -------------------------------
    correlations: dict[str, dict[str, dict[str, float]]] = {}
    python_p1 = [float(cast(float, r["python_pass_at_1"])) for r in rows]
    mmlu_acc = [float(cast(float, r["mmlu_accuracy"])) for r in rows]
    for metric_key in AGGREGATE_METRIC_KEYS:
        metric_values = [float(cast(float, r[metric_key])) for r in rows]
        correlations[metric_key] = {
            "python_pass_at_1": correlate(metric_values, python_p1),
            "mmlu_accuracy": correlate(metric_values, mmlu_acc),
        }

    # --- Source hashes (over on-disk bytes) ------------------------------
    metrics_hash = sha256_file(metrics_path) if metrics_path is not None else ""
    aggregate_hash = sha256_file(aggregate_path) if aggregate_path is not None else ""

    summary: dict[str, object] = {
        "schema": SCHEMA,
        "version": VERSION,
        "source_metrics_version": src_v,
        "aggregation": dict(AGGREGATION_DESCRIPTION),
        "checkpoints": rows,
        "correlations_all_11": correlations,
        "known_duplicate": {
            "checkpoints": list(DUPLICATE_CHECKPOINTS),
            "note": "identical weights/artifacts; do not treat as independent evidence",
        },
        "limitations": list(LIMITATIONS),
        "integrity": dict(INTEGRITY_NOTES),
        "source_paths": {
            METRICS_RELPATH: str(metrics_path) if metrics_path else "",
            AGGREGATE_RELPATH: str(aggregate_path) if aggregate_path else "",
        },
        "source_hashes": {
            METRICS_RELPATH: metrics_hash,
            AGGREGATE_RELPATH: aggregate_hash,
        },
        "config_coordinate": coordinate,
    }

    # --- Build fingerprint (canonical SHA-256 over content) -------------
    summary["build_fingerprint"] = compute_build_fingerprint(summary)
    return summary


# =============================================================================
# Atomic write
# =============================================================================


def write_summary_atomically(
    path: str | os.PathLike[str],
    summary: Mapping[str, object],
) -> str:
    """Write ``summary`` to ``path`` atomically.

    Uses ``tempfile.mkstemp`` in the destination directory, ``fsync`` of the
    file descriptor, then ``os.replace`` so:

    * a crash midway through the write cannot leave a half-written summary --
      the destination either stays at its previous state or is replaced
      wholesale;
    * concurrent readers either see the old file or the new file, never an
      in-between state.

    The temp file is created in the same directory as the destination so
    ``os.replace`` is atomic on POSIX (same filesystem). On any exception the
    temp file is unlinked.

    The on-disk serialization uses ``indent=2`` (for human readability) with
    ``sort_keys=False`` (we want the documented top-level key order to be
    preserved) and a trailing newline. The deterministic build fingerprint is
    computed over the canonical-bytes form, not this pretty-printed form.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                dict(summary),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return str(p)


# =============================================================================
# Strict validator
# =============================================================================


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisSummaryError(message)


def _check_number(value: object, what: str, *, finite: bool = True) -> float:
    """Validate ``value`` is a non-boolean finite number and return it as float.

    Centralizes the validator's numeric narrowing so the type system understands
    that the value is a ``float`` after the check passes.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisSummaryError(f"{what} must be a number, got {value!r}")
    f = float(value)
    if finite and not math.isfinite(f):
        raise AnalysisSummaryError(f"{what} is non-finite: {f}")
    return f


def validate_summary(
    summary: Mapping[str, object],
    *,
    metrics_path: str | os.PathLike[str] | None = None,
    aggregate_path: str | os.PathLike[str] | None = None,
    expected_metrics_schema: str = "rl_zero_code_syntax_metrics",
    expected_metrics_version: int = 2,
) -> None:
    """Strictly validate an ``analysis_summary.json`` dict.

    Verifies, in order:

    * schema/version identity;
    * the recorded source-metrics version;
    * all six aggregation keys are present in every checkpoint row;
    * downstream row fields (incl. ``cpp_pass_at_1``) are present;
    * the duplicate-weight disclosure covers exactly
      :data:`DUPLICATE_CHECKPOINTS`;
    * every :data:`LIMITATIONS` string is preserved verbatim;
    * the correlation block covers all six metrics x both downstream targets
      with finite Pearson/Spearman values;
    * the source-hash block records both artifacts and, if ``metrics_path`` /
      ``aggregate_path`` are supplied, the on-disk SHA-256 must match exactly
      (:class:`SourceHashMismatch` if not, :class:`MissingSource` if the file
      is gone);
    * the config-coordinate fingerprint is internally consistent (recomputing
      it from the recorded coordinate fields matches the recorded
      ``fingerprint_sha256``);
    * the build fingerprint matches a fresh computation over the summary's
      content (:class:`SummaryTamper` if not).

    Raises:
        AnalysisSummaryError: (or a subclass) on the first violation.
    """
    # --- Schema + version ------------------------------------------------
    _check(
        summary.get("schema") == SCHEMA,
        f"summary schema is {summary.get('schema')!r}, expected {SCHEMA!r}",
    )
    v = summary.get("version")
    _check(
        not isinstance(v, bool) and isinstance(v, int) and v == VERSION,
        f"summary version must be {VERSION}, got {v!r}",
    )
    src_v = summary.get("source_metrics_version")
    _check(
        not isinstance(src_v, bool)
        and isinstance(src_v, int)
        and src_v == expected_metrics_version,
        f"source_metrics_version must be {expected_metrics_version}, got {src_v!r}",
    )

    # --- Aggregation semantics block ------------------------------------
    aggregation = summary.get("aggregation")
    _check(isinstance(aggregation, dict), "summary missing 'aggregation' dict")
    for key, expected_text in AGGREGATION_DESCRIPTION.items():
        actual = aggregation.get(key) if isinstance(aggregation, dict) else None
        _check(
            actual == expected_text,
            f"aggregation.{key} mismatch: {actual!r} != {expected_text!r}",
        )

    # --- Checkpoint rows -------------------------------------------------
    rows_raw = summary.get("checkpoints")
    _check(isinstance(rows_raw, list), "summary 'checkpoints' must be a list")
    _check(
        isinstance(rows_raw, list) and len(rows_raw) == 11,
        f"expected 11 checkpoint rows, got "
        f"{len(rows_raw) if isinstance(rows_raw, list) else 'non-list'}",
    )
    rows = cast(list[object], rows_raw)
    row_keys = (
        set(AGGREGATE_METRIC_KEYS)
        | set(DOWNSTREAM_TARGET_KEYS)
        | set(ROW_ONLY_FIELDS)
        | {"checkpoint"}
    )
    for i, row in enumerate(rows):
        _check(isinstance(row, dict), f"row {i} is not a dict")
        assert isinstance(row, dict)  # for mypy
        actual_keys = set(row.keys())
        missing = row_keys - actual_keys
        _check(
            not missing,
            f"row {i} ({row.get('checkpoint', '?')!r}) missing keys: {sorted(missing)}",
        )
        for metric_key in AGGREGATE_METRIC_KEYS:
            _check_number(row.get(metric_key), f"row {i}/{metric_key}")
        for tgt in DOWNSTREAM_TARGET_KEYS:
            _check_number(row.get(tgt), f"row {i}/{tgt}")
        # cpp_pass_at_1 is present (and under the raw protocol is constant 0).
        _check_number(row.get("cpp_pass_at_1"), f"row {i}/cpp_pass_at_1")

    # --- Duplicate disclosure --------------------------------------------
    dup = summary.get("known_duplicate")
    _check(isinstance(dup, dict), "summary missing 'known_duplicate' dict")
    dup_ckpts = dup.get("checkpoints") if isinstance(dup, dict) else None
    _check(
        isinstance(dup_ckpts, list) and tuple(dup_ckpts) == DUPLICATE_CHECKPOINTS,  # type: ignore[arg-type]
        f"known_duplicate.checkpoints must be {list(DUPLICATE_CHECKPOINTS)}, "
        f"got {dup_ckpts!r}",
    )

    # --- Limitations (verbatim) ------------------------------------------
    limitations = summary.get("limitations")
    _check(isinstance(limitations, list), "summary 'limitations' must be a list")
    assert isinstance(limitations, list)  # for mypy
    _check(
        len(limitations) == len(LIMITATIONS),
        f"expected {len(LIMITATIONS)} limitations, got {len(limitations)}",
    )
    for i, expected_text in enumerate(LIMITATIONS):
        _check(
            limitations[i] == expected_text,
            f"limitations[{i}] tampered: {limitations[i]!r} != {expected_text!r}",
        )

    # --- Correlations ----------------------------------------------------
    correlations = summary.get("correlations_all_11")
    _check(
        isinstance(correlations, dict),
        "summary 'correlations_all_11' must be a dict",
    )
    assert isinstance(correlations, dict)  # for mypy
    for metric_key in AGGREGATE_METRIC_KEYS:
        block = correlations.get(metric_key)
        _check(
            isinstance(block, dict),
            f"correlations missing metric {metric_key!r}",
        )
        assert isinstance(block, dict)  # for mypy
        for tgt in DOWNSTREAM_TARGET_KEYS:
            entry = block.get(tgt)
            _check(
                isinstance(entry, dict),
                f"correlations[{metric_key!r}] missing target {tgt!r}",
            )
            assert isinstance(entry, dict)  # for mypy
            stat_values: dict[str, float] = {}
            for stat_key in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p"):
                stat_values[stat_key] = _check_number(
                    entry.get(stat_key),
                    f"correlations[{metric_key!r}][{tgt!r}].{stat_key}",
                )
            pr = stat_values["pearson_r"]
            sr = stat_values["spearman_rho"]
            pp = stat_values["pearson_p"]
            sp = stat_values["spearman_p"]
            _check(
                -1.0 <= pr <= 1.0,
                f"correlations[{metric_key!r}][{tgt!r}].pearson_r out of [-1,1]: {pr}",
            )
            _check(
                -1.0 <= sr <= 1.0,
                f"correlations[{metric_key!r}][{tgt!r}].spearman_rho out of [-1,1]: {sr}",
            )
            _check(
                0.0 <= pp <= 1.0,
                f"correlations[{metric_key!r}][{tgt!r}].pearson_p out of [0,1]: {pp}",
            )
            _check(
                0.0 <= sp <= 1.0,
                f"correlations[{metric_key!r}][{tgt!r}].spearman_p out of [0,1]: {sp}",
            )

    # --- Integrity notes (informational, must be present) ----------------
    integrity = summary.get("integrity")
    _check(isinstance(integrity, dict), "summary missing 'integrity' dict")
    for k, expected_text in INTEGRITY_NOTES.items():
        actual = integrity.get(k) if isinstance(integrity, dict) else None
        _check(
            actual == expected_text,
            f"integrity.{k} mismatch: {actual!r} != {expected_text!r}",
        )

    # --- Source hashes (live re-hash when paths are provided) ------------
    source_hashes = summary.get("source_hashes")
    _check(
        isinstance(source_hashes, dict),
        "summary 'source_hashes' must be a dict",
    )
    assert isinstance(source_hashes, dict)  # for mypy
    for relpath in (METRICS_RELPATH, AGGREGATE_RELPATH):
        _check(
            relpath in source_hashes,
            f"source_hashes missing {relpath!r}",
        )
        recorded = source_hashes.get(relpath)
        if not isinstance(recorded, str) or len(recorded) != 64:
            raise AnalysisSummaryError(
                f"source_hashes[{relpath!r}] must be a 64-char hex SHA-256, got {recorded!r}"
            )
        # Hex decode check.
        try:
            int(recorded, 16)
        except (ValueError, TypeError):
            raise AnalysisSummaryError(
                f"source_hashes[{relpath!r}] is not a valid hex SHA-256: {recorded!r}"
            )

    # Cross-check against on-disk artifacts when paths are given.
    if metrics_path is not None:
        p = Path(metrics_path)
        if not p.exists():
            raise MissingSource(f"metrics source missing during validation: {p}")
        actual = sha256_file(p)
        recorded = source_hashes[METRICS_RELPATH]
        if actual != recorded:
            raise SourceHashMismatch(
                f"metrics.json hash changed: recorded {recorded}, on-disk {actual}"
            )
    if aggregate_path is not None:
        p = Path(aggregate_path)
        if not p.exists():
            raise MissingSource(f"aggregate source missing during validation: {p}")
        actual = sha256_file(p)
        recorded = source_hashes[AGGREGATE_RELPATH]
        if actual != recorded:
            raise SourceHashMismatch(
                f"aggregate_summary.json hash changed: recorded {recorded}, "
                f"on-disk {actual}"
            )

    # --- Config coordinate (recompute fingerprint internally) -----------
    coordinate = summary.get("config_coordinate")
    _check(
        isinstance(coordinate, dict),
        "summary missing 'config_coordinate' dict",
    )
    assert isinstance(coordinate, dict)  # for mypy
    # Recorded fingerprint must be present and well-formed.
    recorded_cfg_fp = coordinate.get("fingerprint_sha256")
    _check(
        isinstance(recorded_cfg_fp, str) and len(recorded_cfg_fp) == 64,
        f"config_coordinate.fingerprint_sha256 invalid: {recorded_cfg_fp!r}",
    )
    # Recompute fingerprint from the recorded coordinate fields; they must
    # agree (a tampered coordinate that doesn't match its own fingerprint is
    # rejected).
    recomputed = config_fingerprint(
        coordinate["checkpoints"],  # type: ignore[index]
        coordinate["layers"],  # type: ignore[index]
        coordinate["concepts"],  # type: ignore[index]
        coordinate["probe_classes"],  # type: ignore[index]
        coordinate["base_checkpoint"],  # type: ignore[index]
        coordinate["optional_references"],  # type: ignore[index]
        n_samples=int(coordinate["n_samples"]),  # type: ignore[index]
        protocol=str(coordinate["protocol"]),  # type: ignore[index]
        humaneval_x_revision=str(coordinate["humaneval_x_revision"]),  # type: ignore[index]
        source_models=coordinate["source_models"],  # type: ignore[index]
        linear_probe=coordinate["linear_probe"],  # type: ignore[index]
    )
    _check(
        recomputed == recorded_cfg_fp,
        f"config_coordinate fingerprint mismatch: recorded {recorded_cfg_fp}, "
        f"recomputed {recomputed}",
    )
    # The coordinate must also match the live experiment constants: a summary
    # describing a *different* experiment is rejected even if its fingerprint
    # is internally consistent.
    _check(
        coordinate["checkpoints"] == list(EXPERIMENT_CHECKPOINTS),
        "config_coordinate.checkpoints does not match the live experiment",
    )
    _check(
        coordinate["layers"] == list(EXPERIMENT_LAYERS),
        "config_coordinate.layers does not match the live experiment",
    )
    _check(
        coordinate["concepts"]["target"] == TARGET_CONCEPT,
        "config_coordinate.concepts.target does not match the live experiment",
    )
    _check(
        coordinate["concepts"]["related"] == list(RELATED_CONCEPTS),
        "config_coordinate.concepts.related does not match the live experiment",
    )
    _check(
        coordinate["concepts"]["control"] == CONTROL_CONCEPT,
        "config_coordinate.concepts.control does not match the live experiment",
    )
    _check(
        coordinate["probe_classes"] == list(PROBE_CLASSES),
        "config_coordinate.probe_classes does not match the live experiment",
    )
    _check(
        coordinate["base_checkpoint"] == BASE_CHECKPOINT,
        "config_coordinate.base_checkpoint does not match the live experiment",
    )

    # --- Build fingerprint (tamper detection) ---------------------------
    recorded_fp = summary.get("build_fingerprint")
    _check(
        isinstance(recorded_fp, str) and len(recorded_fp) == 64,
        f"build_fingerprint invalid: {recorded_fp!r}",
    )
    recomputed_build_fp = compute_build_fingerprint(summary)
    if recomputed_build_fp != recorded_fp:
        raise SummaryTamper(
            f"build fingerprint mismatch: recorded {recorded_fp}, "
            f"recomputed {recomputed_build_fp}"
        )


def validate_summary_file(
    path: str | os.PathLike[str],
    *,
    metrics_path: str | os.PathLike[str] | None = None,
    aggregate_path: str | os.PathLike[str] | None = None,
) -> None:
    """Load ``path`` as JSON and pass it to :func:`validate_summary`."""
    p = Path(path)
    if not p.exists():
        raise MissingSource(f"analysis_summary.json not found: {p}")
    try:
        summary = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise AnalysisSummaryError(f"invalid JSON in {p}: {e}") from e
    if not isinstance(summary, dict):
        raise AnalysisSummaryError(
            f"{p}: expected a JSON object, got {type(summary).__name__}"
        )
    validate_summary(summary, metrics_path=metrics_path, aggregate_path=aggregate_path)


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    "AGGREGATE_METRIC_KEYS",
    "AGGREGATION_DESCRIPTION",
    "AnalysisSummaryError",
    "BASE_CHECKPOINT",
    "CONTROL_CONCEPT",
    "DEFAULT_AGGREGATE_PATH",
    "DEFAULT_METRICS_PATH",
    "DEFAULT_SUMMARY_PATH",
    "DOWNSTREAM_TARGET_KEYS",
    "DUPLICATE_CHECKPOINTS",
    "EXPERIMENT_CHECKPOINTS",
    "EXPERIMENT_LAYERS",
    "INTEGRITY_NOTES",
    "LIMITATIONS",
    "METRICS_RELPATH",
    "AGGREGATE_RELPATH",
    "MissingSource",
    "PROBE_CLASSES",
    "RELATED_CONCEPTS",
    "RL_ZERO_CODE_RESULTS_ROOT",
    "ROW_ONLY_FIELDS",
    "SCHEMA",
    "SourceHashMismatch",
    "SummaryTamper",
    "SYNTAX_LABELS",
    "TARGET_CONCEPT",
    "VERSION",
    "aggregate_m1_cos_vs_main",
    "aggregate_m2_control_delta_cos",
    "aggregate_m2_related_delta_cos",
    "aggregate_m3_norm_delta",
    "aggregate_m4_syntax_metric",
    "build_summary",
    "canonical_json_bytes",
    "compute_build_fingerprint",
    "config_fingerprint",
    "correlate",
    "downstream_value",
    "extract_config_coordinate",
    "load_json",
    "sha256_bytes",
    "sha256_file",
    "validate_summary",
    "validate_summary_file",
    "write_summary_atomically",
]
