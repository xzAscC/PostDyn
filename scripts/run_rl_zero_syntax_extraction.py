#!/usr/bin/env python3
"""Resumable raw-text extraction driver for the RL-Zero-Code syntax experiment.

Runs DiM concept-vector extraction (six concepts) and eight-class probe
activation collection (400 records) across **base ``main`` + ten RL-Zero-Code
checkpoints** (11 total), always with ``use_chat_template=False``.

Why a separate driver?
    The legacy :func:`postdyn.concept_dynamics.run_full_experiment` loads its own
    model internally and unloads it per checkpoint; it cannot share that
    loaded model with the probe-activation pass. This driver implements a
    **narrow checkpoint runner** (:func:`run_checkpoint_extraction`) that loads
    the model and tokenizer **once** per checkpoint, serves both the concept
    and probe extraction paths with that single load, then unloads GPU memory
    before moving to the next checkpoint.

What it writes (all under ``logs/rl_zero_code_syntax`` only)
::

    {output_root}/
        concept_vectors/{model_name}/{checkpoint}/layer_{idx}.{safetensors,json}
        activations/records.json
        activations/{model_name}/{checkpoint}/layer_{idx}.{safetensors,json}
        activations/{model_name}/{checkpoint}/manifest.json
        manifests/{model_name}__{checkpoint}.json   (atomic per-checkpoint manifest)
        extraction_summary.json                       (global progress)

Resume contract
    * A checkpoint is skipped entirely when **both** all six concept vectors
      (at every requested layer) and all probe activations are present on
      disk — validated by reading the actual files, not just a flag.
    * Within a checkpoint, individual concepts and individual probe layers
      are skipped when their output files are present and carry the expected
      sample counts.
    * The per-checkpoint manifest is written **atomically** (temp file +
      ``os.replace``) after extraction, recording six concept statuses and
      completed probe layers.

Usage::

    uv run python scripts/run_rl_zero_syntax_extraction.py [OPTIONS]

Options:
    --only base|rl|all     Select base only, RL only, or all 11 (default: all)
    --checkpoints a,b      Comma-separated checkpoint subset (e.g. main,step_100)
    --limit N              Take only the first N selected checkpoints
    --layers L1,L2         Comma-separated layer indices (default: 10 uniform)
    --samples N            Paired records per concept class (default: 50)
    --output DIR           Output root (default: logs/rl_zero_code_syntax)
    --max-seq-len N        Max tokenization length (default: 2048)
    --keep-hf-cache        Do not delete HF cache entries between checkpoints
    --quick                Smoke test: 1 layer, 5 samples, 1 checkpoint

This script runs no real model during tests (the runner accepts injected
mock model/tokenizer loaders), never writes to ``logs/concept_dynamics_multi``,
and never modifies datasets or ``postdyn/config.py``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

# Make ``src`` importable when run directly via ``python scripts/...``.

from postdyn.config import ModelConfig
from postdyn.concept_dynamics import (
    EXPECTED_D_MODEL,
    _clean_hf_cache,
    _load_model_and_tokenizer,
    compute_concept_vector,
    extract_layer_activations,
    is_concept_layer_v1_complete,
    load_concept_vectors,
    save_concept_vectors,
)
from postdyn.probe_activations import (
    PROTOCOL,
    ProbeRecord,
    build_probe_records,
    compute_records_fingerprint,
    is_layer_complete,
    run_extraction_with_resume,
)
from postdyn.rl_zero_experiment import (
    BASE_CHECKPOINT,
    BASE_MODEL,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_CONCEPTS,
    EXPERIMENT_LAYERS,
    N_SAMPLES,
    PRIMARY_USE_CHAT_TEMPLATE,
    PROBE_CLASSES,
    PAIRED_CONCEPT_RESULTS_ROOT,
    PAIRED_CONCEPT_RESULTS_ROOT_QUICK,
    RL_CHECKPOINTS,
    RL_ZERO_CODE_RESULTS_ROOT,
    TARGET_MODEL,
    is_base_checkpoint,
    is_rl_checkpoint,
    results_root,
)


# =============================================================================
# Constants
# =============================================================================

#: Concept-vector subdirectory inside the output root.
CONCEPT_VECTORS_SUBDIR: str = "concept_vectors"

#: Probe-activation subdirectory inside the output root.
ACTIVATIONS_SUBDIR: str = "activations"

#: Per-checkpoint manifest subdirectory inside the output root.
MANIFESTS_SUBDIR: str = "manifests"

#: Global summary filename.
SUMMARY_FILENAME: str = "extraction_summary.json"

#: Legacy results directories that must NEVER receive writes from this driver.
_FORBIDDEN_ROOTS: frozenset[str] = frozenset(
    {
        PAIRED_CONCEPT_RESULTS_ROOT,
        PAIRED_CONCEPT_RESULTS_ROOT_QUICK,
        os.path.join(os.path.dirname(RL_ZERO_CODE_RESULTS_ROOT), "concept_dynamics"),
        os.path.join(
            os.path.dirname(RL_ZERO_CODE_RESULTS_ROOT), "concept_dynamics_paired"
        ),
    }
)

#: Separator used in manifest filenames (both ``/`` in model names and
#: ``/`` in checkpoint names are replaced with ``__``).
_MANIFEST_NAME_SEP: str = "__"

#: Default max sequence length (matches the concept-dynamics default).
DEFAULT_MAX_SEQ_LEN: int = 2048

#: Type alias for the concept-source-provenance map threaded through the
#: extraction pipeline: ``{concept_name: (positive_texts, negative_texts)}``.
ConceptSources = dict[str, tuple[list[str], list[str]]]


# =============================================================================
# Output-root resolution & isolation guard
# =============================================================================


def resolve_output_root(*, quick: bool, override: str | None) -> str:
    """Resolve the output root, honouring the experiment's own :func:`results_root`.

    An explicit ``override`` wins; otherwise the quick/full default from
    :mod:`postdyn.rl_zero_experiment` is used.
    """
    if override is not None:
        return override
    return results_root(quick=quick)


def assert_output_isolated(output_root: str) -> None:
    """Assert the output root does not collide with any legacy results dir.

    Raises ``ValueError`` if the resolved root is equal to or nested inside
    any forbidden (paired-concept) directory, or vice versa.
    """
    norm = os.path.normpath(output_root)
    for forbidden in _FORBIDDEN_ROOTS:
        f = os.path.normpath(forbidden)
        if norm == f:
            raise ValueError(
                f"output root {output_root!r} must not equal legacy dir {forbidden!r}"
            )
        if norm.startswith(f + os.sep):
            raise ValueError(
                f"output root {output_root!r} is nested inside legacy dir {forbidden!r}"
            )
    if "concept_dynamics_multi" in norm:
        raise ValueError(
            f"output root {output_root!r} must not contain 'concept_dynamics_multi'"
        )


# =============================================================================
# Subdirectory helpers
# =============================================================================


def concept_vectors_dir(output_root: str) -> str:
    return os.path.join(output_root, CONCEPT_VECTORS_SUBDIR)


def activations_dir(output_root: str) -> str:
    return os.path.join(output_root, ACTIVATIONS_SUBDIR)


def manifests_dir(output_root: str) -> str:
    return os.path.join(output_root, MANIFESTS_SUBDIR)


def summary_path(output_root: str) -> str:
    return os.path.join(output_root, SUMMARY_FILENAME)


def manifest_file_path(output_root: str, model_name: str, checkpoint: str) -> str:
    """Return the per-checkpoint manifest path."""
    safe = (
        model_name.replace("/", "_") + _MANIFEST_NAME_SEP + checkpoint.replace("/", "_")
    )
    return os.path.join(manifest_dir(output_root), safe + ".json")


def manifest_dir(output_root: str) -> str:
    return manifests_dir(output_root)


# =============================================================================
# Checkpoint selection
# =============================================================================


def select_checkpoints(
    only: str = "all",
    checkpoint_subset: list[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Select the ordered checkpoint schedule based on CLI flags.

    Args:
        only: ``"base"`` (just ``main``), ``"rl"`` (ten RL steps), or
            ``"all"`` (11 checkpoints in training order).
        checkpoint_subset: Optional explicit checkpoint names. When provided,
            the pool is intersected with these names (pool order preserved).
        limit: If positive, take only the first ``limit`` checkpoints.

    Returns:
        Ordered list of checkpoint names from
        :data:`postdyn.rl_zero_experiment.EXPERIMENT_CHECKPOINTS`.

    Raises:
        ValueError: ``only`` is not one of the three valid values.
    """
    if only == "base":
        pool = [BASE_CHECKPOINT]
    elif only == "rl":
        pool = list(RL_CHECKPOINTS)
    elif only == "all":
        pool = list(EXPERIMENT_CHECKPOINTS)
    else:
        raise ValueError(f"--only must be 'base', 'rl', or 'all'; got {only!r}")

    if checkpoint_subset:
        subset_set = set(checkpoint_subset)
        pool = [c for c in pool if c in subset_set]

    if limit is not None and limit > 0:
        pool = pool[:limit]

    return pool


def model_for_checkpoint(checkpoint: str) -> ModelConfig:
    """Return the :class:`ModelConfig` for a checkpoint name.

    ``main`` -> base model; any RL step -> target model.

    Raises:
        ValueError: the checkpoint is neither base nor RL.
    """
    if is_base_checkpoint(checkpoint):
        return BASE_MODEL
    if is_rl_checkpoint(checkpoint):
        return TARGET_MODEL
    raise ValueError(
        f"unknown checkpoint {checkpoint!r}; expected 'main' or one of {RL_CHECKPOINTS}"
    )


# =============================================================================
# File-based resume checks
# =============================================================================


def is_concept_layer_complete(
    cvec_dir: str,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
    concept: str,
    n_samples: int,
    *,
    expected_d_model: int | None = None,
    expected_max_seq_len: int | None = None,
    expected_protocol: str | None = None,
    expected_use_chat_template: bool | None = None,
    concept_sources: ConceptSources | None = None,
) -> bool:
    """Return ``True`` iff a concept vector file is present and valid.

    When ``concept_sources`` is provided (and contains ``concept``), the check
    delegates to :func:`is_concept_layer_v1_complete`, which requires a v1
    sidecar with matching source-text provenance.  Legacy v0 sidecars are
    rejected so they are re-extracted rather than silently trusted.

    When ``concept_sources`` is ``None`` or does not contain ``concept``, the
    backward-compatible count-only check applies: the file loads, the concept
    is present, and both class counts meet ``n_samples``.
    """
    if concept_sources is not None and concept in concept_sources:
        sources_for_concept = {concept: concept_sources[concept]}
        return is_concept_layer_v1_complete(
            cvec_dir,
            model_name,
            checkpoint,
            layer_idx,
            concept,
            n_samples,
            expected_d_model=expected_d_model,
            expected_max_seq_len=expected_max_seq_len,
            expected_protocol=expected_protocol,
            expected_use_chat_template=expected_use_chat_template,
            expected_concept_sources=sources_for_concept,
        )

    try:
        vectors = load_concept_vectors(cvec_dir, model_name, layer_idx, checkpoint)
    except FileNotFoundError:
        return False
    except Exception:
        return False
    cv = vectors.get(concept)
    if cv is None:
        return False
    if expected_d_model is not None and cv.d_model != expected_d_model:
        return False
    return cv.n_positive >= n_samples and cv.n_negative >= n_samples


def concept_completed_layers(
    cvec_dir: str,
    model_name: str,
    checkpoint: str,
    concept: str,
    layers: list[int],
    n_samples: int,
    *,
    expected_d_model: int | None = None,
    expected_max_seq_len: int | None = None,
    expected_protocol: str | None = None,
    expected_use_chat_template: bool | None = None,
    concept_sources: ConceptSources | None = None,
) -> list[int]:
    """Return the layers whose concept-vector files are present and valid."""
    return [
        ly
        for ly in layers
        if is_concept_layer_complete(
            cvec_dir,
            model_name,
            checkpoint,
            ly,
            concept,
            n_samples,
            expected_d_model=expected_d_model,
            expected_max_seq_len=expected_max_seq_len,
            expected_protocol=expected_protocol,
            expected_use_chat_template=expected_use_chat_template,
            concept_sources=concept_sources,
        )
    ]


def probe_completed_layers(
    acts_dir: str,
    model_name: str,
    checkpoint: str,
    layers: list[int],
    n_records: int,
    *,
    expected_max_seq_len: int | None = None,
    expected_records_fingerprint: str | None = None,
    expected_d_model: int | None = None,
    expected_protocol: str | None = None,
    probe_records: list[ProbeRecord] | None = None,
) -> list[int]:
    """Return the probe layers whose activation files are present and valid.

    When ``probe_records`` is provided, each layer is validated with
    ``is_layer_complete(expected_records=...)`` so the sidecar's per-record
    text/source provenance (``text_sha256``, ordered ``source_ids``, ...) is
    recomputed from the records and any absent or mismatched provenance causes
    the layer to be treated as incomplete. This is what prevents old migrated
    sidecars (which carry a fingerprint + max_seq_len but no ``text_sha256``)
    from being skipped as complete.
    """
    return [
        ly
        for ly in layers
        if is_layer_complete(
            acts_dir,
            model_name,
            checkpoint,
            ly,
            n_records,
            expected_model_name=model_name,
            expected_checkpoint=checkpoint,
            expected_layer_idx=ly,
            expected_max_seq_len=expected_max_seq_len,
            expected_records_fingerprint=expected_records_fingerprint,
            expected_d_model=expected_d_model,
            expected_protocol=expected_protocol,
            expected_records=probe_records,
        )
    ]


# =============================================================================
# Atomic JSON helper
# =============================================================================


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically via a temp file + ``os.replace``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# =============================================================================
# Atomic per-checkpoint manifest
# =============================================================================


def build_checkpoint_manifest(
    output_root: str,
    model_name: str,
    checkpoint: str,
    revision: str,
    hf_id: str,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    n_probe_records: int,
    *,
    max_seq_len: int | None = None,
    records_fingerprint: str | None = None,
    expected_d_model: int | None = None,
    probe_records: list[ProbeRecord] | None = None,
    concept_sources: ConceptSources | None = None,
    use_chat_template: bool | None = None,
) -> dict[str, Any]:
    """Build the per-checkpoint manifest dict by re-validating actual files.

    The manifest records, for each of the six concepts, which layers are
    complete and whether the concept is fully covered, plus the probe
    activation layer coverage. File checks are authoritative — a concept is
    marked complete only when its safetensors + JSON exist at every layer.

    When ``probe_records`` is provided, probe layer completeness additionally
    re-derives the per-record text/source provenance from those records, so old
    sidecars lacking ``text_sha256`` are reported incomplete.

    When ``concept_sources`` is provided, concept layer completeness uses
    :func:`is_concept_layer_v1_complete`, rejecting legacy v0 sidecars.
    """
    cvec_dir = concept_vectors_dir(output_root)
    acts_dir = activations_dir(output_root)

    concept_entries: dict[str, dict[str, Any]] = {}
    for concept in concepts:
        done = concept_completed_layers(
            cvec_dir,
            model_name,
            checkpoint,
            concept,
            layers,
            n_samples,
            expected_d_model=expected_d_model,
            expected_max_seq_len=max_seq_len,
            expected_protocol=PROTOCOL,
            expected_use_chat_template=(
                use_chat_template if use_chat_template is not None else False
            ),
            concept_sources=concept_sources,
        )
        concept_entries[concept] = {
            "complete": len(done) == len(layers),
            "layers": done,
            "n_samples": n_samples,
        }

    probe_done = probe_completed_layers(
        acts_dir,
        model_name,
        checkpoint,
        layers,
        n_probe_records,
        expected_max_seq_len=max_seq_len,
        expected_records_fingerprint=records_fingerprint,
        expected_d_model=expected_d_model,
        expected_protocol=PROTOCOL,
        probe_records=probe_records,
    )

    all_concepts_done = all(e["complete"] for e in concept_entries.values())
    all_probes_done = len(probe_done) == len(layers)

    return {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "revision": revision,
        "hf_id": hf_id,
        "protocol": PROTOCOL,
        "use_chat_template": (
            use_chat_template if use_chat_template is not None else False
        ),
        "max_seq_len": max_seq_len,
        "records_fingerprint": records_fingerprint,
        "expected_d_model": expected_d_model,
        "n_samples": n_samples,
        "layers": list(layers),
        "concepts": concept_entries,
        "probe_activations": {
            "complete": all_probes_done,
            "completed_layers": probe_done,
            "n_records": n_probe_records,
        },
        "complete": all_concepts_done and all_probes_done,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_checkpoint_manifest(
    output_root: str,
    model_name: str,
    checkpoint: str,
    revision: str,
    hf_id: str,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    n_probe_records: int,
    *,
    max_seq_len: int | None = None,
    records_fingerprint: str | None = None,
    expected_d_model: int | None = None,
    probe_records: list[ProbeRecord] | None = None,
    concept_sources: ConceptSources | None = None,
    use_chat_template: bool | None = None,
) -> dict[str, Any]:
    """Build and atomically write the per-checkpoint manifest."""
    manifest = build_checkpoint_manifest(
        output_root,
        model_name,
        checkpoint,
        revision,
        hf_id,
        concepts,
        layers,
        n_samples,
        n_probe_records,
        max_seq_len=max_seq_len,
        records_fingerprint=records_fingerprint,
        expected_d_model=expected_d_model,
        probe_records=probe_records,
        concept_sources=concept_sources,
        use_chat_template=use_chat_template,
    )
    path = manifest_file_path(output_root, model_name, checkpoint)
    _atomic_write_json(path, manifest)
    return manifest


def load_checkpoint_manifest(
    output_root: str, model_name: str, checkpoint: str
) -> dict[str, Any]:
    """Load the per-checkpoint manifest (empty dict if absent or corrupt)."""
    path = manifest_file_path(output_root, model_name, checkpoint)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


# =============================================================================
# Concept-vector extraction (narrow, reusing lower-level APIs)
# =============================================================================


def _extract_concept_vectors(
    model: Any,
    tokenizer: Any,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    cvec_dir: str,
    model_name: str,
    checkpoint: str,
    max_seq_len: int,
    *,
    concept_texts_fn: Callable[[str, int], tuple[list[str], list[str]]] | None = None,
    concept_sources: ConceptSources | None = None,
    revision: str | None = None,
    hf_id: str | None = None,
    expected_d_model: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract missing concept vectors for one checkpoint.

    For each concept, determines which layers are missing, loads the paired
    texts, extracts last-token activations (raw text, no chat template),
    computes the DiM direction, and saves it (merged with any existing
    vectors at that layer) with v1 source provenance.

    Returns ``{concept: {"extracted": [...], "skipped": [...]}}``.
    """
    import torch

    if concept_texts_fn is None:
        from postdyn.contrastive_datasets import load_contrastive_texts as concept_texts_fn

    results: dict[str, dict[str, Any]] = {}

    for concept in concepts:
        missing = [
            ly
            for ly in layers
            if not is_concept_layer_complete(
                cvec_dir,
                model_name,
                checkpoint,
                ly,
                concept,
                n_samples,
                expected_d_model=expected_d_model,
                expected_max_seq_len=max_seq_len,
                expected_protocol=PROTOCOL,
                expected_use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
                concept_sources=concept_sources,
            )
        ]
        done = [ly for ly in layers if ly not in missing]

        if not missing:
            results[concept] = {"extracted": [], "skipped": done}
            continue

        print(f"    Concept '{concept}': extracting layers {missing}")

        pos_texts, neg_texts = concept_texts_fn(concept, n_samples)

        pos_acts = extract_layer_activations(
            model,
            tokenizer,
            pos_texts,
            missing,
            max_seq_len=max_seq_len,
            use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
        )
        neg_acts = extract_layer_activations(
            model,
            tokenizer,
            neg_texts,
            missing,
            max_seq_len=max_seq_len,
            use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
        )

        for ly in missing:
            cv = compute_concept_vector(
                pos_acts[ly],
                neg_acts[ly],
                concept_name=concept,
                model_name=model_name,
                layer_idx=ly,
                normalize=True,
            )
            try:
                existing = load_concept_vectors(cvec_dir, model_name, ly, checkpoint)
            except FileNotFoundError:
                existing = {}
            existing[concept] = cv

            merged_sources: ConceptSources | None = None
            if concept_sources is not None:
                merged_sources = dict(concept_sources)
                merged_sources[concept] = (pos_texts, neg_texts)
            else:
                merged_sources = {concept: (pos_texts, neg_texts)}

            save_concept_vectors(
                existing,
                cvec_dir,
                model_name,
                ly,
                checkpoint,
                protocol=PROTOCOL,
                revision=revision if revision else checkpoint,
                hf_id=hf_id,
                max_seq_len=max_seq_len,
                use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
                concept_sources=merged_sources,
            )

        del pos_acts, neg_acts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results[concept] = {"extracted": missing, "skipped": done}

    return results


# =============================================================================
# Narrow checkpoint runner (single model load for concept + probe)
# =============================================================================


def _is_checkpoint_complete(
    output_root: str,
    model_name: str,
    checkpoint: str,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    n_probe_records: int,
    *,
    max_seq_len: int | None = None,
    records_fingerprint: str | None = None,
    expected_d_model: int | None = None,
    probe_records: list[ProbeRecord] | None = None,
    concept_sources: ConceptSources | None = None,
) -> bool:
    """Return ``True`` if every concept layer AND probe layer is on disk.

    When ``probe_records`` is provided, probe layers are validated with full
    identity re-derivation (``expected_records``), so old sidecars lacking
    ``text_sha256`` are treated as incomplete and the checkpoint is re-run
    instead of skipped.

    When ``concept_sources`` is provided, concept layers are validated with
    :func:`is_concept_layer_v1_complete`, so legacy v0 concept sidecars are
    treated as incomplete and the checkpoint is re-run.
    """
    cvec_dir = concept_vectors_dir(output_root)
    acts_dir = activations_dir(output_root)

    for concept in concepts:
        done = concept_completed_layers(
            cvec_dir,
            model_name,
            checkpoint,
            concept,
            layers,
            n_samples,
            expected_d_model=expected_d_model,
            expected_max_seq_len=max_seq_len,
            expected_protocol=PROTOCOL,
            expected_use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
            concept_sources=concept_sources,
        )
        if len(done) != len(layers):
            return False

    probe_done = probe_completed_layers(
        acts_dir,
        model_name,
        checkpoint,
        layers,
        n_probe_records,
        expected_max_seq_len=max_seq_len,
        expected_records_fingerprint=records_fingerprint,
        expected_d_model=expected_d_model,
        expected_protocol=PROTOCOL,
        probe_records=probe_records,
    )
    return len(probe_done) == len(layers)


def run_checkpoint_extraction(
    checkpoint: str,
    *,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    output_root: str,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    model_loader: Callable[[ModelConfig, str], tuple[Any, Any]] | None = None,
    concept_texts_fn: Callable[[str, int], tuple[list[str], list[str]]] | None = None,
    probe_records_fn: Callable[[], list[ProbeRecord]] | None = None,
    unload_gpu: bool = True,
    expected_d_model: int | None = None,
) -> dict[str, Any]:
    """Extract concept vectors + probe activations for ONE checkpoint.

    This is the **narrow checkpoint runner**: it loads the model and tokenizer
    **once**, serves both the six-concept DiM extraction and the 400-record
    probe activation pass with that single load, then unloads GPU memory.

    Resume:
        Before loading any model, the runner checks every concept layer and
        every probe layer on disk. If all are present and valid, the model is
        not loaded at all. Within a checkpoint, individual concepts/layers
        are skipped when their files exist.

    Args:
        checkpoint: Checkpoint name (``"main"`` or ``"step_100"`` etc.).
        concepts: Six experiment concept keys.
        layers: Layer indices to extract at.
        n_samples: Paired records per concept class.
        output_root: Isolated results root.
        max_seq_len: Max tokenization length for both extraction paths.
        model_loader: Injectable ``(ModelConfig, revision) -> (model, tokenizer)``.
            Defaults to :func:`postdyn.concept_dynamics._load_model_and_tokenizer`.
        concept_texts_fn: Injectable ``(concept, n) -> (pos_texts, neg_texts)``.
            Defaults to :func:`postdyn.contrastive_datasets.load_contrastive_texts`.
        probe_records_fn: Injectable ``() -> list[ProbeRecord]``.
            Defaults to :func:`postdyn.probe_activations.build_probe_records`.
        unload_gpu: If ``True`` (default), call ``gc.collect`` and
            ``torch.cuda.empty_cache`` after extraction.

    Returns:
        Summary dict with per-concept extraction stats, probe stats,
        model metadata, and elapsed time.
    """
    import torch

    model_config = model_for_checkpoint(checkpoint)
    model_name = model_config.name
    revision = checkpoint
    hf_id = model_config.hf_id

    cvec_dir = concept_vectors_dir(output_root)
    acts_dir = activations_dir(output_root)
    os.makedirs(cvec_dir, exist_ok=True)
    os.makedirs(acts_dir, exist_ok=True)

    # Build / load probe records (needed for n_records count + fingerprint).
    if probe_records_fn is None:
        probe_records_fn = build_probe_records
    probe_records = probe_records_fn()
    n_probe_records = len(probe_records)
    records_fingerprint = compute_records_fingerprint(probe_records)

    # Build concept source provenance once so strict v1 checks can bind it.
    if concept_texts_fn is None:
        from postdyn.contrastive_datasets import load_contrastive_texts as concept_texts_fn

    concept_sources: ConceptSources = {}
    for concept in concepts:
        pos_texts, neg_texts = concept_texts_fn(concept, n_samples)
        concept_sources[concept] = (pos_texts, neg_texts)

    expected_d_model_resolved = expected_d_model

    start = time.time()

    print(f"\n{'=' * 60}")
    print(f"Checkpoint: {model_name}/{checkpoint} (hf_id={hf_id}, rev={revision})")
    print(f"  Concepts: {len(concepts)}, Layers: {len(layers)}, Samples: {n_samples}")
    print(f"  Probe records: {n_probe_records}")
    print(f"  Max seq len: {max_seq_len}")
    print(f"  Records fingerprint: {records_fingerprint[:16]}...")
    print(f"  Expected d_model: {expected_d_model_resolved}")
    print(f"  Chat template: off (raw text)")
    print(f"{'=' * 60}")

    # --- Resume: check if everything is already on disk -------------------
    if _is_checkpoint_complete(
        output_root,
        model_name,
        checkpoint,
        concepts,
        layers,
        n_samples,
        n_probe_records,
        max_seq_len=max_seq_len,
        records_fingerprint=records_fingerprint,
        expected_d_model=expected_d_model_resolved,
        probe_records=probe_records,
        concept_sources=concept_sources,
    ):
        print(f"  SKIP: all concept vectors and probe layers already on disk.")
        concept_stats = {
            c: {"extracted": [], "skipped": list(layers)} for c in concepts
        }
        probe_stats = {
            "extracted": [],
            "skipped": list(layers),
            "n_records": n_probe_records,
        }
        elapsed = time.time() - start
        manifest = write_checkpoint_manifest(
            output_root,
            model_name,
            checkpoint,
            revision,
            hf_id,
            concepts,
            layers,
            n_samples,
            n_probe_records,
            max_seq_len=max_seq_len,
            records_fingerprint=records_fingerprint,
            expected_d_model=expected_d_model_resolved,
            probe_records=probe_records,
            concept_sources=concept_sources,
            use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
        )
        return {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "revision": revision,
            "hf_id": hf_id,
            "concepts": concept_stats,
            "probe_activations": probe_stats,
            "model_loaded": False,
            "max_seq_len": max_seq_len,
            "records_fingerprint": records_fingerprint,
            "elapsed_seconds": round(elapsed, 1),
            "manifest": manifest,
        }

    # --- Load model + tokenizer ONCE -------------------------------------
    loader = model_loader or _load_model_and_tokenizer
    print(f"  Loading model {hf_id} rev={revision} (bfloat16, device_map=auto)...")
    model, tokenizer = loader(model_config, revision)
    if hasattr(model, "eval"):
        model.eval()
    device = "cpu"
    try:
        device = str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        pass
    print(f"  Model loaded on {device}")

    try:
        # --- Concept vectors --------------------------------------------------
        concept_stats = _extract_concept_vectors(
            model,
            tokenizer,
            concepts,
            layers,
            n_samples,
            cvec_dir,
            model_name,
            checkpoint,
            max_seq_len,
            concept_texts_fn=concept_texts_fn,
            concept_sources=concept_sources,
            revision=revision,
            hf_id=hf_id,
            expected_d_model=expected_d_model_resolved,
        )

        # --- Probe activations (resumes internally) ---------------------------
        probe_result = run_extraction_with_resume(
            probe_records,
            model,
            tokenizer,
            layers,
            acts_dir,
            model_name,
            checkpoint,
            max_seq_len=max_seq_len,
            expected_d_model=expected_d_model_resolved,
        )
        probe_stats = {
            "extracted": probe_result["extracted"],
            "skipped": probe_result["skipped"],
            "n_records": n_probe_records,
        }
    finally:
        # --- Unload GPU memory -----------------------------------------------
        del model
        del tokenizer
        gc.collect()
        if unload_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    elapsed = time.time() - start
    print(f"  Done: {model_name}/{checkpoint} in {elapsed:.1f}s")

    # --- Write atomic manifest (re-validates files) -----------------------
    manifest = write_checkpoint_manifest(
        output_root,
        model_name,
        checkpoint,
        revision,
        hf_id,
        concepts,
        layers,
        n_samples,
        n_probe_records,
        max_seq_len=max_seq_len,
        records_fingerprint=records_fingerprint,
        expected_d_model=expected_d_model_resolved,
        probe_records=probe_records,
        concept_sources=concept_sources,
        use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
    )

    return {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "revision": revision,
        "hf_id": hf_id,
        "concepts": concept_stats,
        "probe_activations": probe_stats,
        "model_loaded": True,
        "max_seq_len": max_seq_len,
        "records_fingerprint": records_fingerprint,
        "elapsed_seconds": round(elapsed, 1),
        "manifest": manifest,
    }


# =============================================================================
# Global summary
# =============================================================================


def write_summary(
    output_root: str,
    checkpoints: list[str],
    per_checkpoint: dict[str, dict[str, Any]],
    *,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
) -> str:
    """Write the global ``extraction_summary.json`` atomically.

    Called after each checkpoint so the summary is always up to date and
    the driver can be safely interrupted/resumed.
    """
    path = summary_path(output_root)
    n_complete = sum(
        1
        for stats in per_checkpoint.values()
        if isinstance(stats, dict) and stats.get("manifest", {}).get("complete") is True
    )
    n_errors = sum(
        1
        for stats in per_checkpoint.values()
        if isinstance(stats, dict) and "error" in stats
    )
    payload = {
        "protocol": PROTOCOL,
        "use_chat_template": PRIMARY_USE_CHAT_TEMPLATE,
        "output_root": output_root,
        "concepts": concepts,
        "layers": layers,
        "n_samples": n_samples,
        "checkpoints_requested": checkpoints,
        "checkpoints_complete": n_complete,
        "checkpoints_errors": n_errors,
        "per_checkpoint": per_checkpoint,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(path, payload)
    return path


# =============================================================================
# CLI
# =============================================================================


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable raw-text extraction for the RL-Zero-Code syntax "
            "experiment: six DiM concept vectors + 400 probe activations "
            "across base main + 10 RL-Zero-Code checkpoints."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=["base", "rl", "all"],
        default="all",
        help="Select base (main only), RL (10 steps), or all 11 (default: all).",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma-separated checkpoint subset (e.g. 'main,step_100').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Take only the first N selected checkpoints.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: 10 uniform from 32).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help=f"Paired records per concept class (default: {N_SAMPLES}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(f"Output root (default: {RL_ZERO_CODE_RESULTS_ROOT})"),
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Max tokenization length (default: {DEFAULT_MAX_SEQ_LEN}).",
    )
    parser.add_argument(
        "--keep-hf-cache",
        action="store_true",
        help="Do not delete Hugging Face cache entries between checkpoints.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 1 layer, 5 samples, 1 checkpoint (base main).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Resolve parameters ----------------------------------------------
    output_root = resolve_output_root(quick=args.quick, override=args.output)
    try:
        assert_output_isolated(output_root)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.quick:
        concepts = list(EXPERIMENT_CONCEPTS)
        layers = [3]
        n_samples = 5
        checkpoints = select_checkpoints(only="base", limit=1)
        max_seq_len = 512
        print("=" * 60)
        print("QUICK MODE (smoke: 1 base ckpt x 6 concepts x 1 layer x 5 samples)")
        print("=" * 60)
    else:
        concepts = list(EXPERIMENT_CONCEPTS)
        n_samples = args.samples
        max_seq_len = args.max_seq_len

        checkpoint_subset = _split_csv(args.checkpoints)
        checkpoints = select_checkpoints(
            only=args.only,
            checkpoint_subset=checkpoint_subset or None,
            limit=args.limit,
        )

        if args.layers:
            try:
                layers = [int(x) for x in _split_csv(args.layers)]
            except ValueError:
                print(
                    "ERROR: --layers must be comma-separated integers", file=sys.stderr
                )
                return 2
        else:
            layers = list(EXPERIMENT_LAYERS)

    if not checkpoints:
        print("ERROR: no checkpoints selected", file=sys.stderr)
        return 2

    if n_samples <= 0:
        print("ERROR: --samples must be positive", file=sys.stderr)
        return 2

    if any(ly < 0 or ly > 31 for ly in layers):
        print("ERROR: --layers must be integers in [0, 31]", file=sys.stderr)
        return 2

    os.makedirs(output_root, exist_ok=True)

    print(f"\nRL-Zero-Code Syntax Extraction")
    print(f"  Checkpoints ({len(checkpoints)}): {checkpoints}")
    print(f"  Concepts ({len(concepts)}):    {concepts}")
    print(f"  Layers ({len(layers)}):     {layers}")
    print(f"  Samples/concept:  {n_samples}")
    print(f"  Probe classes:    {list(PROBE_CLASSES)}")
    print(f"  Max seq len:      {max_seq_len}")
    print(f"  Chat template:    off (raw text)")
    print(f"  Output:           {output_root}")
    print(f"  Keep HF cache:    {args.keep_hf_cache}")
    print()

    # --- Run extraction per checkpoint -----------------------------------
    per_checkpoint: dict[str, dict[str, Any]] = {}

    # Load existing summary for resume.
    existing_summary_path = summary_path(output_root)
    if os.path.exists(existing_summary_path):
        try:
            with open(existing_summary_path, encoding="utf-8") as f:
                existing = json.load(f)
            per_checkpoint = dict(existing.get("per_checkpoint", {}))
        except (json.JSONDecodeError, OSError):
            pass

    for i, ckpt in enumerate(checkpoints):
        model_config = model_for_checkpoint(ckpt)
        ckpt_key = f"{model_config.name}/{ckpt}"

        try:
            stats = run_checkpoint_extraction(
                ckpt,
                concepts=concepts,
                layers=layers,
                n_samples=n_samples,
                output_root=output_root,
                max_seq_len=max_seq_len,
                expected_d_model=EXPECTED_D_MODEL,
            )
            per_checkpoint[ckpt_key] = stats
        except Exception as e:
            import traceback

            traceback.print_exc()
            per_checkpoint[ckpt_key] = {"error": str(e), "checkpoint": ckpt}

        # Write summary after each checkpoint (resumable).
        write_summary(
            output_root,
            checkpoints,
            per_checkpoint,
            concepts=concepts,
            layers=layers,
            n_samples=n_samples,
        )

        # Clean HF cache after the last checkpoint of each model.
        remaining = [
            c
            for c in checkpoints[i + 1 :]
            if model_for_checkpoint(c).hf_id == model_config.hf_id
        ]
        if not remaining and not args.keep_hf_cache:
            print(f"  Cleaning HF cache for {model_config.hf_id}...")
            _clean_hf_cache(model_config.hf_id)

    # --- Final report -----------------------------------------------------
    n_complete = sum(
        1
        for s in per_checkpoint.values()
        if isinstance(s, dict) and s.get("manifest", {}).get("complete") is True
    )
    n_errors = sum(
        1 for s in per_checkpoint.values() if isinstance(s, dict) and "error" in s
    )

    print(f"\n{'=' * 60}")
    print(f"Extraction complete: {n_complete} OK, {n_errors} errors")
    print(f"Results: {output_root}")
    print(f"{'=' * 60}")

    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
