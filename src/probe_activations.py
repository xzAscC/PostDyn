"""Eight-class raw-text probe-record collection, activation extraction, and
resumable safetensors persistence for the RL-Zero-Code syntax experiment.

This module is the **extraction layer**: it builds exactly 400 probe records
(8 classes x 50 samples), extracts last-token activations via the existing
``extract_layer_activations`` API (no chat template), and persists them
layer-by-layer on CPU as safetensors + JSON sidecars under the isolated
``rl_zero_code_syntax`` results root.

Key invariants
--------------
* **400 records**: 6 code classes x 50 + 2 gender classes x 50.
* **Shared target IDs**: all six code classes use the SAME 50 target task IDs
  from ``python_syntax_pairs.json``. Python valid/error come from the pairs
  file; cpp/js/java/go come from the local ``humaneval_x.json`` so task
  semantics align across languages. Downstream pinned code IDs are never used.
* **Gender pairing**: she and he records come from the same 50 WinoGender
  templates, differing only in the nominative pronoun.
* **Group keys**: keep all six code variants of one task together
  (``code:{task_id}``) and she/he of one template together
  (``gender:{template_index}``).
* **Protocol**: always ``"raw"`` (no chat template applied).

This module is data/extraction only: it runs no model on import, never edits
``src/config.py`` or dataset artifacts, and writes only under
``results/rl_zero_code_syntax``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.concept_dynamics import extract_layer_activations
from src.dataset_store import (
    HUMANEVAL_X_FILE,
    PYTHON_SYNTAX_PAIRS_FILE,
    dataset_path,
    load_dataset_json,
)
from src.rl_zero_experiment import (
    N_SAMPLES,
    PROBE_CLASSES,
    results_root,
)

# =============================================================================
# Constants
# =============================================================================

#: Extraction protocol is always raw text (no chat template).
PROTOCOL: str = "raw"

#: Six code probe classes in canonical order.
CODE_PROBE_CLASSES: tuple[str, ...] = (
    "python_valid",
    "python_syntax_error",
    "cpp",
    "js",
    "java",
    "go",
)

#: Two gender probe classes.
GENDER_PROBE_CLASSES: tuple[str, ...] = ("she", "he")

#: HumanEval-X language key for each non-python code probe class.
_HUMANEVAL_X_LANG: dict[str, str] = {
    "cpp": "cpp",
    "js": "js",
    "java": "java",
    "go": "go",
}

#: Tensor key inside each per-layer safetensors file.
_ACTIVATIONS_KEY: str = "activations"

#: Expected members per code group (6 language variants of one task).
_CODE_GROUP_SIZE: int = 6

#: Expected members per gender group (she + he of one template).
_GENDER_GROUP_SIZE: int = 2


# =============================================================================
# Data model
# =============================================================================


@dataclass(frozen=True)
class ProbeRecord:
    """One raw-text probe record.

    Attributes:
        sample_id: Unique identifier, e.g. ``"python_valid:0"``.
        label: Probe class (one of :data:`PROBE_CLASSES`).
        text: Raw input text (no chat template applied).
        group_id: Grouping key tying aligned records together.
        source_id: Original dataset identifier (task ID or template index).
        protocol: Always ``"raw"``.
    """

    sample_id: str
    label: str
    text: str
    group_id: str
    source_id: str
    protocol: str = PROTOCOL


# =============================================================================
# Record collection
# =============================================================================


def load_target_task_ids() -> list[int]:
    """Return the 50 target task IDs from ``python_syntax_pairs.json``, sorted.

    These IDs are the upstream target concept items, disjoint from the
    downstream HumanEval-X pinned IDs in ``shared_item_ids.json``.
    """
    data = load_dataset_json(PYTHON_SYNTAX_PAIRS_FILE)
    ids = data["selection"]["target_ids"]
    return sorted(int(i) for i in ids)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from a code string."""
    if "```" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("```")
    )


def _load_python_syntax_index() -> dict[int, dict[str, str]]:
    """Return ``{numeric_id: {"positive": str, "negative": str}}``."""
    data = load_dataset_json(PYTHON_SYNTAX_PAIRS_FILE)
    index: dict[int, dict[str, str]] = {}
    for item in data.get("items", []):
        nid = int(item["numeric_id"])
        index[nid] = {
            "positive": str(item["positive"]),
            "negative": str(item["negative"]),
        }
    return index


def _load_humaneval_x_codes() -> dict[str, dict[int, str]]:
    """Return ``{language: {numeric_id: code}}`` from the local JSON."""
    data = load_dataset_json(HUMANEVAL_X_FILE)
    index: dict[str, dict[int, str]] = {}
    for lang, items in data.get("languages", {}).items():
        by_id: dict[int, str] = {}
        for item in items:
            try:
                nid = int(item["numeric_id"])
            except (KeyError, TypeError, ValueError):
                continue
            code = _strip_code_fences(str(item.get("code") or ""))
            if code.strip():
                by_id[nid] = code
        index[lang] = by_id
    return index


def build_code_records(target_ids: list[int] | None = None) -> list[ProbeRecord]:
    """Build 300 code records (6 classes x 50 task IDs).

    All six code classes use the SAME 50 target task IDs from
    ``python_syntax_pairs.json``. Python valid/error come from the pairs file;
    cpp/js/java/go come from the local ``humaneval_x.json`` so task semantics
    align across languages.

    Records are ordered by ``target_ids`` position; within each task ID the
    order follows :data:`CODE_PROBE_CLASSES`.
    """
    if target_ids is None:
        target_ids = load_target_task_ids()
    if len(target_ids) != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} target task IDs, got {len(target_ids)}")

    pairs = _load_python_syntax_index()
    hx = _load_humaneval_x_codes()

    records: list[ProbeRecord] = []
    for position, tid in enumerate(target_ids):
        group_id = f"code:{tid}"
        if tid not in pairs:
            raise ValueError(f"target ID {tid} missing from {PYTHON_SYNTAX_PAIRS_FILE}")
        # python_valid (positive = valid Python program from the pairs file)
        records.append(
            ProbeRecord(
                sample_id=f"python_valid:{position}",
                label="python_valid",
                text=pairs[tid]["positive"],
                group_id=group_id,
                source_id=str(tid),
            )
        )
        # python_syntax_error (negative = one-mutation syntax error)
        records.append(
            ProbeRecord(
                sample_id=f"python_syntax_error:{position}",
                label="python_syntax_error",
                text=pairs[tid]["negative"],
                group_id=group_id,
                source_id=str(tid),
            )
        )
        # cpp, js, java, go -- code loaded by matching numeric_id
        for label, lang in _HUMANEVAL_X_LANG.items():
            if lang not in hx or tid not in hx[lang]:
                raise ValueError(
                    f"target ID {tid} missing from {HUMANEVAL_X_FILE} language {lang!r}"
                )
            records.append(
                ProbeRecord(
                    sample_id=f"{label}:{position}",
                    label=label,
                    text=hx[lang][tid],
                    group_id=group_id,
                    source_id=str(tid),
                )
            )
    return records


def build_gender_records(n_pairs: int = N_SAMPLES) -> list[ProbeRecord]:
    """Build gender records (she + he, one pair per WinoGender template).

    Each she/he pair comes from the SAME WinoGender template, differing only
    in the nominative pronoun. They share a ``gender:{i}`` group_id so the
    pairing is explicit.
    """
    from src.contrastive_datasets import load_winogender_pairs

    pairs = load_winogender_pairs(n_pairs)
    if len(pairs) < n_pairs:
        raise ValueError(
            f"need {n_pairs} WinoGender pairs, only {len(pairs)} available"
        )

    records: list[ProbeRecord] = []
    for i, (he_text, she_text) in enumerate(pairs[:n_pairs]):
        group_id = f"gender:{i}"
        records.append(
            ProbeRecord(
                sample_id=f"she:{i}",
                label="she",
                text=she_text,
                group_id=group_id,
                source_id=str(i),
            )
        )
        records.append(
            ProbeRecord(
                sample_id=f"he:{i}",
                label="he",
                text=he_text,
                group_id=group_id,
                source_id=str(i),
            )
        )
    return records


def build_probe_records() -> list[ProbeRecord]:
    """Build all 400 probe records (8 classes x 50 samples).

    Returns code records (300) followed by gender records (100).
    """
    return [*build_code_records(), *build_gender_records()]


# =============================================================================
# Validation & grouping
# =============================================================================


def validate_probe_records(records: list[ProbeRecord]) -> None:
    """Assert every structural invariant of the 400-record collection.

    Checks: total count, 50-per-class balance, protocol, group sizes, sample-ID
    uniqueness, and code/gender group structure.
    """
    if len(records) != len(PROBE_CLASSES) * N_SAMPLES:
        raise ValueError(
            f"expected {len(PROBE_CLASSES) * N_SAMPLES} records, got {len(records)}"
        )

    # Protocol is always raw.
    for r in records:
        if r.protocol != PROTOCOL:
            raise ValueError(
                f"record {r.sample_id!r} has protocol {r.protocol!r}, "
                f"expected {PROTOCOL!r}"
            )

    # 50 per class.
    by_label: dict[str, int] = {}
    for r in records:
        by_label[r.label] = by_label.get(r.label, 0) + 1
    for label in PROBE_CLASSES:
        count = by_label.get(label, 0)
        if count != N_SAMPLES:
            raise ValueError(
                f"class {label!r} has {count} records, expected {N_SAMPLES}"
            )

    # Sample IDs are unique.
    sample_ids = [r.sample_id for r in records]
    if len(set(sample_ids)) != len(sample_ids):
        dupes = {sid for sid in sample_ids if sample_ids.count(sid) > 1}
        raise ValueError(f"duplicate sample IDs: {sorted(dupes)}")

    # Group structure.
    groups = group_records(records)
    for gid, members in groups.items():
        if gid.startswith("code:"):
            labels = sorted(m.label for m in members)
            if labels != sorted(CODE_PROBE_CLASSES):
                raise ValueError(
                    f"group {gid!r} labels {labels} != {sorted(CODE_PROBE_CLASSES)}"
                )
            # All six variants share the same source_id (task ID).
            source_ids = {m.source_id for m in members}
            if len(source_ids) != 1:
                raise ValueError(
                    f"code group {gid!r} has mixed source IDs: {source_ids}"
                )
        elif gid.startswith("gender:"):
            labels = sorted(m.label for m in members)
            if labels != sorted(GENDER_PROBE_CLASSES):
                raise ValueError(
                    f"gender group {gid!r} labels {labels} != "
                    f"{sorted(GENDER_PROBE_CLASSES)}"
                )
        else:
            raise ValueError(f"unknown group prefix: {gid!r}")

    # No group leakage: every record appears in exactly one group.
    grouped_count = sum(len(m) for m in groups.values())
    if grouped_count != len(records):
        raise ValueError(f"grouped records ({grouped_count}) != total ({len(records)})")


def group_records(records: list[ProbeRecord]) -> dict[str, list[ProbeRecord]]:
    """Group records by ``group_id``, preserving intra-group record order."""
    groups: dict[str, list[ProbeRecord]] = {}
    for r in records:
        groups.setdefault(r.group_id, []).append(r)
    return groups


# =============================================================================
# Activation extraction
# =============================================================================


def extract_probe_activations(
    records: list[ProbeRecord],
    model,
    tokenizer,
    layers: list[int],
    *,
    max_seq_len: int = 512,
    use_chat_template: bool = False,
) -> dict[int, torch.Tensor]:
    """Extract last-token activations for all probe records (raw text).

    Delegates to :func:`src.concept_dynamics.extract_layer_activations` with
    ``use_chat_template=False`` so the primary protocol is raw text.

    Args:
        records: Probe records (defines row order).
        model: HF-style model (or mock) supporting ``output_hidden_states``.
        tokenizer: Tokenizer returning ``{input_ids, attention_mask}``.
        layers: 0-indexed transformer layer indices to extract.
        max_seq_len: Maximum tokenization length.
        use_chat_template: Always ``False`` for the raw protocol.

    Returns:
        ``{layer_idx: (n_records, d_model)}`` CPU float32 tensors.
    """
    texts = [r.text for r in records]
    return extract_layer_activations(
        model,
        tokenizer,
        texts,
        layers,
        max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
    )


# =============================================================================
# Records fingerprint (deterministic SHA-256 over ordered identity)
# =============================================================================


def record_text_sha256(record: ProbeRecord) -> str:
    """Return the SHA-256 hex digest of one record's raw text (UTF-8).

    This is the per-record text provenance hash.  It is persisted in every
    sidecar and in ``records.json`` so that consumers can verify a stored
    activation row really corresponds to the original text without trusting
    an aggregate fingerprint stamped after the fact.
    """
    return hashlib.sha256(record.text.encode("utf-8")).hexdigest()


def compute_records_fingerprint(records: list[ProbeRecord]) -> str:
    """Return a deterministic SHA-256 over the ordered record identity.

    The fingerprint captures ``sample_id``, ``label``, ``group_id``,
    ``source_id``, and a SHA-256 of the ``text`` for every record **in list
    order**.  It is used as a resume-compatibility check: if the ordered
    records change in any identity field or text content, the fingerprint
    changes and stale activations are rejected.

    The ``protocol`` field is deliberately excluded — it is always ``"raw"``
    for this module and is derived, not identity.
    """
    hasher = hashlib.sha256()
    for r in records:
        entry = json.dumps(
            {
                "sample_id": r.sample_id,
                "label": r.label,
                "group_id": r.group_id,
                "source_id": r.source_id,
                "text_sha256": record_text_sha256(r),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        hasher.update(entry.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def validate_sidecar_record_identity(
    sidecar: dict[str, Any],
    records: list[ProbeRecord],
) -> bool:
    """Return ``True`` iff the sidecar's record-provenance fields match records.

    Recomputes the ordered ``sample_ids``, ``labels``, ``group_ids``,
    ``source_ids``, per-record ``text_sha256`` array, and the aggregate
    ``records_fingerprint`` directly from ``records`` and compares each one
    against the sidecar.  Returns ``False`` if any required field is absent or
    mismatches.

    This is the strict, re-derivation-based check that replaces the former
    in-place metadata migration: an activation file can only be considered
    compatible when its stored text/source provenance matches the records the
    caller intends to extract now.  Old sidecars that were stamped with a
    fingerprint without ever persisting (or proving) the per-record text hash
    are therefore rejected and re-extracted.
    """
    expected_sample_ids = [r.sample_id for r in records]
    expected_labels = [r.label for r in records]
    expected_group_ids = [r.group_id for r in records]
    expected_source_ids = [r.source_id for r in records]
    expected_text_sha = [record_text_sha256(r) for r in records]
    expected_fingerprint = compute_records_fingerprint(records)

    if sidecar.get("sample_ids") != expected_sample_ids:
        return False
    if sidecar.get("labels") != expected_labels:
        return False
    if sidecar.get("group_ids") != expected_group_ids:
        return False
    if sidecar.get("source_ids") != expected_source_ids:
        return False
    if sidecar.get("text_sha256") != expected_text_sha:
        return False
    if sidecar.get("records_fingerprint") != expected_fingerprint:
        return False
    return True


# =============================================================================
# Persistence: records
# =============================================================================


def default_activations_root() -> str:
    """Default persistence root: ``{results_root}/activations``."""
    return os.path.join(results_root(), "activations")


def _records_json_path(output_dir: str) -> str:
    return os.path.join(output_dir, "records.json")


def save_records_json(output_dir: str, records: list[ProbeRecord]) -> str:
    """Write the global ``records.json`` (atomic).

    This file is written once per extraction run and shared across all
    models/checkpoints/layers. It records the protocol, the ordered
    ``sample_ids`` / ``labels`` / ``group_ids`` / ``source_ids`` /
    ``text_sha256`` arrays, the aggregate ``records_fingerprint``, and the
    full per-record list (including text). Persisting the per-record
    ``text_sha256`` and ordered ``source_ids`` at extraction time binds the
    actual text/source provenance so consumers never need to trust a
    fingerprint stamped after the fact.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = _records_json_path(output_dir)
    payload = {
        "protocol": PROTOCOL,
        "n_records": len(records),
        "probe_classes": list(PROBE_CLASSES),
        "records_fingerprint": compute_records_fingerprint(records),
        "sample_ids": [r.sample_id for r in records],
        "labels": [r.label for r in records],
        "group_ids": [r.group_id for r in records],
        "source_ids": [r.source_id for r in records],
        "text_sha256": [record_text_sha256(r) for r in records],
        "records": [asdict(r) for r in records],
    }
    _atomic_write_json(path, payload)
    return path


def load_records_json(output_dir: str) -> list[ProbeRecord]:
    """Load ``records.json`` and reconstruct :class:`ProbeRecord` objects."""
    path = _records_json_path(output_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"records.json not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        ProbeRecord(
            sample_id=item["sample_id"],
            label=item["label"],
            text=item["text"],
            group_id=item["group_id"],
            source_id=item["source_id"],
            protocol=item.get("protocol", PROTOCOL),
        )
        for item in data["records"]
    ]


# =============================================================================
# Persistence: per-layer activations
# =============================================================================


def _ckpt_dir(output_dir: str, model_name: str, checkpoint: str) -> str:
    return os.path.join(output_dir, model_name, checkpoint)


def _layer_base(
    output_dir: str, model_name: str, checkpoint: str, layer_idx: int
) -> str:
    return os.path.join(
        _ckpt_dir(output_dir, model_name, checkpoint), f"layer_{layer_idx}"
    )


def save_layer_activations(
    output_dir: str,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
    activations: torch.Tensor,
    records: list[ProbeRecord],
    *,
    max_seq_len: int | None = None,
) -> str:
    """Save one layer's activations atomically with a JSON sidecar.

    The safetensors file holds a single ``"activations"`` tensor of shape
    ``(n_records, d_model)`` in float32 on CPU. The JSON sidecar records
    the protocol, model/checkpoint/layer metadata, shapes, ``max_seq_len``,
    the ordered ``records_fingerprint``, and the ordered ``sample_ids`` /
    ``labels`` / ``group_ids`` / ``source_ids`` so any consumer can verify
    row alignment and extraction-config compatibility.

    Args:
        max_seq_len: Extraction max sequence length.  When ``None`` (the
            default for backward-compatible callers), the sidecar records
            ``null``; strict resume checks will then reject the layer.

    Returns the base path (without extension) of the written files.
    """
    if activations.dim() != 2:
        raise ValueError(
            f"expected 2D activations (n_records, d_model), got {activations.dim()}D"
        )
    if activations.shape[0] != len(records):
        raise ValueError(
            f"activations rows ({activations.shape[0]}) != records ({len(records)})"
        )

    ckpt_dir = _ckpt_dir(output_dir, model_name, checkpoint)
    os.makedirs(ckpt_dir, exist_ok=True)
    base = _layer_base(output_dir, model_name, checkpoint, layer_idx)
    final_safetensors = base + ".safetensors"
    final_json = base + ".json"

    tensor = activations.detach().cpu().to(torch.float32).contiguous()

    sidecar = {
        "protocol": PROTOCOL,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "layer_idx": layer_idx,
        "n_records": int(tensor.shape[0]),
        "d_model": int(tensor.shape[1]),
        "max_seq_len": max_seq_len,
        "records_fingerprint": compute_records_fingerprint(records),
        "activations_key": _ACTIVATIONS_KEY,
        "sample_ids": [r.sample_id for r in records],
        "labels": [r.label for r in records],
        "group_ids": [r.group_id for r in records],
        "source_ids": [r.source_id for r in records],
        "text_sha256": [record_text_sha256(r) for r in records],
    }

    # Atomic publication order: write both files to secure unique temp paths,
    # then publish the safetensors tensor FIRST and the JSON sidecar SECOND,
    # so a reader observing the sidecar is guaranteed the tensor is already
    # visible. On any failure before publication, both temp files are removed.
    tmp_safetensors = _secure_temp_path(ckpt_dir, suffix=".safetensors.tmp")
    tmp_json = _secure_temp_path(ckpt_dir, suffix=".json.tmp")
    try:
        save_file({_ACTIVATIONS_KEY: tensor}, tmp_safetensors)
        _write_json_file(tmp_json, sidecar)
        os.replace(tmp_safetensors, final_safetensors)
        tmp_safetensors = None
        os.replace(tmp_json, final_json)
        tmp_json = None
    finally:
        if tmp_safetensors is not None:
            _safe_remove(tmp_safetensors)
        if tmp_json is not None:
            _safe_remove(tmp_json)
    return base


def load_layer_activations(
    input_dir: str,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one layer's activations and JSON sidecar metadata.

    Returns ``(activations_tensor, sidecar_dict)``.
    """
    base = _layer_base(input_dir, model_name, checkpoint, layer_idx)
    safetensors_path = base + ".safetensors"
    json_path = base + ".json"
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"safetensors not found: {safetensors_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"sidecar JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        sidecar = json.load(f)

    key = sidecar.get("activations_key", _ACTIVATIONS_KEY)
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(key)

    return tensor, sidecar


def _read_safetensors_shape(safetensors_path: str, key: str) -> tuple[int, ...] | None:
    """Return the shape of ``key`` in a safetensors file without loading data.

    Returns ``None`` if the file or key is missing / unreadable.
    """
    try:
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            return tuple(f.get_slice(key).get_shape())
    except Exception:
        return None


def is_layer_complete(
    output_dir: str,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
    expected_n_records: int,
    *,
    expected_d_model: int | None = None,
    expected_protocol: str | None = None,
    expected_model_name: str | None = None,
    expected_checkpoint: str | None = None,
    expected_layer_idx: int | None = None,
    expected_max_seq_len: int | None = None,
    expected_records_fingerprint: str | None = None,
    expected_records: list[ProbeRecord] | None = None,
) -> bool:
    """Return True if a layer file is present, well-formed, and compatible.

    **Always checked** (backward-compatible behaviour):
        * Both the safetensors and the JSON sidecar exist.
        * The sidecar JSON parses.
        * ``n_records`` matches ``expected_n_records``.

    **Strict checks** (only when the corresponding ``expected_*`` kwarg is
    provided — callers that omit them get the old count-only check):
        * ``protocol`` matches ``expected_protocol``.
        * ``model_name`` matches ``expected_model_name``.
        * ``checkpoint`` matches ``expected_checkpoint``.
        * ``layer_idx`` matches ``expected_layer_idx``.
        * ``d_model`` matches ``expected_d_model``.
        * ``max_seq_len`` is present and matches ``expected_max_seq_len``.
        * ``records_fingerprint`` is present and matches
          ``expected_records_fingerprint``.
        * The safetensors tensor shape matches the sidecar's
          ``(n_records, d_model)`` (detects stale/mismatched tensors).

    **Full identity re-derivation** (when ``expected_records`` is provided):
        * ``n_records`` equals ``len(expected_records)``.
        * The safetensors tensor is exactly rank-2 with shape
          ``(n_records, d_model)`` (rank-1 tensors are rejected).
        * The sidecar's ordered ``sample_ids`` / ``labels`` / ``group_ids`` /
          ``source_ids`` / ``text_sha256`` arrays and its aggregate
          ``records_fingerprint`` are recomputed from ``expected_records`` and
          compared exactly. Any absent or mismatched provenance field causes
          the layer to be treated as incomplete (and therefore re-extracted).
          This is the strict mode that replaces the former in-place metadata
          migration: a stored activation is compatible only when its
          extraction-time text/source provenance matches the records the caller
          intends to extract now.
    """
    base = _layer_base(output_dir, model_name, checkpoint, layer_idx)
    safetensors_path = base + ".safetensors"
    json_path = base + ".json"
    if not os.path.exists(safetensors_path) or not os.path.exists(json_path):
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    n_records = int(sidecar.get("n_records", -1))
    if n_records != expected_n_records:
        return False

    if expected_protocol is not None:
        if sidecar.get("protocol") != expected_protocol:
            return False
    if expected_model_name is not None:
        if sidecar.get("model_name") != expected_model_name:
            return False
    if expected_checkpoint is not None:
        if sidecar.get("checkpoint") != expected_checkpoint:
            return False
    if expected_layer_idx is not None:
        if int(sidecar.get("layer_idx", -1)) != expected_layer_idx:
            return False

    sidecar_d_model = sidecar.get("d_model")
    if sidecar_d_model is not None:
        sidecar_d_model = int(sidecar_d_model)

    if expected_d_model is not None:
        if sidecar_d_model != expected_d_model:
            return False

    if expected_max_seq_len is not None:
        recorded = sidecar.get("max_seq_len")
        if recorded is None or int(recorded) != expected_max_seq_len:
            return False

    if expected_records_fingerprint is not None:
        if sidecar.get("records_fingerprint") != expected_records_fingerprint:
            return False

    key = sidecar.get("activations_key", _ACTIVATIONS_KEY)
    actual_shape = _read_safetensors_shape(safetensors_path, key)

    if expected_records is not None:
        if sidecar.get("protocol") != PROTOCOL:
            return False
        if sidecar.get("model_name") != model_name:
            return False
        if sidecar.get("checkpoint") != checkpoint:
            return False
        if int(sidecar.get("layer_idx", -1)) != layer_idx:
            return False
        if n_records != len(expected_records):
            return False
        if actual_shape is None or len(actual_shape) != 2:
            return False
        if actual_shape[0] != len(expected_records):
            return False
        if sidecar_d_model is not None and actual_shape[1] != sidecar_d_model:
            return False
        if not validate_sidecar_record_identity(sidecar, expected_records):
            return False
    else:
        if actual_shape is None or len(actual_shape) < 1:
            return False
        if actual_shape[0] != n_records:
            return False
        if sidecar_d_model is not None and len(actual_shape) >= 2:
            if actual_shape[1] != sidecar_d_model:
                return False

    return True


# =============================================================================
# Persistence: manifest (resume)
# =============================================================================


def _manifest_path(output_dir: str, model_name: str, checkpoint: str) -> str:
    return os.path.join(_ckpt_dir(output_dir, model_name, checkpoint), "manifest.json")


def load_manifest(output_dir: str, model_name: str, checkpoint: str) -> dict[str, Any]:
    """Load the resume manifest for one (model, checkpoint)."""
    path = _manifest_path(output_dir, model_name, checkpoint)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(
    output_dir: str,
    model_name: str,
    checkpoint: str,
    manifest: dict[str, Any],
) -> str:
    path = _manifest_path(output_dir, model_name, checkpoint)
    _atomic_write_json(path, manifest)
    return path


def _update_manifest(
    output_dir: str,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
    n_records: int,
    d_model: int,
    max_seq_len: int,
    records_fingerprint: str,
) -> dict[str, Any]:
    manifest = load_manifest(output_dir, model_name, checkpoint)
    completed = set(manifest.get("completed_layers", []))
    completed.add(layer_idx)
    manifest.update(
        {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "protocol": PROTOCOL,
            "n_records": n_records,
            "d_model": d_model,
            "max_seq_len": max_seq_len,
            "records_fingerprint": records_fingerprint,
            "completed_layers": sorted(completed),
        }
    )
    _save_manifest(output_dir, model_name, checkpoint, manifest)
    return manifest


# =============================================================================
# High-level runner with resume
# =============================================================================


def run_extraction_with_resume(
    records: list[ProbeRecord],
    model,
    tokenizer,
    layers: list[int],
    output_dir: str,
    model_name: str,
    checkpoint: str,
    *,
    max_seq_len: int = 512,
    expected_d_model: int | None = None,
) -> dict[str, Any]:
    """Extract activations layer-by-layer with resume support.

    For each requested layer:
    1. Skip if already complete AND compatible (``is_layer_complete`` with
       strict checks for protocol, model, checkpoint, layer, d_model,
       ``max_seq_len``, and ``records_fingerprint``).
    2. Otherwise extract via :func:`extract_probe_activations` (raw text).
    3. Persist with :func:`save_layer_activations` and update the manifest.

    The global ``records.json`` is (re)written at the start of the run so
    every layer file can be validated against it.

    Args:
        expected_d_model: When provided, existing layer files with a
            different ``d_model`` are treated as incomplete and re-extracted.
    """
    validate_probe_records(records)
    save_records_json(output_dir, records)
    fingerprint = compute_records_fingerprint(records)

    pending = [
        ly
        for ly in layers
        if not is_layer_complete(
            output_dir,
            model_name,
            checkpoint,
            ly,
            len(records),
            expected_protocol=PROTOCOL,
            expected_model_name=model_name,
            expected_checkpoint=checkpoint,
            expected_layer_idx=ly,
            expected_d_model=expected_d_model,
            expected_max_seq_len=max_seq_len,
            expected_records_fingerprint=fingerprint,
            expected_records=records,
        )
    ]
    if not pending:
        return {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "layers": list(layers),
            "skipped": list(layers),
            "extracted": [],
            "n_records": len(records),
            "max_seq_len": max_seq_len,
            "records_fingerprint": fingerprint,
        }

    activations = extract_probe_activations(
        records,
        model,
        tokenizer,
        pending,
        max_seq_len=max_seq_len,
    )

    extracted: list[int] = []
    d_model = 0
    for ly in pending:
        acts = activations[ly]
        d_model = int(acts.shape[1])
        save_layer_activations(
            output_dir,
            model_name,
            checkpoint,
            ly,
            acts,
            records,
            max_seq_len=max_seq_len,
        )
        _update_manifest(
            output_dir,
            model_name,
            checkpoint,
            ly,
            len(records),
            d_model,
            max_seq_len,
            fingerprint,
        )
        extracted.append(ly)

    return {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "layers": list(layers),
        "skipped": [ly for ly in layers if ly not in extracted],
        "extracted": extracted,
        "n_records": len(records),
        "d_model": d_model,
        "max_seq_len": max_seq_len,
        "records_fingerprint": fingerprint,
    }


# =============================================================================
# I/O helpers
# =============================================================================


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    """Write JSON atomically via a temp file + ``os.replace``."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _secure_temp_path(directory: str, *, suffix: str = ".tmp") -> str:
    """Return a unique, predictably-unguessable temp file path inside directory.

    Uses ``tempfile.mkstemp`` so each call gets a fresh, mode-0600 path even
    under concurrent extraction, then closes and removes the placeholder fd so
    the caller (e.g. ``save_file``) can write the path itself.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=".pa_", suffix=suffix, dir=directory)
    os.close(fd)
    os.remove(tmp_path)
    return tmp_path


def _write_json_file(path: str, payload: dict[str, Any]) -> None:
    """Write JSON to a caller-chosen path with fsync (no rename)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _safe_remove(path: str) -> None:
    """Remove a path, swallowing OSError (best-effort cleanup)."""
    try:
        os.remove(path)
    except OSError:
        pass


__all__ = [
    "PROTOCOL",
    "CODE_PROBE_CLASSES",
    "GENDER_PROBE_CLASSES",
    "ProbeRecord",
    "load_target_task_ids",
    "build_code_records",
    "build_gender_records",
    "build_probe_records",
    "compute_records_fingerprint",
    "record_text_sha256",
    "validate_sidecar_record_identity",
    "validate_probe_records",
    "group_records",
    "extract_probe_activations",
    "default_activations_root",
    "save_records_json",
    "load_records_json",
    "save_layer_activations",
    "load_layer_activations",
    "is_layer_complete",
    "load_manifest",
    "run_extraction_with_resume",
]
