"""Raw-vs-chat direction sensitivity analysis (SECONDARY diagnostic).

This module is a deliberately ISOLATED, secondary analysis. It quantifies how
much the DiM concept direction (``r = mu+ - mu-``) changes when the same
contrastive texts are fed to the same model checkpoint under two input
protocols:

* **raw**  -- the paired text is tokenized verbatim (``use_chat_template=False``),
  stored in an isolated *primary raw vectors* directory.
* **chat** -- each text is wrapped with ``tokenizer.apply_chat_template`` before
  the forward pass (``use_chat_template=True``).

For every ``(concept, checkpoint, layer)`` triple it reports two numbers:

* ``cosine``  -- ``cos(r_raw, r_chat)`` (orientation agreement, in ``[-1, 1]``).
* ``norm_diff`` -- ``||r_raw|| - ||r_chat||`` (signed magnitude difference).

Scope (hard-coded protocol)
---------------------------
* Model:        ``olmo3-rl-zero-code`` (the Dolci RL-Zero-Code trajectory).
* Checkpoints:  ``step_100``, ``step_1700``, ``step_2900`` -- early / mid / late
  of the 10-step RL trajectory. These are a SUBSET of the primary raw run
  (which has 11 checkpoints: 1 base + 10 RL steps) and a SUBSET of the old chat
  run (which covers all 10 RL steps for related/control concepts).
* Concepts (6): one ``target`` + four ``related`` + one ``control``:

    - target:  ``python_valid_vs_syntax_error`` (syntax-validity direction;
      NOT present in the old chat results, so its chat direction must be
      extracted separately under ``<sensitivity_dir>/chat_target_vectors/``).
    - related: ``code_python_vs_cpp``, ``code_python_vs_js``,
      ``code_python_vs_java``, ``code_python_vs_go`` (language identity).
    - control: ``gender_she_vs_he`` (out-of-domain WinoGender direction).

* Layers: ``EXPERIMENT_LAYERS_7B = [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]``.
* Samples: 50 per class (must match the old chat vectors' metadata).

Isolation contract
------------------
This module MUST NOT:

* write into ``results/concept_dynamics_multi`` (the primary / old-chat store).
* write into any ``metrics`` location.
* recompute unrelated old vectors.
* run a real model unless the caller explicitly invokes the gated extraction
  helper (``extract_missing_chat_target``) AND a model + chat-template-equipped
  tokenizer are actually available.

The old chat ``related``/``control`` vectors are read READ-ONLY via
``src.concept_dynamics.load_concept_vectors``. The chat ``target`` direction is
the ONLY thing this module ever extracts, and only into its own
``chat_target_vectors`` subdirectory.

Limitations (recorded in every output payload)
----------------------------------------------
* The old chat vector JSON metadata does not record the
  ``use_chat_template`` flag; chat-template-on is INFERRED from the default of
  ``run_concept_dynamics.py``. A mismatch here cannot be detected from
  metadata alone and is documented as a protocol assumption, not verified.
* Sensitivity covers 3 of the 11 primary-checkpoint trajectory points; it is a
  sparse diagnostic, not a full sweep.
* The target chat direction is not in the old results; until it is extracted
  (gated), target entries are emitted with ``status = "chat_missing"``.
* Sensitivity numbers NEVER enter the primary metrics pipeline; consumers must
  read ``sensitivity/sensitivity.json`` explicitly.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import torch

from src.concept_dynamics import ConceptVector, load_concept_vectors
from src.config import EXPERIMENT_LAYERS_7B, OLMO3_VARIANTS

# =============================================================================
# Protocol constants (hard-coded scope of this secondary analysis)
# =============================================================================

#: Model analysed by this sensitivity pass.
SENSITIVITY_MODEL: str = "olmo3-rl-zero-code"

#: The three checkpoints (early / mid / late of the RL-Zero-Code trajectory).
SENSITIVITY_CHECKPOINTS: list[str] = ["step_100", "step_1700", "step_2900"]

#: Layer grid (identical to the primary experiment rule).
SENSITIVITY_LAYERS: list[int] = list(EXPERIMENT_LAYERS_7B)

#: Sample count per class; MUST match the old chat vectors' metadata.
SENSITIVITY_N_SAMPLES: int = 50

#: Hidden dim of Olmo-3-7B.
SENSITIVITY_D_MODEL: int = OLMO3_VARIANTS[SENSITIVITY_MODEL].d_model

#: The single target concept whose chat direction is NOT in the old results.
TARGET_CONCEPT: str = "python_valid_vs_syntax_error"

#: Related code-language concepts (already extracted in the old chat results).
RELATED_CONCEPTS: list[str] = [
    "code_python_vs_cpp",
    "code_python_vs_js",
    "code_python_vs_java",
    "code_python_vs_go",
]

#: Out-of-domain control concept (already extracted in the old chat results).
CONTROL_CONCEPT: str = "gender_she_vs_he"

#: Ordered concept list for this sensitivity pass.
SENSITIVITY_CONCEPTS: list[str] = [
    TARGET_CONCEPT,
    *RELATED_CONCEPTS,
    CONTROL_CONCEPT,
]


def _concept_role(concept: str) -> str:
    if concept == TARGET_CONCEPT:
        return "target"
    if concept == CONTROL_CONCEPT:
        return "control"
    if concept in RELATED_CONCEPTS:
        return "related"
    raise ValueError(
        f"Concept {concept!r} is not part of the sensitivity protocol "
        f"(expected one of {SENSITIVITY_CONCEPTS})."
    )


#: Human-readable protocol block written into every output payload.
PROTOCOL: dict[str, Any] = {
    "name": "raw-vs-chat-direction-sensitivity",
    "version": 1,
    "primary": False,
    "description": (
        "Secondary diagnostic: cosine and signed norm difference between the "
        "DiM concept direction r = mu+ - mu- computed from raw (no chat "
        "template) and chat (tokenizer.apply_chat_template) inputs, on the "
        "same model / checkpoint / layer / concept / samples."
    ),
    "direction_field": "raw_direction",
    "metrics": ["cosine", "norm_diff"],
    "model": SENSITIVITY_MODEL,
    "checkpoints": list(SENSITIVITY_CHECKPOINTS),
    "layers": list(SENSITIVITY_LAYERS),
    "n_samples_per_class": SENSITIVITY_N_SAMPLES,
    "d_model": SENSITIVITY_D_MODEL,
    "concepts": [{"name": c, "role": _concept_role(c)} for c in SENSITIVITY_CONCEPTS],
    "input_protocols": {
        "raw": {
            "use_chat_template": False,
            "source": "isolated primary raw vectors directory (provided by caller)",
        },
        "chat": {
            "use_chat_template": True,
            "related_control_source": (
                "old chat results (results/concept_dynamics_multi/vectors), "
                "READ-ONLY reuse"
            ),
            "target_source": (
                "extracted under <sensitivity_dir>/chat_target_vectors/ via "
                "tokenizer.apply_chat_template (gated, resumable)"
            ),
        },
    },
}

#: Limitations written into every output payload (honest labelling).
LIMITATIONS: list[str] = [
    "Secondary analysis only; sensitivity numbers NEVER enter the primary "
    "metrics pipeline (results/concept_dynamics_multi) and must be read "
    "explicitly from sensitivity/sensitivity.json.",
    "The old chat vector JSON metadata does not record the use_chat_template "
    "flag; chat-template-on is INFERRED from the default of "
    "run_concept_dynamics.py and is NOT verifiable from the stored metadata.",
    "Covers 3 of the 11 primary-checkpoint trajectory points (step_100, "
    "step_1700, step_2900); it is a sparse diagnostic, not a full sweep.",
    "The target chat direction (python_valid_vs_syntax_error) is NOT in the "
    "old chat results and must be extracted separately; until then target "
    "entries are emitted with status='chat_missing'.",
    "Direction is the un-normalized DiM raw_direction; cosine is invariant to "
    "the raw/normalized choice but the norm difference is only meaningful on "
    "the un-normalized vector.",
]


# =============================================================================
# Errors
# =============================================================================


class MetadataMismatchError(Exception):
    """Raised when a stored vector's metadata conflicts with the protocol.

    The ``mismatches`` attribute carries the list of human-readable field
    mismatches (model / checkpoint / layer / concept / sample count / d_model).
    """

    mismatches: list[str]

    def __init__(self, message: str, mismatches: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.mismatches = list(mismatches) if mismatches else []


class ChatTemplateUnavailableError(Exception):
    """Raised when chat-template extraction is requested but unavailable.

    The gated target extractor raises this (instead of silently falling back)
    so callers fail loudly when the model or its chat template cannot be
    loaded.
    """


# =============================================================================
# Spec + entry containers
# =============================================================================


@dataclass(frozen=True)
class VectorSpec:
    """Expected coordinates of a sensitivity vector."""

    model: str
    checkpoint: str
    layer_idx: int
    concept: str
    n_samples: int
    d_model: int

    def label(self) -> str:
        return f"{self.model}/{self.checkpoint}/layer_{self.layer_idx}/{self.concept}"


@dataclass
class SensitivityEntry:
    """One (concept, checkpoint, layer) comparison result."""

    model: str
    checkpoint: str
    layer_idx: int
    concept: str
    role: str
    n_samples: int
    d_model: int
    status: str
    cosine: Optional[float] = None
    norm_raw: Optional[float] = None
    norm_chat: Optional[float] = None
    norm_diff: Optional[float] = None
    raw_source: Optional[str] = None
    chat_source: Optional[str] = None
    detail: Optional[str] = None


# =============================================================================
# Metadata validation
# =============================================================================


def validate_chat_metadata(
    metadata: dict[str, Any],
    spec: VectorSpec,
) -> None:
    """Verify a stored chat vector file matches the sensitivity protocol.

    Checks (all must pass; all failures are collected into one error):

    * ``metadata["model_name"]`` == ``spec.model``
    * ``metadata["layer_idx"]``  == ``spec.layer_idx``
    * a concept named ``spec.concept`` is present in ``metadata["concepts"]``
    * that concept's ``n_positive`` and ``n_negative`` are both ``>= spec.n_samples``
    * that concept's ``d_model`` == ``spec.d_model``

    The checkpoint is encoded in the on-disk directory path (not in the JSON),
    so callers must pass a ``VectorSpec`` whose ``checkpoint`` already matches
    the directory they loaded from; this function does not re-check it.

    Args:
        metadata: The parsed ``layer_<L>.json`` contents.
        spec: The expected coordinates.

    Raises:
        MetadataMismatchError: if any field conflicts with ``spec``. The
            ``mismatches`` attribute lists every problem.
    """
    mismatches: list[str] = []
    model = metadata.get("model_name")
    if model != spec.model:
        mismatches.append(f"model_name: stored {model!r} != expected {spec.model!r}")

    layer = metadata.get("layer_idx")
    if layer != spec.layer_idx:
        mismatches.append(f"layer_idx: stored {layer!r} != expected {spec.layer_idx!r}")

    concepts = metadata.get("concepts", [])
    if not isinstance(concepts, list):
        mismatches.append(f"concepts: expected list, got {type(concepts).__name__}")
        concepts = []

    entry = next(
        (c for c in concepts if isinstance(c, dict) and c.get("name") == spec.concept),
        None,
    )
    if entry is None:
        mismatches.append(f"concept: {spec.concept!r} not present in stored concepts")
    else:
        n_pos = entry.get("n_positive")
        n_neg = entry.get("n_negative")
        if not isinstance(n_pos, int) or n_pos < spec.n_samples:
            mismatches.append(
                f"n_positive: stored {n_pos!r} < expected {spec.n_samples}"
            )
        if not isinstance(n_neg, int) or n_neg < spec.n_samples:
            mismatches.append(
                f"n_negative: stored {n_neg!r} < expected {spec.n_samples}"
            )
        d = entry.get("d_model")
        if d != spec.d_model:
            mismatches.append(f"d_model: stored {d!r} != expected {spec.d_model!r}")

    if mismatches:
        raise MetadataMismatchError(
            f"Chat vector metadata mismatch for {spec.label()}:\n  - "
            + "\n  - ".join(mismatches),
            mismatches=mismatches,
        )


# =============================================================================
# Vector comparison math
# =============================================================================


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-10) -> float:
    """Cosine similarity between two 1-D tensors."""
    if a.shape != b.shape:
        raise ValueError(f"cosine: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    na = a.norm(p=2).item()
    nb = b.norm(p=2).item()
    if na < eps or nb < eps:
        return 0.0
    return float((torch.dot(a, b) / (na * nb)).clamp(-1.0, 1.0).item())


def norm_of(a: torch.Tensor) -> float:
    """L2 norm of a 1-D tensor as a plain float."""
    return float(a.norm(p=2).item())


def compare_vectors(raw_cv: ConceptVector, chat_cv: ConceptVector) -> dict[str, float]:
    """Compute the two sensitivity metrics for a matched vector pair.

    Uses the un-normalized DiM ``raw_direction`` of each ConceptVector. The two
    vectors must share the same ``d_model``; model/checkpoint/layer/concept
    consistency is the caller's responsibility (see ``compare_with_checks``).

    Returns:
        ``{"cosine": ..., "norm_raw": ..., "norm_chat": ..., "norm_diff": ...}``
        where ``norm_diff = norm_raw - norm_chat`` (signed).
    """
    if raw_cv.raw_direction.shape != chat_cv.raw_direction.shape:
        raise ValueError(
            f"raw_direction shape mismatch: raw {tuple(raw_cv.raw_direction.shape)} "
            f"vs chat {tuple(chat_cv.raw_direction.shape)}"
        )
    if raw_cv.d_model != chat_cv.d_model:
        raise ValueError(
            f"d_model mismatch: raw {raw_cv.d_model} vs chat {chat_cv.d_model}"
        )
    n_raw = norm_of(raw_cv.raw_direction)
    n_chat = norm_of(chat_cv.raw_direction)
    return {
        "cosine": cosine_similarity(raw_cv.raw_direction, chat_cv.raw_direction),
        "norm_raw": n_raw,
        "norm_chat": n_chat,
        "norm_diff": n_raw - n_chat,
    }


# =============================================================================
# Vector loading (raw = isolated; chat = old results or extracted target)
# =============================================================================


def _vector_path(base_dir: str, model: str, checkpoint: str, layer_idx: int) -> str:
    """Mirror src.concept_dynamics' on-disk layout for a (model, ckpt, layer)."""
    return os.path.join(base_dir, model, checkpoint, f"layer_{layer_idx}")


def load_raw_vector(
    raw_vectors_dir: str,
    spec: VectorSpec,
) -> ConceptVector:
    """Load an ISOLATED primary raw vector.

    Raises ``FileNotFoundError`` if the isolated raw store does not contain the
    requested (model, checkpoint, layer) triple. Callers treat this as
    ``status='raw_missing'``.
    """
    vectors = load_concept_vectors(
        raw_vectors_dir, spec.model, spec.layer_idx, spec.checkpoint
    )
    if spec.concept not in vectors:
        raise FileNotFoundError(
            f"raw vector missing: concept {spec.concept!r} not in "
            f"{_vector_path(raw_vectors_dir, spec.model, spec.checkpoint, spec.layer_idx)}"
        )
    return vectors[spec.concept]


def load_chat_vector(
    chat_old_vectors_dir: str,
    chat_target_vectors_dir: str,
    spec: VectorSpec,
) -> tuple[ConceptVector, str]:
    """Load a chat vector, choosing the source by concept role.

    * ``related`` / ``control`` concepts are read READ-ONLY from the OLD chat
      results at ``chat_old_vectors_dir`` (never written to).
    * ``target`` concept is read from the EXTRACTED target store at
      ``chat_target_vectors_dir`` (populated by ``extract_missing_chat_target``).

    Returns:
        ``(ConceptVector, source_label)`` where ``source_label`` is one of
        ``"old_chat_results"`` or ``"sensitivity_chat_target_extraction"``.

    Raises:
        FileNotFoundError: if the chosen source does not contain the vector.
        MetadataMismatchError: (re-raised) if old-result metadata conflicts
            with the protocol.
    """
    role = _concept_role(spec.concept)
    if role == "target":
        vectors = load_concept_vectors(
            chat_target_vectors_dir,
            spec.model,
            spec.layer_idx,
            spec.checkpoint,
        )
        if spec.concept not in vectors:
            raise FileNotFoundError(
                f"target chat vector not extracted yet: concept "
                f"{spec.concept!r} not in {_vector_path(chat_target_vectors_dir, spec.model, spec.checkpoint, spec.layer_idx)}"
            )
        return vectors[spec.concept], "sensitivity_chat_target_extraction"

    # related / control: READ-ONLY reuse of old chat results.
    vectors = load_concept_vectors(
        chat_old_vectors_dir, spec.model, spec.layer_idx, spec.checkpoint
    )
    if spec.concept not in vectors:
        raise FileNotFoundError(
            f"old chat vector missing: concept {spec.concept!r} not in "
            f"{_vector_path(chat_old_vectors_dir, spec.model, spec.checkpoint, spec.layer_idx)}"
        )
    # Validate old-result metadata BEFORE reuse (model/layer/concept/samples/d_model).
    base = _vector_path(
        chat_old_vectors_dir, spec.model, spec.checkpoint, spec.layer_idx
    )
    with open(base + ".json") as f:
        metadata = json.load(f)
    validate_chat_metadata(metadata, spec)
    return vectors[spec.concept], "old_chat_results"


# =============================================================================
# Atomic output
# =============================================================================


def write_sensitivity_json(path: str, payload: dict[str, Any]) -> str:
    """Atomically write ``payload`` to ``path`` (temp file + ``os.replace``).

    The temp file is created in the same directory as ``path`` (so the rename
    stays on one filesystem) and is removed on any serialization error. The
    destination directory is created if needed.

    Returns:
        The final ``path`` written.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".sensitivity.", suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        with open(tmp_name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        # Clean up the temp file; never leave a half-written artifact behind.
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise
    return path


# =============================================================================
# Orchestrator
# =============================================================================


def _entry_from_spec(spec: VectorSpec, status: str) -> SensitivityEntry:
    return SensitivityEntry(
        model=spec.model,
        checkpoint=spec.checkpoint,
        layer_idx=spec.layer_idx,
        concept=spec.concept,
        role=_concept_role(spec.concept),
        n_samples=spec.n_samples,
        d_model=spec.d_model,
        status=status,
    )


def run_sensitivity_analysis(
    *,
    raw_vectors_dir: str,
    chat_old_vectors_dir: str,
    chat_target_vectors_dir: str,
    output_dir: str,
    output_filename: str = "sensitivity.json",
    model: str = SENSITIVITY_MODEL,
    checkpoints: Optional[list[str]] = None,
    concepts: Optional[list[str]] = None,
    layers: Optional[list[int]] = None,
    n_samples: int = SENSITIVITY_N_SAMPLES,
    d_model: int = SENSITIVITY_D_MODEL,
) -> dict[str, Any]:
    """Run the raw-vs-chat sensitivity comparison and write ``sensitivity.json``.

    For every ``(concept, checkpoint, layer)`` triple this loads the raw vector
    (from the isolated raw store) and the chat vector (old results for
    related/control, extracted target store for the target), validates
    metadata, and records the cosine + signed norm difference. Missing vectors
    or metadata mismatches are recorded honestly per-entry rather than aborting
    the whole run.

    The output is written ATOMICALLY to
    ``<output_dir>/<output_filename>`` (default ``sensitivity.json``). The
    old chat results directory is ONLY read; nothing is written there.

    Args:
        raw_vectors_dir: Isolated primary raw vectors root (layout mirrors
            ``concept_dynamics_multi/vectors``).
        chat_old_vectors_dir: Old chat results root (READ-ONLY reuse for
            related/control concepts).
        chat_target_vectors_dir: Root of the resumable chat-TARGET extraction
            (only the target concept is ever read/written here).
        output_dir: Where to write ``sensitivity.json``. MUST be distinct from
            ``chat_old_vectors_dir``'s parent (path isolation).
        model, checkpoints, concepts, layers, n_samples, d_model: Protocol
            overrides (default to the hard-coded sensitivity scope).

    Returns:
        The payload dict that was written to disk.

    Raises:
        ValueError: if the output path is not isolated from the old chat store
            (guard against mixing sensitivity outputs into primary stores).
    """
    checkpoints = (
        list(checkpoints) if checkpoints is not None else list(SENSITIVITY_CHECKPOINTS)
    )
    concepts = list(concepts) if concepts is not None else list(SENSITIVITY_CONCEPTS)
    layers = list(layers) if layers is not None else list(SENSITIVITY_LAYERS)

    # Path-isolation guards: refuse to write sensitivity output anywhere inside
    # the old chat results tree or the primary raw store, AND require the final
    # resolved output path to remain strictly under output_dir. The second
    # guard validates output_filename is a plain basename and that a symlinked
    # output_dir cannot escape the approved area after realpath resolution.
    _assert_path_isolation(output_dir, chat_old_vectors_dir, raw_vectors_dir)
    _assert_output_containment(output_dir, output_filename)

    entries: list[SensitivityEntry] = []
    counts = {
        "compared": 0,
        "raw_missing": 0,
        "chat_missing": 0,
        "metadata_rejected": 0,
    }

    for checkpoint in checkpoints:
        for layer_idx in layers:
            for concept in concepts:
                spec = VectorSpec(
                    model=model,
                    checkpoint=checkpoint,
                    layer_idx=layer_idx,
                    concept=concept,
                    n_samples=n_samples,
                    d_model=d_model,
                )
                entry = _compare_one(
                    spec,
                    raw_vectors_dir=raw_vectors_dir,
                    chat_old_vectors_dir=chat_old_vectors_dir,
                    chat_target_vectors_dir=chat_target_vectors_dir,
                )
                entries.append(entry)
                counts[entry.status] = counts.get(entry.status, 0) + 1

    payload = _build_payload(
        entries=entries,
        model=model,
        checkpoints=checkpoints,
        concepts=concepts,
        layers=layers,
        n_samples=n_samples,
        d_model=d_model,
        counts=counts,
        raw_vectors_dir=raw_vectors_dir,
        chat_old_vectors_dir=chat_old_vectors_dir,
        chat_target_vectors_dir=chat_target_vectors_dir,
        output_dir=output_dir,
    )

    out_path = os.path.join(output_dir, output_filename)
    write_sensitivity_json(out_path, payload)
    return payload


def _compare_one(
    spec: VectorSpec,
    *,
    raw_vectors_dir: str,
    chat_old_vectors_dir: str,
    chat_target_vectors_dir: str,
) -> SensitivityEntry:
    """Load raw + chat vectors for one spec and produce a SensitivityEntry."""
    # --- raw side ---
    try:
        raw_cv = load_raw_vector(raw_vectors_dir, spec)
    except FileNotFoundError as e:
        entry = _entry_from_spec(spec, "raw_missing")
        entry.detail = str(e)
        return entry

    # --- chat side ---
    try:
        chat_cv, chat_source = load_chat_vector(
            chat_old_vectors_dir, chat_target_vectors_dir, spec
        )
    except FileNotFoundError as e:
        entry = _entry_from_spec(spec, "chat_missing")
        entry.detail = str(e)
        entry.raw_source = "isolated_raw_store"
        return entry
    except MetadataMismatchError as e:
        entry = _entry_from_spec(spec, "metadata_rejected")
        entry.detail = str(e)
        entry.raw_source = "isolated_raw_store"
        return entry

    # --- compare ---
    try:
        metrics = compare_vectors(raw_cv, chat_cv)
    except ValueError as e:
        entry = _entry_from_spec(spec, "metadata_rejected")
        entry.detail = f"comparison failed: {e}"
        entry.raw_source = "isolated_raw_store"
        entry.chat_source = (
            "old_chat_results"
            if spec.concept != TARGET_CONCEPT
            else "sensitivity_chat_target_extraction"
        )
        return entry

    entry = _entry_from_spec(spec, "compared")
    entry.cosine = metrics["cosine"]
    entry.norm_raw = metrics["norm_raw"]
    entry.norm_chat = metrics["norm_chat"]
    entry.norm_diff = metrics["norm_diff"]
    entry.raw_source = "isolated_raw_store"
    entry.chat_source = chat_source
    return entry


# =============================================================================
# Canonical path containment (defeats symlinks + parent traversal)
# =============================================================================


def _canonical(path: str) -> str:
    """Return the canonical real path of ``path``.

    ``os.path.realpath`` (strict=False, the default) resolves symlinks on the
    EXISTING prefix and appends any non-existing tail verbatim. This is exactly
    what a containment guard needs: the existing part is de-symlinked so a
    symlinked directory cannot smuggle output into a forbidden area, while the
    non-existing tail (e.g. a brand-new ``output_dir``) is still comparable
    against the canonical output root via ``os.path.commonpath``.
    """
    return os.path.realpath(path)


def _is_at_or_under(path: str, root: str) -> bool:
    """True iff canonical ``path`` is ``root`` itself or strictly beneath it.

    Uses ``os.path.commonpath`` on canonical real paths -- a true path-component
    containment test. The lexical ``abspath`` + ``startswith`` it replaces was
    bypassable two ways: symlinks (abspath does not resolve them) and prefix
    collisions (``"/a/bad"``.startswith(``"/a/b"``) is True but they are
    siblings, not nested).
    """
    can_path = _canonical(path)
    can_root = _canonical(root)
    try:
        return os.path.commonpath([can_path, can_root]) == can_root
    except ValueError:
        # commonpath raises on mixed drives (Windows) or empty inputs.
        return False


def _is_strictly_below(path: str, root: str) -> bool:
    """True iff canonical ``path`` is strictly beneath (not equal to) ``root``."""
    can_path = _canonical(path)
    can_root = _canonical(root)
    if can_path == can_root:
        return False
    try:
        return os.path.commonpath([can_path, can_root]) == can_root
    except ValueError:
        return False


def _validate_output_filename(filename: str) -> None:
    """Require ``output_filename`` to be a simple, nonempty basename.

    Defends against ``os.path.join(output_dir, output_filename)`` escapes:
    ``../``, absolute paths, separator-containing names, and the ``.``/``..``
    special names would all write outside the approved ``output_dir``. This is
    the PRIMARY defence; the resolved-path containment check in
    :func:`_assert_output_containment` is the SECONDARY defence against
    symlinked ``output_dir`` values.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError("output_filename must be a non-empty string.")
    if "/" in filename or "\\" in filename:
        raise ValueError(
            f"output_filename must be a plain basename (contains a path "
            f"separator): {filename!r}"
        )
    if filename in (".", ".."):
        raise ValueError(f"output_filename must be a real file name, not {filename!r}.")
    # Detect Windows-style drive prefixes (``C:foo``) on ANY platform, not just
    # Windows: ``os.path.splitdrive`` only recognises drives on Windows, so on
    # POSIX it would let ``C:evil.json`` through as a "relative basename" even
    # though it would resolve to drive C on a Windows host.
    if (
        len(filename) >= 2
        and filename[1] == ":"
        and filename[0].isascii()
        and filename[0].isalpha()
    ):
        raise ValueError(
            f"output_filename must be a plain basename without a drive "
            f"prefix: {filename!r}"
        )
    if os.path.isabs(filename):
        raise ValueError(
            f"output_filename must be a relative basename, not an absolute "
            f"path: {filename!r}"
        )


def _assert_output_containment(output_dir: str, output_filename: str) -> str:
    """Validate and return the canonical final output FILE path.

    Ensures ``output_filename`` is a safe basename AND that the canonical
    resolved path ``realpath(join(output_dir, output_filename))`` is strictly
    beneath ``realpath(output_dir)``. The second check catches symlinked
    ``output_dir`` values whose canonical target escapes the approved area.
    """
    _validate_output_filename(output_filename)
    can_root = _canonical(output_dir)
    final_path = _canonical(os.path.join(output_dir, output_filename))
    if not _is_strictly_below(final_path, can_root):
        raise ValueError(
            f"Path-isolation violation: resolved output path {final_path!r} "
            f"escapes the canonical output dir {can_root!r}."
        )
    return final_path


def _assert_path_isolation(output_dir: str, *forbidden_roots: str) -> None:
    """Refuse to write sensitivity output inside any primary store.

    Each forbidden root and its PARENT directory (the primary store root, e.g.
    ``results/concept_dynamics_multi`` for the ``.../vectors`` subdir) are both
    blocked, so a caller cannot smuggle sensitivity output into a primary store
    by writing to a sibling of ``vectors/``. The literal name ``metrics`` is
    also blocked anywhere along the canonical output path.

    Containment is tested with ``os.path.realpath`` + ``os.path.commonpath`` so
    that symlinked directories and ``..`` traversal cannot bypass the guard
    (the former lexical ``abspath`` + ``startswith`` check was bypassable via
    symlinks and wrongly accepted ``/a/bad`` as inside ``/a/b``).
    """
    can_out = _canonical(output_dir)
    forbidden: list[str] = []
    for root in forbidden_roots:
        if not root:
            continue
        can_root = _canonical(root)
        forbidden.append(can_root)
        parent = os.path.dirname(can_root.rstrip(os.sep))
        if parent:
            forbidden.append(parent)
    for forb in forbidden:
        if _is_at_or_under(can_out, forb):
            raise ValueError(
                f"Path-isolation violation: sensitivity output dir "
                f"{can_out!r} must not lie inside the primary store {forb!r}."
            )
    parts = can_out.split(os.sep)
    if "metrics" in parts:
        raise ValueError(
            f"Path-isolation violation: sensitivity output dir {can_out!r} "
            f"contains a 'metrics' segment."
        )


#: Designated root for ALL sensitivity outputs (``sensitivity.json`` AND the
#: chat-target extraction subdir). Mirrors the CLI default; library callers
#: may override via ``extract_missing_chat_target(sensitivity_output_root=...)``.
DEFAULT_SENSITIVITY_OUTPUT_ROOT: str = os.path.join("results", "sensitivity")


def _assert_chat_target_isolation(
    chat_target_vectors_dir: str,
    *,
    sensitivity_output_root: str,
    raw_vectors_dir: Optional[str] = None,
    chat_old_vectors_dir: Optional[str] = None,
) -> None:
    """Refuse to extract chat-TARGET vectors outside the sensitivity root.

    The chat-target dir is the ONLY place the sensitivity pass ever WRITES
    (during the gated extraction). It must:

    * nest STRICTLY UNDER ``sensitivity_output_root`` (the same root that holds
      ``sensitivity.json``) -- the target dir cannot BE the root (that would
      collide with the comparison output) and cannot escape it; and
    * NEVER lie inside the raw primary store, the old-chat primary store, or
      either store's parent (reuses :func:`_assert_path_isolation` so the rule
      is identical to the output-dir guard).

    Containment is tested with canonical ``os.path.realpath`` +
    ``os.path.commonpath`` (never raw lexical prefixes), so symlinked
    directories and ``..`` traversal cannot bypass the guard.
    """
    forbidden_roots = tuple(r for r in (chat_old_vectors_dir, raw_vectors_dir) if r)
    if forbidden_roots:
        _assert_path_isolation(chat_target_vectors_dir, *forbidden_roots)

    if _is_strictly_below(chat_target_vectors_dir, sensitivity_output_root):
        return

    can_tgt = _canonical(chat_target_vectors_dir)
    can_root = _canonical(sensitivity_output_root)
    if can_tgt == can_root:
        raise ValueError(
            f"Path-isolation violation: chat_target_vectors_dir {can_tgt!r} "
            "must be a SUBDIR of the sensitivity output root, not the root "
            "itself (it would collide with sensitivity.json)."
        )
    raise ValueError(
        f"Path-isolation violation: chat_target_vectors_dir {can_tgt!r} "
        f"must live strictly under the sensitivity output root "
        f"{can_root!r}; refusing to extract into a primary store or "
        "unrelated location."
    )


def _build_payload(
    *,
    entries: list[SensitivityEntry],
    model: str,
    checkpoints: list[str],
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    d_model: int,
    counts: dict[str, int],
    raw_vectors_dir: str,
    chat_old_vectors_dir: str,
    chat_target_vectors_dir: str,
    output_dir: str,
) -> dict[str, Any]:
    return {
        "protocol": dict(PROTOCOL),
        "limitations": list(LIMITATIONS),
        "scope": {
            "model": model,
            "checkpoints": list(checkpoints),
            "concepts": [{"name": c, "role": _concept_role(c)} for c in concepts],
            "layers": list(layers),
            "n_samples_per_class": n_samples,
            "d_model": d_model,
        },
        "paths": {
            "raw_vectors_dir": raw_vectors_dir,
            "chat_old_vectors_dir": chat_old_vectors_dir,
            "chat_target_vectors_dir": chat_target_vectors_dir,
            "output_dir": output_dir,
            "note": (
                "chat_old_vectors_dir is READ-ONLY; sensitivity writes only to "
                "output_dir and (during gated extraction) chat_target_vectors_dir."
            ),
        },
        "summary": {
            "n_entries": len(entries),
            "n_compared": counts.get("compared", 0),
            "n_raw_missing": counts.get("raw_missing", 0),
            "n_chat_missing": counts.get("chat_missing", 0),
            "n_metadata_rejected": counts.get("metadata_rejected", 0),
        },
        "entries": [asdict(e) for e in entries],
    }


# =============================================================================
# Gated, resumable chat-TARGET extraction
# =============================================================================


def extract_missing_chat_target(
    *,
    chat_target_vectors_dir: str,
    sensitivity_output_root: str = DEFAULT_SENSITIVITY_OUTPUT_ROOT,
    raw_vectors_dir: Optional[str] = None,
    chat_old_vectors_dir: Optional[str] = None,
    checkpoints: Optional[list[str]] = None,
    layers: Optional[list[int]] = None,
    n_samples: int = SENSITIVITY_N_SAMPLES,
    max_seq_len: int = 2048,
    model_name: str = SENSITIVITY_MODEL,
    load_model_and_tokenizer: Optional[Callable[..., tuple[Any, Any]]] = None,
    extract_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Extract ONLY the missing chat-direction for the TARGET concept.

    This is the resumable extraction path for the single concept
    (``python_valid_vs_syntax_error``) whose chat direction is NOT in the old
    chat results. It is GATED: it loads a real model + tokenizer and runs the
    DiM extraction with ``use_chat_template=True``, persisting vectors into
    ``<chat_target_vectors_dir>`` using the standard
    ``save_concept_vectors`` layout so later sensitivity runs can read them
    back with ``load_concept_vectors``.

    The extractor FAILS CLEARLY if:

    * ``chat_target_vectors_dir`` escapes the designated sensitivity output
      root or would land inside a primary store (path-isolation guard runs
      BEFORE any model load, so library callers are protected even without
      the CLI);
    * the model or tokenizer cannot be loaded;
    * the tokenizer has no ``apply_chat_template`` attribute, or the attribute
      is not callable (so we never silently fall back to raw text and label it
      'chat').

    Resumability: checkpoints already containing the target concept (loaded via
    ``load_concept_vectors``) are skipped, so interrupted runs can be re-invoked.

    Args:
        chat_target_vectors_dir: Root to write into; MUST nest strictly under
            ``sensitivity_output_root`` and outside any primary store.
        sensitivity_output_root: The designated sensitivity output root (same
            root that holds ``sensitivity.json``). Defaults to
            :data:`DEFAULT_SENSITIVITY_OUTPUT_ROOT`.
        raw_vectors_dir: Isolated primary raw store root; if supplied, the
            guard refuses any target dir nested inside it or its parent.
        chat_old_vectors_dir: Old chat results root; if supplied, the guard
            refuses any target dir nested inside it or its parent.
        checkpoints: Defaults to :data:`SENSITIVITY_CHECKPOINTS`.
        layers: Defaults to :data:`SENSITIVITY_LAYERS`.
        n_samples, max_seq_len: Extraction hyper-parameters.
        model_name: Model to extract (defaults to the sensitivity model).
        load_model_and_tokenizer: Injectable loader (defaults to
            ``src.concept_dynamics.load_model_and_tokenizer``) for testing.
        extract_fn: Injectable single-model extraction function (defaults to
            ``src.concept_dynamics.run_model_extraction``) for testing.

    Returns:
        A dict summarizing what was extracted / skipped per checkpoint.

    Raises:
        ValueError: if ``chat_target_vectors_dir`` escapes the sensitivity
            output root or lies inside a primary store.
        ChatTemplateUnavailableError: if the tokenizer has no usable chat
            template (the whole point of this extraction).
    """
    # Path-isolation guard: refuse to extract anywhere except strictly under
    # the designated sensitivity root, and never inside a primary store. Runs
    # BEFORE any model/tokenizer load so library callers cannot bypass it.
    _assert_chat_target_isolation(
        chat_target_vectors_dir,
        sensitivity_output_root=sensitivity_output_root,
        raw_vectors_dir=raw_vectors_dir,
        chat_old_vectors_dir=chat_old_vectors_dir,
    )

    # Lazy imports keep the module importable in offline / test environments.
    from src.concept_dynamics import (
        load_concept_vectors as _load_vectors,
        load_model_and_tokenizer as _default_loader,
        run_model_extraction as _default_extract,
    )
    from src.config import OLMO3_VARIANTS

    loader = load_model_and_tokenizer or _default_loader
    extract = extract_fn or _default_extract

    checkpoints = (
        list(checkpoints) if checkpoints is not None else list(SENSITIVITY_CHECKPOINTS)
    )
    layers = list(layers) if layers is not None else list(SENSITIVITY_LAYERS)
    model_config = OLMO3_VARIANTS[model_name]

    summary: dict[str, Any] = {
        "model": model_name,
        "concept": TARGET_CONCEPT,
        "checkpoints": {"extracted": [], "skipped_present": []},
        "layers": list(layers),
        "n_samples": n_samples,
        "chat_template_check": None,
    }

    # --- Fail-fast: load tokenizer once and verify chat template BEFORE any
    #     model work, so we never silently fall back to raw text. ---
    model, tokenizer = loader(model_config, checkpoints[0])
    chat_fn = getattr(tokenizer, "apply_chat_template", None)
    if not callable(chat_fn):
        raise ChatTemplateUnavailableError(
            f"Tokenizer for {model_config.hf_id!r} has no callable "
            "apply_chat_template; cannot extract a CHAT-direction for the "
            "target concept. Refusing to fall back to raw text."
        )
    summary["chat_template_check"] = {
        "tokenizer_has_apply_chat_template": True,
        "verified_at_checkpoint": checkpoints[0],
    }

    # Probe one formatting call so a broken template fails loudly here too.
    try:
        chat_fn(
            [{"role": "user", "content": "probe"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        raise ChatTemplateUnavailableError(
            f"tokenizer.apply_chat_template raised on a probe message for "
            f"{model_config.hf_id!r}: {e!r}. Cannot extract chat target."
        ) from e

    try:
        for checkpoint in checkpoints:
            already_done = True
            for layer_idx in layers:
                try:
                    vecs = _load_vectors(
                        chat_target_vectors_dir, model_name, layer_idx, checkpoint
                    )
                except FileNotFoundError:
                    already_done = False
                    break
                if TARGET_CONCEPT not in vecs:
                    already_done = False
                    break
            if already_done:
                summary["checkpoints"]["skipped_present"].append(checkpoint)
                continue

            stats = extract(
                model_config,
                [TARGET_CONCEPT],
                layers,
                n_samples,
                chat_target_vectors_dir,
                max_seq_len=max_seq_len,
                checkpoint=checkpoint,
                revision=checkpoint,
                use_chat_template=True,
            )
            summary["checkpoints"]["extracted"].append(
                {"checkpoint": checkpoint, "stats": stats}
            )
    finally:
        del model

    return summary


__all__ = [
    "SENSITIVITY_MODEL",
    "SENSITIVITY_CHECKPOINTS",
    "SENSITIVITY_LAYERS",
    "SENSITIVITY_N_SAMPLES",
    "SENSITIVITY_D_MODEL",
    "SENSITIVITY_CONCEPTS",
    "TARGET_CONCEPT",
    "RELATED_CONCEPTS",
    "CONTROL_CONCEPT",
    "PROTOCOL",
    "LIMITATIONS",
    "DEFAULT_SENSITIVITY_OUTPUT_ROOT",
    "MetadataMismatchError",
    "ChatTemplateUnavailableError",
    "VectorSpec",
    "SensitivityEntry",
    "validate_chat_metadata",
    "cosine_similarity",
    "norm_of",
    "compare_vectors",
    "load_raw_vector",
    "load_chat_vector",
    "write_sensitivity_json",
    "run_sensitivity_analysis",
    "extract_missing_chat_target",
    "_validate_output_filename",
]
