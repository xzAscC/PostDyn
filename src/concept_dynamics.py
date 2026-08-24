"""
Concept Dynamics Pipeline for Olmo-3-7B Post-Training Analysis.

Implements the DiM (Difference-in-Means) concept extraction pipeline
from "Tracing Concept Dynamics through Pretraining and Post-training":

    1. extract_layer_activations  — last-token hidden states at specified layers
    2. compute_concept_vector     — DiM direction r = mu+ - mu-, normalized r_hat
    3. cross_model_stability      — cos(r_k^t, r_k^t') across models (per concept)
    4. concept_gram_matrices      — cos(r_i^t, r_j^t) across concepts (per model)

All functions are testable without GPU: extract_layer_activations accepts any
object with the HF transformers calling convention (mockable).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.concept_steering import ConceptSteeringVector


# =============================================================================
# Sidecar schema v1 — versioned concept-vector provenance
# =============================================================================
#
# Legacy v0 sidecars carry only ``concepts`` / ``layer_idx`` / ``model_name``
# with no checkpoint, revision, protocol, max_seq_len, d_model, extraction
# contract, or source-text provenance.  Version 1 binds every field that
# determines reproducibility so that resume and metrics loading can reject
# sidecars whose extraction config or source texts have drifted.
#
# The v1 contract mirrors the hardened ``probe_activations`` sidecar pattern:
# per-concept ``positive_text_sha256`` / ``negative_text_sha256`` arrays and an
# aggregate ``source_fingerprint`` are recomputed from live source texts at
# load time and compared exactly.  A v0 sidecar is always rejected by strict
# validation (``validate_concept_sidecar``) — it must be migrated or
# re-extracted.

#: Sidecar schema identifier stamped in every v1 (and v0-tagged) JSON file.
SIDECAR_SCHEMA: str = "olmo_concept_vectors"

#: Current sidecar schema version.  v0 (legacy) is encoded as ``0``.
SIDECAR_VERSION: int = 1

#: Legacy sidecar version (no provenance fields).  Used only to *identify*
#: legacy files so they can be rejected by strict validation and migrated.
SIDECAR_VERSION_LEGACY: int = 0

#: Expected hidden dimensionality for the OLMo-3-7B model family.
EXPECTED_D_MODEL: int = 4096


# =============================================================================
# Layer Selection
# =============================================================================


def select_uniform_layers(n_layers: int, n: int = 10) -> list[int]:
    """Select n layer indices via the slide formula.

    For j = 0..n-1 and L layers: ell_j = round[(0.1 + 0.8*j/(n-1)) * (L-1)].
    For OLMo-3-7B (32 layers, n=10): [3, 6, 9, 11, 14, 17, 20, 22, 25, 28].

    Args:
        n_layers: Total number of transformer layers.
        n: Number of layers to select (default: 10).

    Returns:
        List of 0-indexed layer indices in slide range (10% to 90% of L-1).
    """
    if n <= 0:
        return []
    if n == 1:
        return [int(round(0.5 * (n_layers - 1)))]
    span = max(n_layers - 1, 0)
    return [int(round((0.1 + 0.8 * j / (n - 1)) * span)) for j in range(n)]


# =============================================================================
# Result Data Structure
# =============================================================================


@dataclass
class ConceptVector:
    """Container for a single concept direction at a specific layer and model.

    Attributes:
        concept_name: e.g. "python_vs_cpp", "french_vs_english_language"
        model_name: e.g. "olmo3-think-sft"
        layer_idx: 0-indexed transformer layer index
        steering_vector: The direction to use for analysis (normalized r_hat
                         by default, per paper requirement)
        raw_direction: Unnormalized DiM direction r = mu+ - mu-
        positive_mean: Mean of positive-class activations
        negative_mean: Mean of negative-class activations
        positive_std: Std of positive-class activations
        negative_std: Std of negative-class activations
        n_positive: Number of positive samples
        n_negative: Number of negative samples
        d_model: Hidden dimensionality
    """

    concept_name: str
    model_name: str
    layer_idx: int
    steering_vector: torch.Tensor
    raw_direction: torch.Tensor
    positive_mean: torch.Tensor
    negative_mean: torch.Tensor
    positive_std: torch.Tensor
    negative_std: torch.Tensor
    n_positive: int
    n_negative: int
    d_model: int


# =============================================================================
# Activation Extraction
# =============================================================================


def extract_layer_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    max_seq_len: int = 512,
    use_chat_template: bool = True,
) -> dict[int, torch.Tensor]:
    """Extract last-token hidden states at specified layers.

    For each text, runs a forward pass with output_hidden_states=True and
    collects the last-token hidden state from each requested layer.

    Convention (matching HF transformers):
        outputs.hidden_states is a tuple of (num_layers + 1) tensors.
        Index 0 = embedding layer output.
        Index i (1..num_layers) = i-th transformer layer output.

    Args:
        model: A HF transformers model (or mock) supporting
            model(**inputs, output_hidden_states=True).
        tokenizer: A tokenizer returning {input_ids, attention_mask}.
        texts: List of input strings.
        layers: List of 0-indexed transformer layer indices to extract.
        max_seq_len: Maximum sequence length for tokenization.
        use_chat_template: If True (default) and the tokenizer exposes
            ``apply_chat_template``, each text is wrapped as a single
            ``[{"role": "user", "content": text}]`` message before
            tokenization. Any failure inside ``apply_chat_template`` falls
            back to the raw text so a missing or partial template cannot
            abort the whole extraction run.

    Returns:
        {layer_idx: (n_texts, d_model)} tensor of last-token activations.

    Raises:
        ValueError: If a requested layer index is out of range.
    """
    if not texts:
        d_model = _detect_d_model(model)
        return {layer: torch.empty(0, d_model) for layer in layers}

    # Validate layer indices
    n_model_layers = _detect_num_layers(model)
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= n_model_layers:
            raise ValueError(
                f"Layer index {layer_idx} out of range for model with "
                f"{n_model_layers} layers"
            )

    chat_template_fn = _resolve_chat_template_fn(tokenizer, use_chat_template)
    formatted_texts = [
        _apply_chat_template_safely(chat_template_fn, text, fallback=text)
        for text in texts
    ]

    layer_features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    device = getattr(model, "device", torch.device("cpu"))
    batch_size = 1
    if hasattr(model, "parameters"):
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass

    for start in range(0, len(formatted_texts), batch_size):
        batch_texts = formatted_texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            last_indices = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[1] - 1,
                device=inputs["input_ids"].device,
                dtype=torch.long,
            )
        else:
            last_indices = attention_mask.to(dtype=torch.long).sum(dim=1) - 1

        batch_idx = torch.arange(
            last_indices.shape[0], device=last_indices.device, dtype=torch.long
        )
        for layer_idx in layers:
            hs = hidden_states[layer_idx + 1]
            last_tok = hs[batch_idx, last_indices, :].detach().cpu().float()
            for row in range(last_tok.shape[0]):
                layer_features[layer_idx].append(last_tok[row])

        del outputs, hidden_states, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {layer: torch.stack(layer_features[layer], dim=0) for layer in layers}


def _resolve_chat_template_fn(tokenizer, use_chat_template: bool):
    """Return ``tokenizer.apply_chat_template`` when it should be used, else None."""
    if not use_chat_template:
        return None
    fn = getattr(tokenizer, "apply_chat_template", None)
    if fn is None or not callable(fn):
        return None
    return fn


def _apply_chat_template_safely(chat_template_fn, text: str, *, fallback: str) -> str:
    """Wrap ``text`` as a user message via ``chat_template_fn`` or return ``fallback``."""
    if chat_template_fn is None:
        return fallback
    try:
        return chat_template_fn(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return fallback


def _detect_num_layers(model) -> int:
    """Detect number of transformer layers from a model or its config."""
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, attr):
                return getattr(config, attr)
    # Fall back to counting model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError("Cannot detect number of layers for model")


def _detect_d_model(model) -> int:
    """Detect hidden dimensionality from a model or its config."""
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            if hasattr(config, attr):
                return getattr(config, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if len(layers) > 0:
            return layers[0].hidden_size if hasattr(layers[0], "hidden_size") else 0
    return 0


# =============================================================================
# DiM Concept Vector Computation
# =============================================================================


def compute_concept_vector(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    concept_name: str = "",
    model_name: str = "",
    layer_idx: int = 0,
    normalize: bool = True,
    eps: float = 1e-10,
) -> ConceptVector:
    """Compute a DiM concept direction with optional normalization.

    Following the paper:
        r   = mu+ - mu-              (difference-in-means)
        r_hat = r / ||r||_2          (normalized, default)

    The normalized direction r_hat is stored in ``steering_vector`` so that
    steering strength is controlled only by a scalar coefficient, not by
    the norm of the estimated direction.

    Args:
        positive_activations: (n_positive, d_model)
        negative_activations: (n_negative, d_model)
        concept_name, model_name, layer_idx: metadata
        normalize: If True (default), steering_vector = r / ||r||.
        eps: Numerical stability guard for zero-norm directions.

    Returns:
        ConceptVector with all statistics.

    Raises:
        ValueError: If tensors are not 2D or d_model dimensions mismatch.
    """
    if positive_activations.dim() != 2 or negative_activations.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors (n_samples, d_model), got "
            f"{positive_activations.dim()}D and {negative_activations.dim()}D"
        )

    n_pos, d_pos = positive_activations.shape
    n_neg, d_neg = negative_activations.shape

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"Need at least one positive and one negative sample; "
            f"got n_pos={n_pos}, n_neg={n_neg}"
        )

    if d_pos != d_neg:
        raise ValueError(
            f"d_model mismatch: positive has {d_pos}, negative has {d_neg}"
        )

    # Means
    positive_mean = positive_activations.mean(dim=0)
    negative_mean = negative_activations.mean(dim=0)

    # Std with Bessel's correction when n > 1
    pos_correction = 1 if n_pos > 1 else 0
    neg_correction = 1 if n_neg > 1 else 0
    positive_std = positive_activations.std(dim=0, correction=pos_correction)
    negative_std = negative_activations.std(dim=0, correction=neg_correction)

    # DiM direction
    raw_direction = positive_mean - negative_mean

    # Normalization (paper requirement: r_hat = r / ||r||)
    if normalize:
        norm = raw_direction.norm(p=2)
        if norm.item() < eps:
            # Zero direction (pos == neg) — avoid division by zero
            steering_vector = raw_direction.clone()
        else:
            steering_vector = raw_direction / norm
    else:
        steering_vector = raw_direction

    return ConceptVector(
        concept_name=concept_name,
        model_name=model_name,
        layer_idx=layer_idx,
        steering_vector=steering_vector,
        raw_direction=raw_direction,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        positive_std=positive_std,
        negative_std=negative_std,
        n_positive=n_pos,
        n_negative=n_neg,
        d_model=d_pos,
    )


# =============================================================================
# Cross-Model Directional Stability
# =============================================================================


def cross_model_stability(
    vectors: dict[str, ConceptVector],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute cosine similarity matrix of a concept across models.

        stability(k; t, t') = cos(r_k^t, r_k^t')

    All input vectors must be for the SAME concept at the SAME layer,
    from different models.

    Args:
        vectors: {model_name: ConceptVector} — one concept, one layer, N models.
        eps: Numerical stability constant.

    Returns:
        (N, N) symmetric cosine similarity matrix. Diagonal ≈ 1.0.
        Axes follow sorted model names.
    """
    names = sorted(vectors.keys())
    stacked = torch.stack([vectors[n].steering_vector for n in names])
    return _cosine_matrix(stacked, eps=eps)


# =============================================================================
# Concept Gram Matrix (Entanglement)
# =============================================================================


def concept_gram_matrices(
    vectors: dict[str, ConceptVector],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute pairwise cosine similarity of concept vectors (entanglement).

        G_ij^t = cos(r_i^t, r_j^t)

    All input vectors must be for the SAME model at the SAME layer,
    for different concepts.

    Args:
        vectors: {concept_name: ConceptVector} — one model, one layer, N concepts.
        eps: Numerical stability constant.

    Returns:
        (N, N) symmetric cosine similarity matrix. Diagonal ≈ 1.0.
        Axes follow sorted concept names.
    """
    names = sorted(vectors.keys())
    stacked = torch.stack([vectors[n].steering_vector for n in names])
    return _cosine_matrix(stacked, eps=eps)


# =============================================================================
# Helper
# =============================================================================


def _cosine_matrix(rows: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Pairwise cosine similarity of rows in a (n, d) tensor.

    Returns (n, n) symmetric matrix with diagonal ≈ 1.0.
    """
    norms = rows.norm(dim=1, keepdim=True)
    normalized = rows / norms.clamp(min=eps)
    return normalized @ normalized.T


# =============================================================================
# Conversion helpers (interop with existing concept_steering)
# =============================================================================


def to_steering_vector(cv: ConceptVector) -> ConceptSteeringVector:
    """Convert ConceptVector to the existing ConceptSteeringVector type.

    This allows reuse of save_steering_vectors / load_steering_vectors
    from src.concept_steering for persistence.
    """
    return ConceptSteeringVector(
        concept_name=cv.concept_name,
        steering_vector=cv.steering_vector,
        positive_mean=cv.positive_mean,
        negative_mean=cv.negative_mean,
        positive_std=cv.positive_std,
        negative_std=cv.negative_std,
        n_positive=cv.n_positive,
        n_negative=cv.n_negative,
        d_model=cv.d_model,
    )


# =============================================================================
# Concept source provenance (v1 sidecar fingerprints)
# =============================================================================


def text_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_concept_source_fingerprint(
    concept_name: str,
    positive_texts: list[str],
    negative_texts: list[str],
) -> str:
    """Deterministic SHA-256 over one concept's ordered paired source texts.

    For each index ``i`` the entry
    ``{concept_name, positive_text_sha256, negative_text_sha256}`` is
    canonicalised (sorted keys, minimal separators) and fed to the hasher.
    Changing any text, swapping order, or altering ``concept_name`` produces
    a different digest.
    """
    if len(positive_texts) != len(negative_texts):
        raise ValueError(
            f"concept {concept_name!r}: positive ({len(positive_texts)}) and "
            f"negative ({len(negative_texts)}) text counts differ"
        )
    hasher = hashlib.sha256()
    for pos, neg in zip(positive_texts, negative_texts):
        entry = json.dumps(
            {
                "concept_name": concept_name,
                "positive_text_sha256": text_sha256(pos),
                "negative_text_sha256": text_sha256(neg),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        hasher.update(entry.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def compute_sidecar_source_fingerprint(
    entries: list[tuple[str, str]],
) -> str:
    """Deterministic SHA-256 over the ordered ``(concept_name, fingerprint)`` pairs.

    ``entries`` MUST be sorted by concept name so the digest is stable regardless
    of insertion order.
    """
    hasher = hashlib.sha256()
    for concept_name, per_concept_fp in entries:
        entry = json.dumps(
            {
                "concept_name": concept_name,
                "source_fingerprint": per_concept_fp,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        hasher.update(entry.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def build_concept_source_entries(
    concept_sources: dict[str, tuple[list[str], list[str]]],
) -> list[tuple[str, str, list[str], list[str]]]:
    """Return ``[(concept_name, fingerprint, pos_sha_list, neg_sha_list)]`` sorted by name."""
    entries: list[tuple[str, str, list[str], list[str]]] = []
    for name in sorted(concept_sources.keys()):
        pos_texts, neg_texts = concept_sources[name]
        fp = compute_concept_source_fingerprint(name, pos_texts, neg_texts)
        pos_sha = [text_sha256(t) for t in pos_texts]
        neg_sha = [text_sha256(t) for t in neg_texts]
        entries.append((name, fp, pos_sha, neg_sha))
    return entries


# =============================================================================
# Persistence (per model × layer)
# =============================================================================


def save_concept_vectors(
    vectors: dict[str, ConceptVector],
    output_dir: str,
    model_name: str,
    layer_idx: int,
    checkpoint: str = "final",
    *,
    protocol: str | None = None,
    revision: str | None = None,
    hf_id: str | None = None,
    max_seq_len: int | None = None,
    use_chat_template: bool | None = None,
    concept_sources: dict[str, tuple[list[str], list[str]]] | None = None,
) -> str:
    """Save all concept vectors for one (model, checkpoint, layer) triple.

    Layout: {output_dir}/{model_name}/{checkpoint}/layer_{layer_idx}.{safetensors,json}

    Publication is crash-safe and symlink-resistant, matching the hardened
    ``save_layer_activations`` pattern:

      * Both files are written to secure unique temp paths (``tempfile.mkstemp``
        with O_CREAT|O_EXCL) inside the destination directory, never a
        predictable ``.tmp`` name an attacker could pre-create as a symlink.
      * The safetensors tensor is published (``os.replace``) BEFORE the JSON
        sidecar, so any reader observing the sidecar is guaranteed the tensor
        is already visible.
      * The JSON sidecar is fsynced before publication.
      * On any failure before publication, both temp files are removed.

    **Sidecar schema** — when ``concept_sources`` is provided the sidecar is
    written as v1 (:data:`SIDECAR_VERSION`), binding ``schema``, ``version``,
    ``protocol``, ``checkpoint``, ``revision``, ``hf_id``, ``layer_idx``,
    ``max_seq_len``, ``use_chat_template``, ``d_model``, an aggregate
    ``source_fingerprint``, and per-concept ``positive_text_sha256`` /
    ``negative_text_sha256`` / ``source_fingerprint``.  When ``concept_sources``
    is ``None`` a legacy v0 sidecar (``version`` = 0) is written; strict
    validation (:func:`validate_concept_sidecar`) will reject it.

    Tensor keys (``concept_XXXX.{field}``), file names, and the return value
    are unchanged from the legacy implementation.
    """
    ckpt_dir = os.path.join(output_dir, model_name, checkpoint)
    os.makedirs(ckpt_dir, exist_ok=True)
    base_path = os.path.join(ckpt_dir, f"layer_{layer_idx}")
    final_safetensors = base_path + ".safetensors"
    final_json = base_path + ".json"

    sorted_names = sorted(vectors.keys())
    tensor_dict: dict[str, torch.Tensor] = {}

    is_v1 = concept_sources is not None
    source_entries_by_name: dict[str, tuple[str, str, list[str], list[str]]] = {}
    aggregate_fp: str = ""

    if is_v1:
        cs: dict[str, tuple[list[str], list[str]]] = concept_sources
        filtered_sources = {name: cs[name] for name in sorted_names if name in cs}
        source_entries = build_concept_source_entries(filtered_sources)
        source_entries_by_name = {e[0]: e for e in source_entries}
        aggregate_fp = compute_sidecar_source_fingerprint(
            [(e[0], e[1]) for e in source_entries]
        )

    metadata: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "version": SIDECAR_VERSION if is_v1 else SIDECAR_VERSION_LEGACY,
        "concepts": [],
        "layer_idx": layer_idx,
        "model_name": model_name,
    }
    if is_v1:
        metadata["checkpoint"] = checkpoint
        metadata["protocol"] = protocol if protocol is not None else "raw"
        metadata["revision"] = revision
        metadata["hf_id"] = hf_id
        metadata["max_seq_len"] = max_seq_len
        metadata["use_chat_template"] = (
            use_chat_template if use_chat_template is not None else False
        )
        metadata["source_fingerprint"] = aggregate_fp
        metadata["provenance_origin"] = "extraction"

    for idx, name in enumerate(sorted_names):
        cv = vectors[name]
        prefix = f"concept_{idx:04d}"
        for field in (
            "steering_vector",
            "raw_direction",
            "positive_mean",
            "negative_mean",
            "positive_std",
            "negative_std",
        ):
            tensor = getattr(cv, field)
            tensor_dict[f"{prefix}.{field}"] = tensor.contiguous().to(torch.float32)

        entry: dict[str, Any] = {
            "name": name,
            "n_positive": cv.n_positive,
            "n_negative": cv.n_negative,
            "d_model": cv.d_model,
        }
        if is_v1:
            se = source_entries_by_name.get(name)
            if se is not None:
                _, fp, pos_sha, neg_sha = se
                entry["positive_text_sha256"] = pos_sha
                entry["negative_text_sha256"] = neg_sha
                entry["source_fingerprint"] = fp
        metadata["concepts"].append(entry)

    if is_v1 and sorted_names:
        metadata["d_model"] = vectors[sorted_names[0]].d_model

    tmp_safetensors = _secure_temp_path(ckpt_dir, suffix=".safetensors.tmp")
    tmp_json = _secure_temp_path(ckpt_dir, suffix=".json.tmp")
    try:
        save_file(tensor_dict, tmp_safetensors)
        _write_json_file(tmp_json, metadata)
        os.replace(tmp_safetensors, final_safetensors)
        tmp_safetensors = None
        os.replace(tmp_json, final_json)
        tmp_json = None
    finally:
        if tmp_safetensors is not None:
            _safe_remove(tmp_safetensors)
        if tmp_json is not None:
            _safe_remove(tmp_json)

    return base_path


def _secure_temp_path(directory: str, *, suffix: str = ".tmp") -> str:
    """Return a unique, unpredictable temp file path inside ``directory``.

    Uses ``tempfile.mkstemp`` (O_CREAT|O_EXCL, mode 0600, randomized name) so
    each call gets a fresh path even under concurrent extraction, defeating
    symlink-injection attacks where an attacker pre-places a symlink at a
    predictable temp name. The placeholder fd is closed and removed so the
    caller can write the path itself.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=".cd_", suffix=suffix, dir=directory)
    os.close(fd)
    os.remove(tmp_path)
    return tmp_path


def _write_json_file(path: str, payload: dict[str, Any]) -> None:
    """Write JSON to a caller-chosen path with fsync (no rename)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _safe_remove(path: str) -> None:
    """Remove a path, swallowing OSError (best-effort cleanup)."""
    try:
        os.remove(path)
    except OSError:
        pass


def load_concept_vectors(
    input_dir: str,
    model_name: str,
    layer_idx: int,
    checkpoint: str = "final",
) -> dict[str, ConceptVector]:
    """Load concept vectors for one (model, checkpoint, layer) triple."""
    import json
    import os

    from safetensors import safe_open

    base_path = os.path.join(input_dir, model_name, checkpoint, f"layer_{layer_idx}")
    safetensors_path = base_path + ".safetensors"
    json_path = base_path + ".json"

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Not found: {safetensors_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")

    with open(json_path) as f:
        metadata = json.load(f)

    vectors: dict[str, ConceptVector] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for idx, entry in enumerate(metadata["concepts"]):
            name = entry["name"]
            prefix = f"concept_{idx:04d}"

            def _get(field):
                return f.get_tensor(f"{prefix}.{field}")

            vectors[name] = ConceptVector(
                concept_name=name,
                model_name=metadata["model_name"],
                layer_idx=metadata["layer_idx"],
                steering_vector=_get("steering_vector"),
                raw_direction=_get("raw_direction"),
                positive_mean=_get("positive_mean"),
                negative_mean=_get("negative_mean"),
                positive_std=_get("positive_std"),
                negative_std=_get("negative_std"),
                n_positive=entry["n_positive"],
                n_negative=entry["n_negative"],
                d_model=entry["d_model"],
            )

    return vectors


def load_concept_sidecar(
    input_dir: str,
    model_name: str,
    layer_idx: int,
    checkpoint: str = "final",
) -> dict[str, Any]:
    """Load the raw JSON sidecar dict for a (model, checkpoint, layer) triple."""
    base_path = os.path.join(input_dir, model_name, checkpoint, f"layer_{layer_idx}")
    json_path = base_path + ".json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Not found: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _read_concept_tensor_shapes(
    safetensors_path: str,
    n_concepts: int,
) -> list[tuple[int, ...]] | None:
    """Return the ``raw_direction`` shape for each concept index, or ``None``.

    Shapes are read metadata-only (no tensor data loaded) so the check is
    cheap even for large ``d_model``.
    """
    try:
        shapes: list[tuple[int, ...]] = []
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for idx in range(n_concepts):
                key = f"concept_{idx:04d}.raw_direction"
                shapes.append(tuple(f.get_slice(key).get_shape()))
        return shapes
    except Exception:
        return None


def validate_concept_sidecar(
    sidecar: dict[str, Any],
    *,
    expected_model_name: str | None = None,
    expected_checkpoint: str | None = None,
    expected_layer_idx: int | None = None,
    expected_d_model: int | None = None,
    expected_max_seq_len: int | None = None,
    expected_protocol: str | None = None,
    expected_use_chat_template: bool | None = None,
    expected_hf_id: str | None = None,
    expected_revision: str | None = None,
    expected_concept_sources: dict[str, tuple[list[str], list[str]]] | None = None,
    allow_migrated_provenance: bool = True,
    require_exact_concept_set: bool = True,
) -> bool:
    """Return ``True`` iff a v1 sidecar matches every provided expectation.

    A v0 (legacy) sidecar always returns ``False`` — it lacks provenance.
    When ``expected_concept_sources`` is provided, the per-concept source
    fingerprints and text-SHA arrays are re-derived from the live source texts
    and compared exactly. With ``require_exact_concept_set=True`` (default) the
    sidecar concept set must equal the expected set; set False for single-concept
    completeness checks against a multi-concept sidecar.
    """
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        return False
    if sidecar.get("version") != SIDECAR_VERSION:
        return False

    origin = sidecar.get("provenance_origin", "extraction")
    if origin not in ("extraction", "migrated_v0_assumed_canonical_sources"):
        return False
    if origin != "extraction" and not allow_migrated_provenance:
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
    if expected_hf_id is not None:
        if sidecar.get("hf_id") != expected_hf_id:
            return False
    if expected_revision is not None:
        if sidecar.get("revision") != expected_revision:
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

    if expected_use_chat_template is not None:
        recorded = sidecar.get("use_chat_template")
        if recorded is None or bool(recorded) != bool(expected_use_chat_template):
            return False

    if expected_concept_sources is not None:
        concepts_list = sidecar.get("concepts", [])
        if not isinstance(concepts_list, list):
            return False
        names = [
            c.get("name")
            for c in concepts_list
            if isinstance(c, dict) and isinstance(c.get("name"), str)
        ]
        if len(names) != len(set(names)):
            return False
        concepts_in_sidecar: dict[str, dict[str, Any]] = {
            c["name"]: c
            for c in concepts_list
            if isinstance(c, dict) and isinstance(c.get("name"), str)
        }
        expected_concept_names = set(expected_concept_sources.keys())
        sidecar_concept_names = set(concepts_in_sidecar.keys())
        if require_exact_concept_set:
            if sidecar_concept_names != expected_concept_names:
                return False
        elif not expected_concept_names.issubset(sidecar_concept_names):
            return False

        expected_entries: list[tuple[str, str]] = []
        for name in sorted(expected_concept_sources.keys()):
            pos_texts, neg_texts = expected_concept_sources[name]
            per_fp = compute_concept_source_fingerprint(name, pos_texts, neg_texts)
            expected_pos_sha = [text_sha256(t) for t in pos_texts]
            expected_neg_sha = [text_sha256(t) for t in neg_texts]
            expected_entries.append((name, per_fp))

            sc_concept = concepts_in_sidecar[name]
            if sc_concept.get("source_fingerprint") != per_fp:
                return False
            if sc_concept.get("positive_text_sha256") != expected_pos_sha:
                return False
            if sc_concept.get("negative_text_sha256") != expected_neg_sha:
                return False
            if int(sc_concept.get("n_positive", -1)) != len(pos_texts):
                return False
            if int(sc_concept.get("n_negative", -1)) != len(neg_texts):
                return False

        if require_exact_concept_set or sidecar_concept_names == expected_concept_names:
            aggregate_fp = compute_sidecar_source_fingerprint(expected_entries)
            if sidecar.get("source_fingerprint") != aggregate_fp:
                return False

    return True


def is_concept_layer_v1_complete(
    input_dir: str,
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
    expected_concept_sources: dict[str, tuple[list[str], list[str]]] | None = None,
) -> bool:
    """Return ``True`` iff a concept's v1 sidecar is present, valid, and compatible.

    Combines :func:`validate_concept_sidecar` with:
      * safetensors + JSON both exist and parse;
      * the concept is present with ``n_positive >= n_samples`` and
        ``n_negative >= n_samples``;
      * each ``raw_direction`` tensor is rank-1 with ``d_model`` matching the
        sidecar (coordinate check).
    """
    base_path = os.path.join(input_dir, model_name, checkpoint, f"layer_{layer_idx}")
    safetensors_path = base_path + ".safetensors"
    json_path = base_path + ".json"
    if not os.path.exists(safetensors_path) or not os.path.exists(json_path):
        return False
    try:
        with open(json_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

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
        # Completeness is checked one concept at a time against a multi-concept
        # layer sidecar; require subset match, not exact set equality.
        require_exact_concept_set=False,
    ):
        return False

    concepts_list = sidecar.get("concepts", [])
    concept_entry: dict[str, Any] | None = None
    concept_index = -1
    for idx, entry in enumerate(concepts_list):
        if isinstance(entry, dict) and entry.get("name") == concept:
            concept_entry = entry
            concept_index = idx
            break
    if concept_entry is None:
        return False
    if int(concept_entry.get("n_positive", -1)) < n_samples:
        return False
    if int(concept_entry.get("n_negative", -1)) < n_samples:
        return False

    shapes = _read_concept_tensor_shapes(safetensors_path, len(concepts_list))
    if shapes is None:
        return False
    if concept_index < 0 or concept_index >= len(shapes):
        return False
    rd_shape = shapes[concept_index]
    if len(rd_shape) != 1:
        return False
    sidecar_d = sidecar.get("d_model")
    if sidecar_d is not None and rd_shape[0] != int(sidecar_d):
        return False
    if expected_d_model is not None and rd_shape[0] != expected_d_model:
        return False

    return True


# =============================================================================
# Model Loading (bfloat16, device_map="auto")
# =============================================================================


def _reject_generic_32b_loading(model_config) -> None:
    if (
        getattr(model_config, "architecture", None) == "olmo3"
        and getattr(model_config, "total_params", None) == "32B"
    ):
        raise ValueError(
            "Generic concept dynamics does not support OLMo-3 32B loading; "
            "use the canonical NF4 experiment loader "
            "src.quantized_model_loader.load_olmo3_32b_think instead."
        )


def _clean_hf_cache(hf_id: str):
    """Remove a model's HF cache entries to free disk space."""
    import os
    import shutil

    cache_name = hf_id.replace("/", "--")
    cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{cache_name}")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
        print(f"  Cleaned HF cache: {cache_name}")


def _load_model_and_tokenizer(model_config, revision=None):
    """Load model (bfloat16) and tokenizer for a ModelConfig at a given revision."""
    _reject_generic_32b_loading(model_config)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rev = revision if revision else model_config.revision

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.hf_id,
            revision=rev,
        )
    except (KeyError, AttributeError):
        print(f"  Tokenizer load failed, falling back to Olmo-3 base tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/Olmo-3-1025-7B",
            revision="a81bae42db3975be1671e27b9c9a56da1a9f980f",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_id,
            revision=rev,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    except (KeyError, ValueError) as e:
        if "olmo2-retrofit" in str(e) or "olmo2_retrofit" in str(e):
            print(f"  olmo2-retrofit detected, loading via Olmo2ForCausalLM")
            from transformers import Olmo2ForCausalLM

            model = Olmo2ForCausalLM.from_pretrained(
                model_config.hf_id,
                revision=rev,
                dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            raise

    model.eval()
    return model, tokenizer


load_model_and_tokenizer = _load_model_and_tokenizer


# =============================================================================
# Single-Model Extraction Pipeline
# =============================================================================


def run_model_extraction(
    model_config,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    output_dir: str,
    max_seq_len: int = 2048,
    checkpoint: str = "final",
    revision: Optional[str] = None,
    use_chat_template: bool = True,
) -> dict[str, Any]:
    """Extract concept vectors for one model checkpoint at specified layers."""
    import gc
    import time

    from src.contrastive_datasets import load_contrastive_texts

    model_name = model_config.name
    start = time.time()

    eff_revision = revision if revision else model_config.revision

    print(f"\n{'=' * 60}")
    print(
        f"Concept extraction: {model_name} / {checkpoint} ({model_config.hf_id} rev={eff_revision})"
    )
    print(f"Concepts: {concepts}, Layers: {layers}, Samples: {n_samples}")
    print(f"Chat template: {'on' if use_chat_template else 'off'}")
    print(f"{'=' * 60}")

    # Step 1: Load all contrastive texts upfront (no model needed)
    concept_texts: dict[str, tuple[list[str], list[str]]] = {}
    for concept in concepts:
        print(f"  Loading contrastive texts for '{concept}'...")
        pos, neg = load_contrastive_texts(concept, n_samples=n_samples)
        concept_texts[concept] = (pos, neg)
        print(f"    positive={len(pos)}, negative={len(neg)}")

    # Step 2: Load model + tokenizer
    print(f"  Loading model {model_config.hf_id} rev={eff_revision} (bfloat16)...")
    model, tokenizer = _load_model_and_tokenizer(model_config, eff_revision)
    print(f"  Model loaded on {next(model.parameters()).device}")

    # Step 3: Extract activations per concept, compute vectors per layer
    try:
        for concept in concepts:
            pos_texts, neg_texts = concept_texts[concept]
            print(f"\n  Extracting activations for '{concept}'...")

            pos_acts = extract_layer_activations(
                model,
                tokenizer,
                pos_texts,
                layers,
                max_seq_len=max_seq_len,
                use_chat_template=use_chat_template,
            )
            neg_acts = extract_layer_activations(
                model,
                tokenizer,
                neg_texts,
                layers,
                max_seq_len=max_seq_len,
                use_chat_template=use_chat_template,
            )

            # Compute + save concept vector per layer
            for layer_idx in layers:
                cv = compute_concept_vector(
                    pos_acts[layer_idx],
                    neg_acts[layer_idx],
                    concept_name=concept,
                    model_name=model_name,
                    layer_idx=layer_idx,
                    normalize=True,
                )

                try:
                    existing = load_concept_vectors(
                        output_dir, model_name, layer_idx, checkpoint
                    )
                except FileNotFoundError:
                    existing = {}

                existing[concept] = cv
                save_concept_vectors(
                    existing, output_dir, model_name, layer_idx, checkpoint
                )

            del pos_acts, neg_acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"    Saved {len(layers)} layers for '{concept}'")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start
    print(f"\n  Done: {model_name} in {elapsed:.1f}s")

    return {
        "model": model_name,
        "checkpoint": checkpoint,
        "concepts": concepts,
        "layers": layers,
        "n_samples": n_samples,
        "use_chat_template": use_chat_template,
        "elapsed_seconds": round(elapsed, 1),
    }


# =============================================================================
# Full Experiment Runner (checkpoint trajectory)
# =============================================================================


def _manifest_covers(
    manifest: dict[str, Any] | None,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
) -> bool:
    if not manifest:
        return False
    per_concept = manifest.get("concept_samples")
    if isinstance(per_concept, dict) and per_concept:
        for concept in concepts:
            if int(per_concept.get(concept, 0)) < n_samples:
                return False
    else:
        stored_concepts = set(manifest.get("concepts", []))
        if not set(concepts).issubset(stored_concepts):
            return False
        if int(manifest.get("n_samples", 0)) < n_samples:
            return False
    per_layer_concepts = manifest.get("layer_concepts")
    if isinstance(per_layer_concepts, dict) and per_layer_concepts:
        layer_set = set(layers)
        for layer in layers:
            entries = (
                per_layer_concepts.get(str(layer))
                or per_layer_concepts.get(layer)
                or []
            )
            if not set(concepts).issubset(set(entries)):
                return False
        _ = layer_set
    else:
        stored_layers = set(manifest.get("layers", []))
        if not set(layers).issubset(stored_layers):
            return False
    return True


def _merge_manifest(
    existing: dict[str, Any] | None,
    concepts: list[str],
    layers: list[int],
    n_samples: int,
) -> dict[str, Any]:
    prior = existing or {}
    concept_samples = dict(prior.get("concept_samples") or {})
    for concept in concepts:
        concept_samples[concept] = max(int(concept_samples.get(concept, 0)), n_samples)
    layer_concepts_raw = dict(prior.get("layer_concepts") or {})
    layer_concepts: dict[str, list[str]] = {}
    for key, value in layer_concepts_raw.items():
        layer_concepts[str(key)] = list(value)
    for layer in layers:
        current = set(layer_concepts.get(str(layer), []))
        current.update(concepts)
        layer_concepts[str(layer)] = sorted(current)
    return {
        "concepts": sorted(set(prior.get("concepts", [])) | set(concepts)),
        "layers": sorted(set(prior.get("layers", [])) | set(layers)),
        "n_samples": max(int(prior.get("n_samples", 0)), n_samples),
        "concept_samples": concept_samples,
        "layer_concepts": layer_concepts,
    }


def run_full_experiment(
    model_names: list[str],
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    output_dir: str,
    max_seq_len: int = 2048,
    clean_hf_cache: bool = True,
    use_chat_template: bool = True,
    max_checkpoints_per_model: int | None = None,
    checkpoint_override: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Run concept extraction across all models × checkpoints, then dynamics."""
    import json
    import os

    from src.config import OLMO3_VARIANTS, MODEL_CHECKPOINTS

    os.makedirs(output_dir, exist_ok=True)
    vectors_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "extraction_results.json")

    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results.get('checkpoints_done', []))} ckpts done")
    else:
        all_results: dict[str, Any] = {
            "checkpoints_done": [],
            "extraction": {},
            "checkpoint_manifests": {},
        }
    all_results.setdefault("checkpoint_manifests", {})

    for name in model_names:
        if name not in OLMO3_VARIANTS:
            print(f"WARNING: '{name}' not in OLMO3_VARIANTS, skipping")
            continue

        config = OLMO3_VARIANTS[name]
        if checkpoint_override and name in checkpoint_override:
            checkpoints = list(checkpoint_override[name])
        else:
            checkpoints = list(MODEL_CHECKPOINTS.get(name, ["main"]))
            if max_checkpoints_per_model is not None:
                checkpoints = checkpoints[: max(0, max_checkpoints_per_model)]

        for ckpt in checkpoints:
            ckpt_key = f"{name}/{ckpt}"
            manifest = all_results["checkpoint_manifests"].get(ckpt_key)
            if ckpt_key in all_results["checkpoints_done"] and _manifest_covers(
                manifest, concepts, layers, n_samples
            ):
                print(f"\nSkipping {ckpt_key} (already done, request covered)")
                continue
            if ckpt_key in all_results["checkpoints_done"]:
                print(
                    f"\nRe-running {ckpt_key} "
                    f"(stored manifest does not cover current request)"
                )

            try:
                stats = run_model_extraction(
                    config,
                    concepts,
                    layers,
                    n_samples,
                    vectors_dir,
                    max_seq_len,
                    checkpoint=ckpt,
                    revision=ckpt,
                    use_chat_template=use_chat_template,
                )
                all_results["extraction"][ckpt_key] = stats
                if ckpt_key not in all_results["checkpoints_done"]:
                    all_results["checkpoints_done"].append(ckpt_key)
                all_results["checkpoint_manifests"][ckpt_key] = _merge_manifest(
                    manifest, concepts, layers, n_samples
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                all_results["extraction"][ckpt_key] = {"error": str(e)}

            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # Clean HF cache for this model to free disk space
        model_ckpts_done = all(
            f"{name}/{c}" in all_results["checkpoints_done"] for c in checkpoints
        )
        if model_ckpts_done and clean_hf_cache:
            _clean_hf_cache(config.hf_id)

    print(f"\n{'=' * 60}")
    print("Computing checkpoint trajectory dynamics...")
    print(f"{'=' * 60}")
    dynamics = compute_dynamics_analysis(output_dir, model_names, concepts, layers)

    all_results["dynamics"] = dynamics
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    return all_results


# =============================================================================
# Dynamics Analysis (per-model checkpoint stability + gram)
# =============================================================================


def compute_dynamics_analysis(
    results_dir: str,
    model_names: list[str],
    concepts: list[str],
    layers: list[int],
) -> dict[str, Any]:
    """Compute per-model checkpoint stability and per-checkpoint gram matrices.

    Stability: for each (model, concept, layer), cosine matrix across
    that model's checkpoints (NxN where N = num checkpoints).

    Gram: for each (model, checkpoint, layer), 4x4 cosine across concepts.
    """
    import json
    import os

    from src.config import MODEL_CHECKPOINTS

    vectors_dir = os.path.join(results_dir, "vectors")
    stability_dir = os.path.join(results_dir, "stability")
    gram_dir = os.path.join(results_dir, "gram")
    os.makedirs(stability_dir, exist_ok=True)
    os.makedirs(gram_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "extraction_results.json")
    completed_keys: set[str] | None = None
    if os.path.exists(results_path):
        with open(results_path) as f:
            extraction_results = json.load(f)
        completed_keys = set(extraction_results.get("checkpoints_done", []))

    available_models = [
        m for m in model_names if os.path.exists(os.path.join(vectors_dir, m))
    ]

    # --- Stability: per model, per concept, per layer, across checkpoints ---
    stability: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for model in available_models:
        stability[model] = {}
        ckpts = MODEL_CHECKPOINTS.get(model, ["main"])
        available_ckpts = [
            c
            for c in ckpts
            if os.path.exists(os.path.join(vectors_dir, model, c))
            and (completed_keys is None or f"{model}/{c}" in completed_keys)
        ]

        for concept in concepts:
            stability[model][concept] = {}
            for layer in layers:
                per_ckpt: dict[str, ConceptVector] = {}
                for ckpt in available_ckpts:
                    try:
                        vecs = load_concept_vectors(vectors_dir, model, layer, ckpt)
                        if concept in vecs:
                            per_ckpt[ckpt] = vecs[concept]
                    except FileNotFoundError:
                        continue

                if len(per_ckpt) >= 2:
                    matrix = cross_model_stability(per_ckpt)
                    stability[model][concept][layer] = {
                        "matrix": matrix.tolist(),
                        "checkpoints": sorted(per_ckpt.keys()),
                    }

    with open(os.path.join(stability_dir, "stability.json"), "w") as f:
        json.dump(stability, f, indent=2)

    # --- Gram: per model, per checkpoint, per layer, across concepts ---
    gram: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for model in available_models:
        gram[model] = {}
        ckpts = MODEL_CHECKPOINTS.get(model, ["main"])
        available_ckpts = [
            c
            for c in ckpts
            if os.path.exists(os.path.join(vectors_dir, model, c))
            and (completed_keys is None or f"{model}/{c}" in completed_keys)
        ]

        for ckpt in available_ckpts:
            gram[model][ckpt] = {}
            for layer in layers:
                try:
                    vecs = load_concept_vectors(vectors_dir, model, layer, ckpt)
                except FileNotFoundError:
                    continue

                avail = {c: vecs[c] for c in concepts if c in vecs}
                if len(avail) >= 2:
                    matrix = concept_gram_matrices(avail)
                    gram[model][ckpt][layer] = {
                        "matrix": matrix.tolist(),
                        "concepts": sorted(avail.keys()),
                    }

    with open(os.path.join(gram_dir, "gram.json"), "w") as f:
        json.dump(gram, f, indent=2)

    n_stab = sum(
        len(layers_data)
        for model_data in stability.values()
        for layers_data in model_data.values()
    )
    n_gram = sum(
        len(layers_data)
        for ckpt_data in gram.values()
        for layers_data in ckpt_data.values()
    )
    print(f"  Stability: {n_stab} matrices")
    print(f"  Gram: {n_gram} matrices")

    return {
        "stability": stability,
        "gram": gram,
        "model_names": available_models,
        "concepts": concepts,
        "layers": layers,
    }
