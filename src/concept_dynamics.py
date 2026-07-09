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

from dataclasses import dataclass

import torch

from src.concept_steering import ConceptSteeringVector


# =============================================================================
# Layer Selection
# =============================================================================


def select_uniform_layers(n_layers: int, n: int = 10) -> list[int]:
    """Select n layers uniformly spaced at 10%, 20%, ..., 100% of depth.

    For OLMo-3-7B (32 layers, n=10): [3, 6, 9, 12, 16, 19, 22, 25, 28, 31].

    Args:
        n_layers: Total number of transformer layers.
        n: Number of layers to select (default: 10).

    Returns:
        List of 0-indexed layer indices.
    """
    percentages = [i / n for i in range(1, n + 1)]
    return [min(int(p * n_layers), n_layers - 1) for p in percentages]


# =============================================================================
# Result Data Structure
# =============================================================================


@dataclass
class ConceptVector:
    """Container for a single concept direction at a specific layer and model.

    Attributes:
        concept_name: e.g. "math", "code", "if", "general"
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

    # Accumulate per-layer last-token hidden states
    layer_features: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    d_model = None

    device = getattr(model, "device", torch.device("cpu"))

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        # Move inputs to model device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        last_token_idx = seq_len - 1

        for layer_idx in layers:
            # hidden_states[layer_idx + 1] because index 0 = embedding
            hs = hidden_states[layer_idx + 1]
            # Shape: (1, seq_len, d_model) → extract last token
            last_tok = hs[0, last_token_idx, :].detach().cpu().float()
            layer_features[layer_idx].append(last_tok)
            if d_model is None:
                d_model = last_tok.shape[0]

        del outputs, hidden_states, inputs

        # Periodic GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stack into (n_texts, d_model) per layer
    return {layer: torch.stack(layer_features[layer], dim=0) for layer in layers}


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
# Persistence (per model × layer)
# =============================================================================


def save_concept_vectors(
    vectors: dict[str, ConceptVector],
    output_dir: str,
    model_name: str,
    layer_idx: int,
) -> str:
    """Save all concept vectors for one (model, layer) pair.

    Layout: {output_dir}/{model_name}/layer_{layer_idx}.{safetensors,json}

    Args:
        vectors: {concept_name: ConceptVector} for one model at one layer.
        output_dir: Base output directory.
        model_name: Model identifier (used in path).
        layer_idx: Layer index (used in path).

    Returns:
        Base path (without extension) where vectors were saved.
    """
    import json
    import os

    from safetensors.torch import save_file

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    base_path = os.path.join(model_dir, f"layer_{layer_idx}")

    tensor_dict: dict[str, torch.Tensor] = {}
    metadata: dict = {"concepts": [], "layer_idx": layer_idx, "model_name": model_name}

    for idx, name in enumerate(sorted(vectors.keys())):
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

        metadata["concepts"].append(
            {
                "name": name,
                "n_positive": cv.n_positive,
                "n_negative": cv.n_negative,
                "d_model": cv.d_model,
            }
        )

    save_file(tensor_dict, base_path + ".safetensors")
    with open(base_path + ".json", "w") as f:
        json.dump(metadata, f, indent=2)

    return base_path


def load_concept_vectors(
    input_dir: str,
    model_name: str,
    layer_idx: int,
) -> dict[str, ConceptVector]:
    """Load concept vectors for one (model, layer) pair.

    Args:
        input_dir: Base directory where vectors were saved.
        model_name: Model identifier.
        layer_idx: Layer index.

    Returns:
        {concept_name: ConceptVector} for the specified model and layer.

    Raises:
        FileNotFoundError: If the safetensors or JSON file is missing.
    """
    import json
    import os

    from safetensors import safe_open

    base_path = os.path.join(input_dir, model_name, f"layer_{layer_idx}")
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


# =============================================================================
# Model Loading (bfloat16, device_map="auto")
# =============================================================================


def _load_model_and_tokenizer(model_config):
    """Load model (bfloat16) and tokenizer for a ModelConfig.

    Returns (model, tokenizer). Caller is responsible for cleanup.
    Handles olmo2-retrofit variants (e.g. RL-Zero-Mix) whose tokenizer
    config may be incomplete by falling back to the shared Olmo-3 base
    tokenizer, and loads them via Olmo2ForCausalLM since the retrofit
    architecture shares Olmo2's layer structure.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    trust = getattr(model_config, "architecture", "") == "olmo3"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.hf_id,
            revision=model_config.revision,
            trust_remote_code=trust,
        )
    except (KeyError, AttributeError):
        print(f"  Tokenizer load failed, falling back to Olmo-3 base tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            "allenai/Olmo-3-1025-7B",
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_id,
            revision=model_config.revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=trust,
        )
    except (KeyError, ValueError) as e:
        if "olmo2-retrofit" in str(e) or "olmo2_retrofit" in str(e):
            print(f"  olmo2-retrofit detected, loading via Olmo2ForCausalLM")
            from transformers import Olmo2ForCausalLM

            model = Olmo2ForCausalLM.from_pretrained(
                model_config.hf_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            raise

    model.eval()
    return model, tokenizer


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
) -> dict:
    """Extract concept vectors for one model at specified layers.

    Pipeline per (concept, layer):
        1. Load contrastive texts (Dolci positive + wikitext negative)
        2. Forward pass, extract last-token activations
        3. Compute DiM concept vector (normalized)
        4. Save

    Memory: one model loaded at a time. Activations extracted for all
    concepts before freeing the model.

    Args:
        model_config: ModelConfig from src.config.
        concepts: e.g. ["math", "code", "if", "general"].
        layers: e.g. [3, 6, 9, 12, 16, 19, 22, 25, 28, 31].
        n_samples: Number of positive/negative samples per concept.
        output_dir: Where to save concept vectors.
        max_seq_len: Max tokenization length.

    Returns:
        Summary dict with extracted concept counts per layer.
    """
    import gc
    import time

    from src.contrastive_datasets import load_contrastive_texts

    model_name = model_config.name
    start = time.time()

    print(f"\n{'=' * 60}")
    print(f"Concept extraction: {model_name} ({model_config.hf_id})")
    print(f"Concepts: {concepts}")
    print(f"Layers: {layers}")
    print(f"Samples per concept: {n_samples}")
    print(f"{'=' * 60}")

    # Step 1: Load all contrastive texts upfront (no model needed)
    concept_texts: dict[str, tuple[list[str], list[str]]] = {}
    for concept in concepts:
        print(f"  Loading contrastive texts for '{concept}'...")
        pos, neg = load_contrastive_texts(concept, n_samples=n_samples)
        concept_texts[concept] = (pos, neg)
        print(f"    positive={len(pos)}, negative={len(neg)}")

    # Step 2: Load model + tokenizer
    print(f"  Loading model {model_config.hf_id} (bfloat16)...")
    model, tokenizer = _load_model_and_tokenizer(model_config)
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
            )
            neg_acts = extract_layer_activations(
                model,
                tokenizer,
                neg_texts,
                layers,
                max_seq_len=max_seq_len,
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

                # Load existing vectors for this (model, layer) or start fresh
                try:
                    existing = load_concept_vectors(output_dir, model_name, layer_idx)
                except FileNotFoundError:
                    existing = {}

                existing[concept] = cv
                save_concept_vectors(existing, output_dir, model_name, layer_idx)

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
        "concepts": concepts,
        "layers": layers,
        "n_samples": n_samples,
        "elapsed_seconds": round(elapsed, 1),
    }


# =============================================================================
# Full Experiment Runner
# =============================================================================


def run_full_experiment(
    model_names: list[str],
    concepts: list[str],
    layers: list[int],
    n_samples: int,
    output_dir: str,
    max_seq_len: int = 2048,
) -> dict:
    """Run concept extraction across all models, then compute dynamics.

    Args:
        model_names: Keys into OLMO3_VARIANTS (e.g. "olmo3-think-sft").
        concepts: e.g. ["math", "code", "if", "general"].
        layers: Layer indices to probe.
        n_samples: Samples per concept per class.
        output_dir: Where to save all results.
        max_seq_len: Max tokenization length.

    Returns:
        Full results summary including per-model extraction stats and
        cross-model dynamics analysis.
    """
    import json
    import os

    from src.config import OLMO3_VARIANTS

    os.makedirs(output_dir, exist_ok=True)
    vectors_dir = os.path.join(output_dir, "vectors")
    os.makedirs(vectors_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "extraction_results.json")

    # Resume support: load existing results
    if os.path.exists(results_path):
        with open(results_path) as f:
            all_results = json.load(f)
        print(
            f"Resuming: {len(all_results.get('models_done', []))} models already done"
        )
    else:
        all_results = {"models_done": [], "extraction": {}}

    for name in model_names:
        if name in all_results["models_done"]:
            print(f"\nSkipping {name} (already extracted)")
            continue

        if name not in OLMO3_VARIANTS:
            print(f"\nWARNING: '{name}' not in OLMO3_VARIANTS, skipping")
            continue

        config = OLMO3_VARIANTS[name]
        try:
            stats = run_model_extraction(
                config,
                concepts,
                layers,
                n_samples,
                vectors_dir,
                max_seq_len,
            )
            all_results["extraction"][name] = stats
            all_results["models_done"].append(name)
        except Exception as e:
            import traceback

            traceback.print_exc()
            all_results["extraction"][name] = {"error": str(e)}
            all_results["models_done"].append(name)

        # Incremental save
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Step 2: Compute dynamics analysis
    print(f"\n{'=' * 60}")
    print("Computing cross-model dynamics...")
    print(f"{'=' * 60}")
    dynamics = compute_dynamics_analysis(
        output_dir,
        model_names,
        concepts,
        layers,
    )

    all_results["dynamics"] = dynamics
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    return all_results


# =============================================================================
# Dynamics Analysis (stability + gram matrices)
# =============================================================================


def compute_dynamics_analysis(
    results_dir: str,
    model_names: list[str],
    concepts: list[str],
    layers: list[int],
) -> dict:
    """Compute cross-model stability and per-model gram matrices.

    Loads all saved concept vectors and computes:
        1. stability[k; t, t'] = cos(r_k^t, r_k^t')  — per concept, per layer
        2. G_ij^t = cos(r_i^t, r_j^t)                — per model, per layer

    Args:
        results_dir: Directory containing saved concept vectors.
        model_names: Ordered list of model names.
        concepts: Ordered list of concept names.
        layers: Ordered list of layer indices.

    Returns:
        {
            "stability": {concept: {layer: {matrix: [[NxN]], models: [...]}}},
            "gram": {model: {layer: [[4x4 matrix]]}},
            "model_names": [...],
            "concepts": [...],
            "layers": [...],
        }
    """
    import json
    import os

    vectors_dir = os.path.join(results_dir, "vectors")
    stability_dir = os.path.join(results_dir, "stability")
    gram_dir = os.path.join(results_dir, "gram")
    os.makedirs(stability_dir, exist_ok=True)
    os.makedirs(gram_dir, exist_ok=True)

    available_models = [
        m for m in model_names if os.path.exists(os.path.join(vectors_dir, m))
    ]

    if len(available_models) < 2:
        print("WARNING: Need ≥2 models for stability analysis")

    # --- Stability: per concept, per layer, across models ---
    stability: dict[str, dict[int, dict]] = {}
    for concept in concepts:
        stability[concept] = {}
        for layer in layers:
            per_model: dict[str, ConceptVector] = {}
            for model in available_models:
                try:
                    vecs = load_concept_vectors(vectors_dir, model, layer)
                    if concept in vecs:
                        per_model[model] = vecs[concept]
                except FileNotFoundError:
                    continue

            if len(per_model) >= 2:
                matrix = cross_model_stability(per_model)
                sorted_models = sorted(per_model.keys())
                stability[concept][layer] = {
                    "matrix": matrix.tolist(),
                    "models": sorted_models,
                }

    # Save stability
    with open(os.path.join(stability_dir, "stability.json"), "w") as f:
        json.dump(stability, f, indent=2)

    # --- Gram: per model, per layer, across concepts ---
    gram: dict[str, dict[int, dict]] = {}
    for model in available_models:
        gram[model] = {}
        for layer in layers:
            try:
                vecs = load_concept_vectors(vectors_dir, model, layer)
            except FileNotFoundError:
                continue

            available_concepts = {c: vecs[c] for c in concepts if c in vecs}
            if len(available_concepts) >= 2:
                matrix = concept_gram_matrices(available_concepts)
                sorted_concepts = sorted(available_concepts.keys())
                gram[model][layer] = {
                    "matrix": matrix.tolist(),
                    "concepts": sorted_concepts,
                }

    # Save gram
    with open(os.path.join(gram_dir, "gram.json"), "w") as f:
        json.dump(gram, f, indent=2)

    print(f"  Stability: {sum(len(v) for v in stability.values())} matrices")
    print(f"  Gram: {sum(len(v) for v in gram.values())} matrices")

    return {
        "stability": stability,
        "gram": gram,
        "model_names": available_models,
        "concepts": concepts,
        "layers": layers,
    }
