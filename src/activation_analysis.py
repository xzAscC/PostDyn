"""
Activation-Level RankMe Analysis Module

Implements activation-space effective rank analysis following arxiv:2509.23024.
Extracts hidden states from LLM forward passes on MMLU Pro questions and computes
per-layer RankMe and alpha-ReQ metrics.

Core metrics:
    RankMe(F) = exp(-sum p_i * ln(p_i))  where p_i = lambda_i / sum(lambda_j)
    and lambda_i are eigenvalues of the centered feature covariance matrix.
    ratio     = RankMe / D  in (0, 1]
    alpha-ReQ = power-law decay rate of singular value spectrum

Feature matrix F in R^{N x d} where each row is a last-token hidden state.
Features are centered before SVD computation.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Optional, cast

import torch
from tqdm import tqdm

from src.config import (
    ModelConfig,
    PYTHIA_CONFIGS,
    PYTHIA_CHECKPOINTS,
    OLMO3_VARIANTS,
    RESULTS_DIR,
)


# =============================================================================
# Data Loading
# =============================================================================

def load_mmlu_questions(num_samples: Optional[int] = None) -> list[str]:
    """Load MMLU Pro questions. Returns list of raw question strings."""
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    examples = cast(Iterable[Mapping[str, object]], ds)
    questions = [question for example in examples if isinstance((question := example.get("question")), str)]
    if num_samples is not None:
        questions = questions[:num_samples]
    print(f"Loaded {len(questions)} MMLU Pro questions")
    return questions


# =============================================================================
# Result Data Classes
# =============================================================================

@dataclass
class ActivationResult:
    """Result from extracting hidden states from a model."""
    model_name: str
    layer_results: dict[int, dict]  # layer_idx -> {rankme, rankme_ratio, alpha_req, d_model, n_samples}
    num_questions: int
    elapsed_seconds: float


# =============================================================================
# RankMe Computation (activation matrices)
# =============================================================================

def compute_activation_rankme(features: torch.Tensor, eps: float = 1e-10) -> tuple[float, float]:
    """
    Compute RankMe of an activation feature matrix.

    Args:
        features: (N, D) tensor where N = samples, D = hidden dim

    Returns:
        (rankme, rankme_ratio) where rankme_ratio = rankme / D

    Algorithm (following Li et al. 2025):
    1. Center: F_centered = F - mean(F, dim=0)
    2. SVD: sigma = svdvals(F_centered)
    3. Convert to covariance eigenspectrum: lambda_i = sigma_i^2 / N
    4. Normalize: p_i = lambda_i / sum(lambda_j)
    5. Entropy: H = -sum(p_i * ln(p_i))
    6. RankMe = exp(H)
    7. Ratio = RankMe / D
    """
    d_model = features.shape[1]

    # Center
    centered = features - features.mean(dim=0, keepdim=True)

    sigma = torch.linalg.svdvals(centered.float())

    # Filter near-zero
    sigma = sigma[sigma > eps]
    if len(sigma) == 0:
        return 1.0, 1.0 / d_model

    covariance_eigenvalues = sigma ** 2
    p = covariance_eigenvalues / covariance_eigenvalues.sum()

    # Shannon entropy
    log_p = torch.log(p.clamp(min=eps))
    entropy = -torch.sum(p * log_p)

    rankme = torch.exp(entropy).item()
    ratio = rankme / d_model
    return rankme, ratio


# =============================================================================
# alpha-ReQ Computation (activation matrices)
# =============================================================================

def compute_activation_alpha_req(
    features: torch.Tensor,
    fit_range: tuple[int, int] = (10, 100),
    eps: float = 1e-10,
) -> float:
    """
    Compute power-law decay rate alpha of the eigenspectrum.

    1. Centers features first
    2. Uses configurable fit_range for log-log regression
    3. Operates on (N, D) feature matrix

    Fits log(lambda_i) ~ -alpha * log(i) + const for i in [fit_range[0], fit_range[1]).
    """
    centered = features - features.mean(dim=0, keepdim=True)
    sigma = torch.linalg.svdvals(centered.float())
    covariance_eigenvalues = sigma[sigma > eps] ** 2
    n = len(covariance_eigenvalues)
    if n < 3:
        return 0.0

    lo, hi = fit_range
    lo = max(lo, 1)
    hi = min(hi, n)

    if hi <= lo:
        # Fall back to full spectrum if range is too narrow
        lo, hi = 1, n

    indices = torch.arange(lo, hi, dtype=torch.float64)
    if len(indices) < 2:
        return 0.0

    log_eigenvalues = torch.log(covariance_eigenvalues[lo - 1 : hi - 1].to(torch.float64).clamp(min=eps))
    log_indices = torch.log(indices)

    x_mean = log_indices.mean()
    y_mean = log_eigenvalues.mean()
    numerator = torch.sum((log_indices - x_mean) * (log_eigenvalues - y_mean))
    denominator = torch.sum((log_indices - x_mean) ** 2)

    if abs(denominator.item()) < eps:
        return 0.0

    alpha = -(numerator / denominator).item()
    return max(0.0, alpha)


# =============================================================================
# Model and Tokenizer Loading
# =============================================================================

def _get_tokenizer(model_config: ModelConfig):
    """Load tokenizer for the model."""
    from transformers import AutoTokenizer
    trust = model_config.architecture == "olmo3"
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_id,
        revision=model_config.revision,
        trust_remote_code=trust,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_config: ModelConfig):
    """Load model with memory-efficient settings."""
    from transformers import AutoModelForCausalLM
    trust = model_config.architecture == "olmo3"
    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_id,
        revision=model_config.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=trust,
    )
    model.eval()
    return model


# =============================================================================
# Core: Extract and Analyze Activations
# =============================================================================

def extract_and_analyze_activations(
    model_config: ModelConfig,
    questions: list[str],
    max_seq_len: int = 512,
    fit_range: tuple[int, int] = (10, 100),
) -> ActivationResult:
    """
    Extract hidden states from ALL layers and compute RankMe per layer.

    Strategy:
    1. Load model with device_map="auto", torch_dtype=bfloat16
    2. Load tokenizer
    3. For each question (batch_size=1):
       a. Tokenize (truncate to max_seq_len)
       b. Forward with output_hidden_states=True
       c. Extract last-token hidden state from each layer
       d. Accumulate into per-layer feature lists
    4. For each layer, stack features into (N, d_model) matrix
    5. Center features, compute SVD, then RankMe and alpha-ReQ
    6. Clean up model, return results

    Memory management:
    - torch.no_grad() for all forward passes
    - Move extracted hidden states to CPU float32 immediately
    - Del outputs after extracting, torch.cuda.empty_cache() periodically
    - After all questions, compute metrics per layer and discard feature matrix
    - After done, del model, gc.collect(), torch.cuda.empty_cache()
    """
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Activation Analysis: {model_config.name} ({model_config.hf_id})")
    print(f"Questions: {len(questions)}, max_seq_len: {max_seq_len}")
    print(f"{'='*60}")

    tokenizer = _get_tokenizer(model_config)
    model = _load_model(model_config)

    # Determine number of layers from model config
    num_layers = _detect_num_layers(model, model_config)

    # Per-layer feature accumulation: layer_idx -> list of 1D tensors
    layer_features: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    d_model = None

    for qi, question in enumerate(tqdm(questions, desc="Extracting activations")):
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
        )
        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        # hidden_states is a tuple of (num_layers + 1,) tensors, each (1, seq_len, d_model)
        # Index 0 = embedding layer, 1..num_layers = transformer layers

        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        last_token_idx = seq_len - 1

        for layer_idx in range(num_layers):
            # layer_idx 0 -> hidden_states[1] (first transformer layer output)
            hs = hidden_states[layer_idx + 1]
            # Extract last-token hidden state, move to CPU float32
            last_tok = hs[0, last_token_idx, :].detach().cpu().float()
            layer_features[layer_idx].append(last_tok)
            if d_model is None:
                d_model = last_tok.shape[0]

        del outputs, hidden_states, inputs

        # Periodic GPU cache cleanup
        if (qi + 1) % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Clean up model before metric computation
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Extracted hidden states: {num_layers} layers, d_model={d_model}")
    print(f"  Computing per-layer RankMe and alpha-ReQ...")

    layer_results: dict[int, dict] = {}
    for layer_idx in range(num_layers):
        # Stack into (N, d_model) matrix
        features = torch.stack(layer_features[layer_idx], dim=0)
        n_samples = features.shape[0]

        rankme, rankme_ratio = compute_activation_rankme(features)
        alpha = compute_activation_alpha_req(features, fit_range=fit_range)

        layer_results[layer_idx] = {
            "rankme": round(rankme, 6),
            "rankme_ratio": round(rankme_ratio, 6),
            "alpha_req": round(alpha, 6),
            "d_model": d_model if d_model else 0,
            "n_samples": n_samples,
        }

        # Free this layer's features
        del features
        layer_features[layer_idx].clear()

    del layer_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    return ActivationResult(
        model_name=model_config.name,
        layer_results=layer_results,
        num_questions=len(questions),
        elapsed_seconds=round(elapsed, 1),
    )


def _detect_num_layers(model, model_config: ModelConfig) -> int:
    """Detect number of transformer layers from the loaded model."""
    # Try common config attributes
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    if hasattr(model.config, "n_layer"):
        return model.config.n_layer
    if hasattr(model.config, "num_layers"):
        return model.config.num_layers
    # Fall back to the configured value
    if model_config.layers > 0:
        return model_config.layers
    # Last resort: count from model modules
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    raise ValueError(f"Cannot detect number of layers for {model_config.name}")


# =============================================================================
# Helpers
# =============================================================================

def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_results(data: dict, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved results to {path}")
    return path


def _activation_result_to_dict(result: ActivationResult) -> dict[str, dict]:
    """Convert ActivationResult.layer_results to a JSON-serializable dict with string keys."""
    return {str(k): v for k, v in result.layer_results.items()}


# =============================================================================
# Analysis: Cross-Model Comparison
# =============================================================================

def analyze_activation_cross_model(
    models: dict[str, ModelConfig],
    num_samples: Optional[int] = None,
    max_seq_len: int = 512,
    fit_range: tuple[int, int] = (10, 100),
) -> dict:
    """
    Compare activation RankMe across multiple models.

    Returns dict with same structure as existing analysis results:
    {
        "analysis": "activation_cross_model",
        "description": "...",
        "models": {model_name: layer_results},
        "summary": {model_name: {last_layer_rankme, last_layer_ratio, ...}},
        "timestamp": "..."
    }
    """
    _ensure_dirs()

    print(f"\n{'#'*60}")
    print(f"# Activation Analysis: Cross-Model Comparison")
    print(f"# Models: {list(models.keys())}")
    print(f"# Samples: {num_samples or 'all'}")
    print(f"{'#'*60}")

    questions = load_mmlu_questions(num_samples=num_samples)

    output_path = os.path.join(RESULTS_DIR, "activation_cross_model.json")
    existing_models: dict[str, dict] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
                existing_models = existing.get("models", {})
                print(f"  Loaded {len(existing_models)} existing model results")
        except Exception:
            pass

    all_results: dict[str, dict] = dict(existing_models)

    for name, config in models.items():
        if name in all_results and "error" not in all_results[name]:
            print(f"  Skipping {name} (already computed)")
            continue

        try:
            result = extract_and_analyze_activations(
                config, questions, max_seq_len=max_seq_len, fit_range=fit_range,
            )
            all_results[name] = _activation_result_to_dict(result)
        except Exception as e:
            print(f"  ERROR ({name}): {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

        # Incremental save
        summary = _build_cross_model_summary(all_results)
        _save_results({
            "analysis": "activation_cross_model",
            "description": "Compare activation RankMe across model scales",
            "models": all_results,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }, "activation_cross_model.json")

    summary = _build_cross_model_summary(all_results)
    output = {
        "analysis": "activation_cross_model",
        "description": "Compare activation RankMe across model scales",
        "models": all_results,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    _save_results(output, "activation_cross_model.json")
    return output


def _build_cross_model_summary(all_results: dict) -> dict:
    """Build summary with last-layer stats for each model."""
    summary = {}
    for name, layer_data in all_results.items():
        if "error" in layer_data:
            summary[name] = {"error": layer_data["error"]}
            continue

        # Find last layer (max numeric key)
        layer_keys = sorted(layer_data.keys(), key=lambda k: int(k))
        if not layer_keys:
            summary[name] = {"error": "no layers"}
            continue

        last_key = layer_keys[-1]
        last_layer = layer_data[last_key]
        mid_key = layer_keys[len(layer_keys) // 2]
        mid_layer = layer_data[mid_key]
        first_key = layer_keys[0]
        first_layer = layer_data[first_key]

        summary[name] = {
            "num_layers": len(layer_keys),
            "first_layer_rankme": first_layer.get("rankme"),
            "first_layer_ratio": first_layer.get("rankme_ratio"),
            "mid_layer_rankme": mid_layer.get("rankme"),
            "mid_layer_ratio": mid_layer.get("rankme_ratio"),
            "last_layer_rankme": last_layer.get("rankme"),
            "last_layer_ratio": last_layer.get("rankme_ratio"),
            "last_layer_alpha_req": last_layer.get("alpha_req"),
        }
    return summary


# =============================================================================
# Analysis: Training Dynamics
# =============================================================================

def analyze_activation_training_dynamics(
    model_name: str = "pythia-70m",
    checkpoints: Optional[list[str]] = None,
    num_samples: Optional[int] = None,
    max_seq_len: int = 512,
    fit_range: tuple[int, int] = (10, 100),
) -> dict:
    """
    Track activation RankMe over training checkpoints.

    Loads each checkpoint, extracts activations, computes per-layer RankMe.
    Returns dict with checkpoints as keys, per-layer results as values.
    """
    if model_name not in PYTHIA_CONFIGS:
        print(f"Model {model_name} not found. Available: {list(PYTHIA_CONFIGS.keys())}")
        return {}

    if checkpoints is None:
        checkpoints = PYTHIA_CHECKPOINTS

    _ensure_dirs()
    config = PYTHIA_CONFIGS[model_name]

    print(f"\n{'#'*60}")
    print(f"# Activation Analysis: Training Dynamics ({model_name})")
    print(f"# Checkpoints: {len(checkpoints)}")
    print(f"{'#'*60}")

    questions = load_mmlu_questions(num_samples=num_samples)

    output_path = os.path.join(RESULTS_DIR, f"activation_training_dynamics_{model_name}.json")
    existing_results: dict[str, dict] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
                existing_results = existing.get("results", {})
                print(f"  Loaded {len(existing_results)} existing checkpoint results")
        except Exception:
            pass

    all_results: dict[str, dict] = dict(existing_results)

    for ckpt in tqdm(checkpoints, desc=f"Training dynamics ({model_name})"):
        if ckpt in all_results and "error" not in all_results[ckpt]:
            print(f"  Skipping {ckpt} (already computed)")
            continue

        ckpt_config = ModelConfig(
            name=f"{model_name}-{ckpt}",
            hf_id=config.hf_id,
            revision=ckpt,
            architecture=config.architecture,
            layers=config.layers,
            d_model=config.d_model,
            intermediate_size=config.intermediate_size,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            total_params=config.total_params,
        )

        try:
            result = extract_and_analyze_activations(
                ckpt_config, questions, max_seq_len=max_seq_len, fit_range=fit_range,
            )
            all_results[ckpt] = _activation_result_to_dict(result)
        except Exception as e:
            print(f"  ERROR ({ckpt}): {e}")
            import traceback
            traceback.print_exc()
            all_results[ckpt] = {"error": str(e)}

        # Incremental save
        _save_results({
            "analysis": f"activation_training_dynamics_{model_name}",
            "model": model_name,
            "checkpoints": checkpoints,
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f"activation_training_dynamics_{model_name}.json")

    output = {
        "analysis": f"activation_training_dynamics_{model_name}",
        "model": model_name,
        "checkpoints": checkpoints,
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    _save_results(output, f"activation_training_dynamics_{model_name}.json")
    return output


# =============================================================================
# Analysis: OLMo-3 Post-Training Comparison
# =============================================================================

def analyze_activation_post_training(
    num_samples: Optional[int] = None,
    max_seq_len: int = 512,
    fit_range: tuple[int, int] = (10, 100),
) -> dict:
    """
    Compare activation RankMe across OLMo-3 post-training variants.

    Analyzes all 12 OLMo-3 variants (base + think/instruct/rl-zero pathways).
    """
    _ensure_dirs()

    print(f"\n{'#'*60}")
    print(f"# Activation Analysis: OLMo-3 Post-Training Comparison")
    print(f"# Variants: {len(OLMO3_VARIANTS)}")
    print(f"# Samples: {num_samples or 'all'}")
    print(f"{'#'*60}")

    questions = load_mmlu_questions(num_samples=num_samples)

    output_path = os.path.join(RESULTS_DIR, "activation_post_training.json")
    existing_results: dict[str, dict] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
                existing_results = existing.get("variants", {})
                print(f"  Loaded {len(existing_results)} existing variant results")
        except Exception:
            pass

    all_results: dict[str, dict] = dict(existing_results)

    for name, config in OLMO3_VARIANTS.items():
        if name in all_results and "error" not in all_results[name]:
            print(f"  Skipping {name} (already computed)")
            continue

        try:
            result = extract_and_analyze_activations(
                config, questions, max_seq_len=max_seq_len, fit_range=fit_range,
            )
            variant_data: dict[str, object] = dict(_activation_result_to_dict(result))
            variant_data["pathway"] = config.pathway
            variant_data["stage"] = config.stage
            all_results[name] = variant_data
        except Exception as e:
            print(f"  ERROR ({name}): {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

        pathway_comparison = _build_pathway_summary(all_results)
        _save_results({
            "analysis": "activation_post_training",
            "description": "Compare activation RankMe across OLMo-3 post-training variants",
            "variants": all_results,
            "pathway_comparison": pathway_comparison,
            "timestamp": datetime.now().isoformat(),
        }, "activation_post_training.json")

    pathway_comparison = _build_pathway_summary(all_results)
    output = {
        "analysis": "activation_post_training",
        "description": "Compare activation RankMe across OLMo-3 post-training variants",
        "variants": all_results,
        "pathway_comparison": pathway_comparison,
        "timestamp": datetime.now().isoformat(),
    }
    _save_results(output, "activation_post_training.json")
    return output


def _build_pathway_summary(all_results: dict) -> dict:
    pathways = {}
    for name, layer_data in all_results.items():
        if "error" in layer_data:
            continue
        pathway = layer_data.get("pathway", "unknown")
        stage = layer_data.get("stage", "unknown")
        if pathway not in pathways:
            pathways[pathway] = {}
        layer_keys = sorted([int(k) for k in layer_data.keys() if str(k).isdigit()])
        if not layer_keys:
            continue
        last_layer = layer_data[str(layer_keys[-1])]
        pathways[pathway][stage] = last_layer.get("rankme_ratio", 0)
    return pathways

if __name__ == "__main__":
    # Quick test: analyze pythia-70m with 100 samples
    questions = load_mmlu_questions(num_samples=100)
    result = extract_and_analyze_activations(PYTHIA_CONFIGS["pythia-70m"], questions)
    last_layer = max(result.layer_results.keys())
    print(f"\nPythia-70m results ({result.num_questions} questions, {result.elapsed_seconds:.1f}s):")
    print(f"  Last layer (idx={last_layer}) RankMe: {result.layer_results[last_layer]['rankme']:.4f}")
    print(f"  Last layer RankMe ratio: {result.layer_results[last_layer]['rankme_ratio']:.4f}")
    print(f"  Last layer alpha-ReQ: {result.layer_results[last_layer]['alpha_req']:.4f}")
    print(f"\nAll layers:")
    for idx in sorted(result.layer_results.keys()):
        lr = result.layer_results[idx]
        print(f"  Layer {idx}: RankMe={lr['rankme']:.4f}, ratio={lr['rankme_ratio']:.4f}, alpha={lr['alpha_req']:.4f}")
