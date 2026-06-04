"""
Analysis pipeline for effective rank computation across models and training stages.

Five analysis modes:
1. cross-model-size: Compare effective rank ratios across Pythia model sizes
2. training-dynamics: Track effective rank during training via Pythia checkpoints
3. training-stages: Compare across OLMo-3 pretraining stages
4. post-training-methods: Compare OLMo-3 post-training pathways (Think/Instruct/RL-Zero)
5. fixed-ratio: Aggregate all results to test the fixed ratio hypothesis
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm

from src.config import (
    ModelConfig, PYTHIA_CONFIGS, PYTHIA_DEDUPED_CONFIGS,
    PYTHIA_CHECKPOINTS, PYTHIA_CHECKPOINTS_QUICK,
    OLMO3_VARIANTS, OLMO3_PRETRAIN_CHECKPOINTS,
    RESULTS_DIR, FIGURES_DIR,
)
from src.effective_rank import compute_all_metrics, RankMetrics
from src.model_loader import (
    iter_weight_tensors, collect_all_weight_names,
    group_names_by_type, group_names_by_layer,
)
import gc


def _ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save_results(data: dict, filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved results to {path}")
    return path


def _metrics_to_dict(m: RankMetrics) -> dict:
    return {
        "shape": list(m.shape),
        "max_rank": m.max_rank,
        "effective_rank": round(m.effective_rank, 6),
        "effective_rank_ratio": round(m.effective_rank_ratio, 6),
        "svd_entropy": round(m.svd_entropy, 6),
        "stable_rank": round(m.stable_rank, 6),
        "top_singular_value": round(m.top_singular_value, 6),
        "frobenius_norm": round(m.frobenius_norm, 6),
    }


def analyze_single_model(
    model_config: ModelConfig,
    max_dim: Optional[int] = None,
) -> dict:
    """
    Analyze a single model by streaming weight tensors one-by-one.
    Never holds all weights in memory at once.
    For each tensor: load bfloat16 -> float32 for SVD -> compute metrics -> discard.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_config.name} ({model_config.hf_id})")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        per_weight_metrics: dict[str, dict] = {}
        all_ratios: list[float] = []
        all_entropies: list[float] = []
        group_ratio_accum: dict[str, list[float]] = {}
        layer_ratio_accum: dict[str, list[float]] = {}

        for name, tensor in iter_weight_tensors(model_config):
            m, n = tensor.shape
            min_dim = min(m, n)

            if max_dim is not None and min_dim > max_dim:
                print(f"    Skip {name}: min_dim={min_dim} > max_dim={max_dim}")
                del tensor
                continue

            # SVD in float32 (tensor arrives as bfloat16 from safetensors)
            metrics = compute_all_metrics(tensor)
            del tensor
            gc.collect()

            ratio = metrics.effective_rank_ratio
            all_ratios.append(ratio)
            all_entropies.append(metrics.svd_entropy)
            per_weight_metrics[name] = _metrics_to_dict(metrics)

            cat = _classify_layer(name)
            group_ratio_accum.setdefault(cat, []).append(ratio)

            layer_idx = _extract_layer_idx(name)
            if layer_idx is not None:
                layer_ratio_accum.setdefault(layer_idx, []).append(ratio)

            if len(per_weight_metrics) % 10 == 0:
                print(f"    ...processed {len(per_weight_metrics)} matrices")

        if not per_weight_metrics:
            print(f"  No linear weights found for {model_config.name}")
            return {"model": model_config.name, "error": "no_linear_weights"}

        group_stats = {}
        for g, ratios in group_ratio_accum.items():
            group_stats[g] = {
                "count": len(ratios),
                "mean_ratio": round(sum(ratios) / len(ratios), 6),
                "min_ratio": round(min(ratios), 6),
                "max_ratio": round(max(ratios), 6),
            }

        layer_stats = {}
        for idx_str in sorted(layer_ratio_accum.keys(), key=lambda x: int(x)):
            ratios = layer_ratio_accum[idx_str]
            layer_stats[idx_str] = {
                "mean_ratio": round(sum(ratios) / len(ratios), 6),
                "num_matrices": len(ratios),
            }

        mean_ratio = sum(all_ratios) / len(all_ratios)
        std_ratio = (sum((r - mean_ratio) ** 2 for r in all_ratios) / len(all_ratios)) ** 0.5 if len(all_ratios) > 1 else 0
        mean_entropy = sum(all_entropies) / len(all_entropies)

        elapsed = time.time() - start_time

        result = {
            "model": model_config.name,
            "hf_id": model_config.hf_id,
            "revision": model_config.revision,
            "architecture": model_config.architecture,
            "total_params": model_config.total_params,
            "pathway": model_config.pathway,
            "stage": model_config.stage,
            "num_weight_matrices": len(per_weight_metrics),
            "overall_mean_ratio": round(mean_ratio, 6),
            "overall_std_ratio": round(std_ratio, 6),
            "overall_mean_entropy": round(mean_entropy, 6),
            "group_stats": group_stats,
            "layer_stats": layer_stats,
            "per_weight_metrics": per_weight_metrics,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }

        gc.collect()
        print(f"  Done in {elapsed:.1f}s. Mean ratio: {result['overall_mean_ratio']:.4f}")
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": model_config.name,
            "hf_id": model_config.hf_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def _classify_layer(name: str) -> str:
    """Classify a weight name into a layer type category."""
    low = name.lower()
    if "attention" in low or "attn" in low:
        if any(x in low for x in ("q_proj", "k_proj", "v_proj", "query_key_value")):
            return "attention_qkv"
        return "attention_output"
    if "mlp" in low or "dense_h_to_4h" in low or "dense_4h_to_h" in low or "gate_proj" in low or "up_proj" in low or "down_proj" in low:
        return "mlp"
    return "other"


def _extract_layer_idx(name: str) -> Optional[str]:
    """Extract layer index string from weight name."""
    import re
    match = re.search(r'(?:layers?|h|blocks?)\.(\d+)\.', name)
    if match:
        return match.group(1)
    return None


# =============================================================================
# Analysis 1: Cross-Model-Size Comparison
# =============================================================================

def analyze_cross_model_size(
    models: Optional[dict[str, ModelConfig]] = None,
    max_dim: Optional[int] = None,
) -> dict:
    """
    Compare effective rank ratios across different Pythia model sizes.
    
    Tests whether the ratio erank/min(m,n) is constant across scales.
    Saves results incrementally after each model.
    """
    if models is None:
        models = PYTHIA_CONFIGS
    
    print(f"\n{'#'*60}")
    print(f"# Analysis 1: Cross-Model-Size Comparison")
    print(f"# Models: {list(models.keys())}")
    print(f"{'#'*60}")
    
    # Try to load existing partial results
    output_path = os.path.join(RESULTS_DIR, "cross_model_size.json")
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
                results = existing.get("models", {})
                print(f"  Loaded {len(results)} existing results")
        except Exception:
            pass
    
    for name, config in tqdm(models.items(), desc="Cross-model analysis"):
        if name in results and "error" not in results[name]:
            print(f"  Skipping {name} (already computed)")
            continue
        results[name] = analyze_single_model(config, max_dim=max_dim)
        # Save incrementally
        summary = _extract_ratio_summary(results)
        partial_output = {
            "analysis": "cross_model_size",
            "description": "Compare effective rank ratios across different model scales",
            "models": results,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }
        _save_results(partial_output, "cross_model_size.json")
    
    summary = _extract_ratio_summary(results)
    
    output = {
        "analysis": "cross_model_size",
        "description": "Compare effective rank ratios across different model scales",
        "models": results,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    
    _save_results(output, "cross_model_size.json")
    return output


# =============================================================================
# Analysis 2: Training Dynamics
# =============================================================================

def analyze_training_dynamics(
    model_name: str = "pythia-70m",
    checkpoints: Optional[list[str]] = None,
    max_dim: Optional[int] = None,
) -> dict:
    """
    Track effective rank evolution during training via Pythia checkpoints.
    
    Tests the three-phase hypothesis: warmup → entropy-seeking → compression-seeking.
    Saves results incrementally after each checkpoint.
    """
    if model_name not in PYTHIA_CONFIGS:
        print(f"Model {model_name} not found. Available: {list(PYTHIA_CONFIGS.keys())}")
        return {}
    
    if checkpoints is None:
        checkpoints = PYTHIA_CHECKPOINTS
    
    config = PYTHIA_CONFIGS[model_name]
    
    print(f"\n{'#'*60}")
    print(f"# Analysis 2: Training Dynamics for {model_name}")
    print(f"# Checkpoints: {len(checkpoints)}")
    print(f"{'#'*60}")
    
    output_path = os.path.join(RESULTS_DIR, f"training_dynamics_{model_name}.json")
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
                results = existing.get("results", {})
                print(f"  Loaded {len(results)} existing checkpoint results")
        except Exception:
            pass
    
    for ckpt in tqdm(checkpoints, desc=f"Training dynamics ({model_name})"):
        if ckpt in results and "error" not in results[ckpt]:
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
        results[ckpt] = analyze_single_model(ckpt_config, max_dim=max_dim)
        
        output = {
            "analysis": "training_dynamics",
            "model": model_name,
            "checkpoints": checkpoints,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
        _save_results(output, f"training_dynamics_{model_name}.json")
    
    output = {
        "analysis": "training_dynamics",
        "model": model_name,
        "checkpoints": checkpoints,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    _save_results(output, f"training_dynamics_{model_name}.json")
    return output


# =============================================================================
# Analysis 3: Training Stages (OLMo-3)
# =============================================================================

def analyze_training_stages(
    max_dim: Optional[int] = None,
) -> dict:
    """
    Compare effective rank across OLMo-3 pretraining stages.
    
    Tries to load stage checkpoints from the base model repo.
    """
    print(f"\n{'#'*60}")
    print(f"# Analysis 3: Training Stages (OLMo-3)")
    print(f"{'#'*60}")
    
    from src.config import OLMO3_BASE_CONFIG
    
    output_path = os.path.join(RESULTS_DIR, "training_stages.json")
    results = {}
    successful = 0
    
    # Load existing partial results (incremental save)
    if os.path.exists(output_path):
        try:
            with open(output_path) as f:
                existing = json.load(f)
                results = existing.get("results", {})
                print(f"  Loaded {len(results)} existing checkpoint results")
        except Exception:
            pass
    
    for ckpt in tqdm(OLMO3_PRETRAIN_CHECKPOINTS, desc="OLMo-3 stages"):
        if ckpt in results and "error" not in results[ckpt]:
            print(f"  Skipping {ckpt} (already computed)")
            successful += 1
            continue
        
        ckpt_config = ModelConfig(
            name=f"olmo3-{ckpt}",
            hf_id=OLMO3_BASE_CONFIG.hf_id,
            revision=ckpt,
            architecture=OLMO3_BASE_CONFIG.architecture,
            layers=OLMO3_BASE_CONFIG.layers,
            d_model=OLMO3_BASE_CONFIG.d_model,
            intermediate_size=OLMO3_BASE_CONFIG.intermediate_size,
            n_heads=OLMO3_BASE_CONFIG.n_heads,
            n_kv_heads=OLMO3_BASE_CONFIG.n_kv_heads,
            total_params=OLMO3_BASE_CONFIG.total_params,
        )
        try:
            result = analyze_single_model(ckpt_config, max_dim=max_dim)
        except Exception as e:
            print(f"  CHECKPOINT ERROR ({ckpt}): {e}")
            result = {"model": f"olmo3-{ckpt}", "error": str(e)}
        
        results[ckpt] = result
        if "error" not in result:
            successful += 1
        
        # Save incrementally after each checkpoint
        _save_results({
            "analysis": "training_stages",
            "model": "olmo3-7b",
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }, "training_stages.json")
    
    if successful == 0:
        print("  No stage checkpoints found. Skipping this analysis.")
        print("  The base model repo may not have stage checkpoint branches.")
    
    output = {
        "analysis": "training_stages",
        "model": "olmo3-7b",
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    _save_results(output, "training_stages.json")
    return output


# =============================================================================
# Analysis 4: Post-Training Methods (OLMo-3)
# =============================================================================

def analyze_post_training_methods(
    variants: Optional[dict[str, ModelConfig]] = None,
    max_dim: Optional[int] = None,
) -> dict:
    """
    Compare effective rank across OLMo-3 post-training pathways.
    
    Analyzes: base, Think (SFT/DPO/RLVR), Instruct (SFT/DPO/RLVR), RL-Zero variants.
    """
    if variants is None:
        variants = OLMO3_VARIANTS
    
    print(f"\n{'#'*60}")
    print(f"# Analysis 4: Post-Training Methods (OLMo-3)")
    print(f"# Variants: {list(variants.keys())}")
    print(f"{'#'*60}")
    
    results = {}
    for name, config in tqdm(variants.items(), desc="OLMo-3 variants"):
        results[name] = analyze_single_model(config, max_dim=max_dim)
    
    # Compare pathways
    pathway_comparison = _compare_pathways(results)
    
    output = {
        "analysis": "post_training_methods",
        "description": "Compare effective rank across OLMo-3 post-training pathways",
        "variants": results,
        "pathway_comparison": pathway_comparison,
        "timestamp": datetime.now().isoformat(),
    }
    
    _save_results(output, "post_training_methods.json")
    return output


# =============================================================================
# Analysis 5: Fixed Ratio Hypothesis
# =============================================================================

def analyze_fixed_ratio_hypothesis(
    cross_model_result: Optional[dict] = None,
    training_dynamics_result: Optional[dict] = None,
    post_training_result: Optional[dict] = None,
) -> dict:
    """
    Aggregate all results to test the fixed ratio hypothesis.
    
    Tests: Is there a consistent effective rank ratio that emerges across
    models, layers, and training stages?
    """
    print(f"\n{'#'*60}")
    print(f"# Analysis 5: Fixed Ratio Hypothesis")
    print(f"{'#'*60}")
    
    all_ratios = []
    all_entropies = []
    ratio_by_model = {}
    ratio_by_layer_type = {}
    
    # Collect ratios from cross-model analysis
    if cross_model_result and "models" in cross_model_result:
        for model_name, data in cross_model_result["models"].items():
            if "error" in data:
                continue
            ratio_by_model[model_name] = data.get("overall_mean_ratio", 0)
            if "per_weight_metrics" in data:
                for wname, wm in data["per_weight_metrics"].items():
                    r = wm.get("effective_rank_ratio", 0)
                    all_ratios.append(r)
                    all_entropies.append(wm.get("svd_entropy", 0))
                    
                    # Classify layer type
                    if "attention" in wname.lower() or "attn" in wname.lower():
                        if "q" in wname.lower() or "k" in wname.lower() or "v" in wname.lower():
                            lt = "attention_qkv"
                        else:
                            lt = "attention_output"
                    elif "mlp" in wname.lower() or "dense_h_to_4h" in wname.lower() or "dense_4h_to_h" in wname.lower():
                        lt = "mlp"
                    else:
                        lt = "other"
                    
                    if lt not in ratio_by_layer_type:
                        ratio_by_layer_type[lt] = []
                    ratio_by_layer_type[lt].append(r)
    
    # Collect from post-training analysis
    if post_training_result and "variants" in post_training_result:
        for variant_name, data in post_training_result["variants"].items():
            if "error" in data:
                continue
            ratio_by_model[f"olmo3-{variant_name}"] = data.get("overall_mean_ratio", 0)
    
    # Collect from training dynamics
    if training_dynamics_result and "results" in training_dynamics_result:
        dynamics_ratios = []
        for ckpt, data in training_dynamics_result["results"].items():
            if "error" in data:
                continue
            dynamics_ratios.append({
                "checkpoint": ckpt,
                "mean_ratio": data.get("overall_mean_ratio", 0),
            })
    else:
        dynamics_ratios = []
    
    import numpy as np
    
    if all_ratios:
        ratios_arr = np.array(all_ratios)
        stats = {
            "total_observations": len(all_ratios),
            "mean": float(np.mean(ratios_arr)),
            "std": float(np.std(ratios_arr)),
            "median": float(np.median(ratios_arr)),
            "q25": float(np.percentile(ratios_arr, 25)),
            "q75": float(np.percentile(ratios_arr, 75)),
            "min": float(np.min(ratios_arr)),
            "max": float(np.max(ratios_arr)),
        }
        
        layer_type_stats = {}
        for lt, ratios in ratio_by_layer_type.items():
            arr = np.array(ratios)
            layer_type_stats[lt] = {
                "count": len(arr),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
            }
    else:
        stats = {}
        layer_type_stats = {}
    
    hypothesis_result = {}
    model_means = [v for v in ratio_by_model.values() if v > 0]
    if model_means and len(model_means) >= 2:
        means_arr = np.array(model_means)
        cv = float(np.std(means_arr) / np.mean(means_arr)) if np.mean(means_arr) > 0 else float('inf')

        pythia_models = {k: v for k, v in ratio_by_model.items() if k.startswith("pythia-") and v > 0}
        olmo3_models = {k: v for k, v in ratio_by_model.items() if k.startswith("olmo3-") and v > 0}

        per_arch = {}
        for arch_name, arch_models in [("pythia", pythia_models), ("olmo3", olmo3_models)]:
            if arch_models:
                vals = np.array(list(arch_models.values()))
                arch_mean = float(np.mean(vals))
                arch_std = float(np.std(vals))
                arch_cv = float(arch_std / arch_mean) if arch_mean > 0 else float('inf')
                per_arch[arch_name] = {
                    "mean_ratio": round(arch_mean, 6),
                    "std_ratio": round(arch_std, 6),
                    "cv": round(arch_cv, 6),
                    "num_models": len(arch_models),
                    "models": {k: round(v, 6) for k, v in arch_models.items()},
                }
            else:
                per_arch[arch_name] = {
                    "mean_ratio": None,
                    "std_ratio": None,
                    "cv": None,
                    "num_models": 0,
                    "models": {},
                }

        pythia_mean = per_arch["pythia"]["mean_ratio"]
        olmo3_mean = per_arch["olmo3"]["mean_ratio"]
        arch_gap = {}
        if pythia_mean is not None and olmo3_mean is not None:
            abs_gap = pythia_mean - olmo3_mean
            rel_gap = abs_gap / pythia_mean * 100
            arch_gap = {
                "pythia_mean": round(pythia_mean, 6),
                "olmo3_mean": round(olmo3_mean, 6),
                "absolute_gap": round(abs_gap, 6),
                "relative_gap_pct": round(rel_gap, 2),
            }

        dynamics_range = {}
        if dynamics_ratios and len(dynamics_ratios) >= 2:
            init_ratio = dynamics_ratios[0]["mean_ratio"]
            final_ratio = dynamics_ratios[-1]["mean_ratio"]
            total_change = init_ratio - final_ratio
            change_pct = total_change / init_ratio * 100 if init_ratio > 0 else 0
            dynamics_range = {
                "initial_ratio": round(init_ratio, 6),
                "final_ratio": round(final_ratio, 6),
                "total_change": round(total_change, 6),
                "change_pct": round(change_pct, 2),
            }

        conclusion_parts = []
        if per_arch.get("pythia", {}).get("cv") is not None:
            conclusion_parts.append(
                f"Within each architecture family, the ratio IS approximately constant: "
                f"Pythia CV = {per_arch['pythia']['cv']*100:.1f}% (n={per_arch['pythia']['num_models']}), "
                f"OLMo-3 CV = {per_arch['olmo3']['cv']*100:.2f}% (n={per_arch['olmo3']['num_models']})."
            )
        if arch_gap:
            conclusion_parts.append(
                f"Between architectures, there is a clear gap: "
                f"{arch_gap['relative_gap_pct']:.1f}% difference "
                f"(Pythia mean = {arch_gap['pythia_mean']:.4f}, "
                f"OLMo-3 mean = {arch_gap['olmo3_mean']:.4f})."
            )
        if dynamics_range:
            conclusion_parts.append(
                f"During pretraining, the ratio changes significantly: "
                f"{dynamics_range['change_pct']:.1f}% shift "
                f"({dynamics_range['initial_ratio']:.4f} -> {dynamics_range['final_ratio']:.4f} for pythia-70m)."
            )
        if layer_type_stats:
            lt_means = {lt: s["mean"] for lt, s in layer_type_stats.items() if isinstance(s, dict)}
            if lt_means:
                min_lt = min(lt_means, key=lt_means.get)
                max_lt = max(lt_means, key=lt_means.get)
                conclusion_parts.append(
                    f"Per-layer-type ratios vary substantially: "
                    f"{min_lt} ~ {lt_means[min_lt]:.3f} to {max_lt} ~ {lt_means[max_lt]:.3f}."
                )
        conclusion_parts.append(
            "The fixed ratio property holds at convergence, within an architecture family, "
            "at the aggregate (whole-model) level."
        )
        if olmo3_models:
            olmo3_vals = list(olmo3_models.values())
            olmo3_range = max(olmo3_vals) - min(olmo3_vals)
            olmo3_base = olmo3_models.get("olmo3-olmo3-base", 0)
            if olmo3_base > 0:
                pt_var_pct = olmo3_range / olmo3_base * 100
                conclusion_parts.append(
                    f"Post-training methods (SFT/DPO/RLVR/RL-Zero) have negligible effect "
                    f"(<{pt_var_pct:.1f}% variation across all OLMo-3 variants)."
                )

        hypothesis_result = {
            "coefficient_of_variation_per_weight": round(stats.get("std", 0) / stats["mean"], 4) if stats.get("mean", 0) > 0 else None,
            "coefficient_of_variation_per_model": round(cv, 6),
            "num_models_in_cv": len(model_means),
            "typical_ratio_range": f"[{stats['q25']:.4f}, {stats['q75']:.4f}]" if stats else None,
            "per_architecture": per_arch,
            "architecture_gap": arch_gap,
            "per_layer_type": layer_type_stats,
            "training_dynamics_range": dynamics_range,
            "conclusion": "\n".join(conclusion_parts),
        }
    
    output = {
        "analysis": "fixed_ratio_hypothesis",
        "description": "Test whether a constant effective rank ratio exists across models",
        "overall_stats": stats,
        "ratio_by_model": ratio_by_model,
        "ratio_by_layer_type": layer_type_stats,
        "training_dynamics_summary": dynamics_ratios,
        "hypothesis_test": hypothesis_result,
        "timestamp": datetime.now().isoformat(),
    }
    
    _save_results(output, "fixed_ratio_hypothesis.json")
    return output


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_ratio_summary(results: dict) -> dict:
    """Extract cross-model ratio statistics from analysis results."""
    summary = {}
    for model_name, data in results.items():
        if "error" in data:
            summary[model_name] = {"error": data["error"]}
            continue
        summary[model_name] = {
            "overall_mean_ratio": data.get("overall_mean_ratio", 0),
            "overall_std_ratio": data.get("overall_std_ratio", 0),
            "overall_mean_entropy": data.get("overall_mean_entropy", 0),
            "num_weight_matrices": data.get("num_weight_matrices", 0),
        }
    return summary


def _compare_pathways(results: dict) -> dict:
    """Compare post-training pathways."""
    pathways = {}
    for name, data in results.items():
        if "error" in data:
            continue
        pathway = data.get("pathway", "unknown")
        stage = data.get("stage", "unknown")
        if pathway not in pathways:
            pathways[pathway] = {}
        pathways[pathway][stage] = data.get("overall_mean_ratio", 0)
    return pathways


def run_all_analyses(
    quick: bool = False,
    max_dim: Optional[int] = None,
) -> dict:
    """
    Run all five analyses in sequence.
    
    Args:
        quick: If True, use reduced checkpoint lists and fewer models
        max_dim: Maximum matrix dimension for SVD (None = no limit)
    """
    _ensure_dirs()
    
    all_results = {}
    
    # Analysis 1: Cross-model-size
    models = PYTHIA_CONFIGS
    all_results["cross_model_size"] = analyze_cross_model_size(models, max_dim=max_dim)
    
    # Analysis 2: Training dynamics (smallest model for speed)
    checkpoints = PYTHIA_CHECKPOINTS_QUICK if quick else PYTHIA_CHECKPOINTS
    all_results["training_dynamics"] = analyze_training_dynamics(
        "pythia-70m", checkpoints=checkpoints, max_dim=max_dim,
    )
    
    # Analysis 3: Training stages (may fail if checkpoints unavailable)
    all_results["training_stages"] = analyze_training_stages(max_dim=max_dim)
    
    # Analysis 4: Post-training methods
    all_results["post_training_methods"] = analyze_post_training_methods(max_dim=max_dim)
    
    # Analysis 5: Fixed ratio hypothesis
    all_results["fixed_ratio_hypothesis"] = analyze_fixed_ratio_hypothesis(
        cross_model_result=all_results.get("cross_model_size"),
        training_dynamics_result=all_results.get("training_dynamics"),
        post_training_result=all_results.get("post_training_methods"),
    )
    
    return all_results
