"""
Visualization module for effective rank analysis results.

Generates publication-quality plots for all five analysis modes:
1. Cross-model-size bar chart
2. Training dynamics line plot
3. Training stages comparison
4. Post-training methods heatmap
5. Fixed ratio distribution histogram
"""

from __future__ import annotations

import json
import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from src.config import FIGURES_DIR, COLOR_PALETTES


def _setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (12, 8),
        }
    )


def _save_fig(fig, name: str, formats=("png", "pdf")):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for fmt in formats:
        path = os.path.join(FIGURES_DIR, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_cross_model_size(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    """
    Bar chart of effective rank ratio vs model scale.
    """
    _setup_style()

    if data is None:
        if results_path is None:
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "cross_model_size.json",
            )
        with open(results_path) as f:
            data = json.load(f)

    models_data = data.get("models", data.get("summary", {}))

    # Extract data
    model_names = []
    mean_ratios = []
    std_ratios = []
    mean_entropies = []
    param_counts = []

    for name, info in models_data.items():
        if isinstance(info, dict) and "error" in info:
            continue
        if isinstance(info, dict):
            model_names.append(name)
            mean_ratios.append(info.get("overall_mean_ratio", 0))
            std_ratios.append(info.get("overall_std_ratio", 0))
            mean_entropies.append(info.get("overall_mean_entropy", 0))

    if not model_names:
        print("  No data for cross-model-size plot")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Mean ratio
    x = range(len(model_names))
    bars = axes[0].bar(
        x,
        mean_ratios,
        yerr=std_ratios,
        capsize=5,
        color=COLOR_PALETTES["training_stages"]["pretrain"],
        alpha=0.8,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=45, ha="right")
    axes[0].set_ylabel("Effective Rank Ratio (erank/min_dim)")
    axes[0].set_title("Effective Rank Ratio vs Model Scale")
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(
        y=np.mean(mean_ratios),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {np.mean(mean_ratios):.3f}",
    )
    axes[0].legend()

    # Plot 2: Mean entropy
    axes[1].bar(
        x, mean_entropies, color=COLOR_PALETTES["training_stages"]["sft"], alpha=0.8
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=45, ha="right")
    axes[1].set_ylabel("Normalized SVD Entropy")
    axes[1].set_title("SVD Entropy vs Model Scale")
    axes[1].set_ylim(0, 1.0)

    fig.suptitle("Cross-Model-Size Comparison (Pythia)", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "cross_model_size")
    return "cross_model_size"


def plot_training_dynamics(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    """
    Line plot of effective rank ratio vs training step.
    """
    _setup_style()

    if data is None:
        if results_path is None:
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "training_dynamics_pythia-70m.json",
            )
        with open(results_path) as f:
            data = json.load(f)

    results = data.get("results", {})

    # Extract data points
    steps = []
    mean_ratios = []
    group_ratios = {"attention_qkv": [], "attention_output": [], "mlp": []}

    for ckpt in sorted(
        results.keys(),
        key=lambda x: int(x.replace("step", "")) if x.startswith("step") else 0,
    ):
        info = results[ckpt]
        if isinstance(info, dict) and "error" in info:
            continue

        step_num = int(ckpt.replace("step", ""))
        steps.append(step_num)
        mean_ratios.append(info.get("overall_mean_ratio", 0))

        gs = info.get("group_stats", {})
        for group in group_ratios:
            group_ratios[group].append(gs.get(group, {}).get("mean_ratio", 0))

    if not steps:
        print("  No data for training dynamics plot")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Overall mean ratio
    axes[0].plot(steps, mean_ratios, "o-", color="#3498db", linewidth=2, markersize=4)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Mean Effective Rank Ratio")
    axes[0].set_title("Effective Rank Ratio During Training")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.0)

    # Add phase annotations if enough data points
    if len(steps) > 5:
        axes[0].axvspan(
            steps[0],
            steps[min(3, len(steps) - 1)],
            alpha=0.1,
            color="red",
            label="Warmup",
        )
        if len(steps) > 6:
            mid = len(steps) // 2
            axes[0].axvspan(
                steps[3], steps[mid], alpha=0.1, color="green", label="Entropy-seeking"
            )
            axes[0].axvspan(
                steps[mid],
                steps[-1],
                alpha=0.1,
                color="blue",
                label="Compression-seeking",
            )
        axes[0].legend()

    # Plot 2: Per group
    colors = {
        "attention_qkv": "#e74c3c",
        "attention_output": "#3498db",
        "mlp": "#2ecc71",
    }
    for group, ratios in group_ratios.items():
        if ratios:
            axes[1].plot(
                steps[: len(ratios)],
                ratios,
                "o-",
                color=colors.get(group, "gray"),
                linewidth=2,
                markersize=4,
                label=group,
            )
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Mean Effective Rank Ratio")
    axes[1].set_title("Effective Rank by Layer Type")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    model_name = data.get("model", "unknown")
    fig.suptitle(f"Training Dynamics ({model_name})", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, f"training_dynamics_{model_name}")
    return f"training_dynamics_{model_name}"


def plot_training_stages(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    """
    Grouped bar chart across OLMo-3 training stages.
    """
    _setup_style()

    if data is None:
        if results_path is None:
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "training_stages.json",
            )
        with open(results_path) as f:
            data = json.load(f)

    results = data.get("results", {})

    stages = []
    mean_ratios = []
    for ckpt, info in results.items():
        if isinstance(info, dict) and "error" in info:
            continue
        stages.append(ckpt)
        mean_ratios.append(info.get("overall_mean_ratio", 0))

    if not stages:
        print("  No data for training stages plot")
        return ""

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLOR_PALETTES["training_stages"].get("pretrain", "#3498db")] * len(
        stages
    )
    for i, s in enumerate(stages):
        if "stage2" in s:
            colors[i] = COLOR_PALETTES["training_stages"].get("midtrain", "#2ecc71")
        elif "stage3" in s:
            colors[i] = COLOR_PALETTES["training_stages"].get("longcontext", "#9b59b6")

    ax.bar(range(len(stages)), mean_ratios, color=colors, alpha=0.8)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha="right")
    ax.set_ylabel("Mean Effective Rank Ratio")
    ax.set_title("Effective Rank Across Training Stages (OLMo-3)")
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    _save_fig(fig, "training_stages")
    return "training_stages"


def plot_post_training_methods(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    """
    Heatmap / grouped bar comparing post-training pathways.
    """
    _setup_style()

    if data is None:
        if results_path is None:
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "post_training_methods.json",
            )
        with open(results_path) as f:
            data = json.load(f)

    variants = data.get("variants", {})
    pathway_comparison = data.get("pathway_comparison", {})

    # Extract pathway-stage matrix
    pathway_colors = COLOR_PALETTES["pathways"]

    variant_names = []
    mean_ratios = []
    pathway_labels = []

    for name, info in variants.items():
        if isinstance(info, dict) and "error" in info:
            continue
        variant_names.append(name.replace("olmo3-", ""))
        mean_ratios.append(info.get("overall_mean_ratio", 0))
        pathway_labels.append(info.get("pathway", "unknown"))

    if not variant_names:
        print("  No data for post-training methods plot")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: All variants
    bar_colors = [pathway_colors.get(p, "#95a5a6") for p in pathway_labels]
    axes[0].barh(range(len(variant_names)), mean_ratios, color=bar_colors, alpha=0.8)
    axes[0].set_yticks(range(len(variant_names)))
    axes[0].set_yticklabels(variant_names)
    axes[0].set_xlabel("Mean Effective Rank Ratio")
    axes[0].set_title("All OLMo-3 Variants")
    axes[0].set_xlim(0, 1.0)

    # Add legend for pathways
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=c, label=p.capitalize())
        for p, c in pathway_colors.items()
        if p in pathway_labels
    ]
    axes[0].legend(handles=legend_elements, loc="lower right")

    # Plot 2: Pathway comparison (mean per pathway)
    if pathway_comparison:
        pw_names = list(pathway_comparison.keys())
        pw_means = []
        pw_stds = []
        for pw in pw_names:
            stages = pathway_comparison[pw]
            vals = [v for v in stages.values() if isinstance(v, (int, float)) and v > 0]
            if vals:
                pw_means.append(np.mean(vals))
                pw_stds.append(np.std(vals) if len(vals) > 1 else 0)
            else:
                pw_means.append(0)
                pw_stds.append(0)

        x = range(len(pw_names))
        axes[1].bar(
            x,
            pw_means,
            yerr=pw_stds,
            capsize=5,
            color=[pathway_colors.get(p, "#95a5a6") for p in pw_names],
            alpha=0.8,
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            [p.capitalize() for p in pw_names], rotation=45, ha="right"
        )
        axes[1].set_ylabel("Mean Effective Rank Ratio")
        axes[1].set_title("Post-Training Pathways Comparison")
        axes[1].set_ylim(0, 1.0)

    fig.suptitle("OLMo-3 Post-Training Methods Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "post_training_methods")
    return "post_training_methods"


def plot_fixed_ratio_distribution(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    """
    Histogram + KDE of all observed effective rank ratios.
    """
    _setup_style()

    if data is None:
        if results_path is None:
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "fixed_ratio_hypothesis.json",
            )
        with open(results_path) as f:
            data = json.load(f)

    stats = data.get("overall_stats", {})
    layer_type_stats = data.get("ratio_by_layer_type", {})
    hypothesis = data.get("hypothesis_test", {})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Distribution by layer type
    for lt, ratios in layer_type_stats.items():
        if isinstance(ratios, dict):
            # Stats dict, not raw ratios - draw a box
            axes[0].barh(
                lt,
                ratios.get("mean", 0),
                xerr=ratios.get("std", 0),
                capsize=5,
                alpha=0.8,
                color=COLOR_PALETTES["layer_types"].get(lt, "#95a5a6"),
            )
    axes[0].set_xlabel("Mean Effective Rank Ratio")
    axes[0].set_title("Ratio by Layer Type")
    axes[0].set_xlim(0, 1.0)

    # Plot 2: Model comparison
    ratio_by_model = data.get("ratio_by_model", {})
    if ratio_by_model:
        pythia_keys = [m for m in ratio_by_model if m.startswith("pythia-")]
        olmo3_keys = [m for m in ratio_by_model if m.startswith("olmo3-")]
        other_keys = [
            m
            for m in ratio_by_model
            if not m.startswith("pythia-") and not m.startswith("olmo3-")
        ]
        ordered_keys = pythia_keys + olmo3_keys + other_keys

        models = ordered_keys
        ratios = [ratio_by_model[m] for m in models]
        colors = []
        for m in models:
            if m.startswith("pythia-"):
                colors.append("#3498db")
            elif m.startswith("olmo3-"):
                colors.append("#e74c3c")
            else:
                colors.append("#95a5a6")

        y_pos = list(range(len(models)))
        axes[1].barh(y_pos, ratios, color=colors, alpha=0.8)

        if pythia_keys and olmo3_keys:
            sep_idx = len(pythia_keys) - 0.5
            axes[1].axhline(
                y=sep_idx, color="black", linewidth=1.5, linestyle="--", alpha=0.6
            )

        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(models, fontsize=8)
        axes[1].set_xlabel("Mean Effective Rank Ratio")
        axes[1].set_title("Ratio by Model (grouped by architecture)")
        axes[1].set_xlim(0, 1.0)

        from matplotlib.patches import Patch

        legend_handles = []
        if pythia_keys:
            legend_handles.append(Patch(facecolor="#3498db", alpha=0.8, label="Pythia"))
        if olmo3_keys:
            legend_handles.append(Patch(facecolor="#e74c3c", alpha=0.8, label="OLMo-3"))
        if legend_handles:
            axes[1].legend(handles=legend_handles, loc="lower right", fontsize=9)

    # Add hypothesis annotation
    if hypothesis:
        text_parts = []
        per_arch = hypothesis.get("per_architecture", {})
        if per_arch:
            for arch_name in ["pythia", "olmo3"]:
                ad = per_arch.get(arch_name, {})
                if ad.get("cv") is not None:
                    text_parts.append(
                        f"{arch_name.title()} CV = {ad['cv'] * 100:.2f}% (n={ad['num_models']})"
                    )
        arch_gap = hypothesis.get("architecture_gap", {})
        if arch_gap:
            text_parts.append(f"Arch gap = {arch_gap['relative_gap_pct']:.1f}%")
        dynamics_range = hypothesis.get("training_dynamics_range", {})
        if dynamics_range:
            text_parts.append(f"Training shift = {dynamics_range['change_pct']:.1f}%")
        text = "\n".join(text_parts) if text_parts else "See conclusion in JSON"
        fig.text(
            0.95,
            0.95,
            text,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    fig.suptitle("Fixed Ratio Hypothesis Test", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "fixed_ratio_distribution")
    return "fixed_ratio_distribution"


def _is_training_dynamics(data: dict) -> bool:
    analysis = data.get("analysis", "")
    return "training_dynamics" in analysis or "training_dynamics" in analysis.lower()


def _sort_checkpoint_keys(keys: list[str]) -> list[str]:
    def _ckpt_sort_key(k: str):
        if k.startswith("step") and k[4:].isdigit():
            return (0, int(k[4:]))
        return (1, k)

    return sorted(keys, key=_ckpt_sort_key)


def _sort_model_by_size(names: list[str]) -> list[str]:
    def _size_key(name: str):
        units = {"m": 1e6, "b": 1e9}
        for suffix, mult in units.items():
            if suffix in name:
                parts = name.split(suffix)[0].rsplit("-", 1)[-1]
                try:
                    return float(parts) * mult
                except ValueError:
                    pass
        return 0.0

    return sorted(names, key=_size_key)


def plot_activation_training_dynamics(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    _setup_style()

    if data is None:
        if results_path is None:
            return ""
        with open(results_path) as f:
            data = json.load(f)

    results = data.get("results", {})
    model_label = data.get("model", "")

    ckpt_names = [
        k
        for k in results.keys()
        if "error" not in (results[k] if isinstance(results[k], dict) else {})
    ]
    ckpt_names = _sort_checkpoint_keys(ckpt_names)

    if not ckpt_names:
        print("  No data for training dynamics plot")
        return ""

    steps = [int(c.replace("step", "")) for c in ckpt_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    layer_indices = set()
    for ckpt in ckpt_names:
        ckpt_data = results[ckpt]
        if isinstance(ckpt_data, dict):
            layer_indices.update(int(k) for k in ckpt_data.keys() if str(k).isdigit())
    layer_indices = sorted(layer_indices)

    cmap = plt.cm.viridis
    n_layers = len(layer_indices)
    layer_colors = {
        li: cmap(i / max(n_layers - 1, 1)) for i, li in enumerate(layer_indices)
    }

    selected_layers = [
        layer_indices[0],
        layer_indices[len(layer_indices) // 4],
        layer_indices[len(layer_indices) // 2],
        layer_indices[3 * len(layer_indices) // 4],
        layer_indices[-1],
    ]
    selected_layers = sorted(set(selected_layers))

    for li in selected_layers:
        ratios = []
        valid_steps = []
        for ci, ckpt in enumerate(ckpt_names):
            ckpt_data = results[ckpt]
            if isinstance(ckpt_data, dict) and str(li) in ckpt_data:
                ratios.append(ckpt_data[str(li)].get("rankme_ratio", 0))
                valid_steps.append(steps[ci])
        if ratios:
            axes[0].plot(
                valid_steps,
                ratios,
                "-o",
                color=layer_colors[li],
                linewidth=2,
                markersize=3,
                label=f"Layer {li}",
            )

    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("RankMe Ratio")
    axes[0].set_title("RankMe Ratio vs Training Step (selected layers)")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=8)

    last_layer = layer_indices[-1]
    last_ratios = []
    valid_last_steps = []
    for ci, ckpt in enumerate(ckpt_names):
        ckpt_data = results[ckpt]
        if isinstance(ckpt_data, dict) and str(last_layer) in ckpt_data:
            last_ratios.append(ckpt_data[str(last_layer)].get("rankme_ratio", 0))
            valid_last_steps.append(steps[ci])

    if last_ratios:
        axes[1].plot(
            valid_last_steps,
            last_ratios,
            "-o",
            color="#e74c3c",
            linewidth=2,
            markersize=4,
        )
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel(f"Last-Layer RankMe Ratio (layer {last_layer})")
        axes[1].set_title(f"Last-Layer RankMe Ratio vs Training Step")
        axes[1].set_xscale("log")
        axes[1].set_ylim(0, None)

    title = f"Activation Training Dynamics ({model_label})"
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    analysis_type = data.get("analysis", f"activation_training_dynamics_{model_label}")
    _save_fig(fig, analysis_type)
    return analysis_type


def plot_activation_analysis(
    results_path: Optional[str] = None,
    data: Optional[dict] = None,
) -> str:
    _setup_style()

    if data is None:
        if results_path is None:
            results_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "activation_cross_model.json",
            )
        with open(results_path) as f:
            data = json.load(f)

    if _is_training_dynamics(data):
        return plot_activation_training_dynamics(data=data)

    models_data = data.get("models", data.get("variants", data.get("results", {})))
    if data.get("layer_results"):
        models_data = {data.get("model", "single"): data["layer_results"]}

    if not models_data:
        print("  No data for activation analysis plot")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cmap = plt.cm.viridis
    model_names = _sort_model_by_size(list(models_data.keys()))
    colors = [cmap(i / max(len(model_names) - 1, 1)) for i in range(len(model_names))]

    for mi, model_name in enumerate(model_names):
        layer_data = models_data[model_name]
        if isinstance(layer_data, dict) and "error" in layer_data:
            continue

        layer_indices = sorted(
            [k for k in layer_data.keys() if str(k).isdigit()],
            key=lambda k: int(k),
        )
        if not layer_indices:
            continue

        x = [int(k) for k in layer_indices]
        ratios = [layer_data[k].get("rankme_ratio", 0) for k in layer_indices]

        axes[0].plot(
            x,
            ratios,
            "-o",
            color=colors[mi],
            linewidth=2,
            markersize=4,
            label=model_name,
        )

    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("RankMe Ratio")
    axes[0].set_title("Per-Layer Activation RankMe Ratio")
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=8)

    last_layer_ratios = []
    last_layer_names = []
    for model_name in model_names:
        layer_data = models_data[model_name]
        if isinstance(layer_data, dict) and "error" in layer_data:
            continue
        layer_indices = sorted(
            [k for k in layer_data.keys() if str(k).isdigit()],
            key=lambda k: int(k),
        )
        if not layer_indices:
            continue
        last_key = layer_indices[-1]
        last_layer_names.append(model_name)
        last_layer_ratios.append(layer_data[last_key].get("rankme_ratio", 0))

    if last_layer_names:
        bar_colors = [colors[model_names.index(n)] for n in last_layer_names]
        x_pos = range(len(last_layer_names))
        axes[1].bar(x_pos, last_layer_ratios, color=bar_colors, alpha=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(last_layer_names, rotation=45, ha="right")
        axes[1].set_ylabel("Last-Layer RankMe Ratio")
        axes[1].set_title("Last-Layer RankMe Ratio Comparison")
        axes[1].set_ylim(0, None)

    analysis_type = data.get("analysis", "activation")
    model_label = data.get("model", "")
    title = "Activation RankMe Analysis"
    if model_label:
        title += f" ({model_label})"
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, analysis_type)
    return analysis_type


def plot_activation_combined(
    results_dir: Optional[str] = None,
) -> str:
    _setup_style()

    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
        )

    combined: dict[str, dict] = {}

    cross_path = os.path.join(results_dir, "activation_cross_model.json")
    if os.path.exists(cross_path):
        with open(cross_path) as f:
            cross = json.load(f)
        for name, layers in cross.get("models", {}).items():
            if isinstance(layers, dict) and "error" not in layers:
                combined[name] = layers

    post_path = os.path.join(results_dir, "activation_post_training.json")
    if os.path.exists(post_path):
        with open(post_path) as f:
            post = json.load(f)
        for name, layers in post.get("variants", {}).items():
            if isinstance(layers, dict) and "error" not in layers:
                combined[name] = layers

    if not combined:
        print("  No data for combined activation plot")
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    pythia_names = sorted(
        [n for n in combined if n.startswith("pythia-")],
        key=_sort_model_by_size_key,
    )
    olmo3_names = sorted(
        [n for n in combined if n.startswith("olmo3-")],
    )
    ordered = pythia_names + olmo3_names

    pythia_cmap = plt.cm.Blues
    olmo3_cmap = plt.cm.Reds
    colors = []
    for i, name in enumerate(pythia_names):
        t = (i + 1) / (len(pythia_names) + 1)
        colors.append(pythia_cmap(0.3 + 0.6 * t))
    for i, name in enumerate(olmo3_names):
        t = (i + 1) / (len(olmo3_names) + 1)
        colors.append(olmo3_cmap(0.3 + 0.6 * t))

    for mi, model_name in enumerate(ordered):
        layer_data = combined[model_name]
        layer_indices = sorted(
            [k for k in layer_data.keys() if str(k).isdigit()],
            key=lambda k: int(k),
        )
        if not layer_indices:
            continue
        x = [int(k) for k in layer_indices]
        ratios = [layer_data[k].get("rankme_ratio", 0) for k in layer_indices]
        axes[0].plot(
            x,
            ratios,
            "-o",
            color=colors[mi],
            linewidth=2,
            markersize=4,
            label=model_name,
        )

    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("RankMe Ratio")
    axes[0].set_title("Per-Layer Activation RankMe Ratio")
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=7, ncol=2)

    last_layer_ratios = []
    last_layer_names = []
    for model_name in ordered:
        layer_data = combined[model_name]
        layer_indices = sorted(
            [k for k in layer_data.keys() if str(k).isdigit()],
            key=lambda k: int(k),
        )
        if not layer_indices:
            continue
        last_key = layer_indices[-1]
        display_name = model_name.replace("olmo3-", "O3-")
        last_layer_names.append(display_name)
        last_layer_ratios.append(layer_data[last_key].get("rankme_ratio", 0))

    if last_layer_names:
        bar_colors = colors[: len(last_layer_names)]
        x_pos = range(len(last_layer_names))
        axes[1].bar(x_pos, last_layer_ratios, color=bar_colors, alpha=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(last_layer_names, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Last-Layer RankMe Ratio")
        axes[1].set_title("Last-Layer RankMe Ratio Comparison")
        axes[1].set_ylim(0, None)

    fig.suptitle("Activation RankMe: Pythia + OLMo-3 Combined", fontsize=16, y=1.02)
    fig.tight_layout()
    _save_fig(fig, "activation_combined")
    return "activation_combined"


def _sort_model_by_size_key(name: str) -> float:
    units = {"m": 1e6, "b": 1e9}
    for suffix, mult in units.items():
        if suffix in name:
            parts = name.split(suffix)[0].rsplit("-", 1)[-1]
            try:
                return float(parts) * mult
            except ValueError:
                pass
    return 0.0


def generate_all_plots(results_dir: Optional[str] = None):
    """Generate all plots from saved results."""
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
        )

    print("\nGenerating all plots...")

    plot_files = [
        ("cross_model_size.json", plot_cross_model_size),
        ("training_dynamics_pythia-70m.json", plot_training_dynamics),
        ("training_stages.json", plot_training_stages),
        ("post_training_methods.json", plot_post_training_methods),
        ("fixed_ratio_hypothesis.json", plot_fixed_ratio_distribution),
    ]

    generated = []
    for filename, plot_fn in plot_files:
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            try:
                name = plot_fn(results_path=path)
                if name:
                    generated.append(name)
            except Exception as e:
                print(f"  Error plotting {filename}: {e}")
        else:
            print(f"  Skipping {filename}: not found")

    import glob as glob_mod

    activation_files = sorted(
        glob_mod.glob(os.path.join(results_dir, "activation_*.json"))
    )
    for act_path in activation_files:
        act_name = os.path.basename(act_path)
        if act_name in (fn for fn, _ in plot_files):
            continue
        try:
            name = plot_activation_analysis(results_path=act_path)
            if name:
                generated.append(name)
        except Exception as e:
            print(f"  Error plotting {act_name}: {e}")

    try:
        name = plot_activation_combined(results_dir=results_dir)
        if name:
            generated.append(name)
    except Exception as e:
        print(f"  Error plotting activation_combined: {e}")

    print(f"\nGenerated {len(generated)} plots in {FIGURES_DIR}")
    return generated


# =============================================================================
# Concept Dynamics: Gram + Stability heatmaps
# =============================================================================


_CONCEPT_DYNAMICS_CMAP = "RdBu_r"
_CONCEPT_DYNAMICS_VMIN = -1.0
_CONCEPT_DYNAMICS_VMAX = 1.0
_LARGE_CONCEPT_THRESHOLD = 20


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _save_concept_dynamics_fig(fig, name: str, formats=("png", "pdf")) -> str:
    target_dir = os.path.join(FIGURES_DIR, "concept_dynamics")
    os.makedirs(target_dir, exist_ok=True)
    saved: list[str] = []
    for fmt in formats:
        path = os.path.join(target_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt)
        saved.append(path)
    plt.close(fig)
    for path in saved:
        print(f"  Saved: {path}")
    return name


def _abbreviate_concept(name: str, *, max_len: int = 18) -> str:
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "\u2026"


def _pick_layer(data: dict, layer) -> str:
    candidates = [str(layer), str(int(layer)) if str(layer).isdigit() else None]
    for key in candidates:
        if key and key in data:
            return key
    if layer in data:
        return layer
    raise KeyError(f"Layer {layer!r} not found; available: {sorted(data.keys())}")


def _filter_concepts(
    matrix: np.ndarray,
    concepts: list[str],
    keep: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if keep is None:
        return matrix, concepts
    keep_set = set(keep)
    indices = [i for i, c in enumerate(concepts) if c in keep_set]
    if not indices:
        return matrix, concepts
    sub = matrix[np.ix_(indices, indices)]
    names = [concepts[i] for i in indices]
    return sub, names


def _heat_label_size(n: int) -> int:
    if n <= 8:
        return 11
    if n <= _LARGE_CONCEPT_THRESHOLD:
        return 9
    return 7


def plot_gram_heatmap(
    gram_json_path: str,
    model: str,
    checkpoint: str,
    layer,
    output_name: str | None = None,
    concepts: list[str] | None = None,
) -> str:
    """Plot a single Gram matrix ``cos(r_i, r_j)`` for one (model, ckpt, layer).

    Reads ``gram.json`` with shape ``{model: {checkpoint: {layer_str:
    {matrix, concepts}}}}`` and renders a diverging heatmap (vmin=-1,
    vmax=1, center=0, cmap RdBu_r). When the matrix holds more than 20
    concepts, tick labels are hidden to keep the figure readable.
    """
    _setup_style()
    data = _load_json(gram_json_path)
    if model not in data:
        raise KeyError(f"Model {model!r} not in gram.json")
    model_block = data[model]
    if checkpoint not in model_block:
        raise KeyError(
            f"Checkpoint {checkpoint!r} not in gram.json for model {model!r}; "
            f"available: {sorted(model_block)}"
        )
    layer_block = model_block[checkpoint]
    layer_key = _pick_layer(layer_block, layer)
    entry = layer_block[layer_key]
    matrix = np.asarray(entry["matrix"], dtype=float)
    entry_concepts = list(entry.get("concepts", []))
    matrix, entry_concepts = _filter_concepts(matrix, entry_concepts, concepts)

    n = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5 + 2), max(5, n * 0.5 + 1.5)))
    show_ticks = n <= _LARGE_CONCEPT_THRESHOLD
    labels = [_abbreviate_concept(c) for c in entry_concepts] if show_ticks else []
    xticklabels = labels if show_ticks else False
    yticklabels = labels if show_ticks else False
    sns.heatmap(
        matrix,
        ax=ax,
        vmin=_CONCEPT_DYNAMICS_VMIN,
        vmax=_CONCEPT_DYNAMICS_VMAX,
        center=0.0,
        cmap=_CONCEPT_DYNAMICS_CMAP,
        square=True,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar_kws={"label": "cosine similarity"},
    )
    if show_ticks:
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            fontsize=_heat_label_size(n),
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=_heat_label_size(n),
        )
    title_bits = [
        f"Concept Gram \u2014 {model}",
        f"ckpt={checkpoint}",
        f"layer={layer_key}",
    ]
    if not show_ticks:
        title_bits.append(f"(n={n} concepts; labels hidden)")
    ax.set_title("  ".join(title_bits))
    fig.tight_layout()

    stem = output_name or f"gram_{model}_{checkpoint}_layer{layer_key}"
    return _save_concept_dynamics_fig(fig, stem)


def plot_stability_heatmap(
    stability_json_path: str,
    model: str,
    concept: str,
    layer,
    output_name: str | None = None,
) -> str:
    """Plot a T\u00d7T checkpoint-stability heatmap for one (model, concept, layer)."""
    _setup_style()
    data = _load_json(stability_json_path)
    if model not in data:
        raise KeyError(f"Model {model!r} not in stability.json")
    if concept not in data[model]:
        raise KeyError(
            f"Concept {concept!r} not in stability.json for model {model!r}; "
            f"available: {sorted(data[model])}"
        )
    layer_block = data[model][concept]
    layer_key = _pick_layer(layer_block, layer)
    entry = layer_block[layer_key]
    matrix = np.asarray(entry["matrix"], dtype=float)
    checkpoints = list(entry.get("checkpoints", []))

    t = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, t * 0.6 + 2), max(5, t * 0.6 + 1.5)))
    show_ticks = t <= _LARGE_CONCEPT_THRESHOLD
    labels = checkpoints if show_ticks else []
    sns.heatmap(
        matrix,
        ax=ax,
        vmin=_CONCEPT_DYNAMICS_VMIN,
        vmax=_CONCEPT_DYNAMICS_VMAX,
        center=0.0,
        cmap=_CONCEPT_DYNAMICS_CMAP,
        square=True,
        xticklabels=labels or False,
        yticklabels=labels or False,
        cbar_kws={"label": "cos(r^t, r^t')"},
    )
    if show_ticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    title_bits = [
        f"Stability \u2014 {model}",
        f"concept={concept}",
        f"layer={layer_key}",
    ]
    if not show_ticks:
        title_bits.append(f"(n={t} checkpoints; labels hidden)")
    ax.set_title("  ".join(title_bits))
    fig.tight_layout()

    stem = output_name or f"stability_{model}_{concept}_layer{layer_key}"
    return _save_concept_dynamics_fig(fig, stem)


def _last_checkpoint(gram_block: dict) -> str | None:
    def _sort_key(name: str):
        import re as _re

        m = _re.search(r"step_?(\d+)", name)
        return (0, int(m.group(1))) if m else (1, name)

    keys = [k for k, v in gram_block.items() if isinstance(v, dict) and v]
    if not keys:
        return None
    return sorted(keys, key=_sort_key)[-1]


def _middle_layer(layer_block: dict):
    numeric = sorted(
        (int(k) for k in layer_block if str(k).lstrip("-").isdigit()),
    )
    if not numeric:
        return next(iter(layer_block), None)
    return numeric[len(numeric) // 2]


def plot_concept_dynamics_summary(
    output_dir: str,
    figures_subdir: str | None = None,
    *,
    models: list[str] | None = None,
    concepts: list[str] | None = None,
) -> list[str]:
    """Render a per-model summary grid from gram + stability JSON.

    For each model: plot the mid-layer Gram matrix at the last available
    checkpoint. For each requested concept (default: first three in the
    catalogue): plot the per-model checkpoint-stability heatmap. Figures
    land under ``results/figures/concept_dynamics/`` (or ``figures_subdir``
    when provided).
    """
    gram_path = os.path.join(output_dir, "gram", "gram.json")
    stability_path = os.path.join(output_dir, "stability", "stability.json")
    if not os.path.exists(gram_path) and not os.path.exists(stability_path):
        raise FileNotFoundError(
            f"No gram.json/stability.json under {output_dir}; run extraction first."
        )

    gram_data = _load_json(gram_path) if os.path.exists(gram_path) else {}
    stability_data = (
        _load_json(stability_path) if os.path.exists(stability_path) else {}
    )

    target_models = models or sorted(gram_block for gram_block in gram_data)
    generated: list[str] = []
    for model in target_models:
        model_gram = gram_data.get(model, {})
        ckpt = _last_checkpoint(model_gram)
        if ckpt is None:
            print(f"  Skipping gram for {model}: no checkpoints available")
            continue
        layer_block = model_gram[ckpt]
        layer = _middle_layer(layer_block)
        if layer is None:
            print(f"  Skipping gram for {model}/{ckpt}: no layers available")
            continue
        try:
            stem = plot_gram_heatmap(
                gram_path,
                model=model,
                checkpoint=ckpt,
                layer=layer,
                output_name=f"summary_gram_{model}_{ckpt}_layer{layer}",
                concepts=concepts,
            )
            generated.append(stem)
        except KeyError as exc:
            print(f"  Skipping gram for {model}/{ckpt}/layer{layer}: {exc}")

    summary_concepts: list[str] = []
    for model_block in stability_data.values():
        summary_concepts.extend(c for c in model_block if c not in summary_concepts)
    if concepts:
        summary_concepts = [c for c in summary_concepts if c in set(concepts)]
    summary_concepts = summary_concepts[:3]

    for model in target_models:
        model_stab = stability_data.get(model, {})
        for concept in summary_concepts:
            layer_block = model_stab.get(concept)
            if not layer_block:
                continue
            layer = _middle_layer(layer_block)
            if layer is None:
                continue
            try:
                stem = plot_stability_heatmap(
                    stability_path,
                    model=model,
                    concept=concept,
                    layer=layer,
                    output_name=f"summary_stability_{model}_{concept}_layer{layer}",
                )
                generated.append(stem)
            except KeyError as exc:
                print(f"  Skipping stability for {model}/{concept}/layer{layer}: {exc}")

    print(f"\nGenerated {len(generated)} concept-dynamics summary plots.")
    if figures_subdir is not None:
        print(
            f"(figures_subdir hint={figures_subdir!r}; output stays under FIGURES_DIR/concept_dynamics/)"
        )
    return generated
