"""Plot per-layer weight-update metrics: rows SFT / RL, columns model families, one line per component.

Reads <logs>/<run>/<label>.jsonl from run_weight_diff.py and writes one PDF per metric,
weights_<label>_<metric>.pdf.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = [  # (column title, SFT run, RL run)
    ("OLMo 3 7B\n", "olmo3-7b_sft", "olmo3-7b_rl"),
    ("OLMo 3 32B\n", "olmo3-32b_sft", "olmo3-32b_rl"),
    ("Qwen2.5-Math-7B\nScaleML-RLHF", "qwen-scaleml_sft", "qwen-scaleml_rl"),
    ("Qwen2.5-Math-7B\ncutelemonlili", "qwen-cutelemonlili_sft", "qwen-cutelemonlili_rl"),
]
ROW_TITLES = ("SFT", "RL")
METRICS = {  # key: (axis label, scale to percent, log y)
    "sparsity": ("Update Sparsity (%)", True, False),
    "one_minus_cos": ("1 − cos (row mean)", False, True),
    "gini": ("Gini of |ΔW|", False, False),
    "rank_ratio": ("Rank Ratio (%)", True, False),
    "erank_ratio": ("Effective Rank Ratio (%)", True, False),
    "overlap": ("Principal Overlap (%)", True, False),
}
_FURTHER = ["#3A6DB5", "#D04A4A", "#B88A12", "#3E8E3E"]
# Matrices: solid. Biases: dotted in their matrix's color. Norms: dashed.
STYLE = {
    "self_attn.q_proj.weight": ("q", "#007E80", "-"),
    "self_attn.k_proj.weight": ("k", "#B77544", "-"),
    "self_attn.v_proj.weight": ("v", "#8471AA", "-"),
    "self_attn.o_proj.weight": ("o", _FURTHER[0], "-"),
    "mlp.gate_proj.weight": ("gate", _FURTHER[1], "-"),
    "mlp.up_proj.weight": ("up", _FURTHER[2], "-"),
    "mlp.down_proj.weight": ("down", _FURTHER[3], "-"),
    "self_attn.q_proj.bias": ("q bias", "#007E80", ":"),
    "self_attn.k_proj.bias": ("k bias", "#B77544", ":"),
    "self_attn.v_proj.bias": ("v bias", "#8471AA", ":"),
    "self_attn.q_norm.weight": ("q_norm", "#007E80", "--"),
    "self_attn.k_norm.weight": ("k_norm", "#B77544", "--"),
    "input_layernorm.weight": ("input_ln", "#203342", "--"),
    "post_attention_layernorm.weight": ("post_attn_ln", "#505963", "--"),
    "post_feedforward_layernorm.weight": ("post_ffn_ln", "#74818A", "--"),
}
INK, MUTED, RULE = "#203342", "#74818A", "#DCE3E6"


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of one checkpoint comparison; the last row wins if a tensor appears twice."""
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["name"]] = row
    return list(rows.values())


def series(rows: list[dict[str, Any]], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    """Component -> (layers, values) for decoder-layer tensors that have the metric."""
    points: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("layer") is not None and row.get(metric) is not None:
            points[row["component"]].append((row["layer"], row[metric]))
    return {c: ([l for l, _ in sorted(p)], [v for _, v in sorted(p)]) for c, p in points.items()}


def _style(plt: Any) -> None:
    from matplotlib import font_manager

    for f in Path("/usr/share/texmf-dist/fonts/opentype/adobe/sourcesanspro").glob("SourceSansPro-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Source Sans Pro", "mathtext.fontset": "custom",
        "mathtext.rm": "Source Sans Pro", "mathtext.it": "Source Sans Pro:italic",
        "mathtext.bf": "Source Sans Pro:bold", "mathtext.fallback": "stixsans", "pdf.fonttype": 42,
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
        "xtick.direction": "out", "ytick.direction": "out",
    })


def plot_metric(logs: Path, label: str, metric: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style(plt)
    ylabel, percent, log = METRICS[metric]
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(5.5, 2.35), layout="constrained", squeeze=False)
    seen: dict[str, tuple[str, str, str]] = {}
    for col, (title, *runs) in enumerate(FAMILIES):
        for row, run in enumerate(runs):
            ax = axes[row, col]
            ax.grid(axis="y", color=RULE, lw=0.4)
            ax.set_axisbelow(True)
            path = logs / run / f"{label}.jsonl"
            if not (logs / run / f"{label}.summary.json").is_file():  # only finished comparisons
                note = "running" if path.is_file() else "not run yet"
                ax.text(0.5, 0.5, note, ha="center", va="center", color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                for component, (layers, values) in series(load_rows(path), metric).items():
                    name, color, ls = STYLE.get(component, (component, MUTED, "-"))
                    ys = [100 * v for v in values] if percent else values
                    if log:  # exact zeros (unchanged tensors) cannot be drawn on a log axis
                        ys = [y if y > 0 else float("nan") for y in ys]
                    ax.plot(layers, ys, color=color, ls=ls, lw=0.9)
                    seen[component] = (name, color, ls)
                if log:
                    ax.set_yscale("log")
            if row == 0:
                ax.set_title(title, fontsize=7, loc="left", linespacing=0.85)
            if col == 0:
                ax.set_ylabel(ROW_TITLES[row], fontweight="bold")
            if row == 1:
                ax.set_xlabel("Layer")
    fig.supylabel(ylabel, fontsize=8, color=INK)
    order = [c for c in STYLE if c in seen] + [c for c in seen if c not in STYLE]
    handles = [Line2D([], [], color=seen[c][1], ls=seen[c][2], lw=0.9) for c in order]
    fig.legend(handles, [seen[c][0] for c in order], loc="outside upper center", ncol=8, frameon=False,
               fontsize=6.5, handlelength=2.2, columnspacing=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=str(ROOT / "logs" / "weights"))
    parser.add_argument("--label", default="final")
    parser.add_argument("--metrics", nargs="*", default=list(METRICS))
    parser.add_argument("--out-dir", nargs="+", default=[str(ROOT / "figs")])
    args = parser.parse_args(argv)
    unknown = set(args.metrics) - set(METRICS)
    if unknown:
        parser.error(f"unknown metrics: {sorted(unknown)}")
    for metric in args.metrics:
        for out_dir in args.out_dir:
            out = Path(out_dir) / f"weights_{args.label}_{metric}.pdf"
            plot_metric(Path(args.logs), args.label, metric, out)
            print(out)


if __name__ == "__main__":
    main()
