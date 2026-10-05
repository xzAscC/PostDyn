"""RQ1 Measurement 3: mean contribution rho_i across the top third of M+."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    "math": "Math\nvs. WikiText",
    "code": "Code\nvs. WikiText",
    "instruction_following": "Instruction following\nvs. WikiText",
    "safety": "Harmful\nvs. benign",
}
W1, OTHER = "#B03A2E", "#2E5E96"


def w1_stats(row: dict) -> tuple[float, float]:
    """|cos(w_1, mu_+)| and v_1 / lambda_1 (variance of w_1 relative to the top of Sigma_+)."""
    mu_norm = math.sqrt(sum(x * x for x in row["u_mu_pos"]))
    cos = math.sqrt(row["m3"]["b"][0]) / mu_norm
    return cos, row["m3"]["v"][0] / row["sigma_eigenvalues"][0]


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(ROOT / "logs" / "rq1_measurements_v2"))
    p.add_argument("--out", default=str(ROOT / "figs" / "rq1_mean_contribution.pdf"))
    args = p.parse_args()

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    rows = [json.loads(l) for l in open(Path(args.run) / "metrics.jsonl")]
    layers = sorted({r["layer"] for r in rows})
    fig, axes = plt.subplots(len(layers), len(TITLES), figsize=(7.0, 4.8), sharex=True, sharey=True)
    for r in rows:
        row, col = layers.index(r["layer"]), list(TITLES).index(r["domain"])
        ax = axes[row, col]
        rho = r["m3"]["rho"]
        rank = list(range(1, len(rho) + 1))
        ax.grid(True, color="0.92", lw=0.5)
        ax.scatter(rank[1:], rho[1:], s=3, color=OTHER, alpha=0.5, lw=0, rasterized=True)
        ax.scatter(rank[:1], rho[:1], s=45, marker="*", color=W1, zorder=5, lw=0)
        cos, ratio = w1_stats(r)
        ax.annotate(
            rf"$\rho_1={rho[0]:.2f}$" "\n" rf"$\cos(w_1,\mu_+)={cos:.4f}$" "\n" rf"$v_1/\lambda_1={ratio:.2f}$",
            xy=(rank[0], rho[0]), xytext=(0.62, 0.42), textcoords="axes fraction",
            ha="center", va="bottom", fontsize=6.3, color="0.15",
            arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.4", mutation_scale=6, shrinkB=3),
        )
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.8, len(rho) * 1.15)
        ax.set_ylim(-0.04, 1.08)
        if row == 0:
            ax.set_title(TITLES[r["domain"]])
        if col == 0:
            ax.set_ylabel(f"Layer {r['layer']}")
    fig.supxlabel(r"Eigenvalue rank $i$ of $M_+$ (1 = largest)", fontsize=7.5)
    fig.supylabel(r"Mean contribution $\rho_i=b_i/\eta_i$", fontsize=7.5)
    handles = [Line2D([], [], marker="*", ls="", color=W1, ms=8), Line2D([], [], marker="o", ls="", color=OTHER, ms=3)]
    fig.legend(handles, [r"$w_1$", r"$w_2,\dots,w_K$"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.4)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
