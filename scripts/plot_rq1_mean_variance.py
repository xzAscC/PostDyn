"""RQ1 Measurement 3: mean contribution rho_i across the top third of M+."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    "math": "Math vs. WikiText",
    "code": "Code vs. WikiText",
    "instruction_following": "Instruction following vs. WikiText",
    "safety": "Harmful vs. Benign",
}


def w1_stats(row: dict) -> tuple[float, float]:
    """|cos(w_1, mu_+)| and v_1 / lambda_1 (variance of w_1 relative to the top of Sigma_+)."""
    mu_norm = math.sqrt(sum(x * x for x in row["u_mu_pos"]))
    cos = math.sqrt(row["m3"]["b"][0]) / mu_norm
    return cos, row["m3"]["v"][0] / row["sigma_eigenvalues"][0]


def main() -> None:
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(ROOT / "logs" / "rq1_measurements_v2"))
    p.add_argument("--out", default=str(ROOT / "figs" / "rq1_mean_contribution.pdf"))
    args = p.parse_args()

    rows = [json.loads(l) for l in open(Path(args.run) / "metrics.jsonl")]
    layers = sorted({r["layer"] for r in rows})
    fig, axes = plt.subplots(len(layers), len(TITLES), figsize=(13, 7.5), sharey=True)
    for r in rows:
        ax = axes[layers.index(r["layer"]), list(TITLES).index(r["domain"])]
        eta, rho = r["m3"]["eta"], r["m3"]["rho"]
        ax.scatter(eta[1:], rho[1:], s=4, color="C0", alpha=0.6, lw=0, rasterized=True)
        ax.scatter(eta[:1], rho[:1], s=60, marker="*", color="C3", zorder=5)
        cos, ratio = w1_stats(r)
        ax.annotate(
            rf"$w_1$: $\rho_1$={rho[0]:.2f}" "\n" rf"cos($w_1,\mu_+$)={cos:.3f}" "\n" rf"$v_1/\lambda_1$={ratio:.2f}",
            xy=(eta[0], rho[0]), xytext=(0.97, 0.6), textcoords="axes fraction",
            ha="right", fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6),
        )
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        if r["layer"] == layers[0]:
            ax.set_title(TITLES[r["domain"]], fontsize=10)
        if r["domain"] == "math":
            ax.set_ylabel(f"Layer {r['layer']}\n" + r"$\rho_i = b_i/\eta_i$")
        if r["layer"] == layers[-1]:
            ax.set_xlabel(r"eigenvalue $\eta_i$ of $M_+$ (top third)")
    fig.tight_layout()
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
