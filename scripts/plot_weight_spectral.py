"""Plot update norm, top-10% principal angles and spectrum drift (run_weight_diff.py --spectral).

Follows Fig. 4 of The Path Not Taken (arXiv 2511.08567). Five figures, columns = model families:
  weights_spectral_final_<metric>.pdf  start vs final per layer, one line per projection matrix, rows SFT / RL,
                                       for pa_u_max (largest principal angle of the top-k left subspace), nss
                                       (normalized spectral shift) and rel_update_norm (||dW||_F / ||W0||_F)
  weights_spectral_exemplar.pdf        one matrix (middle layer, --exemplar component): all top-k principal
                                       angles (U solid, V dashed) and the top-k singular values before/after
  weights_spectral_steps.pdf           rows = the three metrics, geometric mean over all projection matrices
                                       per checkpoint vs training progress (step / last intermediate step);
                                       SFT and RL in one panel, final checkpoint at the right
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_weight_diff import FAMILIES, INK, MUTED, ROW_TITLES, RULE, STYLE, _style, load_rows  # noqa: E402

MATRICES = [c for c in STYLE if c.endswith("_proj.weight")]
METRICS = {
    "pa_u_max": "Max Principal Angle, Top-10% U (°)",
    "nss": "Spectrum Drift (NSS)",
    "rel_update_norm": "Relative Update Norm",
}
ROW_STYLE = {"SFT": ("#203342", "-", "o"), "RL": ("#B77544", "--", "s")}


def spectral_rows(run_dir: Path, label: str) -> list[dict[str, Any]] | None:
    """Rows of a finished spectral comparison, or None."""
    if not (run_dir / f"{label}.spectral.summary.json").is_file():
        return None
    return load_rows(run_dir / f"{label}.spectral.jsonl")


def per_layer(rows: list[dict[str, Any]], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    """Projection matrix -> (layers, values)."""
    points: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        if r.get("layer") is not None and r["component"] in MATRICES and r.get(metric) is not None:
            points.setdefault(r["component"], []).append((r["layer"], r[metric]))
    return {c: ([l for l, _ in sorted(p)], [v for _, v in sorted(p)]) for c, p in points.items()}


def matrix_mean(rows: list[dict[str, Any]], metric: str) -> float:
    """Geometric mean of the metric over all decoder-layer projection matrices."""
    vals = [r[metric] for r in rows if r.get("layer") is not None and r["component"] in MATRICES
            and r.get(metric) is not None and r[metric] > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def _legend(fig: Any, entries: list[tuple[str, str, str, str | None]], ncol: int = 8) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3, lw=0.9) for _, c, ls, mk in entries]
    fig.legend(handles, [n for n, *_ in entries], loc="outside upper center", ncol=ncol, frameon=False,
               fontsize=6.5, handlelength=2.2, columnspacing=1.0)


def plot_final(logs: Path, metric: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(5.65, 3.9), layout="constrained", squeeze=False,
                             sharey=True)
    for col, (title, *runs) in enumerate(FAMILIES):
        axes[0, col].set_title(title, fontsize=7, loc="left", linespacing=0.85)
        for row, run in enumerate(runs):
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.grid(axis="y", color=RULE, lw=0.4)
            ax.set_axisbelow(True)
            rows = spectral_rows(logs / run, "final")
            if rows is None:
                ax.text(0.5, 0.5, "not run yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
                continue
            for comp, (layers, values) in per_layer(rows, metric).items():
                name, color, ls = STYLE[comp]
                ax.plot(layers, [v if v > 0 else float("nan") for v in values], color=color, ls=ls, lw=0.9)
    for row, name in enumerate(ROW_TITLES):
        axes[row, 0].set_ylabel(name, fontweight="bold")
    for ax in axes[1]:
        ax.set_xlabel("Layer")
    fig.supylabel(METRICS[metric], fontsize=8, color=INK)
    _legend(fig, [(STYLE[c][0], STYLE[c][1], STYLE[c][2], None) for c in MATRICES])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_exemplar(logs: Path, out: Path, component: str) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = plt.subplots(4, len(FAMILIES), figsize=(5.65, 6.2), layout="constrained", squeeze=False)
    for col, (title, *runs) in enumerate(FAMILIES):
        axes[0, col].set_title(title, fontsize=7, loc="left", linespacing=0.85)
        for row, run in enumerate(runs):
            ax_pa, ax_sv = axes[row, col], axes[2 + row, col]
            for ax in (ax_pa, ax_sv):
                ax.grid(axis="y", color=RULE, lw=0.4)
                ax.set_axisbelow(True)
            rows = spectral_rows(logs / run, "final")
            cand = [r for r in rows or [] if r["component"] == component and "pa_u" in r]
            if not cand:
                for ax in (ax_pa, ax_sv):
                    ax.text(0.5, 0.5, "not run yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                    ax.set_xticks([])
                continue
            cand.sort(key=lambda r: r["layer"])
            r = cand[len(cand) // 2]  # middle layer
            idx = range(1, len(r["pa_u"]) + 1)
            ax_pa.plot(idx, r["pa_u"], color="#3A6DB5", lw=0.9)
            ax_pa.plot(idx, r["pa_v"], color="#D04A4A", ls="--", lw=0.9)
            ax_pa.text(0.03, 0.92, f"layer {r['layer']}", transform=ax_pa.transAxes, fontsize=6.5, color=MUTED, va="top")
            ax_sv.plot(idx, r["sv0_top"], color="#74818A", lw=0.9)
            ax_sv.plot(idx, r["sv1_top"], color="#8471AA", ls="--", lw=0.9)
    labels = ["SFT\nPrincipal Angle (°)", "RL\nPrincipal Angle (°)", "SFT\nSingular Value", "RL\nSingular Value"]
    for row, lab in enumerate(labels):
        axes[row, 0].set_ylabel(lab, linespacing=0.85)
    for ax in axes[3]:
        ax.set_xlabel("Index i (top 10%)")
    _legend(fig, [("Left (U)", "#3A6DB5", "-", None), ("Right (V)", "#D04A4A", "--", None),
                  ("Start", "#74818A", "-", None), ("Final", "#8471AA", "--", None)], ncol=4)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_steps(logs: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = plt.subplots(len(METRICS), len(FAMILIES), figsize=(5.65, 4.8), layout="constrained", squeeze=False)
    for col, (title, *runs) in enumerate(FAMILIES):
        axes[0, col].set_title(title, fontsize=7, loc="left", linespacing=0.85)
        for row_name, run in zip(ROW_TITLES, runs):
            run_dir = logs / run
            labels = sorted(p.name.removesuffix(".spectral.summary.json")
                            for p in run_dir.glob("step*.spectral.summary.json"))
            steps = [int(re.fullmatch(r"step(\d+)", lab).group(1)) for lab in labels]
            final = spectral_rows(run_dir, "final")
            color, ls, marker = ROW_STYLE[row_name]
            for i, metric in enumerate(METRICS):
                ax = axes[i, col]
                if steps:
                    xs = [s / steps[-1] for s in steps]
                    ys = [matrix_mean(load_rows(run_dir / f"{lab}.spectral.jsonl"), metric) for lab in labels]
                    ax.plot(xs, ys, color=color, ls=ls, marker=marker, ms=2.5, lw=0.9)
                if final:
                    ax.plot([1.2], [matrix_mean(final, metric)], color=color, marker=marker, ms=3.5, ls="none")
    for i, metric in enumerate(METRICS):
        axes[i, 0].set_ylabel(METRICS[metric].replace(", Top-10% U", "\nTop-10% U"), linespacing=0.85)
        for ax in axes[i]:
            ax.set_yscale("log")
            ax.grid(axis="y", color=RULE, lw=0.4)
            ax.set_axisbelow(True)
            ax.axvline(1.1, color=RULE, lw=0.6)
            ax.set_xticks([0, 0.5, 1.0, 1.2], ["0", "0.5", "1", "final"])
    for ax in axes[-1]:
        ax.set_xlabel("Step / Last Step")
    _legend(fig, [("SFT", *ROW_STYLE["SFT"]), ("RL", *ROW_STYLE["RL"])], ncol=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=str(ROOT / "logs" / "weights"))
    parser.add_argument("--out-dir", nargs="+", default=[str(ROOT / "figs")])
    parser.add_argument("--exemplar", default="mlp.down_proj.weight", help="component for the exemplar figure")
    args = parser.parse_args(argv)
    logs = Path(args.logs)
    for out_dir in map(Path, args.out_dir):
        for metric in METRICS:
            plot_final(logs, metric, out_dir / f"weights_spectral_final_{metric}.pdf")
        plot_exemplar(logs, out_dir / "weights_spectral_exemplar.pdf", args.exemplar)
        plot_steps(logs, out_dir / "weights_spectral_steps.pdf")
        print(out_dir)


if __name__ == "__main__":
    main()
