"""Plot per-row rotation angles (degrees) of the weight update, from run_weight_diff.py --angles.

Three figures, rows SFT / RL and columns model families, one shared log y axis per figure:
  weights_angle_final_matrices.pdf  start vs final, projection matrices: mean over rows per layer,
                                    band = +-1 std over rows
  weights_angle_final_vectors.pdf   start vs final, norm weights and biases (one row each)
  weights_angle_steps.pdf           matrices over training: geometric mean over layers per
                                    checkpoint, band = x/÷ 10^std(log10) over layers; final at right
Each panel has a dashed gray line at the bf16 floor: the median over tensors of the angle a row
turns if every entry moves by one bfloat16 step.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_weight_diff import FAMILIES, INK, MUTED, ROW_TITLES, RULE, STYLE, _style, load_rows  # noqa: E402
from scripts.plot_weight_dynamics import _band, _mean_std, _short  # noqa: E402

MATRICES = [c for c in STYLE if c.endswith("_proj.weight")]


def angle_rows(run_dir: Path, label: str) -> list[dict[str, Any]] | None:
    """Rows of a finished angle comparison, or None."""
    if not (run_dir / f"{label}.angles.summary.json").is_file():
        return None
    return load_rows(run_dir / f"{label}.angles.jsonl")


def _is_matrix(row: dict[str, Any]) -> bool:
    return row["component"] in MATRICES


def layer_curves(rows: list[dict[str, Any]], matrices: bool) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Component -> (layers, mean angle, std over rows) for decoder-layer tensors of one kind."""
    points: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        if r.get("layer") is not None and _is_matrix(r) == matrices:
            points.setdefault(r["component"], []).append((r["layer"], r["theta_mean"], r["theta_std"] or 0.0))
    return {c: ([l for l, _, _ in sorted(p)], [m for _, m, _ in sorted(p)], [s for _, _, s in sorted(p)])
            for c, p in points.items()}


def bf16_floor(rows: list[dict[str, Any]], matrices: bool) -> float:
    return statistics.median(r["theta_ulp"] for r in rows if r.get("layer") is not None and _is_matrix(r) == matrices)


def _grid(plt: Any) -> tuple[Any, Any]:
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(5.65, 3.9), layout="constrained", squeeze=False,
                             sharey=True)
    for ax in axes.flat:
        ax.set_yscale("log", nonpositive="clip")
        ax.grid(axis="y", color=RULE, lw=0.4)
        ax.set_axisbelow(True)
    for col, (title, *_) in enumerate(FAMILIES):
        axes[0, col].set_title(title, fontsize=7, loc="left", linespacing=0.85)
    for row, name in enumerate(ROW_TITLES):
        axes[row, 0].set_ylabel(name, fontweight="bold")
    return fig, axes


def _finish(fig: Any, seen: dict[str, tuple[str, str, str]], xlabel: str, out: Path) -> None:
    from matplotlib.lines import Line2D

    for ax in fig.axes[len(FAMILIES):]:
        ax.set_xlabel(xlabel)
    fig.supylabel("Row Rotation Angle (°)", fontsize=8, color=INK)
    order = [c for c in STYLE if c in seen]
    handles = [Line2D([], [], color=seen[c][1], ls=seen[c][2], lw=0.9) for c in order]
    handles.append(Line2D([], [], color=MUTED, ls=(0, (4, 2)), lw=0.8))
    fig.legend(handles, [seen[c][0] for c in order] + ["bf16 floor"], loc="outside upper center",
               ncol=8, frameon=False, fontsize=6.5, handlelength=2.2, columnspacing=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def _empty(ax: Any, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, transform=ax.transAxes)


def plot_final(logs: Path, out: Path, matrices: bool) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = _grid(plt)
    seen: dict[str, tuple[str, str, str]] = {}
    for col, (_, *runs) in enumerate(FAMILIES):
        for row, run in enumerate(runs):
            ax = axes[row, col]
            rows = angle_rows(logs / run, "final")
            if rows is None:
                _empty(ax, "not run yet")
                continue
            for comp, (layers, mean, std) in layer_curves(rows, matrices).items():
                name, color, ls = STYLE.get(comp, (comp, MUTED, "-"))
                ax.plot(layers, mean, color=color, ls=ls, lw=0.9)
                if matrices:
                    ax.fill_between(layers, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)],
                                    color=color, alpha=0.12, lw=0)
                seen[comp] = (name, color, ls)
            ax.axhline(bf16_floor(rows, matrices), color=MUTED, ls=(0, (4, 2)), lw=0.8)
    _finish(fig, seen, "Layer", out)
    plt.close(fig)


def plot_steps(logs: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = _grid(plt)
    seen: dict[str, tuple[str, str, str]] = {}
    for col, (_, *runs) in enumerate(FAMILIES):
        for row, run in enumerate(runs):
            ax = axes[row, col]
            run_dir = logs / run
            labels = sorted(p.name.removesuffix(".angles.summary.json") for p in run_dir.glob("step*.angles.summary.json"))
            steps = [int(re.fullmatch(r"step(\d+)", lab).group(1)) for lab in labels]
            if not steps:
                _empty(ax, "not run yet")
                continue
            per_label = {lab: layer_curves(load_rows(run_dir / f"{lab}.angles.jsonl"), True) for lab in labels}
            final = angle_rows(run_dir, "final")
            final_curves = layer_curves(final, True) if final else {}
            x_final = steps[-1] + 0.2 * max(steps[-1] - steps[0], 1)
            for comp in MATRICES:
                stats = [_mean_std(per_label[lab][comp][1], log=True) for lab in labels if comp in per_label[lab]]
                if not stats:
                    continue
                name, color, ls = STYLE[comp]
                mean, std = [m for m, _ in stats], [s for _, s in stats]
                lo, hi = _band(mean, std, log=True)
                ax.plot(steps, mean, color=color, ls=ls, lw=0.9)
                ax.fill_between(steps, lo, hi, color=color, alpha=0.10, lw=0)
                if comp in final_curves:
                    fm, fs = _mean_std(final_curves[comp][1], log=True)
                    flo, fhi = _band([fm], [fs], log=True)
                    ax.errorbar([x_final], [fm], yerr=[[fm - flo[0]], [fhi[0] - fm]], fmt="o", ms=2, color=color,
                                elinewidth=0.6, capsize=1.5)
                seen[comp] = (name, color, ls)
            ref_rows = final or load_rows(run_dir / f"{labels[-1]}.angles.jsonl")
            ax.axhline(bf16_floor(ref_rows, True), color=MUTED, ls=(0, (4, 2)), lw=0.8)
            ax.axvline((steps[-1] + x_final) / 2, color=RULE, lw=0.6)
            ax.set_xticks([steps[0], steps[-1], x_final], [_short(steps[0]), _short(steps[-1]), "final"])
    _finish(fig, seen, "Training Step", out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", default=str(ROOT / "logs" / "weights"))
    parser.add_argument("--out-dir", nargs="+", default=[str(ROOT / "figs")])
    args = parser.parse_args(argv)
    logs = Path(args.logs)
    for out_dir in map(Path, args.out_dir):
        plot_final(logs, out_dir / "weights_angle_final_matrices.pdf", matrices=True)
        plot_final(logs, out_dir / "weights_angle_final_vectors.pdf", matrices=False)
        plot_steps(logs, out_dir / "weights_angle_steps.pdf")
        print(out_dir)


if __name__ == "__main__":
    main()
