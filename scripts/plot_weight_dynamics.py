"""Plot weight-update metrics over training: rows SFT / RL, columns model families.

Each projection matrix is one line (norm weights and biases are left out): its mean over layers at every finished intermediate checkpoint
(stepNNNNNN.jsonl), with a +-1 std band over layers. The final checkpoint continues the line and its
tick reads "final"; it sits at its real step where the repository name gives one (FINAL_STEP), otherwise
one median checkpoint gap after the last intermediate. Log-scale metrics use the geometric mean and a band of x/÷ 10^std(log10 value).
Writes weights_steps_<metric>.pdf.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_weight_diff import (  # noqa: E402
    FAMILIES, INK, MATRICES, METRICS, MUTED, ROW_TITLES, RULE, STYLE, _style, load_rows, series,
)


FINAL_STEP = {"qwen-scaleml_sft": 320, "qwen-scaleml_rl": 180, "qwen-cutelemonlili_sft": 1818,
              "qwen-cutelemonlili_rl": 620}


@dataclass
class Trajectory:
    steps: list[int]
    mean: list[float]
    std: list[float]
    final: tuple[float, float] | None


def _mean_std(values: list[float], log: bool) -> tuple[float, float]:
    xs = [math.log10(v) for v in values if v > 0] if log else values
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return (10**m, s) if log else (m, s)


def trajectory(run_dir: Path, metric: str, log: bool = False) -> dict[str, Trajectory]:
    """Component -> mean and std over layers per finished checkpoint, intermediates sorted by step."""
    labels = sorted(p.name.removesuffix(".summary.json") for p in run_dir.glob("*.summary.json"))
    out: dict[str, Trajectory] = {}
    for label in labels:
        per_comp = series(load_rows(run_dir / f"{label}.jsonl"), metric)
        for comp, (_, values) in per_comp.items():
            traj = out.setdefault(comp, Trajectory([], [], [], None))
            m, s = _mean_std(values, log)
            if label == "final":
                traj.final = (m, s)
            elif match := re.fullmatch(r"step(\d+)", label):
                traj.steps.append(int(match.group(1)))
                traj.mean.append(m)
                traj.std.append(s)
    return out


def _short(step: int) -> str:
    """1596 -> 1.6k, 43000 -> 43k, 950 -> 950."""
    return f"{step / 1000:.1f}".rstrip("0").rstrip(".") + "k" if step >= 1000 else str(step)


def final_x(run: str, steps: list[int]) -> float:
    """x position of the final checkpoint: its real step, else one median gap after the last intermediate."""
    if run in FINAL_STEP:
        return FINAL_STEP[run]
    gaps = sorted(b - a for a, b in zip(steps, steps[1:]))
    return steps[-1] + (gaps[len(gaps) // 2] if gaps else 1)


def step_ticks(ax: object, x_final: float) -> None:
    """About three step ticks inside the axis, then a "final" tick; drop step ticks crowding it."""
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    lo, _ = ax.get_xlim()
    ticks = [t for t in ax.get_xticks() if lo <= t < x_final - 0.2 * (x_final - lo)]
    ax.set_xticks([*ticks, x_final], [*(_short(int(t)) for t in ticks), "final"])


def _band(mean: list[float], std: list[float], log: bool) -> tuple[list[float], list[float]]:
    if log:
        return [m / 10**s for m, s in zip(mean, std)], [m * 10**s for m, s in zip(mean, std)]
    return [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)]


def plot_metric(logs: Path, metric: str, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style(plt)
    ylabel, percent, log = METRICS[metric]
    scale = 100 if percent else 1
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(5.65, 3.9), layout="constrained", squeeze=False)
    seen: dict[str, tuple[str, str, str]] = {}
    for col, (title, *runs) in enumerate(FAMILIES):
        for row, run in enumerate(runs):
            ax = axes[row, col]
            ax.grid(axis="y", color=RULE, lw=0.4)
            ax.set_axisbelow(True)
            trajs = trajectory(logs / run, metric, log) if (logs / run).is_dir() else {}
            steps = sorted({s for t in trajs.values() for s in t.steps})
            if not steps:
                ax.text(0.5, 0.5, "not run yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                x_final = final_x(run, steps)
                for comp, t in trajs.items():
                    if comp not in MATRICES:
                        continue
                    name, color, ls = STYLE.get(comp, (comp, MUTED, "-"))
                    points = list(zip(t.steps, t.mean, t.std))
                    if t.final is not None:
                        points.append((x_final, *t.final))
                    xs = [x for x, _, _ in points]
                    mean = [scale * m for _, m, _ in points]
                    std = [s if log else scale * s for _, _, s in points]
                    lo, hi = _band(mean, std, log)
                    ax.plot(xs, mean, color=color, ls=ls, lw=0.9)
                    ax.fill_between(xs, lo, hi, color=color, alpha=0.10, lw=0)
                    seen[comp] = (name, color, ls)
                if log:
                    ax.set_yscale("log")
                step_ticks(ax, x_final)
            if row == 0:
                ax.set_title(title, fontsize=7, loc="left", linespacing=0.85)
            if col == 0:
                ax.set_ylabel(ROW_TITLES[row], fontweight="bold")
            if row == 1:
                ax.set_xlabel("Training Step")
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
    parser.add_argument("--metrics", nargs="*", default=list(METRICS))
    parser.add_argument("--out-dir", nargs="+", default=[str(ROOT / "figs")])
    args = parser.parse_args(argv)
    unknown = set(args.metrics) - set(METRICS)
    if unknown:
        parser.error(f"unknown metrics: {sorted(unknown)}")
    for metric in args.metrics:
        for out_dir in args.out_dir:
            out = Path(out_dir) / f"weights_steps_{metric}.pdf"
            plot_metric(Path(args.logs), metric, out)
            print(out)


if __name__ == "__main__":
    main()
