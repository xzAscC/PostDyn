"""Plot DiM cosine (mean ± std) against token position for RQ1 Measurement 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "math": "Math\nvs. WikiText",
    "code": "Code\nvs. WikiText",
    "instruction_following": "Instruction following\nvs. WikiText",
    "safety": "Harmful\nvs. benign",
}
LAYERS = (6, 17, 28)
TARGET, REFERENCE = "#2E5E96", "#B03A2E"


def binned_stats(
    cos: torch.Tensor, lengths: torch.Tensor, edges: list[int], min_count: int
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Token-weighted mean/std of cosines in position bins [lo, hi).

    Bins reached by fewer than ``min_count`` prompts are dropped. Centers are
    1-indexed geometric midpoints.
    """
    pos = torch.cat([torch.arange(int(n)) for n in lengths])
    ok = ~torch.isnan(cos)
    cos, pos = cos[ok].double(), pos[ok]
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if int((lengths > lo).sum()) < min_count:
            continue
        x = cos[(pos >= lo) & (pos < hi)]
        if x.numel() == 0:
            continue
        centers.append(np.sqrt((lo + 1) * hi))
        means.append(x.mean())
        stds.append(x.std(unbiased=False))
    return np.array(centers), torch.stack(means), torch.stack(stds)


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    from safetensors.torch import load_file

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(ROOT / "logs" / "rq1_measurements"))
    p.add_argument("--out", default=str(ROOT / "figs" / "rq1_token_cosine_position.pdf"))
    p.add_argument("--min-count", type=int, default=50)
    args = p.parse_args()

    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    edges = sorted({0, *np.unique(np.round(np.logspace(0, np.log10(2048), 45)).astype(int)).tolist()})
    fig, axes = plt.subplots(len(LAYERS), len(DOMAINS), figsize=(7.0, 4.8), sharex="col", sharey="row")
    for col, (dom, title) in enumerate(DOMAINS.items()):
        data = load_file(str(Path(args.run) / f"token_cosines_{dom}.safetensors"))
        x_max = 1.0
        for row, layer in enumerate(LAYERS):
            ax = axes[row, col]
            ax.axhline(0, color="0.55", lw=0.6, ls=(0, (3, 2)))
            ax.grid(True, which="major", color="0.92", lw=0.5)
            for side, color in (("neg", REFERENCE), ("pos", TARGET)):
                x, m, s = binned_stats(
                    data[f"L{layer}_{side}_cos"], data[f"L{layer}_{side}_len"], edges, args.min_count
                )
                m, s = m.numpy(), s.numpy()
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
                ax.plot(x, m, color=color, lw=1.3)
                x_max = max(x_max, float(x[-1]))
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ticks = [t for t in (1, 10, 100, 1000) if t <= x_max * 1.05]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_xlim(0.9, x_max * 1.1)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"Layer {layer}")
    fig.supxlabel("Token position", fontsize=7.5)
    fig.supylabel("Cosine with DiM", fontsize=7.5)
    handles = [Line2D([], [], color=TARGET, lw=1.3), Line2D([], [], color=REFERENCE, lw=1.3)]
    fig.legend(handles, ["Target domain", "Reference domain"], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, w_pad=0.4)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
