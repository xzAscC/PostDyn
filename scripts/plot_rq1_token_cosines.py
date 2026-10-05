"""Plot per-position DiM cosine (mean ± std) for RQ1 Measurement 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = {
    "math": ("Math", "WikiText"),
    "code": ("Code", "WikiText"),
    "instruction_following": ("Instruction following", "WikiText"),
    "safety": ("Harmful", "Benign"),
}
LAYERS = (6, 17, 28)


def position_stats(
    cos: torch.Tensor, lengths: torch.Tensor, min_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean and std of cosines at each token position with at least ``min_count`` values."""
    pos = torch.cat([torch.arange(int(n)) for n in lengths])
    ok = ~torch.isnan(cos)
    cos, pos = cos[ok].double(), pos[ok]
    count = torch.bincount(pos)
    total = torch.bincount(pos, weights=cos)
    sq = torch.bincount(pos, weights=cos.square())
    keep = torch.nonzero(count >= min_count).squeeze(1)
    mean = total[keep] / count[keep]
    std = (sq[keep] / count[keep] - mean.square()).clamp_min(0).sqrt()
    return keep, mean, std


def main() -> None:
    import matplotlib.pyplot as plt
    from safetensors.torch import load_file

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=str(ROOT / "logs" / "rq1_measurements"))
    p.add_argument("--out", default=str(ROOT / "figs" / "rq1_token_cosine_position.pdf"))
    p.add_argument("--min-count", type=int, default=50)
    args = p.parse_args()

    fig, axes = plt.subplots(len(LAYERS), len(DOMAINS), figsize=(13, 7.5), sharex=True)
    for col, (dom, (pos_name, neg_name)) in enumerate(DOMAINS.items()):
        data = load_file(str(Path(args.run) / f"token_cosines_{dom}.safetensors"))
        for row, layer in enumerate(LAYERS):
            ax = axes[row, col]
            for side, name, color in (("pos", pos_name, "C0"), ("neg", neg_name, "C3")):
                x, m, s = position_stats(
                    data[f"L{layer}_{side}_cos"], data[f"L{layer}_{side}_len"], args.min_count
                )
                x = (x + 1).numpy()
                ax.plot(x, m.numpy(), color=color, lw=1, label=name)
                ax.fill_between(x, (m - s).numpy(), (m + s).numpy(), color=color, alpha=0.2, lw=0)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(f"{pos_name} vs. {neg_name}", fontsize=10)
                ax.legend(fontsize=8, loc="upper left")
            if col == 0:
                ax.set_ylabel(f"Layer {layer}\ncosine with DiM")
            if row == len(LAYERS) - 1:
                ax.set_xlabel("token position (1-indexed)")
    fig.tight_layout()
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
