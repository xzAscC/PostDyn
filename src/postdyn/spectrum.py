"""Hidden-state covariance maps for the spectrum ablation.

For layer l the residual output h is a vector of width d. Within one problem x, mu(x) is the
weighted mean of h over the answer-prefix tokens. The covariance pools (h - mu(x))(h - mu(x))^T
over problems, so it keeps the change along a solution and drops the gap between problems.

A map rewrites base states: h' = h + (h - mu) R L^T + shift, where A - I = L R^T has rank k, so
no d x d matrix is applied on the GPU at every token.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportMissingTypeStubs]
from torch import Tensor

from .intervention import _block_output_tensor, _layer
from .spectra import eigensystem, subsim

KINDS = ("identity", "random", "mean", "spectrum", "reorder", "full")
# Where a map acts: the block's output, or only the MLP's share of it (Olmo 3 adds
# post_feedforward_layernorm(mlp(x)) to the residual stream).
SITES = ("residual", "mlp")


def site_module(block: Any, site: str) -> Any:
    if site not in SITES:
        raise ValueError(f"unknown site: {site}")
    return block if site == "residual" else block.post_feedforward_layernorm


def fold_of(problem: str, folds: int) -> int:
    """A stable fold: Python's own hash changes from run to run."""
    return int(hashlib.sha1(problem.encode()).hexdigest(), 16) % folds


class ProblemMoments:
    """Running first and second moments of one problem, one pair per layer."""

    def __init__(self, widths: Mapping[int, int], device: torch.device) -> None:
        self.weight = 0.0
        self.first = {
            layer: torch.zeros(d, dtype=torch.float64, device=device)
            for layer, d in widths.items()
        }
        self.second = {
            layer: torch.zeros(d, d, dtype=torch.float64, device=device)
            for layer, d in widths.items()
        }

    def update(self, states: Mapping[int, Tensor], weights: Tensor) -> None:
        """Add n tokens: every layer's state has shape (n, d), ``weights`` shape (n,)."""
        self.weight += float(weights.sum().item())
        for layer, h in states.items():
            work = h.to(dtype=torch.float64)
            weight = weights.to(device=work.device, dtype=torch.float64)
            self.first[layer] += (weight[:, None] * work).sum(0)
            self.second[layer] += (work * weight[:, None]).T @ work

    def finish(self) -> tuple[dict[int, Tensor], dict[int, Tensor], float]:
        """The mean, the scatter about it (not divided by the weight), and the weight."""
        means = {layer: first / self.weight for layer, first in self.first.items()}
        scatter = {
            layer: self.second[layer] - torch.outer(self.first[layer], means[layer])
            for layer in self.first
        }
        return means, scatter, self.weight


@dataclass
class LayerStats:
    sigma: Tensor
    mean: Tensor


@dataclass
class FoldMoments:
    """Scatter per fold, so a map can be fit on the folds it is not tested on."""

    widths: Mapping[int, int]
    folds: int
    scatter: dict[int, dict[int, Tensor]] = field(default_factory=dict)
    weight: dict[int, float] = field(default_factory=dict)
    problem_means: dict[str, dict[int, Tensor]] = field(default_factory=dict)
    problem_fold: dict[str, int] = field(default_factory=dict)

    def add(
        self,
        problem: str,
        fold: int,
        means: Mapping[int, Tensor],
        scatter: Mapping[int, Tensor],
        weight: float,
    ) -> None:
        held = self.scatter.setdefault(fold, {})
        for layer, value in scatter.items():
            held[layer] = held[layer] + value.cpu() if layer in held else value.cpu()
        self.weight[fold] = self.weight.get(fold, 0.0) + weight
        self.problem_means[problem] = {
            layer: value.cpu().to(torch.float32) for layer, value in means.items()
        }
        self.problem_fold[problem] = fold

    def stats(
        self, layer: int, leave_out: int | None = None, only: Sequence[int] | None = None
    ) -> LayerStats:
        folds = [
            fold
            for fold in self.scatter
            if fold != leave_out and (only is None or fold in only)
        ]
        total = sum(self.weight[fold] for fold in folds)
        sigma = torch.stack([self.scatter[fold][layer] for fold in folds]).sum(0) / total
        means = [
            self.problem_means[problem][layer]
            for problem, fold in self.problem_fold.items()
            if fold in folds
        ]
        return LayerStats(sigma, torch.stack(means).mean(0).to(torch.float64))


@dataclass(frozen=True)
class StateMap:
    """h' = h + (h - mu) right left^T + shift; ``left`` and ``right`` have shape (d, k)."""

    left: Tensor
    right: Tensor
    shift: Tensor


def _match(u_a: Tensor, u_b: Tensor) -> Tensor:
    """pi[i] = j: column i of u_a goes with column j of u_b, by largest squared overlap."""
    overlap = (u_a.T @ u_b).square().cpu().numpy()
    _, columns = linear_sum_assignment(-overlap)  # pyright: ignore[reportUnknownVariableType]
    return torch.as_tensor(columns, dtype=torch.long)


def build_map(
    kind: str, base: LayerStats, post: LayerStats, k: int, seed: int = 0
) -> StateMap:
    """The map of one level of the ablation, from base statistics towards post statistics.

    spectrum: base vectors with post's values, rank by rank.
    reorder: base vectors with base's own values, each moved to its matched post rank.
    random: as reorder, but a random order: the negative control.
    full: post's values and vectors, rank by rank, with signs aligned to base.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown map: {kind}")
    d = len(base.mean)
    empty = torch.zeros(d, 0, dtype=torch.float64)
    if kind == "identity":
        return StateMap(empty, empty, torch.zeros(d, dtype=torch.float64))
    if kind == "mean":
        return StateMap(empty, empty, post.mean - base.mean)

    base_values, base_vectors = eigensystem(base.sigma)
    post_values, post_vectors = eigensystem(post.sigma)
    u_base, u_post = base_vectors[:, :k], post_vectors[:, :k]
    values, targets = base_values[:k], post_values[:k]
    if bool((values <= 0).any()) or bool((targets <= 0).any()):
        raise ValueError(f"top {k} eigenvalues must be positive: reduce k")
    zero = torch.zeros(d, dtype=torch.float64)
    if kind == "full":
        signs = (u_post * u_base).sum(0).sign()
        signs[signs == 0] = 1.0
        return StateMap(u_post * signs * (targets / values).sqrt() - u_base, u_base, zero)
    if kind == "reorder":
        targets = values[_match(u_base, u_post)]
    elif kind == "random":
        generator = torch.Generator().manual_seed(seed)
        targets = values[torch.randperm(k, generator=generator)]
    return StateMap(u_base * ((targets / values).sqrt() - 1.0), u_base, zero)


def apply_map(h: Tensor, state_map: StateMap, mean: Tensor) -> Tensor:
    """Rewrite states of shape (..., d). bf16 is widened because it cannot hold small updates."""
    dtype = torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
    work = h.to(dtype)
    shift = state_map.shift.to(device=h.device, dtype=dtype)
    if state_map.left.shape[1] == 0:
        return (work + shift).to(h.dtype)
    left = state_map.left.to(device=h.device, dtype=dtype)
    right = state_map.right.to(device=h.device, dtype=dtype)
    centered = work - mean.to(device=h.device, dtype=dtype)
    return (work + (centered @ right) @ left.T + shift).to(h.dtype)


def off_diagonal_share(c_base: Tensor, c_post: Tensor, k: int) -> float:
    """Share of the squared size of post's covariance, in base's top-k basis, off the diagonal.

    Near 0, training scaled base's directions. Large, it turned them, so reorder cannot explain it.
    """
    _, vectors = eigensystem(c_base)
    basis = vectors[:, :k].to(c_post.dtype)
    block = basis.T @ c_post @ basis
    return float(1.0 - (block.diagonal().square().sum() / block.square().sum()).item())


def stable_k(u_a: Tensor, u_b: Tensor, grid: tuple[int, ...], threshold: float) -> int:
    """The largest k in ``grid`` up to which two halves of the data find the same subspace."""
    best = 0
    for k in sorted(grid):
        if subsim(u_a[:, :k], u_b[:, :k]) < threshold:
            break
        best = k
    return best


@contextmanager
def residual_capture(model: Any, site: str = "residual") -> Iterator[dict[int, Tensor]]:
    """Every block's own output (or its MLP's share): hidden_capture would give the normed last
    state instead."""
    captured: dict[int, Tensor] = {}
    handles = []
    for layer in range(len(model.model.layers)):

        def hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
            captured[layer] = _block_output_tensor(output)

        handles.append(site_module(_layer(model, layer), site).register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def spectrum_summary(c_base: Tensor, c_post: Tensor, k: int) -> dict[str, float]:
    """How the two top-k subspaces and scales compare, without touching the model."""
    base_values, base_vectors = eigensystem(c_base)
    post_values, post_vectors = eigensystem(c_post)
    k = min(k, int((base_values > 0).sum()), int((post_values > 0).sum()))
    pi = _match(base_vectors[:, :k], post_vectors[:, :k])
    moved = (pi - torch.arange(k)).abs().to(torch.float64)
    return {
        "k": float(k),
        "subspace_overlap": subsim(base_vectors[:, :k], post_vectors[:, :k]),
        "off_diagonal_share": off_diagonal_share(c_base, c_post, k),
        "mean_rank_shift": float(moved.mean().item()),
        "base_variance_share": float((base_values[:k].sum() / base_values.sum()).item()),
        "post_variance_share": float((post_values[:k].sum() / post_values.sum()).item()),
        "log_trace_ratio": math.log(float((post_values.sum() / base_values.sum()).item())),
    }


__all__ = [
    "FoldMoments",
    "KINDS",
    "LayerStats",
    "ProblemMoments",
    "StateMap",
    "apply_map",
    "build_map",
    "fold_of",
    "off_diagonal_share",
    "SITES",
    "residual_capture",
    "site_module",
    "spectrum_summary",
    "stable_k",
]
