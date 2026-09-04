"""Spectral helpers for covariance eigensystems and subspace comparisons."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import (  # pyright: ignore[reportMissingTypeStubs]
    linear_sum_assignment,  # pyright: ignore[reportUnknownVariableType]
)


def eigensystem(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a float64 eigensystem in descending eigenvalue order.

    ``torch.linalg.eigh`` supplies an orthonormal basis, including for
    rank-deficient matrices.  Reordering only (rather than clipping) keeps
    the returned factors suitable for reconstructing the input matrix.
    """
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {tuple(sigma.shape)}")

    eigh = cast(
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        torch.linalg.eigh,
    )
    values, vectors = eigh(sigma.to(dtype=torch.float64))
    return torch.flip(values, dims=(0,)), torch.flip(vectors, dims=(1,))


def effective_rank(values: torch.Tensor) -> float:
    """Return the participation-ratio effective rank of ``values``."""
    values64 = values.to(dtype=torch.float64)
    denominator = values64.square().sum()
    if denominator.item() == 0.0:
        return 0.0
    numerator = values64.sum().square()
    return float((numerator / denominator).item())


def frobenius_magnitude(values: torch.Tensor) -> float:
    """Return the Frobenius magnitude represented by eigenvalues."""
    values64 = values.to(dtype=torch.float64)
    return float(values64.square().sum().sqrt().item())


def trace_sum(values: torch.Tensor) -> float:
    """Return the trace represented by an eigenvalue vector."""
    return float(values.to(dtype=torch.float64).sum().item())


def band_slices(d: int) -> tuple[slice, slice, slice]:
    """Split ``range(d)`` into high, middle, and low floor-third bands."""
    if d < 0:
        raise ValueError(f"Dimension must be non-negative, got {d}")
    third = d // 3
    return slice(0, third), slice(third, 2 * third), slice(2 * third, d)


def subsim(u_a: torch.Tensor, u_b: torch.Tensor) -> float:
    """Return normalized squared overlap between two equal-width bases."""
    if u_a.ndim != 2 or u_b.ndim != 2:
        raise ValueError("Bases must be rank-2 tensors")
    if u_a.shape[0] != u_b.shape[0] or u_a.shape[1] != u_b.shape[1]:
        raise ValueError("Bases must have matching shape (d, k)")
    k = int(u_a.shape[1])
    if k == 0:
        return 0.0

    overlap = u_a.to(dtype=torch.float64).T @ u_b.to(dtype=torch.float64)
    return float((overlap.square().sum() / k).item())


def match_eigenvectors(u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
    """Match columns by maximum absolute overlap.

    The returned ``pi`` uses the convention ``pi[i] = j`` when column ``i``
    of ``u_a`` is assigned to column ``j`` of ``u_b``.  A machine-scale
    secondary cost makes exact-overlap ties deterministic without changing
    the primary overlap objective at normal floating-point precision.
    """
    if u_a.ndim != 2 or u_b.ndim != 2:
        raise ValueError("Eigenvector matrices must be rank-2 tensors")
    if u_a.shape != u_b.shape:
        raise ValueError("Eigenvector matrices must have matching shape (d, d)")
    if u_a.shape[0] != u_a.shape[1]:
        raise ValueError("Eigenvector matrices must be square")

    overlap = (u_a.to(dtype=torch.float64).T @ u_b.to(dtype=torch.float64)).abs()
    overlap_np = cast(NDArray[np.float64], overlap.detach().cpu().numpy())
    size = int(u_a.shape[0])
    scale = max(1.0, float(overlap.max().item()) if overlap.numel() else 0.0)
    epsilon = np.finfo(np.float64).eps * scale / (16.0 * max(1, size) ** 4)
    cost = -overlap_np.copy()
    column_rank = np.arange(size, dtype=np.float64)
    for row, priority in enumerate(range(size, 0, -1)):
        cost[row] += epsilon * priority * column_rank
    assignment = cast(Callable[..., tuple[object, object]], linear_sum_assignment)
    _, assigned_columns = assignment(cost)
    return torch.as_tensor(assigned_columns, dtype=torch.long, device=u_a.device)


def rank_displacement(pi: torch.Tensor) -> torch.Tensor:
    """Return the absolute displacement of every matched rank."""
    ranks = torch.arange(pi.numel(), dtype=torch.long, device=pi.device)
    return (pi.to(dtype=torch.long) - ranks).abs()


def spectral_metrics(values: torch.Tensor) -> dict[str, float]:
    """Return the contract's scalar metrics for an eigenvalue spectrum."""
    return {
        "effective_rank": effective_rank(values),
        "frobenius": frobenius_magnitude(values),
        "trace": trace_sum(values),
    }


__all__ = [
    "band_slices",
    "effective_rank",
    "eigensystem",
    "frobenius_magnitude",
    "match_eigenvectors",
    "rank_displacement",
    "spectral_metrics",
    "subsim",
    "trace_sum",
]
