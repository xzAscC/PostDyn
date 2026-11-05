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
    """Match columns by maximum squared absolute overlap.

    The returned ``pi`` uses the convention ``pi[i] = j`` when column ``i``
    of ``u_a`` is assigned to column ``j`` of ``u_b``.  A machine-scale
    secondary cost makes exact-overlap ties deterministic without changing
    the primary squared-overlap objective at normal floating-point precision.
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
    cost = -(overlap_np**2)
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


def energy_profile(direction: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
    """Squared projections of a direction onto each eigenvector.

    ``eigenvectors`` has shape ``(d, d)`` with columns sorted by descending
    eigenvalue.  Returns a float64 tensor of shape ``(d,)`` where entry *i*
    is ``(u_i^T hat_d)^2``.  The direction is normalized internally.
    """
    d = direction.to(dtype=torch.float64)
    norm = d.norm()
    if norm.item() == 0.0:
        return torch.zeros(eigenvectors.shape[1], dtype=torch.float64)
    d = d / norm
    projections = eigenvectors.to(dtype=torch.float64).T @ d
    return projections.square()


def energy_high_low(
    direction: torch.Tensor,
    eigenvectors: torch.Tensor,
    k: int,
) -> dict[str, object]:
    """E_high and E_low for a direction against sorted eigenvectors.

    Returns a dict with ``E_high``, ``E_low``, ``k``, ``random_baseline``
    (``k/d``), and ``profile`` (the full per-eigenvector squared projection).
    """
    profile = energy_profile(direction, eigenvectors)
    d = int(profile.shape[0])
    e_high = float(profile[:k].sum().item())
    e_low = float(profile[d - k :].sum().item())
    return {
        "E_high": e_high,
        "E_low": e_low,
        "k": k,
        "random_baseline": k / d,
        "profile": [float(v) for v in profile],
    }


def energy_curve(
    direction: torch.Tensor, eigenvectors: torch.Tensor, ks: list[int]
) -> dict[str, list[float] | list[int]]:
    """E_high(k) and E_low(k) for each k, with random baseline k/d."""
    profile = energy_profile(direction, eigenvectors)
    cum = torch.cumsum(profile, dim=0)
    rev = torch.cumsum(profile.flip(0), dim=0)
    d = int(cum.shape[0])
    return {
        "k": list(ks),
        "E_high": [float(cum[k - 1]) for k in ks],
        "E_low": [float(rev[k - 1]) for k in ks],
        "random_baseline": [k / d for k in ks],
    }


def mean_contributions(
    sigma: torch.Tensor,
    mu: torch.Tensor,
    m_values: torch.Tensor,
    m_vectors: torch.Tensor,
    k: int,
) -> dict[str, object]:
    """Split the top-k eigenvalues of M = Sigma + mu mu^T into variance and mean parts."""
    sigma64 = sigma.to(dtype=torch.float64)
    mu64 = mu.to(dtype=torch.float64)
    positive = m_values[:k] > 0
    n_pos = int(positive.sum())
    w = m_vectors[:, :k][:, positive].to(dtype=torch.float64)
    v = ((sigma64 @ w) * w).sum(dim=0)
    b = (w.T @ mu64).square()
    eta = v + b
    rho = b / eta
    lam = torch.linalg.eigvalsh(sigma64)
    v_rank = [(lam > vi).double().mean().item() for vi in v]
    return {
        "n_positive": n_pos,
        "eta": eta.tolist(),
        "v": v.tolist(),
        "b": b.tolist(),
        "rho": rho.tolist(),
        "rho_bar": float(rho.mean()) if n_pos else 0.0,
        "rho_var": float(rho.var(unbiased=False)) if n_pos else 0.0,
        "R_K": float(b.sum() / eta.sum()) if n_pos else 0.0,
        "v_rank_fraction": v_rank,
    }


def subspace_overlap_curve(
    u: torch.Tensor, v: torch.Tensor, ks: list[int]
) -> dict[str, list[float] | list[int]]:
    """A_k = ||U_k^T V_k||_F^2 / k for each k."""
    overlap = (u.to(dtype=torch.float64).T @ v.to(dtype=torch.float64)).square()
    return {
        "k": list(ks),
        "A": [float(overlap[:k, :k].sum() / k) for k in ks],
    }


def token_cosines(
    hiddens: torch.Tensor, direction: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed cosines of nonzero rows with ``direction``; returns (cosines, row indices)."""
    h = hiddens.to(dtype=torch.float32)
    d = direction.to(device=h.device, dtype=torch.float32)
    norms = h.norm(dim=1)
    keep = torch.nonzero(norms > 0).squeeze(1)
    cos = (h[keep] @ d) / (norms[keep] * d.norm())
    return cos, keep


def _describe(x: torch.Tensor) -> dict[str, float | int]:
    if x.numel() == 0:
        return {"n": 0}
    q = torch.quantile(x.double(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=torch.float64))
    return {
        "n": int(x.numel()),
        "mean": float(x.double().mean()),
        "std": float(x.double().std(unbiased=False)),
        "q05": float(q[0]), "q25": float(q[1]), "median": float(q[2]),
        "q75": float(q[3]), "q95": float(q[4]),
        "frac_positive": float((x > 0).double().mean()),
    }


def cosine_summary(
    sequences: list[torch.Tensor], bins: tuple[int, ...] = (0, 1, 16, 64, 256, 1024)
) -> dict[str, object]:
    """Distribution of per-token cosines overall, by position bin, and without position 0."""
    vals = torch.cat(sequences)
    pos = torch.cat([torch.arange(len(s)) for s in sequences])
    ok = ~torch.isnan(vals)
    vals, pos = vals[ok], pos[ok]
    edges = list(bins) + [int(pos.max()) + 1 if pos.numel() else 1]
    by_position = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        sel = (pos >= lo) & (pos < hi)
        by_position.append({"lo": lo, "hi": hi, **_describe(vals[sel])})
    overall = _describe(vals)
    return {
        "n_tokens": overall["n"],
        **{k: v for k, v in overall.items() if k != "n"},
        "by_position": by_position,
        "excluding_first": _describe(vals[pos > 0]),
    }


def split_indices(n: int, eval_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic disjoint estimation/evaluation split of ``range(n)``."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_eval = int(round(n * eval_fraction))
    return sorted(perm[n_eval:]), sorted(perm[:n_eval])


__all__ = [
    "band_slices",
    "cosine_summary",
    "effective_rank",
    "eigensystem",
    "energy_curve",
    "energy_high_low",
    "energy_profile",
    "mean_contributions",
    "split_indices",
    "subspace_overlap_curve",
    "token_cosines",
    "frobenius_magnitude",
    "match_eigenvectors",
    "rank_displacement",
    "spectral_metrics",
    "subsim",
    "trace_sum",
]
