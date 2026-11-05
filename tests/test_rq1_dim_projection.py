"""Tests for per-example DiM projection statistics."""

from __future__ import annotations

import torch
import pytest


def _dim_projection_stats(
    hiddens: torch.Tensor, dim_vec: torch.Tensor
) -> dict[str, float]:
    """Compute cos_sim and raw projection of each row against dim_vec."""
    h = hiddens.to(dtype=torch.float64)
    d = dim_vec.to(dtype=torch.float64)

    projections = h @ d
    h_norms = h.norm(dim=1)
    d_norm = d.norm()
    cos_sims = projections / (h_norms * d_norm + 1e-30)

    return {
        "proj_mean": float(projections.mean().item()),
        "proj_std": float(projections.std().item()),
        "cos_mean": float(cos_sims.mean().item()),
        "cos_std": float(cos_sims.std().item()),
    }


class TestDimProjectionStats:

    def test_aligned_examples_high_cos(self):
        """When all examples point in the same direction as DiM, cos is high."""
        d = 32
        n = 100
        torch.manual_seed(0)
        dim_vec = torch.zeros(d, dtype=torch.float64)
        dim_vec[0] = 5.0
        hiddens = torch.randn(n, d, dtype=torch.float64) * 0.01
        hiddens[:, 0] = 3.0
        stats = _dim_projection_stats(hiddens, dim_vec)
        assert stats["cos_mean"] > 0.9

    def test_high_variance_direction_large_std(self):
        """When DiM is along a high-variance direction, proj_std is large."""
        d = 32
        n = 500
        torch.manual_seed(1)
        dim_vec = torch.zeros(d, dtype=torch.float64)
        dim_vec[0] = 1.0
        hiddens = torch.randn(n, d, dtype=torch.float64) * 0.1
        hiddens[:, 0] = torch.randn(n, dtype=torch.float64) * 10.0
        stats = _dim_projection_stats(hiddens, dim_vec)
        assert stats["proj_std"] > 5.0

    def test_stable_direction_small_std(self):
        """When DiM is along a low-variance direction, proj_std is small."""
        d = 32
        n = 500
        torch.manual_seed(2)
        dim_vec = torch.zeros(d, dtype=torch.float64)
        dim_vec[0] = 1.0
        hiddens = torch.randn(n, d, dtype=torch.float64) * 5.0
        hiddens[:, 0] = 3.0 + torch.randn(n, dtype=torch.float64) * 0.01
        stats = _dim_projection_stats(hiddens, dim_vec)
        assert stats["proj_std"] < 0.1

    def test_output_keys(self):
        d = 8
        n = 10
        stats = _dim_projection_stats(
            torch.randn(n, d, dtype=torch.float64),
            torch.randn(d, dtype=torch.float64),
        )
        assert set(stats.keys()) == {"proj_mean", "proj_std", "cos_mean", "cos_std"}

    def test_cos_bounded(self):
        d = 16
        n = 50
        torch.manual_seed(3)
        stats = _dim_projection_stats(
            torch.randn(n, d, dtype=torch.float64),
            torch.randn(d, dtype=torch.float64),
        )
        assert -1.0 <= stats["cos_mean"] <= 1.0
