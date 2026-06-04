"""Tests for effective_rank module."""

import math
import torch
import numpy as np
import pytest

from src.effective_rank import (
    effective_rank,
    effective_rank_ratio,
    svd_entropy_normalized,
    stable_rank,
    alpha_req,
    compute_all_metrics,
    batch_effective_rank,
    RankMetrics,
)


class TestEffectiveRank:
    """Test entropy-based effective rank computation."""

    def test_identity_matrix_ratio_is_one(self):
        """Identity matrix: all singular values are equal → erank = n → ratio = 1.0"""
        W = torch.eye(10, dtype=torch.float32)
        ratio = effective_rank_ratio(W)
        assert abs(ratio - 1.0) < 0.01, f"Expected ~1.0, got {ratio}"

    def test_identity_matrix_effective_rank_equals_dim(self):
        """Identity matrix should have erank ≈ min(m,n)."""
        W = torch.eye(8, dtype=torch.float32)
        er = effective_rank(W)
        assert abs(er - 8.0) < 0.01, f"Expected ~8.0, got {er}"

    def test_rank_one_matrix_low_ratio(self):
        """Rank-1 matrix: only one nonzero singular value → ratio ≈ 0."""
        W = torch.outer(torch.ones(10), torch.ones(10))  # rank-1
        ratio = effective_rank_ratio(W)
        assert ratio <= 0.11, f"Expected <= 0.11, got {ratio}"

    def test_rank_one_matrix_effective_rank_near_one(self):
        """Rank-1 matrix: erank ≈ 1."""
        W = torch.outer(torch.randn(20), torch.randn(20))  # rank-1
        er = effective_rank(W)
        assert abs(er - 1.0) < 0.01, f"Expected ~1.0, got {er}"

    def test_random_matrix_ratio_in_range(self):
        """Random matrix: ratio should be in (0, 1]."""
        torch.manual_seed(42)
        W = torch.randn(32, 64)
        ratio = effective_rank_ratio(W)
        assert 0.0 < ratio <= 1.0, f"Ratio {ratio} not in (0, 1]"

    def test_scale_invariance(self):
        """erank(cW) = erank(W) for scalar c."""
        W = torch.randn(16, 16)
        er1 = effective_rank(W)
        er2 = effective_rank(2.5 * W)
        assert abs(er1 - er2) < 0.01, f"Not scale-invariant: {er1} vs {er2}"

    def test_rectangular_matrix(self):
        """Works with m != n."""
        W = torch.randn(10, 50)
        er = effective_rank(W)
        max_rank = min(10, 50)
        assert 1.0 <= er <= max_rank, f"erank {er} not in [1, {max_rank}]"

    def test_nearly_zero_matrix(self):
        """Matrix with very small entries should still produce valid results."""
        W = torch.ones(5, 5) * 1e-15
        er = effective_rank(W)
        # All singular values nearly equal → erank ≈ 5
        assert er > 0, f"Expected positive erank, got {er}"


class TestSVDEntropy:
    """Test normalized SVD entropy."""

    def test_identity_matrix_entropy_is_one(self):
        """Identity matrix: all singular values equal → entropy = 1."""
        W = torch.eye(8, dtype=torch.float32)
        ent = svd_entropy_normalized(W)
        assert abs(ent - 1.0) < 0.01, f"Expected ~1.0, got {ent}"

    def test_rank_one_entropy_near_zero(self):
        """Rank-1 matrix: entropy should be near 0."""
        W = torch.outer(torch.ones(10), torch.ones(10))
        ent = svd_entropy_normalized(W)
        assert ent <= 0.01, f"Expected <= 0.01, got {ent}"

    def test_entropy_in_range(self):
        """Normalized entropy should be in [0, 1]."""
        W = torch.randn(16, 32)
        ent = svd_entropy_normalized(W)
        assert 0.0 <= ent <= 1.0, f"Entropy {ent} not in [0, 1]"


class TestStableRank:
    """Test stable rank computation."""

    def test_identity_matrix_stable_rank(self):
        """Identity matrix: stable_rank = n."""
        W = torch.eye(6, dtype=torch.float32)
        sr = stable_rank(W)
        assert abs(sr - 6.0) < 0.01, f"Expected ~6.0, got {sr}"

    def test_rank_one_stable_rank(self):
        """Rank-1 matrix: stable_rank ≈ 1."""
        W = torch.outer(torch.ones(8), torch.ones(8))
        sr = stable_rank(W)
        assert abs(sr - 1.0) < 0.01, f"Expected ~1.0, got {sr}"


class TestAlphaReQ:
    """Test power-law decay rate estimation."""

    def test_uniform_spectrum_low_alpha(self):
        """Uniform singular values → α ≈ 0."""
        W = torch.eye(16, dtype=torch.float32)
        alpha = alpha_req(W)
        assert alpha < 0.1, f"Expected α ≈ 0 for uniform, got {alpha}"

    def test_fast_decay_high_alpha(self):
        """Matrix with rapidly decaying singular values → α > 0."""
        # Create matrix with σ_i = 1/i^2
        U = torch.randn(20, 20)
        V = torch.randn(20, 20)
        sigma = 1.0 / torch.arange(1, 21, dtype=torch.float32) ** 2
        W = U @ torch.diag(sigma) @ V.T
        alpha = alpha_req(W)
        assert alpha > 0.5, f"Expected α > 0.5 for fast decay, got {alpha}"


class TestComputeAllMetrics:
    """Test comprehensive metrics computation."""

    def test_returns_rank_metrics(self):
        """compute_all_metrics returns a RankMetrics instance."""
        W = torch.randn(10, 20)
        metrics = compute_all_metrics(W)
        assert isinstance(metrics, RankMetrics)

    def test_metrics_fields_valid(self):
        """All metric fields should be reasonable."""
        W = torch.randn(16, 32)
        m = compute_all_metrics(W)
        assert m.shape == (16, 32)
        assert m.max_rank == 16
        assert 1.0 <= m.effective_rank <= 16.0
        assert 0.0 < m.effective_rank_ratio <= 1.0
        assert 0.0 <= m.svd_entropy <= 1.0
        assert m.stable_rank >= 1.0
        assert m.top_singular_value > 0
        assert m.condition_number >= 1.0
        assert m.frobenius_norm > 0

    def test_known_identity_matrix(self):
        """Identity matrix should produce predictable metrics."""
        W = torch.eye(8, dtype=torch.float32)
        m = compute_all_metrics(W)
        assert abs(m.effective_rank_ratio - 1.0) < 0.01
        assert abs(m.svd_entropy - 1.0) < 0.01
        assert abs(m.stable_rank - 8.0) < 0.01


class TestBatchEffectiveRank:
    """Test batch computation."""

    def test_batch_basic(self):
        """batch_effective_rank computes metrics for multiple matrices."""
        weights = {
            "layer1.weight": torch.randn(10, 20),
            "layer2.weight": torch.randn(16, 16),
        }
        results = batch_effective_rank(weights)
        assert len(results) == 2
        assert "layer1.weight" in results
        assert "layer2.weight" in results
        for name, metrics in results.items():
            assert isinstance(metrics, RankMetrics)

    def test_batch_skips_non_2d(self):
        """Non-2D tensors should be skipped."""
        weights = {
            "bias": torch.randn(10),  # 1D - should be skipped
            "weight": torch.randn(10, 20),
        }
        results = batch_effective_rank(weights)
        assert len(results) == 1
        assert "weight" in results

    def test_batch_max_dim_filter(self):
        """max_dim should skip large matrices."""
        weights = {
            "small": torch.randn(4, 4),
            "large": torch.randn(100, 100),
        }
        results = batch_effective_rank(weights, max_dim=10)
        assert "small" in results
        assert "large" not in results


class TestNumericalStability:
    """Test numerical edge cases."""

    def test_very_small_values(self):
        """Matrix with very small but nonzero values."""
        W = torch.randn(8, 8) * 1e-10
        er = effective_rank(W)
        assert er > 0 and not math.isnan(er) and not math.isinf(er)

    def test_very_large_values(self):
        """Matrix with large values."""
        W = torch.randn(8, 8) * 1e6
        er = effective_rank(W)
        assert er > 0 and not math.isnan(er) and not math.isinf(er)

    def test_wide_matrix(self):
        """Wide matrix (m << n)."""
        W = torch.randn(4, 128)
        er = effective_rank(W)
        assert 1.0 <= er <= 4.0

    def test_tall_matrix(self):
        """Tall matrix (m >> n)."""
        W = torch.randn(128, 4)
        er = effective_rank(W)
        assert 1.0 <= er <= 4.0
