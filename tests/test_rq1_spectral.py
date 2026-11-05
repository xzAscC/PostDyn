"""Integration test for the RQ1 spectral overlap experiment."""

from __future__ import annotations

import json
import torch
import pytest

from postdyn.extract import OnlineCovariance
from postdyn.spectra import eigensystem, energy_high_low, subsim


class TestOverlapPipeline:
    """End-to-end test of the three overlap metrics on synthetic data."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic positive and negative hidden states.

        Positive set has a strong signal along dimension 0 (stable, low
        variance) and noise elsewhere. Negative set is pure noise.
        """
        d = 32
        n = 200
        torch.manual_seed(42)

        signal = torch.zeros(n, d, dtype=torch.float64)
        signal[:, 0] = 5.0  # stable concept in dim 0
        signal[:, 1] = torch.randn(n, dtype=torch.float64) * 3.0  # high-var dim 1
        noise = torch.randn(n, d, dtype=torch.float64) * 0.1
        pos = signal + noise

        neg = torch.randn(n, d, dtype=torch.float64) * 0.1
        return pos, neg, d

    def test_dim_direction(self, synthetic_data):
        pos, neg, d = synthetic_data
        mu_pos = pos.mean(dim=0)
        mu_neg = neg.mean(dim=0)
        dim_vec = mu_pos - mu_neg
        dim_hat = dim_vec / dim_vec.norm()
        assert dim_hat[0].abs().item() > 0.8

    def test_metric1_dim_vs_covariance(self, synthetic_data):
        """DiM should NOT land in high-variance directions when the target
        concept is stable.  The high-variance dim (dim 1) dominates the top
        eigenvector, so DiM's E_high should be below 1."""
        pos, neg, d = synthetic_data
        k = d // 3

        cov = OnlineCovariance()
        cov.update(pos)
        mu_pos = cov.mean
        sigma = cov.covariance

        mu_neg = neg.mean(dim=0).to(dtype=torch.float64)
        dim_vec = mu_pos - mu_neg

        vals, vecs = eigensystem(sigma)
        result = energy_high_low(dim_vec, vecs, k)

        assert result["E_high"] < 0.9
        assert len(result["profile"]) == d

    def test_metric2_dim_vs_second_moment(self, synthetic_data):
        """The top eigenvector of M+ should track the mean, which is close
        to dim when negative mean is small."""
        pos, neg, d = synthetic_data

        cov = OnlineCovariance()
        cov.update(pos)
        mu_pos = cov.mean
        sigma = cov.covariance

        M = sigma + mu_pos.unsqueeze(1) @ mu_pos.unsqueeze(0)
        m_vals, m_vecs = eigensystem(M)

        w1 = m_vecs[:, 0]
        mu_neg = neg.mean(dim=0).to(dtype=torch.float64)
        dim_vec = mu_pos - mu_neg
        dim_hat = dim_vec / dim_vec.norm()

        overlap = float((w1 @ dim_hat).square().item())
        assert overlap > 0.5

    def test_metric3_m_top5_vs_covariance(self, synthetic_data):
        """Each M+ top eigenvector should have a well-defined E_high/E_low."""
        pos, neg, d = synthetic_data
        k = d // 3

        cov = OnlineCovariance()
        cov.update(pos)
        sigma = cov.covariance
        mu_pos = cov.mean

        vals, vecs = eigensystem(sigma)
        M = sigma + mu_pos.unsqueeze(1) @ mu_pos.unsqueeze(0)
        m_vals, m_vecs = eigensystem(M)

        for j in range(5):
            result = energy_high_low(m_vecs[:, j], vecs, k)
            assert 0.0 <= result["E_high"] <= 1.0
            assert 0.0 <= result["E_low"] <= 1.0

    def test_second_moment_from_online_covariance(self):
        """Verify M+ = Σ+ + μ+μ+^T matches direct computation."""
        d = 16
        n = 100
        torch.manual_seed(7)
        x = torch.randn(n, d, dtype=torch.float64) + 2.0

        cov = OnlineCovariance()
        cov.update(x)
        mu = cov.mean
        sigma = cov.covariance
        M_from_cov = sigma + mu.unsqueeze(1) @ mu.unsqueeze(0)

        M_direct = (x.T @ x) / n
        assert torch.allclose(M_from_cov, M_direct, atol=1e-10)

    def test_incremental_resume(self, tmp_path):
        """Completed units recorded in metrics.jsonl are skipped on resume."""
        from postdyn.persistence import append_jsonl

        metrics_path = tmp_path / "metrics.jsonl"
        append_jsonl(metrics_path, {"domain": "math", "layer": 6, "done": True})

        completed = set()
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed.add((row["domain"], row["layer"]))

        assert ("math", 6) in completed
        assert ("code", 6) not in completed
