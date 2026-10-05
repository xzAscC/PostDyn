"""Tests for the four RQ1 measurements (paper Eqs. focus-energy to token-dim-cosine)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from postdyn.spectra import (
    eigensystem,
    energy_curve,
    mean_contributions,
    split_indices,
    subspace_overlap_curve,
    token_cosines,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _random_spd(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    return a @ a.T / d


class TestEnergyCurve:
    def test_matches_definitions(self):
        sigma = _random_spd(12, 0)
        _, vecs = eigensystem(sigma)
        d = torch.randn(12, dtype=torch.float64)
        dh = d / d.norm()
        out = energy_curve(d, vecs, [1, 3, 4])
        for k, hi, lo in zip(out["k"], out["E_high"], out["E_low"]):
            assert hi == pytest.approx(float((vecs[:, :k].T @ dh).square().sum()))
            assert lo == pytest.approx(float((vecs[:, 12 - k :].T @ dh).square().sum()))
        assert out["random_baseline"] == pytest.approx([1 / 12, 3 / 12, 4 / 12])

    def test_full_basis_energy_is_one(self):
        _, vecs = eigensystem(_random_spd(9, 1))
        out = energy_curve(torch.randn(9, dtype=torch.float64), vecs, [9])
        assert out["E_high"][0] == pytest.approx(1.0)


class TestMeanContributions:
    def test_eta_decomposes_and_ratios_bounded(self):
        d, K = 10, 4
        sigma = _random_spd(d, 2)
        mu = torch.randn(d, dtype=torch.float64) * 3
        m_vals, m_vecs = eigensystem(sigma + torch.outer(mu, mu))
        out = mean_contributions(sigma, mu, m_vals, m_vecs, K)
        for eta, v, b in zip(out["eta"], out["v"], out["b"]):
            assert eta == pytest.approx(v + b, rel=1e-9)
        assert sum(out["rho"]) <= 1.0 + 1e-9
        assert out["n_positive"] == K
        rho = torch.tensor(out["rho"])
        assert out["rho_bar"] == pytest.approx(float(rho.mean()))
        assert out["rho_var"] == pytest.approx(float(rho.var(unbiased=False)))
        assert out["R_K"] == pytest.approx(sum(out["b"]) / sum(out["eta"]))

    def test_mean_dominant_direction(self):
        d = 6
        sigma = torch.diag(torch.tensor([1.0, 0.5, 0.4, 0.3, 0.2, 0.01], dtype=torch.float64))
        mu = torch.zeros(d, dtype=torch.float64)
        mu[5] = 10.0
        m_vals, m_vecs = eigensystem(sigma + torch.outer(mu, mu))
        out = mean_contributions(sigma, mu, m_vals, m_vecs, 2)
        assert out["rho"][0] > 0.99
        assert out["v"][0] == pytest.approx(0.01, abs=1e-9)
        assert out["v_rank_fraction"][0] == pytest.approx(5 / 6)

    def test_restricts_to_positive_eigenvalues(self):
        sigma = torch.zeros(4, 4, dtype=torch.float64)
        mu = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        m_vals, m_vecs = eigensystem(sigma + torch.outer(mu, mu))
        out = mean_contributions(sigma, mu, m_vals, m_vecs, 3)
        assert out["n_positive"] == 1
        assert out["rho"] == pytest.approx([1.0])


class TestSubspaceOverlap:
    def test_identical_bases(self):
        _, u = eigensystem(_random_spd(8, 3))
        out = subspace_overlap_curve(u, u, [1, 4, 8])
        assert out["A"] == pytest.approx([1.0, 1.0, 1.0])


class TestTokenCosines:
    def test_values_and_zero_rows_skipped(self):
        h = torch.tensor([[1.0, 0.0], [0.0, 0.0], [-2.0, 0.0], [1.0, 1.0]])
        d = torch.tensor([3.0, 0.0])
        cos, pos = token_cosines(h, d)
        assert cos.tolist() == pytest.approx([1.0, -1.0, 1 / math.sqrt(2)])
        assert pos.tolist() == [0, 2, 3]


class TestCosineSummary:
    def test_overall_and_position_bins(self):
        from postdyn.spectra import cosine_summary

        seqs = [torch.tensor([0.9, 0.1, float("nan")]), torch.tensor([0.5, -0.1])]
        out = cosine_summary(seqs, bins=(0, 1, 16))
        assert out["n_tokens"] == 4
        assert out["mean"] == pytest.approx(0.35)
        assert out["frac_positive"] == pytest.approx(0.75)
        first, rest = out["by_position"]
        assert (first["lo"], first["hi"], first["n"]) == (0, 1, 2)
        assert first["mean"] == pytest.approx(0.7)
        assert rest["n"] == 2 and rest["mean"] == pytest.approx(0.0)
        assert out["excluding_first"]["mean"] == pytest.approx(0.0)


class TestSplit:
    def test_disjoint_and_deterministic(self):
        a_est, a_ev = split_indices(100, 0.2, seed=0)
        b_est, b_ev = split_indices(100, 0.2, seed=0)
        assert a_est == b_est and a_ev == b_ev
        assert len(a_ev) == 20 and len(a_est) == 80
        assert not set(a_est) & set(a_ev)


class TestLayerMeasurements:
    def test_low_variance_mean_shift_lands_in_low_band(self):
        from scripts.run_rq1_measurements import k_grid, layer_measurements

        g = torch.Generator().manual_seed(0)
        d, n = 30, 400
        scales = torch.linspace(3.0, 0.05, d, dtype=torch.float64)
        pos = torch.randn(n, d, generator=g, dtype=torch.float64) * scales
        neg = torch.randn(n, d, generator=g, dtype=torch.float64) * scales
        pos[:, -1] += 5.0
        row, dim = layer_measurements(pos[:300], neg[:300], pos[300:], neg[300:])
        assert row["K"] == 10 and row["m1"]["k"] == k_grid(d)
        assert row["m1"]["E_low"][-1] > 0.9
        assert sum(row["m1"]["profile"]) == pytest.approx(1.0)
        assert row["m3"]["rho"][0] > 0.9
        assert row["dim_projection_eval"]["pos"]["mean"] > row["dim_projection_eval"]["neg"]["mean"]
