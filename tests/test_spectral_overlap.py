"""Tests for DiM–covariance and second-moment overlap metrics."""

from __future__ import annotations

import torch
import pytest

from postdyn.spectra import energy_profile, energy_high_low


class TestEnergyProfile:
    def test_eigenvector_direction_concentrates(self):
        """A direction equal to u_3 should have all energy on index 3."""
        d = 16
        vectors = torch.eye(d, dtype=torch.float64)
        direction = vectors[:, 3].clone()
        profile = energy_profile(direction, vectors)
        assert profile.shape == (d,)
        assert pytest.approx(profile[3].item(), abs=1e-12) == 1.0
        assert pytest.approx(profile.sum().item(), abs=1e-12) == 1.0

    def test_uniform_direction(self):
        """A direction equally mixed across all eigenvectors gives 1/d per entry."""
        d = 8
        vectors = torch.eye(d, dtype=torch.float64)
        direction = torch.ones(d, dtype=torch.float64) / d**0.5
        profile = energy_profile(direction, vectors)
        expected = 1.0 / d
        for i in range(d):
            assert pytest.approx(profile[i].item(), abs=1e-12) == expected

    def test_non_unit_direction_is_normalized(self):
        """energy_profile should normalize the direction internally."""
        d = 4
        vectors = torch.eye(d, dtype=torch.float64)
        direction = torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        profile = energy_profile(direction, vectors)
        assert pytest.approx(profile[0].item(), abs=1e-12) == 1.0
        assert pytest.approx(profile.sum().item(), abs=1e-12) == 1.0


class TestEnergyHighLow:
    def test_low_variance_direction(self):
        """Direction equal to the last eigenvector → E_low≈1, E_high≈0."""
        d = 12
        k = 4
        vectors = torch.eye(d, dtype=torch.float64)
        direction = vectors[:, d - 1].clone()
        result = energy_high_low(direction, vectors, k)
        assert pytest.approx(result["E_high"], abs=1e-12) == 0.0
        assert pytest.approx(result["E_low"], abs=1e-12) == 1.0
        assert result["k"] == k
        assert pytest.approx(result["random_baseline"], abs=1e-12) == k / d

    def test_high_variance_direction(self):
        """Direction equal to the first eigenvector → E_high≈1, E_low≈0."""
        d = 12
        k = 4
        vectors = torch.eye(d, dtype=torch.float64)
        direction = vectors[:, 0].clone()
        result = energy_high_low(direction, vectors, k)
        assert pytest.approx(result["E_high"], abs=1e-12) == 1.0
        assert pytest.approx(result["E_low"], abs=1e-12) == 0.0

    def test_uniform_direction(self):
        """A uniform direction gives E_high ≈ E_low ≈ k/d."""
        d = 12
        k = 4
        vectors = torch.eye(d, dtype=torch.float64)
        direction = torch.ones(d, dtype=torch.float64) / d**0.5
        result = energy_high_low(direction, vectors, k)
        expected = k / d
        assert pytest.approx(result["E_high"], abs=1e-10) == expected
        assert pytest.approx(result["E_low"], abs=1e-10) == expected

    def test_rotated_basis(self):
        """Works correctly with a non-identity orthonormal basis."""
        d = 8
        k = 2
        torch.manual_seed(42)
        q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
        direction = q[:, 0].clone()
        result = energy_high_low(direction, q, k)
        assert pytest.approx(result["E_high"], abs=1e-12) == 1.0
        assert pytest.approx(result["E_low"], abs=1e-12) == 0.0

    def test_profile_length(self):
        """Profile is included and has correct length."""
        d = 16
        k = 4
        vectors = torch.eye(d, dtype=torch.float64)
        direction = vectors[:, 0].clone()
        result = energy_high_low(direction, vectors, k)
        assert len(result["profile"]) == d
