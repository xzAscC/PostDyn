"""Numerical contract tests for the frozen spectra API."""

from __future__ import annotations

import math

import pytest
import torch
from postdyn.spectra import (
    band_slices,
    effective_rank,
    eigensystem,
    frobenius_magnitude,
    match_eigenvectors,
    rank_displacement,
    spectral_metrics,
    subsim,
    trace_sum,
)

DTYPE = torch.float64


def _orthonormal_basis(d: int, k: int) -> torch.Tensor:
    matrix = torch.randn(d, k, dtype=DTYPE)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis


def test_eigensystem_orders_and_reconstructs_random_psd_matrix() -> None:
    torch.manual_seed(0)
    d, n = 12, 24
    x = torch.randn(d, n, dtype=DTYPE)
    sigma = x @ x.T / n

    vals, vectors = eigensystem(sigma)

    assert vals.dtype == DTYPE
    assert vectors.dtype == DTYPE
    assert torch.all(vals[:-1] >= vals[1:])
    torch.testing.assert_close(
        vectors.T @ vectors,
        torch.eye(d, dtype=DTYPE),
        atol=1e-8,
        rtol=1e-8,
    )
    reconstructed = (vectors * vals.unsqueeze(0)) @ vectors.T
    torch.testing.assert_close(sigma, reconstructed, atol=1e-8, rtol=1e-8)


def test_eigensystem_rank_deficient_matrix_has_only_expected_zero_eigenvalues() -> None:
    torch.manual_seed(1)
    d = 12
    basis = _orthonormal_basis(d, 3)
    sigma = basis @ torch.diag(torch.tensor([9.0, 4.0, 1.0], dtype=DTYPE)) @ basis.T

    vals, vectors = eigensystem(sigma)
    zero = torch.isclose(vals, torch.zeros_like(vals), atol=1e-8, rtol=0.0)

    assert torch.isfinite(vals).all()
    assert torch.isfinite(vectors).all()
    assert int(zero.sum()) == d - 3
    assert torch.all(vals[~zero] > 0)


def test_effective_rank_identity_is_dimension() -> None:
    vals = torch.ones(12, dtype=DTYPE)

    assert effective_rank(vals) == pytest.approx(12.0, abs=1e-12)


def test_effective_rank_matches_participation_ratio_formula() -> None:
    vals = torch.tensor([4.0] + [1.0] * 11, dtype=DTYPE)
    expected = vals.sum().item() ** 2 / vals.square().sum().item()

    assert effective_rank(vals) == pytest.approx(expected, abs=1e-12)


def test_effective_rank_is_monotonic_for_more_balanced_spectra() -> None:
    balanced = torch.tensor([3.0, 3.0], dtype=DTYPE)
    concentrated = torch.tensor([3.0, 0.001], dtype=DTYPE)

    assert effective_rank(balanced) > effective_rank(concentrated)


def test_spectral_magnitudes() -> None:
    vals = torch.tensor([2.0, 1.0, 0.0], dtype=DTYPE)

    assert frobenius_magnitude(vals) == pytest.approx(math.sqrt(5.0), abs=1e-12)
    assert trace_sum(vals) == pytest.approx(3.0, abs=1e-12)


def test_band_slices_use_floor_thirds_and_cover_all_indices() -> None:
    assert band_slices(12) == (slice(0, 4), slice(4, 8), slice(8, 12))

    bands = band_slices(13)
    assert [band.stop - band.start for band in bands] == [4, 4, 5]
    covered = [index for band in bands for index in range(band.start, band.stop)]
    assert covered == list(range(13))


def test_subsim_is_one_for_equal_bases_and_in_band_rotations() -> None:
    torch.manual_seed(2)
    basis = _orthonormal_basis(12, 4)
    rotation = _orthonormal_basis(4, 4)

    assert subsim(basis, basis) == pytest.approx(1.0, abs=1e-12)
    rotated_value = subsim(basis, basis @ rotation)
    assert 0.0 < rotated_value <= 1.0


def test_subsim_hand_cases() -> None:
    e0 = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    e1 = torch.tensor([[0.0], [1.0]], dtype=DTYPE)

    assert subsim(e0, e1) == pytest.approx(0.0, abs=1e-12)
    assert subsim(e0, e0) == pytest.approx(1.0, abs=1e-12)


def test_match_eigenvectors_recovers_known_signed_permutation_deterministically() -> (
    None
):
    torch.manual_seed(3)
    d = 8
    u_a = _orthonormal_basis(d, d)
    permutation = torch.tensor([2, 0, 7, 4, 1, 6, 3, 5], dtype=torch.long)
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0], dtype=DTYPE)
    u_b = torch.empty_like(u_a)
    u_b[:, permutation] = u_a * signs

    recovered = match_eigenvectors(u_a, u_b)
    recovered_again = match_eigenvectors(u_a, u_b)

    assert torch.equal(recovered, permutation)
    assert torch.equal(torch.sort(recovered).values, torch.arange(d))
    assert torch.equal(recovered, recovered_again)


def test_rank_displacement_is_absolute_rank_difference() -> None:
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)

    assert torch.equal(rank_displacement(permutation), torch.tensor([2, 1, 1]))


def test_spectral_metrics_agree_with_individual_functions() -> None:
    vals = torch.tensor([4.0, 1.0, 1.0], dtype=DTYPE)

    metrics = spectral_metrics(vals)

    assert set(metrics) == {"effective_rank", "frobenius", "trace"}
    assert metrics["effective_rank"] == pytest.approx(effective_rank(vals), abs=1e-12)
    assert metrics["frobenius"] == pytest.approx(frobenius_magnitude(vals), abs=1e-12)
    assert metrics["trace"] == pytest.approx(trace_sum(vals), abs=1e-12)


def test_eigensystem_repeated_eigenvalues_preserve_degenerate_subspace() -> None:
    torch.manual_seed(4)
    basis = _orthonormal_basis(12, 12)
    spectrum = torch.tensor([7.0] * 3 + [2.0] * 4 + [0.5] * 5, dtype=DTYPE)
    sigma = basis @ torch.diag(spectrum) @ basis.T

    vals_a, vectors_a = eigensystem(sigma)
    vals_b, vectors_b = eigensystem(sigma)

    assert torch.all(vals_a[:-1] >= vals_a[1:])
    torch.testing.assert_close(vals_a, vals_b, atol=1e-8, rtol=1e-8)
    assert subsim(vectors_a[:, :3], vectors_b[:, :3]) == pytest.approx(1.0, abs=1e-6)
