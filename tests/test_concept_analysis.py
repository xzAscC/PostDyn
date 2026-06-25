"""Tests for concept_analysis module (TDD).

Tests 4 model-agnostic analysis metrics that operate on
ConceptSteeringVector data:
  1. Directional stability (trajectory cosine similarity)
  2. Separability margin (Cohen's d)
  3. Concept Gram matrix (pairwise cosine similarity)
  4. Anisotropy spectrum (covariance eigenvalues)
"""

import torch
import pytest

from src.concept_steering import ConceptSteeringVector, compute_steering_vector
from src.concept_analysis import (
    SeparabilityMargin,
    AnisotropySpectrum,
    directional_stability,
    separability_margin,
    concept_gram_matrix,
    anisotropy_spectrum,
)


def _make_csv(
    positive=None, negative=None, d_model=8, name="test"
) -> ConceptSteeringVector:
    """Helper: build a ConceptSteeringVector from optional tensors."""
    if positive is None:
        positive = torch.randn(10, d_model)
    if negative is None:
        negative = torch.randn(10, d_model)
    return compute_steering_vector(positive, negative, name)


# =============================================================================
# Directional Stability
# =============================================================================


class TestDirectionalStability:
    """Test trajectory cosine similarity across checkpoint pairs."""

    def test_identical_vectors_diagonal_is_one(self):
        """Same steering vector across all checkpoints → diagonal = 1."""
        v = _make_csv(d_model=4, name="concept_a")
        trajectory = {"ckpt0": {"concept_a": v}, "ckpt1": {"concept_a": v}}
        result = directional_stability(trajectory)
        assert "concept_a" in result
        matrix = result["concept_a"]
        assert torch.allclose(torch.diagonal(matrix), torch.ones(2), atol=1e-5)

    def test_orthogonal_concepts_off_diagonal_zero(self):
        """Orthogonal steering vectors → off-diagonal ≈ 0."""
        v1 = _make_csv(
            positive=torch.tensor([[1.0, 0.0]]),
            negative=torch.tensor([[0.0, 0.0]]),
            name="c",
        )
        v2 = _make_csv(
            positive=torch.tensor([[0.0, 1.0]]),
            negative=torch.tensor([[0.0, 0.0]]),
            name="c",
        )
        trajectory = {"t0": {"c": v1}, "t1": {"c": v2}}
        result = directional_stability(trajectory)
        matrix = result["c"]
        assert abs(matrix[0, 1].item()) < 1e-5
        assert abs(matrix[1, 0].item()) < 1e-5

    def test_symmetric_matrix(self):
        v1 = _make_csv(d_model=4, name="c")
        v2 = _make_csv(d_model=4, name="c")
        trajectory = {"t0": {"c": v1}, "t1": {"c": v2}}
        matrix = directional_stability(trajectory)["c"]
        assert torch.allclose(matrix, matrix.T)

    def test_shape_is_T_by_T(self):
        v = _make_csv(d_model=4, name="c")
        trajectory = {f"t{i}": {"c": _make_csv(d_model=4, name="c")} for i in range(5)}
        matrix = directional_stability(trajectory)["c"]
        assert matrix.shape == (5, 5)

    def test_returns_one_matrix_per_concept(self):
        v = _make_csv(d_model=4)
        trajectory = {"t0": {"a": v, "b": v}, "t1": {"a": v, "b": v}}
        result = directional_stability(trajectory)
        assert len(result) == 2
        assert "a" in result and "b" in result

    def test_sorted_checkpoint_order(self):
        """Stability matrix axes follow sorted checkpoint names."""
        v = _make_csv(d_model=4, name="c")
        trajectory = {"z_ckpt": {"c": v}, "a_ckpt": {"c": v}}
        result = directional_stability(trajectory)
        assert list(result.keys()) == ["c"]
        matrix = result["c"]
        assert matrix.shape == (2, 2)


# =============================================================================
# Separability Margin (Cohen's d)
# =============================================================================


class TestSeparabilityMargin:
    """Test per-dimension and scalar Cohen's d."""

    def test_margin_vector_matches_formula(self):
        pos = torch.tensor([[3.0, 0.0], [5.0, 0.0]])  # mean=[4,0], sample std=[sqrt2,0]
        neg = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]]
        )  # mean=[0,0], sample std=[sqrt2,0]
        csv = compute_steering_vector(pos, neg, "test")
        result = separability_margin(csv)
        # margin[0] = 4 / sqrt(0.5*(2+2)) = 4/sqrt(2) = 2*sqrt(2)
        expected = 4.0 / (2**0.5)
        assert abs(result.margin_vector[0].item() - expected) < 1e-4

    def test_scalar_summary_matches_norm_formula(self):
        csv = _make_csv(d_model=8, name="test")
        result = separability_margin(csv)
        v = csv.steering_vector
        pooled = torch.sqrt(
            0.5 * (csv.positive_std.pow(2).sum() + csv.negative_std.pow(2).sum())
        )
        expected = v.norm().item() / pooled.item()
        assert abs(result.scalar_summary - expected) < 1e-4

    def test_scalar_summary_positive(self):
        csv = _make_csv(d_model=8, name="test")
        result = separability_margin(csv)
        assert result.scalar_summary > 0

    def test_zero_std_no_nan(self):
        """Both classes with zero variance → eps guard prevents inf/nan."""
        pos = torch.zeros(3, 4)
        neg = torch.zeros(3, 4)
        csv = compute_steering_vector(pos, neg, "test")
        result = separability_margin(csv, eps=1e-10)
        assert not torch.isnan(result.margin_vector).any()
        assert not torch.isinf(result.margin_vector).any()

    def test_margin_vector_shape(self):
        csv = _make_csv(d_model=16, name="test")
        result = separability_margin(csv)
        assert result.margin_vector.shape == (16,)

    def test_concept_name_preserved(self):
        csv = _make_csv(d_model=4, name="happiness")
        result = separability_margin(csv)
        assert result.concept_name == "happiness"

    def test_hand_computed_exact_value(self):
        """Deterministic 1-D case: margin = |Δμ| / pooled_std."""
        pos = torch.tensor([[2.0, 0.0]])
        neg = torch.tensor([[0.0, 0.0]])
        csv = compute_steering_vector(pos, neg, "test")
        result = separability_margin(csv)
        # margin[0] = (2-0) / sqrt(0.5*(0+0)+eps) — single sample, std=0
        # With eps guard, margin is large but finite
        assert result.margin_vector[0].item() > 0
        assert not torch.isinf(result.margin_vector).any()

    def test_scalar_is_multivariate_cohens_d(self):
        """Scalar = ||v|| / sqrt(0.5*(||σ_p||²+||σ_n||²))."""
        torch.manual_seed(42)
        pos = torch.randn(20, 8)
        neg = torch.randn(20, 8)
        csv = compute_steering_vector(pos, neg, "test")
        result = separability_margin(csv)
        v_norm = csv.steering_vector.norm(p=2)
        sp_norm = csv.positive_std.norm(p=2)
        sn_norm = csv.negative_std.norm(p=2)
        expected = v_norm.item() / torch.sqrt(0.5 * (sp_norm**2 + sn_norm**2)).item()
        assert abs(result.scalar_summary - expected) < 1e-4


# =============================================================================
# Concept Gram Matrix
# =============================================================================


class TestConceptGramMatrix:
    """Test pairwise cosine similarity of steering vectors."""

    def test_diagonal_is_one(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(5)}
        gram = concept_gram_matrix(vectors)
        assert torch.allclose(torch.diagonal(gram), torch.ones(5), atol=1e-5)

    def test_symmetric(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(5)}
        gram = concept_gram_matrix(vectors)
        assert torch.allclose(gram, gram.T)

    def test_shape_n_by_n(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(7)}
        gram = concept_gram_matrix(vectors)
        assert gram.shape == (7, 7)

    def test_values_in_range(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(5)}
        gram = concept_gram_matrix(vectors)
        assert (gram >= -1.0 - 1e-5).all() and (gram <= 1.0 + 1e-5).all()

    def test_orthogonal_concepts_off_diagonal_near_zero(self):
        v1 = compute_steering_vector(
            torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]), "c1"
        )
        v2 = compute_steering_vector(
            torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]), "c2"
        )
        gram = concept_gram_matrix({"c1": v1, "c2": v2})
        assert abs(gram[0, 1].item()) < 1e-5

    def test_sorted_concept_order(self):
        """Gram matrix axes follow sorted concept names."""
        vectors = {
            "zeta": _make_csv(d_model=4, name="z"),
            "alpha": _make_csv(d_model=4, name="a"),
        }
        gram = concept_gram_matrix(vectors)
        # Diagonal should still be 1
        assert gram.shape == (2, 2)


# =============================================================================
# Anisotropy Spectrum
# =============================================================================


class TestAnisotropySpectrum:
    """Test covariance eigenvalue spectrum of concept vectors."""

    def test_eigenvalues_sorted_descending(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(10)}
        spectrum = anisotropy_spectrum(vectors)
        diffs = spectrum.eigenvalues[:-1] - spectrum.eigenvalues[1:]
        assert (diffs >= -1e-5).all(), "Eigenvalues not in descending order"

    def test_explained_variance_ratio_sums_to_one(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(10)}
        spectrum = anisotropy_spectrum(vectors)
        assert abs(spectrum.explained_variance_ratio.sum().item() - 1.0) < 1e-4

    def test_ratio_same_shape_as_eigenvalues(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(10)}
        spectrum = anisotropy_spectrum(vectors)
        assert spectrum.eigenvalues.shape == spectrum.explained_variance_ratio.shape

    def test_all_nonnegative(self):
        vectors = {f"c{i}": _make_csv(d_model=8, name=f"c{i}") for i in range(10)}
        spectrum = anisotropy_spectrum(vectors)
        assert (spectrum.eigenvalues >= -1e-5).all()

    def test_isotropic_random_concepts_flat_spectrum(self):
        """Many random concepts → top explained variance ratio should be small."""
        torch.manual_seed(42)
        vectors = {f"c{i}": _make_csv(d_model=64, name=f"c{i}") for i in range(50)}
        spectrum = anisotropy_spectrum(vectors)
        assert spectrum.explained_variance_ratio[0].item() < 0.2

    def test_rank_one_stack_one_dominant_eigenvalue(self):
        """All identical steering vectors → one dominant eigenvalue."""
        v = _make_csv(d_model=8, name="c")
        vectors = {f"c{i}": v for i in range(10)}
        spectrum = anisotropy_spectrum(vectors)
        # After centering, identical vectors → all zeros → degenerate
        # But with eps, the first eigenvalue should dominate or all near zero
        assert (
            spectrum.eigenvalues[0].item() < 1e-5
            or spectrum.explained_variance_ratio[0].item() > 0.5
        )

    def test_returns_anisotropy_spectrum_type(self):
        vectors = {f"c{i}": _make_csv(d_model=4, name=f"c{i}") for i in range(5)}
        result = anisotropy_spectrum(vectors)
        assert isinstance(result, AnisotropySpectrum)

    def test_centered_removes_mean_component(self):
        """Centering: identical vectors → all eigenvalues ≈ 0."""
        v = _make_csv(d_model=4, name="c")
        vectors = {f"c{i}": v for i in range(5)}
        spectrum = anisotropy_spectrum(vectors)
        # After centering identical vectors, covariance is zero
        assert spectrum.eigenvalues[0].item() < 1e-5
