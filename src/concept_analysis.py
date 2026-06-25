"""
Concept Analysis Module

Four model-agnostic metrics for analyzing training dynamics of concepts
in activation space. All operate on ConceptSteeringVector data:

1. directional_stability — cosine similarity of concept vectors across checkpoints
2. separability_margin   — per-dimension and scalar Cohen's d
3. concept_gram_matrix   — pairwise cosine similarity of concept vectors
4. anisotropy_spectrum   — covariance eigenvalue spectrum of concept vectors

References:
    TrainingDynamic.tex: "Training Dynamics of Concepts in Activation Space"
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.concept_steering import ConceptSteeringVector


# =============================================================================
# Result Data Structures
# =============================================================================


@dataclass
class SeparabilityMargin:
    """Per-dimension and scalar Cohen's d for a single concept.

    Attributes:
        concept_name: Name of the concept
        margin_vector: Per-dimension Cohen's d, shape (d_model,)
        scalar_summary: Multivariate Cohen's d = ||v|| / pooled ||sigma||
    """

    concept_name: str
    margin_vector: torch.Tensor
    scalar_summary: float


@dataclass
class AnisotropySpectrum:
    """Covariance eigenvalue spectrum of a set of concept vectors.

    Attributes:
        eigenvalues: Sorted descending, shape (r,) where r = min(n-1, d)
        explained_variance_ratio: eigenvalues / sum(eigenvalues), sums to 1.0
    """

    eigenvalues: torch.Tensor
    explained_variance_ratio: torch.Tensor


# =============================================================================
# Helper
# =============================================================================


def _cosine_matrix(rows: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Pairwise cosine similarity of rows in a (n, d) tensor.

    Returns (n, n) symmetric matrix with diagonal ≈ 1.0.
    """
    norms = rows.norm(dim=1, keepdim=True)
    normalized = rows / norms.clamp(min=eps)
    return normalized @ normalized.T


# =============================================================================
# Metric 1: Directional Stability (trajectory analysis)
# =============================================================================


def directional_stability(
    trajectory: dict[str, dict[str, ConceptSteeringVector]],
    eps: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Compute per-concept cosine stability across checkpoint pairs.

        stability(k; t, t') = cos(v_k^{(t)}, v_k^{(t')})

    Args:
        trajectory: {checkpoint_id: {concept_name: ConceptSteeringVector}}
        eps: Numerical stability constant

    Returns:
        {concept_name: (T, T) cosine matrix} where T = len(checkpoints).
        Axes follow sorted checkpoint names.
    """
    checkpoints = sorted(trajectory.keys())

    all_concepts: set[str] = set()
    for ckpt_data in trajectory.values():
        all_concepts.update(ckpt_data.keys())

    results: dict[str, torch.Tensor] = {}
    for concept in all_concepts:
        vectors = []
        for ckpt in checkpoints:
            if concept in trajectory[ckpt]:
                vectors.append(trajectory[ckpt][concept].steering_vector)
            else:
                raise KeyError(f"Concept '{concept}' missing from checkpoint '{ckpt}'")
        stacked = torch.stack(vectors)
        results[concept] = _cosine_matrix(stacked, eps=eps)

    return results


# =============================================================================
# Metric 2: Separability Margin (Cohen's d)
# =============================================================================


def separability_margin(
    vec: ConceptSteeringVector,
    eps: float = 1e-10,
) -> SeparabilityMargin:
    """Compute per-dimension and scalar Cohen's d.

        margin_vector[i] = v[i] / sqrt(0.5 * (sigma_p[i]^2 + sigma_n[i]^2))
        scalar_summary  = ||v||_2 / sqrt(0.5 * (||sigma_p||^2 + ||sigma_n||^2))

    The numerator is the steering vector v = mu_p - mu_n.

    Args:
        vec: A ConceptSteeringVector with means and stds
        eps: Guard against zero-variance dimensions

    Returns:
        SeparabilityMargin with per-dimension vector and scalar summary.
    """
    v = vec.steering_vector
    pooled_var = 0.5 * (vec.positive_std.pow(2) + vec.negative_std.pow(2))
    pooled_std = torch.sqrt(pooled_var.clamp(min=eps**2))

    margin_vector = v / pooled_std

    v_norm = v.norm(p=2)
    sigma_p_norm_sq = vec.positive_std.pow(2).sum()
    sigma_n_norm_sq = vec.negative_std.pow(2).sum()
    scalar_pooled = torch.sqrt(
        (0.5 * (sigma_p_norm_sq + sigma_n_norm_sq)).clamp(min=eps**2)
    )
    scalar_summary = (v_norm / scalar_pooled).item()

    return SeparabilityMargin(
        concept_name=vec.concept_name,
        margin_vector=margin_vector,
        scalar_summary=scalar_summary,
    )


# =============================================================================
# Metric 3: Concept Gram Matrix
# =============================================================================


def concept_gram_matrix(
    vectors: dict[str, ConceptSteeringVector],
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute pairwise cosine similarity of concept steering vectors.

        G_{ij} = cos(v_i, v_j)

    Args:
        vectors: {concept_name: ConceptSteeringVector} for one checkpoint
        eps: Numerical stability constant

    Returns:
        (n_concepts, n_concepts) symmetric matrix; diagonal ≈ 1.0.
        Axes follow sorted concept names.
    """
    names = sorted(vectors.keys())
    stacked = torch.stack([vectors[n].steering_vector for n in names])
    return _cosine_matrix(stacked, eps=eps)


# =============================================================================
# Metric 4: Anisotropy Spectrum
# =============================================================================


def anisotropy_spectrum(
    vectors: dict[str, ConceptSteeringVector],
    eps: float = 1e-10,
) -> AnisotropySpectrum:
    """Compute covariance eigenvalue spectrum of concept vectors.

    Stacks steering vectors into V (n x d), centers rows, forms
    covariance Cov = V_c^T V_c / (n-1), and computes eigenvalues.

    Args:
        vectors: {concept_name: ConceptSteeringVector} for one checkpoint
        eps: Guard for near-zero eigenvalues

    Returns:
        AnisotropySpectrum with eigenvalues (descending) and ratios.
    """
    n = len(vectors)
    stacked = torch.stack([v.steering_vector for v in vectors.values()])

    centered = stacked - stacked.mean(dim=0, keepdim=True)

    if n > 1:
        cov = (centered.T @ centered) / (n - 1)
    else:
        cov = centered.T @ centered

    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.flip(eigvals, dims=[0])
    eigvals = eigvals.clamp(min=0)

    total = eigvals.sum()
    if total.item() < eps:
        ratios = torch.ones_like(eigvals) / len(eigvals)
    else:
        ratios = eigvals / total

    return AnisotropySpectrum(eigenvalues=eigvals, explained_variance_ratio=ratios)
