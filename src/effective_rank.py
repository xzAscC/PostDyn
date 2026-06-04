"""
Effective Rank Computation Module

Implements entropy-based effective rank (Roy & Vetterli, 2007) and related metrics
for analyzing the spectral properties of neural network weight matrices.

Core formula:
    erank(W) = exp(-Σ p_i * ln(p_i))  where  p_i = σ_i / Σσ_j
    ratio    = erank(W) / min(m, n) ∈ (0, 1]

Also computes:
    - Normalized SVD entropy (Moonlight paper style): H_norm = -(1/log n) Σ q_i * log(q_i) where q_i = σ_i² / Σσ_j²
    - Stable rank: ||W||²_F / ||W||²_2
    - α-ReQ: power-law decay rate of eigenspectrum
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

_GPU_DEVICE = None


def get_device() -> torch.device:
    global _GPU_DEVICE
    if _GPU_DEVICE is None:
        _GPU_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _GPU_DEVICE


@dataclass
class RankMetrics:
    """Container for all effective rank metrics of a single weight matrix."""
    shape: tuple[int, int]
    max_rank: int
    effective_rank: float
    effective_rank_ratio: float
    svd_entropy: float  # Normalized SVD entropy [0, 1]
    stable_rank: float
    top_singular_value: float
    condition_number: float
    frobenius_norm: float


def _gpu_svdvals(W: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    GPU-accelerated singular value computation.
    Moves matrix to GPU as float32, runs torch.linalg.svdvals,
    returns singular values on CPU.
    Falls back to numpy CPU if CUDA OOM.
    """
    device = get_device()
    if device.type == "cuda":
        try:
            W_gpu = W.detach().to(device=device, dtype=torch.float32)
            sigma = torch.linalg.svdvals(W_gpu)
            result = sigma.cpu()
            del W_gpu, sigma
            torch.cuda.empty_cache()
            return result[result > eps] if (result > eps).any() else torch.tensor([eps])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

    W_np = W.detach().cpu().to(torch.float32).numpy()
    sigma = np.linalg.svd(W_np, compute_uv=False)
    del W_np
    sigma_t = torch.from_numpy(sigma)
    mask = sigma_t > eps
    return sigma_t[mask] if mask.any() else torch.tensor([eps])


compute_singular_values = _gpu_svdvals


def effective_rank(W: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute entropy-based effective rank (Roy & Vetterli, 2007).
    
    erank(W) = exp(-Σ p_i * ln(p_i))  where p_i = σ_i / Σσ_j
    
    Properties:
        - erank ∈ [1, min(m,n)]
        - erank = 1 when only one nonzero singular value
        - erank = min(m,n) when all singular values are equal
        - Scale invariant: erank(cW) = erank(W)
    
    Args:
        W: Weight matrix of shape (m, n)
        eps: Small value for numerical stability
        
    Returns:
        Effective rank as float
    """
    sigma = compute_singular_values(W, eps)
    
    # Normalize to probability distribution
    p = sigma / sigma.sum()
    
    # Shannon entropy: H = -Σ p_i * ln(p_i)
    # Clamp p to avoid log(0)
    log_p = torch.log(p.clamp(min=eps))
    entropy = -torch.sum(p * log_p)
    
    # Effective rank = exp(entropy)
    return torch.exp(entropy).item()


def effective_rank_ratio(W: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute normalized effective rank ratio ∈ (0, 1].
    
    ratio = erank(W) / min(m, n)
    
    This allows comparison across layers of different dimensions.
    """
    max_rank = min(W.shape)
    er = effective_rank(W, eps)
    return er / max_rank


def svd_entropy_normalized(W: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute normalized SVD entropy as used in the Moonlight paper.
    
    H_norm = -(1/log n) * Σ q_i * log(q_i)  where q_i = σ_i² / Σσ_j²
    
    Returns value in [0, 1] where:
        - 0 = all energy in one singular value (rank-1 like)
        - 1 = all singular values equal (isotropic)
    """
    sigma = compute_singular_values(W, eps)
    n = len(sigma)
    
    if n <= 1:
        return 0.0
    
    # Energy normalization: q_i = σ_i² / Σσ_j²
    sigma_sq = sigma ** 2
    q = sigma_sq / sigma_sq.sum()
    
    # Normalized entropy
    log_q = torch.log(q.clamp(min=eps))
    entropy = -torch.sum(q * log_q)
    
    # Normalize by log(n)
    return (entropy / math.log(n)).item()


def stable_rank(W: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute stable rank: ||W||²_F / ||W||²_2 = (Σ σ_i²) / σ_1²
    
    This is a simpler alternative to entropy-based effective rank.
    Range: [1, min(m,n)]
    """
    sigma = compute_singular_values(W, eps)
    return ((sigma ** 2).sum() / (sigma[0] ** 2)).item()


def alpha_req(W: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Estimate the power-law decay rate α of the singular value spectrum.
    
    Fits log(σ_i) ≈ -α * log(i) + const
    
    Higher α → faster decay → more anisotropic (lower effective rank)
    Lower α → slower decay → more isotropic (higher effective rank)
    """
    sigma = compute_singular_values(W, eps)
    n = len(sigma)
    if n < 3:
        return 0.0
    
    # Log-log regression: ln(σ_i) = -α * ln(i) + c
    indices = torch.arange(1, n + 1, dtype=torch.float64)
    log_sigma = torch.log(sigma.clamp(min=eps))
    log_indices = torch.log(indices)
    
    # Simple linear regression
    x = log_indices
    y = log_sigma
    x_mean = x.mean()
    y_mean = y.mean()
    
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sum((x - x_mean) ** 2)
    
    if abs(denominator.item()) < eps:
        return 0.0
    
    # Slope is negative, α is the absolute value
    alpha = -(numerator / denominator).item()
    return max(0.0, alpha)


def compute_all_metrics(W: torch.Tensor, eps: float = 1e-10) -> RankMetrics:
    """
    Compute all effective rank metrics for a single weight matrix.
    
    Returns a RankMetrics dataclass with all computed values.
    """
    sigma = compute_singular_values(W, eps)
    m, n = W.shape
    max_rank = min(m, n)
    
    # Effective rank (entropy-based)
    p = sigma / sigma.sum()
    log_p = torch.log(p.clamp(min=eps))
    entropy = -torch.sum(p * log_p)
    erank = torch.exp(entropy).item()
    
    # Normalized SVD entropy (Moonlight style)
    sigma_sq = sigma ** 2
    q = sigma_sq / sigma_sq.sum()
    n_sigma = len(sigma)
    if n_sigma <= 1:
        svd_ent = 0.0  # Single singular value = zero entropy
    else:
        log_q = torch.log(q.clamp(min=eps))
        q_entropy = -torch.sum(q * log_q)
        svd_ent = (q_entropy / math.log(n_sigma)).item()
    
    # Stable rank: ||W||²_F / ||W||²_2 = (Σ σ_i²) / σ_1²
    s_rank = (sigma_sq.sum() / sigma_sq[0]).item()
    
    # Condition number
    cond = (sigma[0] / sigma[-1]).item()
    
    # Frobenius norm
    fro_norm = torch.norm(W.float(), p='fro').item()
    
    return RankMetrics(
        shape=(m, n),
        max_rank=max_rank,
        effective_rank=erank,
        effective_rank_ratio=erank / max_rank,
        svd_entropy=svd_ent,
        stable_rank=s_rank,
        top_singular_value=sigma[0].item(),
        condition_number=cond,
        frobenius_norm=fro_norm,
    )


def batch_effective_rank(
    weights: dict[str, torch.Tensor],
    eps: float = 1e-10,
    max_dim: Optional[int] = None,
) -> dict[str, RankMetrics]:
    """
    Compute effective rank metrics for a dictionary of weight matrices.
    
    Args:
        weights: Dict mapping layer names to 2D weight tensors
        eps: Numerical stability constant
        max_dim: If set, skip matrices where min(m,n) > max_dim (memory optimization)
        
    Returns:
        Dict mapping layer names to RankMetrics
    """
    results = {}
    for name, W in weights.items():
        if W.dim() != 2:
            continue
        m, n = W.shape
        if max_dim is not None and min(m, n) > max_dim:
            print(f"Skipping {name}: dim {min(m,n)} > max_dim {max_dim}")
            continue
        try:
            results[name] = compute_all_metrics(W, eps)
        except Exception as e:
            print(f"Error computing metrics for {name}: {e}")
    return results


def batch_effective_rank_numpy(
    weights: dict[str, np.ndarray],
    eps: float = 1e-10,
) -> dict[str, dict]:
    """
    NumPy-based batch effective rank computation (for environments without GPU).
    
    Args:
        weights: Dict mapping layer names to 2D numpy arrays
        eps: Numerical stability constant
        
    Returns:
        Dict mapping layer names to metric dicts
    """
    results = {}
    for name, W in weights.items():
        if W.ndim != 2:
            continue
        m, n = W.shape
        max_rank = min(m, n)
        
        # Compute SVD
        try:
            sigma = np.linalg.svd(W.astype(np.float64), compute_uv=False)
            # Filter near-zero
            sigma = sigma[sigma > eps]
            if len(sigma) == 0:
                continue
        except Exception as e:
            print(f"SVD failed for {name}: {e}")
            continue
        
        # Effective rank
        p = sigma / sigma.sum()
        entropy = -np.sum(p * np.log(np.maximum(p, eps)))
        erank = np.exp(entropy)
        
        # SVD entropy (normalized)
        sigma_sq = sigma ** 2
        q = sigma_sq / sigma_sq.sum()
        q_entropy = -np.sum(q * np.log(np.maximum(q, eps)))
        svd_ent = q_entropy / np.log(len(sigma))
        
        # Stable rank
        s_rank = np.sum(sigma_sq) / sigma_sq[0]
        
        results[name] = {
            'shape': (m, n),
            'max_rank': max_rank,
            'effective_rank': float(erank),
            'effective_rank_ratio': float(erank / max_rank),
            'svd_entropy': float(svd_ent),
            'stable_rank': float(s_rank),
            'top_singular_value': float(sigma[0]),
            'frobenius_norm': float(np.linalg.norm(W)),
        }
    
    return results
