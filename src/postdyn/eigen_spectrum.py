"""Eigenvalue-extreme summaries for domain covariances and their difference.

Companion to the differential-subspace pipeline: instead of retained bases it
reports the spectral *extremes* that the PostDyn analysis needs per layer —

* ``lambda_min`` of each domain covariance Σ_math / Σ_wiki (bulk lower edge,
  computed in float64 for a clean smallest eigenvalue), and
* the signed extremes of ΔΣ = Σ_concept − Σ_ref, i.e. the largest positive
  eigenvalue (math-dominant direction) and the smallest eigenvalue (largest
  negative magnitude, reference-dominant direction).

All computations use the same empirical covariance (divisor ``n`` after
per-group centering) as :mod:`postdyn.differential_subspace`.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from postdyn.differential_subspace import empirical_covariance

DEFAULT_TAIL_K: int = 10


def _prepare(h: torch.Tensor) -> torch.Tensor:
    if h.ndim != 2:
        raise ValueError(f"Expected activations of shape (n, d), got {tuple(h.shape)}")
    if h.shape[0] < 2:
        raise ValueError("Need at least 2 rows to estimate a covariance")
    return h.detach().double().cpu()


def _ascending_extremes(evals: torch.Tensor, k: int) -> dict[str, list[float]]:
    """Tail lists of an ascending eigenvalue vector.

    ``smallest`` keeps ascending order; ``largest`` is reversed to descending.
    """
    k = max(1, int(k))
    smallest = evals[: min(k, evals.numel())]
    largest = torch.flip(evals[-min(k, evals.numel()) :], dims=[0])
    return {
        "smallest": [float(x) for x in smallest.tolist()],
        "largest": [float(x) for x in largest.tolist()],
    }


def _covariance_metrics_from_evals(
    evals: torch.Tensor, sigma: torch.Tensor, n: int, k: int
) -> dict[str, Any]:
    tails = _ascending_extremes(evals, k)
    return {
        "n": int(n),
        "d_model": int(sigma.shape[0]),
        "lambda_min": float(evals[0]),
        "lambda_max": float(evals[-1]),
        "smallest": tails["smallest"],
        "largest": tails["largest"],
        "trace": float(torch.diagonal(sigma).sum().item()),
    }


def covariance_extremes(h: torch.Tensor, *, k: int = DEFAULT_TAIL_K) -> dict[str, Any]:
    """Spectral extremes of the empirical covariance of ``h`` ``(n, d)``."""
    x = _prepare(h)
    sigma = empirical_covariance(x)
    evals = torch.linalg.eigvalsh(sigma)
    return _covariance_metrics_from_evals(evals, sigma, x.shape[0], k)


def _difference_metrics_from_evals(
    evals: torch.Tensor, sigma_c: torch.Tensor, sigma_r: torch.Tensor, k: int
) -> dict[str, Any]:
    max_abs = float(evals.abs().max().item()) if evals.numel() else 0.0
    tol = 1e-6 * max(1.0, max_abs)
    pos = evals[evals > tol]
    neg = evals[evals < -tol]
    top_pos = (
        torch.flip(pos[-min(k, pos.numel()) :], dims=[0])
        if pos.numel()
        else torch.empty(0, dtype=evals.dtype)
    )
    bottom_neg = (
        neg[: min(k, neg.numel())] if neg.numel() else torch.empty(0, dtype=evals.dtype)
    )
    return {
        "lambda_max_pos": float(evals[-1]),
        "lambda_min_neg": float(evals[0]),
        "n_pos": int(pos.numel()),
        "n_neg": int(neg.numel()),
        "top_pos": [float(x) for x in top_pos.tolist()],
        "bottom_neg": [float(x) for x in bottom_neg.tolist()],
        "tr_concept": float(torch.diagonal(sigma_c).sum().item()),
        "tr_ref": float(torch.diagonal(sigma_r).sum().item()),
    }


def difference_extremes(
    h_concept: torch.Tensor, h_ref: torch.Tensor, *, k: int = DEFAULT_TAIL_K
) -> dict[str, Any]:
    """Signed spectral extremes of ΔΣ = Σ_concept − Σ_ref.

    ``lambda_max_pos`` is the largest eigenvalue (max positive, math-dominant);
    ``lambda_min_neg`` is the smallest eigenvalue (largest negative magnitude,
    reference-dominant, kept signed).
    """
    if h_concept.shape[1] != h_ref.shape[1]:
        raise ValueError(
            f"d_model mismatch: concept {h_concept.shape[1]} vs ref {h_ref.shape[1]}"
        )
    x_c = _prepare(h_concept)
    x_r = _prepare(h_ref)
    sigma_c = empirical_covariance(x_c)
    sigma_r = empirical_covariance(x_r)
    evals = torch.linalg.eigvalsh(sigma_c - sigma_r)
    return _difference_metrics_from_evals(evals, sigma_c, sigma_r, k)


def build_layer_metrics(
    h_concept: torch.Tensor, h_ref: torch.Tensor, *, k: int = DEFAULT_TAIL_K
) -> dict[str, Any]:
    """Per-layer extremes for both covariances and their difference.

    Covariances are formed once and reused so ΔΣ shares their numerics.
    """
    if h_concept.shape[1] != h_ref.shape[1]:
        raise ValueError(
            f"d_model mismatch: concept {h_concept.shape[1]} vs ref {h_ref.shape[1]}"
        )
    x_c = _prepare(h_concept)
    x_r = _prepare(h_ref)
    sigma_c = empirical_covariance(x_c)
    sigma_r = empirical_covariance(x_r)
    evals_c = torch.linalg.eigvalsh(sigma_c)
    evals_r = torch.linalg.eigvalsh(sigma_r)
    evals_d = torch.linalg.eigvalsh(sigma_c - sigma_r)
    return {
        "concept": _covariance_metrics_from_evals(evals_c, sigma_c, x_c.shape[0], k),
        "reference": _covariance_metrics_from_evals(evals_r, sigma_r, x_r.shape[0], k),
        "difference": _difference_metrics_from_evals(evals_d, sigma_c, sigma_r, k),
    }


def atomic_write_json(path, obj: Any) -> None:
    """Write JSON atomically (tmp file + rename), creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


_PLOT_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("lambda_min_concept", r"$\lambda_{\min}(\Sigma_{\mathrm{math}})$", True),
    ("lambda_min_reference", r"$\lambda_{\min}(\Sigma_{\mathrm{wiki}})$", True),
    ("lambda_max_pos", r"$\lambda_{\max}(\Delta\Sigma)$", False),
    ("lambda_min_neg", r"$\lambda_{\min}(\Delta\Sigma)$", False),
)


def plot_layer_lines(rows: list[dict[str, Any]], out_pdf) -> Any:
    """2×2 PDF of the four extreme eigenvalues vs layer, one line per model.

    Returns the output path, or ``None`` when ``rows`` is empty.
    """
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", font_scale=1.1)
    except ImportError:  # pragma: no cover - seaborn is a project dependency
        pass

    models = sorted({str(row["model"]) for row in rows})
    by_model: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in models}
    for row in rows:
        by_model[str(row["model"])][int(row["layer"])] = row

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (field, title, log_y) in zip(axes.flat, _PLOT_PANELS):
        for model in models:
            layers = sorted(by_model[model])
            values = [by_model[model][layer][field] for layer in layers]
            ax.plot(layers, values, marker="o", label=model.replace("_", " "))
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Eigenvalue extremes: math vs wikitext (per checkpoint)", fontsize=14)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_pdf
