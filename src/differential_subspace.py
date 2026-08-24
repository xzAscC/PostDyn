"""Differential covariance subspaces (PostDyn.tex eqs. 8–11) and five metrics.

Given two groups of last-token hidden states at a fixed layer,

    center each group, form empirical covariances with divisor n,
    ΔΣ = Σ_c − Σ_ref = U Λ Uᵀ,

retain eigenvectors of positive *and* negative eigenvalues separately, and
choose the smallest K_{+}/K_{-} such that the cumulative squared spectral
mass of that sign reaches τ (default 0.95).

Positive eigenvalues: stronger variation in the target domain.
Negative eigenvalues: stronger variation in the reference domain.

Five metrics (PostDyn.tex Metrics paragraph), computed per sign:
  1. Retained subspace dimension K_c^{(t)}
  2. Subspace stability SubSim_c(a, b)
  3. Inter-subspace relation G_ij^{(t)}
  4. Frobenius geometry strength on each sign spectrum
  5. Effective dimensionality d_eff on all classified eigenvalues of that sign

The retained bases (``u_pos``/``u_neg``) remain the public, backward-compatible
K-column bases.  ``u_pos_full`` and ``u_neg_full`` preserve every eigenvector
whose eigenvalue is classified as positive or negative.  Negative-side
eigenvalues are stored as positive magnitudes; ``eigenvalues_signed`` and
``eigenvectors_signed`` preserve the original signed eigensystem explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch


DEFAULT_TAU: float = 0.95
EPS: float = 1e-12


@dataclass(frozen=True)
class DifferentialSubspace:
    """Positive-eigenspace differential subspace for one concept pair.

    Attributes:
        concept: Concept / pair name (e.g. ``math_vs_text``).
        u: Orthonormal basis ``(d, K)`` of retained positive eigenvectors.
        eigenvalues_pos: All positive eigenvalues of ΔΣ, descending ``(r,)``.
        k: Retained dimension (``u.shape[1]``).
        tau: Spectral threshold used to select ``k``.
        n_concept: Number of concept-group examples.
        n_ref: Number of reference-group examples.
        d_model: Hidden dimension.
        tr_concept: ``tr(Σ_c)``.
        tr_ref: ``tr(Σ_ref)``.
        geometry_strength: Normalized geometry strength S̃.
    d_eff: Participation ratio on all classified positive eigenvalues.
    """

    concept: str
    u: torch.Tensor
    eigenvalues_pos: torch.Tensor
    k: int
    tau: float
    n_concept: int
    n_ref: int
    d_model: int
    tr_concept: float
    tr_ref: float
    geometry_strength: float
    d_eff: float
    u_full: Optional[torch.Tensor] = None
    energy: float = 0.0
    frobenius_strength: float = 0.0
    residual_u: Optional[torch.Tensor] = None
    emergence: Optional[float] = None


@dataclass(frozen=True)
class SignedDifferentialSubspace:
    """Target-dominant (+) and reference-dominant (−) differential subspaces.

    ``eigenvalues_neg`` stores *magnitudes* of negative ΔΣ eigenvalues,
    descending, so K / d_eff use the same formulas as the positive side.
    """

    concept: str
    tau: float
    n_concept: int
    n_ref: int
    d_model: int
    tr_concept: float
    tr_ref: float
    u_pos: torch.Tensor
    eigenvalues_pos: torch.Tensor
    k_pos: int
    d_eff_pos: float
    geometry_strength_pos: float
    u_neg: torch.Tensor
    eigenvalues_neg: torch.Tensor
    k_neg: int
    d_eff_neg: float
    geometry_strength_neg: float
    u_pos_full: Optional[torch.Tensor] = None
    u_neg_full: Optional[torch.Tensor] = None
    eigenvalues_signed: Optional[torch.Tensor] = None
    eigenvectors_signed: Optional[torch.Tensor] = None
    energy_pos: float = 0.0
    energy_neg: float = 0.0
    frobenius_strength_pos: float = 0.0
    frobenius_strength_neg: float = 0.0
    r_pos: float = 0.0
    emergence_pos: Optional[float] = None
    emergence_neg: Optional[float] = None

    @property
    def E_pos(self) -> float:
        return self.energy_pos

    @property
    def E_neg(self) -> float:
        return self.energy_neg

    @property
    def R_pos(self) -> float:
        return self.r_pos

    @property
    def eigenvalues_all(self) -> Optional[torch.Tensor]:
        return self.eigenvalues_signed

    @property
    def eigenvectors_all(self) -> Optional[torch.Tensor]:
        return self.eigenvectors_signed

    def to_positive(self) -> DifferentialSubspace:
        """Project onto the legacy positive-only subspace record."""
        return DifferentialSubspace(
            concept=self.concept,
            u=self.u_pos,
            eigenvalues_pos=self.eigenvalues_pos,
            k=self.k_pos,
            tau=self.tau,
            n_concept=self.n_concept,
            n_ref=self.n_ref,
            d_model=self.d_model,
            tr_concept=self.tr_concept,
            tr_ref=self.tr_ref,
            geometry_strength=self.geometry_strength_pos,
            d_eff=self.d_eff_pos,
            u_full=self.u_pos_full,
            energy=self.energy_pos,
            frobenius_strength=self.frobenius_strength_pos,
            residual_u=None,
            emergence=None,
        )

    def to_negative(self) -> DifferentialSubspace:
        """Treat the reference-dominant subspace as a positive-only record."""
        return DifferentialSubspace(
            concept=f"{self.concept}__neg",
            u=self.u_neg,
            eigenvalues_pos=self.eigenvalues_neg,
            k=self.k_neg,
            tau=self.tau,
            n_concept=self.n_concept,
            n_ref=self.n_ref,
            d_model=self.d_model,
            tr_concept=self.tr_concept,
            tr_ref=self.tr_ref,
            geometry_strength=self.geometry_strength_neg,
            d_eff=self.d_eff_neg,
            u_full=self.u_neg_full,
            energy=self.energy_neg,
            frobenius_strength=self.frobenius_strength_neg,
            residual_u=None,
            emergence=None,
        )


def center_rows(h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-center a hidden-state matrix ``H ∈ R^{n×d}``.

    Returns:
        ``(X, mu)`` where ``X = H − 1 μᵀ`` and ``mu ∈ R^d``.
    """
    if h.ndim != 2:
        raise ValueError(f"Expected H of shape (n, d), got {tuple(h.shape)}")
    if h.shape[0] == 0:
        raise ValueError("Cannot center an empty activation matrix")
    mu = h.mean(dim=0)
    return h - mu.unsqueeze(0), mu


def empirical_covariance(h: torch.Tensor) -> torch.Tensor:
    """Empirical covariance with divisor ``n`` (PostDyn.tex).

    Σ = (1/n) Xᵀ X after per-group centering.
    """
    x, _ = center_rows(h)
    n = x.shape[0]
    return (x.T @ x) / float(n)


def select_k_from_positive_spectrum(
    eigenvalues_pos: torch.Tensor,
    tau: float = DEFAULT_TAU,
) -> int:
    """Smallest K with cumulative squared mass ≥ τ over positive eigenvalues."""
    if eigenvalues_pos.numel() == 0:
        return 0
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    sq = eigenvalues_pos.clamp(min=0.0).square()
    total = float(sq.sum().item())
    if total <= 0.0:
        return 0
    csum = torch.cumsum(sq, dim=0)
    thresh = tau * total
    idx = int(torch.searchsorted(csum, torch.tensor(thresh, device=csum.device)).item())
    return min(idx + 1, int(eigenvalues_pos.numel()))


def participation_ratio(eigenvalues: torch.Tensor, k: int, eps: float = EPS) -> float:
    """Effective rank of the first ``k`` (non-negative) eigenvalues.

    ``k`` is retained for the legacy helper API.  Differential metrics call
    this with the full classified spectrum, rather than the retained K.
    """
    if k <= 0 or eigenvalues.numel() == 0:
        return 0.0
    lam_k = eigenvalues[:k].clamp(min=0.0)
    num = float(lam_k.sum().item()) ** 2
    den = float(lam_k.square().sum().item())
    return num / den if den > 0.0 else 0.0


def _basis_from_spectrum(
    evals_signed: torch.Tensor,
    evecs: torch.Tensor,
    d_model: int,
    tau: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float, torch.Tensor]:
    """Retain τ-mass of a descending non-negative spectrum.

    Returns ``(U_retained, eigenvalues, k, d_eff_all, U_full)``.
    """
    k = select_k_from_positive_spectrum(evals_signed, tau=tau)
    u_full = evecs.contiguous()
    u = (
        u_full[:, :k].contiguous()
        if k
        else torch.zeros(d_model, 0, dtype=evecs.dtype, device=evecs.device)
    )
    return (
        u,
        evals_signed.contiguous(),
        k,
        participation_ratio(evals_signed, int(evals_signed.numel()), eps=eps),
        u_full,
    )


def residual_basis(current: SignedDifferentialSubspace) -> torch.Tensor:
    """Return all current directions outside both retained sign bases.

    Neutral eigendirections and non-retained positive/negative directions are
    all included, giving exactly ``d - K_pos - K_neg`` columns.
    """
    if current.eigenvalues_signed is None or current.eigenvectors_signed is None:
        raise ValueError("current subspace does not contain its full eigensystem")
    evals = current.eigenvalues_signed
    retained = torch.zeros(evals.numel(), dtype=torch.bool, device=evals.device)
    pos_indices = torch.nonzero(evals > 0, as_tuple=False).flatten()
    neg_indices = torch.nonzero(evals < 0, as_tuple=False).flatten()
    retained[pos_indices[: current.k_pos]] = True
    if current.k_neg:
        retained[neg_indices[-current.k_neg :]] = True
    return current.eigenvectors_signed[:, ~retained].contiguous()


def residual_to_later_subspace_overlap(
    current: SignedDifferentialSubspace,
    final: SignedDifferentialSubspace | torch.Tensor,
    final_u_neg: Optional[torch.Tensor] = None,
) -> dict[str, dict[str, float | int | bool | None]]:
    """Measure current residual overlap with final positive and negative bases.

    For each sign, observed is ``||RᵀU_final||²_F / K_final``, chance is
    ``d_res / d_model``, and excess is observed minus chance.  A final side
    with ``K_final == 0`` is undefined and returns ``None`` for these values.
    """
    residual = residual_basis(current)
    d_res = int(residual.shape[1])
    d_model = int(current.d_model)
    if isinstance(final, SignedDifferentialSubspace):
        final_bases = (final.u_pos, final.u_neg)
    else:
        if final_u_neg is None:
            raise ValueError(
                "final negative basis is required with an explicit positive basis"
            )
        final_bases = (final, final_u_neg)
    output: dict[str, dict[str, float | int | bool | None]] = {}
    for sign, basis in zip(("pos", "neg"), final_bases):
        k_final = int(basis.shape[1])
        if k_final == 0:
            output[sign] = {
                "defined": False,
                "k_final": 0,
                "d_res": d_res,
                "observed": None,
                "chance": None,
                "excess": None,
            }
            continue
        observed = float((residual.T @ basis).square().sum().item()) / k_final
        chance = d_res / d_model if d_model else 0.0
        output[sign] = {
            "defined": True,
            "k_final": k_final,
            "d_res": d_res,
            "observed": observed,
            "chance": chance,
            "excess": observed - chance,
        }
    return output


compute_chance_adjusted_emergence = residual_to_later_subspace_overlap


def compute_signed_differential_subspace(
    h_concept: torch.Tensor,
    h_ref: torch.Tensor,
    *,
    concept: str = "concept",
    tau: float = DEFAULT_TAU,
    eps: float = EPS,
) -> SignedDifferentialSubspace:
    """Build U₊ and U₋ from ΔΣ = Σ_c − Σ_ref (τ-threshold per sign)."""
    if h_concept.ndim != 2 or h_ref.ndim != 2:
        raise ValueError("Both activation matrices must be rank-2 (n, d)")
    if h_concept.shape[1] != h_ref.shape[1]:
        raise ValueError(
            f"d_model mismatch: concept {h_concept.shape[1]} vs ref {h_ref.shape[1]}"
        )
    if h_concept.shape[0] < 2 or h_ref.shape[0] < 2:
        raise ValueError("Need at least 2 examples in each group")

    h_c = h_concept.detach().float().cpu()
    h_r = h_ref.detach().float().cpu()
    d_model = int(h_c.shape[1])

    sigma_c = empirical_covariance(h_c)
    sigma_r = empirical_covariance(h_r)
    delta = sigma_c - sigma_r

    evals, evecs = torch.linalg.eigh(delta)
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])

    pos_mask = evals > 0
    neg_mask = evals < 0
    evals_pos = evals[pos_mask]
    evecs_pos = evecs[:, pos_mask]
    evals_neg = torch.flip(-evals[neg_mask], dims=[0])
    evecs_neg = torch.flip(evecs[:, neg_mask], dims=[1])

    u_pos, evals_pos, k_pos, d_eff_pos, u_pos_full = _basis_from_spectrum(
        evals_pos, evecs_pos, d_model, tau, eps
    )
    u_neg, evals_neg, k_neg, d_eff_neg, u_neg_full = _basis_from_spectrum(
        evals_neg, evecs_neg, d_model, tau, eps
    )

    tr_c = float(torch.trace(sigma_c).item())
    tr_r = float(torch.trace(sigma_r).item())
    energy_pos = float(evals_pos.square().sum().item()) if evals_pos.numel() else 0.0
    energy_neg = float(evals_neg.square().sum().item()) if evals_neg.numel() else 0.0
    frob_pos = energy_pos**0.5
    frob_neg = energy_neg**0.5
    frob_total = float(delta.square().sum().sqrt().item())
    geometry_strength_pos = frob_pos / frob_total if frob_total > 0.0 else 0.0
    geometry_strength_neg = frob_neg / frob_total if frob_total > 0.0 else 0.0
    energy_total = energy_pos + energy_neg
    r_pos = energy_pos / energy_total if energy_total > 0.0 else 0.0

    return SignedDifferentialSubspace(
        concept=concept,
        tau=float(tau),
        n_concept=int(h_c.shape[0]),
        n_ref=int(h_r.shape[0]),
        d_model=d_model,
        tr_concept=tr_c,
        tr_ref=tr_r,
        u_pos=u_pos,
        eigenvalues_pos=evals_pos,
        k_pos=k_pos,
        d_eff_pos=d_eff_pos,
        geometry_strength_pos=geometry_strength_pos,
        u_neg=u_neg,
        eigenvalues_neg=evals_neg,
        k_neg=k_neg,
        d_eff_neg=d_eff_neg,
        geometry_strength_neg=geometry_strength_neg,
        u_pos_full=u_pos_full,
        u_neg_full=u_neg_full,
        eigenvalues_signed=evals.contiguous(),
        eigenvectors_signed=evecs.contiguous(),
        energy_pos=energy_pos,
        energy_neg=energy_neg,
        frobenius_strength_pos=frob_pos,
        frobenius_strength_neg=frob_neg,
        r_pos=r_pos,
        emergence_pos=None,
        emergence_neg=None,
    )


def compute_differential_subspace(
    h_concept: torch.Tensor,
    h_ref: torch.Tensor,
    *,
    concept: str = "concept",
    tau: float = DEFAULT_TAU,
    eps: float = EPS,
) -> DifferentialSubspace:
    """Build U_c from ΔΣ = Σ_c − Σ_ref (positive eigenspace, τ-threshold).

    Args:
        h_concept: Concept-group activations ``(n_c, d)``.
        h_ref: Reference-group activations ``(n_ref, d)``.
        concept: Name stored on the result.
        tau: Cumulative squared-eigenvalue threshold (default 0.95).
        eps: Numerical floor for positivity / denominators.

    Returns:
        :class:`DifferentialSubspace` with basis ``U ∈ R^{d×K}`` and metrics
        that depend only on this single pair/checkpoint/layer.
    """
    return compute_signed_differential_subspace(
        h_concept,
        h_ref,
        concept=concept,
        tau=tau,
        eps=eps,
    ).to_positive()


def subspace_stability(
    u_a: torch.Tensor,
    u_b: torch.Tensor,
    *,
    k: Optional[int] = None,
) -> float:
    """SubSim between two orthonormal bases (PostDyn.tex).

    Uses the full retained bases and ``K = min(K_a, K_b)``, then

        SubSim = (1/K) || U_aᵀ U_b ||_F²
    """
    if u_a.ndim != 2 or u_b.ndim != 2:
        raise ValueError("U must be rank-2 (d, K)")
    if u_a.shape[0] != u_b.shape[0]:
        raise ValueError("U bases must share the same ambient dimension")
    ka, kb = int(u_a.shape[1]), int(u_b.shape[1])
    k_use = min(ka, kb)
    if k_use <= 0:
        return 0.0
    gram = u_a.float().T @ u_b.float()
    value = float((gram.square().sum() / float(k_use)).item())
    return max(0.0, min(1.0, value))


def inter_subspace_relation(
    u_i: torch.Tensor,
    u_j: torch.Tensor,
    *,
    k: Optional[int] = None,
) -> float:
    """Inter-subspace relation G_ij (same formula as SubSim on two concepts)."""
    return subspace_stability(u_i, u_j, k=k)


def compute_pair_metrics_at_checkpoint(
    subspaces: dict[str, DifferentialSubspace],
) -> dict[str, Any]:
    """Per-checkpoint metrics that need only the subspaces at that checkpoint.

    Returns retained K, geometry strength, d_eff for each concept, and the
    pairwise inter-subspace relation matrix among concepts present.
    """
    names = sorted(subspaces.keys())
    per_concept: dict[str, dict[str, float | int]] = {}
    for name in names:
        s = subspaces[name]
        per_concept[name] = {
            "k": s.k,
            "geometry_strength": s.geometry_strength,
            "d_eff": s.d_eff,
            "n_pos_eigs": int(s.eigenvalues_pos.numel()),
            "tr_concept": s.tr_concept,
            "tr_ref": s.tr_ref,
        }

    g: dict[str, dict[str, float]] = {a: {} for a in names}
    for i, a in enumerate(names):
        for b in names[i:]:
            val = inter_subspace_relation(subspaces[a].u, subspaces[b].u)
            g[a][b] = val
            g[b][a] = val

    return {
        "per_concept": per_concept,
        "inter_subspace_relation": g,
    }


def compute_stability_trajectory(
    subspaces_by_checkpoint: dict[str, dict[str, DifferentialSubspace]],
    checkpoint_order: list[str],
    *,
    reference_checkpoint: Optional[str] = None,
) -> dict[str, Any]:
    """Subspace stability along a checkpoint trajectory.

    For each concept, reports:
      - pairwise SubSim between consecutive checkpoints
      - SubSim of each checkpoint vs a fixed reference (default: first ckpt)
    """
    if not checkpoint_order:
        return {
            "pairwise": {},
            "consecutive": {},
            "vs_reference": {},
            "reference": None,
        }

    ref_name = (
        reference_checkpoint
        if reference_checkpoint is not None
        else checkpoint_order[0]
    )
    if ref_name not in checkpoint_order:
        raise ValueError(
            f"Reference checkpoint {ref_name!r} is not in checkpoint_order"
        )
    concepts = sorted(
        {
            c
            for ck in checkpoint_order
            if ck in subspaces_by_checkpoint
            for c in subspaces_by_checkpoint[ck]
        }
    )

    consecutive: dict[str, list[dict[str, Any]]] = {c: [] for c in concepts}
    vs_ref: dict[str, list[dict[str, Any]]] = {c: [] for c in concepts}
    pairwise: dict[str, list[dict[str, Any]]] = {c: [] for c in concepts}

    for concept in concepts:
        ref_u = None
        if ref_name in subspaces_by_checkpoint:
            ref_u = subspaces_by_checkpoint[ref_name].get(concept)

        prev_ck: Optional[str] = None
        prev_u: Optional[torch.Tensor] = None
        for ck in checkpoint_order:
            block = subspaces_by_checkpoint.get(ck, {})
            sub = block.get(concept)
            if sub is None:
                continue
            if prev_ck is not None and prev_u is not None:
                consecutive[concept].append(
                    {
                        "a": prev_ck,
                        "b": ck,
                        "subsim": subspace_stability(prev_u, sub.u),
                        "k": min(prev_u.shape[1], sub.u.shape[1]),
                    }
                )
            if ref_u is not None:
                vs_ref[concept].append(
                    {
                        "checkpoint": ck,
                        "reference": ref_name,
                        "subsim": subspace_stability(ref_u.u, sub.u),
                        "k": min(ref_u.u.shape[1], sub.u.shape[1]),
                    }
                )
            prev_ck = ck
            prev_u = sub.u

        for first_idx, first_ck in enumerate(checkpoint_order):
            first = subspaces_by_checkpoint.get(first_ck, {}).get(concept)
            if first is None:
                continue
            for second_ck in checkpoint_order[first_idx + 1 :]:
                second = subspaces_by_checkpoint.get(second_ck, {}).get(concept)
                if second is None:
                    continue
                pairwise[concept].append(
                    {
                        "a": first_ck,
                        "b": second_ck,
                        "subsim": subspace_stability(first.u, second.u),
                        "k": min(first.u.shape[1], second.u.shape[1]),
                    }
                )

    return {
        "pairwise": pairwise,
        "reference": ref_name,
        "consecutive": consecutive,
        "vs_reference": vs_ref,
    }


def subspace_to_serializable(sub: DifferentialSubspace) -> dict[str, Any]:
    """JSON-friendly metadata (tensors excluded)."""
    meta = asdict(sub)
    for key in ("u", "u_full", "eigenvalues_pos", "residual_u"):
        meta.pop(key, None)
    meta["eigenvalues_pos"] = [float(x) for x in sub.eigenvalues_pos.tolist()]
    meta["u_full_shape"] = (
        list(sub.u_full.shape) if sub.u_full is not None else [sub.d_model, 0]
    )
    return meta


def signed_subspace_to_serializable(sub: SignedDifferentialSubspace) -> dict[str, Any]:
    """JSON-friendly metadata for both eigenvalue signs (tensors excluded)."""
    return {
        "concept": sub.concept,
        "tau": sub.tau,
        "n_concept": sub.n_concept,
        "n_ref": sub.n_ref,
        "d_model": sub.d_model,
        "tr_concept": sub.tr_concept,
        "tr_ref": sub.tr_ref,
        "k_pos": sub.k_pos,
        "k_neg": sub.k_neg,
        "d_eff_pos": sub.d_eff_pos,
        "d_eff_neg": sub.d_eff_neg,
        "geometry_strength_pos": sub.geometry_strength_pos,
        "geometry_strength_neg": sub.geometry_strength_neg,
        "eigenvalues_pos": [float(x) for x in sub.eigenvalues_pos.tolist()],
        "eigenvalues_neg": [float(x) for x in sub.eigenvalues_neg.tolist()],
        "u_pos_shape": list(sub.u_pos.shape),
        "u_neg_shape": list(sub.u_neg.shape),
        "u_pos_full_shape": list(sub.u_pos_full.shape)
        if sub.u_pos_full is not None
        else [sub.d_model, 0],
        "u_neg_full_shape": list(sub.u_neg_full.shape)
        if sub.u_neg_full is not None
        else [sub.d_model, 0],
        "eigenvalues_signed": [float(x) for x in sub.eigenvalues_signed.tolist()]
        if sub.eigenvalues_signed is not None
        else [],
    }


def signed_subspace_stability(
    a: SignedDifferentialSubspace,
    b: SignedDifferentialSubspace,
) -> dict[str, Any]:
    """SubSim on the + and − retained bases between two models/checkpoints."""
    return {
        "pos": {
            "subsim": subspace_stability(a.u_pos, b.u_pos),
            "k": min(a.k_pos, b.k_pos),
        },
        "neg": {
            "subsim": subspace_stability(a.u_neg, b.u_neg),
            "k": min(a.k_neg, b.k_neg),
        },
    }
