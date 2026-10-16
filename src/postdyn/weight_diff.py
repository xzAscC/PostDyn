"""Per-tensor statistics of the weight update between two checkpoints.

For one tensor, W0 is its start value, W1 its end value and dW = W1 - W0.
Metrics (rows of a matrix are output neurons):
  sparsity       1 - #{|dW_ij| > 1e-5} / numel           (Mukherjee et al., 2025)
  rank_ratio     numerical rank of dW / min(m, n)
  one_minus_cos  1 - mean over rows of cos(W0[i], W1[i])  (whole vector for 1-D tensors)
  gini           Gini index of |dW|
  erank_ratio    exp(entropy of sigma(dW) / sum sigma) / min(m, n)  (Roy & Vetterli, 2007)
  overlap        ||U_k^T dW V_k||_F^2 / ||dW||_F^2, U_k, V_k top-10% singular vectors of W0
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from postdyn.persistence import append_jsonl

TOL = 1e-5
OVERLAP_FRAC = 0.1
_LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
_EPS32 = torch.finfo(torch.float32).eps
T = TypeVar("T")


def tensor_files(ckpt_dir: str | os.PathLike[str]) -> dict[str, Path]:
    """Map each tensor name to the safetensors file that stores it."""
    root = Path(ckpt_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {name: root / file for name, file in weight_map.items()}
    files: dict[str, Path] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handle:
            files.update(dict.fromkeys(handle.keys(), path))
    return files


def parse_name(name: str) -> tuple[int | None, str]:
    """Split a decoder-layer tensor name into (layer index, component)."""
    match = _LAYER.fullmatch(name)
    return (int(match.group(1)), match.group(2)) if match else (None, name)


def select_uniform(items: Sequence[T], n: int = 10) -> list[T]:
    """Pick n evenly spaced items including both ends; all items if there are at most n."""
    if len(items) <= n:
        return list(items)
    last = len(items) - 1
    return [items[int(i * last / (n - 1) + 0.5)] for i in range(n)]


def gini(x: torch.Tensor, chunk: int = 2**26) -> float | None:
    """Gini index of |x|: 0 when all entries are equal, near 1 when one entry holds everything."""
    c = x.detach().abs().flatten().float().sort().values
    total = c.double().sum()
    if total == 0:
        return None
    n = c.numel()
    # sum_k c_(k) * (n - k + 1/2) / n, accumulated in chunks to bound memory
    weighted = torch.zeros((), dtype=torch.float64, device=c.device)
    for lo in range(0, n, chunk):
        part = c[lo : lo + chunk].double()
        k = torch.arange(lo + 1, lo + 1 + part.numel(), dtype=torch.float64, device=c.device)
        weighted += (part * (n - k + 0.5)).sum()
    return float(1 - 2 * weighted / (total * n))


def row_one_minus_cos(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> float:
    """Mean over rows of 1 - cos(a_i, b_i), computed as ||a_i/|a_i| - b_i/|b_i|||^2 / 2 in float64.

    Fine-tuned rows have 1 - cos around 1e-6 or smaller, below the float32 resolution of cos itself.
    """
    total = 0.0
    for lo in range(0, a.shape[0], chunk):
        x, y = a[lo : lo + chunk].double(), b[lo : lo + chunk].double()
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-300)
        y = y / y.norm(dim=1, keepdim=True).clamp_min(1e-300)
        total += float((x - y).square().sum(dim=1).sum() / 2)
    return total / a.shape[0]


def bf16_ulp(w: torch.Tensor) -> torch.Tensor:
    """Spacing between neighbouring bfloat16 values at each entry (0 for zeros)."""
    _, exponent = torch.frexp(w.float())  # |w| in [2^(e-1), 2^e)
    ulp = torch.ldexp(torch.ones_like(w, dtype=torch.float32), exponent - 8)
    return torch.where(w == 0, torch.zeros_like(ulp), ulp)


def angle_stats(start: torch.Tensor, end: torch.Tensor, chunk: int = 4096) -> dict[str, float | int | None]:
    """Rotation angle (degrees) of each row from start to end: mean and std over rows.

    theta_i = 2 asin(d_i / 2), d_i = || a_i/|a_i| - b_i/|b_i| ||, exact and stable for tiny angles.
    theta_ulp = mean over rows of ||ulp(a_i)|| / |a_i| in degrees: the rotation of a long row whose
    entries each move by one bfloat16 step in random directions (the storage-precision floor).
    A 1-D tensor counts as one row.
    """
    a_all = start.reshape(1, -1) if start.ndim == 1 else start
    b_all = end.reshape(1, -1) if end.ndim == 1 else end
    thetas, refs = [], []
    for lo in range(0, a_all.shape[0], chunk):
        a, b = a_all[lo : lo + chunk].double(), b_all[lo : lo + chunk].double()
        na, nb = a.norm(dim=1, keepdim=True).clamp_min(1e-300), b.norm(dim=1, keepdim=True).clamp_min(1e-300)
        d = (a / na - b / nb).norm(dim=1)
        thetas.append(torch.rad2deg(2 * torch.asin((d / 2).clamp(max=1.0))))
        refs.append(torch.rad2deg(bf16_ulp(a_all[lo : lo + chunk]).double().norm(dim=1) / na.squeeze(1)))
    theta, ref = torch.cat(thetas), torch.cat(refs)
    return {
        "rows": theta.numel(),
        "theta_mean": float(theta.mean()),
        "theta_std": float(theta.std(correction=0)) if theta.numel() > 1 else None,
        "theta_ulp": float(ref.mean()),
    }


def effective_rank(sv: torch.Tensor) -> float:
    """exp of the entropy of singular values normalized to sum 1."""
    p = sv.double() / sv.double().sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def _gram(a: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Gram matrix of the smaller side in float64 and whether it is a @ a.T (rows side)."""
    a = a.double()
    return (a @ a.T, True) if a.shape[0] <= a.shape[1] else (a.T @ a, False)


def singular_values(m: torch.Tensor) -> torch.Tensor:
    """Singular values in descending order, from the eigenvalues of the smaller Gram matrix.

    Much faster on GPU than an SVD of a 5120 x 27648 matrix; float64 keeps sigma accurate down to
    about 1e-8 * sigma_1, far below the rank tolerance used here.
    """
    gram, _ = _gram(m)
    return torch.linalg.eigvalsh(gram).flip(0).clamp_min(0).sqrt()


def _top_singular(a: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All singular values (descending) and the top-k left/right singular vectors, from one Gram eigh."""
    a = a.double()
    gram, rows_side = _gram(a)
    evals, evecs = torch.linalg.eigh(gram)
    sv = evals.flip(0).clamp_min(0).sqrt()
    top = evecs[:, -k:].flip(1)
    sigma = sv[:k].clamp_min(1e-300)
    if rows_side:  # gram = W W^T: eigenvectors are left singular vectors
        return sv, top, (a.T @ top) / sigma
    return sv, (a @ top) / sigma, top  # gram = W^T W: eigenvectors are right singular vectors


def principal_basis(w0: torch.Tensor, frac: float = OVERLAP_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k left and right singular vectors of W0, k = round(frac * min(m, n)) (at least 1)."""
    k = max(1, round(frac * min(w0.shape)))
    _, u, v = _top_singular(w0, k)
    return u.float().contiguous(), v.float().contiguous()


def principal_angles(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Principal angles (degrees, ascending) between the column spans of orthonormal a and b.

    Uses sines: the singular values of b - a a^T b are sin(theta_i). Unlike arccos of the singular
    values of a^T b, this keeps tiny angles accurate.
    """
    a, b = a.double(), b.double()
    sines = torch.linalg.svdvals(b - a @ (a.T @ b)).clamp(max=1.0)
    return sorted(float(x) for x in torch.rad2deg(torch.asin(sines)))


def _rounded(values: Any) -> list[float]:
    return [float(f"{float(x):.5g}") for x in values]


def spectral_stats(
    start: torch.Tensor,
    end: torch.Tensor,
    basis0: tuple[torch.Tensor, torch.Tensor] | None,
    sv0: torch.Tensor | None,
    keep_curves: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Update norm, top-k principal angles and spectrum drift of one tensor (Zhu et al., 2025).

    rel_update_norm = ||W1 - W0||_F / ||W0||_F. For matrices, with U0, V0 the top-k singular vectors of
    W0 (basis0) and U1, V1 those of W1: principal angles between span(U0) and span(U1) (pa_u) and
    between span(V0) and span(V1) (pa_v), and nss = ||sigma(W1) - sigma(W0)||_2 / ||sigma(W0)||_2 over
    the full spectrum. keep_curves stores every angle and the top-k singular values of W0 and W1.
    """
    a0, a1 = start.to(device, torch.float64), end.to(device, torch.float64)
    w0_norm, dw_norm = float(a0.norm()), float((a1 - a0).norm())
    stats: dict[str, Any] = {"dw_fro": dw_norm, "w0_fro": w0_norm,
                             "rel_update_norm": dw_norm / w0_norm if w0_norm > 0 else None}
    if a0.ndim != 2 or basis0 is None or sv0 is None:
        return stats
    u0, v0 = (x.to(device, torch.float64) for x in basis0)
    k = u0.shape[1]
    sv1, u1, v1 = _top_singular(a1, k)
    s0 = sv0.to(device, torch.float64)
    pa_u, pa_v = principal_angles(u0, u1), principal_angles(v0, v1)
    stats.update(k=k, pa_u_max=max(pa_u), pa_u_mean=sum(pa_u) / k, pa_v_max=max(pa_v), pa_v_mean=sum(pa_v) / k,
                 nss=float((sv1 - s0).norm() / s0.norm()))
    if keep_curves:
        stats.update(pa_u=_rounded(pa_u), pa_v=_rounded(pa_v), sv0_top=_rounded(s0[:k]), sv1_top=_rounded(sv1[:k]))
    return stats


class BasisCache:
    """Store the principal basis of each start tensor so later checkpoints reuse it."""

    def __init__(self, root: str | os.PathLike[str] | None) -> None:
        self.root = Path(root) if root is not None else None

    def get(self, name: str, w0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.root is None:
            return principal_basis(w0)
        path = self.root / f"{name}.safetensors"
        if path.is_file():
            saved = load_file(str(path), device=str(w0.device))
            return saved["u"], saved["v"]
        u, v = principal_basis(w0)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        save_file({"u": u.cpu(), "v": v.cpu()}, str(tmp))
        os.replace(tmp, path)  # atomic: concurrent jobs never read a partial file
        return u, v

    def spectrum(self, name: str, w0: torch.Tensor) -> torch.Tensor:
        """All singular values of W0 (float64), cached beside the basis."""
        if self.root is None:
            return singular_values(w0)
        path = self.root / f"{name}.sv.safetensors"
        if path.is_file():
            return load_file(str(path), device=str(w0.device))["s"]
        s = singular_values(w0)
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        save_file({"s": s.cpu()}, str(tmp))
        os.replace(tmp, path)
        return s


def update_stats(
    start: torch.Tensor,
    end: torch.Tensor,
    basis: tuple[torch.Tensor, torch.Tensor] | None = None,
    tol: float = TOL,
    device: str = "cpu",
) -> dict[str, float | int | bool | None]:
    """All agreed metrics for one tensor; matrix-only metrics are skipped for vectors."""
    w0, w1 = start.to(device, torch.float32), end.to(device, torch.float32)
    dw = w1 - w0
    changed = int((dw.abs() > tol).sum())
    zero = not bool(dw.any())
    rows0, rows1 = (w0, w1) if w0.ndim == 2 else (w0.reshape(1, -1), w1.reshape(1, -1))
    stats: dict[str, float | int | bool | None] = {
        "numel": dw.numel(),
        "changed_tol": changed,
        "sparsity": 1 - changed / dw.numel(),
        "one_minus_cos": row_one_minus_cos(rows0, rows1),
        "gini": gini(dw),
        "zero_update": zero,
    }
    if dw.ndim != 2:
        return stats
    max_rank = min(dw.shape)
    sv = singular_values(dw)
    # PyTorch's default matrix_rank tolerance for float32 input
    rank = 0 if zero else int((sv > sv[0] * max(dw.shape) * _EPS32).sum())
    erank = None if zero else effective_rank(sv)
    stats.update(
        rank=rank,
        max_rank=max_rank,
        rank_ratio=rank / max_rank,
        erank=erank,
        erank_ratio=None if erank is None else erank / max_rank,
    )
    if basis is not None:
        u, v = (t.to(device) for t in basis)
        stats["k"] = u.shape[1]
        stats["overlap"] = None if zero else float((u.T @ dw @ v).square().sum() / dw.square().sum())
    return stats


def _done_names(out_path: Path) -> set[str]:
    if not out_path.is_file():
        return set()
    data = out_path.read_bytes()
    if data and not data.endswith(b"\n"):  # a killed job left half a line: drop it
        with out_path.open("r+b") as handle:
            handle.truncate(data.rfind(b"\n") + 1)
    return {json.loads(line)["name"] for line in out_path.read_text().splitlines() if line.strip()}


def compare(
    start_dir: str | os.PathLike[str],
    end_dir: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    basis_dir: str | os.PathLike[str] | None = None,
    device: str = "cpu",
    log: Callable[[str], None] = print,
    mode: str = "metrics",
    keep_curves: bool = False,
) -> None:
    """Append one JSON row per tensor to ``out_path``, skipping rows already written.

    mode: "metrics" (update_stats), "angles" (angle_stats: per-row rotation) or "spectral"
    (spectral_stats: update norm, top-k principal angles, spectrum drift).
    """
    out = Path(out_path)
    start_files, end_files = tensor_files(start_dir), tensor_files(end_dir)
    if set(start_files) != set(end_files):
        diff = sorted(set(start_files) ^ set(end_files))[:5]
        raise ValueError(f"tensor names differ between checkpoints, e.g. {diff}")
    cache = BasisCache(basis_dir)
    done = _done_names(out)
    todo = [name for name in sorted(start_files) if name not in done]
    log(f"{len(todo)} tensors to compare ({len(done)} already done) -> {out}")
    handles: dict[Path, object] = {}

    def load(files: dict[str, Path], name: str) -> torch.Tensor:
        path = files[name]
        if path not in handles:
            handles[path] = safe_open(str(path), "pt", device=device)
        return handles[path].get_tensor(name)  # type: ignore[attr-defined]

    for i, name in enumerate(todo):
        start, end = load(start_files, name), load(end_files, name)
        if start.shape != end.shape:
            raise ValueError(f"{name}: shape {tuple(start.shape)} != {tuple(end.shape)}")
        layer, component = parse_name(name)
        row: dict[str, object] = {"name": name, "layer": layer, "component": component, "shape": list(start.shape)}
        if mode == "angles":
            row.update(angle_stats(start.to(device), end.to(device)))
        elif mode == "spectral":
            matrix = start.ndim == 2
            w0 = start.to(device, torch.float32)
            row.update(spectral_stats(start, end, cache.get(name, w0) if matrix else None,
                                      cache.spectrum(name, w0) if matrix else None, keep_curves, device))
        else:
            basis = cache.get(name, start.to(device, torch.float32)) if start.ndim == 2 else None
            row.update(update_stats(start, end, basis=basis, device=device))
        for key in ("sparsity", "one_minus_cos", "gini", "erank", "overlap", "theta_mean", "theta_std",
                    "rel_update_norm", "pa_u_max", "pa_v_max", "nss"):
            value = row.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name}: non-finite {key}={value}")
        append_jsonl(out, row)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            key = {"angles": "theta_mean", "spectral": "rel_update_norm"}.get(mode, "sparsity")
            log(f"  {i + 1}/{len(todo)} {name} {key}={row[key]:.4g}")
