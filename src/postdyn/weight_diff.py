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
from typing import TypeVar

import torch
import torch.nn.functional as F
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


def effective_rank(sv: torch.Tensor) -> float:
    """exp of the entropy of singular values normalized to sum 1."""
    p = sv.double() / sv.double().sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def principal_basis(w0: torch.Tensor, frac: float = OVERLAP_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k left and right singular vectors of W0, k = round(frac * min(m, n)) (at least 1)."""
    u, _, vh = torch.linalg.svd(w0.float(), full_matrices=False)
    k = max(1, round(frac * min(w0.shape)))
    return u[:, :k].contiguous(), vh[:k].T.contiguous()


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
        "one_minus_cos": float(1 - F.cosine_similarity(rows0, rows1, dim=1).double().mean()),
        "gini": gini(dw),
        "zero_update": zero,
    }
    if dw.ndim != 2:
        return stats
    max_rank = min(dw.shape)
    sv = torch.linalg.svdvals(dw)
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
    return {json.loads(line)["name"] for line in out_path.read_text().splitlines() if line.strip()}


def compare(
    start_dir: str | os.PathLike[str],
    end_dir: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    basis_dir: str | os.PathLike[str] | None = None,
    device: str = "cpu",
    log: Callable[[str], None] = print,
) -> None:
    """Append one JSON row per tensor to ``out_path``, skipping rows already written."""
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
        basis = cache.get(name, start.to(device, torch.float32)) if start.ndim == 2 else None
        layer, component = parse_name(name)
        row: dict[str, object] = {"name": name, "layer": layer, "component": component, "shape": list(start.shape)}
        row.update(update_stats(start, end, basis=basis, device=device))
        for key in ("sparsity", "one_minus_cos", "gini", "erank", "overlap"):
            value = row.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name}: non-finite {key}={value}")
        append_jsonl(out, row)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            log(f"  {i + 1}/{len(todo)} {name} sparsity={row['sparsity']:.4f}")
