"""Residual-stream subspace interventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def project_out(
    h: Tensor,
    U: Tensor,
    alpha: float,
    mode: str = "dimensionless",
    r_bar: float | None = None,
) -> Tensor:
    """Remove a scaled projection of ``h`` onto the columns of ``U``."""
    if mode not in {"dimensionless", "norm"}:
        raise ValueError(f"unknown projection mode: {mode}")
    if h.shape[-1] != U.shape[0]:
        raise ValueError("hidden states and projection basis have incompatible widths")
    work_dtype = torch.promote_types(h.dtype, U.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    work = h.to(work_dtype)
    basis = U.to(device=h.device, dtype=work_dtype)
    projection = (work @ basis) @ basis.transpose(-1, -2)
    if mode == "dimensionless":
        result = work - alpha * projection
    else:
        if r_bar is None:
            raise ValueError("r_bar is required for norm projection")
        norm = torch.linalg.vector_norm(projection, dim=-1, keepdim=True)
        eps = torch.finfo(work.dtype).eps
        if bool((norm <= eps).any()):
            raise ValueError("projection norm is degenerate")
        result = work - (alpha * r_bar / norm) * projection
    return result.to(dtype=h.dtype)


def matched_random_basis(d: int, k: int, seed: int) -> Tensor:
    """Return a deterministic fp32 orthonormal Gaussian basis."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    gaussian = torch.randn((d, k), generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(gaussian, mode="reduced")
    return q[:, :k].contiguous()


def _layer(model: Any, index: int) -> Any:
    for parent_name, layers_name in (("transformer", "h"), ("model", "layers")):
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None)
        if layers is not None:
            try:
                return layers[index]
            except IndexError as exc:
                raise IndexError(f"layer {index} is out of range") from exc
    raise AttributeError("model has no transformer.h or model.layers blocks")


def procrustes_align(U_source: Tensor, U_target: Tensor) -> Tensor:
    """Orthogonal map R* = A B^T from the SVD of U_source^T U_target.

    Minimizes ||U_target - U_source R||_F over orthogonal R (slide formula:
    U_R^T U_S = A Sigma B^T gives R* = A B^T).
    """
    if U_source.shape != U_target.shape:
        raise ValueError("alignment bases must share shape (d, k)")
    cross = (U_source.T.to(torch.float64)) @ U_target.to(torch.float64)
    A, _, Bt = torch.linalg.svd(cross)
    return (A @ Bt).to(dtype=U_source.dtype)


def replace_basis(
    h: Tensor,
    U_from: Tensor,
    U_to: Tensor,
    alpha: float = 1.0,
) -> Tensor:
    """Swap the ``U_from``-spanned component for its ``U_to`` re-expression.

    h' = h + alpha * (U_to - U_from) @ (U_from^T h); at alpha = 1 this is
    h - U_from U_from^T h + U_to U_from^T h (the slide's boxed replacement).
    """
    if U_from.shape != U_to.shape:
        raise ValueError("replacement bases must share shape (d, k)")
    work_dtype = torch.promote_types(h.dtype, U_from.dtype)
    if work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32
    work = h.to(work_dtype)
    source = U_from.to(device=h.device, dtype=work_dtype)
    target = U_to.to(device=h.device, dtype=work_dtype)
    coefficients = source.T @ work
    return work + alpha * ((target - source) @ coefficients)


def register_replacement_hook(
    model: Any,
    layer: int,
    U_from: Tensor,
    U_to: Tensor,
    alpha: float,
) -> torch.utils.hooks.RemovableHandle:
    """Register a removable post-block residual replacement."""

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            if not output or not torch.is_tensor(output[0]):
                raise TypeError("block output tuple must start with hidden states")
            return (
                replace_basis(
                    output[0],
                    U_from.to(output[0].device),
                    U_to.to(output[0].device),
                    alpha,
                ),
                *output[1:],
            )
        if torch.is_tensor(output):
            return replace_basis(
                output, U_from.to(output.device), U_to.to(output.device), alpha
            )
        raise TypeError("block output must be a tensor or tuple")

    return _layer(model, layer).register_forward_hook(hook)


def register_ablation_hook(
    model: Any,
    layer: int,
    U: Tensor,
    alpha: float,
    mode: str,
    r_bar: float | None = None,
) -> torch.utils.hooks.RemovableHandle:
    """Register a removable post-block residual rewrite."""

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if isinstance(output, tuple):
            if not output or not torch.is_tensor(output[0]):
                raise TypeError("block output tuple must start with hidden states")
            return (
                project_out(output[0], U.to(output[0].device), alpha, mode, r_bar),
                *output[1:],
            )
        if torch.is_tensor(output):
            return project_out(output, U.to(output.device), alpha, mode, r_bar)
        raise TypeError("block output must be a tensor or tuple")

    return _layer(model, layer).register_forward_hook(hook)


def mean_hidden_norm(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layer: int,
    batch_size: int = 8,
) -> float:
    """Measure mean final-token norm at transformer block ``layer`` output."""
    old_padding = getattr(tokenizer, "padding_side", None)
    old_pad_token = getattr(tokenizer, "pad_token", None)
    tokenizer.padding_side = "left"
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    values: list[Tensor] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            encoded = tokenizer(batch, return_tensors="pt", padding=True)
            device = (
                next(model.parameters()).device
                if hasattr(model, "parameters")
                else None
            )
            if device is not None:
                encoded = {key: value.to(device) for key, value in encoded.items()}
            if "attention_mask" in encoded:
                mask = encoded["attention_mask"]
                encoded["position_ids"] = (
                    mask.to(dtype=torch.long).cumsum(dim=1) - 1
                ).masked_fill(mask == 0, 0)
            with torch.no_grad():
                try:
                    result = model(**encoded, output_hidden_states=True)
                except TypeError:
                    encoded.pop("position_ids", None)
                    result = model(**encoded, output_hidden_states=True)
            hidden = result.hidden_states[layer + 1]
            mask = encoded.get("attention_mask")
            if mask is not None:
                positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
                positions = torch.where(mask.to(dtype=torch.bool), positions, -1).amax(
                    dim=1
                )
            else:
                positions = torch.full(
                    (hidden.shape[0],), hidden.shape[1] - 1, device=hidden.device
                )
            values.append(
                hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
                .norm(dim=-1)
                .cpu()
            )
    finally:
        if old_padding is not None:
            tokenizer.padding_side = old_padding
        if hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = old_pad_token
    return float(torch.cat(values).mean().item()) if values else 0.0
