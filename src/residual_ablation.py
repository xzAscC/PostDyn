"""Residual-stream subspace ablation for decoder-model generation.

The hook is attached to ``model.model.layers[layer]`` and removes the
projection of the first residual-stream output tensor onto a column basis
``U``.  A basis is selected explicitly: target/positive and reference/negative
bases are never combined implicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import TracebackType
from typing import cast

import torch
from torch import nn

__all__ = [
    "ResidualStreamAblation",
    "ResidualSubspaceAblation",
    "register_residual_ablation",
    "register_residual_stream_ablation",
    "residual_stream_ablation",
    "residual_subspace_ablation",
]


def _resolve_layer_index(
    layer: int | None,
    layer_idx: int | None,
) -> int:
    if layer is not None and layer_idx is not None and layer != layer_idx:
        raise ValueError("layer and layer_idx must refer to the same layer")
    selected = layer if layer is not None else layer_idx
    if selected is None:
        raise TypeError("a layer index is required")
    if type(selected) is not int:
        raise TypeError("layer must be an integer")
    if selected < 0:
        raise ValueError("layer must be non-negative")
    return selected


def _get_layer(model: object, layer_idx: int) -> nn.Module:
    try:
        layers = getattr(getattr(model, "model"), "layers")
        block = cast(Sequence[object], layers)[layer_idx]
    except AttributeError as exc:
        raise AttributeError(
            "model must expose transformer blocks at model.model.layers"
        ) from exc
    except IndexError as exc:
        raise IndexError(f"layer index {layer_idx} is out of range") from exc
    except TypeError as exc:
        raise TypeError("model.model.layers must support integer indexing") from exc

    if not isinstance(block, nn.Module):
        raise TypeError("model.model.layers[layer] must be a torch.nn.Module")

    return block


def _select_basis(
    basis: object | None,
    U: object | None,
    U_target: object | None,
    U_reference: object | None,
    U_pos: object | None,
    U_neg: object | None,
) -> tuple[torch.Tensor, str]:
    candidates = [
        ("basis", basis),
        ("U", U),
        ("U_target", U_target),
        ("U_reference", U_reference),
        ("U_pos", U_pos),
        ("U_neg", U_neg),
    ]
    selected = [(name, value) for name, value in candidates if value is not None]
    if len(selected) != 1:
        raise ValueError(
            "provide exactly one basis; positive and negative bases are not combined"
        )

    name, value = selected[0]
    try:
        tensor = torch.as_tensor(value)
    except Exception as exc:
        raise TypeError(f"{name} must be convertible to a torch.Tensor") from exc

    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must have shape (d_model, K), got {tuple(tensor.shape)}"
        )
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")

    if tensor.shape[1] != 0:
        validation_basis = tensor.detach().to(device="cpu", dtype=torch.float64)
        gram = validation_basis.transpose(0, 1) @ validation_basis
        identity = torch.eye(
            validation_basis.shape[1], dtype=torch.float64, device="cpu"
        )
        if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
            error = (gram - identity).abs().max().item()
            raise ValueError(
                f"{name} columns must be orthonormal; maximum Gram error is {error:.3g}"
            )

    return tensor.detach(), name


def _projection_dtype(hidden_states: torch.Tensor, basis: torch.Tensor) -> torch.dtype:
    dtype = torch.promote_types(hidden_states.dtype, basis.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class ResidualStreamAblation:
    """A removable post-block residual-stream projection hook.

    ``U`` is a ``(d_model, K)`` matrix with orthonormal columns.  Each hook
    invocation applies ``x - (x @ U) @ U.T`` to every leading position of the
    block output, including every token in a prefill sequence.
    """

    def __init__(
        self,
        model: object,
        layer: int | None = None,
        basis: object | None = None,
        *,
        layer_idx: int | None = None,
        U: object | None = None,
        U_target: object | None = None,
        U_reference: object | None = None,
        U_pos: object | None = None,
        U_neg: object | None = None,
    ) -> None:
        selected_layer = _resolve_layer_index(layer, layer_idx)
        self._module: nn.Module = _get_layer(model, selected_layer)
        self._basis, self.basis_name = _select_basis(
            basis, U, U_target, U_reference, U_pos, U_neg
        )
        self._basis: torch.Tensor
        self.basis_name: str
        self._basis_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self.layer_idx: int = selected_layer
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def basis(self) -> torch.Tensor:
        return self._basis

    @property
    def is_registered(self) -> bool:
        return self._handle is not None

    def register(self) -> "ResidualStreamAblation":
        if self._handle is None:
            self._handle = self._module.register_forward_hook(self._forward_hook)
        return self

    def remove(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.remove()

    def __enter__(self) -> "ResidualStreamAblation":
        return self.register()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.remove()
        return False

    def _ablate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim == 0:
            raise ValueError(
                "residual-stream output must have a final hidden dimension"
            )
        if hidden_states.shape[-1] != self._basis.shape[0]:
            raise ValueError(
                f"residual-stream width does not match basis: {hidden_states.shape[-1]} != {self._basis.shape[0]}"
            )
        if self._basis.shape[1] == 0:
            return hidden_states
        if not torch.is_floating_point(hidden_states):
            raise TypeError("residual-stream output must have a floating-point dtype")

        dtype = _projection_dtype(hidden_states, self._basis)
        work = hidden_states.to(dtype=dtype)
        cache_key = (hidden_states.device, dtype)
        basis = self._basis_cache.get(cache_key)
        if basis is None:
            basis = self._basis.to(device=hidden_states.device, dtype=dtype)
            self._basis_cache[cache_key] = basis
        return (work - (work @ basis) @ basis.transpose(0, 1)).to(
            dtype=hidden_states.dtype
        )

    def _forward_hook(
        self, module: nn.Module, inputs: object, output: object
    ) -> object:
        del module, inputs
        if torch.is_tensor(output):
            return self._ablate(output)
        if isinstance(output, tuple):
            if not output:
                raise TypeError("block output tuple/list must contain residual states")
            if not torch.is_tensor(output[0]):
                raise TypeError("the first block output must be residual states")
            typed_output = cast(tuple[object, ...], output)
            values: list[object] = list(typed_output)
            values[0] = self._ablate(cast(torch.Tensor, values[0]))
            if type(output) is tuple:
                return tuple(values)
            constructor = cast(Callable[..., object], type(output))
            try:
                rebuilt = constructor(*values)
            except TypeError:
                try:
                    rebuilt = constructor(values)
                except TypeError as exc:
                    raise TypeError(
                        "tuple output type could not be reconstructed"
                    ) from exc
            if not isinstance(rebuilt, tuple):
                raise TypeError("tuple output type could not be reconstructed")
            return cast(tuple[object, ...], rebuilt)
        if isinstance(output, list):
            if not output:
                raise TypeError("block output tuple/list must contain residual states")
            if not torch.is_tensor(output[0]):
                raise TypeError("the first block output must be residual states")
            typed_output = cast(list[object], output)
            values = list(typed_output)
            values[0] = self._ablate(cast(torch.Tensor, values[0]))
            if type(output) is list:
                return values
            constructor = cast(Callable[..., object], type(output))
            try:
                rebuilt = constructor(values)
            except TypeError as exc:
                raise TypeError("list output type could not be reconstructed") from exc
            if not isinstance(rebuilt, list):
                raise TypeError("list output type could not be reconstructed")
            return cast(list[object], rebuilt)
        raise TypeError("block output must be a tensor, tuple, or list")


ResidualSubspaceAblation = ResidualStreamAblation


def register_residual_ablation(
    model: object,
    layer: int | None = None,
    basis: object | None = None,
    *,
    layer_idx: int | None = None,
    U: object | None = None,
    U_target: object | None = None,
    U_reference: object | None = None,
    U_pos: object | None = None,
    U_neg: object | None = None,
) -> ResidualStreamAblation:
    ablation = ResidualStreamAblation(
        model,
        layer,
        basis,
        layer_idx=layer_idx,
        U=U,
        U_target=U_target,
        U_reference=U_reference,
        U_pos=U_pos,
        U_neg=U_neg,
    )
    return ablation.register()


register_residual_stream_ablation = register_residual_ablation


@contextmanager
def residual_stream_ablation(
    model: object,
    layer: int | None = None,
    basis: object | None = None,
    *,
    layer_idx: int | None = None,
    U: object | None = None,
    U_target: object | None = None,
    U_reference: object | None = None,
    U_pos: object | None = None,
    U_neg: object | None = None,
) -> Iterator[ResidualStreamAblation]:
    ablation = ResidualStreamAblation(
        model,
        layer,
        basis,
        layer_idx=layer_idx,
        U=U,
        U_target=U_target,
        U_reference=U_reference,
        U_pos=U_pos,
        U_neg=U_neg,
    )
    try:
        _ = ablation.register()
        yield ablation
    finally:
        ablation.remove()


residual_subspace_ablation = residual_stream_ablation
