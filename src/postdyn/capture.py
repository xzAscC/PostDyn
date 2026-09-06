"""Selective transformer hidden-state capture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def _resolved_blocks(model: Any) -> list[Any]:
    for parent_name, layers_name in (("transformer", "h"), ("model", "layers")):
        parent = getattr(model, parent_name, None)
        layers = getattr(parent, layers_name, None)
        if layers is not None:
            return list(layers)
    layers = getattr(model, "layers", None)
    if layers is not None:
        return list(layers)
    raise AttributeError("model has no transformer.h or model.layers blocks")


def _tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError("hook output must contain a hidden-states tensor")
    return value


class _HiddenCapture:
    def __init__(self, model: Any, layers: Sequence[int]) -> None:
        self.tensors: dict[int, torch.Tensor] = {}
        self._layers = list(layers)
        blocks = _resolved_blocks(model)
        invalid = [layer for layer in self._layers if layer < 0 or layer >= len(blocks)]
        if invalid:
            state_count = len(blocks) + 1
            raise ValueError(
                f"Requested layer(s) {invalid} outside model layer range "
                f"[0, {state_count - 1})"
            )
        if self._layers and len(blocks) > 0:
            final_norm: Any = None
            if len(blocks) - 1 in self._layers:
                transformer = getattr(model, "transformer", None)
                final_norm = getattr(transformer, "ln_f", None)
                if final_norm is None:
                    model_parent = getattr(model, "model", None)
                    final_norm = getattr(model_parent, "norm", None)
                if final_norm is None:
                    final_norm = blocks[-1]
            self._targets = {
                layer: final_norm if layer == len(blocks) - 1 else blocks[layer]
                for layer in self._layers
            }
        else:
            self._targets = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "_HiddenCapture":
        for layer, module in self._targets.items():
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: (
                        self.tensors.__setitem__(layer, _tensor(output))
                    )
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def hidden_capture(model: Any, layers: Sequence[int]) -> _HiddenCapture:
    """Capture selected block outputs, including the post-final-norm state."""
    return _HiddenCapture(model, layers)


__all__ = ["hidden_capture"]
