"""
Memory-efficient model loading via safetensors streaming.

Never instantiates AutoModelForCausalLM. Reads weight tensors one-by-one
from safetensors shard files. All tensors kept in original dtype (bfloat16);
SVD computation converts to float32 per-matrix.
"""

from __future__ import annotations

import gc
import os
from typing import Generator, Optional

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

from src.config import ModelConfig, categorize_weight, get_layer_index


def _is_linear_weight(name: str) -> bool:
    if any(skip in name.lower() for skip in ("norm", "embed", "bias", "layernorm", "rmsnorm")):
        return False
    return True


def iter_weight_tensors(
    model_config: ModelConfig,
    cache_dir: Optional[str] = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """
    Stream 2D linear-layer weight tensors from safetensors shards.
    Yields (param_name, tensor) one at a time in original dtype.
    Caller processes and discards each before the next is yielded.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    print(f"  Streaming {model_config.hf_id} (rev={model_config.revision})...")

    model_path = snapshot_download(
        model_config.hf_id,
        revision=model_config.revision,
        cache_dir=cache_dir,
    )

    sf_files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))

    if not sf_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    count = 0
    for sf_file in sf_files:
        sf_path = os.path.join(model_path, sf_file)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not _is_linear_weight(key):
                    continue
                tensor = f.get_tensor(key)
                if tensor.dim() != 2:
                    continue
                yield key, tensor
                count += 1

    print(f"  Streamed {count} linear weight matrices from {len(sf_files)} shards")


def collect_all_weight_names(
    model_config: ModelConfig,
    cache_dir: Optional[str] = None,
) -> list[str]:
    """Collect only the parameter names (shapes) without loading tensor data."""
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    model_path = snapshot_download(
        model_config.hf_id,
        revision=model_config.revision,
        cache_dir=cache_dir,
    )

    sf_files = sorted(f for f in os.listdir(model_path) if f.endswith(".safetensors"))
    names = []
    for sf_file in sf_files:
        sf_path = os.path.join(model_path, sf_file)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _is_linear_weight(key):
                    names.append(key)
    return names


def group_names_by_type(
    names: list[str],
    architecture: str,
) -> dict[str, list[str]]:
    """Group weight names by their category (no tensors held)."""
    groups: dict[str, list[str]] = {
        "attention_qkv": [],
        "attention_output": [],
        "mlp": [],
        "other": [],
    }
    for name in names:
        cat = categorize_weight(name, architecture)
        groups.setdefault(cat, []).append(name)
    return groups


def group_names_by_layer(
    names: list[str],
) -> dict[int, list[str]]:
    """Group weight names by layer index (no tensors held)."""
    layers: dict[int, list[str]] = {}
    for name in names:
        idx = get_layer_index(name)
        if idx is None:
            idx = -1
        layers.setdefault(idx, []).append(name)
    return layers
