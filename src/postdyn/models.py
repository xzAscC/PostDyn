"""Model loading and small model-inspection helpers."""

from __future__ import annotations

import os
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

_config = import_module("postdyn.config")
CheckpointRef = _config.CheckpointRef
MODEL_FAMILIES = _config.MODEL_FAMILIES


_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_NF4_RESERVE_BYTES = 1 * 1024**3
_NF4_MINIMUM_BUDGET_BYTES = 20 * 1024**3


def pretrained_config(repo: str, revision: str) -> Any:
    """Fetch a checkpoint configuration without loading its weights."""

    return AutoConfig.from_pretrained(repo, revision=revision)


def _family_for_repo(repo: str) -> Any:
    for family in MODEL_FAMILIES.values():
        if repo in {
            family.base_repo,
            family.sft_repo,
            family.dpo_repo,
            family.rlvr_repo,
        }:
            return family
    return MODEL_FAMILIES["7b"]


def _config_dimension(config: Any) -> int | None:
    for name in ("d_model", "hidden_size", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _config_layers(config: Any) -> int | None:
    for name in ("n_layers", "num_hidden_layers", "num_layers"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _h100_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name().lower()
    except Exception:
        return False
    return "h100" in name or "h200" in name


def _check_nf4_budget() -> dict[str, str]:
    free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes) - _NF4_RESERVE_BYTES
    if budget < _NF4_MINIMUM_BUDGET_BYTES:
        raise ValueError(
            "insufficient measured free VRAM for NF4 32B load: "
            f"{int(free_bytes) / 1024**3:.2f} GiB free; "
            f"need at least {_NF4_MINIMUM_BUDGET_BYTES / 1024**3:.0f} GiB "
            "after reserve"
        )
    return {"cuda:0": f"{budget // 1024**3}GiB"}


def load_model(
    checkpoint: CheckpointRef,
    dtype: str = "bfloat16",
    quantization: str | None = None,
    device: str = "cuda",
) -> Any:
    """Validate and load one pinned causal language-model checkpoint."""

    if quantization not in (None, "nf4"):
        raise ValueError(f"unknown quantization: {quantization!r}")
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype: {dtype!r}")

    family = _family_for_repo(checkpoint.repo)
    if family.key == "32b" and quantization is None and dtype in {"float32", "float16"}:
        if not _h100_available():
            raise ValueError(
                "32B float32/float16 loading requires NF4 or H100-class memory"
            )

    config = pretrained_config(checkpoint.repo, revision=checkpoint.revision)
    actual_dimension = _config_dimension(config)
    actual_layers = _config_layers(config)
    if actual_dimension != family.d_model or actual_layers != family.n_layers:
        raise ValueError(
            f"checkpoint config does not match {family.key} family: "
            f"d_model={actual_dimension!r} (expected {family.d_model}), "
            f"n_layers={actual_layers!r} (expected {family.n_layers})"
        )

    kwargs: dict[str, Any] = {
        "revision": checkpoint.revision,
        "dtype": _DTYPES[dtype],
        "device_map": {"": device},
        "attn_implementation": "sdpa",
    }
    if quantization == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if device.lower().startswith("cuda"):
            kwargs["max_memory"] = _check_nf4_budget()

    return AutoModelForCausalLM.from_pretrained(checkpoint.repo, **kwargs)


def release_model(model: Any) -> None:
    """Clear CUDA's allocator cache after the caller drops its own reference.

    Callers must rebind or delete their model variable: this function only
    receives a borrowed reference and cannot clear external references.
    """

    import gc

    gc.collect()
    torch.cuda.empty_cache()


def prune_revision_cache(
    checkpoint: CheckpointRef, hub_cache: str | Path | None = None
) -> None:
    """Remove only the cached snapshot and ref for ``checkpoint.revision``."""

    cache_root = Path(
        hub_cache
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache"))
        / "huggingface"
        / "hub"
    )
    repo_dir = cache_root / ("models--" + checkpoint.repo.replace("/", "--"))
    shutil.rmtree(repo_dir / "snapshots" / checkpoint.revision, ignore_errors=True)
    ref = repo_dir / "refs" / checkpoint.revision
    if ref.is_file() or ref.is_symlink():
        ref.unlink()


def hidden_dimension(model: Any) -> int:
    """Return the hidden dimension from a loaded model configuration."""

    value = _config_dimension(model.config)
    if value is None:
        raise ValueError("model config has no hidden dimension")
    return value


def layer_count(model: Any) -> int:
    """Return the transformer layer count from a loaded model configuration."""

    value = _config_layers(model.config)
    if value is None:
        raise ValueError("model config has no layer count")
    return value
