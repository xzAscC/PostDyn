"""Safe loading helpers for Olmo-3 32B Think checkpoints in 4-bit NF4.

The loader deliberately performs GPU-only placement. CPU/disk offload is
unsupported because it changes throughput and downstream placement assumptions.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, cast

from src.config import (
    OLMO3_VARIANTS,
    THINK_32B_RLVR_REVISIONS,
    THINK_32B_SFT_REVISIONS,
)

MIN_BNB_VERSION = "0.49.2"
MIN_TRANSFORMERS_VERSION = "4.57.1"
OLMO3_32B_HIDDEN_SIZE = 5120
OLMO3_32B_LAYERS = 64
GPU_MEMORY_RESERVE_BYTES = 1 * 1024**3
MIN_32B_LOAD_BUDGET_BYTES = 20 * 1024**3
DEFAULT_MODEL_ID = "allenai/Olmo-3-32B-Think-SFT"
CANONICAL_32B_MODEL_REVISIONS = {
    OLMO3_VARIANTS["olmo3-32b-think-sft"].hf_id: frozenset(
        revision
        for schedule in THINK_32B_SFT_REVISIONS.values()
        for revision in schedule.values()
    ),
    OLMO3_VARIANTS["olmo3-32b-think-rlvr"].hf_id: frozenset(
        THINK_32B_RLVR_REVISIONS.values()
    ),
}
CANONICAL_32B_MODEL_IDS = frozenset(CANONICAL_32B_MODEL_REVISIONS)
CANONICAL_NF4_PROVENANCE: dict[str, Any] = {
    "loader": "src.quantized_model_loader.load_olmo3_32b_think",
    "quantization": {
        "bits": 4,
        "type": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
    },
    "requested_device_map": {"": "cuda:0"},
    "allow_auto_fallback": False,
    "use_safetensors": True,
    "max_memory_policy": {
        "source": "cuda.mem_get_info",
        "reserve_bytes": GPU_MEMORY_RESERVE_BYTES,
        "minimum_budget_bytes": MIN_32B_LOAD_BUDGET_BYTES,
    },
}


class QuantizedLoaderError(RuntimeError):
    """Base error for actionable quantized-loader failures."""


class MissingQuantizationDependencyError(QuantizedLoaderError):
    """Raised when the int4 dependency stack is unavailable or incompatible."""


class ModelArchitectureError(QuantizedLoaderError):
    """Raised when a checkpoint is not the expected Olmo-3 32B shape."""


@dataclass
class LoadDiagnostics:
    """Machine-readable evidence about one load attempt."""

    placement: str
    quant_backend: str = "bitsandbytes"
    safetensors: bool = True
    bitsandbytes_version: str | None = None
    transformers_version: str | None = None
    accelerate_version: str | None = None
    device_map: dict[str, Any] = field(default_factory=dict)
    peak_vram_bytes: int | None = None
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "placement": self.placement,
            "quant_backend": self.quant_backend,
            "safetensors": self.safetensors,
            "bitsandbytes_version": self.bitsandbytes_version,
            "transformers_version": self.transformers_version,
            "accelerate_version": self.accelerate_version,
            "device_map": dict(self.device_map),
            "peak_vram_bytes": self.peak_vram_bytes,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class LoadedQuantizedModel:
    model: Any
    tokenizer: Any
    diagnostics: LoadDiagnostics


LOAD_DIAGNOSTICS_KEYS = frozenset(
    {
        "placement",
        "quant_backend",
        "safetensors",
        "bitsandbytes_version",
        "transformers_version",
        "accelerate_version",
        "device_map",
        "peak_vram_bytes",
        "fallback_reason",
    }
)


def validate_nf4_load_diagnostics(value: Any) -> str | None:
    """Validate the exact GPU-only evidence emitted by ``LoadDiagnostics``."""

    if not isinstance(value, dict):
        return "loader diagnostics must be an object"
    if set(value) != LOAD_DIAGNOSTICS_KEYS:
        return "loader diagnostics fields are incomplete or unexpected"
    if value["placement"] != "gpu-only":
        return "loader diagnostics placement is not gpu-only"
    if value["quant_backend"] != "bitsandbytes":
        return "loader diagnostics quant_backend is not bitsandbytes"
    if value["safetensors"] is not True:
        return "loader diagnostics safetensors must be true"
    for key in ("bitsandbytes_version", "transformers_version", "accelerate_version"):
        if not isinstance(value[key], str) or not value[key]:
            return f"loader diagnostics {key} is missing or invalid"
    device_map = value["device_map"]
    if not isinstance(device_map, dict) or not device_map:
        return "loader diagnostics device_map is missing or empty"
    if any(
        not (
            isinstance(device, str)
            and re.fullmatch(r"cuda(?::[0-9]+)?", device) is not None
        )
        for device in device_map.values()
    ):
        return "loader diagnostics device_map is not GPU-only"
    peak = value["peak_vram_bytes"]
    if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
        return "loader diagnostics peak_vram_bytes is invalid"
    if value["fallback_reason"] is not None:
        return "loader diagnostics contains a fallback reason"
    return None


def validate_canonical_32b_request(model_id: Any, revision: Any) -> None:
    """Reject mutable or non-canonical 32B requests before side effects."""

    if model_id is None:
        raise QuantizedLoaderError("--model-id is required with --load")
    if model_id not in CANONICAL_32B_MODEL_IDS:
        raise QuantizedLoaderError(
            "32B loader accepts only the canonical configured SFT/RLVR checkpoints "
            f"{sorted(CANONICAL_32B_MODEL_IDS)!r}; got {model_id!r}."
        )
    if revision is None:
        raise QuantizedLoaderError("--revision is required with --load")
    if (
        not isinstance(revision, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise QuantizedLoaderError(
            "32B checkpoint revision must be an explicit 40-hex SHA"
        )
    if revision.lower() not in {
        candidate.lower() for candidate in CANONICAL_32B_MODEL_REVISIONS[model_id]
    }:
        raise QuantizedLoaderError(
            f"32B checkpoint revision is not configured for {model_id!r}"
        )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_quantization_dependencies() -> dict[str, str]:
    """Validate the optional stack before any checkpoint access."""

    missing = [
        name
        for name in ("transformers", "accelerate", "bitsandbytes")
        if _version(name) is None
    ]
    if missing:
        raise MissingQuantizationDependencyError(
            "NF4 loading requires the optional quantization stack; missing: "
            + ", ".join(missing)
            + ". Install with `uv sync --extra quantization`."
        )

    versions = {
        name: _version(name) or "unknown"
        for name in ("transformers", "accelerate", "bitsandbytes")
    }
    from packaging.version import InvalidVersion, Version

    try:
        if Version(versions["transformers"]) < Version(MIN_TRANSFORMERS_VERSION):
            raise MissingQuantizationDependencyError(
                f"Transformers {MIN_TRANSFORMERS_VERSION}+ is required; found {versions['transformers']}."
            )
        if Version(versions["bitsandbytes"]) < Version(MIN_BNB_VERSION):
            raise MissingQuantizationDependencyError(
                f"bitsandbytes {MIN_BNB_VERSION}+ is required; found {versions['bitsandbytes']}."
            )
    except InvalidVersion as exc:
        raise MissingQuantizationDependencyError(
            f"Could not parse quantization package version: {exc}"
        ) from exc

    try:
        importlib.import_module("bitsandbytes")
    except Exception as exc:
        raise MissingQuantizationDependencyError(
            "bitsandbytes is installed but cannot initialize (usually a CUDA/runtime mismatch): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return versions


def build_nf4_config(torch_module: Any) -> Any:
    """Build the exact NF4 configuration used by the 32B loader."""

    try:
        BitsAndBytesConfig = importlib.import_module("transformers").BitsAndBytesConfig
    except (ImportError, AttributeError) as exc:
        raise MissingQuantizationDependencyError(
            "The installed Transformers does not expose BitsAndBytesConfig; install the quantization extra."
        ) from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_module.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _validate_config(config: Any) -> None:
    hidden_size = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    if hidden_size != OLMO3_32B_HIDDEN_SIZE or layers != OLMO3_32B_LAYERS:
        raise ModelArchitectureError(
            "Expected Olmo-3 32B config with hidden_size=5120 and num_hidden_layers=64; "
            f"found hidden_size={hidden_size!r}, num_hidden_layers={layers!r}."
        )


def _device_map(model: Any) -> dict[str, Any]:
    if hasattr(model, "hf_device_map"):
        value = getattr(model, "hf_device_map")
    else:
        value = getattr(model, "device_map", {})
    return dict(value) if isinstance(value, Mapping) else {"model": str(value)}


def _embedding_device(model: Any) -> str | None:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if not callable(get_input_embeddings):
        return None
    embeddings = get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    device = getattr(weight, "device", None)
    return str(device) if device is not None else None


def _is_bfloat16(value: Any) -> bool:
    return value is not None and (
        value == "bfloat16" or str(value).lower().removeprefix("torch.") == "bfloat16"
    )


def _validate_loaded_quantization(model: Any) -> None:
    if getattr(model, "is_loaded_in_4bit", None) is not True:
        raise QuantizedLoaderError("loaded model is not attested as 4-bit")
    model_config = getattr(model, "config", None)
    quant_config = getattr(model_config, "quantization_config", None)
    if quant_config is None:
        quant_config = getattr(model, "quantization_config", None)
    expected = CANONICAL_NF4_PROVENANCE["quantization"]
    method = getattr(model, "quantization_method", None)
    method = method() if callable(method) else method
    if method is None:
        method = getattr(model_config, "quant_method", None)
    normalized_method = str(method).lower()
    if "bitsandbytes" not in normalized_method and normalized_method != "bnb":
        raise QuantizedLoaderError(
            "loaded model quantization method is not bitsandbytes"
        )
    if (
        quant_config is None
        or getattr(quant_config, "load_in_4bit", None) is not True
        or str(getattr(quant_config, "bnb_4bit_quant_type", "")).lower()
        != expected["type"]
        or getattr(quant_config, "bnb_4bit_use_double_quant", None)
        is not expected["double_quant"]
        or not _is_bfloat16(getattr(quant_config, "bnb_4bit_compute_dtype", None))
    ):
        raise QuantizedLoaderError(
            "loaded model quantization is not canonical 4-bit NF4+bfloat16"
        )


def _genuine_params4bit_type() -> type[Any] | None:
    try:
        params4bit = getattr(
            importlib.import_module("bitsandbytes.nn"), "Params4bit", None
        )
    except Exception:
        return None
    return params4bit if isinstance(params4bit, type) else None


def _is_4bit_parameter(parameter: Any) -> bool:
    params4bit_type = _genuine_params4bit_type()
    return params4bit_type is not None and isinstance(parameter, params4bit_type)


def _is_floating_dtype(dtype: Any) -> bool:
    text = str(dtype).lower().removeprefix("torch.")
    return "float" in text or "bfloat" in text


def _validate_nf4_parameter_attestation(model: Any) -> None:
    iterator = getattr(model, "named_parameters", None)
    if not callable(iterator):
        raise QuantizedLoaderError("loaded model does not expose named parameters")
    found_4bit = False
    for name, parameter in cast(Iterable[tuple[str, Any]], iterator()):
        if _is_4bit_parameter(parameter):
            quant_state = getattr(parameter, "quant_state", None)
            quant_type = getattr(quant_state, "quant_type", None)
            if str(quant_type).strip().lower() != "nf4":
                raise QuantizedLoaderError(
                    f"loaded model genuine NF4 Params4bit parameter {name!r} is not NF4"
                )
            found_4bit = True
        elif _is_floating_dtype(getattr(parameter, "dtype", None)) and not _is_bfloat16(
            getattr(parameter, "dtype", None)
        ):
            raise QuantizedLoaderError(
                f"loaded model ordinary floating parameter {name!r} is not bfloat16"
            )
    if not found_4bit:
        raise QuantizedLoaderError("loaded model has no genuine NF4 Params4bit")


def _validate_concrete_cuda_tensors(model: Any) -> None:
    for kind in ("parameters", "buffers"):
        iterator = getattr(model, f"named_{kind}", None)
        if not callable(iterator):
            raise QuantizedLoaderError(f"loaded model does not expose named {kind}")
        for name, tensor in cast(Iterable[tuple[str, Any]], iterator()):
            device = str(getattr(tensor, "device", "missing"))
            if re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
                raise QuantizedLoaderError(
                    f"loaded model {kind[:-1]} {name!r} is not concrete CUDA: {device!r}"
                )


def _validate_loaded_runtime(model: Any, diagnostics: LoadDiagnostics) -> None:
    diagnostics.device_map = _device_map(model)
    error = validate_nf4_load_diagnostics(diagnostics.as_dict())
    if error is not None:
        raise QuantizedLoaderError(error)
    if any(
        re.fullmatch(r"cuda(?::[0-9]+)?", str(device)) is None
        for device in diagnostics.device_map.values()
    ):
        raise QuantizedLoaderError("loaded model device_map is not GPU-only")
    embedding_device = _embedding_device(model)
    if (
        embedding_device is None
        or re.fullmatch(r"cuda(?::[0-9]+)?", embedding_device) is None
    ):
        raise QuantizedLoaderError(
            f"loaded model embedding placement is not GPU-only: {embedding_device!r}"
        )
    _validate_loaded_quantization(model)
    _validate_nf4_parameter_attestation(model)
    _validate_concrete_cuda_tensors(model)


def _reset_peak_vram(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.reset_peak_memory_stats()


def _peak_vram(torch_module: Any) -> int | None:
    if torch_module.cuda.is_available():
        return int(torch_module.cuda.max_memory_allocated())
    return None


def measured_gpu_memory_budget(
    torch_module: Any,
    *,
    reserve_bytes: int = GPU_MEMORY_RESERVE_BYTES,
    minimum_bytes: int = MIN_32B_LOAD_BUDGET_BYTES,
) -> dict[str, str]:
    """Return a bounded, measured CUDA budget for GPU-only 32B loading.

    ``max_memory`` is deliberately derived from the currently free memory rather
    than being an unbounded device-map hint.  The reserve protects the driver
    and activation workspace; callers must still run checkpoints sequentially.
    """
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise QuantizedLoaderError(
            "cannot measure a 32B CUDA memory budget without CUDA"
        )
    mem_get_info = getattr(cuda, "mem_get_info", None)
    if not callable(mem_get_info):
        raise QuantizedLoaderError(
            "CUDA free-memory telemetry is unavailable; refusing an unbounded 32B load"
        )
    free_bytes, _ = cast(Callable[[int], tuple[int, int]], mem_get_info)(0)
    budget = int(free_bytes) - int(reserve_bytes)
    if budget < minimum_bytes:
        raise QuantizedLoaderError(
            "insufficient measured free VRAM for GPU-only 32B NF4 load: "
            f"{int(free_bytes) / 1024**3:.2f} GiB free, "
            f"{int(reserve_bytes) / 1024**3:.2f} GiB reserved"
        )
    return {"cuda:0": f"{budget // 1024**3}GiB"}


def _memory_limit_bytes(value: Any) -> int:
    if isinstance(value, bool):
        raise QuantizedLoaderError("max_memory values must be positive byte limits")
    if isinstance(value, (int, float)):
        limit = int(value)
    elif isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(GiB|MiB|GB|MB)\s*", value)
        if match is None:
            raise QuantizedLoaderError(f"unsupported max_memory value {value!r}")
        amount, unit = float(match.group(1)), match.group(2)
        multiplier = 1024**3 if unit in {"GiB", "GB"} else 1024**2
        limit = int(amount * multiplier)
    else:
        raise QuantizedLoaderError(f"unsupported max_memory value {value!r}")
    if limit <= 0:
        raise QuantizedLoaderError("max_memory values must be positive")
    return limit


def validate_measured_max_memory(
    torch_module: Any, max_memory: Mapping[Any, str] | None
) -> dict[str, str]:
    measured = measured_gpu_memory_budget(torch_module)
    if max_memory is None:
        return measured
    if set(max_memory) != {"cuda:0"}:
        raise QuantizedLoaderError("32B max_memory must target only cuda:0")
    requested = _memory_limit_bytes(max_memory["cuda:0"])
    measured_bytes = _memory_limit_bytes(measured["cuda:0"])
    if requested > measured_bytes:
        raise QuantizedLoaderError(
            f"requested max_memory exceeds measured safe budget: {requested} > {measured_bytes} bytes"
        )
    return {"cuda:0": str(max_memory["cuda:0"])}


def load_olmo3_32b_think(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    revision: str | None = None,
    max_memory: Mapping[Any, str] | None = None,
    trust_remote_code: bool = False,
) -> LoadedQuantizedModel:
    """Load an Olmo-3 32B Think model with NF4 and explicit placement policy."""

    validate_canonical_32b_request(model_id, revision)

    versions = check_quantization_dependencies()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise QuantizedLoaderError(
            "NF4 32B loading requires CUDA; no CUDA device is available."
        )
    measured_max_memory = validate_measured_max_memory(torch, max_memory)

    transformers = importlib.import_module("transformers")
    config = transformers.AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    _validate_config(config)
    quant_config = build_nf4_config(torch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust_remote_code
    )
    diagnostics = LoadDiagnostics(
        placement="gpu-only",
        bitsandbytes_version=versions["bitsandbytes"],
        transformers_version=versions["transformers"],
        accelerate_version=versions["accelerate"],
    )
    kwargs = {
        "revision": revision,
        "quantization_config": quant_config,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": "cuda:0"},
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
        "use_safetensors": True,
    }
    kwargs["max_memory"] = measured_max_memory
    _reset_peak_vram(torch)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as exc:
        raise QuantizedLoaderError(
            f"GPU-only placement failed: {type(exc).__name__}: {exc}"
        ) from exc

    diagnostics.peak_vram_bytes = _peak_vram(torch)
    _validate_loaded_runtime(model, diagnostics)
    return LoadedQuantizedModel(
        model=model, tokenizer=tokenizer, diagnostics=diagnostics
    )


def forward_with_hidden_states(model: Any, **inputs: Any) -> Any:
    """Run a regular forward while requesting all hidden states."""

    inputs["output_hidden_states"] = True
    return model(**inputs)


def register_forward_hooks(
    model: Any,
    hook: Callable[[Any, tuple[Any, ...], Any], Any],
    *,
    layer_indices: tuple[int, ...] | None = None,
) -> list[Any]:
    """Register hooks that remain active during both forward and generate."""

    if layer_indices is not None:
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer selection contains duplicate indices")
        if any(index < 0 or index >= OLMO3_32B_LAYERS for index in layer_indices):
            raise ValueError(
                f"layer selection must use indices in [0, {OLMO3_32B_LAYERS - 1}]"
            )
    handles = []
    for name, module in model.named_modules():
        if layer_indices is not None:
            match = re.fullmatch(r"model\.layers\.(\d+)", name)
            if match is None or int(match.group(1)) not in layer_indices:
                continue
        handles.append(module.register_forward_hook(hook))
    return handles


def generate_with_hooks(
    model: Any,
    inputs: Mapping[str, Any],
    hook: Callable[[Any, tuple[Any, ...], Any], Any],
    *,
    layer_indices: tuple[int, ...] | None = None,
    **generate_kwargs: Any,
) -> Any:
    """Generate with temporary hooks; dispatched models are never moved."""

    handles = register_forward_hooks(model, hook, layer_indices=layer_indices)
    try:
        return model.generate(**dict(inputs), **generate_kwargs)
    finally:
        for handle in handles:
            handle.remove()
