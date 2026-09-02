#!/usr/bin/env python3
"""Extract signed Think differential subspaces across the memo domain grid.

Uses 1,000 streaming prompts per memo domain and the final attention token,
with target-vs-WikiText and Math robustness pairs, while:

  * models are the released Think-SFT and Think checkpoints
  * both positive (math-dominant) and negative (text-dominant) eigenspaces
  * reports K, d_eff, and SubSim(SFT, Think) per sign

Default: 7B bfloat16. 32B uses the optional NF4 loader and requires
``--allow-32b``.

Usage::

    uv run python scripts/run_think_sft_differential_subspace.py
    uv run python scripts/run_think_sft_differential_subspace.py --quick
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, cast

import torch
from safetensors.torch import load_file, save_file


from scripts.run_math_differential_subspace import (
    _token_lengths,
    preflight_tokenizer_prompts,
    quick_sample_count,
)
from postdyn.concept_dynamics import _load_model_and_tokenizer
from postdyn.quantized_model_loader import (
    CANONICAL_NF4_PROVENANCE,
    validate_nf4_load_diagnostics,
)
from postdyn.differential_subspace import (
    SignedDifferentialSubspace,
    compute_signed_differential_subspace,
    compute_stability_trajectory,
    signed_subspace_to_serializable,
    subspace_stability,
)
import postdyn.differential_subspace as differential_core
from postdyn.domain_datasets import (
    DOLCI_HF_IDS,
    DOLCI_HF_REVISIONS,
    WIKITEXT_CONFIG,
    WIKITEXT_HF_ID,
    WIKITEXT_HF_REVISION,
    WIKITEXT_SPLIT,
    load_domain_prompt_selection,
    resolve_hub_dataset_revision,
)
from postdyn.think_sft_differential_experiment import (
    CONCEPT_PAIRS,
    DTYPE,
    EXTRACTION_CONTRACT,
    FIGURES_SUBDIR,
    MANIFESTS_SUBDIR,
    MAX_SEQ_LEN,
    METRICS_SUBDIR,
    PROMPTS_SUBDIR,
    SAMPLE_SEED,
    SCALE_32B,
    SCALE_7B,
    TAU,
    FAMILY_THINK,
    TRAJECTORY_SFT,
    U_SUBDIR,
    USE_CHAT_TEMPLATE,
    covariance_n_samples,
    extraction_protocol_payload,
    extraction_protocols_equal,
    fixed_point_configs,
    validate_extraction_protocol,
    validate_root_ownership,
    validate_extraction_root_not_other_trajectory,
    claim_root_ownership,
    root_for_trajectory,
    ensure_root_ownership,
    trajectory_config,
    layers_for_scale,
    model_keys_for_scale,
    model_config,
    sft_model_key,
)

ModelLoader = Callable[[Any, Optional[str]], tuple[Any, Any]]
NF4_PROVENANCE = CANONICAL_NF4_PROVENANCE


_SKIP_7B_GATE = False


def set_skip_7b_gate(value: bool) -> None:
    global _SKIP_7B_GATE
    _SKIP_7B_GATE = value


def _require_canonical_7b(*, project_root: Path | None = None) -> None:
    if _SKIP_7B_GATE:
        print("WARNING: canonical 7B preflight gate skipped (--skip-7b-gate)")
        return
    import importlib

    require = importlib.import_module(
        "postdyn.cross_pipeline_integrity"
    ).require_canonical_7b_extraction
    if project_root is None:
        require()
    else:
        require(project_root=project_root)


def apply_concept_filter(selection: str) -> None:
    global CONCEPT_PAIRS
    wanted = {name.strip() for name in selection.split(",") if name.strip()}
    if not wanted:
        raise ValueError("--concepts selection is empty")
    valid = {name for name, _, _ in CONCEPT_PAIRS}
    unknown = sorted(wanted - valid)
    if unknown:
        raise ValueError(f"unknown concept(s) {unknown}; valid: {sorted(valid)}")
    CONCEPT_PAIRS = tuple(pair for pair in CONCEPT_PAIRS if pair[0] in wanted)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _prompt_fingerprint(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _u_paths(
    root: Path, model_name: str, checkpoint: str, layer: int, concept: str
) -> tuple[Path, Path]:
    base = root / U_SUBDIR / model_name / checkpoint / f"layer_{layer}"
    return base / f"{concept}.safetensors", base / f"{concept}.json"


def _layer_metrics_path(
    root: Path, model_name: str, checkpoint: str, layer: int
) -> Path:
    return root / METRICS_SUBDIR / model_name / checkpoint / f"layer_{layer}.json"


def _expected_prompt_source(domain: str) -> dict[str, str]:
    if domain == "wikitext":
        return {
            "kind": "huggingface",
            "hf_id": WIKITEXT_HF_ID,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "revision": resolve_hub_dataset_revision(
                WIKITEXT_HF_ID, WIKITEXT_HF_REVISION
            ),
        }
    return {
        "kind": "dolci",
        "hf_id": DOLCI_HF_IDS[domain],
        "revision": resolve_hub_dataset_revision(
            DOLCI_HF_IDS[domain], DOLCI_HF_REVISIONS[domain]
        ),
    }


def _cached_prompt_source_is_immutable(domain: str, source: object) -> bool:
    if not isinstance(source, dict):
        return False
    if domain == "wikitext":
        expected = {
            "kind": "huggingface",
            "hf_id": WIKITEXT_HF_ID,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
        }
    else:
        expected = {"kind": "dolci", "hf_id": DOLCI_HF_IDS[domain]}
    return (
        set(source) == set(expected) | {"revision"}
        and all(source.get(key) == value for key, value in expected.items())
        and isinstance(source.get("revision"), str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]) is not None
    )


def prepare_domain_prompts(
    root: Path,
    *,
    n_samples: int,
    seed: int,
    allow_hf: bool,
    max_seq_len: int,
    use_chat_template: bool,
) -> dict[str, list[str]]:
    domains = sorted({d for _, c, r in CONCEPT_PAIRS for d in (c, r)})
    prompt_dir = root / PROMPTS_SUBDIR
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompts_by_domain: dict[str, list[str]] = {}
    for domain in domains:
        cache_path = prompt_dir / f"{domain}.json"
        if cache_path.is_file():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            prompts = data.get("prompts")
            if (
                data.get("domain") == domain
                and data.get("n_samples") == n_samples
                and data.get("seed") == seed
                and isinstance(prompts, list)
                and len(prompts) == n_samples
                and data.get("prompt_fingerprint") == _prompt_fingerprint(prompts)
                and _cached_prompt_source_is_immutable(domain, data.get("source"))
                and data.get("extraction_contract") == EXTRACTION_CONTRACT
                and data.get("max_seq_len") == max_seq_len
                and data.get("use_chat_template") == use_chat_template
            ):
                prompts_by_domain[domain] = prompts
                continue
            raise ValueError(f"Incompatible prompt cache for {domain}: {cache_path}")
        if not allow_hf:
            raise ValueError(
                "memo domain protocol requires streaming HuggingFace sources"
            )
        selection = load_domain_prompt_selection(
            domain, n_samples=n_samples, seed=seed, prefer_local=False
        )
        prompts = selection.as_list()
        source = dict(selection.source)
        if source != _expected_prompt_source(domain):
            raise ValueError(f"unexpected streaming source provenance for {domain}")
        prompts_by_domain[domain] = prompts
        _atomic_write_json(
            cache_path,
            {
                "domain": domain,
                "n_samples": n_samples,
                "seed": seed,
                "source": source,
                "prompt_fingerprint": _prompt_fingerprint(prompts),
                "use_chat_template": use_chat_template,
                "max_seq_len": max_seq_len,
                "extraction_contract": EXTRACTION_CONTRACT,
                "prompts": prompts,
            },
        )
    return prompts_by_domain


def _manifest_path(root: Path, model_name: str, checkpoint: str) -> Path:
    return root / MANIFESTS_SUBDIR / f"{model_name}__{checkpoint}.json"


def setup_signature(
    *,
    pairs: list[tuple[str, str, str]],
    model_keys: list[str],
    checkpoints: list[str],
    layers: list[int],
    model_ids: dict[str, str],
    dataset_sources: dict[str, dict[str, str]],
    prompt_fingerprints: dict[str, str],
    n_samples: int,
    tau: float,
    max_seq_len: int,
    use_chat_template: bool,
    seed: int,
    extraction_contract: str,
    dtype: str,
    signed: bool,
    scale: str,
    loader_provenance: dict[str, Any] | None = None,
    trajectory: str | None = None,
    checkpoint_revisions: dict[str, str] | None = None,
) -> str:
    payload = {
        "pairs": pairs,
        "model_keys": model_keys,
        "checkpoints": checkpoints,
        "layers": layers,
        "model_ids": model_ids,
        "dataset_sources": dataset_sources,
        "prompt_fingerprints": list(prompt_fingerprints.items()),
        "n_samples": n_samples,
        "tau": tau,
        "max_seq_len": max_seq_len,
        "use_chat_template": use_chat_template,
        "seed": seed,
        "extraction_contract": extraction_contract,
        "dtype": dtype,
        "signed": signed,
        "scale": scale,
    }
    if loader_provenance is not None:
        payload["loader_provenance"] = loader_provenance
    if scale == SCALE_32B:
        payload["trajectory"] = trajectory or ""
        payload["checkpoint_revisions"] = list(
            (checkpoint, (checkpoint_revisions or {}).get(checkpoint) or "")
            for checkpoint in checkpoints
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def gpu_memory_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return float(props.total_memory) / (1024**3)


_COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def resolve_model_revision(
    model_id: str,
    revision: str,
    *,
    api_factory: Callable[[], Any] | None = None,
) -> str:
    """Resolve a symbolic Hub revision to an immutable commit SHA."""
    if _COMMIT_SHA_RE.fullmatch(revision):
        return revision.lower()
    if not revision:
        raise ValueError(f"empty revision for {model_id!r}")
    try:
        if api_factory is None:
            from huggingface_hub import HfApi

            api_factory = HfApi
        info = api_factory().model_info(model_id, revision=revision)
        sha = getattr(info, "sha", None)
    except Exception as exc:
        raise ValueError(
            f"could not resolve symbolic revision {model_id}@{revision!r}"
        ) from exc
    if not isinstance(sha, str) or _COMMIT_SHA_RE.fullmatch(sha) is None:
        raise ValueError(
            f"Hub returned no immutable commit SHA for {model_id}@{revision}"
        )
    return sha.lower()


def resolve_model_revisions(
    model_key: str, revisions: dict[str, str]
) -> dict[str, str]:
    model_id = model_config(model_key).hf_id
    return {
        checkpoint: resolve_model_revision(model_id, revision)
        for checkpoint, revision in revisions.items()
    }


def _input_device(model: Any) -> Any:
    embeddings = getattr(model, "get_input_embeddings", lambda: None)()
    device = getattr(getattr(embeddings, "weight", None), "device", None)
    if device is not None and str(device) != "meta":
        return device
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for name in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            if name in device_map and str(device_map[name]) != "meta":
                return device_map[name]
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model has no concrete input device") from exc


def _layer_module(model: Any, layer: int) -> Any | None:
    for name, module in model.named_modules():
        if name in (f"model.layers.{layer}", f"transformer.h.{layer}"):
            return module
    return None


def extract_raw_layer_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    *,
    token_budget: int = 8192,
    lengths: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract selected residual layers without retaining the full layer tuple."""
    device = _input_device(model)
    total = len(texts)
    features: dict[int, list[torch.Tensor | None]] = {
        layer: [None] * total for layer in layers
    }
    modules = {layer: _layer_module(model, layer) for layer in layers}
    if any(module is None for module in modules.values()):
        raise ValueError("model does not expose all requested transformer layers")
    if lengths is None:
        lengths = _token_lengths(tokenizer, texts)
    order = sorted(range(total), key=lambda index: lengths[index])
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def capture(layer: int):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(value, torch.Tensor):
                captured[layer] = value

        return hook

    model.config.use_cache = False
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "sdpa"
    try:
        for layer, module in modules.items():
            handles.append(module.register_forward_hook(capture(layer)))
        position = 0
        while position < len(order):
            take = 1
            padded = lengths[order[position]]
            while position + take < len(order):
                candidate = max(padded, lengths[order[position + take]])
                if (take + 1) * candidate > token_budget:
                    break
                padded = candidate
                take += 1
            batch_indices = order[position : position + take]
            position += take
            encoded = tokenizer(
                [texts[index] for index in batch_indices],
                return_tensors="pt",
                truncation=False,
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in encoded.items()}
            captured.clear()
            with torch.no_grad():
                model(**inputs, use_cache=False, output_hidden_states=False)
            attention_mask = inputs.get("attention_mask")
            for row, index in enumerate(batch_indices):
                if attention_mask is None:
                    last_index = inputs["input_ids"].shape[1] - 1
                else:
                    last_index = int(attention_mask[row].sum().item()) - 1
                for layer in layers:
                    value = captured.get(layer)
                    if value is None:
                        raise ValueError(
                            f"selected layer {layer} produced no activation"
                        )
                    features[layer][index] = (
                        value[row, last_index].detach().cpu().float()
                    )
    finally:
        for handle in handles:
            handle.remove()
    extracted: dict[int, torch.Tensor] = {}
    for layer, values in features.items():
        present = [value for value in values if value is not None]
        if len(present) != total:
            raise ValueError(f"layer {layer} is missing extracted activations")
        extracted[layer] = torch.stack(present)
    return extracted


def validate_scale(scale: str, allow_32b: bool) -> None:
    if scale == SCALE_7B:
        return
    if scale != SCALE_32B:
        raise ValueError(f"Unknown scale {scale!r}")
    if not allow_32b:
        raise ValueError("32B NF4 execution is opt-in; pass --allow-32b.")


def build_model_loader(
    scale: str,
    *,
    runtime_provenance: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> ModelLoader:
    if scale == SCALE_7B:
        return _load_model_and_tokenizer
    if scale != SCALE_32B:
        raise ValueError(f"Unknown scale {scale!r}")

    def load_32b(cfg: Any, revision: Optional[str]) -> tuple[Any, Any]:
        _require_canonical_7b(project_root=project_root)
        if revision is None:
            raise ValueError("32B trajectory loads require a pinned revision")
        from postdyn.quantized_model_loader import load_olmo3_32b_think

        loaded = load_olmo3_32b_think(
            cfg.hf_id,
            revision=revision,
        )
        config = getattr(loaded.model, "config", None)
        if config is not None:
            config.use_cache = False
            if hasattr(config, "attn_implementation"):
                config.attn_implementation = "sdpa"
        if getattr(loaded.tokenizer, "pad_token", None) is None:
            loaded.tokenizer.pad_token = loaded.tokenizer.eos_token
        if runtime_provenance is not None:
            runtime_provenance.clear()
            runtime_provenance.update(loaded.diagnostics.as_dict())
        return loaded.model, loaded.tokenizer

    return load_32b


def _core_value(sub: Any, name: str, default: Any = None) -> Any:
    if isinstance(sub, Mapping):
        return sub.get(name, default)
    return getattr(sub, name, default)


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _tensor_value(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().contiguous().clone()
    if isinstance(value, (list, tuple)):
        try:
            return torch.as_tensor(value, dtype=torch.float32).contiguous()
        except (TypeError, ValueError):
            return None
    return None


def _new_core_metrics(sub: Any) -> dict[str, Any]:
    names = (
        "energy_pos",
        "energy_neg",
        "frobenius_strength_pos",
        "frobenius_strength_neg",
        "r_pos",
        "d_eff_pos",
        "d_eff_neg",
        "emergence_pos",
        "emergence_neg",
    )
    return {
        name: _json_value(value)
        for name in names
        if (value := _core_value(sub, name)) is not None
    }


CORE_SCALAR_METRIC_FIELDS = frozenset(
    {
        "k_pos",
        "k_neg",
        "d_eff_pos",
        "d_eff_neg",
        "energy_pos",
        "energy_neg",
        "frobenius_strength_pos",
        "frobenius_strength_neg",
        "r_pos",
        "emergence_pos",
        "emergence_neg",
        "tr_concept",
        "tr_ref",
    }
)


def _summary_core_metrics(concepts: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            key: value
            for key, value in item.items()
            if key in CORE_SCALAR_METRIC_FIELDS
        }
        for name, item in concepts.items()
        if isinstance(item, Mapping)
    }


def _shared_histogram_metadata(values: list[float], bins: int = 32) -> dict[str, Any]:
    finite = [float(value) for value in values if torch.isfinite(torch.tensor(value))]
    if not finite:
        low, high = 0.0, 1.0
        return {
            "bins": bins,
            "range": [low, high],
            "edges": torch.linspace(low, high, bins + 1).tolist(),
        }
    low, high = min(finite), max(finite)
    if low == high:
        padding = max(abs(low) * 0.05, 1e-6)
        low, high = low - padding, high + padding
    return {
        "bins": bins,
        "range": [low, high],
        "edges": torch.linspace(low, high, bins + 1).tolist(),
    }


def residual_to_final_analysis(current: Any, final: Any) -> dict[str, Any]:
    helper = getattr(differential_core, "residual_to_later_subspace_overlap", None)
    if not callable(helper):
        raise RuntimeError("upgraded core residual-overlap helper is unavailable")
    result = helper(current, final)
    if not isinstance(result, Mapping):
        raise TypeError("core residual-overlap helper returned a non-mapping")
    return cast(dict[str, Any], result)


def save_signed_subspace(
    root: Path,
    model_name: str,
    checkpoint: str,
    layer: int,
    sub: SignedDifferentialSubspace,
    setup_sig: str | None = None,
    revision: str | None = None,
    loader_provenance: dict[str, Any] | None = None,
    extraction_protocol: dict[str, object] | None = None,
    save_tensors: bool = False,
) -> None:
    st_path, js_path = _u_paths(root, model_name, checkpoint, layer, sub.concept)
    st_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_st_path: Path | None = None
    payload: dict[str, Any] = {}
    if save_tensors:
        payload = {
            "U_pos": sub.u_pos.contiguous().float(),
            "U_neg": sub.u_neg.contiguous().float(),
            "eigenvalues_pos": sub.eigenvalues_pos.contiguous().float(),
            "eigenvalues_neg": sub.eigenvalues_neg.contiguous().float(),
        }
        tensor_fields = {
            "U_pos_full": _core_value(sub, "u_pos_full"),
            "U_neg_full": _core_value(sub, "u_neg_full"),
            "residual_U_pos": _core_value(sub, "residual_u_pos"),
            "residual_U_neg": _core_value(sub, "residual_u_neg"),
            "eigenvalues_signed": _core_value(sub, "eigenvalues_signed"),
            "eigenvectors_signed": _core_value(sub, "eigenvectors_signed"),
        }
        for name, value in tensor_fields.items():
            tensor = _tensor_value(value)
            if tensor is not None:
                payload[name] = tensor
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{st_path.name}.", suffix=".tmp", dir=st_path.parent
        )
        os.close(fd)
        tmp_st_path = Path(tmp_name)
    try:
        if tmp_st_path is not None:
            save_file(payload, str(tmp_st_path))
        meta = signed_subspace_to_serializable(sub)
        meta.update(_new_core_metrics(sub))
        meta["core_api_fields"] = sorted(_new_core_metrics(sub))
        for key in (
            "eigenvalues_pos",
            "eigenvalues_neg",
            "eigenvalues_signed",
            "eigenvectors_signed",
            "u_pos_full",
            "u_neg_full",
        ):
            meta.pop(key, None)
        meta.update(
            {
                "model": model_name,
                "checkpoint": checkpoint,
                "layer": layer,
                "setup_signature": setup_sig,
                "revision": revision,
                "tensors_saved": save_tensors,
            }
        )
        if loader_provenance is not None:
            meta["loader_provenance"] = dict(loader_provenance)
        if extraction_protocol is not None:
            meta["extraction_protocol"] = dict(extraction_protocol)
        _atomic_write_json(js_path, meta)
        if tmp_st_path is not None:
            os.replace(tmp_st_path, st_path)
    except BaseException:
        if tmp_st_path is not None:
            tmp_st_path.unlink(missing_ok=True)
        raise


def load_signed_subspace(
    root: Path, model_name: str, checkpoint: str, layer: int, concept: str
) -> SignedDifferentialSubspace:
    st_path, js_path = _u_paths(root, model_name, checkpoint, layer, concept)
    if not st_path.exists():
        raise FileNotFoundError(
            f"{st_path}: tensors were not saved for this subspace "
            "(JSON-only run); re-run with --save-tensors to enable tensor reuse"
        )
    tensors = load_file(str(st_path))
    meta = json.loads(js_path.read_text(encoding="utf-8"))
    return SignedDifferentialSubspace(
        concept=concept,
        tau=float(meta["tau"]),
        n_concept=int(meta["n_concept"]),
        n_ref=int(meta["n_ref"]),
        d_model=int(meta["d_model"]),
        tr_concept=float(meta["tr_concept"]),
        tr_ref=float(meta["tr_ref"]),
        u_pos=tensors["U_pos"],
        eigenvalues_pos=tensors["eigenvalues_pos"],
        k_pos=int(meta["k_pos"]),
        d_eff_pos=float(meta["d_eff_pos"]),
        geometry_strength_pos=float(meta["geometry_strength_pos"]),
        u_neg=tensors["U_neg"],
        eigenvalues_neg=tensors["eigenvalues_neg"],
        k_neg=int(meta["k_neg"]),
        d_eff_neg=float(meta["d_eff_neg"]),
        geometry_strength_neg=float(meta["geometry_strength_neg"]),
        energy_pos=float(meta.get("energy_pos", 0.0)),
        energy_neg=float(meta.get("energy_neg", 0.0)),
        frobenius_strength_pos=float(meta.get("frobenius_strength_pos", 0.0)),
        frobenius_strength_neg=float(meta.get("frobenius_strength_neg", 0.0)),
        r_pos=float(meta.get("r_pos", 0.0)),
        emergence_pos=(
            None if meta.get("emergence_pos") is None else float(meta["emergence_pos"])
        ),
        emergence_neg=(
            None if meta.get("emergence_neg") is None else float(meta["emergence_neg"])
        ),
        u_pos_full=tensors.get("U_pos_full"),
        u_neg_full=tensors.get("U_neg_full"),
        eigenvalues_signed=tensors.get("eigenvalues_signed"),
        eigenvectors_signed=tensors.get("eigenvectors_signed"),
    )


def subspace_complete(
    root: Path,
    model_name: str,
    checkpoint: str,
    layer: int,
    concept: str,
    setup_sig: str | None = None,
    expected_revision: str | None = None,
) -> bool:
    st_path, js_path = _u_paths(root, model_name, checkpoint, layer, concept)
    if not st_path.is_file() or not js_path.is_file():
        return False
    try:
        meta = json.loads(js_path.read_text(encoding="utf-8"))
        tensors = load_file(str(st_path))
    except (OSError, json.JSONDecodeError, KeyError, RuntimeError, ValueError):
        return False
    if not isinstance(meta, dict):
        return False
    try:
        if setup_sig is not None and meta.get("setup_signature") != setup_sig:
            return False
        if expected_revision is not None and meta.get("revision") != expected_revision:
            return False
        if meta.get("model") != model_name or meta.get("concept") != concept:
            return False
        if meta.get("checkpoint") != checkpoint:
            return False
        if int(meta.get("layer", -1)) != layer:
            return False
        u_pos = tensors.get("U_pos")
        u_neg = tensors.get("U_neg")
        return (
            {
                "U_pos",
                "U_neg",
                "eigenvalues_pos",
                "eigenvalues_neg",
                "U_pos_full",
                "U_neg_full",
                "eigenvalues_signed",
                "eigenvectors_signed",
            }.issubset(tensors)
            and u_pos is not None
            and u_neg is not None
            and list(u_pos.shape) == meta.get("u_pos_shape")
            and list(u_neg.shape) == meta.get("u_neg_shape")
            and int(meta.get("k_pos", -1)) == int(u_pos.shape[1])
            and int(meta.get("k_neg", -1)) == int(u_neg.shape[1])
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def model_complete(
    root: Path,
    model_name: str,
    checkpoint: str,
    layers: list[int],
    concepts: list[str],
    setup_sig: str | None = None,
    expected_revision: str | None = None,
    expected_loader_provenance: dict[str, Any] | None = None,
    expected_extraction_protocol: dict[str, object] | None = None,
) -> bool:
    for layer in layers:
        for concept in concepts:
            if not subspace_complete(
                root,
                model_name,
                checkpoint,
                layer,
                concept,
                setup_sig,
                expected_revision,
            ):
                return False
        metrics_path = _layer_metrics_path(root, model_name, checkpoint, layer)
        if not metrics_path.is_file():
            return False
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(metrics, dict):
            return False
        if (
            metrics.get("setup_signature") != setup_sig
            or metrics.get("model") != model_name
            or metrics.get("checkpoint") != checkpoint
            or metrics.get("layer") != layer
            or (
                expected_revision is not None
                and metrics.get("revision") != expected_revision
            )
        ):
            return False
    manifest_path = _manifest_path(root, model_name, checkpoint)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("status") == "ok"
        and manifest.get("model") == model_name
        and manifest.get("checkpoint") == checkpoint
        and (expected_revision is None or manifest.get("revision") == expected_revision)
        and (
            expected_loader_provenance is None
            or manifest.get("loader_provenance") == expected_loader_provenance
        )
        and (
            expected_loader_provenance is None
            or validate_nf4_load_diagnostics(manifest.get("runtime_provenance")) is None
        )
        and manifest.get("setup_signature") == setup_sig
        and (
            expected_extraction_protocol is None
            or extraction_protocols_equal(
                manifest.get("extraction_protocol"), expected_extraction_protocol
            )
        )
    )


def _build_setup_sig(
    *,
    scale: str,
    model_key: str,
    checkpoints: list[str],
    layers: list[int],
    n_samples: int,
    tau: float,
    max_seq_len: int,
    seed: int,
    domain_prompts: dict[str, list[str]],
    trajectory: str | None = None,
    revisions: dict[str, str] | None = None,
) -> str:
    ordered_domains = list(
        dict.fromkeys(
            domain
            for _, concept_domain, ref_domain in CONCEPT_PAIRS
            for domain in (concept_domain, ref_domain)
        )
    )
    dataset_sources = {
        domain: _expected_prompt_source(domain) for domain in ordered_domains
    }
    prompt_fingerprints = {
        domain: _prompt_fingerprint(domain_prompts[domain][:n_samples])
        for domain in ordered_domains
    }
    return setup_signature(
        pairs=list(CONCEPT_PAIRS),
        model_keys=[model_key],
        checkpoints=checkpoints,
        layers=layers,
        model_ids={model_key: model_config(model_key).hf_id},
        dataset_sources=dataset_sources,
        prompt_fingerprints=prompt_fingerprints,
        n_samples=n_samples,
        tau=tau,
        max_seq_len=max_seq_len,
        use_chat_template=USE_CHAT_TEMPLATE,
        seed=seed,
        extraction_contract=EXTRACTION_CONTRACT,
        dtype=DTYPE,
        signed=True,
        scale=scale,
        loader_provenance=NF4_PROVENANCE if scale == SCALE_32B else None,
        trajectory=trajectory,
        checkpoint_revisions=revisions,
    )


def validate_selection(
    checkpoints: list[str],
    layers: list[int],
    canonical_checkpoints: list[str],
    canonical_layers: list[int],
) -> None:
    if not checkpoints or not layers:
        raise ValueError("checkpoint and layer selection must be non-empty")
    if len(set(checkpoints)) != len(checkpoints) or any(
        checkpoint not in canonical_checkpoints for checkpoint in checkpoints
    ):
        raise ValueError("checkpoint selection contains unknown or duplicate selection")
    if len(set(layers)) != len(layers) or any(
        layer not in canonical_layers for layer in layers
    ):
        raise ValueError("layer selection contains unknown or duplicate selection")


def _run_checkpoint(
    model_key: str,
    checkpoint: str,
    revision: str,
    *,
    root: Path,
    scale: str,
    layers: list[int],
    n_samples: int,
    max_seq_len: int,
    tau: float,
    domain_prompts: dict[str, list[str]],
    setup_sig: str,
    model_loader: ModelLoader = _load_model_and_tokenizer,
    runtime_provenance: dict[str, Any] | None = None,
    canonical_protocol: bool = True,
    trajectory: str | None = None,
    project_root: Path | None = None,
    validate_32b_checkpoint: bool = True,
    save_tensors: bool = False,
    token_budget: int = 8192,
) -> dict[str, Any]:
    allowed_model_keys = model_keys_for_scale(scale) + tuple(
        model_key for model_key, _revision in fixed_point_configs(scale).values()
    )
    if model_key not in allowed_model_keys:
        raise ValueError(
            f"model key {model_key!r} does not belong to scale {scale!r}; "
            f"expected one of {allowed_model_keys!r}"
        )
    if _COMMIT_SHA_RE.fullmatch(revision) is None:
        raise ValueError(
            f"model execution requires an immutable commit SHA, got {revision!r}"
        )
    if scale == SCALE_32B and validate_32b_checkpoint:
        _require_canonical_7b(project_root=project_root)
    cfg = model_config(model_key)
    extraction_protocol = extraction_protocol_payload(
        n_samples=n_samples,
        tau=tau,
        max_seq_len=max_seq_len,
        use_chat_template=USE_CHAT_TEMPLATE,
        extraction_contract=EXTRACTION_CONTRACT,
        dtype=DTYPE,
        signed=True,
    )
    if scale == SCALE_32B:
        validate_extraction_protocol(
            extraction_protocol, canonical=canonical_protocol, scale=scale
        )
    active_pairs = tuple(
        pair
        for pair in CONCEPT_PAIRS
        if pair[1] in domain_prompts and pair[2] in domain_prompts
    )
    if not active_pairs:
        raise ValueError(
            "domain prompt selection does not contain a complete concept pair"
        )
    concepts = [name for name, _, _ in active_pairs]
    if model_complete(
        root,
        cfg.name,
        checkpoint,
        layers,
        concepts,
        setup_sig,
        revision,
        NF4_PROVENANCE if scale == SCALE_32B else None,
        extraction_protocol if scale == SCALE_32B else None,
    ):
        print(f"[skip] {cfg.name}/{checkpoint} already complete")
        return {"model": cfg.name, "checkpoint": checkpoint, "status": "skipped"}

    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Signed ΔΣ: {cfg.name} / {checkpoint} ({cfg.hf_id}) dtype={DTYPE}")
    print(f"Layers={layers} n_samples={n_samples} tau={tau}")
    print(f"{'=' * 60}")

    needed_domains = sorted({d for _, c, r in active_pairs for d in (c, r)})
    model, tokenizer = model_loader(cfg, revision)
    runtime_config = getattr(model, "config", None)
    if runtime_config is not None:
        runtime_config.use_cache = False
        if hasattr(runtime_config, "attn_implementation"):
            runtime_config.attn_implementation = "sdpa"
    domain_acts: dict[str, dict[int, torch.Tensor]] = {}
    try:
        for domain in needed_domains:
            texts = domain_prompts[domain][:n_samples]
            print(f"  Extracting domain={domain} n={len(texts)} ...")
            prompt_lengths = preflight_tokenizer_prompts(tokenizer, texts, max_seq_len)
            domain_acts[domain] = extract_raw_layer_activations(
                model,
                tokenizer,
                texts,
                layers,
                token_budget=token_budget,
                lengths=prompt_lengths,
            )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    layer_results: dict[str, Any] = {}
    for layer in layers:
        subs: dict[str, SignedDifferentialSubspace] = {}
        for concept_name, c_dom, r_dom in active_pairs:
            sub = compute_signed_differential_subspace(
                domain_acts[c_dom][layer],
                domain_acts[r_dom][layer],
                concept=concept_name,
                tau=tau,
            )
            save_signed_subspace(
                root,
                cfg.name,
                checkpoint,
                layer,
                sub,
                setup_sig,
                revision,
                loader_provenance=(NF4_PROVENANCE if scale == SCALE_32B else None),
                extraction_protocol=extraction_protocol if scale == SCALE_32B else None,
                save_tensors=save_tensors,
            )
            subs[concept_name] = sub
            print(
                f"  layer={layer} {concept_name}: "
                f"K+={sub.k_pos} K-={sub.k_neg} "
                f"d_eff+={sub.d_eff_pos:.3f} d_eff-={sub.d_eff_neg:.3f}"
            )
        concept_payloads = {}
        histogram_values: dict[str, list[float]] = {}
        for name, sub in subs.items():
            item = signed_subspace_to_serializable(sub)
            item.update(_new_core_metrics(sub))
            item.pop("eigenvalues_pos", None)
            item.pop("eigenvalues_neg", None)
            item.pop("eigenvalues_signed", None)
            item.pop("eigenvectors_signed", None)
            concept_payloads[name] = item
            signed_values = _core_value(sub, "eigenvalues_signed")
            if isinstance(signed_values, torch.Tensor):
                histogram_values[name] = [float(x) for x in signed_values.tolist()]
        payload: dict[str, Any] = {
            "model": cfg.name,
            "checkpoint": checkpoint,
            "layer": layer,
            "tau": tau,
            "n_samples": n_samples,
            "revision": revision,
            "setup_signature": setup_sig,
            "concepts": concept_payloads,
            "three_metrics": {
                "retained_K": {
                    name: {"pos": sub.k_pos, "neg": sub.k_neg}
                    for name, sub in subs.items()
                },
                "d_eff": {
                    name: {"pos": sub.d_eff_pos, "neg": sub.d_eff_neg}
                    for name, sub in subs.items()
                },
                "subspace_stability": "see metrics/stability.json",
            },
            "histogram": {
                name: _shared_histogram_metadata(values)
                for name, values in histogram_values.items()
            },
        }
        if scale == SCALE_32B:
            payload["extraction_protocol"] = extraction_protocol
        _atomic_write_json(
            _layer_metrics_path(root, cfg.name, checkpoint, layer), payload
        )
        layer_results[str(layer)] = payload["three_metrics"]

    elapsed = time.time() - t0
    manifest = {
        "model": cfg.name,
        "hf_id": cfg.hf_id,
        "checkpoint": checkpoint,
        "revision": revision,
        "scale": scale,
        "layers": layers,
        "concepts": concepts,
        "n_samples": n_samples,
        "tau": tau,
        "dtype": DTYPE,
        "setup_signature": setup_sig,
        "elapsed_seconds": round(elapsed, 2),
        "status": "ok",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "layer_three_metrics": layer_results,
    }
    if scale == SCALE_32B:
        manifest["extraction_protocol"] = extraction_protocol
        manifest["canonical_protocol"] = canonical_protocol
    if runtime_provenance:
        manifest["runtime_provenance"] = dict(runtime_provenance)
        manifest["loader_provenance"] = NF4_PROVENANCE
    if scale == SCALE_32B and validate_32b_checkpoint:
        from postdyn.think_32b_differential_validator import validate_checkpoint_tree

        report = validate_checkpoint_tree(
            root,
            trajectory or "rlvr",
            checkpoint,
            layers=layers,
            expected_setup_signature=setup_sig,
            require_publication=False,
        )
        if not report.ok:
            manifest["status"] = "failed"
            manifest["validation_errors"] = report.errors
            _atomic_write_json(_manifest_path(root, cfg.name, checkpoint), manifest)
            raise ValueError(f"32B checkpoint validation failed: {report.errors[0]}")
    _atomic_write_json(_manifest_path(root, cfg.name, checkpoint), manifest)
    print(f"  Done {cfg.name}/{checkpoint} in {elapsed:.1f}s")
    return manifest


def _validate_32b_tree_or_raise(
    root: Path,
    trajectory: str,
    checkpoints: list[str],
    layers: list[int],
    setup_sig: str,
    require_publications: bool = True,
) -> None:
    from postdyn.think_32b_differential_validator import validate_result_tree

    report = validate_result_tree(
        root,
        trajectory,
        checkpoints=checkpoints,
        layers=layers,
        expected_setup_signature=setup_sig,
        require_publications=require_publications,
    )
    if not report.ok:
        raise ValueError(f"32B publication validation failed: {report.errors[0]}")


def _validate_7b_tree_or_raise(root: Path, trajectory: str) -> None:
    from postdyn.think_sft_differential_validator import validate_result_tree

    report = validate_result_tree(root, trajectory)
    if not report.ok:
        raise ValueError(f"7B publication validation failed: {report.errors[0]}")


def finalize_stability(
    root: Path,
    scale: str,
    checkpoints: list[str],
    layers: list[int],
    concepts: list[str],
    setup_sig: str,
    model_key: str | None = None,
    revisions: dict[str, str] | None = None,
    final_root: Path | None = None,
    final_checkpoint: str | None = None,
    final_setup_sig: str | None = None,
    final_revision: str | None = None,
) -> dict[str, Any]:
    model_key = model_key or sft_model_key(scale)
    model_name = model_config(model_key).name
    out: dict[str, Any] = {
        "model": model_name,
        "checkpoint_order": list(checkpoints),
        "layers": {},
        "setup_signature": setup_sig,
        "layers_order": list(layers),
        "reference": checkpoints[0] if checkpoints else None,
        "final_main": {
            "checkpoint": final_checkpoint,
            "setup_signature": final_setup_sig,
            "revision": final_revision,
            "root": str(final_root) if final_root is not None else None,
        },
    }
    if final_root is None or final_checkpoint is None or final_setup_sig is None:
        raise ValueError("final main artifacts are required for residual emergence")
    final_loaded: dict[int, dict[str, Any]] = {}
    for layer in layers:
        by_ck_pos: dict[str, dict[str, Any]] = {}
        by_ck_neg: dict[str, dict[str, Any]] = {}
        loaded: dict[str, dict[str, Any]] = {}
        final_loaded[layer] = {}
        for ck in checkpoints:
            if not all(
                subspace_complete(
                    root,
                    model_name,
                    ck,
                    layer,
                    concept,
                    setup_sig,
                    revisions.get(ck) if revisions else ck,
                )
                for concept in concepts
            ):
                raise ValueError(
                    f"incomplete signed artifacts for checkpoint={ck}, layer={layer}"
                )
            by_ck_pos[ck] = {}
            by_ck_neg[ck] = {}
            for concept in concepts:
                signed = load_signed_subspace(root, model_name, ck, layer, concept)
                loaded.setdefault(ck, {})[concept] = signed
                by_ck_pos[ck][concept] = signed.to_positive()
                by_ck_neg[ck][concept] = signed.to_negative()
        for concept in concepts:
            if not subspace_complete(
                final_root,
                model_name,
                final_checkpoint,
                layer,
                concept,
                final_setup_sig,
                final_revision,
            ):
                raise ValueError(
                    f"incomplete final-main artifact for layer={layer}, concept={concept}"
                )
            final_loaded[layer][concept] = load_signed_subspace(
                final_root, model_name, final_checkpoint, layer, concept
            )
        residual_to_final: dict[str, Any] = {}
        for ck in checkpoints:
            for concept in concepts:
                current = loaded[ck][concept]
                final = final_loaded[layer][concept]
                residual_to_final.setdefault(ck, {})[concept] = (
                    residual_to_final_analysis(current, final)
                )
        reference_names = (
            "math_vs_code",
            "math_vs_instruction_following",
            "math_vs_general_reasoning",
        )
        reference_robustness: dict[str, Any] = {}
        for ck in checkpoints:
            baseline = loaded[ck].get("math_vs_wikitext")
            if baseline is None:
                continue
            reference_robustness[ck] = {}
            for concept in reference_names:
                comparison = loaded[ck].get(concept)
                if comparison is None:
                    continue
                reference_robustness[ck][concept] = {
                    "pos": {
                        "subsim": subspace_stability(baseline.u_pos, comparison.u_pos),
                        "k": min(baseline.k_pos, comparison.k_pos),
                    },
                    "neg": {
                        "subsim": subspace_stability(baseline.u_neg, comparison.u_neg),
                        "k": min(baseline.k_neg, comparison.k_neg),
                    },
                }
        histograms: dict[str, dict[str, Any]] = {}
        for concept in concepts:
            eigenvalue_values: list[float] = []
            for checkpoint_data in loaded.values():
                values = getattr(checkpoint_data[concept], "eigenvalues_signed", None)
                if isinstance(values, torch.Tensor):
                    eigenvalue_values.extend(float(value) for value in values.tolist())
            histograms[concept] = _shared_histogram_metadata(eigenvalue_values)
        out.setdefault("histogram", {})[str(layer)] = histograms
        for ck in checkpoints:
            metrics_path = _layer_metrics_path(root, model_name, ck, layer)
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics["histogram"] = histograms
                _atomic_write_json(metrics_path, metrics)
        out["layers"][str(layer)] = {
            "pos": compute_stability_trajectory(
                by_ck_pos, checkpoints, reference_checkpoint=checkpoints[0]
            ),
            "neg": compute_stability_trajectory(
                by_ck_neg, checkpoints, reference_checkpoint=checkpoints[0]
            ),
            "residual_to_final": residual_to_final,
            "reference_robustness": reference_robustness,
        }
    _atomic_write_json(root / METRICS_SUBDIR / "stability.json", out)
    return out


def build_summary(
    root: Path,
    scale: str,
    checkpoints: list[str],
    layers: list[int],
    setup_sig: str,
    model_key: str | None = None,
    fixed_points: dict[str, dict[str, Any]] | None = None,
    final_main: dict[str, Any] | None = None,
) -> dict[str, Any]:
    concepts = [n for n, _, _ in CONCEPT_PAIRS]
    model_key = model_key or sft_model_key(scale)
    model_name = model_config(model_key).name
    rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        for layer in layers:
            path = _layer_metrics_path(root, model_name, ck, layer)
            if not path.is_file():
                raise ValueError(f"incomplete summary for {ck} layer={layer}")
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("setup_signature") != setup_sig:
                raise ValueError(f"mismatched summary for {ck} layer={layer}")
            three = data.get("three_metrics", {})
            rows.append(
                {
                    "model": model_name,
                    "hf_id": model_config(model_key).hf_id,
                    "checkpoint": ck,
                    "layer": layer,
                    "retained_K": three.get("retained_K"),
                    "d_eff": three.get("d_eff"),
                    "core_metrics": _summary_core_metrics(data.get("concepts", {})),
                    "artifact_paths": {
                        concept: str(_u_paths(root, model_name, ck, layer, concept)[0])
                        for concept in concepts
                    },
                }
            )
    stability_data = json.loads(
        (root / METRICS_SUBDIR / "stability.json").read_text(encoding="utf-8")
    )
    summary = {
        "experiment": "think_sft_differential_subspace",
        "scale": scale,
        "dtype": DTYPE,
        "pairs": [list(p) for p in CONCEPT_PAIRS],
        "models": [model_config(model_key).hf_id],
        "model_keys": [model_key],
        "fixed_points": fixed_points or {},
        "final_main": final_main or {},
        "checkpoints": checkpoints,
        "layers": layers,
        "setup_signature": setup_sig,
        "concepts": concepts,
        "n_rows": len(rows),
        "rows": rows,
        "stability_path": str(root / METRICS_SUBDIR / "stability.json"),
        "residual_to_final_path": str(root / METRICS_SUBDIR / "stability.json"),
        "histogram": stability_data.get("histogram", {}),
        "figures_dir": str(root / FIGURES_SUBDIR),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(root / METRICS_SUBDIR / "summary.json", summary)
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--project-root", type=Path, default=None)
    p.add_argument("--scale", choices=[SCALE_7B, SCALE_32B], default=SCALE_7B)
    p.add_argument("--family", choices=[FAMILY_THINK], default=FAMILY_THINK)
    p.add_argument(
        "--trajectory",
        choices=("sft", "rlvr", "sft_lr_1e-4", "sft_lr_5e-5"),
        default=TRAJECTORY_SFT,
    )
    p.add_argument("--checkpoints", type=str, default=None)
    p.add_argument("--layers", type=str, default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument(
        "--extract-token-budget",
        type=int,
        default=8192,
        help="Max padded tokens per forward batch during extraction",
    )
    p.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="Comma-separated concept names to run (default: all concept pairs)",
    )
    p.add_argument(
        "--save-tensors",
        action="store_true",
        help="Also dump subspace tensors (.safetensors); default is JSON metrics only",
    )
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--keep-hf-cache", action="store_true")
    p.add_argument("--no-hf", action="store_true")
    p.add_argument("--allow-32b", action="store_true")
    p.add_argument(
        "--skip-7b-gate",
        action="store_true",
        help=(
            "Skip the canonical 7B preflight gate (for subset/verification runs; "
            "recorded in the run manifest)"
        ),
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dry-run", "--preflight-only", action="store_true")
    p.add_argument(
        "--fixed-points",
        type=str,
        default=None,
        help="Comma-separated 7B fixed points: base,dpo; unavailable on 32B",
    )
    p.add_argument(
        "--no-fixed-points",
        action="store_true",
        help="Opt out of the mandatory 7B base and DPO fixed points",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    validate_scale(args.scale, args.allow_32b)
    if args.concepts:
        apply_concept_filter(args.concepts)
    set_skip_7b_gate(args.skip_7b_gate)
    config = trajectory_config(args.family, args.scale, args.trajectory)
    model_key = config.model_key
    checkpoints = list(config.checkpoints)
    revisions = dict(config.revisions)
    layers = layers_for_scale(args.scale)
    n_samples = (
        args.samples if args.samples is not None else covariance_n_samples(args.scale)
    )
    root = (
        Path(args.output)
        if args.output
        else root_for_trajectory(
            args.family, args.scale, args.trajectory, project_root=args.project_root
        )
    )
    if args.quick:
        root = (
            Path(args.output)
            if args.output
            else root_for_trajectory(
                args.family,
                args.scale,
                args.trajectory,
                quick=True,
                project_root=args.project_root,
            )
        )
        checkpoints = checkpoints[:1]
        layers = layers[:2]
        n_samples = quick_sample_count(n_samples)
    if args.checkpoints:
        requested = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
        validate_selection(
            requested, layers, list(config.checkpoints), layers_for_scale(args.scale)
        )
        wanted = set(requested)
        checkpoints = [c for c in checkpoints if c in wanted]
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    validate_selection(
        checkpoints,
        layers,
        list(config.checkpoints),
        layers_for_scale(args.scale),
    )
    if not checkpoints or not layers or n_samples <= 0:
        raise ValueError("checkpoints, layers, and samples must be positive")
    if args.no_fixed_points:
        requested_fixed_points: tuple[str, ...] = ()
    elif args.fixed_points is None and args.scale == SCALE_7B:
        requested_fixed_points = ("base", "dpo")
    else:
        requested_fixed_points = tuple(
            value.strip()
            for value in (args.fixed_points or "").split(",")
            if value.strip()
        )
    available_fixed_points = fixed_point_configs(args.scale)
    unknown_fixed_points = set(requested_fixed_points) - set(available_fixed_points)
    if unknown_fixed_points:
        if args.scale == SCALE_32B:
            raise ValueError(
                "32B fixed points are unavailable; no configured 32B base/DPO IDs or revisions"
            )
        raise ValueError(f"unknown fixed point(s): {sorted(unknown_fixed_points)}")
    if not args.dry_run:
        revisions = resolve_model_revisions(model_key, revisions)

    is_canonical_root = (
        root.resolve()
        == root_for_trajectory(
            args.family,
            args.scale,
            args.trajectory,
            quick=args.quick,
            project_root=args.project_root,
        ).resolve()
    )
    validate_extraction_root_not_other_trajectory(
        root,
        family=args.family,
        scale=args.scale,
        trajectory=args.trajectory,
        project_root=args.project_root,
        quick=args.quick,
    )
    validate_root_ownership(
        root,
        family=args.family,
        scale=args.scale,
        trajectory=args.trajectory,
        checkpoints=checkpoints,
        revisions=revisions,
        model_key=model_key,
        purpose="extraction",
        canonical=is_canonical_root,
    )

    if args.scale == SCALE_32B:
        if not args.quick:
            validate_extraction_protocol(
                extraction_protocol_payload(
                    n_samples=n_samples,
                    tau=args.tau,
                    max_seq_len=args.max_seq_len,
                    use_chat_template=USE_CHAT_TEMPLATE,
                    extraction_contract=EXTRACTION_CONTRACT,
                    dtype=DTYPE,
                    signed=True,
                ),
                scale=args.scale,
            )
        _require_canonical_7b(project_root=args.project_root)

    if not is_canonical_root:
        claim_root_ownership(
            root,
            family=args.family,
            scale=args.scale,
            trajectory=args.trajectory,
            checkpoints=checkpoints,
            revisions=revisions,
            model_key=model_key,
            purpose="extraction",
        )
    print(f"Output root: {root}")
    print(f"Scale: {args.scale} dtype={DTYPE}")
    print(f"Model: {model_config(model_key).hf_id}")
    print(f"Checkpoints ({len(checkpoints)}): {checkpoints}")
    print(f"Layers ({len(layers)}): {layers}")
    print(f"Samples/domain: {n_samples}")
    if requested_fixed_points:
        print(f"Fixed points: {list(requested_fixed_points)}")
    mem = gpu_memory_gib()
    if mem is not None:
        print(f"GPU memory: {mem:.1f} GiB")
    if args.dry_run:
        print(
            "Dry run: selection and ownership validated; no datasets or models loaded."
        )
        return 0

    domain_prompts = prepare_domain_prompts(
        root,
        n_samples=n_samples,
        seed=args.seed,
        allow_hf=not args.no_hf,
        max_seq_len=args.max_seq_len,
        use_chat_template=USE_CHAT_TEMPLATE,
    )
    setup_sig = _build_setup_sig(
        scale=args.scale,
        model_key=model_key,
        checkpoints=checkpoints,
        layers=layers,
        n_samples=n_samples,
        tau=args.tau,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        domain_prompts=domain_prompts,
        trajectory=args.trajectory,
        revisions=revisions,
    )
    runtime_provenance: dict[str, Any] = {}
    model_loader = build_model_loader(
        args.scale,
        runtime_provenance=runtime_provenance if args.scale == SCALE_32B else None,
        project_root=args.project_root,
    )
    for ck in checkpoints:
        _run_checkpoint(
            model_key,
            ck,
            revisions[ck],
            root=root,
            scale=args.scale,
            layers=layers,
            n_samples=n_samples,
            max_seq_len=args.max_seq_len,
            tau=args.tau,
            domain_prompts=domain_prompts,
            setup_sig=setup_sig,
            model_loader=model_loader,
            runtime_provenance=runtime_provenance if args.scale == SCALE_32B else None,
            canonical_protocol=not args.quick,
            trajectory=args.trajectory,
            project_root=args.project_root,
            save_tensors=args.save_tensors,
            token_budget=args.extract_token_budget,
        )
        if not args.keep_hf_cache:
            from postdyn.concept_dynamics import _clean_hf_cache

            try:
                _clean_hf_cache(model_config(model_key).hf_id)
            except Exception as exc:
                print(f"  cache clean warning: {exc}")

    fixed_point_records: dict[str, dict[str, Any]] = {}
    if args.scale == SCALE_32B:
        fixed_point_records = {
            label: {
                "status": "unavailable",
                "reason": "no configured 32B fixed-point model ID or immutable revision",
            }
            for label in ("base", "dpo")
        }
    elif not requested_fixed_points:
        fixed_point_records = {
            label: {"status": "opted_out"} for label in ("base", "dpo")
        }
    for label in requested_fixed_points:
        fixed_model_key, fixed_revision = available_fixed_points[label]
        fixed_revision = resolve_model_revision(
            model_config(fixed_model_key).hf_id, fixed_revision
        )
        fixed_setup_sig = _build_setup_sig(
            scale=args.scale,
            model_key=fixed_model_key,
            checkpoints=[label],
            layers=layers,
            n_samples=n_samples,
            tau=args.tau,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            domain_prompts=domain_prompts,
            trajectory=f"fixed:{label}",
            revisions={label: fixed_revision},
        )
        record = _run_checkpoint(
            fixed_model_key,
            label,
            fixed_revision,
            root=root,
            scale=args.scale,
            layers=layers,
            n_samples=n_samples,
            max_seq_len=args.max_seq_len,
            tau=args.tau,
            domain_prompts=domain_prompts,
            setup_sig=fixed_setup_sig,
            model_loader=model_loader,
            canonical_protocol=not args.quick,
            trajectory=f"fixed:{label}",
            project_root=args.project_root,
            save_tensors=args.save_tensors,
            token_budget=args.extract_token_budget,
        )
        fixed_point_records[label] = {
            "model_key": fixed_model_key,
            "model": model_config(fixed_model_key).hf_id,
            "revision": fixed_revision,
            "setup_signature": fixed_setup_sig,
            "status": record.get("status", "ok"),
        }

    concepts = [n for n, _, _ in CONCEPT_PAIRS]
    final_label = "sft_main" if args.trajectory.startswith("sft") else "rlvr_main"
    final_root = root / "final_points"
    final_revision = resolve_model_revision(
        model_config(model_key).hf_id, model_config(model_key).revision
    )
    final_setup_sig = _build_setup_sig(
        scale=args.scale,
        model_key=model_key,
        checkpoints=[final_label],
        layers=layers,
        n_samples=n_samples,
        tau=args.tau,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        domain_prompts=domain_prompts,
        trajectory=f"{args.trajectory}:main",
        revisions={final_label: final_revision},
    )
    final_record = _run_checkpoint(
        model_key,
        final_label,
        final_revision,
        root=final_root,
        scale=args.scale,
        layers=layers,
        n_samples=n_samples,
        max_seq_len=args.max_seq_len,
        tau=args.tau,
        domain_prompts=domain_prompts,
        setup_sig=final_setup_sig,
        model_loader=model_loader,
        runtime_provenance=runtime_provenance if args.scale == SCALE_32B else None,
        canonical_protocol=not args.quick,
        trajectory=f"{args.trajectory}:main",
        project_root=args.project_root,
        validate_32b_checkpoint=False,
        save_tensors=args.save_tensors,
        token_budget=args.extract_token_budget,
    )
    final_main_record = {
        "model_key": model_key,
        "model": model_config(model_key).hf_id,
        "checkpoint": final_label,
        "revision": final_revision,
        "setup_signature": final_setup_sig,
        "root": str(final_root),
        "status": final_record.get("status", "ok"),
        "artifact_paths": {
            concept: {
                str(layer): str(
                    _u_paths(
                        final_root,
                        model_config(model_key).name,
                        final_label,
                        layer,
                        concept,
                    )[0]
                )
                for layer in layers
            }
            for concept in concepts
        },
    }

    if args.scale == SCALE_32B and not args.quick:
        _validate_32b_tree_or_raise(
            root,
            args.trajectory,
            checkpoints,
            layers,
            setup_sig,
            require_publications=False,
        )
    stability = finalize_stability(
        root,
        args.scale,
        checkpoints,
        layers,
        concepts,
        setup_sig,
        model_key,
        revisions,
        final_root=final_root,
        final_checkpoint=final_label,
        final_setup_sig=final_setup_sig,
        final_revision=final_revision,
    )
    if args.scale == SCALE_32B and not args.quick:
        _validate_32b_tree_or_raise(
            root,
            args.trajectory,
            checkpoints,
            layers,
            setup_sig,
            require_publications=False,
        )
    summary = build_summary(
        root,
        args.scale,
        checkpoints,
        layers,
        setup_sig,
        model_key,
        fixed_points=fixed_point_records,
        final_main=final_main_record,
    )
    if args.scale == SCALE_7B and not args.quick:
        _validate_7b_tree_or_raise(root, args.trajectory)
    if args.scale == SCALE_32B and not args.quick:
        from postdyn.think_32b_differential_validator import (
            validate_full_canonical_publication,
        )

        report = validate_full_canonical_publication(
            root, args.trajectory, expected_setup_signature=setup_sig
        )
        if not report.ok:
            raise ValueError(f"32B publication validation failed: {report.errors[0]}")
    print("\n===== SIGNED METRICS READY =====")
    print(f"summary: {root / METRICS_SUBDIR / 'summary.json'}")
    print(f"stability: {root / METRICS_SUBDIR / 'stability.json'}")
    print(f"rows: {summary['n_rows']}")
    if layers and checkpoints:
        sample = (
            stability["layers"][str(layers[0])]["pos"]
            .get("vs_reference", {})
            .get(concepts[0], [])
        )
        print(f"SubSim vs first, layer {layers[0]}: {sample[:2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
