#!/usr/bin/env python3
"""Extract differential subspaces for Dolci Math vs Dolci General.

Pair (PostDyn.tex):
  * math_vs_text  — ΔΣ = Σ_math − Σ_text

Schedule: 10 RL-Zero-Math checkpoints loaded at immutable revisions.
Layers: 10 slide-formula indices. Samples: 1,000 raw prompts / domain.
Extraction uses no chat template; a 2,048-token, no-truncation preflight runs
before activation extraction. Stability includes all unordered checkpoint-pair
SubSim values (45 pairs per concept and layer), alongside consecutive/reference
summaries.

Writes under ``logs/math_differential_subspace_setup_raw_prompt/``::

    U/{model}/{checkpoint}/layer_{L}/{concept}.safetensors
    U/{model}/{checkpoint}/layer_{L}/{concept}.json
    metrics/{model}/{checkpoint}/layer_{L}.json
    metrics/stability.json
    metrics/summary.json
    prompts/{domain}.json
    manifests/{model}__{checkpoint}.json

Usage::

    uv run python scripts/run_math_differential_subspace.py
    uv run python scripts/run_math_differential_subspace.py --quick
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from safetensors.torch import load_file, save_file


from postdyn.concept_dynamics import (
    _clean_hf_cache,
    _load_model_and_tokenizer,
)
from postdyn.differential_subspace import (
    DifferentialSubspace,
    compute_differential_subspace,
    compute_pair_metrics_at_checkpoint,
    compute_stability_trajectory,
    subspace_to_serializable,
)
from postdyn.domain_datasets import (
    DOLCI_HF_IDS,
    DOLCI_HF_REVISIONS,
    load_dolci_domain_prompts,
)
from postdyn.math_differential_experiment import (
    CONCEPT_PAIRS,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_LAYERS,
    EXTRACTION_CONTRACT,
    MANIFESTS_SUBDIR,
    MAX_SEQ_LEN,
    METRICS_SUBDIR,
    N_SAMPLES,
    PROMPTS_SUBDIR,
    RESULTS_ROOT,
    RESULTS_ROOT_QUICK,
    SAMPLE_SEED,
    TAU,
    TARGET_MODEL_KEY,
    U_SUBDIR,
    USE_CHAT_TEMPLATE,
    model_for_checkpoint,
    revision_for_checkpoint,
)

ModelLoader = Callable[[Any, Optional[str]], tuple[Any, Any]]


def _concrete_device(device: Any) -> Any | None:
    if device is None or str(device) == "meta":
        return None
    return device


def _input_device(model: Any) -> Any:
    if hasattr(model, "get_input_embeddings"):
        weight = getattr(model.get_input_embeddings(), "weight", None)
        device = _concrete_device(getattr(weight, "device", None))
        if device is not None:
            return device

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for name in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            device = _concrete_device(device_map.get(name))
            if device is not None:
                return device

    if hasattr(model, "parameters"):
        try:
            device = _concrete_device(next(model.parameters()).device)
            if device is not None:
                return device
        except StopIteration:
            pass

    device = _concrete_device(getattr(model, "device", None))
    if device is not None:
        return device
    raise ValueError("no concrete execution device is available for model inputs")


def quick_sample_count(requested: int) -> int:
    return min(16, requested)


def validate_selection(
    checkpoints: list[str], layers: list[int], n_samples: int, limit: int | None
) -> None:
    if not checkpoints:
        raise ValueError("checkpoint selection is empty")
    if not layers:
        raise ValueError("layer selection is empty")
    if n_samples <= 0:
        raise ValueError(f"samples must be positive, got {n_samples}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if len(set(checkpoints)) != len(checkpoints) or any(
        checkpoint not in EXPERIMENT_CHECKPOINTS for checkpoint in checkpoints
    ):
        raise ValueError("checkpoint selection contains unknown or duplicate selection")
    if len(set(layers)) != len(layers) or any(
        layer not in EXPERIMENT_LAYERS for layer in layers
    ):
        raise ValueError("layer selection contains unknown or duplicate selection")


def preflight_tokenizer_prompts(
    tokenizer: Any, prompts: list[str], max_seq_len: int
) -> list[int]:
    lengths: list[int] = []
    for index, prompt in enumerate(prompts):
        encoded = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=False,
            padding=True,
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            length = int(encoded["input_ids"].shape[-1])
        else:
            length = int(attention_mask[0].sum().item())
        lengths.append(length)
        if length > max_seq_len:
            raise ValueError(
                f"Prompt {index} has {length} tokens, exceeds max_seq_len={max_seq_len}"
            )
    return lengths


def extract_raw_layer_activations(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layers: list[int],
    *,
    token_budget: int = 8192,
    lengths: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    device = _input_device(model)
    total = len(texts)
    features: dict[int, list[torch.Tensor | None]] = {
        layer: [None] * total for layer in layers
    }
    if lengths is None:
        lengths = _token_lengths(tokenizer, texts)
    order = sorted(range(total), key=lambda index: lengths[index])
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
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        attention_mask = inputs.get("attention_mask")
        for row, index in enumerate(batch_indices):
            if attention_mask is None:
                last_index = inputs["input_ids"].shape[1] - 1
            else:
                last_index = int(attention_mask[row].sum().item()) - 1
            for layer in layers:
                features[layer][index] = (
                    outputs.hidden_states[layer + 1][row, last_index, :]
                    .detach()
                    .cpu()
                    .float()
                )
    extracted: dict[int, torch.Tensor] = {}
    for layer, values in features.items():
        present = [value for value in values if value is not None]
        if len(present) != total:
            raise ValueError(f"layer {layer} is missing extracted activations")
        extracted[layer] = torch.stack(present)
    return extracted


def _token_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    lengths: list[int] = []
    for text in texts:
        encoded = tokenizer([text], return_tensors="pt", truncation=False)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None and torch.is_tensor(attention_mask):
            length = int(attention_mask[0].sum().item())
        else:
            length = int(encoded["input_ids"].shape[-1])
        lengths.append(length)
    return lengths


def load_preflight_tokenizer(checkpoint: str) -> Any:
    from transformers import AutoTokenizer

    cfg = model_for_checkpoint(checkpoint)
    return AutoTokenizer.from_pretrained(
        cfg.hf_id, revision=revision_for_checkpoint(checkpoint)
    )


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def _u_paths(
    root: Path, model_name: str, checkpoint: str, layer: int, concept: str
) -> tuple[Path, Path]:
    base = root / U_SUBDIR / model_name / checkpoint / f"layer_{layer}"
    return base / f"{concept}.safetensors", base / f"{concept}.json"


def _layer_metrics_path(
    root: Path, model_name: str, checkpoint: str, layer: int
) -> Path:
    return root / METRICS_SUBDIR / model_name / checkpoint / f"layer_{layer}.json"


def _manifest_path(root: Path, model_name: str, checkpoint: str) -> Path:
    return root / MANIFESTS_SUBDIR / f"{model_name}__{checkpoint}.json"


def _prompt_fingerprint(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def setup_signature(
    *,
    pairs: list[tuple[str, str, str]],
    checkpoints: list[str],
    layers: list[int],
    model_id: str,
    checkpoint_revisions: dict[str, str],
    dataset_sources: dict[str, dict[str, str]],
    prompt_fingerprints: dict[str, str],
    n_samples: int,
    tau: float,
    max_seq_len: int,
    use_chat_template: bool,
    seed: int,
    extraction_contract: str,
) -> str:
    payload = {
        "pairs": pairs,
        "checkpoints": checkpoints,
        "layers": layers,
        "model_id": model_id,
        "checkpoint_revisions": checkpoint_revisions,
        "dataset_sources": dataset_sources,
        "prompt_fingerprints": list(prompt_fingerprints.items()),
        "n_samples": n_samples,
        "tau": tau,
        "max_seq_len": max_seq_len,
        "use_chat_template": use_chat_template,
        "seed": seed,
        "extraction_contract": extraction_contract,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_subspace(
    root: Path,
    model_name: str,
    checkpoint: str,
    layer: int,
    sub: DifferentialSubspace,
    setup_sig: str | None = None,
    revision: str | None = None,
) -> None:
    st_path, js_path = _u_paths(root, model_name, checkpoint, layer, sub.concept)
    st_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "U": sub.u.contiguous().float(),
        "eigenvalues_pos": sub.eigenvalues_pos.contiguous().float(),
    }
    tmp_st_path = st_path.with_suffix(st_path.suffix + ".tmp")
    save_file(payload, str(tmp_st_path))
    os.replace(tmp_st_path, st_path)
    meta = subspace_to_serializable(sub)
    meta.update(
        {
            "model": model_name,
            "checkpoint": checkpoint,
            "layer": layer,
            "u_shape": list(sub.u.shape),
            "eigenvalues_pos_shape": list(sub.eigenvalues_pos.shape),
            "setup_signature": setup_sig,
            "revision": revision,
        }
    )
    _atomic_write_json(js_path, meta)


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
    except (OSError, json.JSONDecodeError):
        return False
    except (KeyError, RuntimeError, ValueError):
        return False
    if not isinstance(meta, dict):
        return False
    try:
        if setup_sig is not None and meta.get("setup_signature") != setup_sig:
            return False
        if expected_revision is not None and meta.get("revision") != expected_revision:
            return False
        if meta.get("model") != model_name or meta.get("checkpoint") != checkpoint:
            return False
        if int(meta.get("layer", -1)) != layer or meta.get("concept") != concept:
            return False
        u = tensors.get("U")
        eigenvalues = tensors.get("eigenvalues_pos")
        shape = meta.get("u_shape")
        return (
            set(tensors) == {"U", "eigenvalues_pos"}
            and u is not None
            and eigenvalues is not None
            and isinstance(shape, list)
            and list(u.shape) == shape
            and list(eigenvalues.shape) == meta.get("eigenvalues_pos_shape")
            and int(meta.get("k", -1)) == int(u.shape[1])
            and int(eigenvalues.numel()) >= int(meta.get("k", -1))
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def checkpoint_complete(
    root: Path,
    model_name: str,
    checkpoint: str,
    layers: list[int],
    concepts: list[str],
    setup_sig: str | None = None,
    expected_revision: str | None = None,
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
    if not isinstance(manifest, dict):
        return False
    return (
        manifest.get("status") == "ok"
        and manifest.get("model") == model_name
        and manifest.get("checkpoint") == checkpoint
        and manifest.get("revision") == expected_revision
        and manifest.get("setup_signature") == setup_sig
    )


def prepare_domain_prompts(
    root: Path,
    n_samples: int,
    seed: int,
    allow_hf: bool = True,
    max_seq_len: int = MAX_SEQ_LEN,
    use_chat_template: bool = USE_CHAT_TEMPLATE,
) -> dict[str, list[str]]:
    domains = sorted({d for _, c, r in CONCEPT_PAIRS for d in (c, r)})
    out: dict[str, list[str]] = {}
    prompt_dir = root / PROMPTS_SUBDIR
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for domain in domains:
        cache_path = prompt_dir / f"{domain}.json"
        expected_source = {
            "kind": "dolci",
            "hf_id": DOLCI_HF_IDS[domain],
            "revision": DOLCI_HF_REVISIONS[domain],
        }
        if cache_path.is_file():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("cache root must be an object")
                prompts = data["prompts"]
                if not isinstance(prompts, list) or not all(
                    isinstance(prompt, str) for prompt in prompts
                ):
                    raise ValueError("prompts must be a list of strings")
            except (
                OSError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Malformed prompt cache for {domain}: {cache_path}"
                ) from exc
            if (
                data.get("n_samples") == n_samples
                and data.get("seed") == seed
                and data.get("source") == expected_source
                and data.get("prompt_fingerprint") == _prompt_fingerprint(prompts)
                and len(prompts) == n_samples
                and data.get("use_chat_template") == use_chat_template
                and data.get("max_seq_len") == max_seq_len
                and data.get("extraction_contract") == EXTRACTION_CONTRACT
            ):
                out[domain] = prompts[:n_samples]
                continue
            raise ValueError(f"Incompatible prompt cache for {domain}: {cache_path}")
        if not allow_hf:
            raise ValueError("This experiment requires the strict Dolci sources")
        prompts = load_dolci_domain_prompts(domain, n_samples=n_samples, seed=seed)
        out[domain] = prompts
        _atomic_write_json(
            cache_path,
            {
                "domain": domain,
                "n": len(prompts),
                "n_samples": n_samples,
                "seed": seed,
                "source": expected_source,
                "prompt_fingerprint": _prompt_fingerprint(prompts),
                "use_chat_template": use_chat_template,
                "max_seq_len": max_seq_len,
                "extraction_contract": EXTRACTION_CONTRACT,
                "prompts": prompts,
            },
        )
    return out


def run_checkpoint(
    checkpoint: str,
    *,
    root: Path,
    layers: list[int],
    n_samples: int,
    max_seq_len: int,
    tau: float,
    use_chat_template: bool,
    domain_prompts: dict[str, list[str]],
    keep_hf_cache: bool,
    seed: int = SAMPLE_SEED,
    requested_checkpoints: list[str] | None = None,
    model_loader: ModelLoader = _load_model_and_tokenizer,
) -> dict[str, Any]:
    concepts = [name for name, _, _ in CONCEPT_PAIRS]
    model_cfg = model_for_checkpoint(checkpoint)
    model_name = model_cfg.name
    revision = revision_for_checkpoint(checkpoint)
    if use_chat_template:
        raise ValueError("The raw prompt protocol does not use a chat template")
    ordered_domains = list(
        dict.fromkeys(
            domain
            for _, concept_domain, ref_domain in CONCEPT_PAIRS
            for domain in (concept_domain, ref_domain)
        )
    )
    dataset_sources = {
        domain: {
            "id": DOLCI_HF_IDS[domain],
            "revision": DOLCI_HF_REVISIONS[domain],
        }
        for domain in ordered_domains
    }
    prompt_fingerprints = {
        domain: _prompt_fingerprint(domain_prompts[domain][:n_samples])
        for domain in ordered_domains
    }
    requested = list(requested_checkpoints or EXPERIMENT_CHECKPOINTS)
    setup_sig = setup_signature(
        pairs=list(CONCEPT_PAIRS),
        checkpoints=requested,
        layers=layers,
        model_id=model_cfg.hf_id,
        checkpoint_revisions={ck: revision_for_checkpoint(ck) for ck in requested},
        dataset_sources=dataset_sources,
        prompt_fingerprints=prompt_fingerprints,
        n_samples=n_samples,
        tau=tau,
        max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
        seed=seed,
        extraction_contract=EXTRACTION_CONTRACT,
    )

    if checkpoint_complete(
        root, model_name, checkpoint, layers, concepts, setup_sig, revision
    ):
        print(f"[skip] {model_name}/{checkpoint} already complete")
        return {
            "model": model_name,
            "checkpoint": checkpoint,
            "status": "skipped",
        }

    t0 = time.time()
    print(f"\n{'=' * 60}")
    print(f"Differential U: {model_name} / {checkpoint} (rev={revision})")
    print(f"Layers={layers} n_samples={n_samples} tau={tau}")
    print(f"{'=' * 60}")

    # Extract unique domain activations once per checkpoint.
    needed_domains = sorted({d for _, c, r in CONCEPT_PAIRS for d in (c, r)})
    model, tokenizer = model_loader(model_cfg, revision)
    domain_acts: dict[str, dict[int, torch.Tensor]] = {}
    try:
        for domain in needed_domains:
            texts = domain_prompts[domain][:n_samples]
            print(f"  Extracting domain={domain} n={len(texts)} ...")
            preflight_tokenizer_prompts(tokenizer, texts, max_seq_len)
            domain_acts[domain] = extract_raw_layer_activations(
                model, tokenizer, texts, layers
            )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not keep_hf_cache and model_cfg.name != "olmo3-base":
            try:
                _clean_hf_cache(model_cfg.hf_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  cache clean warning: {exc}")

    layer_results: dict[str, Any] = {}
    subspaces_by_layer: dict[int, dict[str, DifferentialSubspace]] = {}

    for layer in layers:
        subs: dict[str, DifferentialSubspace] = {}
        for concept_name, c_dom, r_dom in CONCEPT_PAIRS:
            if subspace_complete(
                root, model_name, checkpoint, layer, concept_name, setup_sig
            ):
                # Still recompute from acts if we have them — simpler to always recompute
                # when we already extracted activations this run.
                pass
            sub = compute_differential_subspace(
                domain_acts[c_dom][layer],
                domain_acts[r_dom][layer],
                concept=concept_name,
                tau=tau,
            )
            save_subspace(root, model_name, checkpoint, layer, sub, setup_sig, revision)
            subs[concept_name] = sub
            print(
                f"  layer={layer} {concept_name}: K={sub.k} "
                f"S̃={sub.geometry_strength:.4f} d_eff={sub.d_eff:.3f}"
            )

        metrics = compute_pair_metrics_at_checkpoint(subs)
        metrics_path = _layer_metrics_path(root, model_name, checkpoint, layer)
        payload = {
            "model": model_name,
            "checkpoint": checkpoint,
            "layer": layer,
            "tau": tau,
            "n_samples": n_samples,
            "revision": revision,
            "setup_signature": setup_sig,
            "concepts": {
                name: subspace_to_serializable(sub) for name, sub in subs.items()
            },
            "metrics": metrics,
            # Explicit five-metric block for easy reading.
            "five_metrics": {
                "1_retained_subspace_dimension": {
                    name: sub.k for name, sub in subs.items()
                },
                "2_subspace_stability": "see metrics/stability.json (trajectory-level)",
                "3_inter_subspace_relation": metrics["inter_subspace_relation"],
                "4_geometry_strength": {
                    name: sub.geometry_strength for name, sub in subs.items()
                },
                "5_effective_dimensionality": {
                    name: sub.d_eff for name, sub in subs.items()
                },
            },
        }
        _atomic_write_json(metrics_path, payload)
        layer_results[str(layer)] = payload["five_metrics"]
        subspaces_by_layer[layer] = subs

    elapsed = time.time() - t0
    manifest = {
        "model": model_name,
        "checkpoint": checkpoint,
        "revision": revision,
        "layers": layers,
        "concepts": concepts,
        "n_samples": n_samples,
        "tau": tau,
        "setup_signature": setup_sig,
        "elapsed_seconds": round(elapsed, 2),
        "status": "ok",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "layer_five_metrics": layer_results,
    }
    _atomic_write_json(_manifest_path(root, model_name, checkpoint), manifest)
    print(f"  Done {model_name}/{checkpoint} in {elapsed:.1f}s")
    return manifest


def load_saved_subspace(
    root: Path,
    model_name: str,
    checkpoint: str,
    layer: int,
    concept: str,
) -> DifferentialSubspace:
    from safetensors.torch import load_file

    st_path, js_path = _u_paths(root, model_name, checkpoint, layer, concept)
    tensors = load_file(str(st_path))
    meta = json.loads(js_path.read_text(encoding="utf-8"))
    return DifferentialSubspace(
        concept=concept,
        u=tensors["U"],
        eigenvalues_pos=tensors["eigenvalues_pos"],
        k=int(meta["k"]),
        tau=float(meta["tau"]),
        n_concept=int(meta["n_concept"]),
        n_ref=int(meta["n_ref"]),
        d_model=int(meta["d_model"]),
        tr_concept=float(meta["tr_concept"]),
        tr_ref=float(meta["tr_ref"]),
        geometry_strength=float(meta["geometry_strength"]),
        d_eff=float(meta["d_eff"]),
    )


def finalize_stability(
    root: Path,
    checkpoints: list[str],
    layers: list[int],
    concepts: list[str],
    setup_sig: str | None = None,
) -> dict[str, Any]:
    """Aggregate SubSim across the full trajectory for every layer/concept."""
    stability_out: dict[str, Any] = {"layers": {}}
    for layer in layers:
        by_ck: dict[str, dict[str, DifferentialSubspace]] = {}
        ck_order: list[str] = []
        for ck in checkpoints:
            model_name = model_for_checkpoint(ck).name
            if not all(
                subspace_complete(
                    root,
                    model_name,
                    ck,
                    layer,
                    c,
                    setup_sig,
                    revision_for_checkpoint(ck) if setup_sig is not None else None,
                )
                for c in concepts
            ):
                raise ValueError(
                    f"incomplete stability artifacts for checkpoint={ck}, layer={layer}"
                )
            by_ck[ck] = {
                c: load_saved_subspace(root, model_name, ck, layer, c) for c in concepts
            }
            ck_order.append(ck)
        stab = compute_stability_trajectory(
            by_ck, ck_order, reference_checkpoint=ck_order[0] if ck_order else None
        )
        stability_out["layers"][str(layer)] = stab
    stability_out["setup_signature"] = setup_sig
    stability_out["checkpoint_order"] = list(checkpoints)
    stability_out["layers_order"] = list(layers)
    path = root / METRICS_SUBDIR / "stability.json"
    _atomic_write_json(path, stability_out)
    return stability_out


def build_summary(
    root: Path, checkpoints: list[str], layers: list[int], setup_sig: str
) -> dict[str, Any]:
    """Collect five-metric snapshots from every completed layer file."""
    concepts = [n for n, _, _ in CONCEPT_PAIRS]
    rows: list[dict[str, Any]] = []
    for ck in checkpoints:
        model_name = model_for_checkpoint(ck).name
        for layer in layers:
            path = _layer_metrics_path(root, model_name, ck, layer)
            if not path.is_file():
                raise ValueError(
                    f"incomplete summary artifacts for checkpoint={ck}, layer={layer}"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                data.get("setup_signature") != setup_sig
                or data.get("model") != model_name
                or data.get("checkpoint") != ck
                or data.get("layer") != layer
            ):
                raise ValueError(
                    f"mismatched summary artifact for checkpoint={ck}, layer={layer}"
                )
            five = data.get("five_metrics", {})
            rows.append(
                {
                    "model": model_name,
                    "checkpoint": ck,
                    "layer": layer,
                    "retained_K": five.get("1_retained_subspace_dimension"),
                    "inter_subspace_G": five.get("3_inter_subspace_relation"),
                    "geometry_strength": five.get("4_geometry_strength"),
                    "d_eff": five.get("5_effective_dimensionality"),
                }
            )
    summary = {
        "experiment": "math_differential_subspace",
        "target_model": TARGET_MODEL_KEY,
        "pairs": [list(p) for p in CONCEPT_PAIRS],
        "checkpoints": checkpoints,
        "checkpoint_order": list(checkpoints),
        "layers": layers,
        "setup_signature": setup_sig,
        "concepts": concepts,
        "n_rows": len(rows),
        "rows": rows,
        "stability_path": str(root / METRICS_SUBDIR / "stability.json"),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(root / METRICS_SUBDIR / "summary.json", summary)
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, default=None, help="Output root directory")
    p.add_argument(
        "--checkpoints", type=str, default=None, help="Comma-separated subset"
    )
    p.add_argument("--limit", type=int, default=None, help="First N checkpoints only")
    p.add_argument(
        "--layers", type=str, default=None, help="Comma-separated layer indices"
    )
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--tau", type=float, default=TAU)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--keep-hf-cache", action="store_true")
    p.add_argument("--no-hf", action="store_true", help="Do not download Dolci from HF")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 1 checkpoint, 2 layers, 16 samples",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.quick:
        root = Path(args.output) if args.output else RESULTS_ROOT_QUICK
        checkpoints = [EXPERIMENT_CHECKPOINTS[0]]
        layers = EXPERIMENT_LAYERS[:2]
        n_samples = quick_sample_count(args.samples)
    else:
        root = Path(args.output) if args.output else RESULTS_ROOT
        checkpoints = list(EXPERIMENT_CHECKPOINTS)
        layers = list(EXPERIMENT_LAYERS)
        n_samples = args.samples

    if args.checkpoints:
        requested = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
        validate_selection(requested, layers, n_samples, args.limit)
        wanted = set(requested)
        checkpoints = [c for c in checkpoints if c in wanted]
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    validate_selection(checkpoints, layers, n_samples, args.limit)

    root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {root}")
    print(f"Checkpoints ({len(checkpoints)}): {checkpoints}")
    print(f"Layers ({len(layers)}): {layers}")
    print(f"Samples/domain: {n_samples}")

    domain_prompts = prepare_domain_prompts(
        root,
        n_samples=n_samples,
        seed=args.seed,
        allow_hf=not args.no_hf,
        max_seq_len=args.max_seq_len,
        use_chat_template=USE_CHAT_TEMPLATE,
    )
    for d, texts in domain_prompts.items():
        print(f"  domain {d}: {len(texts)} prompts")

    if args.preflight_only:
        tokenizer = load_preflight_tokenizer(checkpoints[0])
        for domain, prompts in domain_prompts.items():
            lengths = preflight_tokenizer_prompts(tokenizer, prompts, args.max_seq_len)
            print(f"  preflight {domain}: max_tokens={max(lengths, default=0)}")
        return 0

    manifests: list[dict[str, Any]] = []
    for ck in checkpoints:
        man = run_checkpoint(
            ck,
            root=root,
            layers=layers,
            n_samples=n_samples,
            max_seq_len=args.max_seq_len,
            tau=args.tau,
            use_chat_template=USE_CHAT_TEMPLATE,
            domain_prompts=domain_prompts,
            keep_hf_cache=args.keep_hf_cache,
            seed=args.seed,
            requested_checkpoints=checkpoints,
        )
        manifests.append(man)
        # Free HF cache between heavy RL checkpoints (not base).
        if not args.keep_hf_cache:
            cfg = model_for_checkpoint(ck)
            try:
                _clean_hf_cache(cfg.hf_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  cache clean warning: {exc}")

    concepts = [n for n, _, _ in CONCEPT_PAIRS]
    ordered_domains = list(
        dict.fromkeys(
            domain
            for _, concept_domain, ref_domain in CONCEPT_PAIRS
            for domain in (concept_domain, ref_domain)
        )
    )
    dataset_sources = {
        domain: {"id": DOLCI_HF_IDS[domain], "revision": DOLCI_HF_REVISIONS[domain]}
        for domain in ordered_domains
    }
    prompt_fingerprints = {
        domain: _prompt_fingerprint(domain_prompts[domain])
        for domain in ordered_domains
    }
    setup_sig = setup_signature(
        pairs=list(CONCEPT_PAIRS),
        checkpoints=checkpoints,
        layers=layers,
        model_id=model_for_checkpoint(checkpoints[0]).hf_id,
        checkpoint_revisions={ck: revision_for_checkpoint(ck) for ck in checkpoints},
        dataset_sources=dataset_sources,
        prompt_fingerprints=prompt_fingerprints,
        n_samples=n_samples,
        tau=args.tau,
        max_seq_len=args.max_seq_len,
        use_chat_template=USE_CHAT_TEMPLATE,
        seed=args.seed,
        extraction_contract=EXTRACTION_CONTRACT,
    )
    stability = finalize_stability(root, checkpoints, layers, concepts, setup_sig)
    summary = build_summary(root, checkpoints, layers, setup_sig)

    print("\n===== FIVE METRICS READY =====")
    print(f"summary: {root / METRICS_SUBDIR / 'summary.json'}")
    print(f"stability: {root / METRICS_SUBDIR / 'stability.json'}")
    print(f"rows: {summary['n_rows']}")
    if summary["rows"]:
        sample = summary["rows"][0]
        print("sample row keys:", sorted(sample.keys()))
        print("retained_K:", sample.get("retained_K"))
        print("geometry_strength:", sample.get("geometry_strength"))
        print("d_eff:", sample.get("d_eff"))
        print("inter_subspace_G:", sample.get("inter_subspace_G"))
    print(
        "subspace_stability concepts:",
        list(
            stability.get("layers", {})
            .get(str(layers[0]), {})
            .get("consecutive", {})
            .keys()
        )
        if layers
        else [],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
