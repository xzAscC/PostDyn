"""Strict, model-free validator for canonical 32B differential artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from postdyn.config import (
    EXPERIMENT_LAYERS_32B,
    OLMO3_VARIANTS,
    THINK_32B_RLVR_CHECKPOINTS,
    THINK_32B_RLVR_REVISIONS,
    THINK_32B_SFT_CHECKPOINTS,
    THINK_32B_SFT_REVISIONS,
)
from postdyn.domain_datasets import (
    DOLCI_HF_IDS,
    DOLCI_HF_REVISIONS,
    WIKITEXT_CONFIG,
    WIKITEXT_HF_ID,
    WIKITEXT_SPLIT,
)
from postdyn.think_sft_differential_experiment import CONCEPT_PAIRS
from postdyn.quantized_model_loader import (
    CANONICAL_NF4_PROVENANCE,
    validate_nf4_load_diagnostics,
)
from postdyn.differential_subspace import subspace_stability

MODEL_BY_TRAJECTORY = {
    "rlvr": "olmo3-32b-think-rlvr",
    "sft_lr_1e-4": "olmo3-32b-think-sft",
    "sft_lr_5e-5": "olmo3-32b-think-sft",
}
CHECKPOINTS_BY_TRAJECTORY = {
    "rlvr": tuple(THINK_32B_RLVR_CHECKPOINTS),
    "sft_lr_1e-4": tuple(THINK_32B_SFT_CHECKPOINTS),
    "sft_lr_5e-5": tuple(THINK_32B_SFT_CHECKPOINTS),
}
REVISIONS_BY_TRAJECTORY = {
    "rlvr": dict(THINK_32B_RLVR_REVISIONS),
    "sft_lr_1e-4": dict(THINK_32B_SFT_REVISIONS["1e-4"]),
    "sft_lr_5e-5": dict(THINK_32B_SFT_REVISIONS["5e-5"]),
}
LAYERS = tuple(EXPERIMENT_LAYERS_32B)
CONCEPTS = tuple(name for name, _, _ in CONCEPT_PAIRS)
EXPECTED_D_MODEL = 5120
ORTHONORMALITY_TOLERANCE = 1e-4
# Full Gram validation is retained for small bases.  Large bases use every
# column norm plus this fixed-shape deterministic pair sample: spaced adjacent
# pairs, stride-2/stride-7 pairs, explicit edge pairs, and seeded random pairs.
EXACT_GRAM_MAX_COLUMNS = 256
SAMPLED_ADJACENT_PAIRS = 16
SAMPLED_STRIDED_PAIRS = 16
SAMPLED_RANDOM_PAIRS = 64
SAMPLED_PAIR_SEED = 0x504F535444594E
EXPECTED_PROTOCOL = {
    "n_samples": 1000,
    "tau": 0.95,
    "max_seq_len": 2048,
    "use_chat_template": False,
    "extraction_contract": "raw_prompt_final_attention_token_v1",
    "dtype": "bfloat16",
    "signed": True,
}
STATIC_NF4_PROVENANCE = CANONICAL_NF4_PROVENANCE
STABILITY_TOLERANCE = 1e-6
CORE_METRIC_FIELDS = {
    "energy_pos",
    "energy_neg",
    "frobenius_strength_pos",
    "frobenius_strength_neg",
    "r_pos",
    "d_eff_pos",
    "d_eff_neg",
}


@dataclass
class ValidationReport:
    root: str
    trajectory: str
    ok: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def fail(self, check: str, message: str) -> None:
        self.ok = False
        self.checks[check] = False
        self.errors.append(message)

    def pass_check(self, check: str) -> None:
        self.checks[check] = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "trajectory": self.trajectory,
            "ok": self.ok,
            "checks": self.checks,
            "errors": self.errors,
        }


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    return (
        isinstance(value, bool) or not isinstance(value, (int, float)) or _finite(value)
    )


def _protocol_ok(value: Any) -> bool:
    return isinstance(value, dict) and value == EXPECTED_PROTOCOL


def _orthonormal_error(tensor: torch.Tensor, path: Path, name: str) -> str | None:
    columns = tensor.shape[1]
    if columns == 0:
        return None
    values = tensor.to(torch.float64)
    if columns <= EXACT_GRAM_MAX_COLUMNS:
        gram = values.T @ values
        identity = torch.eye(columns, dtype=gram.dtype, device=gram.device)
        error = (gram - identity).abs().max().item()
    else:
        norms = values.square().sum(dim=0)
        error = (norms - 1).abs().max().item()
        if error <= ORTHONORMALITY_TOLERANCE:
            pairs = _sampled_column_pairs(columns)
            left = values[:, [first for first, _ in pairs]]
            right = values[:, [second for _, second in pairs]]
            cross_error = (left * right).sum(dim=0).abs().max().item()
            error = max(error, cross_error)
    if error > ORTHONORMALITY_TOLERANCE:
        return f"{path}: {name} is not orthonormal"
    return None


def _sampled_column_pairs(columns: int) -> list[tuple[int, int]]:
    """Return a deterministic bounded sample of distinct column pairs."""
    if columns < 2:
        return []
    pairs: set[tuple[int, int]] = set()

    def add(first: int, second: int) -> None:
        if first != second:
            pairs.add((min(first, second), max(first, second)))

    for count, stride in (
        (SAMPLED_ADJACENT_PAIRS, 1),
        (SAMPLED_STRIDED_PAIRS, 2),
        (SAMPLED_STRIDED_PAIRS, 7),
    ):
        for index in range(min(count, columns - stride)):
            first = (index * max(1, (columns - stride) // count)) % (columns - stride)
            add(first, first + stride)
    add(0, columns - 1)
    add(0, min(1, columns - 1))
    add(max(0, columns - 2), columns - 1)
    rng = random.Random(SAMPLED_PAIR_SEED)
    target = min(
        columns * (columns - 1) // 2,
        SAMPLED_ADJACENT_PAIRS + 2 * SAMPLED_STRIDED_PAIRS + 3 + SAMPLED_RANDOM_PAIRS,
    )
    while len(pairs) < target:
        add(rng.randrange(columns), rng.randrange(columns))
    return sorted(pairs)


def _full_eigensystem_error(tensors: dict[str, torch.Tensor], path: Path) -> str | None:
    signed = tensors["eigenvalues_signed"].to(torch.float64)
    vectors = tensors["eigenvectors_signed"]
    orth_error = _orthonormal_error(vectors, path, "eigenvectors_signed")
    if orth_error:
        return orth_error
    positive = signed > 0
    negative = signed < 0
    expected_values = {
        "pos": signed[positive],
        "neg": (-signed[negative]).flip(0),
    }
    expected_vectors = {
        "pos": vectors[:, positive],
        "neg": vectors[:, negative].flip(1),
    }
    for sign in ("pos", "neg"):
        values = tensors[f"eigenvalues_{sign}"].to(torch.float64)
        if not torch.allclose(values, expected_values[sign], rtol=1e-5, atol=1e-6):
            return f"{path}: eigenvalues_{sign} disagrees with eigenvalues_signed sign selection"
        full_basis = tensors[f"U_{sign}_full"]
        orth_error = _orthonormal_error(full_basis, path, f"U_{sign}_full")
        if orth_error:
            return orth_error
        if not torch.allclose(
            full_basis.abs(), expected_vectors[sign].abs(), rtol=1e-5, atol=1e-6
        ):
            return f"{path}: U_{sign}_full disagrees with eigenvectors_signed sign selection"
        retained = tensors[f"U_{sign}"]
        if not torch.allclose(
            retained.abs(),
            full_basis[:, : retained.shape[1]].abs(),
            rtol=1e-5,
            atol=1e-6,
        ):
            return (
                f"{path}: U_{sign} disagrees with the retained columns of U_{sign}_full"
            )
    return None


def _expected_setup_signature(
    root: Path,
    model_key: str,
    checkpoints: list[str],
    layers: list[int],
    trajectory: str,
    revisions: dict[str, str],
) -> str:
    fingerprints = []
    sources = {}
    seed: int | None = None
    domains = tuple(dict.fromkeys(d for _, c, r in CONCEPT_PAIRS for d in (c, r)))
    for domain in domains:
        data = _read(root / "prompts" / f"{domain}.json")
        if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
            raise ValueError(f"invalid prompts/{domain}.json")
        prompts = data["prompts"]
        if len(prompts) != EXPECTED_PROTOCOL["n_samples"] or any(
            not isinstance(x, str) for x in prompts
        ):
            raise ValueError(f"prompts/{domain}.json has incomplete prompt coverage")
        expected_source = (
            {
                "kind": "huggingface",
                "hf_id": WIKITEXT_HF_ID,
                "config": WIKITEXT_CONFIG,
                "split": WIKITEXT_SPLIT,
            }
            if domain == "wikitext"
            else {
                "kind": "dolci",
                "hf_id": DOLCI_HF_IDS[domain],
            }
        )
        source = data.get("source")
        if (
            data.get("domain") != domain
            or not isinstance(source, dict)
            or set(source) != set(expected_source) | {"revision"}
            or any(source.get(key) != value for key, value in expected_source.items())
            or not re.fullmatch(r"[0-9a-fA-F]{40}", str(source.get("revision", "")))
        ):
            raise ValueError(f"prompt provenance mismatch for {domain}")
        if (
            data.get("n_samples") != 1000
            or data.get("max_seq_len") != 2048
            or data.get("use_chat_template") is not False
        ):
            raise ValueError(f"prompt protocol mismatch for {domain}")
        if data.get("extraction_contract") != EXPECTED_PROTOCOL["extraction_contract"]:
            raise ValueError(f"prompt extraction contract mismatch for {domain}")
        payload = json.dumps(
            prompts, ensure_ascii=False, separators=(",", ":")
        ).encode()
        fingerprint = hashlib.sha256(payload).hexdigest()
        if data.get("prompt_fingerprint") != fingerprint:
            raise ValueError(f"prompt fingerprint mismatch for {domain}")
        if seed is None:
            seed = data.get("seed")
        if data.get("seed") != seed:
            raise ValueError("prompt seed mismatch")
        fingerprints.append((domain, fingerprint))
        sources[domain] = {str(key): str(value) for key, value in source.items()}
    payload = {
        "pairs": [list(pair) for pair in CONCEPT_PAIRS],
        "model_keys": [model_key],
        "checkpoints": checkpoints,
        "layers": layers,
        "model_ids": {model_key: OLMO3_VARIANTS[model_key].hf_id},
        "dataset_sources": sources,
        "prompt_fingerprints": fingerprints,
        "n_samples": 1000,
        "tau": 0.95,
        "max_seq_len": 2048,
        "use_chat_template": False,
        "seed": seed,
        "extraction_contract": EXPECTED_PROTOCOL["extraction_contract"],
        "dtype": "bfloat16",
        "signed": True,
        "scale": "32b",
        "loader_provenance": STATIC_NF4_PROVENANCE,
        "trajectory": trajectory,
        "checkpoint_revisions": [
            (checkpoint, revisions[checkpoint]) for checkpoint in checkpoints
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _basis_error(
    path: Path,
    meta_path: Path,
    model: str,
    checkpoint: str,
    revision: str,
    setup_sig: str,
    layer: int,
    expected_concept: str | None = None,
    require_full: bool = False,
) -> str | None:
    try:
        tensors, meta = load_file(str(path)), _read(meta_path)
    except Exception as exc:
        return f"{path}: unreadable artifact ({exc})"
    required_tensors = {
        "U_pos",
        "U_neg",
        "eigenvalues_pos",
        "eigenvalues_neg",
    }
    if require_full:
        required_tensors.update(
            {"U_pos_full", "U_neg_full", "eigenvalues_signed", "eigenvectors_signed"}
        )
    if not isinstance(meta, dict) or not required_tensors.issubset(tensors):
        return f"{path}: exact tensor keys or sidecar object invalid"
    for key, tensor in tensors.items():
        if key in {
            "U_pos_full",
            "U_neg_full",
            "eigenvalues_signed",
            "eigenvectors_signed",
            "residual_U_pos",
            "residual_U_neg",
        } and (
            tensor.ndim not in (1, 2)
            or not tensor.is_floating_point()
            or not torch.isfinite(tensor).all()
        ):
            return f"{path}: {key} spectrum tensor is invalid"
    if require_full:
        if tensors["eigenvalues_signed"].ndim != 1:
            return f"{path}: eigenvalues_signed must be a vector"
        if tensors["eigenvalues_signed"].shape != (EXPECTED_D_MODEL,):
            return f"{path}: eigenvalues_signed must have shape ({EXPECTED_D_MODEL},)"
        if tensors["eigenvectors_signed"].ndim != 2:
            return f"{path}: eigenvectors_signed must be a matrix"
        if tensors["eigenvectors_signed"].shape != (
            EXPECTED_D_MODEL,
            EXPECTED_D_MODEL,
        ):
            return (
                f"{path}: eigenvectors_signed must have shape "
                f"({EXPECTED_D_MODEL}, {EXPECTED_D_MODEL})"
            )
        signed = tensors["eigenvalues_signed"]
        for sign in ("pos", "neg"):
            full_basis = tensors[f"U_{sign}_full"]
            expected_columns = (
                int((signed > 0).sum().item())
                if sign == "pos"
                else int((signed < 0).sum().item())
            )
            if full_basis.ndim != 2 or full_basis.shape != (
                EXPECTED_D_MODEL,
                expected_columns,
            ):
                return (
                    f"{path}: U_{sign}_full is incompatible with the signed eigensystem"
                )
        full_error = _full_eigensystem_error(tensors, path)
        if full_error:
            return full_error
    for sign in ("pos", "neg"):
        basis = tensors[f"U_{sign}"]
        if (
            basis.ndim != 2
            or basis.shape[0] != EXPECTED_D_MODEL
            or not basis.is_floating_point()
            or not torch.isfinite(basis).all()
        ):
            return f"{path}: U_{sign} shape, dtype, or finiteness invalid"
        orth_error = _orthonormal_error(basis, path, f"U_{sign}")
        if orth_error:
            return orth_error
        spectrum = tensors[f"eigenvalues_{sign}"]
        values = (
            spectrum.to(torch.float64)
            if spectrum.ndim == 1 and spectrum.is_floating_point()
            else None
        )
        if (
            values is None
            or not torch.isfinite(values).all()
            or (values <= 0).any()
            or (values[:-1] < values[1:]).any()
        ):
            return f"{path}: eigenvalues_{sign} sign, order, or finiteness invalid"
        k = meta.get(f"k_{sign}")
        if type(k) is not int or k != basis.shape[1] or values.numel() < k:
            return f"{meta_path}: k_{sign} disagrees with retained spectrum"
        if meta.get(f"u_{sign}_shape") != list(basis.shape):
            return f"{meta_path}: {sign} tensor metadata disagrees"
    required = (
        "concept",
        "tau",
        "n_concept",
        "n_ref",
        "d_model",
        "tr_concept",
        "tr_ref",
        "d_eff_pos",
        "d_eff_neg",
        "geometry_strength_pos",
        "geometry_strength_neg",
        *CORE_METRIC_FIELDS,
    )
    if any(
        key not in meta or (key != "concept" and not _finite(meta[key]))
        if key not in {"concept"}
        else key not in meta
        for key in required
    ):
        return f"{meta_path}: scalar metadata is incomplete or nonfinite"
    if any(
        meta.get(key) != value
        for key, value in {
            "concept": expected_concept
            if expected_concept is not None
            else meta.get("concept"),
            "model": model,
            "checkpoint": checkpoint,
            "revision": revision,
            "setup_signature": setup_sig,
            "layer": layer,
            "d_model": EXPECTED_D_MODEL,
            "tau": 0.95,
        }.items()
    ):
        return f"{meta_path}: model/checkpoint/layer/setup provenance mismatch"
    if meta.get("loader_provenance") != STATIC_NF4_PROVENANCE or not _protocol_ok(
        meta.get("extraction_protocol")
    ):
        return f"{meta_path}: static NF4 or extraction protocol provenance mismatch"
    return None


def _selection(
    trajectory: str, checkpoints: list[str] | None, layers: list[int] | None
):
    if trajectory not in MODEL_BY_TRAJECTORY:
        raise ValueError(f"unknown 32B trajectory {trajectory!r}")
    expected = list(CHECKPOINTS_BY_TRAJECTORY[trajectory])
    chosen = list(checkpoints or expected)
    chosen_layers = list(layers or LAYERS)
    if (
        not chosen
        or len(set(chosen)) != len(chosen)
        or any(x not in expected for x in chosen)
    ):
        raise ValueError("checkpoint selection is not canonical")
    if (
        not chosen_layers
        or len(set(chosen_layers)) != len(chosen_layers)
        or any(x not in LAYERS for x in chosen_layers)
    ):
        raise ValueError("layer selection is not canonical")
    revisions = REVISIONS_BY_TRAJECTORY[trajectory]
    return (
        MODEL_BY_TRAJECTORY[trajectory],
        chosen,
        chosen_layers,
        {x: revisions[x] for x in chosen},
    )


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _artifact_paths(value: Any) -> list[Path]:
    if isinstance(value, str) and value.endswith(".safetensors"):
        return [Path(value)]
    if isinstance(value, dict):
        return [path for item in value.values() for path in _artifact_paths(item)]
    if isinstance(value, list):
        return [path for item in value for path in _artifact_paths(item)]
    return []


def _histogram_error(value: Any, concepts: list[str], layers: list[int]) -> str | None:
    if not isinstance(value, dict) or set(value) != {str(layer) for layer in layers}:
        return "histogram layer coverage is incomplete"
    for layer in layers:
        block = value.get(str(layer))
        if not isinstance(block, dict) or list(block) != concepts:
            return f"histogram concept coverage is incomplete for layer {layer}"
        for concept in concepts:
            item = block[concept]
            if (
                not isinstance(item, dict)
                or item.get("bins") != 32
                or not isinstance(item.get("range"), list)
                or len(item["range"]) != 2
                or not all(_finite(x) for x in item["range"])
                or not isinstance(item.get("edges"), list)
                or len(item["edges"]) != 33
                or not all(_finite(x) for x in item["edges"])
            ):
                return f"histogram payload is malformed for {concept} layer {layer}"
    return None


def _residual_error(value: Any) -> str | None:
    if not isinstance(value, dict) or set(value) != {"pos", "neg"}:
        return "residual sign coverage is incomplete"
    for sign in ("pos", "neg"):
        item = value[sign]
        required = {"defined", "k_final", "d_res", "observed", "chance", "excess"}
        if not isinstance(item, dict) or set(item) != required:
            return f"residual {sign} payload is malformed"
        if (
            not isinstance(item["defined"], bool)
            or not _nonnegative_integer(item["k_final"])
            or not _nonnegative_integer(item["d_res"])
        ):
            return f"residual {sign} dimensions are invalid"
        if item["defined"]:
            if not all(_finite(item[key]) for key in ("observed", "chance", "excess")):
                return f"residual {sign} values are invalid"
            if not math.isclose(
                item["excess"],
                item["observed"] - item["chance"],
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                return f"residual {sign} algebra is inconsistent"
        elif item["k_final"] != 0 or any(
            item[key] is not None for key in ("observed", "chance", "excess")
        ):
            return f"residual {sign} undefined payload is inconsistent"
    return None


def _resolve_output_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidates = (
        path if path.is_absolute() else None,
        None if path.is_absolute() else Path.cwd() / path,
        None if path.is_absolute() else root / path,
        None if path.is_absolute() else root.parent / path,
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return (path if path.is_absolute() else root / path).resolve()


def _reference_errors(
    root: Path,
    model: str,
    selected: list[str],
    selected_layers: list[int],
) -> list[str]:
    errors: list[str] = []
    expected_concepts = [
        concept
        for concept in (
            "math_vs_code",
            "math_vs_instruction_following",
            "math_vs_general_reasoning",
        )
        if concept in CONCEPTS
    ]
    for layer in selected_layers:
        for checkpoint in selected:
            metrics_path = root / "metrics" / model / checkpoint / f"layer_{layer}.json"
            try:
                metrics = _read(metrics_path)
                concepts = metrics["concepts"]
                baseline = concepts["math_vs_wikitext"]
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(
                    f"{metrics_path}: reference retained dimensions unavailable ({exc})"
                )
                continue
            block = None
            try:
                block = _read(root / "metrics" / "stability.json")["layers"][
                    str(layer)
                ]["reference_robustness"][checkpoint]
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(block, dict) or set(block) != set(expected_concepts):
                errors.append(
                    f"stability.json: reference concept coverage is invalid for {checkpoint} layer {layer}"
                )
                continue
            for concept in expected_concepts:
                item = block[concept]
                if not isinstance(item, dict) or set(item) != {"pos", "neg"}:
                    errors.append(
                        f"stability.json: reference sign coverage is invalid for {checkpoint} {concept}"
                    )
                    continue
                comparison = concepts.get(concept)
                for sign in ("pos", "neg"):
                    sign_item = item[sign]
                    if (
                        not isinstance(sign_item, dict)
                        or set(sign_item) != {"subsim", "k"}
                        or not _finite(sign_item.get("subsim"))
                        or not 0.0 <= float(sign_item["subsim"]) <= 1.0
                        or not _nonnegative_integer(sign_item.get("k"))
                    ):
                        errors.append(
                            f"stability.json: reference {sign} payload is invalid for {checkpoint} {concept}"
                        )
                        continue
                    if (
                        not isinstance(comparison, dict)
                        or not _nonnegative_integer(baseline.get(f"k_{sign}"))
                        or not _nonnegative_integer(comparison.get(f"k_{sign}"))
                    ):
                        continue
                    if sign_item["k"] != min(
                        baseline[f"k_{sign}"], comparison[f"k_{sign}"]
                    ):
                        errors.append(
                            f"stability.json: reference {sign} k disagrees with retained dimensions for {checkpoint} {concept}"
                        )
    return errors


def _final_record_errors(
    root: Path,
    record: Any,
    model: str,
    selected_layers: list[int],
    concepts: list[str],
    expected_checkpoint: str,
    expected_model_key: str,
) -> list[str]:
    if not isinstance(record, dict):
        return ["summary.json: final_main is not an object"]
    status = record.get("status")
    if status == "skipped":
        return []
    if status != "ok":
        return ["summary.json: final_main has invalid status"]
    errors: list[str] = []
    revision = record.get("revision")
    setup = record.get("setup_signature")
    expected_model = next(
        config.hf_id for config in OLMO3_VARIANTS.values() if config.name == model
    )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(revision)):
        errors.append("summary.json: final_main revision is not immutable")
    if (
        record.get("model") != expected_model
        or record.get("model_key") != expected_model_key
    ):
        errors.append("summary.json: final_main model provenance mismatch")
    if not isinstance(setup, str) or not setup:
        errors.append("summary.json: final_main setup signature is missing")
    if record.get("checkpoint") != expected_checkpoint:
        errors.append("summary.json: final_main checkpoint provenance mismatch")
    root_value = record.get("root")
    if not isinstance(root_value, str) or not root_value:
        errors.append("summary.json: final_main root is missing")
    paths = record.get("artifact_paths")
    if not isinstance(paths, dict) or list(paths) != concepts:
        return errors + [
            "summary.json: final_main concept artifact coverage is incomplete"
        ]
    expected_count = len(concepts) * len(selected_layers)
    found = _artifact_paths(paths)
    if len(found) != expected_count:
        errors.append("summary.json: final_main layer artifact coverage is incomplete")
    final_root = _resolve_output_path(root, root_value)
    expected_final_root = (root / "final_points").resolve()
    if final_root != expected_final_root:
        errors.append(
            "summary.json: final_main root is not the canonical final_points path"
        )
        final_root = expected_final_root
    for concept in concepts:
        layer_paths = paths.get(concept)
        if not isinstance(layer_paths, dict) or set(layer_paths) != {
            str(x) for x in selected_layers
        }:
            errors.append(
                f"summary.json: final_main layer coverage is incomplete for {concept}"
            )
            continue
        for layer in selected_layers:
            path = Path(layer_paths[str(layer)])
            resolved = _resolve_output_path(root, layer_paths[str(layer)])
            if resolved is None:
                errors.append(
                    f"summary.json: final_main artifact path is invalid for {concept} layer {layer}"
                )
                continue
            try:
                resolved.relative_to(root.resolve())
                resolved.relative_to(expected_final_root)
            except ValueError:
                errors.append(
                    f"summary.json: final_main artifact path is inconsistent for {concept} layer {layer}"
                )
                continue
            sidecar = resolved.with_suffix(".json")
            if not resolved.is_file() or not sidecar.is_file():
                errors.append(
                    f"summary.json: final_main artifact is missing for {concept} layer {layer}"
                )
                continue
            try:
                meta = _read(sidecar)
                error = _basis_error(
                    resolved,
                    sidecar,
                    model,
                    record["checkpoint"],
                    str(revision).lower(),
                    str(setup),
                    layer,
                    expected_concept=concept,
                    require_full=True,
                )
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(
                    f"summary.json: final_main artifact sidecar is invalid ({exc})"
                )
                continue
            if error:
                errors.append(error)
    return errors


def _optional_record_errors(
    root: Path,
    summary: dict[str, Any],
    stability: dict[str, Any],
    model: str,
    selected: list[str],
    selected_layers: list[int],
    trajectory: str,
) -> list[str]:
    errors: list[str] = []
    if summary.get("concepts") != list(CONCEPTS):
        errors.append("summary.json: concept coverage is not canonical")
    fixed = summary.get("fixed_points")
    if not isinstance(fixed, dict) or set(fixed) != {"base", "dpo"}:
        errors.append("summary.json: fixed_points must contain base and dpo")
    elif any(
        not isinstance(fixed[name], dict) or fixed[name].get("status") != "unavailable"
        for name in ("base", "dpo")
    ):
        errors.append("summary.json: 32B fixed_points must be unavailable")
    if "final_main" not in summary:
        errors.append("summary.json: top-level final_main is missing")
    else:
        final_record = summary["final_main"]
        errors.extend(
            _final_record_errors(
                root,
                final_record,
                model,
                selected_layers,
                list(CONCEPTS),
                "rlvr_main" if trajectory == "rlvr" else "sft_main",
                MODEL_BY_TRAJECTORY[trajectory],
            )
        )
        if isinstance(final_record, dict) and final_record.get("status") == "ok":
            stability_final = stability.get("final_main")
            if not isinstance(stability_final, dict) or any(
                stability_final.get(key) != final_record.get(key)
                for key in ("checkpoint", "setup_signature", "revision", "root")
            ):
                errors.append(
                    "stability.json: final_main provenance disagrees with summary"
                )
    histogram_error = _histogram_error(
        summary.get("histogram"), list(CONCEPTS), selected_layers
    )
    if histogram_error:
        errors.append(f"summary.json: {histogram_error}")
    if stability.get("histogram") != summary.get("histogram"):
        errors.append("stability.json: histogram disagrees with summary")
    for layer in selected_layers:
        block = stability.get("layers", {}).get(str(layer), {})
        residual = block.get("residual_to_final")
        if not isinstance(residual, dict) or set(residual) != set(selected):
            errors.append(
                f"stability.json: residual coverage is incomplete for layer {layer}"
            )
        else:
            for checkpoint in selected:
                if not isinstance(residual[checkpoint], dict) or list(
                    residual[checkpoint]
                ) != list(CONCEPTS):
                    errors.append(
                        f"stability.json: residual concept coverage is incomplete for layer {layer}"
                    )
                else:
                    for concept in CONCEPTS:
                        residual_error = _residual_error(residual[checkpoint][concept])
                        if residual_error:
                            errors.append(
                                f"stability.json: {residual_error} for layer {layer}"
                            )
        robustness = block.get("reference_robustness")
        expected_reference_concepts = [
            concept
            for concept in (
                "math_vs_code",
                "math_vs_instruction_following",
                "math_vs_general_reasoning",
            )
            if concept in CONCEPTS
        ]
        if not isinstance(robustness, dict) or set(robustness) != set(selected):
            errors.append(
                f"stability.json: reference coverage is incomplete for layer {layer}"
            )
        else:
            for checkpoint in selected:
                if (
                    not isinstance(robustness[checkpoint], dict)
                    or list(robustness[checkpoint]) != expected_reference_concepts
                ):
                    errors.append(
                        f"stability.json: reference concept coverage is incomplete for layer {layer}"
                    )
    errors.extend(_reference_errors(root, model, selected, selected_layers))
    return errors


def _publication_error(
    root: Path,
    model: str,
    selected: list[str],
    selected_layers: list[int],
    setup_sig: str,
) -> list[str]:
    errors: list[str] = []
    expected_hf_id = next(
        config.hf_id for config in OLMO3_VARIANTS.values() if config.name == model
    )
    metrics_by_layer: dict[tuple[str, int], dict[str, Any]] = {}
    for checkpoint in selected:
        for layer in selected_layers:
            path = root / "metrics" / model / checkpoint / f"layer_{layer}.json"
            try:
                metrics = _read(path)
            except Exception as exc:
                errors.append(f"{path}: cannot read publication metrics ({exc})")
                continue
            three = metrics.get("three_metrics", {})
            for concept_name in CONCEPTS:
                concept = metrics.get("concepts", {}).get(concept_name)
                retained = three.get("retained_K", {}).get(concept_name)
                effective = three.get("d_eff", {}).get(concept_name)
                required = ("k_pos", "k_neg", "d_eff_pos", "d_eff_neg")
                if (
                    not isinstance(concept, dict)
                    or not CORE_METRIC_FIELDS.issubset(concept)
                    or any(
                        key not in concept or not _finite(concept[key])
                        for key in required
                    )
                    or retained
                    != {"pos": concept.get("k_pos"), "neg": concept.get("k_neg")}
                    or effective
                    != {
                        "pos": concept.get("d_eff_pos"),
                        "neg": concept.get("d_eff_neg"),
                    }
                    or not _nonnegative_integer(concept.get("k_pos"))
                    or not _nonnegative_integer(concept.get("k_neg"))
                ):
                    errors.append(
                        f"{path}: complete metrics are inconsistent for {concept_name}"
                    )
            metrics_by_layer[(checkpoint, layer)] = metrics

    summary_path = root / "metrics" / "summary.json"
    try:
        summary = _read(summary_path)
        rows = summary.get("rows")
        expected_keys = {"model", "hf_id", "checkpoint", "layer", "retained_K", "d_eff"}
        expected_ids = [
            (model, checkpoint, layer)
            for checkpoint in selected
            for layer in selected_layers
        ]
        if (
            not isinstance(rows, list)
            or summary.get("setup_signature") != setup_sig
            or summary.get("checkpoints") != selected
            or summary.get("layers") != selected_layers
            or summary.get("n_rows") != len(expected_ids)
            or len(rows) != len(expected_ids)
        ):
            raise ValueError("summary coverage or ordering is incomplete")
        for row, (expected_model, checkpoint, layer) in zip(rows, expected_ids):
            if not isinstance(row, dict) or not expected_keys.issubset(row):
                raise ValueError("summary row metric payload is incomplete")
            if (row.get("model"), row.get("checkpoint"), row.get("layer")) != (
                expected_model,
                checkpoint,
                layer,
            ):
                raise ValueError("summary rows are not in canonical order")
            if row.get("hf_id") != expected_hf_id:
                raise ValueError("summary row model identity is inconsistent")
            metrics = metrics_by_layer.get((checkpoint, layer), {})
            expected_retained = {
                concept_name: {
                    "pos": metrics.get("concepts", {})
                    .get(concept_name, {})
                    .get("k_pos"),
                    "neg": metrics.get("concepts", {})
                    .get(concept_name, {})
                    .get("k_neg"),
                }
                for concept_name in CONCEPTS
            }
            expected_effective = {
                concept_name: {
                    "pos": metrics.get("concepts", {})
                    .get(concept_name, {})
                    .get("d_eff_pos"),
                    "neg": metrics.get("concepts", {})
                    .get(concept_name, {})
                    .get("d_eff_neg"),
                }
                for concept_name in CONCEPTS
            }
            if (
                row.get("retained_K") != expected_retained
                or row.get("d_eff") != expected_effective
            ):
                raise ValueError("summary metrics disagree with per-layer metrics")
    except Exception as exc:
        errors.append(f"summary.json missing or invalid ({exc})")

    stability_path = root / "metrics" / "stability.json"
    try:
        stability = _read(stability_path)
        if (
            stability.get("model") != model
            or stability.get("checkpoint_order") != selected
            or stability.get("layers_order") != selected_layers
            or stability.get("setup_signature") != setup_sig
            or stability.get("reference") != (selected[0] if selected else None)
        ):
            raise ValueError("stability provenance or ordering is invalid")
        layers = stability.get("layers")
        if not isinstance(layers, dict) or set(layers) != {
            str(layer) for layer in selected_layers
        }:
            raise ValueError("stability layer coverage is incomplete")
        expected_sequences = {
            "pairwise": [
                (a, b)
                for index, a in enumerate(selected)
                for b in selected[index + 1 :]
            ],
            "consecutive": list(zip(selected, selected[1:])),
            "vs_reference": [(checkpoint, selected[0]) for checkpoint in selected],
        }
        expected_counts = {
            name: len(sequence) for name, sequence in expected_sequences.items()
        }
        for layer in selected_layers:
            bases: dict[tuple[str, str, str], torch.Tensor] = {}
            for checkpoint in selected:
                for concept in CONCEPTS:
                    basis_path = (
                        root
                        / "U"
                        / model
                        / checkpoint
                        / f"layer_{layer}"
                        / f"{concept}.safetensors"
                    )
                    tensors = load_file(str(basis_path), device="cpu")
                    bases.update(
                        ((checkpoint, concept, sign), tensors[f"U_{sign}"])
                        for sign in ("pos", "neg")
                    )
            layer_block = layers[str(layer)]
            if not isinstance(layer_block, dict):
                raise ValueError(f"stability layer {layer} is not an object")
            for concept in CONCEPTS:
                for sign in ("pos", "neg"):
                    sign_block = layer_block.get(sign)
                    if not isinstance(sign_block, dict):
                        raise ValueError(
                            f"stability layer {layer} {sign} block is missing"
                        )
                    retained_k = {
                        checkpoint: metrics_by_layer[(checkpoint, layer)]
                        .get("concepts", {})
                        .get(concept, {})
                        .get(f"k_{sign}")
                        for checkpoint in selected
                    }
                    if any(
                        not _nonnegative_integer(value) for value in retained_k.values()
                    ):
                        raise ValueError(
                            f"stability layer {layer} {sign} {concept} retained k is invalid"
                        )
                    for name, expected_sequence in expected_sequences.items():
                        rows = (
                            sign_block.get(name, {}).get(concept)
                            if isinstance(sign_block.get(name), dict)
                            else None
                        )
                        if (
                            not isinstance(rows, list)
                            or len(rows) != expected_counts[name]
                        ):
                            raise ValueError(
                                f"stability layer {layer} {sign} {name} {concept} row count is invalid"
                            )
                        actual_sequence = [
                            (row.get("checkpoint"), row.get("reference"))
                            if name == "vs_reference"
                            else (row.get("a"), row.get("b"))
                            for row in rows
                        ]
                        if actual_sequence != expected_sequence:
                            raise ValueError(
                                f"stability layer {layer} {sign} {name} {concept} ordering is invalid"
                            )
                        for row in rows:
                            if (
                                not isinstance(row, dict)
                                or not _finite(row.get("subsim"))
                                or not 0.0 <= float(row["subsim"]) <= 1.0
                            ):
                                raise ValueError(
                                    f"stability layer {layer} {sign} {name} {concept} SubSim is invalid"
                                )
                            first, second = (
                                (row.get("checkpoint"), row.get("reference"))
                                if name == "vs_reference"
                                else (row.get("a"), row.get("b"))
                            )
                            if first not in retained_k or second not in retained_k:
                                raise ValueError(
                                    f"stability layer {layer} {sign} {name} {concept} references an unexpected checkpoint"
                                )
                            expected_k = min(retained_k[first], retained_k[second])
                            if row.get("k") != expected_k:
                                raise ValueError(
                                    f"stability layer {layer} {sign} {name} {concept} retained k is inconsistent"
                                )
                            actual_subsim = subspace_stability(
                                bases[(first, concept, sign)],
                                bases[(second, concept, sign)],
                                k=expected_k,
                            )
                            if not math.isclose(
                                float(row["subsim"]),
                                actual_subsim,
                                rel_tol=STABILITY_TOLERANCE,
                                abs_tol=STABILITY_TOLERANCE,
                            ):
                                raise ValueError(
                                    f"stability layer {layer} {sign} {name} {concept} SubSim disagrees with signed bases"
                                )
    except Exception as exc:
        errors.append(f"stability.json missing or invalid ({exc})")
    return errors


def validate_full_canonical_publication(
    root: Path,
    trajectory: str,
    *,
    expected_setup_signature: str | None = None,
) -> ValidationReport:
    report = ValidationReport(str(root), trajectory)
    try:
        model, checkpoints, layers, revisions = _selection(trajectory, None, None)
        setup_sig = expected_setup_signature or _expected_setup_signature(
            Path(root), model, checkpoints, layers, trajectory, revisions
        )
    except Exception as exc:
        report.fail("setup_signature", str(exc))
        return report
    return validate_result_tree(
        Path(root),
        trajectory,
        checkpoints=checkpoints,
        layers=layers,
        expected_setup_signature=setup_sig,
        require_publications=True,
    )


def validate_result_tree(
    root: Path,
    trajectory: str,
    *,
    checkpoints: list[str] | None = None,
    layers: list[int] | None = None,
    expected_setup_signature: str | None = None,
    require_publications: bool = True,
) -> ValidationReport:
    root = Path(root)
    report = ValidationReport(str(root), trajectory)
    try:
        model, selected, selected_layers, revisions = _selection(
            trajectory, checkpoints, layers
        )
        setup_sig = expected_setup_signature or _expected_setup_signature(
            root, model, selected, selected_layers, trajectory, revisions
        )
    except Exception as exc:
        report.fail("setup_signature", str(exc))
        return report
    report.pass_check("setup_signature")
    errors = []
    manifest_dir = root / "manifests"
    if require_publications:
        expected_manifests = {f"{model}__{x}.json" for x in selected}
        actual = (
            {p.name for p in manifest_dir.glob(f"{model}__*.json")}
            if manifest_dir.is_dir()
            else set()
        )
        if actual != expected_manifests:
            errors.append(
                "manifest coverage is incomplete or contains another trajectory"
            )
    for checkpoint in selected:
        manifest_path = manifest_dir / f"{model}__{checkpoint}.json"
        try:
            if require_publications or manifest_path.is_file():
                manifest = _read(manifest_path)
                expected = {
                    "model": model,
                    "checkpoint": checkpoint,
                    "revision": revisions[checkpoint],
                    "scale": "32b",
                    "layers": selected_layers,
                    "concepts": list(CONCEPTS),
                    "setup_signature": setup_sig,
                    "status": "ok",
                    "extraction_protocol": EXPECTED_PROTOCOL,
                    "canonical_protocol": True,
                    "loader_provenance": STATIC_NF4_PROVENANCE,
                }
                if not isinstance(manifest, dict) or any(
                    manifest.get(k) != v for k, v in expected.items()
                ):
                    errors.append(f"{manifest_path}: manifest provenance mismatch")
                if (
                    validate_nf4_load_diagnostics(manifest.get("runtime_provenance"))
                    is not None
                ):
                    errors.append(f"{manifest_path}: runtime provenance is invalid")
        except Exception as exc:
            errors.append(f"{manifest_path}: unreadable manifest ({exc})")
        for layer in selected_layers:
            metrics_path = root / "metrics" / model / checkpoint / f"layer_{layer}.json"
            try:
                metrics = _read(metrics_path)
                if not isinstance(metrics, dict) or not _finite_tree(metrics):
                    errors.append(f"{metrics_path}: metrics are malformed or nonfinite")
                expected_metrics = {
                    "model": model,
                    "checkpoint": checkpoint,
                    "revision": revisions[checkpoint],
                    "layer": layer,
                    "setup_signature": setup_sig,
                    "tau": 0.95,
                    "n_samples": 1000,
                    "extraction_protocol": EXPECTED_PROTOCOL,
                }
                if (
                    not isinstance(metrics, dict)
                    or any(metrics.get(k) != v for k, v in expected_metrics.items())
                    or list(metrics.get("concepts", {})) != list(CONCEPTS)
                ):
                    errors.append(
                        f"{metrics_path}: metrics provenance or concept coverage mismatch"
                    )
                for concept in CONCEPTS:
                    concepts = metrics.get("concepts", {})
                    if (
                        not isinstance(concepts.get(concept), dict)
                        or concepts[concept].get("concept") != concept
                    ):
                        errors.append(f"{metrics_path}: concept metrics are incomplete")
                three = metrics.get("three_metrics", {})
                for concept in CONCEPTS:
                    item = metrics.get("concepts", {}).get(concept, {})
                    retained = three.get("retained_K", {}).get(concept, {})
                    effective = three.get("d_eff", {}).get(concept, {})
                    if retained != {
                        "pos": item.get("k_pos"),
                        "neg": item.get("k_neg"),
                    } or effective != {
                        "pos": item.get("d_eff_pos"),
                        "neg": item.get("d_eff_neg"),
                    }:
                        errors.append(
                            f"{metrics_path}: retained-K or d_eff consistency mismatch"
                        )
            except Exception as exc:
                errors.append(f"{metrics_path}: missing or unreadable ({exc})")
            for concept in CONCEPTS:
                st = (
                    root
                    / "U"
                    / model
                    / checkpoint
                    / f"layer_{layer}"
                    / f"{concept}.safetensors"
                )
                meta = st.with_suffix(".json")
                if not st.is_file() or not meta.is_file():
                    errors.append(f"{st}: signed artifact coverage is incomplete")
                else:
                    error = _basis_error(
                        st,
                        meta,
                        model,
                        checkpoint,
                        revisions[checkpoint],
                        setup_sig,
                        layer,
                        expected_concept=concept,
                        require_full=True,
                    )
                    if error:
                        errors.append(error)
    if errors:
        report.fail("artifacts", errors[0])
        report.errors.extend(errors[1:])
        return report
    report.pass_check("artifacts")
    if not require_publications:
        report.pass_check("publications")
        return report
    optional_errors: list[str] = []
    try:
        summary = _read(root / "metrics" / "summary.json")
        stability = _read(root / "metrics" / "stability.json")
        optional_errors.extend(
            _optional_record_errors(
                root, summary, stability, model, selected, selected_layers, trajectory
            )
        )
    except Exception as exc:
        optional_errors.append(f"optional publication payload is invalid ({exc})")
    if optional_errors:
        report.checks["optional_records"] = False
    else:
        report.pass_check("optional_records")
    publication_errors = _publication_error(
        root, model, selected, selected_layers, setup_sig
    )
    all_publication_errors = optional_errors + publication_errors
    if all_publication_errors:
        report.fail("summary", all_publication_errors[0])
        report.errors.extend(all_publication_errors[1:])
        report.checks["stability"] = False
    else:
        report.pass_check("summary")
        report.pass_check("stability")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--trajectory", choices=tuple(MODEL_BY_TRAJECTORY), required=True
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = validate_full_canonical_publication(args.root, args.trajectory)
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("VALID" if report.ok else "INVALID")
        for error in report.errors:
            print(f"- {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


def validate_checkpoint_tree(
    root: Path,
    trajectory: str,
    checkpoint: str,
    *,
    layers: list[int],
    expected_setup_signature: str,
    require_publication: bool = True,
) -> ValidationReport:
    return validate_result_tree(
        root,
        trajectory,
        checkpoints=[checkpoint],
        layers=layers,
        expected_setup_signature=expected_setup_signature,
        require_publications=require_publication,
    )
