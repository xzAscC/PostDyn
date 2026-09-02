"""Read-only integrity validator for complete 7B Think signed-subspace trees.

This module deliberately imports no model-loading code.  It checks the files
written by ``run_think_sft_differential_subspace.py`` before downstream work is
allowed to consume them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from postdyn.config import (
    MODEL_CHECKPOINTS,
    OLMO3_VARIANTS,
    THINK_7B_RLVR_CHECKPOINTS,
    THINK_7B_RLVR_REVISIONS,
)
from postdyn.domain_datasets import (
    DOLCI_HF_IDS,
    DOLCI_HF_REVISIONS,
    WIKITEXT_CONFIG,
    WIKITEXT_HF_ID,
    WIKITEXT_SPLIT,
)
from postdyn.think_sft_differential_experiment import (
    CONCEPT_PAIRS,
    SCALE_7B,
    covariance_n_samples,
)

EXPECTED_N_SAMPLES = covariance_n_samples(SCALE_7B)


CHECKPOINTS_SFT: tuple[str, ...] = tuple(MODEL_CHECKPOINTS["olmo3-think-sft"])
CHECKPOINTS_RLVR: tuple[str, ...] = THINK_7B_RLVR_CHECKPOINTS
LAYERS: tuple[int, ...] = (3, 6, 9, 11, 14, 17, 20, 22, 25, 28)
CONCEPTS: tuple[str, ...] = tuple(name for name, _, _ in CONCEPT_PAIRS)
MODEL_SFT = "olmo3-think-sft"
MODEL_RLVR = "olmo3-think-rlvr"
EXPECTED_D_MODEL = 4096
ORTHONORMALITY_TOLERANCE = 1e-4
# Full Gram validation is retained for small bases.  Large bases use every
# column norm plus this fixed-shape deterministic pair sample: spaced adjacent
# pairs, stride-2/stride-7 pairs, explicit edge pairs, and seeded random pairs.
EXACT_GRAM_MAX_COLUMNS = 256
SAMPLED_ADJACENT_PAIRS = 16
SAMPLED_STRIDED_PAIRS = 16
SAMPLED_RANDOM_PAIRS = 64
SAMPLED_PAIR_SEED = 0x504F535444594E
STABILITY_TOLERANCE = 1e-6
COMMIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
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
    """Machine-readable result of one tree validation."""

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


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _prompt_fingerprint(prompts: list[str]) -> str:
    payload = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (int, float)) or _is_finite_number(value)


def _stability_row_error(row: Any, kind: str, retained_k: dict[str, int]) -> str | None:
    if not isinstance(row, dict):
        return "stability row is not an object"
    if not _is_finite_number(row.get("subsim")):
        return "stability subsim is not finite"
    subsim = float(row["subsim"])
    if not -STABILITY_TOLERANCE <= subsim <= 1.0 + STABILITY_TOLERANCE:
        return "stability subsim is outside [0, 1]"
    if kind == "vs_reference":
        first, second = row.get("checkpoint"), row.get("reference")
    else:
        first, second = row.get("a"), row.get("b")
    if first not in retained_k or second not in retained_k:
        return "stability row references an unexpected checkpoint"
    k = row.get("k")
    if not _is_nonnegative_integer(k):
        return "stability k is not a nonnegative integer"
    expected_k = min(retained_k[first], retained_k[second])
    if k != expected_k:
        return "stability k disagrees with retained dimensions"
    return None


def _expected_setup_signature(
    root: Path, model_key: str, checkpoints: list[str]
) -> str:
    prompt_fingerprints: list[tuple[str, str]] = []
    dataset_sources: dict[str, dict[str, str]] = {}
    domains = tuple(dict.fromkeys(d for _, c, r in CONCEPT_PAIRS for d in (c, r)))
    for domain in domains:
        data = _read_json(root / "prompts" / f"{domain}.json")
        if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
            raise ValueError(f"invalid prompts/{domain}.json")
        prompts = data["prompts"]
        if len(prompts) < EXPECTED_N_SAMPLES or any(
            not isinstance(p, str) for p in prompts
        ):
            raise ValueError(
                f"prompts/{domain}.json does not contain {EXPECTED_N_SAMPLES} strings"
            )
        if data.get("domain") != domain or data.get("n_samples") != EXPECTED_N_SAMPLES:
            raise ValueError(f"prompt metadata mismatch for {domain}")
        if (
            data.get("use_chat_template") is not False
            or data.get("max_seq_len") != 2048
        ):
            raise ValueError(f"prompt extraction metadata mismatch for {domain}")
        if data.get("extraction_contract") != "raw_prompt_final_attention_token_v1":
            raise ValueError(f"prompt extraction contract mismatch for {domain}")
        source = data.get("source")
        expected_prompt_source = (
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
        if (
            not isinstance(source, dict)
            or set(source) != set(expected_prompt_source) | {"revision"}
            or any(
                source.get(key) != value
                for key, value in expected_prompt_source.items()
            )
            or not _is_commit_sha(source.get("revision"))
        ):
            raise ValueError(f"prompt source mismatch for {domain}")
        fingerprint = _prompt_fingerprint(prompts[:EXPECTED_N_SAMPLES])
        if data.get("prompt_fingerprint") != fingerprint:
            raise ValueError(f"prompt fingerprint mismatch for {domain}")
        prompt_fingerprints.append((domain, fingerprint))
        dataset_sources[domain] = {
            str(key): str(value) for key, value in source.items()
        }

    payload = {
        "pairs": [list(pair) for pair in CONCEPT_PAIRS],
        "model_keys": [model_key],
        "checkpoints": checkpoints,
        "layers": list(LAYERS),
        "model_ids": {model_key: OLMO3_VARIANTS[model_key].hf_id},
        "dataset_sources": dataset_sources,
        "prompt_fingerprints": prompt_fingerprints,
        "n_samples": EXPECTED_N_SAMPLES,
        "tau": 0.95,
        "max_seq_len": 2048,
        "use_chat_template": False,
        "seed": 42,
        "extraction_contract": "raw_prompt_final_attention_token_v1",
        "dtype": "bfloat16",
        "signed": True,
        "scale": "7b",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_commit_sha(value: Any) -> bool:
    return isinstance(value, str) and COMMIT_SHA_RE.fullmatch(value) is not None


def _revision_matches(actual: Any, expected: str) -> bool:
    if _is_commit_sha(actual) and _is_commit_sha(expected):
        return str(actual).lower() == expected.lower()
    return actual == expected


def _orthonormal_error(tensor: torch.Tensor, path: Path, name: str) -> str | None:
    columns = tensor.shape[1]
    if columns == 0:
        return None
    gram_dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    values = tensor.to(dtype=gram_dtype)
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
    signed = tensors["eigenvalues_signed"].to(dtype=torch.float64)
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
        values = tensors[f"eigenvalues_{sign}"].to(dtype=torch.float64)
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


def _artifact_paths(value: Any) -> list[Path]:
    if isinstance(value, str) and value.endswith(".safetensors"):
        return [Path(value)]
    if isinstance(value, dict):
        paths: list[Path] = []
        for item in value.values():
            paths.extend(_artifact_paths(item))
        return paths
    if isinstance(value, list):
        paths: list[Path] = []
        for item in value:
            paths.extend(_artifact_paths(item))
        return paths
    return []


def _optional_record_errors(root: Path, summary: dict[str, Any]) -> list[str]:
    records: dict[str, Any] = {}
    fixed_points = summary.get("fixed_points")
    if fixed_points is not None:
        if not isinstance(fixed_points, dict):
            return ["summary.json: fixed_points must be an object"]
        records.update(fixed_points)
    for label in ("sft_main", "rlvr_main"):
        if label in summary:
            records[label] = summary[label]
    if "final_main" in summary:
        records["final_main"] = summary["final_main"]
    errors: list[str] = []
    for label, record in records.items():
        if label not in {"base", "dpo", "sft_main", "rlvr_main", "final_main"}:
            errors.append(f"summary.json: unknown optional record {label!r}")
            continue
        if not isinstance(record, dict):
            errors.append(f"summary.json: optional record {label} is not an object")
            continue
        status = record.get("status")
        if status in {"opted_out", "skipped", "unavailable"}:
            if label not in {"base", "dpo", "sft_main", "rlvr_main", "final_main"}:
                errors.append(f"summary.json: {label} cannot be {status}")
            continue
        if status != "ok":
            errors.append(f"summary.json: optional record {label} has invalid status")
            continue
        revision = record.get("revision")
        setup_signature = record.get("setup_signature")
        if not _is_commit_sha(revision):
            errors.append(
                f"summary.json: optional record {label} has no immutable revision"
            )
            continue
        if not isinstance(setup_signature, str) or not setup_signature:
            errors.append(
                f"summary.json: optional record {label} has no setup signature"
            )
            continue
        artifact_values = _artifact_paths(record.get("artifact_paths"))
        artifact_values.extend(_artifact_paths(record.get("artifacts")))
        artifact_declared = "artifact_paths" in record or "artifacts" in record
        if label in {"sft_main", "rlvr_main", "final_main"} and not artifact_values:
            errors.append(f"summary.json: {label} has no valid artifact paths")
            continue
        if artifact_declared and not artifact_values:
            errors.append(
                f"summary.json: optional record {label} has malformed artifact paths"
            )
            continue
        for artifact_path in artifact_values:
            resolved = (
                artifact_path if artifact_path.is_absolute() else root / artifact_path
            ).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(
                    f"summary.json: {label} artifact path escapes the result root: {resolved}"
                )
                continue
            meta_path = resolved.with_suffix(".json")
            if not resolved.is_file() or not meta_path.is_file():
                errors.append(
                    f"summary.json: {label} artifact path is missing: {resolved}"
                )
                continue
            try:
                meta = _read_json(meta_path)
                model = meta["model"]
                checkpoint = meta["checkpoint"]
                layer = meta["layer"]
                concept = meta["concept"]
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(
                    f"summary.json: {label} artifact sidecar is invalid ({exc})"
                )
                continue
            error = _basis_error(
                resolved,
                meta_path,
                model,
                checkpoint,
                str(revision).lower(),
                setup_signature,
                layer,
                expected_concept=concept,
                require_full=True,
            )
            if error:
                errors.append(error)
    return errors


def _basis_error(
    st_path: Path,
    meta_path: Path,
    model: str,
    checkpoint: str,
    revision: str,
    setup_signature: str,
    layer: int,
    expected_concept: str | None = None,
    require_full: bool = False,
) -> str | None:
    try:
        meta = _read_json(meta_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"{meta_path}: unreadable sidecar ({exc})"
    if not isinstance(meta, dict):
        return f"{meta_path}: sidecar is not an object"
    if meta.get("tensors_saved", True) is False:
        return None
    try:
        tensors = load_file(str(st_path))
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        return f"{st_path}: unreadable artifact ({exc})"
    required_tensors = {"U_pos", "U_neg", "eigenvalues_pos", "eigenvalues_neg"}
    if require_full:
        required_tensors.update(
            {"U_pos_full", "U_neg_full", "eigenvalues_signed", "eigenvectors_signed"}
        )
    if not required_tensors.issubset(tensors):
        return f"{st_path}: signed tensor keys are incomplete or unexpected"
    for key, tensor in tensors.items():
        if key in {
            "U_pos_full",
            "U_neg_full",
            "eigenvalues_signed",
            "eigenvectors_signed",
            "residual_U_pos",
            "residual_U_neg",
        }:
            if tensor.ndim not in (1, 2) or not tensor.is_floating_point():
                return f"{st_path}: {key} has invalid spectrum shape or dtype"
            if not torch.isfinite(tensor).all():
                return f"{st_path}: {key} contains nonfinite values"
    if require_full:
        if tensors["eigenvalues_signed"].ndim != 1:
            return f"{st_path}: eigenvalues_signed must be a vector"
        if tensors["eigenvalues_signed"].shape != (EXPECTED_D_MODEL,):
            return (
                f"{st_path}: eigenvalues_signed must have shape ({EXPECTED_D_MODEL},)"
            )
        if tensors["eigenvectors_signed"].ndim != 2:
            return f"{st_path}: eigenvectors_signed must be a matrix"
        if tensors["eigenvectors_signed"].shape != (
            EXPECTED_D_MODEL,
            EXPECTED_D_MODEL,
        ):
            return (
                f"{st_path}: eigenvectors_signed must have shape "
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
                return f"{st_path}: U_{sign}_full is incompatible with the signed eigensystem"
        full_error = _full_eigensystem_error(tensors, st_path)
        if full_error:
            return full_error
    for key in ("U_pos", "U_neg"):
        tensor = tensors[key]
        if tensor.ndim != 2 or tensor.shape[0] != EXPECTED_D_MODEL:
            return f"{st_path}: {key} must have shape ({EXPECTED_D_MODEL}, k), got {tuple(tensor.shape)}"
        if not tensor.is_floating_point():
            return f"{st_path}: {key} must use a floating-point dtype"
        if not torch.isfinite(tensor).all():
            return f"{st_path}: {key} must contain only finite values"
        orth_error = _orthonormal_error(tensor, st_path, key)
        if orth_error:
            return orth_error
    for sign in ("pos", "neg"):
        basis = tensors[f"U_{sign}"]
        eigenvalues = tensors[f"eigenvalues_{sign}"]
        if eigenvalues.ndim != 1:
            return f"{st_path}: eigenvalues_{sign} must be a vector"
        if not eigenvalues.is_floating_point():
            return f"{st_path}: eigenvalues_{sign} must use a floating-point dtype"
        values = eigenvalues.detach().to(dtype=torch.float64)
        if not torch.isfinite(values).all() or (values <= 0).any():
            return (
                f"{st_path}: eigenvalues_{sign} must be finite and positive magnitudes"
            )
        if (values[:-1] < values[1:]).any():
            return f"{st_path}: eigenvalues_{sign} are not descending"
        retained_k = meta.get(f"k_{sign}")
        if (
            not isinstance(retained_k, int)
            or isinstance(retained_k, bool)
            or retained_k < 0
        ):
            return f"{meta_path}: k_{sign} is not a nonnegative integer"
        retained_k_int = int(retained_k)
        if retained_k_int != basis.shape[1]:
            return f"{meta_path}: k_{sign} disagrees with tensor shape"
        if eigenvalues.numel() < retained_k_int:
            return f"{st_path}: eigenvalues_{sign} does not cover retained k"
        if list(meta.get(f"u_{sign}_shape", [])) != list(basis.shape):
            return f"{meta_path}: u_{sign}_shape disagrees with tensor shape"
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
    if any(key not in meta for key in required):
        return f"{meta_path}: scalar metadata is incomplete"
    if (
        meta.get("concept")
        != (expected_concept if expected_concept is not None else meta.get("concept"))
        or (expected_concept is None and meta.get("concept") not in CONCEPTS)
        or meta.get("model") != model
        or meta.get("checkpoint") != checkpoint
        or not _revision_matches(meta.get("revision"), revision)
        or meta.get("setup_signature") != setup_signature
        or meta.get("layer") != layer
    ):
        return f"{meta_path}: model/checkpoint/layer provenance mismatch"
    if meta.get("d_model") != EXPECTED_D_MODEL or meta.get("tau") != 0.95:
        return f"{meta_path}: tensor metadata dimensions or tau mismatch"
    for key in (
        "d_eff_pos",
        "d_eff_neg",
        "geometry_strength_pos",
        "geometry_strength_neg",
        "tr_concept",
        "tr_ref",
    ):
        if not _is_finite_number(meta[key]):
            return f"{meta_path}: {key} is not finite"
    return None


def validate_result_tree(root: Path, trajectory: str | None = None) -> ValidationReport:
    """Validate a complete 7B SFT or RLVR result tree without loading a model."""
    root = Path(root)
    if trajectory is None:
        trajectory = (
            "rlvr" if root.name == "think_7b_rlvr_differential_subspace" else "sft"
        )
    if trajectory not in {"sft", "rlvr"}:
        raise ValueError("trajectory must be 'sft' or 'rlvr'")
    model = MODEL_SFT if trajectory == "sft" else MODEL_RLVR
    checkpoints = list(CHECKPOINTS_SFT if trajectory == "sft" else CHECKPOINTS_RLVR)
    revisions = dict(THINK_7B_RLVR_REVISIONS) if trajectory == "rlvr" else {}
    report = ValidationReport(str(root), trajectory)
    errors: list[str] = []
    try:
        setup_signature = _expected_setup_signature(root, model, checkpoints)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        report.fail("setup_signature", str(exc))
        return report
    report.pass_check("setup_signature")
    expected_manifest_names = {
        f"{model}__{checkpoint}.json" for checkpoint in checkpoints
    }
    manifests_root = root / "manifests"
    if not manifests_root.is_dir():
        errors.append(f"{manifests_root}: canonical manifest directory is missing")
    else:
        actual_manifest_names = {
            path.name
            for path in manifests_root.iterdir()
            if path.is_file() and path.name.startswith(f"{model}__")
        }
        allowed_extra_names = {f"{model}__main.json"}
        if not expected_manifest_names.issubset(actual_manifest_names) or not (
            actual_manifest_names - expected_manifest_names
        ).issubset(allowed_extra_names):
            errors.append(
                f"{manifests_root}: canonical manifest cardinality or trajectory mismatch"
            )
    for artifact_dir in (root / "metrics", root / "U"):
        if artifact_dir.is_dir():
            model_dir = artifact_dir / model
            if model_dir.is_dir():
                checkpoint_dirs = {
                    path.name for path in model_dir.iterdir() if path.is_dir()
                }
                expected_checkpoint_dirs = set(checkpoints)
                if checkpoint_dirs != expected_checkpoint_dirs:
                    errors.append(
                        f"{model_dir}: checkpoint cardinality or trajectory mismatch"
                    )
    for checkpoint in checkpoints:
        manifest_path = root / "manifests" / f"{model}__{checkpoint}.json"
        try:
            manifest = _read_json(manifest_path)
            expected = {
                "model": model,
                "checkpoint": checkpoint,
                "scale": "7b",
                "layers": list(LAYERS),
                "concepts": list(CONCEPTS),
                "setup_signature": setup_signature,
                "status": "ok",
            }
            if not isinstance(manifest, dict):
                errors.append(f"{manifest_path}: manifest is not an object")
            else:
                for key, value in expected.items():
                    if manifest.get(key) != value:
                        errors.append(f"{manifest_path}: {key} mismatch")
                if trajectory == "sft":
                    if not _is_commit_sha(manifest.get("revision")):
                        errors.append(
                            f"{manifest_path}: revision is not an immutable commit SHA"
                        )
                    else:
                        revisions[checkpoint] = str(manifest["revision"]).lower()
                elif manifest.get("revision") != revisions[checkpoint]:
                    errors.append(f"{manifest_path}: revision mismatch")
        except (
            OSError,
            ValueError,
            AttributeError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(f"{manifest_path}: unreadable manifest ({exc})")
        for layer in LAYERS:
            metrics_path = root / "metrics" / model / checkpoint / f"layer_{layer}.json"
            try:
                metrics = _read_json(metrics_path)
                if not _finite_tree(metrics):
                    errors.append(f"{metrics_path}: metrics contain nonfinite values")
                if (
                    metrics.get("model") != model
                    or metrics.get("checkpoint") != checkpoint
                    or metrics.get("layer") != layer
                    or metrics.get("setup_signature") != setup_signature
                    or metrics.get("tau") != 0.95
                    or metrics.get("n_samples") != EXPECTED_N_SAMPLES
                ):
                    errors.append(f"{metrics_path}: provenance mismatch")
                if "revision" in metrics and not _revision_matches(
                    metrics.get("revision"), revisions.get(checkpoint, "")
                ):
                    errors.append(f"{metrics_path}: revision mismatch")
                concepts = metrics.get("concepts")
                if (
                    not isinstance(concepts, dict)
                    or list(concepts) != list(CONCEPTS)
                    or any(
                        concepts[name].get("concept") != name
                        or concepts[name].get("tau") != 0.95
                        or not CORE_METRIC_FIELDS.issubset(concepts[name])
                        for name in CONCEPTS
                    )
                ):
                    errors.append(f"{metrics_path}: concept metrics are incomplete")
            except (
                OSError,
                ValueError,
                AttributeError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
            ) as exc:
                errors.append(f"{metrics_path}: missing or unreadable ({exc})")
            for concept in CONCEPTS:
                st_path = (
                    root
                    / "U"
                    / model
                    / checkpoint
                    / f"layer_{layer}"
                    / f"{concept}.safetensors"
                )
                meta_path = st_path.with_suffix(".json")
                if not st_path.is_file() or not meta_path.is_file():
                    errors.append(f"{st_path}: missing signed artifact or sidecar")
                    continue
                error = _basis_error(
                    st_path,
                    meta_path,
                    model,
                    checkpoint,
                    revisions.get(checkpoint, ""),
                    setup_signature,
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
    metrics_root = root / "metrics"
    summary: Any = None
    try:
        summary = _read_json(metrics_root / "summary.json")
        expected_rows = [
            {"model": model, "checkpoint": checkpoint, "layer": layer}
            for checkpoint in checkpoints
            for layer in LAYERS
        ]
        rows = summary.get("rows")
        actual_rows = (
            [
                {
                    "model": row.get("model"),
                    "checkpoint": row.get("checkpoint"),
                    "layer": row.get("layer"),
                }
                for row in rows
            ]
            if isinstance(rows, list)
            else []
        )
        row_payloads_ok = True
        if isinstance(rows, list):
            for row in rows:
                metrics = _read_json(
                    root
                    / "metrics"
                    / model
                    / row["checkpoint"]
                    / f"layer_{row['layer']}.json"
                )
                expected_retained = {
                    name: {
                        "pos": metrics["concepts"][name]["k_pos"],
                        "neg": metrics["concepts"][name]["k_neg"],
                    }
                    for name in CONCEPTS
                }
                expected_effective = {
                    name: {
                        "pos": metrics["concepts"][name]["d_eff_pos"],
                        "neg": metrics["concepts"][name]["d_eff_neg"],
                    }
                    for name in CONCEPTS
                }
                row_payloads_ok = (
                    row_payloads_ok
                    and row.get("retained_K") == expected_retained
                    and row.get("d_eff") == expected_effective
                )
        if (
            summary.get("checkpoints") != checkpoints
            or summary.get("layers") != list(LAYERS)
            or summary.get("setup_signature") != setup_signature
            or summary.get("n_rows") != len(expected_rows)
            or actual_rows != expected_rows
            or not row_payloads_ok
        ):
            report.fail(
                "summary",
                "summary checkpoint/layer order, count, or setup signature is incomplete",
            )
        else:
            report.pass_check("summary")
    except (
        OSError,
        ValueError,
        AttributeError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        report.fail("summary", f"summary.json missing or unreadable ({exc})")
    if "summary" in locals() and isinstance(summary, dict):
        optional_errors = _optional_record_errors(root, summary)
        if optional_errors:
            report.fail("summary", optional_errors[0])
            report.errors.extend(optional_errors[1:])
    try:
        stability = _read_json(metrics_root / "stability.json")
        if (
            stability.get("model") != model
            or stability.get("checkpoint_order") != checkpoints
            or stability.get("layers_order") != list(LAYERS)
            or stability.get("setup_signature") != setup_signature
            or stability.get("reference") != checkpoints[0]
        ):
            raise ValueError("stability provenance or order mismatch")
        for layer in LAYERS:
            block = stability.get("layers", {}).get(str(layer))
            if not isinstance(block, dict):
                raise ValueError(f"stability layer {layer} missing")
            for sign in ("pos", "neg"):
                sign_block = block.get(sign)
                if not isinstance(sign_block, dict):
                    raise ValueError(f"stability layer {layer} {sign} missing")
                for concept_name in CONCEPTS:
                    for name, expected_count in (
                        ("pairwise", 45),
                        ("consecutive", 9),
                        ("vs_reference", 10),
                    ):
                        rows = sign_block.get(name, {}).get(concept_name)
                        if not isinstance(rows, list) or len(rows) != expected_count:
                            raise ValueError(
                                f"stability layer {layer} {sign} {name} {concept_name} incomplete"
                            )
                        expected_sequence = {
                            "pairwise": [
                                (a, b)
                                for index, a in enumerate(checkpoints)
                                for b in checkpoints[index + 1 :]
                            ],
                            "consecutive": list(zip(checkpoints, checkpoints[1:])),
                            "vs_reference": [
                                (checkpoint, checkpoints[0])
                                for checkpoint in checkpoints
                            ],
                        }[name]
                        actual_sequence = [
                            (row.get("a"), row.get("b"))
                            if name != "vs_reference"
                            else (row.get("checkpoint"), row.get("reference"))
                            for row in rows
                        ]
                        if actual_sequence != expected_sequence:
                            raise ValueError(
                                f"stability layer {layer} {sign} {name} {concept_name} order mismatch"
                            )
                        retained_k = {}
                        for checkpoint in checkpoints:
                            metrics = _read_json(
                                root
                                / "metrics"
                                / model
                                / checkpoint
                                / f"layer_{layer}.json"
                            )
                            concept = metrics["concepts"][concept_name]
                            value = concept[f"k_{sign}"]
                            if not _is_nonnegative_integer(value):
                                raise ValueError(
                                    f"stability layer {layer} {sign} has invalid retained k"
                                )
                            retained_k[checkpoint] = value
                        for row in rows:
                            row_error = _stability_row_error(row, name, retained_k)
                            if row_error:
                                raise ValueError(
                                    f"stability layer {layer} {sign} {name}: {row_error}"
                                )
        report.pass_check("stability")
    except (
        OSError,
        ValueError,
        AttributeError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        report.fail("stability", str(exc))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--trajectory", choices=("sft", "rlvr"), default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = validate_result_tree(args.root, args.trajectory)
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("VALID" if report.ok else "INVALID")
        for error in report.errors:
            print(f"- {error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
