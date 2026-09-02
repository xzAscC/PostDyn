"""Model-free integrity checks for MATH-500 ablation result trees."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.config import EXPERIMENT_LAYERS_7B, OLMO3_VARIANTS
from src.math500_eval import MATH500_COUNT, load_authoritative_summary
from src.think_sft_differential_experiment import (
    FAMILY_THINK,
    SCALE_32B,
    SCALE_7B,
    canonical_extraction_protocol,
    layers_for_scale,
    root_for_trajectory,
    trajectory_config,
    validate_extraction_protocol,
)
from src.quantized_model_loader import (
    CANONICAL_NF4_PROVENANCE,
    validate_nf4_load_diagnostics,
)

CONDITIONS_FILENAME = "summary.json"
NF4_CONFIG = {
    "load_in_4bit": CANONICAL_NF4_PROVENANCE["quantization"]["bits"] == 4,
    "bnb_4bit_quant_type": CANONICAL_NF4_PROVENANCE["quantization"]["type"],
    "bnb_4bit_compute_dtype": CANONICAL_NF4_PROVENANCE["quantization"]["compute_dtype"],
    "bnb_4bit_use_double_quant": CANONICAL_NF4_PROVENANCE["quantization"][
        "double_quant"
    ],
}
CANONICAL_32B_MAX_NEW_TOKENS = 2048
STATIC_NF4_PROVENANCE = CANONICAL_NF4_PROVENANCE


@dataclass
class ValidationReport:
    root: str
    trajectory: str
    errors: list[str] = field(default_factory=list)
    conditions: list[dict[str, object]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def canonical_conditions(layers: tuple[int, ...] | None = None) -> tuple[str, ...]:
    selected_layers = tuple(EXPERIMENT_LAYERS_7B) if layers is None else layers
    return ("baseline",) + tuple(
        f"layer_{layer}_{sign}"
        for layer in selected_layers
        for sign in ("U_pos", "U_neg")
    )


def _condition_dir(root: Path, checkpoint: str, condition: str) -> Path:
    return root / "checkpoints" / checkpoint / condition


def _item_files(path: Path) -> tuple[list[Path], list[str]]:
    files = sorted(path.glob("math500_*.json"))
    expected = {f"math500_{index:02d}.json" for index in range(MATH500_COUNT)}
    actual = {file.name for file in files}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return files, [
        *(f"missing {name}" for name in missing),
        *(f"unexpected {name}" for name in extra),
    ]


def _read_item_identity(path: Path) -> Mapping[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    identity = raw.get("identity") if isinstance(raw, dict) else None
    return identity if isinstance(identity, dict) else None


def _basis_paths(
    root: Path, model: str, checkpoint: str, layer: int
) -> tuple[Path, Path]:
    base = root / "U" / model / checkpoint / f"layer_{layer}" / "math_vs_text"
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def _validate_extraction_manifest(
    root: Path,
    model: str,
    checkpoint: str,
    revision: str,
    layer: int,
    setup_signature: str | None,
) -> tuple[dict[str, object] | None, str | None]:
    path = root / "manifests" / f"{model}__{checkpoint}.json"
    if not path.is_file():
        return None, "basis extraction manifest is missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "basis extraction manifest is unreadable"
    if not isinstance(raw, dict):
        return None, "basis extraction manifest is not an object"
    expected = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "scale": "32b",
        "status": "ok",
        "loader_provenance": STATIC_NF4_PROVENANCE,
    }
    if setup_signature is not None:
        expected["setup_signature"] = setup_signature
    for key, value in expected.items():
        if raw.get(key) != value:
            return None, f"basis extraction manifest {key} mismatch"
    try:
        validate_extraction_protocol(raw.get("extraction_protocol"))
    except ValueError:
        return None, "basis extraction manifest extraction protocol mismatch"
    layers = raw.get("layers")
    concepts = raw.get("concepts")
    if not isinstance(layers, list) or layer not in layers:
        return None, "basis extraction manifest layer mismatch"
    if not isinstance(concepts, list) or "math_vs_text" not in concepts:
        return None, "basis extraction manifest concept mismatch"
    diagnostics_error = validate_nf4_load_diagnostics(raw.get("runtime_provenance"))
    if diagnostics_error is not None:
        return None, diagnostics_error
    return raw, None


def _validate_basis_artifact(
    *,
    root: Path,
    model: str,
    checkpoint: str,
    revision: str,
    layer: int,
    provenance: Mapping[str, object],
    require_static_provenance: bool = False,
) -> str | None:
    tensor_path, sidecar_path = _basis_paths(root, model, checkpoint, layer)
    try:
        tensor_bytes = tensor_path.read_bytes()
        sidecar_bytes = sidecar_path.read_bytes()
        metadata = json.loads(sidecar_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "basis sidecar or tensor missing/unreadable"
    if not isinstance(metadata, dict):
        return "basis sidecar is not an object"
    for key, expected in {
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "layer": layer,
    }.items():
        if metadata.get(key) != expected:
            return f"basis metadata {key} mismatch"
    if metadata.get("setup_signature") != provenance.get("setup_signature"):
        return "basis setup_signature mismatch"
    if require_static_provenance:
        if metadata.get("loader_provenance") != STATIC_NF4_PROVENANCE:
            return "basis static NF4 loader provenance mismatch"
        _, manifest_error = _validate_extraction_manifest(
            root, model, checkpoint, revision, layer, str(provenance["setup_signature"])
        )
        if manifest_error is not None:
            return manifest_error
    if provenance.get("sidecar_sha256") != hashlib.sha256(sidecar_bytes).hexdigest():
        return "basis sidecar hash mismatch"
    if provenance.get("tensor_sha256") != hashlib.sha256(tensor_bytes).hexdigest():
        return "basis tensor hash mismatch"
    return None


def _validate_condition(
    *,
    root: Path,
    dataset_path: Path,
    trajectory: str,
    checkpoint: str,
    revision: str,
    condition: str,
    max_new_tokens: int,
    dtype: str,
    quantization: str,
    artifact_root: Path,
    scale: str,
) -> tuple[dict[str, object] | None, list[str]]:
    path = _condition_dir(root, checkpoint, condition)
    if not path.is_dir():
        return None, [f"{checkpoint}/{condition}: condition directory missing"]
    files, file_errors = _item_files(path)
    if file_errors:
        return None, [f"{checkpoint}/{condition}: {error}" for error in file_errors]
    first = _read_item_identity(files[0])
    if first is None:
        return None, [f"{checkpoint}/{condition}: unreadable item identity"]
    model_key = first.get("model_key")
    model = first.get("model")
    experiment = first.get("experiment_identity")
    if not isinstance(model_key, str) or not isinstance(model, str):
        return None, [f"{checkpoint}/{condition}: missing model provenance"]
    if not isinstance(experiment, dict):
        return None, [f"{checkpoint}/{condition}: missing experiment provenance"]
    expected_experiment = {
        "ablation_contract": "residual-ablation-all-tokens-v1",
        "checkpoint": checkpoint,
        "condition": condition,
    }
    for key, expected in expected_experiment.items():
        if experiment.get(key) != expected:
            return None, [
                f"{checkpoint}/{condition}: experiment {key}={experiment.get(key)!r}, expected {expected!r}"
            ]
    if experiment.get("dataset") != str(dataset_path):
        return None, [f"{checkpoint}/{condition}: stale dataset provenance"]
    if model_key not in OLMO3_VARIANTS:
        return None, [f"{checkpoint}/{condition}: unknown model key"]
    expected_model_key = trajectory_config(FAMILY_THINK, scale, trajectory).model_key
    expected_model = OLMO3_VARIANTS[model_key]
    if model_key != expected_model_key or model != expected_model.hf_id:
        return None, [f"{checkpoint}/{condition}: model provenance mismatch"]
    if first.get("revision") != revision:
        return None, [f"{checkpoint}/{condition}: revision provenance mismatch"]
    generation = first.get("generation_contract")
    if generation != "raw-prompt-greedy-v1":
        return None, [f"{checkpoint}/{condition}: generation contract mismatch"]
    if (
        first.get("max_new_tokens") != max_new_tokens
        or first.get("dtype") != dtype
        or first.get("quantization") != quantization
    ):
        return None, [f"{checkpoint}/{condition}: generation provenance mismatch"]
    basis = experiment.get("basis")
    runtime = experiment.get("runtime_provenance")
    if scale == SCALE_32B:
        manifest_setup_signature = (
            basis.get("setup_signature") if isinstance(basis, dict) else None
        )
        manifest, manifest_error = _validate_extraction_manifest(
            artifact_root,
            expected_model.name,
            checkpoint,
            revision,
            int(condition.removeprefix("layer_").split("_", 1)[0])
            if condition != "baseline"
            else layers_for_scale(scale)[0],
            manifest_setup_signature,
        )
        if manifest_error is not None:
            return None, [f"{checkpoint}/{condition}: {manifest_error}"]
        assert manifest is not None
        if not isinstance(runtime, dict):
            return None, [f"{checkpoint}/{condition}: missing NF4 runtime provenance"]
        if runtime.get("loader") != "load_olmo3_32b_think":
            return None, [f"{checkpoint}/{condition}: NF4 loader provenance mismatch"]
        if runtime.get("nf4_config") != NF4_CONFIG:
            return None, [f"{checkpoint}/{condition}: NF4 config provenance mismatch"]
        if not isinstance(runtime.get("diagnostics"), dict):
            return None, [f"{checkpoint}/{condition}: missing loader diagnostics"]
        diagnostics_error = validate_nf4_load_diagnostics(runtime["diagnostics"])
        if diagnostics_error is not None:
            return None, [f"{checkpoint}/{condition}: {diagnostics_error}"]
        if runtime["diagnostics"] != manifest["runtime_provenance"]:
            return None, [
                f"{checkpoint}/{condition}: extraction/runtime diagnostics mismatch"
            ]
        runtime_path = root / "checkpoints" / checkpoint / "runtime_provenance.json"
        try:
            persisted_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, [
                f"{checkpoint}/{condition}: runtime provenance missing/unreadable"
            ]
        if persisted_runtime != runtime:
            return None, [f"{checkpoint}/{condition}: runtime provenance mismatch"]
    if condition == "baseline":
        if basis is not None:
            return None, [f"{checkpoint}/{condition}: baseline has basis provenance"]
    else:
        if not isinstance(basis, dict) or not isinstance(
            basis.get("setup_signature"), str
        ):
            return None, [f"{checkpoint}/{condition}: missing basis provenance"]
        if not basis.get("sidecar_sha256") or not basis.get("tensor_sha256"):
            return None, [f"{checkpoint}/{condition}: incomplete basis provenance"]
        if (
            basis.get("model") != expected_model.name
            or basis.get("checkpoint") != checkpoint
            or basis.get("revision") != revision
        ):
            return None, [f"{checkpoint}/{condition}: basis provenance mismatch"]
        layer_text, _ = condition.removeprefix("layer_").split("_", 1)
        if basis.get("layer") != int(layer_text):
            return None, [f"{checkpoint}/{condition}: basis layer mismatch"]
        basis_error = _validate_basis_artifact(
            root=artifact_root,
            model=expected_model.name,
            checkpoint=checkpoint,
            revision=revision,
            layer=int(layer_text),
            provenance=basis,
            require_static_provenance=scale == SCALE_32B,
        )
        if basis_error is not None:
            return None, [f"{checkpoint}/{condition}: {basis_error}"]
    summary = load_authoritative_summary(
        output_dir=path,
        model=model,
        model_key=model_key,
        revision=revision,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        quantization=quantization,
        experiment_identity=experiment,
    )
    if summary is None:
        return None, [
            f"{checkpoint}/{condition}: missing, stale, corrupt, or error result"
        ]
    record = dict(summary)
    record.update(
        {
            "checkpoint": checkpoint,
            "condition": condition,
            "basis_provenance": basis,
        }
    )
    return record, []


def collect_valid_conditions(
    root: Path,
    *,
    trajectory: str,
    dataset_path: Path,
    max_new_tokens: int,
    dtype: str,
    quantization: str,
    artifact_root: Path | None = None,
    scale: str = SCALE_7B,
    selected_checkpoints: tuple[str, ...] | None = None,
    selected_layers: tuple[int, ...] | None = None,
    project_root: Path | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    if scale == SCALE_32B and max_new_tokens != CANONICAL_32B_MAX_NEW_TOKENS:
        raise ValueError("32b requires max_new_tokens=2048")
    if scale == SCALE_32B and (dtype != "bfloat16" or quantization != "nf4"):
        raise ValueError("32b requires dtype=bfloat16 and quantization=nf4")
    if artifact_root is None:
        artifact_root = root_for_trajectory(
            FAMILY_THINK, scale, trajectory, project_root=project_root
        )
    config = trajectory_config(FAMILY_THINK, scale, trajectory)
    checkpoints = (
        config.checkpoints if selected_checkpoints is None else selected_checkpoints
    )
    layers = (
        tuple(layers_for_scale(scale)) if selected_layers is None else selected_layers
    )
    if any(checkpoint not in config.checkpoints for checkpoint in checkpoints):
        raise ValueError("selected checkpoint is not canonical for the trajectory")
    if any(layer not in layers_for_scale(scale) for layer in layers):
        raise ValueError("selected layer is not canonical for the scale")
    records: list[dict[str, object]] = []
    errors: list[str] = []
    for checkpoint in checkpoints:
        for condition in canonical_conditions(layers):
            record, condition_errors = _validate_condition(
                root=root,
                dataset_path=dataset_path,
                trajectory=trajectory,
                checkpoint=checkpoint,
                revision=config.revisions[checkpoint],
                condition=condition,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                quantization=quantization,
                artifact_root=artifact_root,
                scale=scale,
            )
            if record is not None:
                records.append(record)
            errors.extend(condition_errors)
    return records, errors


def validate_result_tree(
    root: Path,
    *,
    trajectory: str,
    dataset_path: Path,
    max_new_tokens: int,
    dtype: str,
    quantization: str,
    artifact_root: Path | None = None,
    scale: str = SCALE_7B,
    selected_checkpoints: tuple[str, ...] | None = None,
    selected_layers: tuple[int, ...] | None = None,
    project_root: Path | None = None,
) -> ValidationReport:
    report = ValidationReport(str(root), trajectory)
    if scale == SCALE_32B:
        from src.think_32b_differential_validator import (
            validate_full_canonical_publication,
        )

        publication = validate_full_canonical_publication(
            artifact_root
            or root_for_trajectory(
                FAMILY_THINK, scale, trajectory, project_root=project_root
            ),
            trajectory,
        )
        if not publication.ok:
            report.errors = [
                "32B extraction publication validation failed: " + publication.errors[0]
            ]
            return report
    records, errors = collect_valid_conditions(
        root,
        trajectory=trajectory,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        quantization=quantization,
        artifact_root=artifact_root,
        scale=scale,
        selected_checkpoints=selected_checkpoints,
        selected_layers=selected_layers,
        project_root=project_root,
    )
    report.conditions = records
    report.errors = errors
    config = trajectory_config(FAMILY_THINK, scale, trajectory)
    checkpoints = (
        config.checkpoints if selected_checkpoints is None else selected_checkpoints
    )
    layers = (
        tuple(layers_for_scale(scale)) if selected_layers is None else selected_layers
    )
    expected = {
        (checkpoint, condition)
        for checkpoint in checkpoints
        for condition in canonical_conditions(layers)
    }
    actual = {(str(row["checkpoint"]), str(row["condition"])) for row in records}
    report.errors.extend(
        f"missing canonical condition {checkpoint}/{condition}"
        for checkpoint, condition in sorted(expected - actual)
    )
    if len(actual) != len(records):
        report.errors.append("duplicate checkpoint/condition records")
    aggregate_path = root / "aggregate.json"
    if not aggregate_path.is_file():
        report.errors.append("aggregate.json missing")
    else:
        try:
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            report.errors.append("aggregate.json is unreadable")
        else:
            if (
                not isinstance(aggregate, dict)
                or aggregate.get("trajectory") != trajectory
                or aggregate.get("model_key") != config.model_key
                or aggregate.get("conditions") != records
            ):
                report.errors.append(
                    "aggregate.json does not match authoritative conditions"
                )
    for checkpoint in checkpoints:
        checkpoint_path = root / "checkpoints" / checkpoint / CONDITIONS_FILENAME
        expected_records = [
            record for record in records if record.get("checkpoint") == checkpoint
        ]
        if not checkpoint_path.is_file():
            report.errors.append(f"{checkpoint}: checkpoint summary missing")
            continue
        try:
            checkpoint_summary = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            report.errors.append(f"{checkpoint}: checkpoint summary unreadable")
            continue
        if checkpoint_summary != {
            "checkpoint": checkpoint,
            "conditions": expected_records,
        }:
            report.errors.append(
                f"{checkpoint}: checkpoint summary is stale or incomplete"
            )
    checkpoint_root = root / "checkpoints"
    expected_checkpoints = set(checkpoints)
    if checkpoint_root.is_dir():
        for directory in checkpoint_root.iterdir():
            if directory.is_dir() and directory.name not in expected_checkpoints:
                report.errors.append(
                    f"unexpected checkpoint directory {directory.name}"
                )
        expected_conditions = set(canonical_conditions(layers))
        for checkpoint in checkpoints:
            condition_root = checkpoint_root / checkpoint
            if condition_root.is_dir():
                for directory in condition_root.iterdir():
                    if directory.is_dir() and directory.name not in expected_conditions:
                        report.errors.append(
                            f"{checkpoint}: unexpected condition directory {directory.name}"
                        )
    return report
