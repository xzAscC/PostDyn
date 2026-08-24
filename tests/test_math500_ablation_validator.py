from __future__ import annotations

from pathlib import Path
import importlib
import hashlib
import json

import pytest

from src.math500_eval import MATH500_COUNT

validator = importlib.import_module("src.math500_ablation_validator")
canonical_conditions = validator.canonical_conditions
collect_valid_conditions = validator.collect_valid_conditions
validate_result_tree = validator.validate_result_tree
validate_basis_artifact = validator._validate_basis_artifact
validate_condition = validator._validate_condition
validate_cli = importlib.import_module("experiments.validate_math500_ablation")


DATASET = Path(__file__).parents[1] / "datasets" / "math500.json"
REAL_ROOT = Path(__file__).parents[1] / "results" / "math500_ablation_first50"


@pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason="real MATH-500 cache is not available"
)
def test_real_cache_accepts_hf_id_items_and_name_basis_provenance() -> None:
    records, errors = collect_valid_conditions(
        REAL_ROOT,
        trajectory="sft",
        dataset_path=Path("datasets/math500.json"),
        max_new_tokens=2048,
        dtype="bfloat16",
        quantization="none",
    )

    pairs = {(row["checkpoint"], row["condition"]) for row in records}
    assert {("step1000", "baseline"), ("step1000", "layer_3_U_pos")} <= pairs, errors
    assert not any(
        "step1000/baseline" in error or "step1000/layer_3_U_pos" in error
        for error in errors
    )


def test_validator_requires_full_canonical_coverage_without_loading_a_model(
    tmp_path: Path,
) -> None:
    report = validate_result_tree(
        tmp_path,
        trajectory="sft",
        dataset_path=DATASET,
        max_new_tokens=2048,
        dtype="bfloat16",
        quantization="none",
    )

    assert report.ok is False
    assert len(canonical_conditions()) == 21
    assert any("missing canonical condition" in error for error in report.errors)


def test_basis_validation_binds_actual_files_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    model = "Olmo-3-7B-Think-SFT"
    checkpoint = "step1000"
    revision = checkpoint
    layer = 3
    base = tmp_path / "U" / model / checkpoint / f"layer_{layer}" / "math_vs_text"
    tensor_path = base.with_suffix(".safetensors")
    sidecar_path = base.with_suffix(".json")
    tensor_bytes = b"basis tensor bytes"
    sidecar = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "layer": layer,
        "setup_signature": "setup-v1",
    }
    sidecar_bytes = json.dumps(sidecar).encode()
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(tensor_bytes)
    sidecar_path.write_bytes(sidecar_bytes)
    provenance = {
        "setup_signature": "setup-v1",
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
    }
    assert (
        validate_basis_artifact(
            root=tmp_path,
            model=model,
            checkpoint=checkpoint,
            revision=revision,
            layer=layer,
            provenance=provenance,
        )
        is None
    )
    tensor_path.write_bytes(b"tampered")
    assert (
        validate_basis_artifact(
            root=tmp_path,
            model=model,
            checkpoint=checkpoint,
            revision=revision,
            layer=layer,
            provenance=provenance,
        )
        == "basis tensor hash mismatch"
    )


def test_32b_basis_requires_exact_static_nf4_provenance(tmp_path: Path) -> None:
    model = "Olmo-3-32B-Think-SFT"
    checkpoint = "step1000"
    layer = 6
    base = tmp_path / "U" / model / checkpoint / f"layer_{layer}" / "math_vs_text"
    tensor_path = base.with_suffix(".safetensors")
    sidecar_path = base.with_suffix(".json")
    tensor_bytes = b"basis tensor bytes"
    sidecar = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": checkpoint,
        "layer": layer,
        "setup_signature": "setup-v1",
        "loader_provenance": validator.STATIC_NF4_PROVENANCE,
    }
    sidecar_bytes = json.dumps(sidecar).encode()
    tensor_path.parent.mkdir(parents=True)
    tensor_path.write_bytes(tensor_bytes)
    sidecar_path.write_bytes(sidecar_bytes)
    provenance = {
        "setup_signature": "setup-v1",
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
    }
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / f"{model}__{checkpoint}.json").write_text(
        json.dumps(
            {
                "model": model,
                "checkpoint": checkpoint,
                "revision": checkpoint,
                "scale": "32b",
                "layers": [6],
                "concepts": ["math_vs_text"],
                "setup_signature": "setup-v1",
                "status": "ok",
                "loader_provenance": validator.STATIC_NF4_PROVENANCE,
                "extraction_protocol": validator.canonical_extraction_protocol(),
                "runtime_provenance": {
                    "placement": "gpu-only",
                    "quant_backend": "bitsandbytes",
                    "safetensors": True,
                    "bitsandbytes_version": "0.49.2",
                    "transformers_version": "4.57.1",
                    "accelerate_version": "1.10.1",
                    "device_map": {"": "cuda:0"},
                    "peak_vram_bytes": 1,
                    "fallback_reason": None,
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        validate_basis_artifact(
            root=tmp_path,
            model=model,
            checkpoint=checkpoint,
            revision=checkpoint,
            layer=layer,
            provenance=provenance,
            require_static_provenance=True,
        )
        is None
    )
    sidecar["loader_provenance"] = {"use_safetensors": False}
    tampered_sidecar_bytes = json.dumps(sidecar).encode()
    sidecar_path.write_bytes(tampered_sidecar_bytes)
    provenance["sidecar_sha256"] = hashlib.sha256(tampered_sidecar_bytes).hexdigest()
    assert (
        validate_basis_artifact(
            root=tmp_path,
            model=model,
            checkpoint=checkpoint,
            revision=checkpoint,
            layer=layer,
            provenance=provenance,
            require_static_provenance=True,
        )
        == "basis static NF4 loader provenance mismatch"
    )


def test_32b_basis_rejects_missing_extraction_manifest(tmp_path: Path) -> None:
    model = "Olmo-3-32B-Think-SFT"
    checkpoint = "step1000"
    layer = 6
    base = tmp_path / "U" / model / checkpoint / f"layer_{layer}" / "math_vs_text"
    base.parent.mkdir(parents=True)
    sidecar = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": checkpoint,
        "layer": layer,
        "setup_signature": "setup-v1",
        "loader_provenance": validator.STATIC_NF4_PROVENANCE,
    }
    base.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    base.with_suffix(".safetensors").write_bytes(b"tensor")
    provenance = {
        "setup_signature": "setup-v1",
        "sidecar_sha256": hashlib.sha256(
            base.with_suffix(".json").read_bytes()
        ).hexdigest(),
        "tensor_sha256": hashlib.sha256(b"tensor").hexdigest(),
    }
    assert (
        validate_basis_artifact(
            root=tmp_path,
            model=model,
            checkpoint=checkpoint,
            revision=checkpoint,
            layer=layer,
            provenance=provenance,
            require_static_provenance=True,
        )
        == "basis extraction manifest is missing"
    )


def test_32b_condition_manifest_must_be_under_artifact_root(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    result_root = tmp_path / "results"
    model = "olmo3-32b-think-sft"
    checkpoint = "step1000"
    revision = checkpoint
    layer = 6
    setup_signature = "setup-v1"
    runtime = {
        "placement": "gpu-only",
        "quant_backend": "bitsandbytes",
        "safetensors": True,
        "bitsandbytes_version": "0.49.2",
        "transformers_version": "4.57.1",
        "accelerate_version": "1.10.1",
        "device_map": {"": "cuda:0"},
        "peak_vram_bytes": 1,
        "fallback_reason": None,
    }
    manifest = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "scale": "32b",
        "layers": [layer],
        "concepts": ["math_vs_text"],
        "setup_signature": setup_signature,
        "status": "ok",
        "loader_provenance": validator.STATIC_NF4_PROVENANCE,
        "runtime_provenance": runtime,
        "extraction_protocol": validator.canonical_extraction_protocol(),
    }
    manifest_path = artifact_root / "manifests" / f"{model}__{checkpoint}.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    basis_base = (
        artifact_root / "U" / model / checkpoint / f"layer_{layer}" / "math_vs_text"
    )
    basis_base.parent.mkdir(parents=True)
    sidecar = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "layer": layer,
        "setup_signature": setup_signature,
        "loader_provenance": validator.STATIC_NF4_PROVENANCE,
    }
    sidecar_bytes = json.dumps(sidecar).encode()
    tensor_bytes = b"basis tensor bytes"
    basis_base.with_suffix(".json").write_bytes(sidecar_bytes)
    basis_base.with_suffix(".safetensors").write_bytes(tensor_bytes)
    provenance = {
        "setup_signature": setup_signature,
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
        "tensor_sha256": hashlib.sha256(tensor_bytes).hexdigest(),
    }

    condition = "layer_6_U_pos"
    condition_dir = result_root / "checkpoints" / checkpoint / condition
    condition_dir.mkdir(parents=True)
    identity = {
        "model_key": "olmo3-32b-think-sft",
        "model": "allenai/Olmo-3-32B-Think-SFT",
        "revision": revision,
        "generation_contract": "raw-prompt-greedy-v1",
        "max_new_tokens": 2048,
        "dtype": "bfloat16",
        "quantization": "nf4",
        "experiment_identity": {
            "ablation_contract": "residual-ablation-all-tokens-v1",
            "checkpoint": checkpoint,
            "condition": condition,
            "dataset": str(DATASET),
            "basis": {
                **provenance,
                "model": model,
                "checkpoint": checkpoint,
                "revision": revision,
                "layer": layer,
            },
            "runtime_provenance": {
                "loader": "load_olmo3_32b_think",
                "nf4_config": validator.NF4_CONFIG,
                "diagnostics": runtime,
            },
        },
        "basis": {
            **provenance,
            "model": model,
            "checkpoint": checkpoint,
            "revision": revision,
            "layer": layer,
        },
        "runtime_provenance": {
            "loader": "load_olmo3_32b_think",
            "nf4_config": validator.NF4_CONFIG,
            "diagnostics": runtime,
        },
    }
    for index in range(MATH500_COUNT):
        (condition_dir / f"math500_{index:02d}.json").write_text(
            json.dumps({"identity": identity}), encoding="utf-8"
        )
    runtime_path = result_root / "checkpoints" / checkpoint / "runtime_provenance.json"
    runtime_path.write_text(
        json.dumps(identity["runtime_provenance"]), encoding="utf-8"
    )
    monkeypatch.setattr(
        validator,
        "load_authoritative_summary",
        lambda **kwargs: {"n_processed": 50},
    )

    kwargs = {
        "root": result_root,
        "dataset_path": DATASET,
        "trajectory": "sft_lr_1e-4",
        "checkpoint": checkpoint,
        "revision": revision,
        "condition": condition,
        "max_new_tokens": 2048,
        "dtype": "bfloat16",
        "quantization": "nf4",
        "artifact_root": artifact_root,
        "scale": "32b",
    }
    record, errors = validate_condition(**kwargs)
    assert record is not None, errors
    assert not errors

    manifest_path.unlink()
    result_manifest = result_root / "manifests" / manifest_path.name
    result_manifest.parent.mkdir()
    result_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    record, errors = validate_condition(**kwargs)
    assert record is None
    assert errors == [f"{checkpoint}/{condition}: basis extraction manifest is missing"]


def test_32b_validator_routes_selected_canonical_subset_without_model(
    tmp_path: Path, monkeypatch
) -> None:
    seen: list[tuple[str, int]] = []

    def fake_condition(**kwargs):
        seen.append((kwargs["checkpoint"], kwargs["condition"]))
        return (
            {"checkpoint": kwargs["checkpoint"], "condition": kwargs["condition"]},
            [],
        )

    monkeypatch.setattr(validator, "_validate_condition", fake_condition)
    records, errors = collect_valid_conditions(
        tmp_path,
        trajectory="sft_lr_1e-4",
        dataset_path=DATASET,
        max_new_tokens=2048,
        dtype="bfloat16",
        quantization="nf4",
        scale="32b",
        selected_checkpoints=("step1000",),
        selected_layers=(6,),
    )

    assert not errors
    assert len(records) == 3
    assert seen == [
        ("step1000", "baseline"),
        ("step1000", "layer_6_U_pos"),
        ("step1000", "layer_6_U_neg"),
    ]


@pytest.mark.parametrize(
    ("dtype", "quantization"),
    [("float16", "nf4"), ("bfloat16", "none")],
)
def test_direct_32b_validator_rejects_noncanonical_runtime_identity(
    tmp_path: Path, dtype: str, quantization: str
) -> None:
    with pytest.raises(
        ValueError, match="32b requires dtype=bfloat16 and quantization=nf4"
    ):
        collect_valid_conditions(
            tmp_path,
            trajectory="sft_lr_1e-4",
            dataset_path=DATASET,
            max_new_tokens=2048,
            dtype=dtype,
            quantization=quantization,
            scale="32b",
            selected_checkpoints=("step1000",),
            selected_layers=(6,),
        )


@pytest.mark.parametrize("option", ["--dtype", "--quantization"])
def test_cli_32b_rejects_wrong_runtime_identity(monkeypatch, option: str) -> None:
    called = False

    def fail_if_validated(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid 32b identity must fail before validation")

    monkeypatch.setattr(validate_cli, "validate_result_tree", fail_if_validated)
    value = "float16" if option == "--dtype" else "none"
    assert validate_cli.main(["/tmp/results", "--scale", "32b", option, value]) == 2
    assert called is False


def test_cli_32b_preflights_before_artifact_or_result_validation(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    project_root = tmp_path / "project"

    def fail_preflight(**kwargs):
        events.append(f"preflight:{kwargs['project_root']}")
        raise RuntimeError("7b incomplete")

    monkeypatch.setattr(
        "src.cross_pipeline_integrity.require_canonical_7b", fail_preflight
    )
    monkeypatch.setattr(
        validate_cli,
        "validate_full_canonical_publication",
        lambda *args, **kwargs: events.append("artifact-validation"),
    )
    monkeypatch.setattr(
        validate_cli,
        "validate_result_tree",
        lambda *args, **kwargs: events.append("result-validation"),
    )

    assert (
        validate_cli.main(
            [
                str(tmp_path / "results"),
                "--scale",
                "32b",
                "--project-root",
                str(project_root),
            ]
        )
        == 2
    )
    assert events == [f"preflight:{project_root}"]
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("bad_tokens", [1, 2047, 2049])
def test_direct_32b_validator_rejects_noncanonical_max_new_tokens(
    tmp_path: Path, bad_tokens: int
) -> None:
    with pytest.raises(ValueError, match="max_new_tokens=2048"):
        collect_valid_conditions(
            tmp_path,
            trajectory="sft_lr_1e-4",
            dataset_path=DATASET,
            max_new_tokens=bad_tokens,
            dtype="bfloat16",
            quantization="nf4",
            scale="32b",
            selected_checkpoints=("step1000",),
            selected_layers=(6,),
        )


def test_direct_7b_validator_keeps_custom_max_new_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        validator,
        "_validate_condition",
        lambda **kwargs: (
            {"checkpoint": kwargs["checkpoint"], "condition": kwargs["condition"]},
            [],
        ),
    )

    records, errors = collect_valid_conditions(
        tmp_path,
        trajectory="sft",
        dataset_path=DATASET,
        max_new_tokens=17,
        dtype="bfloat16",
        quantization="none",
        scale="7b",
        selected_checkpoints=("step1000",),
        selected_layers=(),
    )

    assert not errors
    assert records == [{"checkpoint": "step1000", "condition": "baseline"}]


def test_cli_32b_rejects_noncanonical_max_new_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_cli,
        "validate_result_tree",
        lambda *args, **kwargs: pytest.fail("invalid token budget reached validator"),
    )

    assert (
        validate_cli.main(
            ["/tmp/results", "--scale", "32b", "--max-new-tokens", "2049"]
        )
        == 2
    )


def test_cli_7b_keeps_custom_max_new_tokens(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_validate(*args, **kwargs):
        seen.update(kwargs)
        return validator.ValidationReport("/tmp/results", "sft")

    monkeypatch.setattr(validate_cli, "validate_result_tree", fake_validate)

    assert validate_cli.main(["/tmp/results", "--max-new-tokens", "17"]) == 0
    assert seen["max_new_tokens"] == 17
