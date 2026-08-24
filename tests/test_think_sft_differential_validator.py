from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

validator = importlib.import_module("src.think_sft_differential_validator")


SFT_ROOT = Path("results/think_sft_differential_subspace")


def test_canonical_sft_tree_is_accepted_without_model_loading():
    if not (SFT_ROOT / "metrics/summary.json").is_file():
        pytest.skip("canonical generated SFT tree is unavailable")
    report = validator.validate_result_tree(SFT_ROOT, "sft")
    assert report.ok, report.errors[:3]


def test_missing_signed_artifact_is_rejected(tmp_path):
    error = validator._basis_error(
        tmp_path / "missing.safetensors",
        tmp_path / "missing.json",
        "olmo3-think-sft",
        "step1000",
        "step1000",
        "signature",
        3,
    )
    assert error is not None
    assert "unreadable artifact" in error


def test_corrupt_sidecar_shape_is_rejected(tmp_path):
    source = SFT_ROOT / "U/olmo3-think-sft/step1000/layer_3/math_vs_wikitext"
    sidecar = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar["u_pos_shape"] = [4096, 999]
    corrupt_sidecar = tmp_path / "math_vs_wikitext.json"
    corrupt_sidecar.write_text(json.dumps(sidecar), encoding="utf-8")
    error = validator._basis_error(
        source.with_suffix(".safetensors"),
        corrupt_sidecar,
        "olmo3-think-sft",
        "step1000",
        "step1000",
        sidecar["setup_signature"],
        3,
    )
    assert error is not None
    assert "u_pos_shape" in error


def test_summary_checkpoint_order_is_rejected(monkeypatch):
    original = validator._read_json
    summary_path = SFT_ROOT / "metrics/summary.json"

    def reordered(path):
        value = original(path)
        if path == summary_path:
            value["rows"] = list(reversed(value["rows"]))
        return value

    monkeypatch.setattr(validator, "_read_json", reordered)
    report = validator.validate_result_tree(SFT_ROOT, "sft")
    assert not report.ok
    assert report.errors


def test_optional_skipped_records_are_accepted():
    summary = {
        "fixed_points": {
            label: {"status": "skipped"}
            for label in ("base", "dpo", "sft_main", "rlvr_main", "final_main")
        }
    }
    assert validator._optional_record_errors(Path("."), summary) == []


@pytest.mark.parametrize(
    ("field", "expected_value"),
    [
        ("revision", "corrupt"),
        ("setup_signature", "corrupt"),
        ("model", "corrupt"),
        ("layers", []),
        ("concepts", []),
        ("status", "failed"),
    ],
)
def test_manifest_provenance_is_rejected(monkeypatch, field, expected_value):
    original = validator._read_json
    manifest_path = SFT_ROOT / "manifests/olmo3-think-sft__step1000.json"

    def altered(path):
        value = original(path)
        if path == manifest_path:
            value[field] = expected_value
        return value

    monkeypatch.setattr(validator, "_read_json", altered)
    report = validator.validate_result_tree(SFT_ROOT, "sft")
    assert not report.ok
    assert report.errors


def test_sft_preflight_requires_canonical_manifests_without_phantom_main(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        validator,
        "_expected_setup_signature",
        lambda *args: "setup-v1",
    )
    report = validator.validate_result_tree(tmp_path, "sft")
    assert not report.ok
    assert any("manifest" in error for error in report.errors)
    assert not any("main" in error for error in report.errors)


def test_sft_preflight_rejects_missing_expected_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        validator,
        "_expected_setup_signature",
        lambda *args: "setup-v1",
    )
    (tmp_path / "manifests").mkdir()
    report = validator.validate_result_tree(tmp_path, "sft")
    assert not report.ok
    assert any("unreadable manifest" in error for error in report.errors)


def _basis_fixture(*, dtype=torch.float32, value=1.0, k=2, eigenvalues=None):
    tensors = {
        "U_pos": torch.zeros((validator.EXPECTED_D_MODEL, k), dtype=dtype),
        "U_neg": torch.zeros((validator.EXPECTED_D_MODEL, k), dtype=dtype),
        "eigenvalues_pos": torch.tensor(eigenvalues or [2.0, 1.0], dtype=torch.float32),
        "eigenvalues_neg": torch.tensor(eigenvalues or [2.0, 1.0], dtype=torch.float32),
    }
    tensors["U_pos"].diagonal().fill_(value)
    tensors["U_neg"].diagonal().fill_(value)
    meta = {
        "concept": validator.CONCEPTS[0],
        "model": "olmo3-think-sft",
        "checkpoint": "step1000",
        "revision": "step1000",
        "setup_signature": "signature",
        "layer": 3,
        "tau": 0.95,
        "n_concept": 1000,
        "n_ref": 1000,
        "d_model": validator.EXPECTED_D_MODEL,
        "tr_concept": 1.0,
        "tr_ref": 1.0,
        "d_eff_pos": 1.0,
        "d_eff_neg": 1.0,
        "geometry_strength_pos": 1.0,
        "geometry_strength_neg": 1.0,
        "energy_pos": 4.0,
        "energy_neg": 1.0,
        "frobenius_strength_pos": 2.0,
        "frobenius_strength_neg": 1.0,
        "r_pos": 1.0,
        "r_neg": 1.0,
        "k_pos": k,
        "k_neg": k,
        "u_pos_shape": [validator.EXPECTED_D_MODEL, k],
        "u_neg_shape": [validator.EXPECTED_D_MODEL, k],
    }
    return tensors, meta


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test_non_floating_basis_is_rejected(monkeypatch, tmp_path, dtype):
    tensors, meta = _basis_fixture(dtype=dtype)
    monkeypatch.setattr(validator, "load_file", lambda _: tensors)
    monkeypatch.setattr(validator, "_read_json", lambda _: meta)
    error = validator._basis_error(
        tmp_path / "artifact.safetensors",
        tmp_path / "artifact.json",
        "olmo3-think-sft",
        "step1000",
        "step1000",
        "signature",
        3,
    )
    assert error is not None
    assert "floating" in error


def test_nonfinite_basis_is_rejected_before_gram_check(monkeypatch, tmp_path):
    tensors, meta = _basis_fixture(value=float("nan"))
    monkeypatch.setattr(validator, "load_file", lambda _: tensors)
    monkeypatch.setattr(validator, "_read_json", lambda _: meta)
    error = validator._basis_error(
        tmp_path / "artifact.safetensors",
        tmp_path / "artifact.json",
        "olmo3-think-sft",
        "step1000",
        "step1000",
        "signature",
        3,
    )
    assert error is not None
    assert "finite" in error


def test_compact_sidecar_accepts_spectrum_tensors_and_rejects_corruption(
    monkeypatch, tmp_path
):
    tensors, meta = _basis_fixture()
    monkeypatch.setattr(validator, "load_file", lambda _: tensors)
    monkeypatch.setattr(validator, "_read_json", lambda _: meta)
    assert (
        validator._basis_error(
            tmp_path / "artifact.safetensors",
            tmp_path / "artifact.json",
            "olmo3-think-sft",
            "step1000",
            "step1000",
            "signature",
            3,
        )
        is None
    )

    tensors["eigenvalues_pos"][0] = 0.0
    error = validator._basis_error(
        tmp_path / "artifact.safetensors",
        tmp_path / "artifact.json",
        "olmo3-think-sft",
        "step1000",
        "step1000",
        "signature",
        3,
    )
    assert error is not None
    assert "positive magnitudes" in error


def test_large_float32_basis_uses_elementwise_gram_error(monkeypatch, tmp_path):
    n = 4096
    epsilon = 5e-8
    a = (1.0 - epsilon) ** 0.5
    b = ((1.0 + (n - 1) * epsilon) ** 0.5 - a) / n
    basis = (
        a * torch.eye(n, dtype=torch.float64)
        + b * torch.ones((n, n), dtype=torch.float64)
    ).to(torch.float32)
    gram_error = basis.to(torch.float64).T @ basis.to(torch.float64) - torch.eye(
        n, dtype=torch.float64
    )
    assert gram_error.abs().max().item() < validator.ORTHONORMALITY_TOLERANCE
    assert (
        torch.linalg.norm(gram_error, ord=float("inf")).item()
        > validator.ORTHONORMALITY_TOLERANCE
    )
    assert validator._orthonormal_error(basis, tmp_path / "valid", "basis") is None

    corrupted = basis.clone()
    corrupted[0, 0] += 1e-2
    assert (
        validator._orthonormal_error(corrupted, tmp_path / "bad", "basis") is not None
    )


@pytest.mark.parametrize("corruption", ["scale", "adjacent"])
def test_large_basis_rejects_scale_and_sampled_cross_column_corruption(
    monkeypatch, tmp_path, corruption
):
    monkeypatch.setattr(validator, "EXACT_GRAM_MAX_COLUMNS", 4)
    basis = torch.eye(8, dtype=torch.float32)
    if corruption == "scale":
        basis[:, 7] *= 1.01
    else:
        basis[:, 0] += 0.01 * basis[:, 1]
    error = validator._orthonormal_error(basis, tmp_path / corruption, "basis")
    assert error is not None


def test_empty_basis_passes_and_sampled_pairs_are_bounded_and_deterministic(
    tmp_path,
):
    assert (
        validator._orthonormal_error(torch.empty((8, 0)), tmp_path / "empty", "basis")
        is None
    )
    first = validator._sampled_column_pairs(4096)
    assert first == validator._sampled_column_pairs(4096)
    assert (0, 1) in first
    assert (0, 2) in first
    assert (0, 7) in first
    assert len(first) <= 16 + 2 * 16 + 3 + 64


def test_eigenvalue_spectrum_must_cover_retained_k(monkeypatch, tmp_path):
    tensors, meta = _basis_fixture(eigenvalues=[2.0])
    monkeypatch.setattr(validator, "load_file", lambda _: tensors)
    monkeypatch.setattr(validator, "_read_json", lambda _: meta)
    error = validator._basis_error(
        tmp_path / "artifact.safetensors",
        tmp_path / "artifact.json",
        "olmo3-think-sft",
        "step1000",
        "step1000",
        "signature",
        3,
    )
    assert error is not None
    assert "retained k" in error


def test_partial_full_eigensystem_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(validator, "EXPECTED_D_MODEL", 4)
    tensors, meta = _basis_fixture(k=1, eigenvalues=[2.0])
    tensors.update(
        {
            "U_pos_full": torch.eye(4)[:, :1],
            "U_neg_full": torch.eye(4)[:, 1:2],
            "eigenvalues_signed": torch.tensor([2.0, -1.0]),
            "eigenvectors_signed": torch.eye(4)[:, :2],
        }
    )
    monkeypatch.setattr(validator, "load_file", lambda _: tensors)
    monkeypatch.setattr(validator, "_read_json", lambda _: meta)
    error = validator._basis_error(
        tmp_path / "artifact.safetensors",
        tmp_path / "artifact.json",
        "olmo3-think-sft",
        "step1000",
        "step1000",
        "signature",
        3,
        expected_concept=validator.CONCEPTS[0],
        require_full=True,
    )
    assert error is not None
    assert "eigenvalues_signed" in error


def test_sft_sha_revisions_are_shared_by_all_checkpoint_artifacts(
    tmp_path, monkeypatch
):
    checkpoints = tuple(f"step{i}" for i in range(10))
    concept = "math_vs_wikitext"
    layer = 3
    model = "olmo3-think-sft"
    setup = "setup-v1"
    revision = "a" * 40
    monkeypatch.setattr(validator, "CHECKPOINTS_SFT", checkpoints)
    monkeypatch.setattr(validator, "LAYERS", (layer,))
    monkeypatch.setattr(validator, "CONCEPTS", (concept,))
    monkeypatch.setattr(validator, "EXPECTED_D_MODEL", 4)
    monkeypatch.setattr(validator, "_expected_setup_signature", lambda *args: setup)

    tensors = {
        "U_pos": torch.eye(4)[:, :1].contiguous(),
        "U_neg": torch.eye(4)[:, 3:4].contiguous(),
        "eigenvalues_pos": torch.tensor([2.0]),
        "eigenvalues_neg": torch.tensor([1.0]),
        "U_pos_full": torch.eye(4)[:, :1].contiguous(),
        "U_neg_full": torch.eye(4)[:, 3:4].contiguous(),
        "eigenvalues_signed": torch.tensor([2.0, 0.0, 0.0, -1.0]),
        "eigenvectors_signed": torch.eye(4),
    }

    def meta(checkpoint: str) -> dict[str, object]:
        return {
            "concept": concept,
            "model": model,
            "checkpoint": checkpoint,
            "revision": revision,
            "setup_signature": setup,
            "layer": layer,
            "tau": 0.95,
            "n_concept": 1000,
            "n_ref": 1000,
            "d_model": 4,
            "tr_concept": 1.0,
            "tr_ref": 1.0,
            "d_eff_pos": 1.0,
            "d_eff_neg": 1.0,
            "geometry_strength_pos": 1.0,
            "geometry_strength_neg": 1.0,
            "energy_pos": 4.0,
            "energy_neg": 1.0,
            "frobenius_strength_pos": 2.0,
            "frobenius_strength_neg": 1.0,
            "r_pos": 1.0,
            "r_neg": 1.0,
            "k_pos": 1,
            "k_neg": 1,
            "u_pos_shape": [4, 1],
            "u_neg_shape": [4, 1],
        }

    metrics_root = tmp_path / "metrics" / model
    u_root = tmp_path / "U" / model
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    rows = []
    for checkpoint in checkpoints:
        manifest = {
            "model": model,
            "checkpoint": checkpoint,
            "revision": revision,
            "scale": "7b",
            "layers": [layer],
            "concepts": [concept],
            "setup_signature": setup,
            "status": "ok",
        }
        (manifest_root / f"{model}__{checkpoint}.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        artifact_dir = u_root / checkpoint / f"layer_{layer}"
        artifact_dir.mkdir(parents=True)
        save_file(tensors, str(artifact_dir / f"{concept}.safetensors"))
        (artifact_dir / f"{concept}.json").write_text(
            json.dumps(meta(checkpoint)), encoding="utf-8"
        )
        metric_dir = metrics_root / checkpoint
        metric_dir.mkdir(parents=True)
        concept_metrics = {
            "concept": concept,
            "tau": 0.95,
            "n_concept": 1000,
            "n_ref": 1000,
            "d_model": 4,
            "tr_concept": 1.0,
            "tr_ref": 1.0,
            "k_pos": 1,
            "k_neg": 1,
            "d_eff_pos": 1.0,
            "d_eff_neg": 1.0,
            "geometry_strength_pos": 1.0,
            "geometry_strength_neg": 1.0,
            "energy_pos": 4.0,
            "energy_neg": 1.0,
            "frobenius_strength_pos": 2.0,
            "frobenius_strength_neg": 1.0,
            "r_pos": 1.0,
            "r_neg": 1.0,
        }
        (metric_dir / f"layer_{layer}.json").write_text(
            json.dumps(
                {
                    "model": model,
                    "checkpoint": checkpoint,
                    "layer": layer,
                    "setup_signature": setup,
                    "tau": 0.95,
                    "n_samples": 1000,
                    "concepts": {concept: concept_metrics},
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "model": model,
                "checkpoint": checkpoint,
                "layer": layer,
                "retained_K": {concept: {"pos": 1, "neg": 1}},
                "d_eff": {concept: {"pos": 1.0, "neg": 1.0}},
            }
        )

    def stability_rows(name: str) -> list[dict[str, object]]:
        if name == "pairwise":
            pairs = [
                (a, b)
                for index, a in enumerate(checkpoints)
                for b in checkpoints[index + 1 :]
            ]
            return [{"a": a, "b": b, "subsim": 1.0, "k": 1} for a, b in pairs]
        if name == "consecutive":
            return [
                {"a": a, "b": b, "subsim": 1.0, "k": 1}
                for a, b in zip(checkpoints, checkpoints[1:])
            ]
        return [
            {
                "checkpoint": checkpoint,
                "reference": checkpoints[0],
                "subsim": 1.0,
                "k": 1,
            }
            for checkpoint in checkpoints
        ]

    stability_sign = {
        name: {concept: stability_rows(name)}
        for name in ("pairwise", "consecutive", "vs_reference")
    }
    metrics_root.parent.mkdir(parents=True, exist_ok=True)
    (metrics_root.parent / "summary.json").write_text(
        json.dumps(
            {
                "checkpoints": list(checkpoints),
                "layers": [layer],
                "setup_signature": setup,
                "n_rows": len(rows),
                "rows": rows,
                "fixed_points": {
                    "sft_main": {
                        "status": "ok",
                        "revision": revision,
                        "setup_signature": setup,
                        "artifact_paths": {
                            concept: str(
                                u_root
                                / checkpoints[0]
                                / f"layer_{layer}"
                                / f"{concept}.safetensors"
                            )
                        },
                    }
                },
                "final_main": {
                    "status": "ok",
                    "revision": revision,
                    "setup_signature": setup,
                    "artifact_paths": {
                        concept: {
                            str(layer): str(
                                u_root
                                / checkpoints[0]
                                / f"layer_{layer}"
                                / f"{concept}.safetensors"
                            )
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (metrics_root.parent / "stability.json").write_text(
        json.dumps(
            {
                "model": model,
                "checkpoint_order": list(checkpoints),
                "layers_order": [layer],
                "setup_signature": setup,
                "reference": checkpoints[0],
                "layers": {"3": {"pos": stability_sign, "neg": stability_sign}},
            }
        ),
        encoding="utf-8",
    )

    report = validator.validate_result_tree(tmp_path, "sft")
    assert report.ok, report.errors
    bad_manifest = manifest_root / f"{model}__{checkpoints[0]}.json"
    changed = json.loads(bad_manifest.read_text())
    changed["revision"] = checkpoints[0]
    bad_manifest.write_text(json.dumps(changed), encoding="utf-8")
    report = validator.validate_result_tree(tmp_path, "sft")
    assert not report.ok
    assert any("immutable commit SHA" in error for error in report.errors)


def test_stability_values_and_k_are_strictly_validated():
    row = {"a": "step1000", "b": "step6000", "subsim": float("nan"), "k": 2}
    error = validator._stability_row_error(
        row,
        "pairwise",
        {"step1000": 2, "step6000": 2},
    )
    assert error is not None
    assert "subsim" in error


@pytest.mark.parametrize("bad_k", [-1, 1.5, True])
def test_stability_k_must_be_nonnegative_integer(bad_k):
    row = {"a": "step1000", "b": "step6000", "subsim": 0.5, "k": bad_k}
    error = validator._stability_row_error(
        row,
        "pairwise",
        {"step1000": 2, "step6000": 2},
    )
    assert error is not None
    assert "k" in error
