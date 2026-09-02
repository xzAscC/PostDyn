from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from postdyn import think_32b_differential_validator as validator


def _tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(validator, "EXPECTED_D_MODEL", 4)
    monkeypatch.setattr(validator, "LAYERS", (6,))
    monkeypatch.setitem(validator.CHECKPOINTS_BY_TRAJECTORY, "rlvr", ("step_050",))
    monkeypatch.setitem(validator.REVISIONS_BY_TRAJECTORY, "rlvr", {"step_050": "r1"})
    root = tmp_path
    model = "olmo3-32b-think-rlvr"
    ck = "step_050"
    setup = "setup-v1"
    concepts = list(validator.CONCEPTS)
    basis_pos = torch.eye(4)[:, :1].contiguous()
    basis_neg = torch.eye(4)[:, 3:4].contiguous()
    tensors = {
        "U_pos": basis_pos,
        "U_neg": basis_neg,
        "eigenvalues_pos": torch.tensor([2.0]),
        "eigenvalues_neg": torch.tensor([1.0]),
        "U_pos_full": basis_pos.clone(),
        "U_neg_full": basis_neg.clone(),
        "eigenvalues_signed": torch.tensor([2.0, 0.0, 0.0, -1.0]),
        "eigenvectors_signed": torch.eye(4),
    }

    def metadata(concept: str) -> dict[str, object]:
        return {
            "concept": concept,
            "model": model,
            "checkpoint": ck,
            "revision": "r1",
            "setup_signature": setup,
            "layer": 6,
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
            "loader_provenance": validator.STATIC_NF4_PROVENANCE,
            "extraction_protocol": validator.EXPECTED_PROTOCOL,
        }

    meta = metadata(concepts[0])
    base = root / "U" / model / ck / "layer_6"
    base.mkdir(parents=True)
    for concept in concepts:
        save_file(tensors, str(base / f"{concept}.safetensors"))
        (base / f"{concept}.json").write_text(
            json.dumps(metadata(concept)), encoding="utf-8"
        )
    metrics = {
        "model": model,
        "checkpoint": ck,
        "revision": "r1",
        "layer": 6,
        "setup_signature": setup,
        "tau": 0.95,
        "n_samples": 1000,
        "extraction_protocol": validator.EXPECTED_PROTOCOL,
        "concepts": {
            concept: {
                "concept": concept,
                "k_pos": 1,
                "k_neg": 1,
                "d_eff_pos": 1.0,
                "d_eff_neg": 1.0,
                "energy_pos": 4.0,
                "energy_neg": 1.0,
                "frobenius_strength_pos": 2.0,
                "frobenius_strength_neg": 1.0,
                "r_pos": 1.0,
                "r_neg": 1.0,
            }
            for concept in concepts
        },
        "three_metrics": {
            "retained_K": {concept: {"pos": 1, "neg": 1} for concept in concepts},
            "d_eff": {concept: {"pos": 1.0, "neg": 1.0} for concept in concepts},
        },
    }
    metrics_dir = root / "metrics" / model / ck
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "layer_6.json").write_text(json.dumps(metrics), encoding="utf-8")
    runtime = {
        "placement": "gpu-only",
        "quant_backend": "bitsandbytes",
        "safetensors": True,
        "bitsandbytes_version": "0.49.2",
        "transformers_version": "4.57.1",
        "accelerate_version": "1.0",
        "device_map": {"": "cuda:0"},
        "peak_vram_bytes": 1,
        "fallback_reason": None,
    }
    manifest = {
        "model": model,
        "checkpoint": ck,
        "revision": "r1",
        "scale": "32b",
        "layers": list(validator.LAYERS),
        "concepts": concepts,
        "setup_signature": setup,
        "status": "ok",
        "extraction_protocol": validator.EXPECTED_PROTOCOL,
        "canonical_protocol": True,
        "loader_provenance": validator.STATIC_NF4_PROVENANCE,
        "runtime_provenance": runtime,
    }
    manifest_dir = root / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / f"{model}__{ck}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (metrics_dir.parent.parent / "summary.json").write_text(
        json.dumps(
            {
                "concepts": concepts,
                "setup_signature": setup,
                "checkpoints": [ck],
                "layers": [6],
                "fixed_points": {
                    "base": {"status": "unavailable"},
                    "dpo": {"status": "unavailable"},
                },
                "final_main": {"status": "skipped"},
                "histogram": {
                    "6": {
                        concept: {"bins": 32, "range": [0.0, 1.0], "edges": [0.0] * 33}
                        for concept in concepts
                    }
                },
                "n_rows": 1,
                "rows": [
                    {
                        "model": model,
                        "hf_id": validator.OLMO3_VARIANTS["olmo3-32b-think-rlvr"].hf_id,
                        "checkpoint": ck,
                        "layer": 6,
                        "retained_K": {
                            concept: {"pos": 1, "neg": 1} for concept in concepts
                        },
                        "d_eff": {
                            concept: {"pos": 1.0, "neg": 1.0} for concept in concepts
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir.parent.parent / "stability.json").write_text(
        json.dumps(
            {
                "setup_signature": setup,
                "model": model,
                "checkpoint_order": [ck],
                "layers_order": [6],
                "reference": ck,
                "histogram": {
                    "6": {
                        concept: {"bins": 32, "range": [0.0, 1.0], "edges": [0.0] * 33}
                        for concept in concepts
                    }
                },
                "layers": {
                    "6": {
                        "pos": {
                            "pairwise": {concept: [] for concept in concepts},
                            "consecutive": {concept: [] for concept in concepts},
                            "vs_reference": {
                                concept: [
                                    {
                                        "checkpoint": ck,
                                        "reference": ck,
                                        "subsim": 1.0,
                                        "k": 1,
                                    }
                                ]
                                for concept in concepts
                            },
                        },
                        "neg": {
                            "pairwise": {concept: [] for concept in concepts},
                            "consecutive": {concept: [] for concept in concepts},
                            "vs_reference": {
                                concept: [
                                    {
                                        "checkpoint": ck,
                                        "reference": ck,
                                        "subsim": 1.0,
                                        "k": 1,
                                    }
                                ]
                                for concept in concepts
                            },
                            "residual_to_final": {
                                ck: {
                                    concept: {"pos": {}, "neg": {}}
                                    for concept in concepts
                                }
                            },
                            "reference_robustness": {
                                ck: {
                                    concept: {
                                        "pos": {"subsim": 1.0, "k": 1},
                                        "neg": {"subsim": 1.0, "k": 1},
                                    }
                                    for concept in (
                                        "math_vs_code",
                                        "math_vs_instruction_following",
                                        "math_vs_general_reasoning",
                                    )
                                }
                            },
                        },
                        "residual_to_final": {
                            ck: {
                                concept: {
                                    sign: {
                                        "defined": True,
                                        "k_final": 1,
                                        "d_res": 1,
                                        "observed": 1.0,
                                        "chance": 0.5,
                                        "excess": 0.5,
                                    }
                                    for sign in ("pos", "neg")
                                }
                                for concept in concepts
                            }
                        },
                        "reference_robustness": {
                            ck: {
                                concept: {
                                    "pos": {"subsim": 1.0, "k": 1},
                                    "neg": {"subsim": 1.0, "k": 1},
                                }
                                for concept in (
                                    "math_vs_code",
                                    "math_vs_instruction_following",
                                    "math_vs_general_reasoning",
                                )
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return root, Path(setup)


def test_valid_current_writer_fixture_passes(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert report.ok


@pytest.mark.parametrize("mutation", ["sidecar_core", "fixed_point", "histogram"])
def test_writer_contract_mutations_are_rejected(tmp_path, monkeypatch, mutation):
    root, setup = _tree(tmp_path, monkeypatch)
    if mutation == "sidecar_core":
        path = (
            root
            / f"U/olmo3-32b-think-rlvr/step_050/layer_6/{validator.CONCEPTS[0]}.json"
        )
        data = json.loads(path.read_text())
        del data["energy_pos"]
        path.write_text(json.dumps(data))
    else:
        path = root / "metrics/summary.json"
        data = json.loads(path.read_text())
        if mutation == "fixed_point":
            del data["fixed_points"]["dpo"]
        else:
            del data["histogram"]["6"][validator.CONCEPTS[0]]
        path.write_text(json.dumps(data))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok


def test_cli_json_report_and_trajectory_are_model_free(tmp_path, monkeypatch, capsys):
    root, setup = _tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        validator, "_expected_setup_signature", lambda *args: str(setup)
    )
    assert validator.main([str(root), "--trajectory", "rlvr", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["trajectory"] == "rlvr"


def _add_final_main(root: Path, relative_paths: bool = False) -> None:
    model = "olmo3-32b-think-rlvr"
    final_root = root / "final_points"
    for concept in validator.CONCEPTS:
        source = root / "U" / model / "step_050" / "layer_6" / f"{concept}.safetensors"
        target = final_root / "U" / model / "rlvr_main" / "layer_6" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sidecar = json.loads(source.with_suffix(".json").read_text())
        sidecar.update(
            {
                "checkpoint": "rlvr_main",
                "revision": "a" * 40,
                "setup_signature": "final-setup",
            }
        )
        target.with_suffix(".json").write_text(json.dumps(sidecar))
    path_value = lambda path: (
        str(path.relative_to(root.parent)) if relative_paths else str(path)
    )
    summary_path = root / "metrics" / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["final_main"] = {
        "model_key": validator.MODEL_BY_TRAJECTORY["rlvr"],
        "model": validator.OLMO3_VARIANTS[validator.MODEL_BY_TRAJECTORY["rlvr"]].hf_id,
        "checkpoint": "rlvr_main",
        "revision": "a" * 40,
        "setup_signature": "final-setup",
        "root": path_value(final_root),
        "status": "ok",
        "artifact_paths": {
            concept: {
                "6": path_value(
                    final_root
                    / "U"
                    / model
                    / "rlvr_main"
                    / "layer_6"
                    / f"{concept}.safetensors"
                )
            }
            for concept in validator.CONCEPTS
        },
    }
    summary_path.write_text(json.dumps(summary))
    stability_path = root / "metrics" / "stability.json"
    stability = json.loads(stability_path.read_text())
    stability["final_main"] = {
        key: summary["final_main"][key]
        for key in ("checkpoint", "setup_signature", "revision", "root")
    }
    stability_path.write_text(json.dumps(stability))


def test_writer_shaped_successful_final_main_accepts_relative_output_paths(
    tmp_path, monkeypatch
):
    root, setup = _tree(tmp_path, monkeypatch)
    _add_final_main(root, relative_paths=True)
    monkeypatch.chdir(root.parent)
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert report.ok


@pytest.mark.parametrize("mutation", ["model_key", "missing_artifact", "linkage"])
def test_successful_final_main_provenance_mutations_are_rejected(
    tmp_path, monkeypatch, mutation
):
    root, setup = _tree(tmp_path, monkeypatch)
    _add_final_main(root)
    summary_path = root / "metrics" / "summary.json"
    summary = json.loads(summary_path.read_text())
    if mutation == "model_key":
        summary["final_main"]["model_key"] = "olmo3-32b-think-sft"
    elif mutation == "missing_artifact":
        path = Path(summary["final_main"]["artifact_paths"][validator.CONCEPTS[0]]["6"])
        path.unlink()
    else:
        stability_path = root / "metrics" / "stability.json"
        stability = json.loads(stability_path.read_text())
        stability["final_main"]["setup_signature"] = "wrong"
        stability_path.write_text(json.dumps(stability))
    summary_path.write_text(json.dumps(summary))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok


@pytest.mark.parametrize("mutation", ["extra", "missing", "range", "k"])
def test_reference_robustness_mutations_are_rejected(tmp_path, monkeypatch, mutation):
    root, setup = _tree(tmp_path, monkeypatch)
    path = root / "metrics" / "stability.json"
    data = json.loads(path.read_text())
    reference = data["layers"]["6"]["reference_robustness"]["step_050"]["math_vs_code"]
    if mutation == "extra":
        reference["extra"] = 1
    elif mutation == "missing":
        del reference["neg"]
    elif mutation == "range":
        reference["pos"]["subsim"] = 2.0
    else:
        reference["pos"]["k"] = -1
    path.write_text(json.dumps(data))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok


def test_residual_algebra_mutation_is_rejected(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    path = root / "metrics" / "stability.json"
    data = json.loads(path.read_text())
    residual = data["layers"]["6"]["residual_to_final"]["step_050"][
        validator.CONCEPTS[0]
    ]["pos"]
    residual["excess"] = 0.25
    path.write_text(json.dumps(data))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok


def test_large_float32_basis_uses_elementwise_gram_error(tmp_path):
    n = 5120
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
    first = validator._sampled_column_pairs(5120)
    assert first == validator._sampled_column_pairs(5120)
    assert (0, 1) in first
    assert (0, 2) in first
    assert (0, 7) in first
    assert len(first) <= 16 + 2 * 16 + 3 + 64


def test_partial_signed_eigensystem_is_rejected(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    concept = validator.CONCEPTS[0]
    path = root / f"U/olmo3-32b-think-rlvr/step_050/layer_6/{concept}.safetensors"
    save_file(
        {
            "U_pos": torch.eye(4)[:, :1].contiguous(),
            "U_neg": torch.eye(4)[:, 1:2].contiguous(),
            "eigenvalues_pos": torch.tensor([2.0]),
            "eigenvalues_neg": torch.tensor([1.0]),
            "U_pos_full": torch.eye(4)[:, :1].contiguous(),
            "U_neg_full": torch.eye(4)[:, 1:2].contiguous(),
            "eigenvalues_signed": torch.tensor([2.0, -1.0]),
            "eigenvectors_signed": torch.eye(4)[:, :2].contiguous(),
        },
        str(path),
    )
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert any("eigenvalues_signed" in error for error in report.errors)


@pytest.mark.parametrize(
    "corruption", ["eigenvectors", "positive_basis", "retained_basis"]
)
def test_full_signed_eigensystem_integrity_is_rejected(
    tmp_path, monkeypatch, corruption
):
    root, setup = _tree(tmp_path, monkeypatch)
    concept = validator.CONCEPTS[0]
    path = root / f"U/olmo3-32b-think-rlvr/step_050/layer_6/{concept}.safetensors"
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
    if corruption == "eigenvectors":
        tensors["eigenvectors_signed"][0, 0] = 2.0
    elif corruption == "positive_basis":
        tensors["U_pos_full"] = torch.eye(4)[:, 1:2].contiguous()
    else:
        tensors["U_pos"] = torch.eye(4)[:, 1:2].contiguous()
    save_file(tensors, str(path))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert any(
        "orthonormal" in error
        or "sign selection" in error
        or "retained columns" in error
        for error in report.errors
    )


def test_empty_sign_spectra_are_accepted_when_metadata_matches(tmp_path, monkeypatch):
    root, _ = _tree(tmp_path, monkeypatch)
    concept = validator.CONCEPTS[0]
    sidecar = root / f"U/olmo3-32b-think-rlvr/step_050/layer_6/{concept}.json"
    meta = json.loads(sidecar.read_text())
    meta.update(
        {
            "k_pos": 0,
            "k_neg": 0,
            "u_pos_shape": [4, 0],
            "u_neg_shape": [4, 0],
        }
    )
    tensors = {
        "U_pos": torch.zeros((4, 0)),
        "U_neg": torch.zeros((4, 0)),
        "eigenvalues_pos": torch.empty(0),
        "eigenvalues_neg": torch.empty(0),
        "U_pos_full": torch.zeros((4, 0)),
        "U_neg_full": torch.zeros((4, 0)),
        "eigenvalues_signed": torch.zeros(4),
        "eigenvectors_signed": torch.eye(4),
    }
    monkeypatch.setattr(validator, "load_file", lambda *args, **kwargs: tensors)
    monkeypatch.setattr(validator, "_read", lambda _: meta)
    assert (
        validator._basis_error(
            sidecar.with_suffix(".safetensors"),
            sidecar,
            "olmo3-32b-think-rlvr",
            "step_050",
            "r1",
            "setup-v1",
            6,
            expected_concept=concept,
            require_full=True,
        )
        is None
    )


def test_full_publication_helper_uses_complete_trajectory(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setitem(
        validator.CHECKPOINTS_BY_TRAJECTORY,
        "rlvr",
        ("step_050", "step_100"),
    )
    monkeypatch.setitem(
        validator.REVISIONS_BY_TRAJECTORY,
        "rlvr",
        {"step_050": "r1", "step_100": "r2"},
    )
    monkeypatch.setattr(
        validator,
        "_expected_setup_signature",
        lambda *args: "full-setup",
    )

    def fake_validate(*args, **kwargs):
        seen.update(kwargs)
        return validator.ValidationReport(str(tmp_path), "rlvr")

    monkeypatch.setattr(validator, "validate_result_tree", fake_validate)
    validator.validate_full_canonical_publication(tmp_path, "rlvr")
    assert seen == {
        "checkpoints": ["step_050", "step_100"],
        "layers": list(validator.LAYERS),
        "expected_setup_signature": "full-setup",
        "require_publications": True,
    }


def test_altered_bounded_subsim_is_rejected(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    path = root / "metrics/stability.json"
    data = json.loads(path.read_text())
    data["layers"]["6"]["pos"]["vs_reference"][validator.CONCEPTS[0]][0]["subsim"] = 0.5
    path.write_text(json.dumps(data))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert any("SubSim disagrees" in error for error in report.errors)


def test_subset_only_publication_is_rejected_by_full_gate(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    monkeypatch.setitem(
        validator.CHECKPOINTS_BY_TRAJECTORY,
        "rlvr",
        ("step_050", "step_100"),
    )
    monkeypatch.setitem(
        validator.REVISIONS_BY_TRAJECTORY,
        "rlvr",
        {"step_050": "r1", "step_100": "r2"},
    )
    report = validator.validate_full_canonical_publication(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert any("coverage" in error or "step_100" in error for error in report.errors)


@pytest.mark.parametrize(
    "mutation", ["nan", "keys", "orthonormal", "provenance", "status"]
)
def test_mutation_matrix_rejects_publication(tmp_path, monkeypatch, mutation):
    root, setup = _tree(tmp_path, monkeypatch)
    concept = validator.CONCEPTS[0]
    path = root / f"U/olmo3-32b-think-rlvr/step_050/layer_6/{concept}.safetensors"
    if mutation == "nan":
        save_file(
            {
                "U_pos": torch.tensor([[float("nan")], [0.0], [0.0], [0.0]]),
                "U_neg": torch.eye(4)[:, :1].contiguous(),
                "eigenvalues_pos": torch.tensor([2.0]),
                "eigenvalues_neg": torch.tensor([1.0]),
            },
            str(path),
        )
    elif mutation == "keys":
        save_file({"U_pos": torch.eye(4)[:, :1].contiguous()}, str(path))
    elif mutation == "orthonormal":
        bad = torch.full((4, 1), 2.0)
        save_file(
            {
                "U_pos": bad,
                "U_neg": torch.eye(4)[:, :1].contiguous(),
                "eigenvalues_pos": torch.tensor([2.0]),
                "eigenvalues_neg": torch.tensor([1.0]),
            },
            str(path),
        )
    elif mutation == "provenance":
        sidecar = root / f"U/olmo3-32b-think-rlvr/step_050/layer_6/{concept}.json"
        data = json.loads(sidecar.read_text())
        data["revision"] = "wrong"
        sidecar.write_text(json.dumps(data))
    else:
        manifest = root / "manifests/olmo3-32b-think-rlvr__step_050.json"
        data = json.loads(manifest.read_text())
        data["status"] = "failed"
        manifest.write_text(json.dumps(data))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok


def test_checkpoint_failure_cannot_be_ok_manifest(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    manifest = root / "manifests/olmo3-32b-think-rlvr__step_050.json"
    data = json.loads(manifest.read_text())
    data["status"] = "failed"
    manifest.write_text(json.dumps(data))
    report = validator.validate_checkpoint_tree(
        root, "rlvr", "step_050", layers=[6], expected_setup_signature=str(setup)
    )
    assert not report.ok


def test_full_failure_blocks_summary(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    summary = root / "metrics/summary.json"
    summary.write_text(json.dumps({"setup_signature": "wrong", "rows": []}))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert "summary" in report.checks


def test_empty_stability_publication_is_rejected(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    stability = json.loads((root / "metrics/stability.json").read_text())
    stability["layers"]["6"] = {}
    (root / "metrics/stability.json").write_text(json.dumps(stability))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert any("stability" in error for error in report.errors)


def test_misordered_stability_row_and_inconsistent_k_are_rejected(
    tmp_path, monkeypatch
):
    root, setup = _tree(tmp_path, monkeypatch)
    stability = json.loads((root / "metrics/stability.json").read_text())
    row = stability["layers"]["6"]["pos"]["vs_reference"][validator.CONCEPTS[0]][0]
    row["k"] = 0
    (root / "metrics/stability.json").write_text(json.dumps(stability))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
    assert any("stability" in error for error in report.errors)


def test_missing_stability_sign_block_is_rejected(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    stability = json.loads((root / "metrics/stability.json").read_text())
    stability["layers"]["6"] = {"pos": {}}
    (root / "metrics/stability.json").write_text(json.dumps(stability))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok


def test_summary_metric_payload_must_match_layer_metrics(tmp_path, monkeypatch):
    root, setup = _tree(tmp_path, monkeypatch)
    summary = json.loads((root / "metrics/summary.json").read_text())
    summary["rows"][0]["retained_K"] = {"math_vs_text": {"pos": 9, "neg": 1}}
    (root / "metrics/summary.json").write_text(json.dumps(summary))
    report = validator.validate_result_tree(
        root, "rlvr", expected_setup_signature=str(setup)
    )
    assert not report.ok
