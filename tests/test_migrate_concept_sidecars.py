"""Tests for scripts/migrate_concept_sidecars.py."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.migrate_concept_sidecars import (
    build_v1_sidecar,
    migrate_one,
    migrate_tree,
)
from postdyn.concept_dynamics import (
    EXPECTED_D_MODEL,
    SIDECAR_SCHEMA,
    SIDECAR_VERSION,
    validate_concept_sidecar,
)
from postdyn.rl_zero_experiment import EXPERIMENT_CONCEPTS, N_SAMPLES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_legacy_layer(
    root: Path,
    model: str,
    checkpoint: str,
    layer: int,
    concepts: list[str] | None = None,
) -> tuple[Path, Path]:
    concepts = list(concepts or EXPERIMENT_CONCEPTS)
    d = root / "concept_vectors" / model / checkpoint
    d.mkdir(parents=True, exist_ok=True)
    base = d / f"layer_{layer}"
    tensors: dict[str, torch.Tensor] = {}
    meta_concepts = []
    for idx, name in enumerate(sorted(concepts)):
        prefix = f"concept_{idx:04d}"
        for field in (
            "steering_vector",
            "raw_direction",
            "positive_mean",
            "negative_mean",
            "positive_std",
            "negative_std",
        ):
            tensors[f"{prefix}.{field}"] = torch.zeros(EXPECTED_D_MODEL)
        meta_concepts.append(
            {
                "name": name,
                "n_positive": N_SAMPLES,
                "n_negative": N_SAMPLES,
                "d_model": EXPECTED_D_MODEL,
            }
        )
    st = base.with_suffix(".safetensors")
    js = base.with_suffix(".json")
    save_file(tensors, str(st))
    js.write_text(
        json.dumps(
            {
                "concepts": meta_concepts,
                "layer_idx": layer,
                "model_name": model,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return js, st


def _fake_sources(n: int = N_SAMPLES) -> dict[str, tuple[list[str], list[str]]]:
    out: dict[str, tuple[list[str], list[str]]] = {}
    for name in EXPERIMENT_CONCEPTS:
        pos = [f"{name}-pos-{i}" for i in range(n)]
        neg = [f"{name}-neg-{i}" for i in range(n)]
        out[name] = (pos, neg)
    return out


def test_build_v1_and_validate(tmp_path: Path) -> None:
    js, _st = _write_legacy_layer(tmp_path, "olmo3-base", "main", 3)
    legacy = json.loads(js.read_text(encoding="utf-8"))
    sources = _fake_sources()
    v1 = build_v1_sidecar(
        legacy,
        model_name="olmo3-base",
        checkpoint="main",
        layer_idx=3,
        concept_sources=sources,
    )
    assert v1["version"] == SIDECAR_VERSION
    assert v1["schema"] == SIDECAR_SCHEMA
    assert validate_concept_sidecar(
        v1,
        expected_model_name="olmo3-base",
        expected_checkpoint="main",
        expected_layer_idx=3,
        expected_d_model=EXPECTED_D_MODEL,
        expected_max_seq_len=2048,
        expected_protocol="raw",
        expected_use_chat_template=False,
        expected_concept_sources=sources,
    )


def test_migrate_one_preserves_tensor_bytes(tmp_path: Path) -> None:
    js, st = _write_legacy_layer(tmp_path, "olmo3-base", "main", 3)
    before = _sha(st)
    sources = _fake_sources()
    row = migrate_one(js, sources, dry_run=False)
    assert row["status"] == "migrated"
    assert _sha(st) == before
    loaded = json.loads(js.read_text(encoding="utf-8"))
    assert loaded["version"] == SIDECAR_VERSION
    assert validate_concept_sidecar(
        loaded,
        expected_model_name="olmo3-base",
        expected_checkpoint="main",
        expected_layer_idx=3,
        expected_d_model=EXPECTED_D_MODEL,
        expected_max_seq_len=2048,
        expected_protocol="raw",
        expected_use_chat_template=False,
        expected_concept_sources=sources,
    )


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    js, st = _write_legacy_layer(tmp_path, "olmo3-base", "main", 3)
    before_js = js.read_text(encoding="utf-8")
    before_st = _sha(st)
    sources = _fake_sources()
    row = migrate_one(js, sources, dry_run=True)
    assert row["status"] == "would_migrate"
    assert js.read_text(encoding="utf-8") == before_js
    assert _sha(st) == before_st


def test_idempotent_second_run(tmp_path: Path) -> None:
    js, st = _write_legacy_layer(tmp_path, "olmo3-rl-zero-code", "step_100", 14)
    sources = _fake_sources()
    first = migrate_one(js, sources, dry_run=False)
    second = migrate_one(js, sources, dry_run=False)
    assert first["status"] == "migrated"
    assert second["status"] == "verified"
    assert first["tensor_sha256"] == second["tensor_sha256"] == _sha(st)


def test_migrate_tree_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_legacy_layer(tmp_path, "olmo3-base", "main", 3)
    _write_legacy_layer(tmp_path, "olmo3-rl-zero-code", "step_100", 6)
    sources = _fake_sources()
    monkeypatch.setattr(
        "experiments.migrate_concept_sidecars.load_canonical_concept_sources",
        lambda n_samples=N_SAMPLES: sources,
    )
    report = migrate_tree(tmp_path, dry_run=False)
    assert report["n_sidecars"] == 2
    assert report["counts"]["migrated"] == 2
    assert report["counts"]["re_extract_required"] == 0
