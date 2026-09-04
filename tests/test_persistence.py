from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from postdyn.persistence import (
    RunDir,
    append_jsonl,
    atomic_write_json,
    load_eigensystem,
    save_eigensystem,
    tee_log,
)


def test_atomic_write_json_writes_valid_json_and_leaves_no_tmp(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    atomic_write_json(path, {"ok": [1, 2]})
    assert json.loads(path.read_text()) == {"ok": [1, 2]}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_preserves_original_on_serialize_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.json"
    path.write_text('{"original": true}')
    with pytest.raises((TypeError, ValueError)):
        atomic_write_json(path, {"bad": object()})
    assert path.read_text() == '{"original": true}'


def test_append_jsonl_appends_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    for i in range(3):
        append_jsonl(path, {"i": i})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"i": 0}, {"i": 1}, {"i": 2}]


def test_rundir_paths_completed_units_and_manifest(tmp_path: Path) -> None:
    run = RunDir(tmp_path)
    assert run.path("logs", "run.log") == tmp_path / "logs" / "run.log"
    run.path("metrics.jsonl").write_text(
        '{"checkpoint": "c1", "layer": 2, "domain": "math"}\n'
        '{"checkpoint": "c2", "layer": 3, "domain": "code"}\n'
    )
    assert run.completed_units() == {("c1", 2, "math"), ("c2", 3, "code")}
    manifest = run.manifest()
    values = vars(manifest) if hasattr(manifest, "__dict__") else manifest
    text = repr(values)
    for expected in ("repo", "revision", "dataset", "seed", "params"):
        assert expected in text


def test_eigensystem_roundtrip_uses_fp32_safetensors(tmp_path: Path) -> None:
    base = tmp_path / "eig" / "math"
    vals = torch.tensor([3.0, 1.0], dtype=torch.float64)
    vecs = torch.eye(2, dtype=torch.float64)
    save_eigensystem(base, vals, vecs)
    loaded_vals, loaded_vecs = load_eigensystem(base)
    assert loaded_vecs.shape == (2, 2)
    assert loaded_vecs.dtype == torch.float32
    torch.testing.assert_close(loaded_vals, vals)
    torch.testing.assert_close(loaded_vecs, vecs.float())


def test_tee_log_duplicates_stdout(tmp_path: Path, capsys) -> None:
    run = RunDir(tmp_path / "logs" / "run")
    with tee_log(run) as _:
        print("tee-marker")
    assert "tee-marker" in capsys.readouterr().out
    assert "tee-marker" in run.path("run.log").read_text()
