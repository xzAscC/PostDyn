from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

q1 = importlib.import_module("scripts.run_q1")
robustness = importlib.import_module("scripts.run_q1_robustness")


def _run_args(output: Path) -> list[str]:
    return [
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--output",
        str(output),
        "--domains",
        "math,code",
        "--checkpoints",
        "base,sft,dpo,rlvr",
        "--limit",
        "6",
    ]


def test_q1_writes_metrics_eigensystems_analysis_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "q1"

    assert q1.main(_run_args(output)) == 0

    rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 * 2 * 2
    assert {tuple(row["unit"]) for row in rows} == {
        (checkpoint, layer, domain)
        for checkpoint in ("base", "sft", "dpo", "rlvr")
        for layer in (0, 1)
        for domain in ("math", "code")
    }
    eig_json = output / "eigensystems" / "base" / "0" / "math.json"
    eig_values = json.loads(eig_json.read_text())
    assert len(eig_values) == 8
    from postdyn.persistence import load_eigensystem

    values, vecs = load_eigensystem(eig_json.with_suffix(""))
    assert values.shape == (8,)
    assert vecs.shape == (8, 8)
    assert vecs.dtype == torch.float32

    summary = json.loads((output / "analysis" / "summary.json").read_text())
    assert summary["adjacent_pairs"]
    assert {"high", "mid", "low"}.issubset(summary["adjacent_pairs"][0]["subsim"])
    assert "mean" in summary["adjacent_pairs"][0]["reordering"]
    assert "vs_base" in summary["checkpoints"]["rlvr"]
    assert "vs_stage_final" in summary["checkpoints"]["sft"]

    before = (output / "metrics.jsonl").read_text()
    assert q1.main(_run_args(output)) == 0
    assert (output / "metrics.jsonl").read_text() == before


def test_q1_recomputes_unit_when_safetensors_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    assert q1.main(_run_args(output)) == 0

    missing = output / "eigensystems" / "base" / "0" / "math.safetensors"
    missing.unlink()

    assert q1.main(_run_args(output)) == 0
    assert missing.is_file()


def test_q1_rejects_resume_with_different_domains(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    assert q1.main(_run_args(output)) == 0

    changed = _run_args(output)
    changed[changed.index("math,code")] = "math"
    with pytest.raises(SystemExit, match="mismatch.*fresh --output"):
        q1.main(changed)


def test_robustness_writes_repeat_files_and_spread_summary(tmp_path: Path) -> None:
    output = tmp_path / "robustness"
    args = [
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--checkpoint",
        "rlvr",
        "--domain",
        "math",
        "--repeats",
        "3",
        "--output",
        str(output),
        "--limit",
        "6",
    ]

    assert robustness.main(args) == 0

    root = output / "7b" / "rlvr" / "math"
    for repeat in range(3):
        payload = json.loads((root / f"repeat_{repeat}.json").read_text())
        assert payload["repeat"] == repeat
        assert len(payload["eigenvalues"]) == 2
        assert "rank_stability" in payload
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary["spread"]["std_effective_rank"]) == 2
    assert "mean_subsim_high_across_pairs" in summary["spread"]


def test_cli_validation_rejects_bad_family_and_small_repeat_count() -> None:
    with pytest.raises(SystemExit):
        q1.parse_args(["--family", "bad", "--scale", "tiny"])
    with pytest.raises(SystemExit):
        robustness.parse_args(["--family", "7b", "--repeats", "1"])
