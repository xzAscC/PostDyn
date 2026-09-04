from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp1_missing_final_bases_fails_clearly(tmp_path: Path) -> None:
    exp1 = load_script("run_q2_exp1")
    with pytest.raises(SystemExit, match="Q1.*rlvr.*bases"):
        exp1.require_bases(tmp_path, "7b", "math", [0], 1, "rlvr")


def test_selection_ties_layer_then_alpha() -> None:
    exp1 = load_script("run_q2_exp1")
    rows = [
        {"layer": 2, "alpha": 1.0, "accuracy": 1.0},
        {"layer": 1, "alpha": 10.0, "accuracy": 1.0},
        {"layer": 1, "alpha": 0.1, "accuracy": 1.0},
    ]
    assert exp1.select_best(rows) == {"layer": 1, "alpha": 0.1}


def test_random_basis_is_stable_per_domain_layer() -> None:
    exp1 = load_script("run_q2_exp1")
    first = exp1.random_basis(6, 2, "math", 3)
    second = exp1.random_basis(6, 2, "math", 3)
    assert torch.equal(first, second)


def test_exp2_sentence_final_states_use_offsets() -> None:
    exp2 = load_script("run_q2_exp2")

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs["return_offsets_mapping"] is True
            return {"offset_mapping": [[(0, 3), (3, 4), (5, 8), (8, 9)]]}

    captured = torch.arange(6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    states = exp2.sentence_final_states(
        Tokenizer(), [10, 11, 12, 13], "One. Two!", {2: captured}, prompt_token_len=2
    )
    torch.testing.assert_close(states, captured[0, [3, 5]])


def test_exp1_resume_with_completed_validation_preserves_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "exp1"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_q2_exp1.py"),
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--q1-root",
        str(tmp_path / "q1"),
        "--domains",
        "math",
        "--limit",
        "1",
        "--output",
        str(output),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    selected = (output / "selected.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, env=env)
    assert second.returncode == 0, second.stderr
    assert (output / "selected.json").read_bytes() == selected


def test_exp3_resume_with_completed_validation_preserves_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "exp3"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_q2_exp3.py"),
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--q1-root",
        str(tmp_path / "q1"),
        "--domains",
        "math",
        "--limit",
        "1",
        "--output",
        str(output),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    selected = (output / "selected.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True, env=env)
    assert second.returncode == 0, second.stderr
    assert (output / "selected.json").read_bytes() == selected


def test_exp2_covariance_uses_float64_and_t_minus_one_cap() -> None:
    exp2 = load_script("run_q2_exp2")
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    vals, vecs, k = exp2.solution_eigensystem(x, K=9)
    assert vals.dtype == torch.float64
    assert vecs.dtype == torch.float64
    assert k == 1
    assert vals.shape == (2,)


def test_exp3_has_exact_conditions_and_separate_basis_stages() -> None:
    exp3 = load_script("run_q2_exp3")
    assert exp3.CONDITIONS == ("baseline", "sft_low", "rlvr_low")
    assert exp3.basis_stage("sft_low") == "sft"
    assert exp3.basis_stage("rlvr_low") == "rlvr"


def test_cli_rejects_invalid_family_and_alpha() -> None:
    exp1 = load_script("run_q2_exp1")
    with pytest.raises(SystemExit):
        exp1.parse_args(["--family", "bad", "--q1-root", "/tmp/q1"])
    with pytest.raises(SystemExit):
        exp1.parse_args(["--family", "7b", "--q1-root", "/tmp/q1", "--batch-size", "0"])


def test_incremental_resume_helpers(tmp_path: Path) -> None:
    exp1 = load_script("run_q2_exp1")
    path = tmp_path / "validation.jsonl"
    path.write_text(
        json.dumps({"domain": "math", "layer": 0, "alpha": 1.0, "condition": "high"})
        + "\n"
    )
    assert exp1.completed_keys(path) == {("math", 0, 1.0, "high")}
