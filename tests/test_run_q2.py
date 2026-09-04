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
        exp1.require_bases(tmp_path, "math", [0], "rlvr")


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
            return {
                "offset_mapping": [
                    [(0, 1), (1, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8)]
                ]
            }

    # Captured states longer than prompt + re-tokenized generation (trailing
    # tokens beyond the generated span, e.g. boundary merges or appended
    # EOS). The old `state[-len(offsets):]` formula then misaligns by the
    # surplus: it would select rows [6, 8, 11]; prompt-relative selection
    # must be rows [4, 6, 9].
    trailing = torch.arange(12 * 3, dtype=torch.float32).reshape(1, 12, 3)
    states = exp2.sentence_final_states(
        Tokenizer(), [10, 11, 12], "a. b. c.", {2: trailing}, prompt_token_len=3
    )
    torch.testing.assert_close(states, trailing[0, [4, 6, 9]])
    assert not torch.allclose(states, trailing[0, [6, 8, 11]])

    aligned = torch.arange(10 * 3, dtype=torch.float32).reshape(1, 10, 3)
    states = exp2.sentence_final_states(
        Tokenizer(), [10, 11, 12], "a. b. c.", {2: aligned}, prompt_token_len=3
    )
    torch.testing.assert_close(states, aligned[0, [4, 6, 9]])

    zero_prompt = exp2.sentence_final_states(
        Tokenizer(), [], "a. b. c.", {2: aligned[0, :7]}, prompt_token_len=0
    )
    torch.testing.assert_close(zero_prompt, aligned[0, [1, 3, 6]])


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


def test_exp1_resume_rejects_different_q1_root(tmp_path: Path) -> None:
    output = tmp_path / "exp1"
    base = [
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
    subprocess.run(base, check=True, capture_output=True, text=True, env=env)
    mismatch = subprocess.run(
        [
            *base[: base.index(str(tmp_path / "q1"))],
            str(tmp_path / "other-q1"),
            *base[base.index(str(tmp_path / "q1")) + 1 :],
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert mismatch.returncode != 0
    assert "resume identity mismatch" in mismatch.stderr


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


def test_exp3_resume_rejects_different_sft_lr(tmp_path: Path) -> None:
    output = tmp_path / "exp3"
    base = [
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
    subprocess.run(base, check=True, capture_output=True, text=True, env=env)
    mismatch = subprocess.run(
        [*base, "--sft-lr", "5e-5"], capture_output=True, text=True, env=env
    )
    assert mismatch.returncode != 0
    assert "resume identity mismatch" in mismatch.stderr


def test_exp2_covariance_uses_float64_and_t_minus_one_cap() -> None:
    exp2 = load_script("run_q2_exp2")
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    vals, vecs, k = exp2.solution_eigensystem(x, K=9)
    assert vals.dtype == torch.float64
    assert vecs.dtype == torch.float64
    assert k == 1
    assert vals.shape == (2,)


def test_exp2_comparisons_slice_global_bands_to_solution_k() -> None:
    exp2 = load_script("run_q2_exp2")
    solution = torch.eye(4, dtype=torch.float64)[:, :2]
    high = torch.eye(4, dtype=torch.float64)[:, :3]
    low = torch.flip(torch.eye(4, dtype=torch.float64), dims=(1,))[:, :3]

    assert exp2.comparison_subsims(solution, high, low) == (1.0, 0.0)


def test_exp2_summary_groups_both_subsims_and_variance() -> None:
    exp2 = load_script("run_q2_exp2")
    rows = [
        {"correct": True, "V_i": 2.0, "subsim_high": 0.8, "subsim_low": 0.2},
        {"correct": False, "V_i": 4.0, "subsim_high": 0.4, "subsim_low": 0.6},
    ]

    summary = exp2.group_summary(rows)

    assert summary == {
        "correct": {
            "n": 1,
            "mean_V_i": 2.0,
            "mean_subsim_high": 0.8,
            "mean_subsim_low": 0.2,
        },
        "incorrect": {
            "n": 1,
            "mean_V_i": 4.0,
            "mean_subsim_high": 0.4,
            "mean_subsim_low": 0.6,
        },
    }


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
