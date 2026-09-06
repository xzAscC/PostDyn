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


def test_exp1_identity_includes_runtime_and_rlvr_checkpoint() -> None:
    exp1 = load_script("run_q2_exp1")
    identity = exp1.identity_for(
        exp1.parse_args(
            [
                "--family",
                "7b",
                "--scale",
                "tiny",
                "--q1-root",
                "/tmp/q1",
                "--device",
                "cpu",
                "--batch-size",
                "3",
                "--limit",
                "2",
            ]
        )
    )
    assert identity["checkpoints"] == [["allenai/Olmo-3-7B-Think", "main"]]
    assert identity["device"] == "cpu"
    assert identity["batch_size"] == 3
    assert identity["limit"] == 2


def test_exp2_identity_binds_effective_selection_source_and_override(
    tmp_path: Path,
) -> None:
    exp2 = load_script("run_q2_exp2")
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps({"math": {"layer": 6, "alpha": 10.0}}))
    identity = exp2.identity_for(
        exp2.parse_args(
            [
                "--family",
                "7b",
                "--scale",
                "tiny",
                "--q1-root",
                "/tmp/q1",
                "--exp1-output",
                str(tmp_path),
                "--domains",
                "math",
                "--layer",
                "3",
                "--alpha",
                "1.0",
            ]
        ),
        selected_path,
        {"math": {"layer": 3, "alpha": 1.0}},
    )
    assert identity["selected_layers"] == {"math": 3}
    assert identity["selected_alphas"] == {"math": 1.0}
    assert identity["layer"] == {"math": 3}
    assert identity["alpha"] == {"math": 1.0}
    assert identity["selection_source"] == str(selected_path.resolve())
    assert identity["selection_source_hash"]


def test_exp3_identity_includes_both_base_checkpoints_and_sft_lr() -> None:
    exp3 = load_script("run_q2_exp3")
    identity = exp3.identity_for(
        exp3.parse_args(
            [
                "--family",
                "7b",
                "--scale",
                "tiny",
                "--q1-root",
                "/tmp/q1",
                "--sft-lr",
                "5e-5",
            ]
        )
    )
    assert identity["checkpoints"] == [
        ["allenai/Olmo-3-7B-Think-SFT", "main"],
        ["allenai/Olmo-3-7B-Think", "main"],
    ]
    assert identity["sft_lr"] == "5e-5"


@pytest.mark.parametrize(
    ("flag", "value", "identity_key"),
    [("--device", "cpu", "device"), ("--limit", "2", "limit")],
)
def test_exp1_resume_identity_rejects_runtime_changes(
    flag: str, value: str, identity_key: str
) -> None:
    exp1 = load_script("run_q2_exp1")
    first = {"family": "7b", "scale": "tiny", identity_key: "original"}
    second = dict(first, **{identity_key: value})
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        exp1.common.validate_identity(first, second)


def test_exp2_resume_identity_rejects_layer_override() -> None:
    exp2 = load_script("run_q2_exp2")
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        exp2.common.validate_identity(
            {"selected_layers": {"math": 3}},
            {"selected_layers": {"math": 6}},
        )


def test_exp3_resume_identity_rejects_sft_lr() -> None:
    exp3 = load_script("run_q2_exp3")
    with pytest.raises(SystemExit, match="resume identity mismatch"):
        exp3.common.validate_identity(
            {"sft_lr": "1e-4"},
            {"sft_lr": "5e-5"},
        )


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


def test_exp2_item_k_is_recomputed_for_each_solution() -> None:
    exp2 = load_script("run_q2_exp2")
    high = torch.eye(4, dtype=torch.float64)[:, :3]
    low = torch.eye(4, dtype=torch.float64)[:, 1:]
    first = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    second = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    _, first_subsims, first_k = exp2.item_subsims(first, 3, high, low)
    _, second_subsims, second_k = exp2.item_subsims(second, 3, high, low)

    assert first_k == 1
    assert first_subsims == (1.0, 0.0)
    assert second_k == 3
    assert second_subsims == pytest.approx((0.75, 0.75))


def test_exp2_subsim_vs_band_normalizes_by_solution_width() -> None:
    exp2 = load_script("run_q2_exp2")
    solution = torch.eye(4, dtype=torch.float64)[:, :2]
    band = torch.eye(4, dtype=torch.float64)[:, :3]

    assert exp2.subsim_vs_band(solution, band) == 1.0
    assert (
        exp2.subsim_vs_band(solution, torch.eye(4, dtype=torch.float64)[:, 2:]) == 0.0
    )
    rectangular_solution = torch.eye(4, dtype=torch.float64)[:, :1]
    rectangular_band = torch.eye(4, dtype=torch.float64)[:, :2]
    assert exp2.subsim_vs_band(rectangular_solution, rectangular_band) == 1.0


def test_exp2_global_bands_use_exact_first_and_last_k_columns() -> None:
    exp2 = load_script("run_q2_exp2")
    eigenvectors = torch.eye(8, dtype=torch.float64)
    K = 2
    high = eigenvectors[:, :K]
    low = eigenvectors[:, -K:]
    assert torch.equal(high, eigenvectors[:, [0, 1]])
    assert torch.equal(low, eigenvectors[:, [6, 7]])
    solution = eigenvectors[:, :1]
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
    assert exp3.CONDITIONS == ("baseline", "own_only", "replace")
    assert exp3.SELECTION_CONDITION == "replace"


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


def test_exp3_uses_aligned_replacement_conditions(tmp_path: Path) -> None:
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
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr

    for condition in ("baseline", "own_only", "replace"):
        assert (output / f"eval_math_{condition}.jsonl").is_file()
    validation = [
        json.loads(line)
        for line in (output / "validation.jsonl").read_text().splitlines()
    ]
    assert validation
    assert {row["condition"] for row in validation} == {"replace"}
    summary = json.loads((output / "summary.json").read_text())
    assert "math" in summary["selected"]
    assert summary["alignment"]["math"]


def test_exp3_replacement_formula_matches_slide_spec() -> None:
    exp3 = load_script("run_q2_exp3")
    exp1 = load_script("run_q2_exp1")
    from postdyn.intervention import procrustes_align, replace_basis

    torch.manual_seed(2)
    d, k = 9, 3
    U_s, _ = torch.linalg.qr(torch.randn(d, k))
    U_r, _ = torch.linalg.qr(torch.randn(d, k))
    rotation = procrustes_align(U_r, U_s)
    h = torch.randn(d)

    assert exp3.CONDITIONS == ("baseline", "own_only", "replace")
    assert exp3.SELECTION_CONDITION == "replace"
    spec = h - U_s @ (U_s.T @ h) + (U_r @ rotation) @ (U_s.T @ h)  # noqa: F841
    torch.testing.assert_close(
        replace_basis(h, U_s, U_r @ rotation, alpha=1.0), spec, atol=1e-6, rtol=1e-6
    )


def test_exp1_and_exp3_run_on_both_model_checkpoints(tmp_path: Path) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(script: str, extra: list[str], out: Path) -> None:
        done = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
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
                str(out),
                *extra,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert done.returncode == 0, done.stderr

    run("run_q2_exp1.py", ["--model", "sft"], tmp_path / "exp1_sft")
    identity = json.loads((tmp_path / "exp1_sft" / "manifest.json").read_text())
    assert identity["model"] == "sft"
    assert identity["checkpoints"] == [["allenai/Olmo-3-7B-Think-SFT", "main"]]

    run("run_q2_exp3.py", ["--model", "rlvr"], tmp_path / "exp3_rlvr")
    mirror = json.loads((tmp_path / "exp3_rlvr" / "manifest.json").read_text())
    assert mirror["model"] == "rlvr"
    for condition in ("baseline", "own_only", "replace"):
        assert (tmp_path / "exp3_rlvr" / f"eval_math_{condition}.jsonl").is_file()
    summary = json.loads((tmp_path / "exp3_rlvr" / "summary.json").read_text())
    assert summary["alignment"]["math"]


def test_exp3_grid_searches_layer_only_with_fixed_alpha() -> None:
    exp3 = load_script("run_q2_exp3")
    identity = exp3.identity_for(
        exp3.parse_args(["--family", "7b", "--scale", "tiny", "--q1-root", "/tmp/q1"])
    )
    assert identity["alpha"] == 1.0
    assert "alphas" not in identity

    rows = [
        {"layer": 3, "alpha": 1.0, "accuracy": 0.5},
        {"layer": 6, "alpha": 1.0, "accuracy": 0.8},
        {"layer": 9, "alpha": 1.0, "accuracy": 0.8},
    ]
    assert exp3.select_layer(rows) == {"layer": 6, "alpha": 1.0}


def test_exp1_identity_includes_sft_lr() -> None:
    exp1 = load_script("run_q2_exp1")
    args = exp1.parse_args(
        [
            "--family",
            "32b",
            "--scale",
            "tiny",
            "--q1-root",
            "/tmp/q1",
            "--sft-lr",
            "5e-5",
        ]
    )
    identity = exp1.identity_for(args)
    assert identity["sft_lr"] == "5e-5"
    assert identity["checkpoints"] == exp1.common.checkpoint_pairs(
        "32b", (args.model,), "5e-5"
    )


def test_exp2_identity_includes_sft_lr(tmp_path: Path) -> None:
    exp2 = load_script("run_q2_exp2")
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps({"math": {"layer": 0, "alpha": 1.0}}))
    args = exp2.parse_args(
        [
            "--family",
            "32b",
            "--scale",
            "tiny",
            "--q1-root",
            "/tmp/q1",
            "--sft-lr",
            "5e-5",
        ]
    )
    identity = exp2.identity_for(
        args, selected_path, {"math": {"layer": 0, "alpha": 1.0}}
    )
    assert identity["sft_lr"] == "5e-5"
    assert identity["checkpoints"] == exp2.common.checkpoint_pairs(
        "32b", (args.model,), "5e-5"
    )
