from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_args(q1_root: Path) -> list[str]:
    return [
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--q1-root",
        str(q1_root),
        "--device",
        "cpu",
        "--domains",
        "math",
        "--limit",
        "1",
    ]


def test_parser_accepts_common_and_model_flags() -> None:
    runner = load_script("run_q2_model")
    args = runner.parse_args(
        [
            "--family",
            "7b",
            "--q1-root",
            "/tmp/q",
            "--model",
            "sft",
            "--output-root",
            "/tmp/o",
            "--sft-lr",
            "5e-5",
            "--domains",
            "math",
            "--limit",
            "4",
        ]
    )
    assert args.model == "sft"
    assert args.output_root == Path("/tmp/o")
    assert args.sft_lr == "5e-5"
    assert args.domains == ["math"]
    assert args.limit == 4


def test_builds_exp_namespaces_via_each_exp_parse_args() -> None:
    runner = load_script("run_q2_model")
    args = runner.parse_args(
        [
            "--family",
            "7b",
            "--q1-root",
            "/tmp/q",
            "--model",
            "sft",
            "--output-root",
            "/tmp/o",
            "--sft-lr",
            "5e-5",
            "--domains",
            "math",
        ]
    )
    a1, a2, a3 = runner.build_experiment_args(args)
    assert a1.output == Path("/tmp/o/exp1_sft")
    assert a2.exp1_output == a1.output
    assert a2.output == Path("/tmp/o/exp2_sft")
    assert a3.output == Path("/tmp/o/exp3_sft")
    assert a1.sft_lr == "5e-5"
    assert a2.sft_lr == "5e-5"
    assert a3.sft_lr == "5e-5"
    assert all(hasattr(stage, "model") for stage in (a1, a2, a3))


def test_orchestrator_runs_exps_in_order_with_single_runtime(monkeypatch) -> None:
    runner = load_script("run_q2_model")
    args = runner.parse_args(
        ["--family", "7b", "--q1-root", "/tmp/q", "--scale", "tiny"]
    )
    sentinel = (object(), object())
    loaded = []
    calls = []

    monkeypatch.setattr(
        runner.common, "load_runtime", lambda *values: loaded.append(values) or sentinel
    )
    for name in ("exp1", "exp2", "exp3"):
        module = getattr(runner, name)

        def record(stage, module=module):
            def run_with(stage_args, loader):
                calls.append((stage, stage_args))
                assert loader() == sentinel

            return run_with

        monkeypatch.setattr(module, "run_with", record(name))

    runner.run(args)

    assert len(loaded) == 1
    assert [stage for stage, _ in calls] == ["exp1", "exp2", "exp3"]


def test_exp2_run_with_uses_injected_runtime(tmp_path: Path, monkeypatch) -> None:
    exp2 = load_script("run_q2_exp2")
    args = exp2.parse_args(
        [
            *_base_args(tmp_path / "q1"),
            "--exp1-output",
            str(tmp_path / "exp1"),
            "--output",
            str(tmp_path / "exp2"),
        ]
    )
    (tmp_path / "exp1").mkdir()
    (tmp_path / "exp1" / "selected.json").write_text(
        '{"math": {"layer": 0, "alpha": 1.0}}'
    )
    loaded = []
    monkeypatch.setattr(exp2.common, "load_runtime", lambda *_: loaded.append(True))
    exp2.run_with(args, lambda: (exp2.common.TinyModel(), exp2.common.TinyTokenizer()))
    assert loaded == []


def test_exp3_run_with_uses_injected_runtime(tmp_path: Path, monkeypatch) -> None:
    exp3 = load_script("run_q2_exp3")
    args = exp3.parse_args(
        [*_base_args(tmp_path / "q1"), "--output", str(tmp_path / "exp3")]
    )
    loaded = []
    monkeypatch.setattr(exp3.common, "load_runtime", lambda *_: loaded.append(True))
    exp3.run_with(args, lambda: (exp3.common.TinyModel(), exp3.common.TinyTokenizer()))
    assert loaded == []


def _run(script: str, output: Path, q1_root: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            *_base_args(q1_root),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr


def _without_r_bar(value):
    if isinstance(value, dict):
        return {
            key: _without_r_bar(item) for key, item in value.items() if key != "r_bar"
        }
    if isinstance(value, list):
        return [_without_r_bar(item) for item in value]
    return value


def test_orchestrator_tiny_end_to_end_matches_standalone(tmp_path: Path) -> None:
    standalone = tmp_path / "standalone"
    orchestrated = tmp_path / "orchestrated"
    q1_root = tmp_path / "q1"
    _run("run_q2_exp1.py", standalone / "exp1_rlvr", q1_root)
    _run("run_q2_exp2.py", standalone / "exp2_rlvr", q1_root)
    _run("run_q2_exp3.py", standalone / "exp3_rlvr", q1_root)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_q2_model.py"),
            *_base_args(q1_root),
            "--output-root",
            str(orchestrated),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr

    for stage in ("exp1_rlvr", "exp2_rlvr", "exp3_rlvr"):
        left = standalone / stage
        right = orchestrated / stage
        assert {
            path.relative_to(left) for path in left.rglob("*") if path.is_file()
        } == {path.relative_to(right) for path in right.rglob("*") if path.is_file()}
        for path in left.rglob("*"):
            if not path.is_file() or path.name in {"manifest.json", "run.log"}:
                continue
            other = right / path.relative_to(left)
            if path.name in {"selected.json", "summary.json"}:
                assert _without_r_bar(json.loads(path.read_text())) == _without_r_bar(
                    json.loads(other.read_text())
                )
            else:
                assert path.read_bytes() == other.read_bytes()
