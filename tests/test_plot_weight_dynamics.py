from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_weight_dynamics import main, trajectory  # noqa: E402


def _write(run_dir: Path, label: str, values: list[float], done: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"name": f"model.layers.{i}.self_attn.q_proj.weight", "layer": i,
             "component": "self_attn.q_proj.weight", "sparsity": v, "one_minus_cos": v} for i, v in enumerate(values)]
    (run_dir / f"{label}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if done:
        (run_dir / f"{label}.summary.json").write_text("{}")


def test_trajectory_mean_std_over_layers_and_final_last(tmp_path: Path) -> None:
    run = tmp_path / "olmo3-7b_sft"
    _write(run, "step000020", [0.2, 0.4])
    _write(run, "step000010", [0.1, 0.3])
    _write(run, "step000030", [0.9, 0.9], done=False)  # still running: skipped
    _write(run, "final", [0.5, 0.7])
    traj = trajectory(run, "sparsity")["self_attn.q_proj.weight"]
    assert traj.steps == [10, 20]
    assert traj.mean == pytest.approx([0.2, 0.3])
    assert traj.std == pytest.approx([0.1, 0.1])
    assert traj.final == pytest.approx((0.6, 0.1))


def test_trajectory_uses_log_space_for_log_metrics(tmp_path: Path) -> None:
    run = tmp_path / "r"
    _write(run, "step000001", [1e-4, 1e-2])
    traj = trajectory(run, "one_minus_cos", log=True)["self_attn.q_proj.weight"]
    assert traj.mean == pytest.approx([1e-3])  # geometric mean
    assert traj.std == pytest.approx([math.log10(1e-2) - math.log10(1e-3)])  # std of log10 values


def test_main_writes_pdfs(tmp_path: Path) -> None:
    logs = tmp_path / "weights"
    _write(logs / "olmo3-7b_sft", "step000010", [0.1, 0.3])
    _write(logs / "olmo3-7b_sft", "final", [0.5, 0.7])
    main(["--logs", str(logs), "--out-dir", str(tmp_path / "figs"), "--metrics", "sparsity", "one_minus_cos"])
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == [
        "weights_steps_one_minus_cos.pdf", "weights_steps_sparsity.pdf"]


def test_final_x_uses_known_step_else_median_gap() -> None:
    from scripts.plot_weight_dynamics import final_x

    assert final_x("qwen-scaleml_sft", [20, 60, 80, 300]) == 320
    assert final_x("olmo3-7b_sft", [1000, 6000, 10000, 15000]) == 15000 + 5000  # median of 5000, 4000, 5000


def test_linear_band_is_clipped_to_valid_range() -> None:
    from scripts.plot_weight_dynamics import _band

    assert _band([99.0, 1.0], [2.0, 2.0], log=False, top=100) == ([97.0, 0.0], [100.0, 3.0])
    assert _band([99.0], [2.0], log=False) == ([97.0], [101.0])


def test_short_step_labels() -> None:
    from scripts.plot_weight_dynamics import _short

    assert [_short(s) for s in (950, 1596, 10790, 43000)] == ["950", "1.6k", "10.8k", "43k"]
