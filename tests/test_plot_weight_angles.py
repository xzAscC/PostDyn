from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_weight_angles import angle_rows, bf16_floor, layer_curves, main  # noqa: E402


def _write(run_dir: Path, label: str, rows: list[dict], done: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{label}.angles.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if done:
        (run_dir / f"{label}.angles.summary.json").write_text("{}")


def _rows(scale: float = 1.0) -> list[dict]:
    rows = []
    for layer in (1, 0):
        rows.append({"name": f"model.layers.{layer}.mlp.up_proj.weight", "layer": layer, "component": "mlp.up_proj.weight",
                     "theta_mean": scale * (layer + 1), "theta_std": 0.5, "theta_ulp": 0.3 + layer})
        rows.append({"name": f"model.layers.{layer}.input_layernorm.weight", "layer": layer,
                     "component": "input_layernorm.weight", "theta_mean": 0.01, "theta_std": None, "theta_ulp": 0.2})
    return rows


def test_angle_rows_requires_finished_summary(tmp_path: Path) -> None:
    _write(tmp_path / "r", "final", _rows(), done=False)
    assert angle_rows(tmp_path / "r", "final") is None
    _write(tmp_path / "r", "final", _rows())
    assert len(angle_rows(tmp_path / "r", "final")) == 4


def test_layer_curves_split_matrices_and_vectors() -> None:
    mats = layer_curves(_rows(), matrices=True)
    assert mats == {"mlp.up_proj.weight": ([0, 1], [1.0, 2.0], [0.5, 0.5])}
    vecs = layer_curves(_rows(), matrices=False)
    assert vecs == {"input_layernorm.weight": ([0, 1], [0.01, 0.01], [0.0, 0.0])}


def test_bf16_floor_is_median_over_matrices() -> None:
    assert bf16_floor(_rows(), matrices=True) == pytest.approx(0.8)  # median of 0.3, 1.3
    assert bf16_floor(_rows(), matrices=False) == pytest.approx(0.2)


def test_main_writes_three_figures(tmp_path: Path) -> None:
    logs = tmp_path / "weights"
    _write(logs / "olmo3-7b_sft", "final", _rows())
    _write(logs / "olmo3-7b_sft", "step000010", _rows(0.5))
    _write(logs / "qwen-scaleml_rl", "final", _rows(2.0))
    main(["--logs", str(logs), "--out-dir", str(tmp_path / "figs")])
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == [
        "weights_angle_final_matrices.pdf", "weights_angle_final_vectors.pdf", "weights_angle_steps.pdf"]
