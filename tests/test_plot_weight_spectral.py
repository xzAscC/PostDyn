from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_weight_spectral import main, matrix_mean, per_layer, spectral_rows  # noqa: E402


def _rows(scale: float, curves: bool) -> list[dict]:
    rows = []
    for layer in (1, 0):
        r = {"name": f"model.layers.{layer}.mlp.down_proj.weight", "layer": layer, "component": "mlp.down_proj.weight",
             "pa_u_max": scale * (layer + 1), "pa_v_max": scale, "nss": scale / 10, "rel_update_norm": scale / 100,
             "k": 3}
        if curves:
            r.update(pa_u=[0.1, 0.2, scale], pa_v=[0.1, 0.3, scale], sv0_top=[3.0, 2.0, 1.0], sv1_top=[3.1, 2.0, 0.9])
        rows.append(r)
        rows.append({"name": f"model.layers.{layer}.input_layernorm.weight", "layer": layer,
                     "component": "input_layernorm.weight", "rel_update_norm": 0.5})
    return rows


def _write(run_dir: Path, label: str, rows: list[dict], done: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{label}.spectral.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    if done:
        (run_dir / f"{label}.spectral.summary.json").write_text("{}")


def test_spectral_rows_requires_summary(tmp_path: Path) -> None:
    _write(tmp_path / "r", "final", _rows(1, True), done=False)
    assert spectral_rows(tmp_path / "r", "final") is None
    _write(tmp_path / "r", "final", _rows(1, True))
    assert len(spectral_rows(tmp_path / "r", "final")) == 4


def test_per_layer_uses_matrices_only() -> None:
    assert per_layer(_rows(2.0, False), "pa_u_max") == {"mlp.down_proj.weight": ([0, 1], [2.0, 4.0])}
    assert "input_layernorm.weight" not in per_layer(_rows(2.0, False), "rel_update_norm")


def test_matrix_mean_is_geometric_over_matrices() -> None:
    assert matrix_mean(_rows(2.0, False), "pa_u_max") == pytest.approx((2.0 * 4.0) ** 0.5)


def test_main_writes_five_figures(tmp_path: Path) -> None:
    logs = tmp_path / "weights"
    _write(logs / "olmo3-7b_sft", "final", _rows(10.0, True))
    _write(logs / "olmo3-7b_rl", "final", _rows(0.1, True))
    _write(logs / "olmo3-7b_sft", "step000010", _rows(5.0, False))
    main(["--logs", str(logs), "--out-dir", str(tmp_path / "figs")])
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == [
        "weights_spectral_exemplar.pdf", "weights_spectral_final_nss.pdf", "weights_spectral_final_pa_u_max.pdf",
        "weights_spectral_final_rel_update_norm.pdf", "weights_spectral_steps.pdf"]
