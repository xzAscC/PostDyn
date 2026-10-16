from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plot_weight_diff import load_rows, main, series  # noqa: E402


def _rows() -> list[dict]:
    rows = []
    for layer in (1, 0):
        rows.append({"name": f"model.layers.{layer}.self_attn.q_proj.weight", "layer": layer,
                     "component": "self_attn.q_proj.weight", "sparsity": 0.5 + layer / 10, "rank_ratio": 1.0})
        rows.append({"name": f"model.layers.{layer}.input_layernorm.weight", "layer": layer,
                     "component": "input_layernorm.weight", "sparsity": 0.9})
    rows.append({"name": "lm_head.weight", "layer": None, "component": "lm_head.weight", "sparsity": 0.1})
    return rows


def test_series_sorts_layers_and_skips_missing_metrics() -> None:
    out = series(_rows(), "rank_ratio")
    assert out == {"self_attn.q_proj.weight": ([0, 1], [1.0, 1.0])}
    sparsity = series(_rows(), "sparsity")
    assert sparsity["self_attn.q_proj.weight"] == ([0, 1], [0.5, 0.6])
    assert "lm_head.weight" not in sparsity


def test_load_rows_keeps_last_row_per_tensor(tmp_path: Path) -> None:
    path = tmp_path / "final.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"name": "a", "x": 1}, {"name": "a", "x": 2}]) + "\n")
    assert load_rows(path) == [{"name": "a", "x": 2}]


def test_main_writes_one_pdf_per_metric(tmp_path: Path) -> None:
    logs = tmp_path / "weights"
    for run in ("olmo3-7b_sft", "qwen-scaleml_rl"):
        (logs / run).mkdir(parents=True)
        (logs / run / "final.jsonl").write_text("\n".join(json.dumps(r) for r in _rows()) + "\n")
        (logs / run / "final.summary.json").write_text("{}")
    main(["--logs", str(logs), "--out-dir", str(tmp_path / "figs"), "--metrics", "sparsity", "rank_ratio"])
    # only sparsity has norm-weight values, so only it gets a vectors figure
    assert sorted(p.name for p in (tmp_path / "figs").iterdir()) == [
        "weights_final_rank_ratio.pdf", "weights_final_sparsity.pdf", "weights_final_sparsity_vectors.pdf"]
    with pytest.raises(SystemExit):
        main(["--logs", str(logs), "--out-dir", str(tmp_path / "figs"), "--metrics", "nope"])
