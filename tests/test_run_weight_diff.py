from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from postdyn.persistence import close_all_jsonl_handles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_weight_diff import build_tasks, main  # noqa: E402


def _ckpt(root: Path, w: torch.Tensor) -> dict[str, str]:
    root.mkdir(parents=True)
    save_file({"model.layers.0.self_attn.q_proj.weight": w}, str(root / "model.safetensors"))
    return {"path": str(root)}


def _config(tmp_path: Path) -> Path:
    w = torch.eye(8, dtype=torch.bfloat16)
    mid, end = w.clone(), w.clone()
    mid[0, 1] = 0.5
    end[0, 1] = 1.0
    run = {
        "name": "toy_sft", "family": "toy", "row": "SFT",
        "start": _ckpt(tmp_path / "base", w),
        "final": _ckpt(tmp_path / "final", end),
        "intermediates": [{**_ckpt(tmp_path / "mid", mid), "step": 5}],
    }
    config = tmp_path / "runs.json"
    config.write_text(json.dumps({"runs": [run]}))
    return config


def test_build_tasks_labels_and_shards(tmp_path: Path) -> None:
    runs = json.loads(_config(tmp_path).read_text())["runs"]
    assert [t.label for t in build_tasks(runs, "final")] == ["final"]
    assert [t.label for t in build_tasks(runs, "intermediate")] == ["step000005"]
    assert [t.label for t in build_tasks(runs, "intermediate", shard=1, num_shards=2)] == []


def test_main_writes_rows_and_summary(tmp_path: Path) -> None:
    out = tmp_path / "logs" / "weights"
    args = ["--config", str(_config(tmp_path)), "--output", str(out), "--device", "cpu",
            "--basis-root", str(tmp_path / "basis")]
    main([*args, "--which", "final"])
    main([*args, "--which", "intermediate"])
    close_all_jsonl_handles()
    summary = json.loads((out / "toy_sft" / "final.summary.json").read_text())
    assert summary["sparsity"] == pytest.approx(63 / 64)
    assert summary["start"] == str(tmp_path / "base")
    assert (out / "toy_sft" / "step000005.jsonl").is_file()
    assert "toy_sft" in (out / "final-s0of1" / "run.log").read_text()


def test_main_rejects_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--config", str(_config(tmp_path)), "--output", str(tmp_path / "o"), "--runs", "nope"])
