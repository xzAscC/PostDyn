from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from postdyn.persistence import close_all_jsonl_handles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_weight_diff import build_tasks, main, ready  # noqa: E402


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
    assert [t.label for t in build_tasks(runs, "all")] == ["final", "step000005"]


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


def test_main_skips_done_locked_and_unready_tasks(tmp_path: Path) -> None:
    import fcntl

    config = _config(tmp_path)
    runs = json.loads(config.read_text())
    runs["runs"][0]["intermediates"].append({"path": str(tmp_path / "not_downloaded"), "step": 9})
    config.write_text(json.dumps(runs))
    out = tmp_path / "logs" / "weights"
    lock = out / "toy_sft" / "final.lock"
    lock.parent.mkdir(parents=True)
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another worker owns the final task
        undone = main(["--config", str(config), "--output", str(out), "--device", "cpu", "--which", "all"])
    close_all_jsonl_handles()
    assert not (out / "toy_sft" / "final.jsonl").exists()
    assert (out / "toy_sft" / "step000005.summary.json").is_file()
    assert sorted(undone) == [("toy_sft", "final"), ("toy_sft", "step000009")]
    # a second worker finishes the final task and skips the one already done
    undone = main(["--config", str(config), "--output", str(out), "--device", "cpu", "--which", "all"])
    close_all_jsonl_handles()
    assert undone == [("toy_sft", "step000009")]
    assert len((out / "toy_sft" / "step000005.jsonl").read_text().splitlines()) == 1


def test_worker_does_not_wait_for_tasks_held_by_others(tmp_path: Path) -> None:
    import fcntl
    import time

    out = tmp_path / "logs" / "weights"
    config = _config(tmp_path)
    for label in ("final", "step000005"):
        (out / "toy_sft").mkdir(parents=True, exist_ok=True)
    locks = [(out / "toy_sft" / f"{label}.lock").open("w") for label in ("final", "step000005")]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    t0 = time.time()
    undone = main(["--config", str(config), "--output", str(out), "--device", "cpu",
                   "--which", "all", "--wait-minutes", "5"])
    assert time.time() - t0 < 30
    assert len(undone) == 2
    for lock in locks:
        lock.close()


def test_main_angles_mode_writes_separate_files(tmp_path: Path) -> None:
    out = tmp_path / "logs" / "weights"
    args = ["--config", str(_config(tmp_path)), "--output", str(out), "--device", "cpu", "--which", "all"]
    main([*args, "--angles"])
    close_all_jsonl_handles()
    run = out / "toy_sft"
    assert (run / "final.angles.summary.json").is_file() and (run / "step000005.angles.jsonl").is_file()
    assert not (run / "final.summary.json").exists()  # the metric pass is separate
    row = json.loads((run / "final.angles.jsonl").read_text().splitlines()[0])
    assert row["theta_mean"] > 0 and "sparsity" not in row
    summary = json.loads((run / "final.angles.summary.json").read_text())
    assert summary["theta_mean"] == pytest.approx(row["theta_mean"])


def test_ready_waits_for_index_of_sharded_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_file({"a": torch.zeros(2)}, str(ckpt / "model-00001-of-00002.safetensors"))
    assert ready({"path": str(ckpt)}) is None  # shards arrive before the index
    save_file({"b": torch.zeros(2)}, str(ckpt / "model-00002-of-00002.safetensors"))
    (ckpt / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    assert ready({"path": str(ckpt)}) == ckpt


def test_failing_task_is_recorded_and_skipped(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runs = json.loads(config.read_text())
    bad = tmp_path / "bad"
    bad.mkdir()
    save_file({"other": torch.zeros(2)}, str(bad / "model.safetensors"))  # tensor names differ
    runs["runs"][0]["intermediates"].insert(0, {"path": str(bad), "step": 1})
    config.write_text(json.dumps(runs))
    out = tmp_path / "logs" / "weights"
    undone = main(["--config", str(config), "--output", str(out), "--device", "cpu", "--which", "intermediate"])
    close_all_jsonl_handles()
    assert "tensor names differ" in (out / "toy_sft" / "step000001.failed").read_text()
    assert (out / "toy_sft" / "step000005.summary.json").is_file()  # the worker went on
    assert undone == [("toy_sft", "step000001")]


def test_main_spectral_mode_keeps_curves_only_for_final(tmp_path: Path) -> None:
    out = tmp_path / "logs" / "weights"
    main(["--config", str(_config(tmp_path)), "--output", str(out), "--device", "cpu", "--which", "all",
          "--basis-root", str(tmp_path / "basis"), "--spectral"])
    close_all_jsonl_handles()
    run = out / "toy_sft"
    final = json.loads((run / "final.spectral.jsonl").read_text().splitlines()[0])
    mid = json.loads((run / "step000005.spectral.jsonl").read_text().splitlines()[0])
    assert "pa_u" in final and "pa_u" not in mid
    assert final["rel_update_norm"] > mid["rel_update_norm"] > 0
    summary = json.loads((run / "final.spectral.summary.json").read_text())
    assert summary["nss"] == pytest.approx(final["nss"])


def test_main_rejects_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--config", str(_config(tmp_path)), "--output", str(tmp_path / "o"), "--runs", "nope"])
