"""Tests for BF16 XOR delta chains of training checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import postdyn.delta_chain as dc

STEPS = [1, 2, 3, 5]  # step 4 exists but is unfinished (no DONE marker)


def _save(folder: Path, tensors: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    folder.mkdir(parents=True)
    save_file({k: v.to(dtype).contiguous() for k, v in tensors.items()}, folder / "model.safetensors")
    (folder / "config.json").write_text(json.dumps({"dtype": str(dtype).removeprefix("torch.")}))
    (folder / "tokenizer.json").write_text("{}")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).view(torch.int16)


@pytest.fixture
def run(tmp_path: Path) -> dict:
    """A base model in BF16 and FP32 exports of a run that starts from it."""
    g = torch.Generator().manual_seed(0)
    base = {
        "embed.weight": torch.randn(100, 48, generator=g) * 0.02,
        "layer.weight": -torch.rand(64, 48, generator=g) * 0.05,  # negative values too
        "norm.weight": torch.ones(48),
    }
    base = {k: v.to(torch.bfloat16) for k, v in base.items()}
    _save(tmp_path / "base", base, torch.bfloat16)
    master = {k: v.float() for k, v in base.items()}
    expected = {}
    exports = tmp_path / "exports"
    for step in range(1, 6):
        for k in master:
            master[k] += torch.randn(master[k].shape, generator=g) * 3e-4
        folder = exports / f"step_{step}"
        _save(folder / "huggingface", master, torch.float32)
        if step in STEPS:
            (folder / "DONE").touch()
            expected[step] = {k: _bits(v) for k, v in master.items()}
    return {"base": tmp_path / "base", "exports": exports, "expected": expected, "out": tmp_path / "chain"}


def _build(run: dict, **kw) -> None:
    dc.build(dc.find_exports([run["exports"]]), run["base"], "org/base", "abc123", run["out"], **kw)


def _equal(bits: dict, expected: dict) -> bool:
    return bits.keys() == expected.keys() and all(
        torch.equal(torch.from_numpy(bits[k].view("int16")), expected[k]) for k in expected
    )


def test_find_exports_skips_unfinished_and_takes_the_first_dir_per_step(tmp_path: Path) -> None:
    for root, steps in (("a", [1, 2, 3]), ("b", [2, 5])):
        for s in steps:
            (tmp_path / root / f"step_{s}" / "huggingface").mkdir(parents=True)
            (tmp_path / root / f"step_{s}" / "DONE").touch()
    (tmp_path / "a" / "step_4" / "huggingface").mkdir(parents=True)  # unfinished
    found = dc.find_exports([tmp_path / "a", tmp_path / "b"])
    assert [s for s, _ in found] == [1, 2, 3, 5]
    assert dict(found)[2] == tmp_path / "a" / "step_2" / "huggingface"


def test_rebuild_is_bit_exact_for_every_step(run: dict) -> None:
    _build(run, part_bytes=4000)  # small parts: every step is split into several files
    manifest = json.loads((run["out"] / "manifest.json").read_text())
    assert [e["step"] for e in manifest["steps"]] == STEPS
    assert all(len(e["parts"]) > 1 for e in manifest["steps"])
    seen = []
    for step, bits in dc.rebuild(run["out"], base_dir=run["base"], verify=True):
        assert _equal(bits, run["expected"][step]), step
        seen.append(step)
    assert seen == STEPS


def test_final_is_a_loadable_bf16_checkpoint(run: dict) -> None:
    _build(run)
    final = run["out"] / "final"
    tensors = load_file(final / "model.safetensors")
    assert all(t.dtype == torch.bfloat16 for t in tensors.values())
    assert {k: t.view(torch.int16) for k, t in tensors.items()}.keys() == run["expected"][5].keys()
    assert all(torch.equal(tensors[k].view(torch.int16), v) for k, v in run["expected"][5].items())
    assert json.loads((final / "config.json").read_text())["dtype"] == "bfloat16"
    assert (final / "tokenizer.json").exists()
    assert (run["out"] / "delta_chain.py").read_text() == Path(dc.__file__).read_text()


def test_corrupt_delta_is_detected(run: dict) -> None:
    _build(run)
    part = sorted((run["out"] / "deltas").iterdir())[0]
    data = bytearray(part.read_bytes())
    data[-1] ^= 1
    part.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="corrupt delta"):
        list(dc.rebuild(run["out"], base_dir=run["base"]))


def test_a_different_base_is_detected(run: dict) -> None:
    _build(run)
    tensors = load_file(run["base"] / "model.safetensors")
    tensors["norm.weight"] = tensors["norm.weight"] * 2
    save_file(tensors, run["base"] / "model.safetensors")
    with pytest.raises(ValueError, match="base"):
        list(dc.rebuild(run["out"], base_dir=run["base"]))


def test_build_resumes_after_the_last_finished_step(run: dict, tmp_path: Path) -> None:
    _build(run, max_steps=2)
    assert [e["step"] for e in json.loads((run["out"] / "manifest.json").read_text())["steps"]] == [1, 2]
    _build(run)
    resumed = json.loads((run["out"] / "manifest.json").read_text())
    dc.build(dc.find_exports([run["exports"]]), run["base"], "org/base", "abc123", tmp_path / "fresh")
    fresh = json.loads((tmp_path / "fresh" / "manifest.json").read_text())
    assert resumed == fresh


def test_restore_writes_one_step(run: dict, tmp_path: Path) -> None:
    _build(run)
    out = tmp_path / "step2"
    dc.main(["restore", str(run["out"]), "--base-dir", str(run["base"]), "--step", "2", "--out", str(out)])
    tensors = load_file(out / "model.safetensors")
    assert all(torch.equal(tensors[k].view(torch.int16), v) for k, v in run["expected"][2].items())
    assert json.loads((out / "config.json").read_text())["dtype"] == "bfloat16"


def test_verify_compares_with_the_exports(run: dict) -> None:
    _build(run)
    args = ["restore", str(run["out"]), "--base-dir", str(run["base"]), "--verify",
            "--export-dir", str(run["exports"]), "--compare-steps", "1", "5"]
    dc.main(args)
    # an export that no longer matches the chain must fail the comparison
    tensors = load_file(run["exports"] / "step_5" / "huggingface" / "model.safetensors")
    tensors["norm.weight"] = tensors["norm.weight"] + 1
    save_file(tensors, run["exports"] / "step_5" / "huggingface" / "model.safetensors")
    with pytest.raises(ValueError, match="step 5"):
        dc.main(args)


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def test_upload_creates_a_private_repo_by_default(run: dict) -> None:
    _build(run)
    api = FakeApi()
    dc.upload(run["out"], "user/run-bf16delta", api=api)
    (c1, kw1), (c2, kw2) = api.calls
    assert (c1, kw1["repo_id"], kw1["private"]) == ("create_repo", "user/run-bf16delta", True)
    assert (c2, kw2["repo_id"], kw2["folder_path"]) == ("upload_folder", "user/run-bf16delta", str(run["out"]))


def test_rebuild_downloads_the_pinned_base_revision(run: dict, monkeypatch) -> None:
    _build(run)
    calls = []

    def fake_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(run["base"])

    monkeypatch.setattr(dc, "snapshot_download", fake_download)
    steps = [s for s, _ in dc.rebuild(run["out"], verify=True)]
    assert steps == STEPS
    assert calls == [("org/base", "abc123", ["model.safetensors"])]
