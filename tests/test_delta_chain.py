"""Tests for BF16 XOR delta chains of training checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

import postdyn.delta_chain as dc

STEPS = [1, 2, 3, 5]  # step 4 exists but is unfinished (no DONE marker)
REV = "0123456789abcdef0123456789abcdef01234567"


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
    dc.build(dc.find_exports([run["exports"]]), run["base"], "org/base", REV, run["out"], **kw)


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
    dc.build(dc.find_exports([run["exports"]]), run["base"], "org/base", REV, tmp_path / "fresh")
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
    def __init__(self, base_sha: str, public: bool = False, revision: str = REV) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.base_sha, self.public, self.revision = base_sha, public, revision

    def model_info(self, repo_id, revision=None, files_metadata=False):
        lfs = SimpleNamespace(sha256=self.base_sha)
        return SimpleNamespace(sha=self.revision, siblings=[SimpleNamespace(rfilename="model.safetensors", lfs=lfs)])

    def repo_info(self, repo_id):
        return SimpleNamespace(private=not self.public)

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw))


def _base_sha(run: dict) -> str:
    return dc.file_sha(run["base"] / "model.safetensors")


def test_upload_creates_a_private_repo_with_only_the_chain_files(run: dict) -> None:
    _build(run)
    (run["out"] / "deltas" / "leftover.safetensors").write_bytes(b"x")  # e.g. from a killed build
    api = FakeApi(_base_sha(run))
    dc.upload(run["out"], "user/run-bf16delta", api=api)
    (c1, kw1), (c2, kw2) = api.calls
    assert (c1, kw1["repo_id"], kw1["private"]) == ("create_repo", "user/run-bf16delta", True)
    assert (c2, kw2["repo_id"], kw2["folder_path"]) == ("upload_folder", "user/run-bf16delta", str(run["out"]))
    assert "deltas/leftover.safetensors" not in kw2["allow_patterns"]
    assert {"manifest.json", "delta_chain.py", "final/*"} <= set(kw2["allow_patterns"])


def test_upload_refuses_an_existing_public_repo(run: dict) -> None:
    _build(run)
    api = FakeApi(_base_sha(run), public=True)
    with pytest.raises(ValueError, match="public"):
        dc.upload(run["out"], "user/run-bf16delta", api=api)
    assert [c for c, _ in api.calls] == ["create_repo"]


def test_upload_refuses_a_base_that_differs_from_the_hub(run: dict) -> None:
    _build(run)
    for api in (FakeApi("0" * 64), FakeApi(_base_sha(run), revision="f" * 40)):
        with pytest.raises(ValueError, match="Hub"):
            dc.upload(run["out"], "user/run-bf16delta", api=api)
        assert api.calls == []


def test_upload_refuses_an_unfinished_chain(run: dict, monkeypatch) -> None:
    _build(run, max_steps=2)

    def killed(*args, **kw):
        raise RuntimeError("job killed while writing final/")

    monkeypatch.setattr(dc, "write_checkpoint", killed)
    with pytest.raises(RuntimeError):
        _build(run)
    with pytest.raises(ValueError, match="final"):
        dc.upload(run["out"], "user/run-bf16delta", api=FakeApi(_base_sha(run)))
    monkeypatch.undo()
    _build(run)  # a rerun finishes the chain
    assert json.loads((run["out"] / "manifest.json").read_text())["final"]["step"] == 5


def test_build_rejects_a_branch_name_as_base_revision(run: dict) -> None:
    with pytest.raises(ValueError, match="commit"):
        dc.build(dc.find_exports([run["exports"]]), run["base"], "org/base", "main", run["out"])


def test_build_resumes_after_the_exports_of_built_steps_are_deleted(run: dict) -> None:
    import shutil

    _build(run, max_steps=2)
    for s in (1, 2):
        shutil.rmtree(run["exports"] / f"step_{s}")
    _build(run)
    steps = [e["step"] for e in json.loads((run["out"] / "manifest.json").read_text())["steps"]]
    assert steps == STEPS
    assert all(_equal(bits, run["expected"][s]) for s, bits in dc.rebuild(run["out"], base_dir=run["base"]))


def test_rerun_with_fewer_steps_leaves_the_chain_alone(run: dict) -> None:
    _build(run)
    before = (run["out"] / "manifest.json").read_text()
    final = (run["out"] / "final" / "model.safetensors").stat().st_mtime_ns
    _build(run, max_steps=2)
    assert (run["out"] / "manifest.json").read_text() == before
    assert (run["out"] / "final" / "model.safetensors").stat().st_mtime_ns == final


def test_build_rejects_a_tensor_whose_shape_changed(run: dict) -> None:
    folder = run["exports"] / "step_2" / "huggingface"
    tensors = load_file(folder / "model.safetensors")
    tensors["norm.weight"] = tensors["norm.weight"][None, :].repeat(3, 1)  # (48,) -> (3, 48)
    save_file(tensors, folder / "model.safetensors")
    with pytest.raises(ValueError, match="shape"):
        _build(run)


def test_restore_checks_the_last_step_without_verify(run: dict, tmp_path: Path) -> None:
    _build(run)
    mpath = run["out"] / "manifest.json"
    m = json.loads(mpath.read_text())
    m["steps"][-1]["bf16_sha256"] = "0" * 64
    mpath.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="step 5"):
        dc.main(["restore", str(run["out"]), "--base-dir", str(run["base"]), "--out", str(tmp_path / "o")])


def test_compare_steps_that_are_not_rebuilt_fail(run: dict) -> None:
    _build(run, max_steps=3)
    base = ["restore", str(run["out"]), "--base-dir", str(run["base"]), "--export-dir", str(run["exports"])]
    with pytest.raises(ValueError, match="not compared"):
        dc.main([*base, "--step", "2", "--compare-steps", "3"])  # after the stop
    with pytest.raises(ValueError, match="not compared"):
        dc.main([*base, "--compare-steps", "5"])  # has an export, but is not in the chain


def test_rebuild_downloads_the_pinned_base_revision(run: dict, monkeypatch) -> None:
    _build(run)
    calls = []

    def fake_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(run["base"])

    monkeypatch.setattr(dc, "snapshot_download", fake_download)
    steps = [s for s, _ in dc.rebuild(run["out"], verify=True)]
    assert steps == STEPS
    assert calls == [("org/base", REV, ["model.safetensors"])]
