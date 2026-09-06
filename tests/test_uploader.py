"""Tests for the background artifact uploader."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postdyn import uploader as up


class FakeApi:
    def __init__(self, failures: int = 0) -> None:
        self.calls: list[dict] = []
        self.failures = failures

    def upload_file(
        self,
        path_or_fileobj=None,
        path_in_repo=None,
        repo_id=None,
        repo_type=None,
        **kwargs,
    ):
        self.calls.append(
            {"path": str(path_or_fileobj), "in_repo": path_in_repo, "repo": repo_id}
        )
        if self.failures > 0:
            self.failures -= 1
            raise OSError("flaky network")


def _write(tmp_path: Path, rel: str, content: str = "x") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def test_uploader_streams_files_and_mirrors_layout(tmp_path: Path) -> None:
    api = FakeApi()
    artifact = _write(tmp_path, "logs/q1/7b/eigensystems/base/3/math.safetensors")
    state = tmp_path / ".upload_state.json"

    handle = up.ArtifactUploader("me/postdyn-artifacts", state_path=state, api=api)
    handle.start()
    handle.submit(artifact, relative_to=tmp_path)
    summary = handle.finish()

    assert summary["uploaded"] == 1 and summary["failed"] == 0
    assert api.calls[0]["in_repo"] == "logs/q1/7b/eigensystems/base/3/math.safetensors"
    assert api.calls[0]["repo"] == "me/postdyn-artifacts"
    recorded = json.loads(state.read_text())
    assert recorded["uploaded"]["logs/q1/7b/eigensystems/base/3/math.safetensors"]


def test_uploader_skips_already_uploaded_via_state(tmp_path: Path) -> None:
    api = FakeApi()
    artifact = _write(tmp_path, "data/domain_prompts/math.json")
    state = tmp_path / ".upload_state.json"
    state.write_text(
        json.dumps({"uploaded": {"data/domain_prompts/math.json": "done"}})
    )

    handle = up.ArtifactUploader("me/r", state_path=state, api=api)
    handle.start()
    handle.submit(artifact, relative_to=tmp_path)
    summary = handle.finish()

    assert summary["uploaded"] == 0 and summary["skipped"] == 1
    assert api.calls == []


def test_uploader_retries_then_records_failure_without_raising(
    tmp_path: Path,
) -> None:
    api = FakeApi(failures=4)
    artifact = _write(tmp_path, "logs/run.log")

    handle = up.ArtifactUploader(
        "me/r",
        state_path=tmp_path / ".upload_state.json",
        api=api,
        retries=3,
        retry_delay=0,
    )
    handle.start()
    handle.submit(artifact, relative_to=tmp_path)
    summary = handle.finish()

    assert summary["failed"] == 1 and summary["uploaded"] == 0
    recorded = json.loads((tmp_path / ".upload_state.json").read_text())
    assert "logs/run.log" in recorded["failed"]


def test_uploader_submits_trees_recursively(tmp_path: Path) -> None:
    api = FakeApi()
    _write(tmp_path, "logs/q1/manifest.json")
    _write(tmp_path, "logs/q1/eigensystems/base/3/math.json")

    handle = up.ArtifactUploader("me/r", state_path=tmp_path / ".s.json", api=api)
    handle.start()
    handle.submit_tree(tmp_path / "logs", relative_to=tmp_path)
    handle.finish()

    uploaded = sorted(call["in_repo"] for call in api.calls)
    assert uploaded == [
        "logs/q1/eigensystems/base/3/math.json",
        "logs/q1/manifest.json",
    ]


def test_uploader_deduplicates_queued_path(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    api = FakeApi()

    def blocking_upload(**kwargs):
        api.calls.append(
            {"path": kwargs["path_or_fileobj"], "in_repo": kwargs["path_in_repo"]}
        )
        entered.set()
        release.wait(timeout=5)

    setattr(api, "upload_file", blocking_upload)
    artifact = _write(tmp_path, "logs/run.log")
    handle = up.ArtifactUploader("me/r", state_path=tmp_path / ".s.json", api=api)
    handle.start()
    handle.submit(artifact, relative_to=tmp_path)
    assert entered.wait(timeout=5)
    handle.submit(artifact, relative_to=tmp_path)
    release.set()
    summary = handle.finish()

    assert len(api.calls) == 1
    assert summary["uploaded"] == 1
    assert summary["skipped"] == 1


def test_uploader_terminal_failure_clears_pending_for_resubmit(tmp_path: Path) -> None:
    api = FakeApi(failures=100)
    artifact = _write(tmp_path, "logs/run.log")
    handle = up.ArtifactUploader(
        "me/r", state_path=tmp_path / ".s.json", api=api, retries=0, retry_delay=0
    )
    handle.start()
    handle.submit(artifact, relative_to=tmp_path)
    first = handle.finish()

    assert first["failed"] == 1
    assert handle._pending == set()
    handle.submit(artifact, relative_to=tmp_path)
    assert handle._summary.skipped == 0
