"""Tests for the upload_artifacts CLI and the prune rule."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postdyn import uploader as up


def test_determine_prunable_keeps_finals_and_requires_analysis(
    tmp_path: Path,
) -> None:
    intermediate = tmp_path / "logs/q1/7b/eigensystems/sft_step6000/3/math.json"
    intermediate.parent.mkdir(parents=True)
    intermediate.write_text("{}")
    final = tmp_path / "logs/q1/7b/eigensystems/rlvr/3/math.json"
    final.parent.mkdir(parents=True)
    final.write_text("{}")
    analysis = tmp_path / "logs/q1/7b/analysis/summary.json"
    analysis.parent.mkdir(parents=True)
    analysis.write_text("{}")

    state = {
        "uploaded": {
            "logs/q1/7b/eigensystems/sft_step6000/3/math.json": "t",
            "logs/q1/7b/eigensystems/rlvr/3/math.json": "t",
            "data/domain_prompts/math.json": "t",
        }
    }
    prunable = up.determine_prunable(state, tmp_path)
    assert prunable == [intermediate]

    analysis.unlink()
    assert up.determine_prunable(state, tmp_path) == []


def test_upload_cli_uploads_trees_and_reports(monkeypatch, tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.upload_artifacts")
    calls: list[str] = []

    class FakeApi:
        def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=None):
            calls.append(f"create:{repo_id}")

        def upload_file(self, path_or_fileobj=None, path_in_repo=None, **kwargs):
            calls.append(f"upload:{path_in_repo}")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/domain_prompts").mkdir(parents=True)
    (tmp_path / "data/domain_prompts/math.json").write_text("{}")
    monkeypatch.setattr(up, "HfApi", FakeApi, raising=False)
    original = up.ArtifactUploader

    def patched_factory(repo, state_path, api=None, **kwargs):
        return original(repo, state_path=state_path, api=FakeApi(), **kwargs)

    monkeypatch.setattr(cli.uploader, "ArtifactUploader", patched_factory)
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    status = cli.main(["--repo", "me/r", "--paths", "data"])
    assert status == 0
    assert "create:me/r" in calls
    assert "upload:data/domain_prompts/math.json" in calls
