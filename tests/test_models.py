from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import postdyn.models as models
from postdyn.config import CheckpointRef


def _checkpoint(name: str = "base", repo: str = "org/model") -> CheckpointRef:
    return CheckpointRef(name=name, repo=repo, revision="rev-123", stage="base")


def test_models_exports_load_model_with_contract_signature() -> None:
    import inspect

    assert list(inspect.signature(models.load_model).parameters) == [
        "checkpoint",
        "dtype",
        "quantization",
        "device",
    ]


def test_load_model_rejects_unknown_quantization_without_loading() -> None:
    with pytest.raises(ValueError, match="quantization"):
        models.load_model(_checkpoint(), quantization="int2", device="cpu")


def test_32b_float32_requires_nf4_or_h100_without_loading() -> None:
    checkpoint = _checkpoint(repo="allenai/Olmo-3-1125-32B")
    with pytest.raises(ValueError, match="(?i)NF4.*H100|H100.*NF4"):
        models.load_model(checkpoint, dtype="float32", device="cpu")


def test_config_preflight_rejects_wrong_architecture_before_weight_load(
    monkeypatch,
) -> None:
    config = SimpleNamespace(d_model=17, n_layers=19)
    monkeypatch.setattr(models, "pretrained_config", Mock(return_value=config))
    from_pretrained = Mock(return_value=SimpleNamespace(config=config))
    monkeypatch.setattr(models.AutoModelForCausalLM, "from_pretrained", from_pretrained)

    with pytest.raises(ValueError, match="d_model|n_layers"):
        models.load_model(_checkpoint(), device="cpu")
    from_pretrained.assert_not_called()


def test_hidden_dimension_and_layer_count_read_model_config() -> None:
    model = SimpleNamespace(config=SimpleNamespace(d_model=4096, n_layers=32))
    assert models.hidden_dimension(model) == 4096
    assert models.layer_count(model) == 32


def test_release_model_empties_cuda_cache_and_drops_reference(monkeypatch) -> None:
    empty_cache = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    model = SimpleNamespace()
    models.release_model(model)
    empty_cache.assert_called_once_with()
    _ = gc.collect()


def test_prune_revision_cache_removes_hash_snapshot_and_ref_but_keeps_blobs(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "models--org--model"
    snapshot = repo_dir / "snapshots" / ("a" * 40)
    ref = repo_dir / "refs" / "rev-123"
    blob = repo_dir / "blobs" / "shared-blob"
    snapshot.mkdir(parents=True)
    ref.parent.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_text("cached")
    (snapshot / "weights.safetensors").symlink_to(blob)
    ref.write_text("a" * 40 + "\n")

    models.prune_revision_cache(_checkpoint(), hub_cache=tmp_path)

    assert not snapshot.exists()
    assert not ref.exists()
    assert blob.exists()


def test_prune_revision_cache_missing_revision_is_noop(tmp_path: Path) -> None:
    models.prune_revision_cache(_checkpoint(), hub_cache=tmp_path)


def test_prune_revision_cache_undecodable_ref_is_noop(tmp_path: Path) -> None:
    ref = tmp_path / "models--org--model" / "refs" / "rev-123"
    ref.parent.mkdir(parents=True)
    ref.write_text("not a commit hash\nextra content\n")

    models.prune_revision_cache(_checkpoint(), hub_cache=tmp_path)

    assert ref.exists()


def test_start_prefetch_downloads_in_background(monkeypatch) -> None:
    from postdyn import models
    from postdyn.config import CheckpointRef

    calls = []

    def fake_snapshot_download(repo, revision=None, allow_patterns=None):
        calls.append((repo, revision, tuple(allow_patterns or ())))
        return f"/cache/{repo}@{revision}"

    monkeypatch.setattr(models, "snapshot_download", fake_snapshot_download)
    checkpoint = CheckpointRef("sft_step6000", "allenai/Olmo-3-7B-Think-SFT", "step6000", "sft")

    join = models.start_prefetch(checkpoint)
    assert join() is True
    assert calls == [("allenai/Olmo-3-7B-Think-SFT", "step6000", ("*.safetensors", "*.json"))]


def test_start_prefetch_join_reports_failure(monkeypatch) -> None:
    from postdyn import models
    from postdyn.config import CheckpointRef

    def fake_snapshot_download(repo, revision=None, allow_patterns=None):
        raise OSError("network down")

    monkeypatch.setattr(models, "snapshot_download", fake_snapshot_download)
    checkpoint = CheckpointRef("sft_step6000", "allenai/Olmo-3-7B-Think-SFT", "step6000", "sft")

    join = models.start_prefetch(checkpoint)
    assert join() is False
