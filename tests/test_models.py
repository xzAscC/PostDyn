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


def test_prune_revision_cache_removes_revision_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "models--org--model" / "snapshots" / "rev-123"
    snapshot.mkdir(parents=True)
    (snapshot / "weights.safetensors").write_text("cached")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))

    models.prune_revision_cache(_checkpoint())

    assert not snapshot.exists()
