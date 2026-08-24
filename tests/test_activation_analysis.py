import sys
import types

import pytest
import torch

from src.activation_analysis import (
    _get_tokenizer,
    _load_model,
    compute_activation_alpha_req,
    compute_activation_rankme,
)
from src.config import OLMO3_BASE_CONFIG, OLMO3_VARIANTS


@pytest.mark.parametrize("loader", [_get_tokenizer, _load_model])
@pytest.mark.parametrize(
    "model_config",
    [
        OLMO3_VARIANTS["olmo3-32b-think-sft"],
        OLMO3_VARIANTS["olmo3-32b-think-rlvr"],
    ],
)
def test_generic_32b_loaders_reject_before_transformers_import(
    monkeypatch, loader, model_config
):
    def fail_on_transformers_import(name, *args, **kwargs):
        if name == "transformers":
            raise AssertionError("generic 32B loading imported Transformers")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_on_transformers_import)

    with pytest.raises(ValueError, match="canonical NF4 experiment loader"):
        loader(model_config)


def test_generic_7b_loaders_keep_using_transformers(monkeypatch):
    tokenizer = types.SimpleNamespace(pad_token=None, eos_token="eos")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return tokenizer

    model = types.SimpleNamespace(eval=lambda: None)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model

    fake_transformers = types.ModuleType("transformers")
    setattr(fake_transformers, "AutoTokenizer", FakeAutoTokenizer)
    setattr(fake_transformers, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    assert _get_tokenizer(OLMO3_BASE_CONFIG) is tokenizer
    assert tokenizer.pad_token == tokenizer.eos_token
    assert _load_model(OLMO3_BASE_CONFIG) is model


def test_activation_rankme_uses_covariance_eigenvalues():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )

    rankme, ratio = compute_activation_rankme(features)

    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    expected_rankme = torch.exp(-torch.sum(p * torch.log(p))).item()

    assert abs(rankme - expected_rankme) < 1e-6
    assert abs(ratio - expected_rankme / features.shape[1]) < 1e-6


def test_activation_rankme_differs_from_raw_singular_value_entropy():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )

    rankme, _ = compute_activation_rankme(features)

    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    p_raw = singular_values / singular_values.sum()
    raw_svd_rankme = torch.exp(-torch.sum(p_raw * torch.log(p_raw))).item()

    assert abs(rankme - raw_svd_rankme) > 0.05


def test_activation_alpha_req_uses_covariance_eigenvalue_decay():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 / 2.0, 0.0, 0.0],
            [0.0, -1.0 / 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, -1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / 4.0],
            [0.0, 0.0, 0.0, -1.0 / 4.0],
        ]
    )

    alpha = compute_activation_alpha_req(features, fit_range=(1, 4))

    assert abs(alpha - 2.0) < 1e-5
