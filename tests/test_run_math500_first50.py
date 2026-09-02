from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import scripts.run_math500_first50 as cli
from postdyn.math500_eval import MATH500_COUNT


class FakeModel:
    def eval(self) -> object:
        return self

    def get_input_embeddings(self) -> object:
        return SimpleNamespace(weight=SimpleNamespace(device="cpu"))


def test_cli_selects_model_key_and_revision_without_downloading(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, Any] = {}

    def fake_loader(model_key: str, revision: str, dtype: str, quantization: str):
        calls["loader"] = (model_key, revision, dtype, quantization)
        return FakeModel(), object()

    def fake_generator(**kwargs: object) -> object:
        calls["generator"] = kwargs
        return object()

    def fake_eval(**kwargs: object):
        calls["evaluate"] = kwargs
        return SimpleNamespace(
            n_processed=MATH500_COUNT, n_expected=MATH500_COUNT, accuracy=1.0
        )

    monkeypatch.setattr(cli, "load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(cli, "GreedyGenerator", fake_generator)
    monkeypatch.setattr(cli, "evaluate_first50", fake_eval)
    args = cli.parse_args(
        [
            "--model-key",
            "olmo3-think-dpo",
            "--revision",
            "step_123",
            "--dtype",
            "float32",
            "--quantization",
            "none",
            "--output",
            str(tmp_path),
        ]
    )
    assert cli.run(args) == 0
    assert calls["loader"] == ("olmo3-think-dpo", "step_123", "float32", "none")
    assert calls["evaluate"]["revision"] == "step_123"
    assert calls["evaluate"]["model_key"] == "olmo3-think-dpo"
