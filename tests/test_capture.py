from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM
import importlib.util
from pathlib import Path
from typing import Any

from postdyn.capture import hidden_capture


def _load_run_q1():
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_q1_capture_test", root / "scripts" / "run_q1.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_gpt2():
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2", local_files_only=True
        )
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"tiny-gpt2 is not cached: {exc}")
    return model.eval()


def test_hidden_capture_matches_hidden_states() -> None:
    model = _tiny_gpt2()
    inputs = {"input_ids": torch.tensor([[10, 11, 12]])}
    with hidden_capture(model, [0, 1]) as store:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
    assert torch.equal(store.tensors[0], outputs.hidden_states[1])
    assert torch.equal(store.tensors[1], outputs.hidden_states[2])


def test_run_q1_tiny_model_capture_matches_hidden_states() -> None:
    run_q1 = _load_run_q1()
    model = run_q1._TinyModel(seed=17).eval()
    inputs = {"input_ids": torch.tensor([[10, 11, 12]])}
    with hidden_capture(model, [0, 1]) as store:
        with torch.no_grad():
            model(**inputs)
    with torch.no_grad():
        reference_output = model(**inputs, output_hidden_states=True)
    assert torch.equal(store.tensors[0], reference_output.hidden_states[1])
    assert torch.equal(store.tensors[1], reference_output.hidden_states[2])


def test_hidden_capture_rejects_invalid_layers() -> None:
    model = _tiny_gpt2()
    with pytest.raises(ValueError, match="outside model layer range"):
        hidden_capture(model, [-1, 2])


def test_hidden_capture_removes_hooks_on_exit() -> None:
    model: Any = _tiny_gpt2()
    blocks = model.transformer.h
    final_norm = model.transformer.ln_f
    with hidden_capture(model, [0, 1]):
        assert blocks[0]._forward_hooks
        assert final_norm._forward_hooks
    assert not blocks[0]._forward_hooks
    assert not final_norm._forward_hooks


def test_hidden_capture_removes_hooks_when_forward_raises(monkeypatch) -> None:
    model: Any = _tiny_gpt2()
    block = model.transformer.h[0]

    def fail(*args, **kwargs):
        raise RuntimeError("forced forward failure")

    monkeypatch.setattr(block, "forward", fail)
    with pytest.raises(RuntimeError, match="forced forward failure"):
        with hidden_capture(model, [0, 1]):
            model(input_ids=torch.tensor([[10, 11]]))
    assert not block._forward_hooks
    assert not model.transformer.ln_f._forward_hooks
