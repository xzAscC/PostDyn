"""Extraction and streaming-covariance contract tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

import pytest
import torch
from postdyn.extract import (
    OnlineCovariance,
    extract_layer_hiddens,
    iterate_prompt_batches,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
)


class _FallbackTokenizer:
    """Small tokenizer used only when the offline Hugging Face cache is empty."""

    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _encode(self, text: str, max_length: int | None) -> list[int]:
        ids = [(ord(character) % 510) + 2 for character in text] or [1]
        return ids[:max_length] if max_length is not None else ids

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        return_tensors: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        del truncation
        text_list = [texts] if isinstance(texts, str) else list(texts)
        encoded = [self._encode(text, max_length) for text in text_list]
        target_length = max(len(ids) for ids in encoded)
        if padding == "max_length" and max_length is not None:
            target_length = max_length
        if padding:
            padded: list[list[int]] = []
            masks: list[list[int]] = []
            for ids in encoded:
                pad_count = target_length - len(ids)
                if self.padding_side == "left":
                    padded.append([self.pad_token_id] * pad_count + ids)
                    masks.append([0] * pad_count + [1] * len(ids))
                else:
                    padded.append(ids + [self.pad_token_id] * pad_count)
                    masks.append([1] * len(ids) + [0] * pad_count)
            encoded = padded
        else:
            masks = [[1] * len(ids) for ids in encoded]
        if return_tensors != "pt":
            raise ValueError(
                "The fallback tokenizer only supports return_tensors='pt'."
            )
        return {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def apply_chat_template(self, *_: object, **__: object) -> None:
        raise AssertionError(
            "chat templates are not available in the fallback tokenizer"
        )


class _Tokenizer(Protocol):
    padding_side: str
    pad_token: str | None
    pad_token_id: int | None
    eos_token: str

    def __call__(
        self, texts: str | Sequence[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]: ...


class _TrackingTokenizer:
    def __init__(self, tokenizer: _Tokenizer) -> None:
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "calls", [])

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_tokenizer"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_tokenizer", "calls", "apply_chat_template"}:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_tokenizer"), name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = object.__getattribute__(
            self, "calls"
        )
        calls.append((args, kwargs))
        tokenizer: _Tokenizer = object.__getattribute__(self, "_tokenizer")
        return tokenizer(*args, **kwargs)


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer() -> tuple[Any, _Tokenizer]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        try:
            tokenizer = cast(
                _Tokenizer,
                AutoTokenizer.from_pretrained(
                    "sshleifer/tiny-gpt2", local_files_only=True
                ),
            )
        except (OSError, ValueError):
            tokenizer = cast(_Tokenizer, cast(object, _FallbackTokenizer()))

        model: Any
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "sshleifer/tiny-gpt2", local_files_only=True
            )
        except (OSError, ValueError):
            config = GPT2Config(
                vocab_size=512,
                n_positions=64,
                n_embd=16,
                n_layer=2,
                n_head=2,
                bos_token_id=0,
                eos_token_id=1,
                pad_token_id=0,
            )
            model = GPT2LMHeadModel(config)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to("cpu").eval()

    # GPT-2's absolute position table otherwise shifts real tokens after left padding.
    position_embedding = getattr(getattr(model, "transformer", None), "wpe", None)
    if isinstance(position_embedding, torch.nn.Embedding):
        with torch.no_grad():
            position_embedding.weight.zero_()
    return model, tokenizer


def _forward_hidden_states(
    model: Any,
    tokenizer: _Tokenizer,
    prompts: Sequence[str],
    padding_side: str,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True)
    finally:
        tokenizer.padding_side = previous_side
    return encoded, outputs.hidden_states


def test_online_covariance_matches_centered_covariance_for_uneven_batches() -> None:
    torch.manual_seed(10)
    n, d = 200, 10
    observations = torch.randn(n, d, dtype=torch.float64)
    expected = observations - observations.mean(dim=0, keepdim=True)
    expected = expected.T @ expected / n

    accumulator = OnlineCovariance()
    offset = 0
    for batch_size in [7, 31, 4, 53, 105]:
        accumulator.update(observations[offset : offset + batch_size])
        offset += batch_size

    assert offset == n
    assert accumulator.count == n
    torch.testing.assert_close(
        accumulator.mean,
        observations.mean(dim=0),
        atol=1e-8,
        rtol=1e-8,
    )
    torch.testing.assert_close(accumulator.covariance, expected, atol=1e-8, rtol=1e-8)


def test_online_covariance_single_batch_matches_direct_covariance() -> None:
    torch.manual_seed(11)
    observations = torch.randn(200, 10, dtype=torch.float64)
    centered = observations - observations.mean(dim=0, keepdim=True)
    expected = centered.T @ centered / observations.shape[0]

    accumulator = OnlineCovariance()
    accumulator.update(observations)

    assert accumulator.count == observations.shape[0]
    torch.testing.assert_close(accumulator.covariance, expected, atol=1e-8, rtol=1e-8)


def test_extract_layer_hiddens_uses_raw_text_left_padding_and_final_tokens(
    tiny_model_and_tokenizer: tuple[Any, _Tokenizer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, tokenizer = tiny_model_and_tokenizer
    prompts = [
        "hi",
        "a short prompt",
        "a somewhat longer prompt here",
        "please compare these hidden states carefully",
        "this is the longest prompt in the tiny extraction fixture",
    ]
    tracking_tokenizer = _TrackingTokenizer(tokenizer)

    def fail_chat_template(*_: object, **__: object) -> None:
        raise AssertionError("chat_template=False must not apply a chat template")

    monkeypatch.setattr(tracking_tokenizer, "apply_chat_template", fail_chat_template)
    extracted = extract_layer_hiddens(
        model,
        tracking_tokenizer,
        prompts,
        layers=[0, 1],
        batch_size=5,
        max_length=64,
        chat_template=False,
    )

    observed_prompts: list[str] = []
    for args, kwargs in tracking_tokenizer.calls:
        batch = args[0] if args else kwargs["text"]
        observed_prompts.extend([batch] if isinstance(batch, str) else batch)
    assert observed_prompts == prompts
    assert set(extracted) == {0, 1}
    for hidden in extracted.values():
        assert hidden.shape[0] == len(prompts)
        assert hidden.shape[1] == model.config.hidden_size
        assert hidden.dtype == torch.float32
        assert hidden.device.type == "cpu"

    left_inputs, left_hidden = _forward_hidden_states(
        model, tokenizer, prompts, padding_side="left"
    )
    short_last = int(left_inputs["attention_mask"][0].nonzero()[-1].item())
    assert short_last == left_inputs["input_ids"].shape[1] - 1
    torch.testing.assert_close(
        extracted[1][0], left_hidden[1][0, short_last].cpu(), atol=1e-4, rtol=1e-4
    )

    right_inputs, right_hidden = _forward_hidden_states(
        model, tokenizer, prompts, padding_side="right"
    )
    assert right_inputs["attention_mask"][0, -1].item() == 0
    assert not torch.allclose(
        extracted[1][0], right_hidden[1][0, -1].cpu(), atol=1e-4, rtol=1e-4
    )


def test_extract_final_token_matches_single_unpadded_forward(
    tiny_model_and_tokenizer: tuple[Any, _Tokenizer],
) -> None:
    model, tokenizer = tiny_model_and_tokenizer
    prompt = "the short prompt"
    prompts = [prompt, "this second prompt is substantially longer than the first"]
    extracted = extract_layer_hiddens(
        model,
        tokenizer,
        prompts,
        layers=[0, 1],
        batch_size=2,
        max_length=64,
        chat_template=False,
    )

    tokenizer.padding_side = "left"
    single_inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    assert single_inputs["attention_mask"].all()
    with torch.no_grad():
        single_hidden = model(**single_inputs, output_hidden_states=True).hidden_states

    for layer in [0, 1]:
        torch.testing.assert_close(
            extracted[layer][0], single_hidden[layer][0, -1].cpu(), atol=1e-4, rtol=1e-4
        )


def test_iterate_prompt_batches_preserves_order_and_batch_sizes() -> None:
    prompts = [f"prompt-{index}" for index in range(5)]

    batches: list[list[str]] = list(iterate_prompt_batches(prompts, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [prompt for batch in batches for prompt in batch] == prompts
    assert all(isinstance(batch, list) for batch in batches)
