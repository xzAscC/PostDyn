"""Benchmark loading and deterministic generation helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - exercised only without optional dependency
    load_dataset = None


@dataclass(frozen=True)
class BenchItem:
    id: str
    prompt: str
    reference: dict[str, Any]


@dataclass(frozen=True)
class Generation:
    item_id: str
    text: str
    captured: dict[int, torch.Tensor] | None = None
    prompt_token_len: int = 0


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    """Apply the user-turn template while preserving a generation prompt."""
    if not getattr(tokenizer, "chat_template", None) or not hasattr(
        tokenizer, "apply_chat_template"
    ):
        raise ValueError("tokenizer does not provide a chat template")
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def _rows(dataset: Any) -> list[Any]:
    if isinstance(dataset, dict):
        dataset = dataset.get("train") or next(iter(dataset.values()))
    return list(dataset)


def _dataset(path: str, name: str | None = None) -> list[Any]:
    if load_dataset is None:
        raise ImportError("datasets is required to load benchmarks")
    return _rows(load_dataset(path, name=name) if name else load_dataset(path))


def _load_math500() -> list[BenchItem]:
    return [
        BenchItem(
            str(r.get("id", i)),
            r.get("problem", r.get("prompt", "")),
            {"answer": r.get("answer", "")},
        )
        for i, r in enumerate(_dataset("HuggingFaceH4/MATH-500"))
    ]


def _load_mmlu_pro() -> list[BenchItem]:
    result = []
    for i, r in enumerate(_dataset("TIGER-Lab/MMLU-Pro")):
        options = r.get("options", list("ABCDEFGHIJ"))
        result.append(
            BenchItem(
                str(r.get("id", i)),
                r.get("question", r.get("prompt", "")),
                {"answer": str(r.get("answer", "A")), "options": options},
            )
        )
    return result


def _load_ifeval() -> list[BenchItem]:
    return [
        BenchItem(
            str(r.get("id", i)),
            r.get("prompt", ""),
            {
                "instructions": r.get("instr", r.get("instructions", [])),
                "kwargs": r.get("kwargs", []),
            },
        )
        for i, r in enumerate(_dataset("google/IFEval"))
    ]


def _load_livecodebench() -> list[BenchItem]:
    return [
        BenchItem(
            str(r.get("id", i)),
            r.get("prompt", r.get("question", "")),
            {
                "code": r.get("code", r.get("canonical_solution", "")),
                "cases": r.get("test-cases", r.get("cases", [])),
            },
        )
        for i, r in enumerate(_dataset("livecodebench/code_generation_lite"))
    ]


def _ensure_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def load_benchmark(
    name: str, n_val: int = 30, seed: int = 42
) -> tuple[list[BenchItem], list[BenchItem]]:
    """Load, hash-sort, and split a benchmark; ``seed`` is part of the ordering key."""
    loaders = {
        "math500": _load_math500,
        "mmlu_pro": _load_mmlu_pro,
        "ifeval": _load_ifeval,
        "livecodebench": _load_livecodebench,
    }
    if name not in loaders:
        raise ValueError(f"unknown benchmark: {name}")
    items = sorted(
        loaders[name](),
        key=lambda item: hashlib.sha256(f"{seed}|{item.id}".encode()).hexdigest(),
    )
    return items[:n_val], sorted(items[n_val:], key=lambda item: item.id)


def generate(
    model: Any,
    tokenizer: Any,
    items: list[BenchItem],
    chat_template: bool = True,
    greedy: bool = True,
    max_new_tokens: int = 2048,
    batch_size: int = 8,
    capture_layers: list[int] | None = None,
) -> list[Generation]:
    """Generate in fixed-size batches and optionally capture full-sequence hiddens."""
    old_padding = getattr(tokenizer, "padding_side", None)
    old_pad_token = getattr(tokenizer, "pad_token", None)
    tokenizer.padding_side = "left"
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    output: list[Generation] = []
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            prompts = [
                apply_chat_template(tokenizer, x.prompt) if chat_template else x.prompt
                for x in batch
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            encoded = _ensure_device(encoded, _model_device(model))
            kwargs: dict[str, Any] = {
                "do_sample": not greedy,
                "num_return_sequences": 1,
                "max_new_tokens": max_new_tokens,
            }
            pad_id = getattr(tokenizer, "pad_token_id", None)
            if pad_id is not None:
                kwargs["pad_token_id"] = pad_id
            generated = model.generate(**encoded, **kwargs)
            rows = generated.tolist() if torch.is_tensor(generated) else generated
            input_width = (
                encoded["input_ids"].shape[-1]
                if torch.is_tensor(encoded["input_ids"])
                else len(encoded["input_ids"][0])
            )
            texts = tokenizer.batch_decode(
                [row[input_width:] for row in rows], skip_special_tokens=True
            )
            captures: list[dict[int, torch.Tensor] | None] = [None] * len(batch)
            if capture_layers and torch.is_tensor(generated):
                full_mask = torch.ones_like(
                    generated, dtype=encoded["attention_mask"].dtype
                )
                full_mask[:, : encoded["attention_mask"].shape[1]] = encoded[
                    "attention_mask"
                ]
                with torch.no_grad():
                    forward = model(
                        input_ids=generated,
                        attention_mask=full_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                captures = [
                    {
                        layer: forward.hidden_states[layer + 1][row]
                        .detach()
                        .float()
                        .cpu()
                        for layer in capture_layers
                    }
                    for row in range(len(batch))
                ]
            output.extend(
                Generation(item.id, text, capture, input_width)
                for item, text, capture in zip(batch, texts, captures)
            )
    finally:
        if old_padding is not None:
            tokenizer.padding_side = old_padding
        if hasattr(tokenizer, "pad_token"):
            tokenizer.pad_token = old_pad_token
    return output
