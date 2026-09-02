#!/usr/bin/env python3
"""Run the resumable raw-prompt MATH-500 first-50 baseline."""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OLMO3_VARIANTS
from src.downstream_eval import GreedyGenerator, TokenizerLike
from src.math500_eval import (
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_QUANTIZATION,
    evaluate_first50,
)

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL_KEY = "olmo3-think-sft"
DEFAULT_REVISION = "main"
DEFAULT_DATASET = "datasets/math500.json"
DEFAULT_OUTPUT = "results/math500_first50"
THINK_MODEL_KEYS = tuple(
    key for key in OLMO3_VARIANTS if key.startswith("olmo3-think-")
)


class _Model(Protocol):
    def generate(
        self,
        input_ids: "torch.Tensor",
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> "torch.Tensor": ...

    def eval(self) -> object: ...

    def get_input_embeddings(self) -> object: ...


def load_model_and_tokenizer(
    model_key: str,
    revision: str,
    dtype: str = DEFAULT_DTYPE,
    quantization: str = DEFAULT_QUANTIZATION,
) -> tuple[_Model, TokenizerLike]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    if quantization not in {"none", "4bit", "8bit"}:
        raise ValueError(f"unsupported quantization: {quantization!r}")

    hf_id = OLMO3_VARIANTS[model_key].hf_id
    raw_tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if getattr(raw_tokenizer, "pad_token", None) is None:
        setattr(raw_tokenizer, "pad_token", getattr(raw_tokenizer, "eos_token", None))
    tokenizer = cast(TokenizerLike, cast(object, raw_tokenizer))
    load_kwargs: dict[str, object] = {
        "revision": revision,
        "dtype": dtype_map[dtype],
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if quantization != "none":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
        )
        load_kwargs.pop("dtype")
    model = cast(
        _Model,
        cast(
            object,
            AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs),
        ),
    )
    model.eval()
    return model, tokenizer


def _model_device(model: _Model) -> str:
    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    device = getattr(weight, "device", None)
    if device is None:
        device = getattr(model, "device", "cuda")
    return str(device)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the ordered first 50 MATH-500 items"
    )
    parser.add_argument(
        "--model-key", choices=THINK_MODEL_KEYS, default=DEFAULT_MODEL_KEY
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.max_new_tokens <= 0:
        print("ERROR: --max-new-tokens must be positive", file=sys.stderr)
        return 2
    hf_id = OLMO3_VARIANTS[args.model_key].hf_id
    model, tokenizer = load_model_and_tokenizer(
        args.model_key, args.revision, args.dtype, args.quantization
    )
    try:
        generator = GreedyGenerator(
            model=model, tokenizer=tokenizer, device=_model_device(model)
        )
        summary = evaluate_first50(
            model=hf_id,
            model_key=args.model_key,
            revision=args.revision,
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output),
            tokenizer=tokenizer,
            generator=generator,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            quantization=args.quantization,
            force=args.force,
            progress=lambda message: print(message, flush=True),
        )
    finally:
        del model, tokenizer
        gc.collect()
    print(
        f"DONE: {summary.n_processed}/{summary.n_expected} items; accuracy={summary.accuracy:.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
