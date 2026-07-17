#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_CHECKPOINTS, OLMO3_VARIANTS
from src.math_pairs import (
    GenerateFn,
    GenerationMode,
    ProblemRecord,
    VerifyFn,
    append_math_pair_jsonl,
    build_math_pairs,
    is_meaningfully_concise,
    read_math_pairs_jsonl,
    sort_math_pairs,
    write_math_pairs_jsonl,
)


MATH_500_DATASET = "HuggingFaceH4/MATH-500"
MATH_500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
DEFAULT_MODEL = OLMO3_VARIANTS["olmo3-rl-zero-math"].hf_id
DEFAULT_MODEL_REVISION = MODEL_CHECKPOINTS["olmo3-rl-zero-math"][-1]
DEFAULT_OUTPUT = "data/math-500.jsonl"


class _GenerativeModel(Protocol):
    @property
    def device(self) -> torch.device: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> torch.Tensor: ...


class _MathVerifyModule(Protocol):
    def parse(self, expression: str, *, fallback_mode: str) -> list[object]: ...

    def verify(self, gold: list[object], target: list[object]) -> bool: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare verified MATH-500 concise/verbose reasoning pairs",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--n-pairs", type=int, default=50)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _prompt(
    problem: str,
    mode: GenerationMode,
    verbose_reference: str | None,
) -> str:
    if mode == "verbose":
        return (
            "Solve the following problem step by step. Give a complete derivation, "
            "using equations for every material step and prose only where needed. "
            "End with the final answer on its own line in the form Answer: <answer>. "
            "Do not add a style or version heading.\n\n"
            f"Problem: {problem}\n"
        )
    if verbose_reference is None:
        raise ValueError("Concise generation requires a verbose reference")
    return (
        "Solve the problem using exactly the same mathematical method as the "
        "reference solution. Rewrite it as a compact, formula-dense derivation: "
        "omit explanatory prose and retain only equations and indispensable logical "
        "connectors. Use substantially fewer words than the reference and do not "
        "repeat checks or alternative methods. End with the final answer on its own line in the form "
        "Answer: <answer>. Do not add a style or version heading.\n\n"
        f"Problem: {problem}\n\n"
        f"Reference solution:\n{verbose_reference}\n\n"
        "Compact derivation:\n"
    )


def _remove_style_heading(response: str) -> str:
    return re.sub(
        r"^\s*(?:concise|verbose|detailed)(?:\s+(?:version|solution))?\s*:\s*",
        "",
        response,
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def _make_generator(args: argparse.Namespace) -> GenerateFn:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = cast(
        _GenerativeModel,
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ),
    )
    model.eval()
    torch.manual_seed(args.seed)

    def generate(
        problem: ProblemRecord,
        mode: GenerationMode,
        verbose_reference: str | None = None,
    ) -> str:
        prompt = _prompt(str(problem["problem"]), mode, verbose_reference)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model.device)
        prompt_length = encoded["input_ids"].shape[1]
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs.update(
                {"do_sample": True, "temperature": args.temperature}
            )
        else:
            generation_kwargs["do_sample"] = False
        with torch.inference_mode():
            output = model.generate(**encoded, **generation_kwargs)
        response = tokenizer.decode(
            output[0, prompt_length:],
            skip_special_tokens=True,
        )
        response = _remove_style_heading(response)
        if mode == "concise" and verbose_reference is not None:
            if not is_meaningfully_concise(response, verbose_reference):
                return ""
        return response

    return generate


def _make_verifier() -> VerifyFn:
    math_verify = cast(
        _MathVerifyModule,
        importlib.import_module("math_verify"),
    )

    def is_correct(gold_answer: str, candidate: str) -> bool:
        gold = math_verify.parse(gold_answer, fallback_mode="no_fallback")
        target = math_verify.parse(candidate, fallback_mode="no_fallback")
        return bool(gold and target and math_verify.verify(gold, target))

    return is_correct


def _load_math_500() -> list[ProblemRecord]:
    from datasets import load_dataset

    dataset = load_dataset(
        MATH_500_DATASET,
        name="default",
        split="test",
        revision=MATH_500_REVISION,
    )
    if len(dataset) != 500:
        raise ValueError(f"Expected 500 MATH-500 rows, received {len(dataset)}")
    return [dict(row) for row in dataset]


def main() -> None:
    args = parse_args()
    if args.n_pairs <= 0:
        raise ValueError("--n-pairs must be positive")
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        raise ValueError("Token limits must be positive")

    output_path = Path(args.output)
    existing = read_math_pairs_jsonl(output_path) if output_path.exists() else []
    qualified_existing = [
        pair
        for pair in existing
        if is_meaningfully_concise(pair.concise_solution, pair.verbose_solution)
    ]
    if len(qualified_existing) != len(existing):
        removed = len(existing) - len(qualified_existing)
        existing = qualified_existing
        write_math_pairs_jsonl(output_path, existing)
        print(f"Removed {removed} resumed pairs that were not materially concise")
    if len(existing) > args.n_pairs:
        raise ValueError(
            f"Output already contains {len(existing)} pairs, more than {args.n_pairs}"
        )
    if len(existing) == args.n_pairs:
        write_math_pairs_jsonl(output_path, sort_math_pairs(existing))
        print(f"Already have {len(existing)} verified pairs in {args.output}")
        return

    problems = _load_math_500()
    generate_fn = _make_generator(args)
    verify_fn = _make_verifier()

    completed = len(existing)

    def persist(pair) -> None:
        nonlocal completed
        append_math_pair_jsonl(output_path, pair)
        completed += 1
        print(
            f"[{completed}/{args.n_pairs}] verified {pair.unique_id}",
            flush=True,
        )

    pairs = build_math_pairs(
        problems,
        generate_fn,
        verify_fn,
        args.n_pairs,
        initial_pairs=existing,
        on_pair=persist,
    )
    write_math_pairs_jsonl(args.output, pairs)
    print(f"Saved {len(pairs)} verified pairs to {args.output}")


if __name__ == "__main__":
    main()
