"""pass@k from repeated samples: prompts, strict sample scoring and paired intervals.

Scoring is stricter than ``verifiers._math`` on purpose. pass@64 multiplies false positives: a
sample that is wrongly accepted 1% of the time lifts a problem's pass@64 by almost half. So only
the final response counts, an answer must be a ``\\boxed{}`` or an ``Answer:`` line, and a
response cut off by the token limit is never correct.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from .verifiers import _code

KS = (1, 4, 16, 64)
BOOTSTRAPS = 2000

BOXED_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
CODE_INSTRUCTION = "Enclose your final solution in a single ```python code block."
# Olmo 3 report (arXiv 2512.13961), Figure 37: the RL-Zero prompt, written for a base model that
# has no chat template.
SIMPLE_MATH = (
    "Solve the following math problem step by step.\n"
    "The last line of your response should be the answer to the problem in form Answer: $Answer "
    "(without quotes)\nwhere $Answer is the answer to the problem.\n\n"
    '{question}\n\nRemember to put your answer on its own line after "Answer:"'
)
# The report prints only the math prompt, so this one follows its pattern.
SIMPLE_CODE = "Solve the following coding problem step by step.\n\n{question}\n\n" + CODE_INSTRUCTION

ANSWER_LINE = re.compile(r"^\s*\**Answer:\**\s*(.+?)\s*$", re.M)
CODE_BLOCK = re.compile(r"```[\w+-]*[ \t]*\n(.*?)```", re.S)


def pass_at_k(correct: int, samples: int, k: int) -> float:
    """Chance that ``k`` samples drawn without replacement hold a correct one.

    Averaged over all subsets of size ``k``, so one set of ``samples`` answers gives an unbiased
    estimate for every ``k`` up to ``samples``.
    """
    if not 1 <= k <= samples:
        raise ValueError(f"k={k} is outside 1..{samples}")
    return 1.0 - math.comb(samples - correct, k) / math.comb(samples, k)


def final_response(text: str) -> str:
    """The text after the last ``</think>``; all of it if the thinking was never closed."""
    return text.rsplit("</think>", 1)[-1]


def _last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    begin = start + len("\\boxed{")
    depth = 1
    for position in range(begin, len(text)):
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        if depth == 0:
            return text[begin:position]
    return None


def extract_answer(text: str) -> str | None:
    """The last ``\\boxed{}``, else the last ``Answer:`` line; running text does not count."""
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed
    lines = ANSWER_LINE.findall(text)
    return lines[-1].rstrip(" .").strip("$ ") if lines else None


def last_code_block(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def _math_correct(prediction: str, answer: str) -> bool:
    from math_verify import LatexExtractionConfig, parse, verify

    config = [LatexExtractionConfig()]
    gold = parse(f"\\boxed{{{answer}}}", extraction_config=config)
    target = parse(f"\\boxed{{{prediction}}}", extraction_config=config)
    return bool(verify(gold, target))


def classify(kind: str, sample: dict[str, Any], reference: dict[str, Any]) -> str:
    """One of ``correct``, ``wrong``, ``no_answer`` (finished, no final answer), ``truncated``."""
    if sample["finish_reason"] != "stop":
        return "truncated"
    response = final_response(sample["text"])
    if kind == "math":
        prediction = extract_answer(response)
        if prediction is None:
            return "no_answer"
        return "correct" if _math_correct(prediction, reference["answer"]) else "wrong"
    if kind == "code":
        block = last_code_block(response)
        if block is None:
            return "no_answer"
        # ``_code`` reads a fenced block, so hand it only the block chosen here.
        return "correct" if _code(f"```python\n{block}```", reference) else "wrong"
    raise ValueError(f"unknown kind: {kind}")


def user_text(kind: str, question: str, prompt_format: str) -> str:
    """The user's message: ``chat`` for models with a chat template, ``simple`` for raw text."""
    if prompt_format == "chat":
        instruction = {"math": BOXED_INSTRUCTION, "code": CODE_INSTRUCTION}[kind]
        return f"{question}\n\n{instruction}"
    if prompt_format == "simple":
        return {"math": SIMPLE_MATH, "code": SIMPLE_CODE}[kind].format(question=question)
    raise ValueError(f"unknown prompt format: {prompt_format}")


def bootstrap_interval(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean, resampling problems."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    low, high = np.percentile(values[picks].mean(axis=1), [2.5, 97.5])
    return float(low), float(high)


def _sample_count(rows: Sequence[dict[str, Any]]) -> int:
    counts = {len(row["outcomes"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"problems have different sample counts: {sorted(counts)}")
    return counts.pop()


def _per_problem(rows: Sequence[dict[str, Any]], k: int, samples: int) -> np.ndarray:
    return np.array([pass_at_k(row["outcomes"].count("correct"), samples, k) for row in rows])


def task_summary(rows: Sequence[dict[str, Any]], ks: Sequence[int] = KS) -> dict[str, Any]:
    """pass@k with intervals, outcome shares and token use of one task."""
    samples = _sample_count(rows)
    outcomes = Counter(outcome for row in rows for outcome in row["outcomes"])
    tokens = [count for row in rows for count in row["tokens"]]
    summary: dict[str, Any] = {
        "n_problems": len(rows),
        "n_samples": samples,
        "outcomes": {name: count / sum(outcomes.values()) for name, count in outcomes.items()},
        "mean_tokens": float(np.mean(tokens)),
    }
    for k in ks:
        if k <= samples:
            values = _per_problem(rows, k, samples)
            low, high = bootstrap_interval(values)
            summary[f"pass@{k}"] = {"value": float(values.mean()), "low": low, "high": high}
    return summary


def compare(
    a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]], ks: Sequence[int] = KS
) -> dict[str, Any]:
    """b minus a, paired by problem, with intervals that resample problems."""
    by_id = {row["id"]: row for row in b}
    if set(by_id) != {row["id"] for row in a}:
        raise ValueError("the two runs hold different problems")
    paired = [by_id[row["id"]] for row in a]
    samples_a, samples_b = _sample_count(a), _sample_count(paired)
    result: dict[str, Any] = {}
    for k in ks:
        if k <= min(samples_a, samples_b):
            gaps = _per_problem(paired, k, samples_b) - _per_problem(a, k, samples_a)
            low, high = bootstrap_interval(gaps)
            result[f"pass@{k}"] = {"difference": float(gaps.mean()), "low": low, "high": high}
    return result
