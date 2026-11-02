"""pass@k from repeated samples: prompts, strict sample scoring and paired intervals.

Both models get the same user text and answer in the same form, so one scoring path decides for
both and cannot favour either. Scoring is stricter than ``verifiers._math`` on purpose, because
pass@64 multiplies false positives: a sample wrongly accepted 1% of the time lifts a problem's
pass@64 by almost half. Only the final response counts, only its last ``\\boxed{}`` is an answer,
and a response cut off by the token limit is never correct.
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
# LiveCodeBench's import_string (lcb_runner/evaluation/testing_util.py): the official harness puts
# it before every solution, so a solution may use List or defaultdict without importing them.
LCB_PREAMBLE = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\nimport string\nimport re\n"
    "import datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\n"
    "import math\nimport random\nimport statistics\nimport itertools\nimport functools\n"
    "import operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)

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


def last_code_block(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def _math_outcome(response: str, answer: str) -> str:
    """Judge the last box only: math-verify merges all boxes, which fails a corrected answer."""
    from math_verify import LatexExtractionConfig, parse, verify

    start = response.rfind("\\boxed{")
    if start < 0:
        return "no_answer"
    config = [LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0)]
    target = parse(response[start:], extraction_config=config)
    if not target:
        return "no_answer"  # empty or never closed
    gold = parse(f"\\boxed{{{answer}}}", extraction_config=[LatexExtractionConfig()])
    return "correct" if verify(gold, target) else "wrong"


def classify(
    kind: str, sample: dict[str, Any], reference: dict[str, Any], prompt_format: str
) -> str:
    """One of ``correct``, ``wrong``, ``no_answer`` (finished, no final answer), ``truncated``."""
    if sample["finish_reason"] != "stop":
        return "truncated"
    # The chat prompt ends in <think>, so without </think> the model never left its thinking.
    if prompt_format == "chat" and "</think>" not in sample["text"]:
        return "no_answer"
    response = final_response(sample["text"])
    if kind == "math":
        return _math_outcome(response, reference["answer"])
    if kind == "code":
        block = last_code_block(response)
        if block is None:
            return "no_answer"
        # ``_code`` reads a fenced block, so hand it only the block chosen here.
        return "correct" if _code(f"```python\n{LCB_PREAMBLE}{block}```", reference) else "wrong"
    raise ValueError(f"unknown kind: {kind}")


def user_text(kind: str, question: str) -> str:
    """The user's message; the same for every model, whatever template wraps it."""
    instruction = {"math": BOXED_INSTRUCTION, "code": CODE_INSTRUCTION}[kind]
    return f"{question}\n\n{instruction}"


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
