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
SIMPLE_CODE = (
    "Solve the following coding problem step by step.\n\n{question}\n\n"
    "Enclose your final solution in a single ```python code block."
)

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

# "Answer: 5", "**Answer**: 5", "### Final answer: 5", "Answers: 5", as the base writes them.
ANSWER_LINE = re.compile(
    r"^[ \t#>*]*(?:final )?answers?\**[ \t]*:\**[ \t]*(.*)$", re.M | re.I
)
# $...$ that leaves an escaped dollar sign inside alone: $\$36$ is 36 dollars.
MATH_SPAN = re.compile(r"(?<!\\)\$((?:\\.|[^$\\])+)\$")
# "5 people" as \boxed{5 \text{ people}} passes; "5 or 6" must stay as it is and fail.
TRAILING_WORDS = re.compile(r"(?<=[\d}])\s+(?!(?:or|and)\b)[A-Za-z][A-Za-z ]*$")
WORD = re.compile(r"(?<![\\A-Za-z])[A-Za-z]{2,}")
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


def _last_boxed(text: str) -> tuple[int, str] | None:
    """Where the last non-empty ``\\boxed{}`` starts, and what it holds.

    A box left open counts as the model's last word, so an earlier box does not stand in for it.
    """
    end = len(text)
    while (start := text.rfind("\\boxed{", 0, end)) >= 0:
        begin = start + len("\\boxed{")
        depth = 1
        for position in range(begin, len(text)):
            depth += {"{": 1, "}": -1}.get(text[position], 0)
            if depth == 0:
                if content := text[begin:position].strip():
                    return start, content
                break
        else:
            return None
        end = start
    return None


def _clean_line(value: str) -> str:
    """The value of an ``Answer:`` line, without Markdown, dollar signs or trailing units."""
    spans = MATH_SPAN.findall(value)
    if spans:
        # "$-2$, $1$" is a list of values; "$3$ or $5$" becomes one too, and then fails.
        value = ", ".join(spans)
    value = value.replace("°", "^\\circ").strip("* ").rstrip(" .").strip("*$ ")
    return TRAILING_WORDS.sub("", value)


def _last_answer_line(text: str) -> tuple[int, str] | None:
    """Where the last usable ``Answer:`` line starts, and its value.

    The prompt asks for the answer "on its own line after Answer:", so the value may sit on the
    next line if that ends the response. A sentence is not a value.
    """
    match = None
    for match in ANSWER_LINE.finditer(text):
        pass
    if match is None:
        return None
    value = match.group(1)
    if not value.strip("* \t"):
        following = text[match.end() :].strip().splitlines()
        value = following[0] if len(following) == 1 else ""
    value = _clean_line(value)
    if not re.search(r"\w", value) or len(WORD.findall(value)) >= 3:
        return None
    return match.start(), value


def extract_answer(text: str) -> str | None:
    """The later of the last ``\\boxed{}`` and the last ``Answer:`` line, never running text."""
    found = [answer for answer in (_last_boxed(text), _last_answer_line(text)) if answer]
    return max(found)[1] if found else None


def last_code_block(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def _balanced(text: str) -> bool:
    depth = 0
    for character in text:
        depth += {"{": 1, "}": -1}.get(character, 0)
        if depth < 0:
            return False
    return depth == 0


def _math_correct(prediction: str, answer: str) -> bool:
    from math_verify import LatexExtractionConfig, parse, verify

    # A stray } would close the box below and hide the rest of the prediction from the parser.
    if not _balanced(prediction):
        return False
    config = [LatexExtractionConfig()]
    gold = parse(f"\\boxed{{{answer}}}", extraction_config=config)
    target = parse(f"\\boxed{{{prediction}}}", extraction_config=config)
    return bool(verify(gold, target))


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
        prediction = extract_answer(response)
        if prediction is None:
            return "no_answer"
        return "correct" if _math_correct(prediction, reference["answer"]) else "wrong"
    if kind == "code":
        block = last_code_block(response)
        if block is None:
            return "no_answer"
        # ``_code`` reads a fenced block, so hand it only the block chosen here.
        return "correct" if _code(f"```python\n{LCB_PREAMBLE}{block}```", reference) else "wrong"
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
