"""Deterministic benchmark verifiers with isolated execution for code."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation or newlines, including reasoning text."""
    chunks = re.split(
        r"(?<=[.!?])\s+|\n+", text.replace("<think>", "").replace("</think>", "")
    )
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _math(generation: str, answer: str) -> bool:
    try:
        import importlib

        math_verify = importlib.import_module("math_verify")
        parse = math_verify.parse
        verify = math_verify.verify

        target = parse(generation)
        gold = parse(f"${answer}$")
        if verify(gold, target):
            return True
        candidates = re.findall(
            r"\\boxed\{([^{}]+)\}|(?:final answer|answer)\s*[:=]?\s*([^\n.]+)",
            generation,
            re.I,
        )
        for candidate in candidates[::-1]:
            expression = next((part for part in candidate if part), "")
            try:
                if verify(gold, parse(expression)):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return bool(re.search(rf"(?:^|\D){re.escape(answer)}(?:$|\D)", generation))


def _code(generation: str, reference: dict[str, Any]) -> bool:
    code = generation
    fenced = re.search(r"```(?:python)?\s*\n?(.*?)```", generation, re.S | re.I)
    if fenced:
        code = fenced.group(1)
    cases = reference.get("cases", reference.get("test_cases", []))
    if not cases:
        return False
    with tempfile.TemporaryDirectory(prefix="postdyn-code-") as directory:
        script = Path(directory) / "solution.py"
        script.write_text(code, encoding="utf-8")
        for case in cases:
            stdin = case.get("stdin", "") if isinstance(case, dict) else ""
            expected = case.get("stdout", "") if isinstance(case, dict) else ""
            try:
                result = subprocess.run(
                    [sys.executable, "-I", str(script)],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=directory,
                )
            except (subprocess.TimeoutExpired, OSError):
                return False
            if result.returncode != 0 or result.stdout.strip() != str(expected).strip():
                return False
    return True


def _first_option(text: str) -> str | None:
    match = re.search(r"\b([A-J])\b", text.upper())
    return match.group(1) if match else None


def _constraint_checker(
    instruction: str, kwargs: dict[str, Any], response: str
) -> bool:
    name = instruction.lower().replace(" ", "_")
    words = response.split()
    if name in {"num_words", "min_words", "max_words"}:
        count = len(words)
        if name == "num_words":
            return count == int(kwargs.get("num_words", kwargs.get("count", 0)))
        if name == "min_words":
            return count >= int(kwargs.get("num_words", kwargs.get("count", 0)))
        return count <= int(kwargs.get("num_words", kwargs.get("count", 0)))
    if name in {"num_sentences", "min_sentences", "max_sentences"}:
        count = len(split_sentences(response))
        target = int(kwargs.get("num_sentences", kwargs.get("count", 0)))
        return (
            count == target
            if name == "num_sentences"
            else count >= target
            if name == "min_sentences"
            else count <= target
        )
    if name in {"num_bullets", "num_paragraphs"}:
        count = (
            len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", response))
            if name == "num_bullets"
            else len([p for p in re.split(r"\n\s*\n", response) if p.strip()])
        )
        return count == int(kwargs.get(name, kwargs.get("count", 0)))
    if name in {"required_words", "forbidden_words"}:
        required = kwargs.get("words", kwargs.get("word", []))
        required = [required] if isinstance(required, str) else required
        return (
            all(w.lower() in response.lower() for w in required)
            if name == "required_words"
            else all(w.lower() not in response.lower() for w in required)
        )
    if name == "keyword_frequency":
        key = str(kwargs.get("keyword", kwargs.get("word", "")))
        return response.lower().count(key.lower()) == int(
            kwargs.get("frequency", kwargs.get("count", 0))
        )
    if name == "end_phrase":
        return response.rstrip().endswith(
            str(kwargs.get("phrase", kwargs.get("end_phrase", "")))
        )
    if name == "first_word":
        return (
            bool(words)
            and words[0].strip('.,:;!?"').lower() == str(kwargs.get("word", "")).lower()
        )
    if name == "forbidden_words":
        return True
    if name == "comma_frequency":
        return response.count(",") == int(
            kwargs.get("frequency", kwargs.get("count", 0))
        )
    if name == "quotation":
        return response.count('"') >= 2
    logger.warning("Unsupported IFEval instruction: %s", instruction)
    return False


def check_prompt_level(instructions: Any, kwargs_list: Any, response: str) -> bool:
    """Check the common prompt-level IFEval constraints."""
    if isinstance(instructions, str):
        instructions = [instructions]
    kwargs_list = kwargs_list or [{}]
    return all(
        _constraint_checker(
            str(inst), kwargs_list[i] if i < len(kwargs_list) else {}, response
        )
        for i, inst in enumerate(instructions or [])
    )


ifeval_check = check_prompt_level
_DEFAULT_IFEVAL_CHECK = check_prompt_level


def verify(benchmark: str, generation_text: str, reference: dict[str, object]) -> bool:
    """Verify one generated answer against its benchmark payload."""
    if benchmark == "math500":
        return _math(generation_text, str(reference["answer"]))
    if benchmark == "mmlu_pro":
        return _first_option(generation_text) == str(reference["answer"]).upper()
    if benchmark == "livecodebench":
        return _code(generation_text, reference)
    if benchmark == "ifeval":
        prompt = reference.get("instructions", reference.get("instruction", []))
        kwargs = reference.get("kwargs", [])
        checker: Any = (
            check_prompt_level
            if check_prompt_level is not _DEFAULT_IFEVAL_CHECK
            else ifeval_check
        )
        try:
            return checker(prompt, kwargs, generation_text)
        except TypeError:
            return checker(prompt, generation_text)
    raise ValueError(f"unknown benchmark: {benchmark}")
