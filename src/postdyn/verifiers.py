"""Deterministic benchmark verifiers with isolated execution for code."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:
    resource: Any = None

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
            process: subprocess.Popen[bytes] | None = None
            stdout_thread: threading.Thread | None = None
            stderr_thread: threading.Thread | None = None
            try:
                process = subprocess.Popen(
                    [sys.executable, "-I", str(script)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=directory,
                    start_new_session=True,
                    preexec_fn=_resource_limits,
                )
                stdout_chunks: list[bytes] = []
                stderr_chunks: list[bytes] = []
                stdout_thread = threading.Thread(
                    target=_read_capped,
                    args=(process.stdout, stdout_chunks),
                    daemon=True,
                )
                stderr_thread = threading.Thread(
                    target=_read_capped,
                    args=(process.stderr, stderr_chunks),
                    daemon=True,
                )
                stdout_thread.start()
                stderr_thread.start()
                if process.stdin is not None:
                    process.stdin.write(stdin.encode())
                    process.stdin.close()
                process.wait(timeout=5)
                stdout_thread.join()
                stderr_thread.join()
            except subprocess.TimeoutExpired:
                if process is not None:
                    _kill_process_group(process.pid)
                    process.wait()
                    if stdout_thread is not None:
                        stdout_thread.join()
                    if stderr_thread is not None:
                        stderr_thread.join()
                return False
            except OSError:
                return False
            stdout_text = b"".join(stdout_chunks).decode(errors="replace")
            if (
                process is None
                or process.returncode != 0
                or stdout_text.strip() != str(expected).strip()
            ):
                return False
    return True


def _read_capped(stream: Any, chunks: list[bytes], cap: int = 1_048_576) -> None:
    captured = 0
    while True:
        chunk = stream.read(65_536)
        if not chunk:
            return
        if captured < cap:
            kept = chunk[: cap - captured]
            chunks.append(kept)
            captured += len(kept)


def _resource_limits() -> None:
    if resource is None:
        return
    limits = (
        (resource.RLIMIT_CPU, (5, 5)),
        (resource.RLIMIT_AS, (1 * 1024**3, 1 * 1024**3)),
        (resource.RLIMIT_NPROC, (64, 64)),
        (resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2)),
    )
    for kind, value in limits:
        try:
            resource.setrlimit(kind, value)
        except (OSError, ValueError):
            continue


def _kill_process_group(pid: int | None) -> None:
    if pid is None or os.name != "posix":
        return
    try:
        os.killpg(pid, 9)
    except (ProcessLookupError, PermissionError, OSError):
        return


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
