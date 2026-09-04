"""Deterministic benchmark verifiers with isolated execution for code."""

from __future__ import annotations

import json
import importlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
from decimal import Decimal, InvalidOperation
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
    func_name = reference.get("func_name")
    with tempfile.TemporaryDirectory(prefix="postdyn-code-") as directory:
        script = Path(directory) / "solution.py"
        script.write_text(code, encoding="utf-8")
        for case in cases:
            functional = isinstance(case, dict) and (
                case.get("testtype") == "functional"
                or "args" in case
                or "expected_output" in case
            )
            if functional:
                if not isinstance(func_name, str) or not func_name:
                    logger.warning("LiveCodeBench functional case has no func_name")
                    return False
                if not _run_functional_case(directory, script, case, func_name):
                    return False
                continue
            stdin = (
                case.get("input", case.get("stdin", ""))
                if isinstance(case, dict)
                else ""
            )
            expected = (
                case.get("output", case.get("stdout", ""))
                if isinstance(case, dict)
                else ""
            )
            ok, stdout_text = _run_sandboxed(
                [sys.executable, "-I", str(script)], directory, stdin
            )
            if not ok:
                return False
            if not _stdio_outputs_equal(stdout_text, str(expected)):
                return False
    return True


def _run_functional_case(
    directory: str, script: Path, case: dict[str, Any], func_name: str
) -> bool:
    driver = Path(directory) / "functional_driver.py"
    driver.write_text(
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "spec = importlib.util.spec_from_file_location('solution', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "name = sys.argv[2]\n"
        "function = getattr(module, name, None)\n"
        "if function is None and hasattr(module, 'Solution'):\n"
        "    function = getattr(module.Solution(), name, None)\n"
        "if function is None:\n"
        "    raise AttributeError(name)\n"
        "lines = sys.stdin.read().splitlines()\n"
        "args = []\n"
        "for line in lines:\n"
        "    try:\n"
        "        args.append(json.loads(line))\n"
        "    except json.JSONDecodeError:\n"
        "        args.append(line)\n"
        "result = function(*args)\n"
        "sys.stdout.write(json.dumps(result))\n",
        encoding="utf-8",
    )
    value = case.get("input", case.get("args", ""))
    if isinstance(value, list):
        input_text = "\n".join(json.dumps(item) for item in value)
    else:
        input_text = str(value)
    expected = case.get("output", case.get("expected_output", ""))
    ok, captured = _run_sandboxed(
        [sys.executable, "-I", str(driver), str(script), func_name],
        directory,
        input_text,
    )
    if not ok:
        return False
    return _functional_outputs_equal(str(expected), captured)


def _functional_outputs_equal(expected: str, captured: str) -> bool:
    try:
        expected_value = json.loads(expected)
        captured_value = json.loads(captured)
    except (TypeError, json.JSONDecodeError):
        return False
    return _normalize_json_value(expected_value) == _normalize_json_value(
        captured_value
    )


def _stdio_outputs_equal(predicted: str, expected: str) -> bool:
    predicted_lines = [line.strip() for line in predicted.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]
    if len(predicted_lines) != len(expected_lines):
        return False
    for predicted_line, expected_line in zip(predicted_lines, expected_lines):
        if predicted_line == expected_line:
            continue
        try:
            predicted_tokens = [Decimal(token) for token in predicted_line.split()]
            expected_tokens = [Decimal(token) for token in expected_line.split()]
        except (InvalidOperation, ValueError):
            return False
        if predicted_tokens != expected_tokens:
            return False
    return True


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    return value


def _run_sandboxed(command: list[str], directory: str, stdin: str) -> tuple[bool, str]:
    process: subprocess.Popen[Any] | None = None
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    try:
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "cwd": directory,
            "start_new_session": True,
            "env": {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        }
        preexec = _preexec_fn()
        if preexec is not None:
            popen_kwargs["preexec_fn"] = preexec
        process = subprocess.Popen(command, **popen_kwargs)
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_thread = threading.Thread(
            target=_read_capped, args=(process.stdout, stdout_chunks), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_read_capped, args=(process.stderr, stderr_chunks), daemon=True
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
        return False, ""
    except OSError:
        return False, ""
    if process is None or process.returncode != 0:
        return False, ""
    return True, b"".join(stdout_chunks).decode(errors="replace")


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


def _preexec_fn() -> Any:
    if os.name == "posix" and resource is not None and hasattr(resource, "setrlimit"):
        return _resource_limits
    return None


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


def _relation(value: int, target: int, relation: Any) -> bool:
    if relation == "at least":
        return value >= target
    if relation == "less than":
        return value < target
    return False


def _keywords_existence(kwargs: dict[str, Any], response: str) -> bool:
    keywords = kwargs["keywords"]
    return (
        isinstance(keywords, list)
        and bool(keywords)
        and all(
            isinstance(keyword, str) and re.search(keyword, response, re.I)
            for keyword in keywords
        )
    )


def _keywords_frequency(kwargs: dict[str, Any], response: str) -> bool:
    keyword = kwargs["keyword"]
    return (
        isinstance(keyword, str)
        and bool(keyword)
        and _relation(
            len(re.findall(keyword, response, re.I)),
            int(kwargs["frequency"]),
            kwargs["relation"],
        )
    )


def _keywords_forbidden(kwargs: dict[str, Any], response: str) -> bool:
    words = kwargs["forbidden_words"]
    return (
        isinstance(words, list)
        and bool(words)
        and all(
            isinstance(word, str) and not re.search(rf"\b{word}\b", response, re.I)
            for word in words
        )
    )


def _letter_frequency(kwargs: dict[str, Any], response: str) -> bool:
    letter = kwargs["letter"]
    return (
        isinstance(letter, str)
        and len(letter) == 1
        and _relation(
            response.lower().count(letter.lower()),
            int(kwargs["let_frequency"]),
            kwargs["let_relation"],
        )
    )


def _response_language(kwargs: dict[str, Any], response: str) -> bool:
    language = kwargs["language"]
    if not isinstance(language, str) or len(language) != 2:
        return False
    try:
        langdetect = importlib.import_module("langdetect")
    except ImportError:
        logger.warning("langdetect is unavailable; response language check failed")
        return False
    try:
        return langdetect.detect(response) == language.lower()
    except getattr(langdetect, "LangDetectException", Exception):
        return False


def _count_words(response: str) -> int:
    nltk = importlib.import_module("nltk")
    return len(nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(response))


def _count_sentences(response: str) -> int:
    nltk = importlib.import_module("nltk")
    tokenizer = nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    return len(tokenizer.tokenize(response))


def _number_sentences(kwargs: dict[str, Any], response: str) -> bool:
    return _relation(
        _count_sentences(response),
        int(kwargs["num_sentences"]),
        kwargs["relation"],
    )


def _number_words(kwargs: dict[str, Any], response: str) -> bool:
    return _relation(
        _count_words(response), int(kwargs["num_words"]), kwargs["relation"]
    )


def _number_paragraphs(kwargs: dict[str, Any], response: str) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    count = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
    target = int(kwargs["num_paragraphs"])
    return count == target


def _nth_paragraph_first_word(kwargs: dict[str, Any], response: str) -> bool:
    paragraphs = re.split(r"\n\n", response)
    count = len(paragraphs)
    for paragraph in paragraphs:
        if not paragraph.strip():
            count -= 1
    index = int(kwargs["nth_paragraph"]) - 1
    if not 0 <= index < len(paragraphs) or not paragraphs[index].strip():
        return False
    word = paragraphs[index].strip().split()[0].lstrip("'\"")
    punctuation = {".", ",", "?", "!", "'", '"'}
    first_word = ""
    for letter in word:
        if letter in punctuation:
            break
        first_word += letter.lower()
    return (
        count == int(kwargs["num_paragraphs"])
        and first_word == str(kwargs["first_word"]).lower()
    )


def _number_placeholders(kwargs: dict[str, Any], response: str) -> bool:
    return len(re.findall(r"\[.*?\]", response)) >= int(kwargs["num_placeholders"])


def _postscript(kwargs: dict[str, Any], response: str) -> bool:
    marker = kwargs["postscript_marker"]
    if not isinstance(marker, str) or not marker:
        return False
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = rf"\s*{marker.lower()}.*$"
    return bool(re.findall(pattern, response.lower(), re.M))


def _number_bullet_lists(kwargs: dict[str, Any], response: str) -> bool:
    bullets = re.findall(r"^\s*\*[^\*].*$", response, re.M)
    bullets.extend(re.findall(r"^\s*-.*$", response, re.M))
    return len(bullets) == int(kwargs["num_bullets"])


def _constrained_response(kwargs: dict[str, Any], response: str) -> bool:
    choices = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return any(choice in response.strip() for choice in choices)


def _number_highlighted_sections(kwargs: dict[str, Any], response: str) -> bool:
    highlights = re.findall(r"\*[^\n\*]*\*", response)
    double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", response)
    count = sum(bool(highlight.strip("*").strip()) for highlight in highlights)
    count += sum(
        bool(highlight.removeprefix("**").removesuffix("**").strip())
        for highlight in double_highlights
    )
    return count >= int(kwargs["num_highlights"])


def _multiple_sections(kwargs: dict[str, Any], response: str) -> bool:
    marker = kwargs.get("section_spliter")
    if not isinstance(marker, str) or not marker:
        return False
    sections = re.split(rf"\s?{marker}\s?\d+\s?", response)
    return len(sections) - 1 >= int(kwargs["num_sections"])


def _json_format(kwargs: dict[str, Any], response: str) -> bool:
    candidate = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(candidate)
    except (TypeError, ValueError):
        return False
    return True


def _title(kwargs: dict[str, Any], response: str) -> bool:
    return bool(re.search(r"<<[^<>]+>>", response))


def _two_responses(kwargs: dict[str, Any], response: str) -> bool:
    parts = response.split("******")
    if len(parts) != 2:
        return False
    first, second = (part.strip() for part in parts)
    return first != second and bool(first or second)


def _repeat_prompt(kwargs: dict[str, Any], response: str) -> bool:
    prompt = kwargs["prompt_to_repeat"]
    return (
        isinstance(prompt, str)
        and bool(prompt.strip())
        and response.strip().lower().startswith(prompt.strip().lower())
    )


def _end_checker(kwargs: dict[str, Any], response: str) -> bool:
    phrase = kwargs["end_phrase"]
    if not isinstance(phrase, str) or not phrase:
        return False
    candidate = response.strip()
    if candidate.startswith('"') and candidate.endswith('"'):
        candidate = candidate[1:-1].strip()
    return candidate.lower().endswith(phrase.lower())


def _quotation(kwargs: dict[str, Any], response: str) -> bool:
    stripped = response.strip()
    return len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')


def _capital_word_frequency(kwargs: dict[str, Any], response: str) -> bool:
    nltk = importlib.import_module("nltk")
    tokens = nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(response)
    return _relation(
        len([token for token in tokens if token.isupper()]),
        int(kwargs["capital_frequency"]),
        kwargs["capital_relation"],
    )


def _english_capital(kwargs: dict[str, Any], response: str) -> bool:
    return _case_and_language(response, str.isupper)


def _english_lowercase(kwargs: dict[str, Any], response: str) -> bool:
    return _case_and_language(response, str.islower)


def _case_and_language(response: str, case_fn: Any) -> bool:
    if not response or not case_fn(response):
        return False
    try:
        langdetect = importlib.import_module("langdetect")
    except ImportError:
        return False
    try:
        return langdetect.detect(response) == "en"
    except getattr(langdetect, "LangDetectException", Exception):
        return False


def _no_comma(kwargs: dict[str, Any], response: str) -> bool:
    return "," not in response


class _IFEvalRegistry(dict[str, Any]):
    """Complete official 25-ID namespaced IFEval checker registry."""


IFEVAL_CHECKERS = _IFEvalRegistry(
    {
        "keywords:existence": _keywords_existence,
        "keywords:frequency": _keywords_frequency,
        "keywords:forbidden_words": _keywords_forbidden,
        "keywords:letter_frequency": _letter_frequency,
        "language:response_language": _response_language,
        "length_constraints:number_sentences": _number_sentences,
        "length_constraints:number_words": _number_words,
        "length_constraints:number_paragraphs": _number_paragraphs,
        "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
        "detectable_content:number_placeholders": _number_placeholders,
        "detectable_content:postscript": _postscript,
        "detectable_format:number_bullet_lists": _number_bullet_lists,
        "detectable_format:constrained_response": _constrained_response,
        "detectable_format:number_highlighted_sections": _number_highlighted_sections,
        "detectable_format:multiple_sections": _multiple_sections,
        "detectable_format:json_format": _json_format,
        "detectable_format:title": _title,
        "combination:two_responses": _two_responses,
        "combination:repeat_prompt": _repeat_prompt,
        "startend:end_checker": _end_checker,
        "startend:quotation": _quotation,
        "change_case:capital_word_frequency": _capital_word_frequency,
        "change_case:english_capital": _english_capital,
        "change_case:english_lowercase": _english_lowercase,
        "punctuation:no_comma": _no_comma,
    }
)


def _check_ifeval(reference: dict[str, object], response: str) -> bool:
    instruction_ids = reference.get("instruction_id_list")
    kwargs_list = reference.get("kwargs")
    if (
        not isinstance(instruction_ids, list)
        or not instruction_ids
        or not all(
            isinstance(instruction_id, str) for instruction_id in instruction_ids
        )
        or not isinstance(kwargs_list, list)
        or len(kwargs_list) != len(instruction_ids)
        or not all(isinstance(kwargs, dict) for kwargs in kwargs_list)
    ):
        return False
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list):
        checker = IFEVAL_CHECKERS.get(instruction_id.lower())
        if checker is None:
            logger.warning("Unsupported IFEval instruction: %s", instruction_id)
            return False
        try:
            if not checker(kwargs, response):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def verify(benchmark: str, generation_text: str, reference: dict[str, object]) -> bool:
    """Verify one generated answer against its benchmark payload."""
    if benchmark == "math500":
        try:
            return _math(generation_text, str(reference["answer"]))
        except (KeyError, TypeError):
            return False
    if benchmark == "mmlu_pro":
        try:
            return _first_option(generation_text) == str(reference["answer"]).upper()
        except (KeyError, TypeError):
            return False
    if benchmark == "livecodebench":
        try:
            cases = reference.get("cases", reference.get("test_cases", []))
            if not isinstance(cases, list) or any(
                not isinstance(case, dict) for case in cases
            ):
                return False
            return _code(generation_text, reference)
        except (AttributeError, TypeError):
            return False
    if benchmark == "ifeval":
        if not isinstance(reference, dict):
            return False
        return _check_ifeval(reference, generation_text)
    raise ValueError(f"unknown benchmark: {benchmark}")
