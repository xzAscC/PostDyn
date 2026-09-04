from __future__ import annotations

import inspect
import io
import os
import sys
import subprocess
from typing import cast

import pytest

import postdyn.verifiers as verifiers


def test_math500_verifier_accepts_boxed_and_trailing_answers() -> None:
    assert verifiers.verify("math500", "solution \\boxed{42}", {"answer": "42"})
    assert not verifiers.verify("math500", "\\boxed{41}", {"answer": "42"})
    assert verifiers.verify("math500", "The answer is 7.", {"answer": "7"})


def test_mmlu_pro_uses_first_standalone_option_letter() -> None:
    assert verifiers.verify("mmlu_pro", "B", {"answer": "B"})
    assert verifiers.verify("mmlu_pro", "B) explanation", {"answer": "B"})
    assert not verifiers.verify("mmlu_pro", "garbage", {"answer": "B"})
    assert verifiers.verify("mmlu_pro", "...the answer is J...", {"answer": "J"})


def test_malformed_references_are_unverifiable() -> None:
    assert not verifiers.verify("livecodebench", "print(1)", {"cases": [{"stdin": 1}]})
    assert not verifiers.verify("math500", "42", {})
    assert not verifiers.verify("mmlu_pro", "A", {})
    assert not verifiers.verify(
        "ifeval", "anything", {"instruction_id_list": [1], "kwargs": [{}]}
    )


def test_preexec_guard_is_platform_specific(monkeypatch) -> None:
    monkeypatch.setattr(verifiers.os, "name", "nt")
    assert verifiers._preexec_fn() is None


def test_livecodebench_runs_isolated_subprocess_with_timeout(monkeypatch) -> None:
    calls = []

    class FakeProcess:
        pid = 123
        returncode = 0

        def __init__(self):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(b"3\n")
            self.stderr = io.BytesIO()

        def wait(self, **kwargs):
            return self.returncode

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(verifiers.subprocess, "Popen", fake_popen)
    reference = cast(
        dict[str, object],
        {"cases": [{"input": "1 2", "output": "3", "testtype": "stdin"}]},
    )
    assert verifiers.verify("livecodebench", "print(a+b)", reference)
    assert calls and calls[0][1]["start_new_session"]
    assert "exec(" not in inspect.getsource(verifiers.verify)


def test_livecodebench_timeout_is_false(monkeypatch) -> None:
    class TimeoutProcess:
        pid = 123
        returncode = -9
        stdin = io.BytesIO()
        stdout = io.BytesIO()
        stderr = io.BytesIO()

        def __init__(self):
            self.waited = False

        def wait(self, **kwargs):
            if not self.waited:
                self.waited = True
                raise subprocess.TimeoutExpired("python", 1)
            return None

    monkeypatch.setattr(
        verifiers.subprocess,
        "Popen",
        lambda *args, **kwargs: TimeoutProcess(),
    )
    assert not verifiers.verify(
        "livecodebench",
        "print(1)",
        {"cases": [{"input": "", "output": "1", "testtype": "stdin"}]},
    )


def test_livecodebench_timeout_kills_process_group() -> None:
    code = (
        "import subprocess; subprocess.Popen([%r, '-c', 'import time; time.sleep(30)'])"
        % sys.executable
    )
    assert not verifiers.verify(
        "livecodebench",
        code,
        {"cases": [{"input": "", "output": "", "testtype": "stdin"}]},
    )


def test_livecodebench_memory_limit_returns_false() -> None:
    code = "x = bytearray(4 * 1024 * 1024 * 1024)"
    assert not verifiers.verify(
        "livecodebench",
        code,
        {"cases": [{"input": "", "output": "", "testtype": "stdin"}]},
    )


def test_livecodebench_large_output_is_bounded_and_verifies() -> None:
    code = "import sys; print('3'); sys.stderr.write('x' * (10 * 1024 * 1024))"
    assert verifiers.verify(
        "livecodebench",
        code,
        {"cases": [{"input": "", "output": "3", "testtype": "stdin"}]},
    )


def test_ifeval_dispatches_namespaced_punctuation_instruction() -> None:
    reference = cast(
        dict[str, object],
        {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
    )
    assert verifiers.verify("ifeval", "no commas", reference)
    assert not verifiers.verify("ifeval", "has, comma", reference)


def test_ifeval_dispatches_parametrized_common_instruction_types() -> None:
    assert verifiers.verify(
        "ifeval",
        "one two",
        {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 2, "relation": "at least"}],
        },
    )
    assert verifiers.verify(
        "ifeval",
        "keyword keyword",
        {
            "instruction_id_list": ["keywords:frequency"],
            "kwargs": [{"keyword": "keyword", "frequency": 2, "relation": "at least"}],
        },
    )


def test_ifeval_unknown_namespaced_instruction_is_false(caplog) -> None:
    reference = cast(
        dict[str, object],
        {"instruction_id_list": ["unknown:not_implemented"], "kwargs": [{}]},
    )
    assert not verifiers.verify("ifeval", "anything", reference)
    assert "Unsupported IFEval instruction" in caplog.text


IFEVAL_CASES = [
    ("keywords:existence", {"keywords": ["alpha", "beta"]}, "Alpha beta."),
    (
        "keywords:frequency",
        {"keyword": "word", "frequency": 2, "relation": "at least"},
        "word WORD",
    ),
    ("keywords:forbidden_words", {"forbidden_words": ["secret"]}, "safe text"),
    (
        "keywords:letter_frequency",
        {"letter": "a", "let_frequency": 2, "let_relation": "at least"},
        "Alpha",
    ),
    ("language:response_language", {"language": "en"}, "This is an English answer."),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "First sentence. Second sentence.",
    ),
    (
        "length_constraints:number_words",
        {"num_words": 2, "relation": "at least"},
        "one two",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "Start"},
        "First paragraph\n\nStart here",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "First\n\nSecond",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Use [one] and [two].",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "Answer. P.S. Note",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* one\n* two",
    ),
    (
        "detectable_format:constrained_response",
        {},
        "My answer is yes.",
    ),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*one* and **two**",
    ),
    (
        "detectable_format:multiple_sections",
        {"num_sections": 2, "section_spliter": "---"},
        "one\n---\ntwo",
    ),
    ("detectable_format:json_format", {}, '{"answer": 1}'),
    ("detectable_format:title", {}, "<<A title>>\nAnswer"),
    ("combination:two_responses", {}, "First******Second"),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Repeat this"},
        "Repeat this, then answer.",
    ),
    ("startend:end_checker", {"end_phrase": "THE END"}, "Answer. THE END"),
    ("startend:quotation", {}, '"Quoted answer"'),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "NASA API",
    ),
    ("change_case:english_capital", {}, "ALL CAPITAL"),
    ("change_case:english_lowercase", {}, "all lowercase"),
    ("punctuation:no_comma", {}, "No commas"),
]


@pytest.mark.parametrize(("instruction_id", "kwargs", "response"), IFEVAL_CASES)
def test_ifeval_official_registry_true_case(
    instruction_id: str, kwargs: dict[str, object], response: str
) -> None:
    reference = cast(
        dict[str, object], {"instruction_id_list": [instruction_id], "kwargs": [kwargs]}
    )
    assert verifiers.verify("ifeval", response, reference)


@pytest.mark.parametrize(("instruction_id", "kwargs", "response"), IFEVAL_CASES)
def test_ifeval_official_registry_false_case(
    instruction_id: str, kwargs: dict[str, object], response: str
) -> None:
    false_responses = {
        "keywords:existence": "Alpha only.",
        "keywords:frequency": "word",
        "keywords:forbidden_words": "secret text",
        "keywords:letter_frequency": "A",
        "language:response_language": "Bonjour tout le monde.",
        "length_constraints:number_sentences": "Only one sentence.",
        "length_constraints:number_words": "one",
        "length_constraints:nth_paragraph_first_word": "First paragraph\n\nWrong here",
        "length_constraints:number_paragraphs": "Only one paragraph",
        "detectable_content:number_placeholders": "Use [one].",
        "detectable_content:postscript": "Answer only.",
        "detectable_format:number_bullet_lists": "* one",
        "detectable_format:constrained_response": "My answer is yes. My answer is no.",
        "detectable_format:number_highlighted_sections": "*one*",
        "detectable_format:multiple_sections": "one",
        "detectable_format:json_format": "not json",
        "detectable_format:title": "No title",
        "combination:two_responses": "Only one response",
        "combination:repeat_prompt": "The prompt was Repeat this",
        "startend:end_checker": "Answer. THE END.",
        "startend:quotation": "Quoted answer",
        "change_case:capital_word_frequency": "NASA",
        "change_case:english_capital": "ALL Capital",
        "change_case:english_lowercase": "all Lower",
        "punctuation:no_comma": "Has, comma",
    }
    reference = cast(
        dict[str, object], {"instruction_id_list": [instruction_id], "kwargs": [kwargs]}
    )
    assert not verifiers.verify("ifeval", false_responses[instruction_id], reference)


REAL_IFEVAL_IDS = [
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "combination:repeat_prompt",
    "combination:two_responses",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:constrained_response",
    "detectable_format:json_format",
    "detectable_format:multiple_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:number_highlighted_sections",
    "detectable_format:title",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:nth_paragraph_first_word",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "punctuation:no_comma",
    "startend:end_checker",
    "startend:quotation",
]

REAL_IFEVAL_ROW_FIXTURES = [
    ("keywords:existence", {"keywords": ["correlated", "experiencing"]}),
    (
        "keywords:frequency",
        {"relation": "at least", "keyword": "story", "frequency": 2},
    ),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "PARAGRAPH", "num_sections": 2},
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"first_word": "weekend", "num_paragraphs": 4, "nth_paragraph": 1},
    ),
]


def test_ifeval_registry_has_exactly_the_real_ids() -> None:
    assert set(verifiers.IFEVAL_CHECKERS) == set(REAL_IFEVAL_IDS)


def test_ifeval_real_row_fixture_kwargs_are_supported() -> None:
    assert all(
        instruction_id in verifiers.IFEVAL_CHECKERS
        for instruction_id, _ in REAL_IFEVAL_ROW_FIXTURES
    )


def test_livecodebench_functional_case_calls_named_function() -> None:
    assert verifiers.verify(
        "livecodebench",
        "def add(a, b):\n    return a + b",
        {
            "func_name": "add",
            "cases": [{"input": "2\n3", "output": "5", "testtype": "functional"}],
        },
    )


def test_livecodebench_functional_case_requires_function_name(caplog) -> None:
    assert not verifiers.verify(
        "livecodebench",
        "def add(a):\n    return a",
        {"cases": [{"input": "2", "output": "2", "testtype": "functional"}]},
    )
    assert "func_name" in caplog.text


def test_ifeval_malformed_kwargs_are_false() -> None:
    assert not verifiers.verify(
        "ifeval",
        "word",
        {
            "instruction_id_list": ["keywords:frequency"],
            "kwargs": [{"keyword": "word"}],
        },
    )


def test_split_sentences_handles_punctuation_think_blocks_and_whitespace() -> None:
    assert verifiers.split_sentences("A b. C d! E f?\nG h. ") == [
        "A b.",
        "C d!",
        "E f?",
        "G h.",
    ]
    assert verifiers.split_sentences("<think>\na b.\n</think>\nc d.\n") == [
        "a b.",
        "c d.",
    ]
