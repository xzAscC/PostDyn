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
        "First\n***\nSecond",
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
        {"num_sections": 2, "section_spliter": "SECTION"},
        "SECTION 1\nSECTION 2\nSECTION 3",
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
        "detectable_format:number_bullet_lists": "* one\n* two\n* three",
        "detectable_format:constrained_response": "My answer is yes. My answer is no.",
        "detectable_format:number_highlighted_sections": "*one*",
        "detectable_format:multiple_sections": "SECTION 1",
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


def test_ifeval_number_paragraphs_uses_markdown_divider() -> None:
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["length_constraints:number_paragraphs"],
            "kwargs": [{"num_paragraphs": 2}],
        },
    )
    assert verifiers.verify("ifeval", "one\n***\ntwo", reference)
    assert not verifiers.verify("ifeval", "one\n***\ntwo\n***\nthree", reference)


def test_ifeval_number_paragraphs_none_relation_defaults_to_exact() -> None:
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["length_constraints:number_paragraphs"],
            "kwargs": [{"num_paragraphs": 2, "relation": None}],
        },
    )
    assert verifiers.verify("ifeval", "one\n***\ntwo", reference)
    assert not verifiers.verify("ifeval", "one\n***\ntwo\n***\nthree", reference)


def test_ifeval_number_bullets_rejects_surplus_bullets() -> None:
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["detectable_format:number_bullet_lists"],
            "kwargs": [{"num_bullets": 2}],
        },
    )
    assert not verifiers.verify("ifeval", "* one\n* two\n* three", reference)


def test_ifeval_official_edge_checker_semantics() -> None:
    # instructions.py:280-326: the official regex accepts both * and - markers.
    assert verifiers.verify(
        "ifeval",
        "- one\n- two",
        {
            "instruction_id_list": ["detectable_format:number_bullet_lists"],
            "kwargs": [{"num_bullets": 2}],
        },
    )
    # instructions.py:873-905: json.loads accepts every JSON value, including scalars.
    assert verifiers.verify(
        "ifeval",
        "42",
        {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]},
    )
    # instructions.py:329-363: any allowed phrase is sufficient.
    assert verifiers.verify(
        "ifeval",
        "My answer is yes. My answer is no.",
        {
            "instruction_id_list": ["detectable_format:constrained_response"],
            "kwargs": [{}],
        },
    )
    # instructions.py:908-1010: first-word comparison is case-insensitive.
    assert verifiers.verify(
        "ifeval",
        "First paragraph\n\nstart here",
        {
            "instruction_id_list": ["length_constraints:nth_paragraph_first_word"],
            "kwargs": [
                {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "Start"}
            ],
        },
    )


def test_ifeval_official_regex_and_normalization_semantics() -> None:
    # instructions.py:703-810: keyword checks use unescaped regex searches.
    assert verifiers.verify(
        "ifeval",
        "a.b",
        {
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["a.b"]}],
        },
    )
    # instructions.py:1070-1112: forbidden words are whole-word regex matches.
    assert verifiers.verify(
        "ifeval",
        "secretary",
        {
            "instruction_id_list": ["keywords:forbidden_words"],
            "kwargs": [{"forbidden_words": ["secret"]}],
        },
    )
    # instructions_util.py:125-130: words are \w+ tokens, not whitespace tokens.
    assert verifiers.verify(
        "ifeval",
        "one-two",
        {
            "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 3, "relation": "less than"}],
        },
    )
    # instructions.py:1171-1210: exactly two nonempty distinct responses;
    # at most ONE empty fragment, and only at the original first or last index.
    assert verifiers.verify(
        "ifeval",
        "******first******second",
        {"instruction_id_list": ["combination:two_responses"], "kwargs": [{}]},
    )
    assert verifiers.verify(
        "ifeval",
        "first******second******",
        {"instruction_id_list": ["combination:two_responses"], "kwargs": [{}]},
    )
    assert not verifiers.verify(
        "ifeval",
        "************first******second",
        {"instruction_id_list": ["combination:two_responses"], "kwargs": [{}]},
    )
    assert not verifiers.verify(
        "ifeval",
        "first******",
        {"instruction_id_list": ["combination:two_responses"], "kwargs": [{}]},
    )
    # instructions.py:1213-1247: prompt and response are stripped and compared ignoring case.
    assert verifiers.verify(
        "ifeval",
        "  REPEAT THIS, then answer",
        {
            "instruction_id_list": ["combination:repeat_prompt"],
            "kwargs": [{"prompt_to_repeat": "Repeat this"}],
        },
    )


def test_ifeval_language_detection_failure_counts_as_followed(monkeypatch) -> None:
    # instructions.py:113-164: LangDetectException counts as followed.
    class FakeLangDetectException(Exception):
        pass

    class FakeLangdetect:
        LangDetectException = FakeLangDetectException

        @staticmethod
        def detect(_response):
            raise FakeLangDetectException

    monkeypatch.setitem(sys.modules, "langdetect", FakeLangdetect)
    assert verifiers.verify(
        "ifeval",
        "text",
        {
            "instruction_id_list": ["language:response_language"],
            "kwargs": [{"language": "en"}],
        },
    )


def test_ifeval_title_and_end_checker_match_upstream_normalization() -> None:
    # instructions.py:1286-1313: <<[^\n]+>> present with a nonempty title;
    # multiple titles are accepted and empty <<>> is rejected.
    title = cast(
        dict[str, object],
        {"instruction_id_list": ["detectable_format:title"], "kwargs": [{}]},
    )
    assert verifiers.verify("ifeval", "<<one>> <<two>>", title)
    assert not verifiers.verify("ifeval", "<<>>", title)
    # instructions.py:1266-1301: strip outer whitespace and ALL outer quotes
    # (.strip().strip('"')); do not strip whitespace introduced inside them.
    end = cast(
        dict[str, object],
        {
            "instruction_id_list": ["startend:end_checker"],
            "kwargs": [{"end_phrase": "THE END"}],
        },
    )
    assert verifiers.verify("ifeval", '"Answer. THE END"', end)
    assert verifiers.verify("ifeval", '""Answer. THE END""', end)
    assert not verifiers.verify("ifeval", '"Answer. THE END "', end)


def test_ifeval_capital_word_frequency_uses_regexp_tokenizer() -> None:
    # instructions_util.py:125-130: RegexpTokenizer(r"\w+") supplies tokens.
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["change_case:capital_word_frequency"],
            "kwargs": [{"capital_frequency": 2, "capital_relation": "at least"}],
        },
    )
    assert verifiers.verify("ifeval", "NASA-API", reference)


def test_nltk_sentence_resource_is_provisioned_lazily(monkeypatch) -> None:
    class FakeData:
        def __init__(self):
            self.find_calls = []
            self.downloaded = False

        def find(self, resource):
            self.find_calls.append(resource)
            if self.downloaded and resource == "tokenizers/punkt_tab":
                return object()
            if resource == "tokenizers/punkt_tab":
                raise LookupError(resource)
            if resource == "tokenizers/punkt":
                raise LookupError(resource)
            return object()

        def load(self, _resource):
            return type("Tokenizer", (), {"tokenize": lambda _self, text: [text]})()

    data = FakeData()
    downloads = []
    fake_nltk = type(
        "FakeNltk",
        (),
        {
            "data": data,
            "download": staticmethod(
                lambda resource, quiet: (
                    downloads.append((resource, quiet))
                    or setattr(data, "downloaded", True)
                    or True
                )
            ),
        },
    )
    monkeypatch.setattr(verifiers, "_punkt_provision_attempted", False)
    monkeypatch.setattr(verifiers.importlib, "import_module", lambda name: fake_nltk)
    assert verifiers._count_sentences("one") == 1
    assert downloads == [("punkt_tab", True)]


def test_ifeval_multiple_sections_is_at_least_numbered_markers() -> None:
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["detectable_format:multiple_sections"],
            "kwargs": [{"section_spliter": "SECTION", "num_sections": 2}],
        },
    )
    assert verifiers.verify("ifeval", "SECTION 1\nSECTION 2\nSECTION 3", reference)
    assert not verifiers.verify("ifeval", "SECTION 1", reference)


@pytest.mark.parametrize(
    ("case_output", "captured", "expected"),
    [
        ("true", "true", True),
        ("[1, 4]", "[1, 4]", True),
        ('"answer"', '"answer"', True),
        ("false", "true", False),
        ("[1, 2]", "[1, 3]", False),
        ('"answer"', '"other"', False),
        ("8", "9", False),
    ],
)
def test_livecodebench_functional_outputs_compare_as_json_types(
    case_output: str, captured: str, expected: bool
) -> None:
    assert verifiers._functional_outputs_equal(case_output, captured) is expected


def test_livecodebench_official_stdin_decimal_line_comparison() -> None:
    # testing_util.py:389-425: strip each line, then compare Decimal token lists.
    compare = getattr(verifiers, "_stdio_outputs_equal")
    assert compare("4.0\n  5  ", "4\n5")
    assert not compare("4.01", "4")
    assert not compare("4", "4 5")


def test_livecodebench_functional_json_numeric_equality() -> None:
    # testing_util.py:229-307: json.loads values use native Python equality.
    assert verifiers._functional_outputs_equal("4.0", "4")
    assert verifiers._functional_outputs_equal("[4.0, 5]", "[4, 5]")
    assert not verifiers._functional_outputs_equal('"4.0"', "4")


def test_livecodebench_tuple_normalization_is_top_level_only() -> None:
    # testing_util.py:263-266: only the prediction's top-level tuple is listed.
    assert verifiers._normalize_json_value((1, (2,))) == [1, (2,)]


def test_livecodebench_stdin_and_functional_paths_share_sandbox(monkeypatch) -> None:
    calls = []

    class FakeProcess:
        pid = 123
        returncode = 0

        def __init__(self, output: bytes):
            self.stdin = io.BytesIO()
            self.stdout = io.BytesIO(output)
            self.stderr = io.BytesIO()

        def wait(self, **kwargs):
            return self.returncode

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        output = b"5" if "functional_driver.py" in command[2] else b"3\n"
        return FakeProcess(output)

    monkeypatch.setattr(verifiers.subprocess, "Popen", fake_popen)
    assert verifiers.verify(
        "livecodebench",
        "def add(a, b):\n    return a + b",
        {"cases": [{"input": "1 2", "output": "3", "testtype": "stdin"}]},
    )
    assert verifiers.verify(
        "livecodebench",
        "def add(a, b):\n    return a + b",
        {
            "func_name": "add",
            "cases": [{"input": "1\n2", "output": "5", "testtype": "functional"}],
        },
    )
    assert len(calls) == 2
    assert all(call[1]["start_new_session"] for call in calls)
    assert all(call[1]["preexec_fn"] is not None for call in calls)


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

def test_nltk_legacy_punkt_without_punkt_tab_raises_clear_error(monkeypatch) -> None:
    class FakeData:
        def __init__(self):
            self.download_calls = 0

        def find(self, resource):
            if resource == "tokenizers/punkt_tab":
                raise LookupError(resource)
            raise LookupError(resource)

        def load(self, _resource):
            raise LookupError("punkt_tab redirect target missing")

    data = FakeData()
    fake_nltk = type(
        "FakeNltk",
        (),
        {
            "data": data,
            "download": staticmethod(
                lambda resource, quiet: setattr(data, "download_calls", data.download_calls + 1) or False
            ),
        },
    )
    monkeypatch.setattr(verifiers, "_punkt_provision_attempted", False)
    monkeypatch.setattr(verifiers.importlib, "import_module", lambda name: fake_nltk)
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["length_constraints:number_sentences"],
            "kwargs": [{"num_sentences": 1, "relation": "at least"}],
        },
    )
    with pytest.raises(RuntimeError, match="punkt_tab"):
        verifiers.verify("ifeval", "One sentence.", reference)
    assert data.download_calls == 1


def test_nltk_tokenizer_load_failure_raises_clear_error(monkeypatch) -> None:
    class FakeData:
        def find(self, resource):
            return object()

        def load(self, _resource):
            raise ValueError("corrupt tokenizer resource")

    fake_nltk = type("FakeNltk", (), {"data": FakeData(), "download": staticmethod(lambda r, q: True)})
    monkeypatch.setattr(verifiers, "_punkt_provision_attempted", False)
    monkeypatch.setattr(verifiers.importlib, "import_module", lambda name: fake_nltk)
    reference = cast(
        dict[str, object],
        {
            "instruction_id_list": ["length_constraints:number_sentences"],
            "kwargs": [{"num_sentences": 1, "relation": "at least"}],
        },
    )
    with pytest.raises(RuntimeError, match="failed to load"):
        verifiers.verify("ifeval", "One sentence.", reference)
