from __future__ import annotations

import inspect
import io
import os
import sys
import subprocess
from typing import cast

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
        {"code": "print(a+b)", "cases": [{"stdin": "1 2", "stdout": "3"}]},
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
        "livecodebench", "print(1)", {"cases": [{"stdin": "", "stdout": "1"}]}
    )


def test_livecodebench_timeout_kills_process_group() -> None:
    code = (
        "import subprocess; subprocess.Popen([%r, '-c', 'import time; time.sleep(30)'])"
        % sys.executable
    )
    assert not verifiers.verify(
        "livecodebench", code, {"cases": [{"stdin": "", "stdout": ""}]}
    )


def test_livecodebench_memory_limit_returns_false() -> None:
    code = "x = bytearray(4 * 1024 * 1024 * 1024)"
    assert not verifiers.verify(
        "livecodebench", code, {"cases": [{"stdin": "", "stdout": ""}]}
    )


def test_livecodebench_large_output_is_bounded_and_verifies() -> None:
    code = "import sys; print('3'); sys.stderr.write('x' * (10 * 1024 * 1024))"
    assert verifiers.verify(
        "livecodebench",
        code,
        {"cases": [{"stdin": "", "stdout": "3"}]},
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
            "kwargs": [{"num_words": 2}],
        },
    )
    assert verifiers.verify(
        "ifeval",
        "keyword keyword",
        {
            "instruction_id_list": ["keywords:frequency"],
            "kwargs": [{"keyword": "keyword", "frequency": 2}],
        },
    )


def test_ifeval_unknown_namespaced_instruction_is_false(caplog) -> None:
    reference = cast(
        dict[str, object],
        {"instruction_id_list": ["unknown:not_implemented"], "kwargs": [{}]},
    )
    assert not verifiers.verify("ifeval", "anything", reference)
    assert "Unsupported IFEval instruction" in caplog.text


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
