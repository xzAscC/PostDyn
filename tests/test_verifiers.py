from __future__ import annotations

import inspect
import subprocess

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


def test_livecodebench_runs_isolated_subprocess_with_timeout(monkeypatch) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="3\n", stderr="")

    monkeypatch.setattr(verifiers.subprocess, "run", fake_run)
    reference = {"code": "print(a+b)", "cases": [{"stdin": "1 2", "stdout": "3"}]}
    assert verifiers.verify("livecodebench", "print(a+b)", reference)
    assert calls and calls[0][1]["timeout"] > 0
    assert "exec(" not in inspect.getsource(verifiers.verify)


def test_livecodebench_timeout_is_false(monkeypatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 1)

    monkeypatch.setattr(
        verifiers.subprocess,
        "run",
        timeout,
    )
    assert not verifiers.verify(
        "livecodebench", "print(1)", {"cases": [{"stdin": "", "stdout": "1"}]}
    )


def test_ifeval_delegates_prompt_and_response_to_official_checker(monkeypatch) -> None:
    calls = []

    def checker(prompt, response):
        calls.append((prompt, response))
        return response == "no commas"

    monkeypatch.setattr(verifiers, "check_prompt_level", checker)
    reference = {"instruction": "no commas"}
    assert verifiers.verify("ifeval", "no commas", reference)
    assert calls == [("no commas", "no commas")]
    assert not verifiers.verify("ifeval", "has, comma", reference)


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
