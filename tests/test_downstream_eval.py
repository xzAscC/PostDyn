"""Tests for src/downstream_eval.py.

No real model, no real benchmark code, no host execution. Every test injects
a fake ``CompletionGenerator`` and a fake ``SandboxRunner`` so generation is
canned and the bubblewrap path is never spawned.

Coverage matrix:
  * ``generate_completion`` -- greedy kwargs (do_sample=False, num_beams=1,
    explicit max_new_tokens) and output slicing (only generated tokens
    decoded; raw, no special-token stripping).
  * ``build_mmlu_prompt`` / ``parse_mmlu_letter`` -- prompt shape and
    deterministic first-standalone A-D extraction (incl. embedded-letter
    rejection and empty input).
  * HumanEval scoring seams -- pass / fail / timeout / compile_error via a
    scripted sandbox runner; assembly places the *completion* (never the
    canonical solution) between prompt and official test.
  * Atomic per-item writes -- temp file cleaned up; os.replace semantics.
  * Identity + resume -- valid cache reused (no generation/sandbox work);
    hash-mismatch / model-drift cache invalidated and regenerated.
  * Orchestration -- exact 50 python + 50 cpp + 50 MMLU per checkpoint,
    summary metrics (pass@1, accuracy, counts, errors), schema rejection,
    and idempotent resume across a full checkpoint run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from src import downstream_eval as de
from src.downstream_eval import (
    DEFAULT_MAX_NEW_TOKENS_CODE,
    DEFAULT_MAX_NEW_TOKENS_MMLU,
    DEFAULT_TIMEOUT_SECONDS,
    GENERATION_CONTRACT_VERSION,
    HISTORICAL_DEFAULT_TIMEOUT,
    LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1,
    OUTCOME_GENERATION_ERROR,
    SUMMARY_FILENAME,
    TASK_HUMANEVAL_X,
    TASK_MMLU,
    CheckpointSummary,
    DownstreamHumanevalItem,
    DownstreamIdentity,
    DownstreamMMLUItem,
    HumanevalLangFields,
    HumanEvalResult,
    MMLU_LETTERS,
    MMLUResult,
    ScoringConfig,
    build_mmlu_prompt,
    compare_singleton_and_batch_token_ids,
    evaluate_checkpoint,
    evaluate_humaneval_item,
    evaluate_mmlu_item,
    generate_completion,
    generate_completions_batch,
    identity_matches,
    load_cached_item,
    load_downstream_items,
    parse_mmlu_letter,
    rebuild_checkpoint_summary,
    score_humaneval_completion,
    validate_humaneval_cached_body,
    validate_mmlu_cached_body,
    write_item_atomically,
)
from src.humaneval_x_validator import (
    OUTCOME_COMPILE_ERROR,
    OUTCOME_ERROR,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_TIMEOUT,
    assemble_cpp_program,
    assemble_python_program,
    sha256_hex,
)


# =============================================================================
# Fakes & helpers
# =============================================================================


class _FakeTokenizer:
    """Records decode calls; encode maps one stable id per character.

    ``eos_token_id`` / ``pad_token_id`` are ``None`` so the EOS-truncation
    path in ``generate_completion`` is skipped (existing tests rely on the
    full generated slice reaching decode unchanged).
    """

    def __init__(self) -> None:
        self.decode_calls: list[list[int]] = []
        self.last_skip: bool = False
        self.eos_token_id: int | None = None
        self.pad_token_id: int | None = None

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 1000 for ch in text]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        self.decode_calls.append(list(token_ids))
        self.last_skip = skip_special_tokens
        # Deterministic, content-derived string so MMLU parsing is predictable.
        return "A"


class _RecordingGenerator:
    """Returns ``input_ids + extra`` and records the greedy kwargs per call."""

    def __init__(self, extra_ids: Sequence[int]) -> None:
        self.extra_ids = list(extra_ids)
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> list[int]:
        self.calls.append(
            {
                "input_ids": list(input_ids),
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_beams": num_beams,
            }
        )
        return list(input_ids) + self.extra_ids


class _RaisingGenerator:
    def generate(
        self,
        input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> list[int]:
        raise RuntimeError("model exploded")


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _ScriptedRunner:
    """Returns canned CompletedProcess / raises per call, in order."""

    def __init__(
        self, responses: list[subprocess.CompletedProcess[str] | BaseException]
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], Path, float]] = []

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), scratch_dir, timeout))
        if not self.responses:
            raise AssertionError("ScriptedRunner ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class _CapturingRunner:
    """Reads the assembled program source from scratch and returns pass/fail."""

    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.programs: list[str] = []
        self.call_count: int = 0

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        self.call_count += 1
        for path in sorted(scratch_dir.iterdir()):
            if path.suffix in (".py", ".cpp"):
                self.programs.append(path.read_text(encoding="utf-8"))
        return _completed(self.returncode)


def _he_item(nid: int) -> dict[str, object]:
    return {
        "numeric_id": nid,
        "python": {
            "prompt": f"def f{nid}(x):\n    ",
            "canonical_solution": f"    return {nid}\n",
            "test": f"assert f{nid}(0)=={nid}\n",
        },
        "cpp": {
            "prompt": f"// cpp prompt {nid}\n",
            "canonical_solution": f"int f{nid}(){{return {nid};}}\n",
            "test": f"// cpp test {nid}\n",
        },
    }


def _mmlu_item(i: int, letter: str = "A") -> dict[str, object]:
    assert letter in MMLU_LETTERS
    question = f"Question number {i}?"
    return {
        "subject": f"subj_{i}",
        "question": question,
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": MMLU_LETTERS.index(letter),
        "answer_letter": letter,
        "question_sha256": sha256_hex(question),
    }


def _make_downstream(n_he: int = 50, n_mmlu: int = 50) -> dict[str, object]:
    return {
        "humaneval_x": {
            "task_ids": list(range(n_he)),
            "n_items": n_he,
            "items": [_he_item(i) for i in range(n_he)],
        },
        "mmlu": {
            "n_questions": n_mmlu,
            "items": [_mmlu_item(i, letter=MMLU_LETTERS[i % 4]) for i in range(n_mmlu)],
        },
    }


# =============================================================================
# generate_completion: greedy kwargs + output slicing
# =============================================================================


class TestGenerateCompletion:
    def test_forwards_deterministic_greedy_kwargs(self):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[700, 701, 702])
        generate_completion(tok, gen, "hello", max_new_tokens=7)
        assert len(gen.calls) == 1
        call = gen.calls[0]
        assert call["do_sample"] is False
        assert call["num_beams"] == 1
        assert call["max_new_tokens"] == 7

    def test_decode_receives_only_generated_tokens_not_prompt(self):
        tok = _FakeTokenizer()
        prompt = "prompt-body"
        gen = _RecordingGenerator(extra_ids=[900, 901])
        generate_completion(tok, gen, prompt, max_new_tokens=2)
        prompt_ids = tok.encode(prompt)
        # Exactly one decode call, containing only the extra (generated) ids.
        assert tok.decode_calls == [[900, 901]]
        # And the prompt ids were never decoded.
        for ids in tok.decode_calls:
            assert not any(i in prompt_ids for i in ids)

    def test_preserves_raw_completion_without_special_token_stripping(self):
        # decode must be called with skip_special_tokens=False (raw).
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[5])
        generate_completion(tok, gen, "p", max_new_tokens=1)
        assert tok.last_skip is False

    def test_empty_generation_slice_yields_empty_completion(self):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[])  # model echoed nothing new
        completion = generate_completion(tok, gen, "abc", max_new_tokens=4)
        assert tok.decode_calls == [[]]
        assert completion == "A"  # decode of empty -> the fake's fixed string

    def test_rejects_non_positive_max_new_tokens(self):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        with pytest.raises(ValueError, match="max_new_tokens"):
            generate_completion(tok, gen, "p", max_new_tokens=0)

    def test_input_ids_forwarded_verbatim_to_generator(self):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        generate_completion(tok, gen, "hi", max_new_tokens=3)
        assert gen.calls[0]["input_ids"] == tok.encode("hi")


class TestGenerateCompletionsBatch:
    def test_canary_compares_eos_truncated_token_ids_not_decoded_text(self):
        class Tokenizer:
            eos_token_id: int | None = 9
            pad_token_id: int | None = 0

            def encode(self, text):
                return list(range(1, len(text) + 1))

            def decode(self, token_ids, *, skip_special_tokens=False):
                return "same decoded text"

        class Generator:
            def generate(
                self, input_ids, *, max_new_tokens, do_sample=False, num_beams=1
            ):
                return list(input_ids) + [1, 8, 9, 7]

            def generate_batch(
                self,
                input_ids,
                attention_mask,
                *,
                max_new_tokens,
                do_sample=False,
                num_beams=1,
            ):
                return [list(row) + [1, 7, 9, 8] for row in input_ids]

        assert not compare_singleton_and_batch_token_ids(
            Tokenizer(), Generator(), Generator(), ["a", "long"], max_new_tokens=4
        )

    def test_left_padding_masks_and_per_row_eos_slicing(self):
        class BatchTokenizer:
            eos_token_id: int | None = 99
            pad_token_id: int | None = 0

            def __init__(self) -> None:
                self.decoded: list[list[int]] = []

            def encode(self, text: str) -> list[int]:
                return list(range(1, len(text) + 1))

            def decode(
                self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
            ) -> str:
                self.decoded.append(list(token_ids))
                return str(list(token_ids))

        class BatchGenerator:
            def __init__(self) -> None:
                self.input_ids: list[list[int]] = []
                self.masks: list[list[int]] = []

            def generate_batch(
                self,
                input_ids,
                attention_mask,
                *,
                max_new_tokens,
                do_sample=False,
                num_beams=1,
            ):
                self.input_ids = [list(row) for row in input_ids]
                self.masks = [list(row) for row in attention_mask]
                return [
                    list(input_ids[0]) + [10, 99, 11],
                    list(input_ids[1]) + [20, 21],
                ]

        tokenizer = BatchTokenizer()
        generator = BatchGenerator()
        result = generate_completions_batch(
            tokenizer, generator, ["a", "abcd"], max_new_tokens=8
        )
        assert generator.input_ids == [[0, 0, 0, 1], [1, 2, 3, 4]]
        assert generator.masks == [[0, 0, 0, 1], [1, 1, 1, 1]]
        assert tokenizer.decoded == [[10], [20, 21]]
        assert result == ["[10]", "[20, 21]"]

    def test_requires_pad_token(self):
        class NoopBatchGenerator:
            def generate_batch(
                self,
                input_ids,
                attention_mask,
                *,
                max_new_tokens,
                do_sample=False,
                num_beams=1,
            ):
                return []

        with pytest.raises(ValueError, match="pad_token_id"):
            generate_completions_batch(
                _FakeTokenizer(),
                NoopBatchGenerator(),
                ["a", "bb"],
                max_new_tokens=2,
            )


# =============================================================================
# MMLU prompt + parser
# =============================================================================


class TestBuildMmluPrompt:
    def test_emits_question_choices_then_answer(self):
        prompt = build_mmlu_prompt("What is 2+2?", ["1", "3", "4", "5"])
        assert prompt.startswith("What is 2+2?\n\n")
        assert "A. 1\nB. 3\nC. 4\nD. 5" in prompt
        assert prompt.endswith("Answer:")

    def test_is_deterministic(self):
        a = build_mmlu_prompt("Q", ["a", "b", "c", "d"])
        b = build_mmlu_prompt("Q", ["a", "b", "c", "d"])
        assert a == b
        assert sha256_hex(a) == sha256_hex(b)

    def test_rejects_wrong_choice_count(self):
        with pytest.raises(ValueError, match="4 choices"):
            build_mmlu_prompt("Q", ["a", "b"])


class TestParseMmluLetter:
    @pytest.mark.parametrize("text", ["A", "B", "C", "D"])
    def test_single_letter(self, text):
        assert parse_mmlu_letter(text) == text

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("B.", "B"),
            ("B)", "B"),
            ("B:", "B"),
            ("B,", "B"),
            ("The answer is C.", "C"),
            ("answer: D", "D"),
            ("\nA\n", "A"),
        ],
    )
    def test_letter_with_punctuation_or_preamble(self, text, expected):
        assert parse_mmlu_letter(text) == expected

    def test_returns_first_standalone_letter(self):
        # "A" appears inside "CAT" (not standalone) and as a standalone later.
        assert parse_mmlu_letter("CAT is wrong.\nThe answer is B.") == "B"

    def test_rejects_letters_embedded_in_words(self):
        # No standalone A-D anywhere.
        assert parse_mmlu_letter("ABCDEF") == ""
        assert parse_mmlu_letter("the value is xyz") == ""

    def test_empty_or_blank_returns_empty(self):
        assert parse_mmlu_letter("") == ""
        assert parse_mmlu_letter("   \n  ") == ""

    def test_lowercase_is_not_matched(self):
        # Only uppercase A-D count as answer letters.
        assert parse_mmlu_letter("the answer is b") == ""


# =============================================================================
# HumanEval scoring seams (pass / fail / timeout / compile_error)
# =============================================================================


class TestScoreHumanevalCompletion:
    def test_python_pass_assembles_completion_between_prompt_and_test(self):
        runner = _CapturingRunner(returncode=0)
        prompt, completion, test = "def f():\n    ", "    return 1\n", "assert f()==1\n"
        program, outcome = score_humaneval_completion(
            prompt=prompt,
            completion=completion,
            test=test,
            language="python",
            task_id=3,
            runner=runner,
            timeout=1.0,
        )
        assert outcome.status == OUTCOME_PASS
        # The assembled source equals the official helper output with the
        # *completion* in the solution slot.
        assert program == assemble_python_program(prompt, completion, test)

    def test_cpp_compile_error_does_not_run_binary(self):
        runner = _ScriptedRunner([_completed(2, stderr="error: stray ';'")])
        _, outcome = score_humaneval_completion(
            prompt="// p\n",
            completion="garbage;\n",
            test="// t\n",
            language="cpp",
            task_id=5,
            runner=runner,
            timeout=1.0,
        )
        assert outcome.status == OUTCOME_COMPILE_ERROR
        assert len(runner.calls) == 1  # binary never executed

    def test_python_fail_on_nonzero_exit(self):
        runner = _ScriptedRunner([_completed(1, stderr="AssertionError")])
        _, outcome = score_humaneval_completion(
            prompt="p",
            completion="bad",
            test="t",
            language="python",
            task_id=7,
            runner=runner,
            timeout=1.0,
        )
        assert outcome.status == OUTCOME_FAIL

    def test_python_timeout(self):
        runner = _ScriptedRunner([subprocess.TimeoutExpired(cmd=["py"], timeout=1.0)])
        _, outcome = score_humaneval_completion(
            prompt="p",
            completion="while True: pass",
            test="t",
            language="python",
            task_id=9,
            runner=runner,
            timeout=1.0,
        )
        assert outcome.status == OUTCOME_TIMEOUT

    def test_python_oserror_becomes_error_outcome(self):
        runner = _ScriptedRunner([OSError("bwrap missing")])
        _, outcome = score_humaneval_completion(
            prompt="p",
            completion="x",
            test="t",
            language="python",
            task_id=11,
            runner=runner,
            timeout=1.0,
        )
        assert outcome.status == OUTCOME_ERROR

    def test_rejects_unsupported_language(self):
        with pytest.raises(ValueError, match="unsupported HumanEval-X language"):
            score_humaneval_completion(
                prompt="p",
                completion="c",
                test="t",
                language="rust",
                task_id=1,
                runner=_CapturingRunner(),
                timeout=1.0,
            )


# =============================================================================
# evaluate_humaneval_item: no canonical substitution + atomic write
# =============================================================================


def _humaneval_item(nid: int = 1) -> DownstreamHumanevalItem:
    return DownstreamHumanevalItem(
        numeric_id=nid,
        python=HumanevalLangFields(
            prompt=f"def f{nid}():\n    ",
            canonical_solution=f"    return {nid}\n",
            test=f"assert f{nid}()=={nid}\n",
        ),
        cpp=HumanevalLangFields(
            prompt=f"// cpp {nid}\n",
            canonical_solution=f"int f{nid}(){{return {nid};}}\n",
            test=f"// test {nid}\n",
        ),
    )


class TestEvaluateHumanevalItem:
    def test_assembles_model_completion_not_canonical(self, tmp_path):
        # A distinct model completion that differs from the canonical solution.
        completion_ids = [777]
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=completion_ids)
        runner = _CapturingRunner(returncode=0)

        result = evaluate_humaneval_item(
            _humaneval_item(1),
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        # Completion is the raw decoded model output, preserved verbatim.
        assert result.completion == "A"
        # The assembled program used the completion, NOT the canonical solution.
        expected = assemble_python_program(
            result.prompt, result.completion, "assert f1()==1\n"
        )
        assert result.assembled_sha256 == sha256_hex(expected)
        # And it must NOT equal the canonical assembly.
        canonical = assemble_python_program(
            result.prompt, "    return 1\n", "assert f1()==1\n"
        )
        assert result.assembled_sha256 != sha256_hex(canonical)
        # The runner observed exactly that assembled source on disk.
        assert runner.programs == [expected]

    def test_writes_atomic_per_item_json(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_humaneval_item(
            _humaneval_item(2),
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        path = tmp_path / "humaneval_x_python_2.json"
        assert path.exists()
        # No leftover temp file.
        assert not list(tmp_path.glob(".humaneval_x_python_2.json.*.tmp"))

    def test_cpp_item_uses_cpp_assembly_and_compile_run(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _ScriptedRunner([_completed(0), _completed(0)])  # compile + run
        result = evaluate_humaneval_item(
            _humaneval_item(3),
            "cpp",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        assert result.outcome == OUTCOME_PASS
        assert result.language == "cpp"
        assert len(runner.calls) == 2  # compile then run


# =============================================================================
# Identity + resume + hash-mismatch invalidation
# =============================================================================


class TestIdentityAndResume:
    def test_identity_matches_equal_dicts(self):
        ident = DownstreamIdentity(
            "m", "r", TASK_HUMANEVAL_X, "python", "abc", 512, 10.0
        )
        assert identity_matches({"identity": ident.to_dict()}, ident) is True

    def test_identity_mismatch_on_prompt_hash(self):
        ident = DownstreamIdentity(
            "m", "r", TASK_HUMANEVAL_X, "python", "abc", 512, 10.0
        )
        drifted = DownstreamIdentity(
            "m", "r", TASK_HUMANEVAL_X, "python", "zzz", 512, 10.0
        )
        assert identity_matches({"identity": drifted.to_dict()}, ident) is False

    def test_identity_mismatch_on_model(self):
        ident = DownstreamIdentity(
            "m1", "r", TASK_HUMANEVAL_X, "python", "x", 512, 10.0
        )
        other = DownstreamIdentity(
            "m2", "r", TASK_HUMANEVAL_X, "python", "x", 512, 10.0
        )
        assert identity_matches({"identity": other.to_dict()}, ident) is False

    def test_resume_reuses_cache_without_generation_or_sandbox(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(4)
        first = evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        n_gen_before = len(gen.calls)
        n_run_before = len(runner.programs)

        second = evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        # Cache hit: no new generation or sandbox work.
        assert len(gen.calls) == n_gen_before
        assert len(runner.programs) == n_run_before
        assert second.completion == first.completion
        assert second.assembled_sha256 == first.assembled_sha256

    def test_hash_mismatch_invalidates_and_regenerates(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(5)
        path = tmp_path / "humaneval_x_python_5.json"

        # First run produces a valid file.
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        assert path.exists()
        n_gen_before = len(gen.calls)

        # Corrupt the prompt hash so the cache is invalidated.
        cached = json.loads(path.read_text(encoding="utf-8"))
        cached["identity"]["prompt_sha256"] = "0" * 64
        path.write_text(json.dumps(cached), encoding="utf-8")

        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        # Regeneration happened.
        assert len(gen.calls) == n_gen_before + 1
        # File was rewritten with the correct identity.
        refreshed = json.loads(path.read_text(encoding="utf-8"))
        assert refreshed["identity"]["prompt_sha256"] == sha256_hex(item.python.prompt)

    def test_force_regenerates_even_when_cache_valid(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(6)
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        n_gen_before = len(gen.calls)
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            force=True,
        )
        assert len(gen.calls) == n_gen_before + 1

    def test_corrupt_cache_file_is_regenerated(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(7)
        path = tmp_path / "humaneval_x_python_7.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json", encoding="utf-8")

        result = evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        assert result.outcome == OUTCOME_PASS
        # File rewritten cleanly.
        json.loads(path.read_text(encoding="utf-8"))


# =============================================================================
# MMLU item evaluation
# =============================================================================


def _mmlu_typed(i: int, letter: str = "A") -> DownstreamMMLUItem:
    question = f"Question {i}?"
    return DownstreamMMLUItem(
        index=i,
        subject=f"subj_{i}",
        question=question,
        choices=("alpha", "beta", "gamma", "delta"),
        answer=MMLU_LETTERS.index(letter),
        answer_letter=letter,
        question_sha256=sha256_hex(question),
    )


class TestEvaluateMmluItem:
    def test_correct_when_parsed_letter_matches_gold(self, tmp_path):
        tok = _FakeTokenizer()  # decode -> "A"
        gen = _RecordingGenerator(extra_ids=[1])
        result = evaluate_mmlu_item(
            _mmlu_typed(0, letter="A"),
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            output_dir=tmp_path,
        )
        assert result.predicted_letter == "A"
        assert result.correct_letter == "A"
        assert result.is_correct is True

    def test_incorrect_when_parsed_letter_differs(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        result = evaluate_mmlu_item(
            _mmlu_typed(1, letter="C"),  # gold C, model says A
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            output_dir=tmp_path,
        )
        assert result.predicted_letter == "A"
        assert result.is_correct is False

    def test_writes_atomic_per_item_json(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        evaluate_mmlu_item(
            _mmlu_typed(2),
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            output_dir=tmp_path,
        )
        assert (tmp_path / "mmlu_2.json").exists()

    def test_resume_reuses_mmlu_cache(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        item = _mmlu_typed(3)
        evaluate_mmlu_item(
            item,
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            output_dir=tmp_path,
        )
        n_before = len(gen.calls)
        evaluate_mmlu_item(
            item,
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            output_dir=tmp_path,
        )
        assert len(gen.calls) == n_before


# =============================================================================
# Atomic write + cached-item loader
# =============================================================================


class TestAtomicWrite:
    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "item.json"
        write_item_atomically(path, {"k": "v"})
        assert json.loads(path.read_text(encoding="utf-8")) == {"k": "v"}

    def test_no_temp_file_left_after_success(self, tmp_path):
        path = tmp_path / "item.json"
        write_item_atomically(path, {"k": "v"})
        assert not list(tmp_path.glob(".item.json.*.tmp"))

    def test_load_cached_returns_none_when_missing(self, tmp_path):
        ident = DownstreamIdentity("m", "r", TASK_MMLU, TASK_MMLU, "x", 32, 10.0)
        assert load_cached_item(tmp_path / "absent.json", ident) is None

    def test_load_cached_returns_none_on_identity_mismatch(self, tmp_path):
        path = tmp_path / "item.json"
        ident = DownstreamIdentity("m", "r", TASK_MMLU, TASK_MMLU, "x", 32, 10.0)
        other = DownstreamIdentity("m", "r", TASK_MMLU, TASK_MMLU, "y", 32, 10.0)
        write_item_atomically(path, {"identity": other.to_dict()})
        assert load_cached_item(path, ident) is None


# =============================================================================
# Typed downstream loaders + schema validation
# =============================================================================


class TestLoadDownstreamItems:
    def test_parses_typed_lists_for_valid_manifest(self):
        downstream = _make_downstream(2, 2)
        humaneval, mmlu = load_downstream_items(
            downstream, expected_humaneval=2, expected_mmlu=2
        )
        assert len(humaneval) == 2
        assert len(mmlu) == 2
        assert isinstance(humaneval[0], DownstreamHumanevalItem)
        assert isinstance(mmlu[0], DownstreamMMLUItem)
        assert humaneval[0].python.prompt == "def f0(x):\n    "
        assert mmlu[0].choices == ("alpha", "beta", "gamma", "delta")

    def test_rejects_wrong_humaneval_count(self):
        downstream = _make_downstream(3, 2)
        with pytest.raises(Exception):
            load_downstream_items(downstream, expected_humaneval=2, expected_mmlu=2)

    def test_rejects_wrong_mmlu_count(self):
        downstream = _make_downstream(2, 3)
        with pytest.raises(Exception):
            load_downstream_items(downstream, expected_humaneval=2, expected_mmlu=2)

    def test_rejects_missing_humaneval_block(self):
        with pytest.raises(Exception):
            load_downstream_items(
                {"humaneval_x": None, "mmlu": {"n_questions": 0, "items": []}}
            )

    def test_rejects_non_string_prompt(self):
        bad: dict[str, object] = {
            "humaneval_x": {
                "task_ids": [0],
                "n_items": 1,
                "items": [
                    {
                        "numeric_id": 0,
                        "python": {
                            "prompt": 123,
                            "canonical_solution": "s\n",
                            "test": "t\n",
                        },
                        "cpp": {
                            "prompt": "p\n",
                            "canonical_solution": "s\n",
                            "test": "t\n",
                        },
                    }
                ],
            },
            "mmlu": {"n_questions": 1, "items": [_mmlu_item(0)]},
        }
        with pytest.raises(ValueError, match="must be a string"):
            load_downstream_items(bad, expected_humaneval=1, expected_mmlu=1)

    def test_rejects_bad_mmlu_choices(self):
        bad: dict[str, object] = {
            "humaneval_x": {
                "task_ids": [0],
                "n_items": 1,
                "items": [_he_item(0)],
            },
            "mmlu": {
                "n_questions": 1,
                "items": [
                    {
                        "subject": "s",
                        "question": "q?",
                        "choices": ["only", "two"],
                        "answer": 0,
                        "answer_letter": "A",
                        "question_sha256": "x" * 64,
                    }
                ],
            },
        }
        with pytest.raises(ValueError, match="choices"):
            load_downstream_items(bad, expected_humaneval=1, expected_mmlu=1)


# =============================================================================
# Orchestration: exact 50 + 50 + 50 per checkpoint
# =============================================================================


class TestEvaluateCheckpoint:
    def test_exact_fifty_fifty_fifty_counts_and_metrics(self, tmp_path):
        downstream = _make_downstream(50, 50)
        tok = _FakeTokenizer()  # decode -> "A"
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)  # all programs pass

        summary = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        # Exact item counts.
        assert summary.n_humaneval_python == 50
        assert summary.n_humaneval_cpp == 50
        assert summary.n_mmlu == 50
        # All python/cpp pass at 100%.
        assert summary.python_pass_at_1 == 1.0
        assert summary.cpp_pass_at_1 == 1.0
        # MMLU: every completion decodes to "A"; gold rotates A-D.
        # range(50) -> i%4==0 gives 13 "A" items.
        assert summary.mmlu_parsed == 50
        assert summary.mmlu_correct == 13
        assert summary.mmlu_accuracy == pytest.approx(13 / 50)
        # Outcome counts.
        assert summary.python_counts[OUTCOME_PASS] == 50
        assert summary.cpp_counts[OUTCOME_PASS] == 50
        assert summary.errors == 0

    def test_writes_one_json_per_item_plus_summary(self, tmp_path):
        downstream = _make_downstream(50, 50)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        py = list(tmp_path.glob("humaneval_x_python_*.json"))
        cpp = list(tmp_path.glob("humaneval_x_cpp_*.json"))
        mmlu = list(tmp_path.glob("mmlu_*.json"))
        assert len(py) == 50
        assert len(cpp) == 50
        assert len(mmlu) == 50
        assert (tmp_path / SUMMARY_FILENAME).exists()

    def test_summary_round_trips_through_json(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        summary = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        loaded = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
        assert loaded["model"] == "m"
        assert loaded["n_humaneval_python"] == 2
        assert loaded["python_pass_at_1"] == summary.python_pass_at_1

    def test_idempotent_resume_skips_all_work(self, tmp_path):
        downstream = _make_downstream(3, 3)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=3,
            expected_mmlu=3,
        )
        gen_calls_after_first = len(gen.calls)
        run_calls_after_first = len(runner.programs)

        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=3,
            expected_mmlu=3,
        )
        # No new generation or sandbox work on the second pass.
        assert len(gen.calls) == gen_calls_after_first
        assert len(runner.programs) == run_calls_after_first

    def test_mmlu_does_not_invoke_sandbox_runner(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        # 2 python (1 call each) + 2 cpp (compile + run each) = 6 sandbox
        # calls; MMLU contributes none.
        assert runner.call_count == 6

    def test_rejects_manifest_with_wrong_counts(self, tmp_path):
        downstream = _make_downstream(10, 50)  # only 10 humaneval
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        with pytest.raises(Exception):
            evaluate_checkpoint(
                model="m",
                revision="r",
                downstream=downstream,
                tokenizer=tok,
                generator=gen,
                runner=runner,
                output_dir=tmp_path,
                timeout=1.0,
            )
        # Nothing written when the schema is rejected up front.
        assert not (tmp_path / SUMMARY_FILENAME).exists()

    def test_generation_error_isolated_and_counted(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RaisingGenerator()  # every generation raises
        runner = _CapturingRunner(returncode=0)
        summary = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        # 2 python + 2 cpp generation errors, 2 mmlu generation errors.
        assert summary.errors == 6
        assert summary.python_counts[OUTCOME_GENERATION_ERROR] == 2
        assert summary.cpp_counts[OUTCOME_GENERATION_ERROR] == 2
        # No passes.
        assert summary.python_pass_at_1 == 0.0
        assert summary.mmlu_correct == 0

    def test_force_regenerates_everything(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        first = len(gen.calls)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=2,
            expected_mmlu=2,
            force=True,
        )
        # 2 py + 2 cpp + 2 mmlu = 6 generations regenerated.
        assert len(gen.calls) == first + 6


# =============================================================================
# Orchestration: mixed HumanEval outcomes feed counts correctly
# =============================================================================


class TestMixedOutcomes:
    def test_python_fail_and_cpp_compile_error_reflected_in_counts(self, tmp_path):
        downstream = _make_downstream(2, 0)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])

        class _LanguageRunner:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def run_in_sandbox(self, command, scratch_dir, timeout, **_kwargs):
                cmd = list(command)
                self.calls.append(cmd)
                if any(str(c).endswith(".py") for c in cmd):
                    return _completed(1, stderr="AssertionError")
                if "/usr/bin/g++" in cmd:
                    return _completed(2, stderr="compile error")
                return _completed(0)

        runner = _LanguageRunner()

        summary = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
            expected_humaneval=2,
            expected_mmlu=0,
        )
        assert summary.python_counts[OUTCOME_FAIL] == 2
        assert summary.cpp_counts[OUTCOME_COMPILE_ERROR] == 2
        assert summary.python_pass_at_1 == 0.0
        assert summary.cpp_pass_at_1 == 0.0


# =============================================================================
# GreedyGenerator: explicit EOS/pad forwarding to the HF model adapter
# =============================================================================


class TestGreedyGeneratorForwardsEosPad:
    """GreedyGenerator must forward the tokenizer's ``eos_token_id`` and
    ``pad_token_id`` to ``model.generate`` so generation stops at the real
    end-of-text even when ``model.generation_config.eos_token_id`` is ``None``
    (OLMo ships an effectively empty ``generation_config.json``).
    """

    def test_forwards_tokenizer_eos_and_pad_ids_to_model_generate(self):
        import torch

        # Distinctive fixture IDs chosen by the test -- NOT the OLMo magic
        # numbers. The production adapter must read from the tokenizer, never
        # hardcode them.
        eos_id = 998
        pad_id = 999

        class _SpecialTokenTokenizer:
            def __init__(self) -> None:
                self.eos_token_id: int | None = eos_id
                self.pad_token_id: int | None = pad_id

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(
                self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
            ) -> str:
                return ""

        class _RecordingHFModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def generate(
                self,
                input_ids: object,
                *,
                max_new_tokens: int,
                do_sample: bool = False,
                num_beams: int = 1,
                eos_token_id: int | None = None,
                pad_token_id: int | None = None,
            ) -> "torch.Tensor":
                self.calls.append(
                    {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": do_sample,
                        "num_beams": num_beams,
                        "eos_token_id": eos_token_id,
                        "pad_token_id": pad_token_id,
                    }
                )
                return torch.tensor([[10, 20, 30]], dtype=torch.long)

        model = _RecordingHFModel()
        tok = _SpecialTokenTokenizer()
        gen = de.GreedyGenerator(model=model, tokenizer=tok, device="cpu")
        gen.generate([1, 2, 3], max_new_tokens=8)
        assert len(model.calls) == 1
        call = model.calls[0]
        # EOS and pad forwarded verbatim from the tokenizer attributes -- the
        # adapter must not invent or omit them.
        assert call["eos_token_id"] == tok.eos_token_id
        assert call["pad_token_id"] == tok.pad_token_id
        # Greedy kwargs still forwarded unchanged.
        assert call["do_sample"] is False
        assert call["num_beams"] == 1
        assert call["max_new_tokens"] == 8

    def test_forwards_none_pad_safely_without_crashing(self):
        import torch

        class _NoPadTokenizer:
            eos_token_id: int | None = 42
            pad_token_id: int | None = None  # tokenizer has no pad token

            def encode(self, text: str) -> list[int]:
                return [1]

            def decode(
                self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
            ) -> str:
                return ""

        class _RecordingHFModel:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def generate(
                self,
                input_ids: object,
                *,
                max_new_tokens: int,
                do_sample: bool = False,
                num_beams: int = 1,
                eos_token_id: int | None = None,
                pad_token_id: int | None = None,
            ) -> "torch.Tensor":
                self.calls.append(
                    {"eos_token_id": eos_token_id, "pad_token_id": pad_token_id}
                )
                return torch.tensor([[1]], dtype=torch.long)

        model = _RecordingHFModel()
        tok = _NoPadTokenizer()
        gen = de.GreedyGenerator(model=model, tokenizer=tok, device="cpu")
        # Must not raise even though pad_token_id is None.
        gen.generate([1], max_new_tokens=1)
        assert model.calls[0]["pad_token_id"] is None
        assert model.calls[0]["eos_token_id"] == 42


# =============================================================================
# generate_completion: defensive EOS truncation before decode
# =============================================================================


class TestGenerateCompletionEosTruncation:
    """``generate_completion`` must defensively truncate the generated token-id
    slice at the first EOS before decoding, so post-EOS garbage never enters the
    completion. ``skip_special_tokens`` stays ``False`` so other special tokens
    in the payload are preserved verbatim.
    """

    def test_truncates_at_first_eos_excluding_garbage_after(self):
        eos_id = 800

        class _EosAwareTokenizer:
            def __init__(self) -> None:
                self.decode_calls: list[list[int]] = []
                self.last_skip: bool = False
                self.eos_token_id: int | None = eos_id
                self.pad_token_id: int | None = None

            def encode(self, text: str) -> list[int]:
                return [ord(ch) % 100 for ch in text]

            def decode(
                self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
            ) -> str:
                self.decode_calls.append(list(token_ids))
                self.last_skip = skip_special_tokens
                return "decoded(" + ",".join(str(t) for t in token_ids) + ")"

        class _FixedSequenceGenerator:
            def __init__(self, full_sequence: Sequence[int]) -> None:
                self.full_sequence = list(full_sequence)

            def generate(
                self,
                input_ids: Sequence[int],
                *,
                max_new_tokens: int,
                do_sample: bool = False,
                num_beams: int = 1,
            ) -> list[int]:
                return list(self.full_sequence)

        tok = _EosAwareTokenizer()
        prompt = "hi"
        prompt_ids = tok.encode(prompt)
        payload = [100, 101, 102]
        garbage = [200, 201]
        # Model emitted: payload, EOS, garbage, EOS (kept generating past EOS).
        full = list(prompt_ids) + payload + [eos_id] + garbage + [eos_id]
        gen = _FixedSequenceGenerator(full)

        result = generate_completion(tok, gen, prompt, max_new_tokens=10)

        # Only the payload reached decode; EOS and post-EOS garbage excluded.
        assert tok.decode_calls == [payload]
        # skip_special_tokens is still False (other special tokens preserved).
        assert tok.last_skip is False
        # The decoded text reflects only the payload tokens.
        assert result == "decoded(100,101,102)"

    def test_no_truncation_when_eos_token_id_is_none(self):
        class _NoEosTokenizer:
            def __init__(self) -> None:
                self.decode_calls: list[list[int]] = []
                self.eos_token_id: int | None = None
                self.pad_token_id: int | None = None

            def encode(self, text: str) -> list[int]:
                return [ord(ch) % 100 for ch in text]

            def decode(
                self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
            ) -> str:
                self.decode_calls.append(list(token_ids))
                return "x"

        class _FixedSequenceGenerator:
            def __init__(self, full_sequence: Sequence[int]) -> None:
                self.full_sequence = list(full_sequence)

            def generate(
                self,
                input_ids: Sequence[int],
                *,
                max_new_tokens: int,
                do_sample: bool = False,
                num_beams: int = 1,
            ) -> list[int]:
                return list(self.full_sequence)

        tok = _NoEosTokenizer()
        prompt_ids = tok.encode("hi")
        generated = [50, 60, 70]
        gen = _FixedSequenceGenerator(list(prompt_ids) + generated)

        generate_completion(tok, gen, "hi", max_new_tokens=3)

        # No EOS -> no truncation; all generated ids decoded.
        assert tok.decode_calls == [generated]


# =============================================================================
# Generation-contract version: old cached items auto-invalidated
# =============================================================================


class TestGenerationContractVersion:
    """The identity carries a generation-contract version so cached items
    written before the EOS-truncation fix are automatically detected as stale
    and regenerated, without deleting old files."""

    def test_identity_mismatch_when_version_key_missing(self):
        # An old-format identity dict (pre-fix) has no generation_contract_version.
        ident = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=10.0,
        )
        old_dict: dict[str, object] = {
            "model": "m",
            "revision": "r",
            "task": TASK_HUMANEVAL_X,
            "language": "python",
            "prompt_sha256": "abc",
            # No generation_contract_version key.
        }
        assert identity_matches({"identity": old_dict}, ident) is False

    def test_identity_matches_when_version_present_and_equal(self):
        from src.downstream_eval import GENERATION_CONTRACT_VERSION

        ident = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=10.0,
        )
        full_dict: dict[str, object] = dict(ident.to_dict())
        assert full_dict["generation_contract_version"] == GENERATION_CONTRACT_VERSION
        assert identity_matches({"identity": full_dict}, ident) is True

    def test_old_cache_without_version_is_regenerated(self, tmp_path):
        """A pre-fix cached item (identity without generation_contract_version)
        must not match the current identity and must be regenerated."""
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(8)
        path = tmp_path / "humaneval_x_python_8.json"

        # Build an "old format" payload: identical to what the pre-fix code
        # wrote, but the identity dict is missing generation_contract_version.
        old_identity: dict[str, object] = {
            "model": "m",
            "revision": "r",
            "task": TASK_HUMANEVAL_X,
            "language": "python",
            "prompt_sha256": sha256_hex(item.python.prompt),
        }
        old_payload: dict[str, object] = {
            "identity": old_identity,
            "task_id": 8,
            "language": "python",
            "prompt": item.python.prompt,
            "completion": "STALE_PRE_FIX_COMPLETION",
            "completion_sha256": "0" * 64,
            "assembled_sha256": "0" * 64,
            "outcome": OUTCOME_PASS,
            "exit_code": 0,
            "diagnostics": "",
            "max_new_tokens": 512,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(old_payload), encoding="utf-8")

        result = evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=1.0,
        )
        # Stale cache invalidated -> regeneration happened.
        assert len(gen.calls) == 1
        # File rewritten with current identity (including the version key).
        refreshed = json.loads(path.read_text(encoding="utf-8"))
        assert "generation_contract_version" in refreshed["identity"]
        # Stale completion was replaced by the fresh decode.
        assert refreshed["completion"] != "STALE_PRE_FIX_COMPLETION"
        # The returned result reflects the regeneration.
        assert result.completion != "STALE_PRE_FIX_COMPLETION"


# =============================================================================
# eos-truncate-2 identity: max_new_tokens + finite positive timeout
# =============================================================================


class TestEosTruncate2Identity:
    """The eos-truncate-2 identity carries ``max_new_tokens`` and a finite
    positive ``timeout`` so a token-budget or timeout change invalidates the
    cache and forces regeneration, instead of silently reusing stale items."""

    def test_contract_version_is_current(self):
        assert GENERATION_CONTRACT_VERSION == "eos-truncate-3"

    def test_identity_serializes_max_new_tokens_and_timeout(self):
        ident = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=10.0,
        )
        d = ident.to_dict()
        assert d["max_new_tokens"] == 512
        assert d["timeout"] == 10.0
        assert d["generation_contract_version"] == GENERATION_CONTRACT_VERSION

    def test_identity_mismatch_when_max_new_tokens_differs(self):
        ident = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=10.0,
        )
        other = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=256,
            timeout=10.0,
        )
        assert identity_matches({"identity": other.to_dict()}, ident) is False

    def test_identity_mismatch_when_timeout_differs(self):
        ident = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=10.0,
        )
        other = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=5.0,
        )
        assert identity_matches({"identity": other.to_dict()}, ident) is False

    def test_identity_matches_when_all_fields_equal(self):
        ident = DownstreamIdentity(
            model="m",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="abc",
            max_new_tokens=512,
            timeout=10.0,
        )
        assert identity_matches({"identity": ident.to_dict()}, ident) is True

    def test_newly_written_item_carries_budget_timeout_and_version(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_humaneval_item(
            _humaneval_item(11),
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=10.0,
        )
        raw = json.loads(
            (tmp_path / "humaneval_x_python_11.json").read_text(encoding="utf-8")
        )
        assert (
            raw["identity"]["generation_contract_version"]
            == GENERATION_CONTRACT_VERSION
        )
        assert raw["identity"]["max_new_tokens"] == 512
        assert raw["identity"]["timeout"] == 10.0


# =============================================================================
# Legacy eos-truncate-1 adoption (narrow, default-only migration)
# =============================================================================


def _legacy_humaneval_payload(
    item: DownstreamHumanevalItem,
    *,
    body_mnt: int | None = 512,
    completion: str = "LEGACY_COMPLETION",
    prompt_sha256: str | None = None,
    identity_version: str = LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1,
) -> dict[str, object]:
    """Build an eos-truncate-1 (or pre-version) cached item payload.

    Mirrors what the pre-eos-truncate-2 code wrote: the identity dict omits
    ``max_new_tokens`` / ``timeout`` and carries the legacy contract version;
    the body carries ``max_new_tokens``. Body hashes are recomputed so the
    full body-validation path on resume can accept the fixture.
    """
    if prompt_sha256 is None:
        prompt_sha256 = sha256_hex(item.python.prompt)
    program = assemble_python_program(item.python.prompt, completion, item.python.test)
    payload: dict[str, object] = {
        "identity": {
            "model": "m",
            "revision": "r",
            "task": TASK_HUMANEVAL_X,
            "language": "python",
            "prompt_sha256": prompt_sha256,
            "generation_contract_version": identity_version,
        },
        "task_id": item.numeric_id,
        "language": "python",
        "prompt": item.python.prompt,
        "completion": completion,
        "completion_sha256": sha256_hex(completion),
        "assembled_sha256": sha256_hex(program),
        "outcome": OUTCOME_PASS,
        "exit_code": 0,
        "diagnostics": "",
    }
    if body_mnt is not None:
        payload["max_new_tokens"] = body_mnt
    return payload


class TestLegacyEosTruncate1Adoption:
    """Existing EOS-corrected eos-truncate-1 files remain reusable at default
    settings through a narrowly validated migration. Acceptance requires all
    of: core identity match, body max_new_tokens == expected, expected timeout
    == historical default 10.0. Anything else is rejected (regenerated)."""

    def test_adopted_when_body_mnt_matches_and_timeout_is_default(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(12)
        path = tmp_path / "humaneval_x_python_12.json"
        path.write_text(
            json.dumps(_legacy_humaneval_payload(item, body_mnt=512)),
            encoding="utf-8",
        )
        result = evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        # Cache adopted: no generation, no sandbox work.
        assert gen.calls == []
        assert runner.programs == []
        # Legacy completion preserved verbatim (no regeneration).
        assert result.completion == "LEGACY_COMPLETION"

    def test_rejected_when_timeout_is_not_default(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(13)
        path = tmp_path / "humaneval_x_python_13.json"
        path.write_text(
            json.dumps(_legacy_humaneval_payload(item, body_mnt=512)),
            encoding="utf-8",
        )
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=5.0,  # not the historical default -> reject legacy
        )
        # Legacy rejected -> regenerated.
        assert len(gen.calls) == 1
        refreshed = json.loads(path.read_text(encoding="utf-8"))
        assert (
            refreshed["identity"]["generation_contract_version"]
            == GENERATION_CONTRACT_VERSION
        )
        assert refreshed["identity"]["timeout"] == 5.0

    def test_rejected_when_body_mnt_mismatches_expected(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(14)
        path = tmp_path / "humaneval_x_python_14.json"
        # Body says 256 but the run expects 512.
        path.write_text(
            json.dumps(_legacy_humaneval_payload(item, body_mnt=256)),
            encoding="utf-8",
        )
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        assert len(gen.calls) == 1  # rejected -> regenerated

    def test_rejected_when_body_mnt_missing(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(15)
        path = tmp_path / "humaneval_x_python_15.json"
        path.write_text(
            json.dumps(_legacy_humaneval_payload(item, body_mnt=None)),
            encoding="utf-8",
        )
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        assert len(gen.calls) == 1  # malformed -> rejected

    def test_rejected_when_body_mnt_malformed(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(16)
        path = tmp_path / "humaneval_x_python_16.json"
        payload = _legacy_humaneval_payload(item, body_mnt=512)
        payload["max_new_tokens"] = "lots"  # not an int
        path.write_text(json.dumps(payload), encoding="utf-8")
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        assert len(gen.calls) == 1  # malformed -> rejected

    def test_rejected_when_body_mnt_is_bool(self, tmp_path):
        # bool is a subclass of int; the validator must refuse it.
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(17)
        path = tmp_path / "humaneval_x_python_17.json"
        payload = _legacy_humaneval_payload(item, body_mnt=512)
        payload["max_new_tokens"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        assert len(gen.calls) == 1  # bool is not a valid budget -> rejected

    def test_rejected_when_core_identity_drifts(self, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(18)
        path = tmp_path / "humaneval_x_python_18.json"
        path.write_text(
            json.dumps(
                _legacy_humaneval_payload(item, prompt_sha256="0" * 64, body_mnt=512)
            ),
            encoding="utf-8",
        )
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        assert len(gen.calls) == 1  # core drift -> rejected

    def test_pre_version_files_still_rejected(self, tmp_path):
        # Files written before the version key existed (no
        # generation_contract_version) must never be adopted.
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        item = _humaneval_item(19)
        path = tmp_path / "humaneval_x_python_19.json"
        payload = _legacy_humaneval_payload(item, body_mnt=512)
        legacy_identity = payload["identity"]
        assert isinstance(legacy_identity, dict)
        del legacy_identity["generation_contract_version"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        evaluate_humaneval_item(
            item,
            "python",
            model="m",
            revision="r",
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens=512,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
        )
        assert len(gen.calls) == 1  # legacy sentinel -> rejected


# =============================================================================
# Timeout validation (NaN / inf / nonpositive rejected in the library)
# =============================================================================


class TestTimeoutValidation:
    @pytest.mark.parametrize(
        "bad", [0, -1, -0.5, float("nan"), float("inf"), float("-inf")]
    )
    def test_evaluate_humaneval_item_rejects_bad_timeout(self, bad, tmp_path):
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        with pytest.raises(ValueError, match="timeout"):
            evaluate_humaneval_item(
                _humaneval_item(20),
                "python",
                model="m",
                revision="r",
                tokenizer=tok,
                generator=gen,
                runner=runner,
                output_dir=tmp_path,
                max_new_tokens=512,
                timeout=bad,
            )

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
    def test_score_humaneval_completion_rejects_bad_timeout(self, bad):
        with pytest.raises(ValueError, match="timeout"):
            score_humaneval_completion(
                prompt="p",
                completion="c",
                test="t",
                language="python",
                task_id=1,
                runner=_CapturingRunner(),
                timeout=bad,
            )

    def test_evaluate_checkpoint_rejects_nan_timeout(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        with pytest.raises(ValueError, match="timeout"):
            evaluate_checkpoint(
                model="m",
                revision="r",
                downstream=downstream,
                tokenizer=tok,
                generator=gen,
                runner=runner,
                output_dir=tmp_path,
                timeout=float("nan"),
                expected_humaneval=1,
                expected_mmlu=1,
            )

    def test_evaluate_checkpoint_rejects_inf_timeout(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        with pytest.raises(ValueError, match="timeout"):
            evaluate_checkpoint(
                model="m",
                revision="r",
                downstream=downstream,
                tokenizer=tok,
                generator=gen,
                runner=runner,
                output_dir=tmp_path,
                timeout=float("inf"),
                expected_humaneval=1,
                expected_mmlu=1,
            )


# =============================================================================
# ScoringConfig: typed scoring config carried by every checkpoint summary
# =============================================================================


class TestScoringConfig:
    def test_round_trips_through_json(self):
        cfg = ScoringConfig(
            max_new_tokens_code=512,
            max_new_tokens_mmlu=32,
            timeout=10.0,
            generation_contract_version=GENERATION_CONTRACT_VERSION,
        )
        d = cfg.to_dict()
        assert d == {
            "max_new_tokens_code": 512,
            "max_new_tokens_mmlu": 32,
            "timeout": 10.0,
            "generation_contract_version": GENERATION_CONTRACT_VERSION,
        }

    def test_summary_carries_scoring_config(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        loaded = json.loads((tmp_path / SUMMARY_FILENAME).read_text(encoding="utf-8"))
        assert loaded["scoring_config"]["max_new_tokens_code"] == 512
        assert loaded["scoring_config"]["max_new_tokens_mmlu"] == 32
        assert loaded["scoring_config"]["timeout"] == 10.0
        assert (
            loaded["scoring_config"]["generation_contract_version"]
            == GENERATION_CONTRACT_VERSION
        )

    def test_summary_scoring_config_reflects_run_params(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        summary = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            max_new_tokens_code=256,
            max_new_tokens_mmlu=16,
            timeout=7.5,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        assert summary.scoring_config.max_new_tokens_code == 256
        assert summary.scoring_config.max_new_tokens_mmlu == 16
        assert summary.scoring_config.timeout == 7.5


# =============================================================================
# No-model summary rebuild from existing per-item JSON
# =============================================================================


class TestRebuildCheckpointSummary:
    """``rebuild_checkpoint_summary`` rebuilds a checkpoint summary from the
    existing per-item JSON files without loading any model or invoking the
    sandbox. It reads exactly the expected files, validates each identity
    (with legacy adoption), recounts outcomes/correctness, and writes the
    summary atomically."""

    def test_rebuilds_from_existing_items_without_generation(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        # First pass: produce items + summary with the model.
        first = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        gen_calls_before = len(gen.calls)
        # Delete the summary so the rebuild cannot just copy it.
        (tmp_path / SUMMARY_FILENAME).unlink()

        rebuilt = rebuild_checkpoint_summary(
            model="m",
            revision="r",
            downstream=downstream,
            output_dir=tmp_path,
            max_new_tokens_code=DEFAULT_MAX_NEW_TOKENS_CODE,
            max_new_tokens_mmlu=DEFAULT_MAX_NEW_TOKENS_MMLU,
            timeout=10.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        # No generation or sandbox work happened during the rebuild.
        assert len(gen.calls) == gen_calls_before
        # Recounted metrics match the original summary.
        assert rebuilt.python_counts == first.python_counts
        assert rebuilt.cpp_counts == first.cpp_counts
        assert rebuilt.mmlu_correct == first.mmlu_correct
        assert rebuilt.mmlu_parsed == first.mmlu_parsed
        assert rebuilt.errors == first.errors
        assert rebuilt.scoring_config == first.scoring_config
        # Summary was written atomically.
        assert (tmp_path / SUMMARY_FILENAME).exists()

    def test_rebuild_rejects_when_an_item_is_missing(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        # Remove one item -> rebuild must refuse (incomplete checkpoint).
        (tmp_path / "humaneval_x_python_0.json").unlink()
        with pytest.raises((ValueError, FileNotFoundError)):
            rebuild_checkpoint_summary(
                model="m",
                revision="r",
                downstream=downstream,
                output_dir=tmp_path,
                max_new_tokens_code=DEFAULT_MAX_NEW_TOKENS_CODE,
                max_new_tokens_mmlu=DEFAULT_MAX_NEW_TOKENS_MMLU,
                timeout=10.0,
                expected_humaneval=2,
                expected_mmlu=2,
            )

    def test_rebuild_rejects_identity_drift(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        # Asking for a different model -> identity drift -> rebuild refuses.
        with pytest.raises((ValueError, FileNotFoundError)):
            rebuild_checkpoint_summary(
                model="other/model",
                revision="r",
                downstream=downstream,
                output_dir=tmp_path,
                max_new_tokens_code=DEFAULT_MAX_NEW_TOKENS_CODE,
                max_new_tokens_mmlu=DEFAULT_MAX_NEW_TOKENS_MMLU,
                timeout=10.0,
                expected_humaneval=1,
                expected_mmlu=1,
            )

    def test_rebuild_recounts_outcomes_from_item_bodies(self, tmp_path):
        downstream = _make_downstream(2, 2)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        first = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        # Mutate one item's outcome on disk so the recounted summary differs.
        path = tmp_path / "humaneval_x_python_0.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["outcome"] = OUTCOME_FAIL
        write_item_atomically(path, raw)
        (tmp_path / SUMMARY_FILENAME).unlink()

        rebuilt = rebuild_checkpoint_summary(
            model="m",
            revision="r",
            downstream=downstream,
            output_dir=tmp_path,
            max_new_tokens_code=DEFAULT_MAX_NEW_TOKENS_CODE,
            max_new_tokens_mmlu=DEFAULT_MAX_NEW_TOKENS_MMLU,
            timeout=10.0,
            expected_humaneval=2,
            expected_mmlu=2,
        )
        # The mutated pass became a fail; pass@1 dropped accordingly.
        assert (
            rebuilt.python_counts[OUTCOME_PASS] == first.python_counts[OUTCOME_PASS] - 1
        )
        assert rebuilt.python_counts[OUTCOME_FAIL] == 1

    def test_rebuild_adopts_legacy_items_at_default_timeout(self, tmp_path):
        downstream = _make_downstream(1, 1)
        # Derive the typed items from the manifest so the legacy files'
        # prompt hashes line up with what the rebuild expects.
        he_items, mmlu_typed_list = load_downstream_items(
            downstream, expected_humaneval=1, expected_mmlu=1
        )
        ds_item = he_items[0]

        def _legacy_item(
            fields: HumanevalLangFields, language: str
        ) -> dict[str, object]:
            if language == "python":
                program = assemble_python_program(fields.prompt, "LEGACY", fields.test)
            else:
                program = assemble_cpp_program(fields.prompt, "LEGACY", fields.test)
            return {
                "identity": {
                    "model": "m",
                    "revision": "r",
                    "task": TASK_HUMANEVAL_X,
                    "language": language,
                    "prompt_sha256": sha256_hex(fields.prompt),
                    "generation_contract_version": LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1,
                },
                "task_id": ds_item.numeric_id,
                "language": language,
                "prompt": fields.prompt,
                "completion": "LEGACY",
                "completion_sha256": sha256_hex("LEGACY"),
                "assembled_sha256": sha256_hex(program),
                "outcome": OUTCOME_PASS,
                "exit_code": 0,
                "diagnostics": "",
                "max_new_tokens": DEFAULT_MAX_NEW_TOKENS_CODE,
            }

        (tmp_path / "humaneval_x_python_0.json").write_text(
            json.dumps(_legacy_item(ds_item.python, "python")), encoding="utf-8"
        )
        (tmp_path / "humaneval_x_cpp_0.json").write_text(
            json.dumps(_legacy_item(ds_item.cpp, "cpp")), encoding="utf-8"
        )
        mmlu_typed = mmlu_typed_list[0]
        mmlu_prompt = build_mmlu_prompt(mmlu_typed.question, mmlu_typed.choices)
        mmlu_payload: dict[str, object] = {
            "identity": {
                "model": "m",
                "revision": "r",
                "task": TASK_MMLU,
                "language": TASK_MMLU,
                "prompt_sha256": sha256_hex(mmlu_prompt),
                "generation_contract_version": LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1,
            },
            "index": 0,
            "subject": mmlu_typed.subject,
            "question_sha256": mmlu_typed.question_sha256,
            "prompt": mmlu_prompt,
            "completion": "A",
            "predicted_letter": "A",
            "correct_letter": mmlu_typed.answer_letter,
            "is_correct": True,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS_MMLU,
        }
        (tmp_path / "mmlu_0.json").write_text(
            json.dumps(mmlu_payload), encoding="utf-8"
        )

        rebuilt = rebuild_checkpoint_summary(
            model="m",
            revision="r",
            downstream=downstream,
            output_dir=tmp_path,
            max_new_tokens_code=DEFAULT_MAX_NEW_TOKENS_CODE,
            max_new_tokens_mmlu=DEFAULT_MAX_NEW_TOKENS_MMLU,
            timeout=HISTORICAL_DEFAULT_TIMEOUT,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        # Legacy items adopted: python pass recounted from the body outcome.
        assert rebuilt.python_counts[OUTCOME_PASS] == 1
        assert rebuilt.cpp_counts[OUTCOME_PASS] == 1
        assert rebuilt.mmlu_correct == 1


# =============================================================================
# Cache body validators: HumanEval-X body integrity (hash + pinned-input)
# =============================================================================
#


def _he_identity(language: str = "python", nid: int = 1) -> DownstreamIdentity:
    item = _humaneval_item(nid)
    fields = item.python if language == "python" else item.cpp
    return DownstreamIdentity(
        model="m",
        revision="r",
        task=TASK_HUMANEVAL_X,
        language=language,
        prompt_sha256=sha256_hex(fields.prompt),
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS_CODE,
        timeout=10.0,
        task_id=nid,
        test_sha256=sha256_hex(fields.test),
    )


def _he_clean_body(
    *,
    language: str = "python",
    nid: int = 1,
    completion: str | None = None,
    outcome: str = OUTCOME_PASS,
    exit_code: int | None = 0,
    contract: str = GENERATION_CONTRACT_VERSION,
) -> tuple[dict[str, object], DownstreamHumanevalItem, DownstreamIdentity]:
    """Build an internally-consistent HumanEval body + manifest item + identity.

    The assembled/completion hashes are always computed from the stored
    completion so a clean body validates under the recomputing validator.
    """
    item = _humaneval_item(nid)
    fields = item.python if language == "python" else item.cpp
    comp = completion if completion is not None else fields.canonical_solution
    if language == "python":
        program = assemble_python_program(fields.prompt, comp, fields.test)
    else:
        program = assemble_cpp_program(fields.prompt, comp, fields.test)
    ident = _he_identity(language, nid)
    ident_dict = dict(ident.to_dict())
    ident_dict["generation_contract_version"] = contract
    if contract == LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1:
        ident_dict.pop("max_new_tokens", None)
        ident_dict.pop("timeout", None)
    ident_dict.pop("task_id", None)
    ident_dict.pop("test_sha256", None)
    if contract == GENERATION_CONTRACT_VERSION:
        ident_dict["task_id"] = nid
        ident_dict["test_sha256"] = sha256_hex(fields.test)
    body: dict[str, object] = {
        "identity": ident_dict,
        "task_id": nid,
        "language": language,
        "prompt": fields.prompt,
        "completion": comp,
        "completion_sha256": sha256_hex(comp),
        "assembled_sha256": sha256_hex(program),
        "outcome": outcome,
        "exit_code": exit_code,
        "diagnostics": "",
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS_CODE,
    }
    return body, item, ident


class TestValidateHumanevalCachedBody:
    def test_accepts_clean_current_contract_body(self):
        body, item, ident = _he_clean_body(language="python", outcome=OUTCOME_PASS)
        fields = item.python
        validated = validate_humaneval_cached_body(
            body,
            expected_identity=ident,
            expected_prompt=fields.prompt,
            expected_test=fields.test,
            expected_task_id=item.numeric_id,
            expected_language="python",
        )
        assert validated.outcome == OUTCOME_PASS
        assert validated.exit_code == 0
        # Recomputed assembled program is returned for downstream rescore use.
        assert validated.assembled_sha256 == body["assembled_sha256"]

    def test_accepts_clean_cpp_body(self):
        body, item, ident = _he_clean_body(
            language="cpp", outcome=OUTCOME_COMPILE_ERROR, exit_code=1
        )
        fields = item.cpp
        validated = validate_humaneval_cached_body(
            body,
            expected_identity=ident,
            expected_prompt=fields.prompt,
            expected_test=fields.test,
            expected_task_id=item.numeric_id,
            expected_language="cpp",
        )
        assert validated.outcome == OUTCOME_COMPILE_ERROR

    def test_rejects_completion_tamper_without_hash_update(self):
        body, item, ident = _he_clean_body(language="python")
        # Edit the completion but leave completion_sha256 stale.
        body["completion"] = "    return 999\n"
        with pytest.raises(ValueError, match="completion_sha256"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_rejects_assembled_hash_drift(self):
        body, item, ident = _he_clean_body(language="python")
        # Update completion + completion_sha256 consistently, but leave the
        # assembled hash stale so the reassembled program no longer matches.
        tampered = "    return 999\n"
        body["completion"] = tampered
        body["completion_sha256"] = sha256_hex(tampered)
        with pytest.raises(ValueError, match="assembled_sha256"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_rejects_prompt_not_matching_pinned_manifest(self):
        body, item, ident = _he_clean_body(language="python")
        # Body claims a different prompt than the manifest expects.
        body["prompt"] = "def tampered():\n    "
        with pytest.raises(ValueError, match="prompt"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_rejects_task_id_mismatch(self):
        body, item, ident = _he_clean_body(language="python", nid=1)
        body["task_id"] = 999
        with pytest.raises(ValueError, match="task_id"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_rejects_language_mismatch(self):
        body, item, ident = _he_clean_body(language="python")
        with pytest.raises(ValueError, match="language"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="cpp",
            )

    def test_rejects_invalid_outcome_enum(self):
        body, item, ident = _he_clean_body(language="python")
        body["outcome"] = "spectacular_success"
        with pytest.raises(ValueError, match="outcome"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_rejects_identity_mismatch(self):
        body, _item, _ident = _he_clean_body(language="python")
        other = DownstreamIdentity(
            model="other/model",
            revision="r",
            task=TASK_HUMANEVAL_X,
            language="python",
            prompt_sha256="0" * 64,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS_CODE,
            timeout=10.0,
        )
        with pytest.raises(ValueError, match="identity"):
            validate_humaneval_cached_body(
                body,
                expected_identity=other,
                expected_prompt="p",
                expected_test="t",
                expected_task_id=1,
                expected_language="python",
            )

    def test_rejects_bad_exit_code_type(self):
        body, item, ident = _he_clean_body(language="python")
        body["exit_code"] = "zero"
        with pytest.raises(ValueError, match="exit_code"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_rejects_bad_diagnostics_type(self):
        body, item, ident = _he_clean_body(language="python")
        body["diagnostics"] = 123
        with pytest.raises(ValueError, match="diagnostics"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test=item.python.test,
                expected_task_id=item.numeric_id,
                expected_language="python",
            )

    def test_accepts_legacy_contract_body_with_real_hashes(self):
        # Legacy eos-truncate-1 bodies carry real completion/assembled hashes
        # (the contract only affects identity shape, not body integrity).
        body, item, ident = _he_clean_body(
            language="python", contract=LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1
        )
        validated = validate_humaneval_cached_body(
            body,
            expected_identity=ident,
            expected_prompt=item.python.prompt,
            expected_test=item.python.test,
            expected_task_id=item.numeric_id,
            expected_language="python",
        )
        assert validated.outcome == OUTCOME_PASS

    def test_rejects_when_manifest_test_drifted(self):
        # If the pinned manifest test changed since generation, the reassembled
        # program hash will not match the stored assembled_sha256.
        body, item, ident = _he_clean_body(language="python")
        with pytest.raises(ValueError, match="assembled_sha256"):
            validate_humaneval_cached_body(
                body,
                expected_identity=ident,
                expected_prompt=item.python.prompt,
                expected_test="assert DRIFTED\n",
                expected_task_id=item.numeric_id,
                expected_language="python",
            )


# =============================================================================
# Cache body validators: MMLU body integrity (re-parse + recompute is_correct)
# =============================================================================
#


def _mmlu_identity(
    index: int = 0, letter: str = "A"
) -> tuple[DownstreamMMLUItem, DownstreamIdentity]:
    downstream = _make_downstream(1, 1)
    he_items, mmlu_items = load_downstream_items(
        downstream, expected_humaneval=1, expected_mmlu=1
    )
    item = mmlu_items[index]
    prompt = build_mmlu_prompt(item.question, item.choices)
    ident = DownstreamIdentity(
        model="m",
        revision="r",
        task=TASK_MMLU,
        language=TASK_MMLU,
        prompt_sha256=sha256_hex(prompt),
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS_MMLU,
        timeout=10.0,
        task_id=item.index,
        test_sha256="",
    )
    return item, ident


def _mmlu_clean_body(
    *, completion: str = "A", contract: str = GENERATION_CONTRACT_VERSION
) -> tuple[dict[str, object], DownstreamMMLUItem, DownstreamIdentity]:
    item, ident = _mmlu_identity()
    prompt = build_mmlu_prompt(item.question, item.choices)
    ident_dict = dict(ident.to_dict())
    ident_dict["generation_contract_version"] = contract
    if contract == LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1:
        ident_dict.pop("max_new_tokens", None)
        ident_dict.pop("timeout", None)
    ident_dict.pop("task_id", None)
    ident_dict.pop("test_sha256", None)
    if contract == GENERATION_CONTRACT_VERSION:
        ident_dict["task_id"] = item.index
        ident_dict["test_sha256"] = ""
    predicted = parse_mmlu_letter(completion)
    body: dict[str, object] = {
        "identity": ident_dict,
        "index": item.index,
        "subject": item.subject,
        "question_sha256": item.question_sha256,
        "prompt": prompt,
        "completion": completion,
        "predicted_letter": predicted,
        "correct_letter": item.answer_letter,
        "is_correct": (predicted == item.answer_letter),
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS_MMLU,
    }
    return body, item, ident


class TestValidateMmluCachedBody:
    def test_accepts_clean_body_and_recomputes_is_correct(self):
        body, item, ident = _mmlu_clean_body(completion="A")
        validated = validate_mmlu_cached_body(
            body, expected_identity=ident, expected_item=item
        )
        assert validated.predicted_letter == "A"
        # answer_letter for index 0 is "A" (MMLU_LETTERS[0]).
        assert validated.is_correct is True

    def test_recomputes_is_correct_ignoring_stored_field(self):
        # Stored is_correct lies (True) but the completion parses to a wrong
        # letter. The validator must recompute, not trust the stored bool.
        body, item, ident = _mmlu_clean_body(completion="B")
        body["is_correct"] = True  # tampered
        validated = validate_mmlu_cached_body(
            body, expected_identity=ident, expected_item=item
        )
        assert validated.predicted_letter == "B"
        # "B" != answer_letter "A" -> recomputed False, regardless of stored.
        assert validated.is_correct is False

    def test_reparses_predicted_ignoring_stored_field(self):
        body, item, ident = _mmlu_clean_body(completion="B")
        body["predicted_letter"] = "C"  # tampered
        validated = validate_mmlu_cached_body(
            body, expected_identity=ident, expected_item=item
        )
        # Re-parsed from completion "B", not the stored "C".
        assert validated.predicted_letter == "B"

    def test_rejects_question_sha256_mismatch(self):
        body, item, ident = _mmlu_clean_body(completion="A")
        body["question_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="question_sha256"):
            validate_mmlu_cached_body(body, expected_identity=ident, expected_item=item)

    def test_rejects_subject_mismatch(self):
        body, item, ident = _mmlu_clean_body(completion="A")
        body["subject"] = "tampered_subject"
        with pytest.raises(ValueError, match="subject"):
            validate_mmlu_cached_body(body, expected_identity=ident, expected_item=item)

    def test_rejects_correct_letter_mismatch(self):
        body, item, ident = _mmlu_clean_body(completion="A")
        body["correct_letter"] = "D"
        with pytest.raises(ValueError, match="correct_letter"):
            validate_mmlu_cached_body(body, expected_identity=ident, expected_item=item)

    def test_rejects_prompt_drift_against_manifest(self):
        body, item, ident = _mmlu_clean_body(completion="A")
        body["prompt"] = "tampered prompt\nAnswer:"
        with pytest.raises(ValueError, match="prompt"):
            validate_mmlu_cached_body(body, expected_identity=ident, expected_item=item)

    def test_rejects_index_mismatch(self):
        body, item, ident = _mmlu_clean_body(completion="A")
        body["index"] = 999
        with pytest.raises(ValueError, match="index"):
            validate_mmlu_cached_body(body, expected_identity=ident, expected_item=item)

    def test_rejects_identity_mismatch(self):
        body, _item, _ident = _mmlu_clean_body(completion="A")
        other = DownstreamIdentity(
            model="other/model",
            revision="r",
            task=TASK_MMLU,
            language=TASK_MMLU,
            prompt_sha256="0" * 64,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS_MMLU,
            timeout=10.0,
        )
        # Build a minimal item-like via the manifest for the call signature.
        ds = _make_downstream(1, 1)
        _he, mmlu = load_downstream_items(ds, expected_humaneval=1, expected_mmlu=1)
        with pytest.raises(ValueError, match="identity"):
            validate_mmlu_cached_body(
                body, expected_identity=other, expected_item=mmlu[0]
            )

    def test_accepts_legacy_contract_body(self):
        body, item, ident = _mmlu_clean_body(
            completion="A", contract=LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1
        )
        validated = validate_mmlu_cached_body(
            body, expected_identity=ident, expected_item=item
        )
        assert validated.is_correct is True


# =============================================================================
# rebuild_checkpoint_summary: validator-backed rebuild + rescore mode
# =============================================================================
#


class TestRebuildValidatorBacked:
    def test_rebuild_rejects_tampered_completion(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        # Tamper with one completion without updating its hashes -> rebuild
        # must refuse (the validator detects completion_sha256 drift).
        path = tmp_path / "humaneval_x_python_0.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["completion"] = "    return 999\n"  # completion_sha256 now stale
        write_item_atomically(path, raw)
        (tmp_path / SUMMARY_FILENAME).unlink()
        with pytest.raises(ValueError, match="completion_sha256"):
            rebuild_checkpoint_summary(
                model="m",
                revision="r",
                downstream=downstream,
                output_dir=tmp_path,
                timeout=10.0,
                expected_humaneval=1,
                expected_mmlu=1,
            )

    def test_rebuild_recomputes_mmlu_is_correct(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        first = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        # Fake tokenizer decodes completion "A"; index 0 correct letter is "A".
        assert first.mmlu_correct == 1
        # Flip the stored is_correct to lie; completion stays "A" (correct).
        path = tmp_path / "mmlu_0.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["is_correct"] = False  # real parse says correct
        write_item_atomically(path, raw)
        (tmp_path / SUMMARY_FILENAME).unlink()
        rebuilt = rebuild_checkpoint_summary(
            model="m",
            revision="r",
            downstream=downstream,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        # Recomputed is_correct (True) wins over the tampered stored False.
        assert rebuilt.mmlu_correct == 1

    def test_rebuild_rejects_invalid_outcome_enum(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        path = tmp_path / "humaneval_x_python_0.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["outcome"] = "bogus_outcome"
        write_item_atomically(path, raw)
        (tmp_path / SUMMARY_FILENAME).unlink()
        with pytest.raises(ValueError, match="outcome"):
            rebuild_checkpoint_summary(
                model="m",
                revision="r",
                downstream=downstream,
                output_dir=tmp_path,
                timeout=10.0,
                expected_humaneval=1,
                expected_mmlu=1,
            )


class TestRebuildRescoreCached:
    def test_rescore_requires_runner(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        runner = _CapturingRunner(returncode=0)
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        with pytest.raises(ValueError, match="runner"):
            rebuild_checkpoint_summary(
                model="m",
                revision="r",
                downstream=downstream,
                output_dir=tmp_path,
                timeout=10.0,
                expected_humaneval=1,
                expected_mmlu=1,
                rescore_cached=True,
                runner=None,
            )

    def test_rescore_reexecutes_every_completion_no_drift(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        # First-pass runner returns rc=0 (pass) for all programs.
        runner = _CapturingRunner(returncode=0)
        first = evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=runner,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        (tmp_path / SUMMARY_FILENAME).unlink()
        # Rescore with a runner that also returns rc=0 -> no drift -> ok.
        rescore_runner = _CapturingRunner(returncode=0)
        rebuilt = rebuild_checkpoint_summary(
            model="m",
            revision="r",
            downstream=downstream,
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
            rescore_cached=True,
            runner=rescore_runner,
        )
        # python (1 run) + cpp (1 compile + 1 run) -> 3 sandbox invocations.
        assert rescore_runner.call_count == 3
        assert rebuilt.python_counts == first.python_counts
        assert rebuilt.cpp_counts == first.cpp_counts

    def test_rescore_detects_outcome_drift_and_raises(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        # Original pass (rc=0) cached on disk.
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=_CapturingRunner(returncode=0),
            output_dir=tmp_path,
            timeout=10.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        (tmp_path / SUMMARY_FILENAME).unlink()
        # Rescore runner now returns rc=1 (fail) -> drift from cached pass.
        drift_runner = _CapturingRunner(returncode=1)
        with pytest.raises(ValueError, match="drift"):
            rebuild_checkpoint_summary(
                model="m",
                revision="r",
                downstream=downstream,
                output_dir=tmp_path,
                timeout=10.0,
                expected_humaneval=1,
                expected_mmlu=1,
                rescore_cached=True,
                runner=drift_runner,
            )

    def test_rescore_uses_current_timeout(self, tmp_path):
        downstream = _make_downstream(1, 1)
        tok = _FakeTokenizer()
        gen = _RecordingGenerator(extra_ids=[1])
        evaluate_checkpoint(
            model="m",
            revision="r",
            downstream=downstream,
            tokenizer=tok,
            generator=gen,
            runner=_CapturingRunner(returncode=0),
            output_dir=tmp_path,
            timeout=5.0,
            expected_humaneval=1,
            expected_mmlu=1,
        )
        (tmp_path / SUMMARY_FILENAME).unlink()
        # A scripted runner that records the timeout forwarded per call.
        scripted = _ScriptedRunner([_completed(0), _completed(0), _completed(0)])
        rebuild_checkpoint_summary(
            model="m",
            revision="r",
            downstream=downstream,
            output_dir=tmp_path,
            timeout=5.0,
            expected_humaneval=1,
            expected_mmlu=1,
            rescore_cached=True,
            runner=scripted,
        )
        forwarded = {call[2] for call in scripted.calls}
        assert forwarded == {5.0}
