"""RED contracts for src.math_pairs (TDD).

Correctness-gated MATH-500 concise/verbose pair construction for the
concept ``concise_math_reasoning_vs_verbose_math_reasoning``.

Pipeline invariants pinned by these tests:
  1. Problems are processed in ascending ``unique_id`` order.
  2. The verbose solution is generated first.
  3. The concise solution is generated conditioned on the verbose
     solution (to preserve the solution method).
  4. The verifier is invoked gold-first ``verify(gold, candidate)``
     for both the verbose and the concise output.
  5. Only pairs where BOTH outputs verify against the gold answer are
     retained.
  6. Exactly ``n_pairs`` (default 50) valid pairs are required; the
     build hard-fails when fewer than ``n_pairs`` valid pairs exist.
  7. JSONL serialization uses a fixed schema and round-trips.

No network, no model, no math-verify: ``generate_fn`` and ``verify_fn``
are plain callables supplied by the test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.math_pairs import (
    CONCEPT_NAME,
    MathPair,
    append_math_pair_jsonl,
    build_math_pairs,
    is_meaningfully_concise,
    read_math_pairs_jsonl,
    write_math_pairs_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _problems(unique_ids):
    return [
        {
            "unique_id": uid,
            "problem": f"What is {uid}+1?",
            "answer": f"{uid + 1}",
        }
        for uid in unique_ids
    ]


def _candidate_uid(candidate: str) -> int:
    return int(candidate[candidate.index("[") + 1 : candidate.index("]")])


def _recorder_gen():
    calls: list[dict] = []

    def generate(problem, mode, verbose_reference=None):
        calls.append(
            {
                "unique_id": problem["unique_id"],
                "mode": mode,
                "verbose_reference": verbose_reference,
            }
        )
        uid = problem["unique_id"]
        if mode == "verbose":
            return f"VERBOSE[{uid}]"
        return f"CONCISE[{uid}]"

    return generate, calls


def _recorder_verify(decider=None):
    calls: list[dict] = []

    def verify(gold, candidate):
        calls.append({"gold": gold, "candidate": candidate})
        if decider is not None:
            return decider(gold, candidate)
        return True

    return verify, calls


# =============================================================================
# Concept contract
# =============================================================================


class TestConceptContract:
    def test_concept_name_is_exact(self):
        assert CONCEPT_NAME == "concise_math_reasoning_vs_verbose_math_reasoning"

    def test_direction_is_concise_minus_verbose(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        assert len(pairs) == 1
        assert pairs[0].concise_solution.startswith("CONCISE")
        assert pairs[0].verbose_solution.startswith("VERBOSE")

    def test_concise_style_requires_material_length_reduction(self):
        verbose = "A detailed derivation with several explanatory words and equations."
        assert is_meaningfully_concise("x=1\nAnswer: 1", verbose)
        assert not is_meaningfully_concise(verbose.removeprefix("A "), verbose)


# =============================================================================
# MathPair schema
# =============================================================================


class TestMathPairSchema:
    def test_constructs_with_required_fields(self):
        pair = MathPair(
            unique_id=1,
            problem="p",
            gold_answer="g",
            verbose_solution="v",
            concise_solution="c",
        )
        assert pair.unique_id == 1
        assert pair.problem == "p"
        assert pair.gold_answer == "g"
        assert pair.verbose_solution == "v"
        assert pair.concise_solution == "c"

    def test_equality_is_field_wise(self):
        a = MathPair(1, "p", "g", "v", "c")
        b = MathPair(1, "p", "g", "v", "c")
        assert a == b


# =============================================================================
# Ascending unique_id processing
# =============================================================================


class TestAscendingUniqueId:
    def test_processes_in_ascending_unique_id_regardless_of_input_order(self):
        gen, calls = _recorder_gen()
        verify, _ = _recorder_verify()
        build_math_pairs(_problems([3, 1, 2]), gen, verify, n_pairs=3)
        seen = [c["unique_id"] for c in calls]
        assert seen == [1, 1, 2, 2, 3, 3]

    def test_returned_pairs_sorted_by_unique_id_ascending(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems([7, 2, 5]), gen, verify, n_pairs=3)
        assert [p.unique_id for p in pairs] == [2, 5, 7]

    def test_preserves_and_sorts_canonical_path_ids(self):
        problems = [
            {
                "unique_id": unique_id,
                "problem": "problem",
                "answer": "answer",
            }
            for unique_id in (
                "test/algebra/2.json",
                "test/algebra/10.json",
            )
        ]

        def generate(problem, mode, verbose_reference=None):
            return f"{mode}: {problem['unique_id']}"

        pairs = build_math_pairs(problems, generate, lambda gold, candidate: True, 2)
        assert [pair.unique_id for pair in pairs] == [
            "test/algebra/10.json",
            "test/algebra/2.json",
        ]


# =============================================================================
# Verbose generated first
# =============================================================================


class TestVerboseGeneratedFirst:
    def test_verbose_call_precedes_concise_call_per_problem(self):
        gen, calls = _recorder_gen()
        verify, _ = _recorder_verify()
        build_math_pairs(_problems([1, 2]), gen, verify, n_pairs=2)
        by_problem: dict = {}
        for c in calls:
            by_problem.setdefault(c["unique_id"], []).append(c["mode"])
        for modes in by_problem.values():
            assert modes == ["verbose", "concise"]


# =============================================================================
# Concise conditioned on verbose
# =============================================================================


class TestConciseConditionedOnVerbose:
    def test_concise_receives_verbose_solution_as_reference(self):
        gen, calls = _recorder_gen()
        verify, _ = _recorder_verify()
        build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        verbose_call = next(c for c in calls if c["mode"] == "verbose")
        concise_call = next(c for c in calls if c["mode"] == "concise")
        assert verbose_call["verbose_reference"] is None
        assert concise_call["verbose_reference"] == "VERBOSE[1]"


# =============================================================================
# Verifier gold-first for both outputs
# =============================================================================


class TestVerifierGoldFirst:
    def test_gold_is_first_argument_when_verifying_verbose(self):
        gen, _ = _recorder_gen()
        verify, calls = _recorder_verify()
        build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        verbose_verify = next(c for c in calls if c["candidate"] == "VERBOSE[1]")
        assert verbose_verify["gold"] == "2"
        assert verbose_verify["candidate"] == "VERBOSE[1]"

    def test_gold_is_first_argument_when_verifying_concise(self):
        gen, _ = _recorder_gen()
        verify, calls = _recorder_verify()
        build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        concise_verify = next(c for c in calls if c["candidate"] == "CONCISE[1]")
        assert concise_verify["gold"] == "2"
        assert concise_verify["candidate"] == "CONCISE[1]"

    def test_verifier_called_once_per_output(self):
        gen, _ = _recorder_gen()
        verify, calls = _recorder_verify()
        build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        assert len(calls) == 2


# =============================================================================
# Only both-correct pairs retained
# =============================================================================


class TestBothCorrectRetention:
    def test_keeps_pair_when_both_correct(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify(lambda gold, cand: True)
        pairs = build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        assert [p.unique_id for p in pairs] == [1]

    def test_drops_pair_when_verbose_wrong(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify(lambda gold, cand: cand != "VERBOSE[1]")
        pairs = build_math_pairs(_problems([1, 2]), gen, verify, n_pairs=1)
        assert [p.unique_id for p in pairs] == [2]

    def test_does_not_generate_concise_when_verbose_is_wrong(self):
        modes = []

        def generate(problem, mode, verbose_reference=None):
            modes.append(mode)
            if mode == "concise":
                raise AssertionError("concise generation should have been skipped")
            return "wrong verbose answer"

        with pytest.raises(ValueError, match="1"):
            build_math_pairs(
                _problems([1]),
                generate,
                lambda gold, candidate: False,
                n_pairs=1,
            )
        assert modes == ["verbose"]

    def test_drops_pair_when_concise_wrong(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify(lambda gold, cand: cand != "CONCISE[1]")
        pairs = build_math_pairs(_problems([1, 2]), gen, verify, n_pairs=1)
        assert [p.unique_id for p in pairs] == [2]

    def test_retains_only_both_correct_from_mixed(self):
        wrong = {("VERBOSE", 1), ("CONCISE", 2), ("VERBOSE", 3)}

        def decider(gold, cand):
            mode = cand.split("[", 1)[0]
            return (mode, _candidate_uid(cand)) not in wrong

        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify(decider)
        pairs = build_math_pairs(_problems([1, 2, 3, 4]), gen, verify, n_pairs=1)
        assert [p.unique_id for p in pairs] == [4]


class TestResumableBuild:
    def test_verified_pair_is_persisted_before_later_generation_crashes(self):
        persisted = []

        def generate(problem, mode, verbose_reference=None):
            if problem["unique_id"] == 2:
                raise RuntimeError("generation interrupted")
            return f"{mode}[{problem['unique_id']}]"

        with pytest.raises(RuntimeError, match="interrupted"):
            build_math_pairs(
                _problems([1, 2]),
                generate,
                lambda gold, candidate: True,
                n_pairs=2,
                on_pair=persisted.append,
            )
        assert [pair.unique_id for pair in persisted] == [1]

    def test_resume_skips_ids_already_persisted(self):
        existing = MathPair(1, "p", "g", "verbose", "concise")
        generated_ids = []

        def generate(problem, mode, verbose_reference=None):
            generated_ids.append(problem["unique_id"])
            return f"{mode}[{problem['unique_id']}]"

        pairs = build_math_pairs(
            _problems([1, 2]),
            generate,
            lambda gold, candidate: True,
            n_pairs=2,
            initial_pairs=[existing],
        )
        assert [pair.unique_id for pair in pairs] == [1, 2]
        assert generated_ids == [2, 2]

    def test_resume_returns_pairs_in_global_id_order(self):
        existing = MathPair(2, "p2", "g2", "verbose", "concise")

        def generate(problem, mode, verbose_reference=None):
            return f"{mode}[{problem['unique_id']}]"

        pairs = build_math_pairs(
            _problems([1, 2]),
            generate,
            lambda gold, candidate: True,
            n_pairs=2,
            initial_pairs=[existing],
        )
        assert [pair.unique_id for pair in pairs] == [1, 2]


# =============================================================================
# Exactly fifty required (hard fail below)
# =============================================================================


class TestExactlyFiftyRequired:
    def test_default_n_pairs_is_fifty(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems(range(50)), gen, verify)
        assert len(pairs) == 50

    def test_hard_fails_when_fewer_problems_than_required(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        with pytest.raises(ValueError, match="50"):
            build_math_pairs(_problems(range(49)), gen, verify)

    def test_hard_fails_when_invalid_outputs_leave_too_few(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify(lambda gold, cand: False)
        with pytest.raises(ValueError, match="50"):
            build_math_pairs(_problems(range(60)), gen, verify)

    def test_returns_exactly_fifty_from_more_candidates(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems(range(60)), gen, verify)
        assert len(pairs) == 50
        assert [p.unique_id for p in pairs] == list(range(50))

    def test_continues_past_failures_until_quota_met(self):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify(lambda gold, cand: _candidate_uid(cand) != 0)
        pairs = build_math_pairs(_problems(range(51)), gen, verify)
        assert len(pairs) == 50
        assert [p.unique_id for p in pairs] == list(range(1, 51))


# =============================================================================
# JSONL schema
# =============================================================================


class TestJsonlSchema:
    def test_record_has_exactly_required_fields(self, tmp_path):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems([1]), gen, verify, n_pairs=1)
        path = tmp_path / "math_pairs.jsonl"
        write_math_pairs_jsonl(path, pairs)
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert set(record) == {
            "unique_id",
            "problem",
            "gold_answer",
            "verbose_solution",
            "concise_solution",
        }
        assert record["unique_id"] == 1

    def test_one_json_object_per_line(self, tmp_path):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems([1, 2, 3]), gen, verify, n_pairs=3)
        path = tmp_path / "math_pairs.jsonl"
        write_math_pairs_jsonl(path, pairs)
        lines = path.read_text().splitlines()
        assert len(lines) == 3
        for line in lines:
            assert isinstance(json.loads(line), dict)


# =============================================================================
# JSONL round-trip
# =============================================================================


class TestJsonlRoundtrip:
    def test_write_then_read_roundtrips(self, tmp_path):
        gen, _ = _recorder_gen()
        verify, _ = _recorder_verify()
        pairs = build_math_pairs(_problems([1, 2, 3]), gen, verify, n_pairs=3)
        path = tmp_path / "math_pairs.jsonl"
        write_math_pairs_jsonl(path, pairs)
        loaded = read_math_pairs_jsonl(path)
        assert loaded == pairs
        assert all(isinstance(p, MathPair) for p in loaded)

    def test_read_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            read_math_pairs_jsonl(tmp_path / "does_not_exist.jsonl")

    def test_canonical_path_id_roundtrips(self, tmp_path):
        pair = MathPair(
            "test/algebra/10.json",
            "problem",
            "answer",
            "verbose",
            "concise",
        )
        path = tmp_path / "math_pairs.jsonl"
        write_math_pairs_jsonl(path, [pair])
        assert read_math_pairs_jsonl(path) == [pair]

    def test_append_persists_a_pair_without_overwriting_existing_rows(self, tmp_path):
        path = tmp_path / "math_pairs.jsonl"
        first = MathPair(1, "p1", "g1", "v1", "c1")
        second = MathPair(2, "p2", "g2", "v2", "c2")
        append_math_pair_jsonl(path, first)
        append_math_pair_jsonl(path, second)
        assert read_math_pairs_jsonl(path) == [first, second]


class TestPreparationCli:
    def test_help_exposes_required_generation_controls(self):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "experiments" / "prepare_math_pairs.py"),
                "--help",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--output" in result.stdout
        assert "--n-pairs" in result.stdout
        assert "--model" in result.stdout
