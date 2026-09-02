"""Focused tests for the RL-Zero-Code Python syntax-validity concept (data-only).

These tests validate the data-only phase deliverables under
``data/allenai/Dolci-RL-Zero-Code-7B/``:

* ``python_syntax_pairs.json`` -- 50 paired records for the
  ``python_valid_vs_syntax_error`` concept.
* ``downstream.json`` -- HumanEval-X python/cpp + MMLU downstream items.

Coverage areas (per the experiment brief):

1. **Deterministic mutations** -- the six mutation kinds are deterministic and
   balanced, and ``apply_mutation`` always returns a verified kind.
2. **Syntax validity / invalidity** -- every positive compiles under
   ``compile(..., mode='exec')``; every negative raises ``SyntaxError`` or
   ``IndentationError`` (never a semantic/runtime bug).
3. **Counts** -- exactly 50 paired records, exactly 50 HumanEval-X downstream
   pairs, exactly 50 MMLU questions.
4. **Provenance** -- source dataset / revision / hashes / honest validation
   metadata are present and self-consistent.
5. **Disjoint ids** -- target ids are disjoint from both the pinned downstream
   ``humaneval_x_task_ids`` and the legacy ``0..49`` validator report, and the
   downstream answers are never syntax-invalid code.

The mutation-function unit tests need no data files. The data-file tests skip
gracefully if the built artifacts are absent (e.g. a fresh clone before running
the builder), but fully validate the artifacts when they exist.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# -----------------------------------------------------------------------------
# Load the builder module from its file path (scripts/ is not a package).
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILDER_PATH = REPO_ROOT / "scripts" / "build_rl_zero_syntax_concept.py"

_spec = importlib.util.spec_from_file_location(
    "build_rl_zero_syntax_concept", BUILDER_PATH
)
assert _spec is not None and _spec.loader is not None
builder = importlib.util.module_from_spec(_spec)
sys.modules["build_rl_zero_syntax_concept"] = builder
_spec.loader.exec_module(builder)

CONCEPT_DIR = REPO_ROOT / "data" / "allenai" / "Dolci-RL-Zero-Code-7B"
PAIRS_PATH = CONCEPT_DIR / "python_syntax_pairs.json"
DOWNSTREAM_PATH = CONCEPT_DIR / "downstream.json"

SAMPLE_VALID = (
    "from typing import List\n"
    "\n\n"
    "def add_one(numbers: List[int]) -> List[int]:\n"
    "    result = []\n"
    "    for n in numbers:\n"
    "        if n > 0:\n"
    "            result.append(n + 1)\n"
    "    return result\n"
)


# =============================================================================
# Helpers
# =============================================================================


def _compile_ok(code: str) -> bool:
    try:
        compile(code, "<test>", "exec")
        return True
    except (SyntaxError, IndentationError):
        return False


def _compile_error_type(code: str) -> str | None:
    try:
        compile(code, "<test>", "exec")
        return None
    except (SyntaxError, IndentationError) as exc:
        return type(exc).__name__


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _common_suffix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def _line_diffs(positive: str, negative: str) -> list[tuple[int, str, str]] | None:
    """Line-by-line diff. Returns ``None`` if line counts differ.

    Otherwise returns ``(index, positive_line, negative_line)`` for each line
    that is not byte-identical between the two programs.
    """
    p = positive.splitlines()
    n = negative.splitlines()
    if len(p) != len(n):
        return None
    return [(i, p[i], n[i]) for i in range(len(p)) if p[i] != n[i]]


def _reference_negative(positive: str, kind: str) -> str | None:
    """Independently re-derive the expected full-program negative for ``kind``.

    This does NOT call the builder's mutation helpers, so it cannot reproduce a
    helper bug. Each branch performs exactly one localized, in-place edit on the
    full program and returns the rejoined source.
    """
    lines = positive.splitlines(keepends=True)
    def_idx = -1
    for i, ln in enumerate(lines):
        if re.match(r"^[ \t]*def[ \t]+\w+", ln):
            def_idx = i
            break
    if def_idx < 0:
        return None
    header = lines[def_idx]
    if kind == "drop_def_colon":
        body = header.rstrip("\n").rstrip()
        if not body.endswith(":"):
            return None
        lines[def_idx] = body[:-1] + "\n"
    elif kind == "drop_open_paren_def":
        if "(" not in header:
            return None
        j = header.index("(")
        lines[def_idx] = header[:j] + header[j + 1 :]
    elif kind == "drop_close_paren_def":
        if ")" not in header:
            return None
        j = header.rindex(")")
        lines[def_idx] = header[:j] + header[j + 1 :]
    elif kind == "drop_def_keyword":
        m = re.match(r"^(?P<indent>[ \t]*)def[ \t]+", header)
        if m is None:
            return None
        lines[def_idx] = m.group("indent") + header[m.end() :]
    elif kind == "unindent_body":
        body_idx = None
        for k in range(def_idx + 1, len(lines)):
            if lines[k].strip() != "":
                body_idx = k
                break
        if body_idx is None or not lines[body_idx].endswith("\n"):
            return None
        lines[body_idx] = lines[body_idx].lstrip()
    elif kind == "indent_def_header":
        if (len(header) - len(header.lstrip(" \t"))) != 0:
            return None
        lines[def_idx] = "    " + header
    else:
        return None
    return "".join(lines)


def _require_pairs() -> dict[str, Any]:
    if not PAIRS_PATH.exists():
        pytest.skip(
            f"{PAIRS_PATH} not built; run scripts/build_rl_zero_syntax_concept.py"
        )
    return json.loads(PAIRS_PATH.read_text(encoding="utf-8"))


def _require_downstream() -> dict[str, Any]:
    if not DOWNSTREAM_PATH.exists():
        pytest.skip(
            f"{DOWNSTREAM_PATH} not built; run scripts/build_rl_zero_syntax_concept.py"
        )
    return json.loads(DOWNSTREAM_PATH.read_text(encoding="utf-8"))


# =============================================================================
# 1. Deterministic mutations (unit tests, no data files)
# =============================================================================


class TestMutationKinds:
    """Each mutation kind is deterministic and produces a verified syntax error."""

    @pytest.mark.parametrize("kind", list(builder.MUTATION_KINDS))
    def test_each_kind_breaks_sample(self, kind: str) -> None:
        assert _compile_ok(SAMPLE_VALID), "fixture must be valid Python"
        mutated = builder._MUTATIONS[kind](SAMPLE_VALID)
        assert mutated is not None, f"mutation {kind} did not apply to the fixture"
        assert mutated != SAMPLE_VALID
        err = _compile_error_type(mutated)
        assert err in ("SyntaxError", "IndentationError"), (
            f"mutation {kind} produced {err!r} (should be a syntax/indent error)"
        )

    def test_mutations_are_deterministic(self) -> None:
        for kind in builder.MUTATION_KINDS:
            fn = builder._MUTATIONS[kind]
            assert fn(SAMPLE_VALID) == fn(SAMPLE_VALID)

    @pytest.mark.parametrize("kind", list(builder.MUTATION_KINDS))
    def test_mutation_returns_full_program_with_single_local_edit(
        self, kind: str
    ) -> None:
        """Regression guard: a mutation must return the WHOLE program, edited on
        exactly one line, not just the mutated header line (a prior defect)."""
        mutated = builder._MUTATIONS[kind](SAMPLE_VALID)
        assert mutated is not None, f"mutation {kind} did not apply"
        diffs = _line_diffs(SAMPLE_VALID, mutated)
        assert diffs is not None, f"{kind}: line count changed (not a full program)"
        assert len(diffs) == 1, (
            f"{kind}: expected exactly one edited line, got {len(diffs)}: {diffs}"
        )
        shared = _common_prefix_len(SAMPLE_VALID, mutated) + _common_suffix_len(
            SAMPLE_VALID, mutated
        )
        assert shared >= 0.9 * len(SAMPLE_VALID), f"{kind}: edit is not localized"
        assert _compile_error_type(mutated) in ("SyntaxError", "IndentationError")

    def test_drop_def_keyword_keeps_function_name(self) -> None:
        mutated = builder._MUTATIONS["drop_def_keyword"](SAMPLE_VALID)
        assert mutated is not None
        assert "def add_one" not in mutated
        assert "add_one(" in mutated, "function name must be preserved"

    def test_rotation_is_balanced_and_complete(self) -> None:
        # Each rotation starting index yields a full ordering of all kinds.
        n = len(builder.MUTATION_KINDS)
        for i in range(n):
            order = builder._rotation_for(i)
            assert sorted(order) == sorted(builder.MUTATION_KINDS)
            assert order[0] == builder.MUTATION_KINDS[i % n]

    def test_rotation_covers_every_kind_as_first_across_pool(self) -> None:
        # Over a pool the size of the kind count, every kind leads once.
        firsts = [
            builder._rotation_for(i)[0] for i in range(len(builder.MUTATION_KINDS))
        ]
        assert sorted(firsts) == sorted(builder.MUTATION_KINDS)


class TestApplyMutation:
    def test_returns_verified_kind(self) -> None:
        mutated, kind = builder.apply_mutation(SAMPLE_VALID, 0)
        assert kind in builder.MUTATION_KINDS
        assert not _compile_ok(mutated)
        assert _compile_error_type(mutated) in ("SyntaxError", "IndentationError")

    def test_is_deterministic(self) -> None:
        a = builder.apply_mutation(SAMPLE_VALID, 3)
        b = builder.apply_mutation(SAMPLE_VALID, 3)
        assert a == b

    def test_balanced_over_synthetic_pool(self) -> None:
        # A pool larger than the kind count should exercise every kind.
        codes = [SAMPLE_VALID] * (len(builder.MUTATION_KINDS) * 4)
        used = {builder.apply_mutation(c, i)[1] for i, c in enumerate(codes)}
        assert used == set(builder.MUTATION_KINDS)


# =============================================================================
# 2-5. python_syntax_pairs.json validation
# =============================================================================


class TestPythonSyntaxPairs:
    """Validate counts, syntax validity, provenance, and disjoint ids."""

    def test_has_exactly_50_records(self) -> None:
        data = _require_pairs()
        assert len(data["items"]) == 50

    def test_concept_metadata(self) -> None:
        data = _require_pairs()
        assert data["concept"]["name"] == "python_valid_vs_syntax_error"
        assert data["concept"]["positive_class"] == "syntax_valid"
        assert data["concept"]["negative_class"] == "syntax_error"

    def test_positives_compile_and_negatives_break(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            assert _compile_ok(it["positive"]), (
                f"positive for {it['numeric_id']} must compile"
            )
            err = _compile_error_type(it["negative"])
            assert err in ("SyntaxError", "IndentationError"), (
                f"negative for {it['numeric_id']} must raise syntax/indent error, got {err!r}"
            )
            assert it["validation"]["negative_compile_error_type"] == err

    def test_hashes_are_self_consistent(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            assert (
                hashlib.sha256(it["positive"].encode()).hexdigest()
                == it["positive_sha256"]
            )
            assert (
                hashlib.sha256(it["negative"].encode()).hexdigest()
                == it["negative_sha256"]
            )

    def test_record_fields_present(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            for key in (
                "id",
                "source_task_id",
                "numeric_id",
                "mutation_kind",
                "positive",
                "negative",
                "positive_sha256",
                "negative_sha256",
                "provenance",
                "validation",
            ):
                assert key in it, f"record {it.get('numeric_id')} missing {key}"

    def test_provenance_fields(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            prov = it["provenance"]
            assert prov["source_dataset"] == builder.HUMANEVAL_X_DATASET
            assert prov["source_revision"] == builder.HUMANEVAL_X_REVISION
            assert "positive" in prov and isinstance(prov["positive"], str)
            assert "negative" in prov and isinstance(prov["negative"], str)

    def test_validation_is_honest_compile_only(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            v = it["validation"]
            assert v["positive_compiles"] is True
            assert v["negative_compiles"] is False
            assert v["positive_tests_executed"] is False
            assert v["negative_tests_executed"] is False
            assert "compile" in v["validation_method"]

    def test_mutation_kinds_are_balanced(self) -> None:
        data = _require_pairs()
        counts = data["mutation_kind_counts"]
        # Every declared kind must appear at least once.
        for kind in data["mutation_kinds"]:
            assert kind in counts and counts[kind] > 0, f"kind {kind} unused"
        # Total equals record count.
        assert sum(counts.values()) == len(data["items"])
        # No single kind dominates (balanced distribution).
        assert max(counts.values()) <= min(counts.values()) + 2

    def test_mutation_kind_matches_applied(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            kind = it["mutation_kind"]
            assert kind in data["mutation_kinds"]
            repro = builder._MUTATIONS[kind](it["positive"])
            assert repro == it["negative"], (
                f"stored kind {kind} does not reproduce negative for {it['numeric_id']}"
            )

    def test_negative_preserves_rest_of_program(self) -> None:
        """Strong invariant: each negative is the positive with exactly one
        localized edit, independently re-derived (not via the builder helpers)."""
        data = _require_pairs()
        for it in data["items"]:
            positive = it["positive"]
            negative = it["negative"]
            kind = it["mutation_kind"]
            diffs = _line_diffs(positive, negative)
            assert diffs is not None, (
                f"task {it['numeric_id']}: negative line count differs from positive "
                f"(not a full-program mutation)"
            )
            assert len(diffs) == 1, (
                f"task {it['numeric_id']} ({kind}): expected exactly one edited line, "
                f"got {len(diffs)}"
            )
            shared = _common_prefix_len(positive, negative) + _common_suffix_len(
                positive, negative
            )
            assert shared >= 0.9 * len(positive), (
                f"task {it['numeric_id']} ({kind}): edit is not localized "
                f"(shared {shared} < 90% of {len(positive)})"
            )
            expected = _reference_negative(positive, kind)
            assert expected is not None, (
                f"task {it['numeric_id']} ({kind}): reference transform not applicable"
            )
            assert negative == expected, (
                f"task {it['numeric_id']} ({kind}): negative does not match the "
                f"independent full-program reference transform"
            )

    def test_drop_def_keyword_records_keep_function_name(self) -> None:
        data = _require_pairs()
        for it in data["items"]:
            if it["mutation_kind"] != "drop_def_keyword":
                continue
            pos_lines = it["positive"].splitlines()
            neg_lines = it["negative"].splitlines()
            for line in pos_lines:
                m = re.match(r"^[ \t]*def[ \t]+(\w+)", line)
                if m:
                    name = m.group(1)
                    assert any(name + "(" in nl for nl in neg_lines), (
                        f"task {it['numeric_id']}: function name {name!r} not preserved"
                    )
                    break

    def test_unique_ids_and_task_ids(self) -> None:
        data = _require_pairs()
        ids = [it["id"] for it in data["items"]]
        tids = [it["numeric_id"] for it in data["items"]]
        assert len(set(ids)) == len(ids)
        assert len(set(tids)) == len(tids)
        assert len(tids) == 50

    def test_target_disjoint_from_downstream(self) -> None:
        data = _require_pairs()
        target = set(data["selection"]["target_ids"])
        downstream = set(data["selection"]["excluded_downstream_ids"])
        assert target.isdisjoint(downstream)
        assert data["selection"]["disjointness_verified"] is True

    def test_target_disjoint_from_legacy_report(self) -> None:
        data = _require_pairs()
        target = set(data["selection"]["target_ids"])
        legacy = set(data["selection"]["excluded_legacy_report_ids"])
        assert target.isdisjoint(legacy)

    def test_source_revision_pinned(self) -> None:
        data = _require_pairs()
        assert data["source"]["revision"] == "62c78627f3072a1454fa0cb0184737cafe5e4198"


# =============================================================================
# downstream.json validation
# =============================================================================


class TestDownstreamHumanEvalX:
    def test_has_exactly_50_pairs(self) -> None:
        data = _require_downstream()
        hx = data["humaneval_x"]
        assert hx["n_items"] == 50
        assert len(hx["items"]) == 50

    def test_downstream_ids_match_pinned(self) -> None:
        data = _require_downstream()
        shared = json.loads(
            (REPO_ROOT / "data" / "shared_item_ids.json").read_text("utf-8")
        )
        pinned = shared["humaneval_x_task_ids"]
        assert data["humaneval_x"]["task_ids"] == pinned

    def test_python_and_cpp_fields_preserved(self) -> None:
        data = _require_downstream()
        for it in data["humaneval_x"]["items"]:
            for lang in ("python", "cpp"):
                entry = it[lang]
                for field in ("prompt", "canonical_solution", "code", "test"):
                    assert isinstance(entry[field], str)
                    assert entry[field].strip(), (
                        f"{lang} task {it['numeric_id']} has empty {field}"
                    )
                assert (
                    entry["code_sha256"]
                    == hashlib.sha256(entry["code"].encode()).hexdigest()
                )

    def test_downstream_python_compiles(self) -> None:
        # Downstream answers must be valid code (never syntax-invalid).
        data = _require_downstream()
        for it in data["humaneval_x"]["items"]:
            py = it["python"]["code"]
            assert _compile_ok(py), f"downstream python {it['numeric_id']} must compile"
            assert it["python"]["compiles"] is True

    def test_downstream_disjoint_from_target(self) -> None:
        pairs = json.loads(PAIRS_PATH.read_text("utf-8"))
        down = _require_downstream()
        target = set(pairs["selection"]["target_ids"])
        downstream = set(down["humaneval_x"]["task_ids"])
        assert target.isdisjoint(downstream)


class TestDownstreamMMLU:
    def test_has_exactly_50_questions(self) -> None:
        data = _require_downstream()
        mmlu = data["mmlu"]
        assert mmlu["n_questions"] == 50
        assert len(mmlu["items"]) == 50

    def test_50_distinct_subjects(self) -> None:
        data = _require_downstream()
        subjects = [it["subject"] for it in data["mmlu"]["items"]]
        assert len(set(subjects)) == 50
        assert data["mmlu"]["n_subjects_selected"] == 50
        assert data["mmlu"]["n_subjects_total"] == 57

    def test_fields_preserved_and_consistent(self) -> None:
        data = _require_downstream()
        letters = ("A", "B", "C", "D")
        for it in data["mmlu"]["items"]:
            assert len(it["choices"]) == 4
            assert isinstance(it["answer"], int)
            assert 0 <= it["answer"] < 4
            assert it["answer_letter"] == letters[it["answer"]]
            assert (
                it["question_sha256"]
                == hashlib.sha256(it["question"].encode()).hexdigest()
            )
            assert it["source_revision"] == builder.MMLU_REVISION

    def test_seed_and_source_pinned(self) -> None:
        data = _require_downstream()
        mmlu = data["mmlu"]
        assert mmlu["seed"] == 42
        assert mmlu["config"] == "all"
        assert mmlu["split"] == "test"
        assert mmlu["source_revision"] == "c30699e8356da336a370243923dbaf21066bb9fe"
        assert mmlu["source_dataset"] == "cais/mmlu"
