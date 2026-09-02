#!/usr/bin/env python3
"""Build the data-only phase of the RL-Zero-Code *Python syntax validity* concept.

This script is **data-only**: it constructs two auditable JSON artifacts under
``data/allenai/Dolci-RL-Zero-Code-7B/`` and runs no model, registers no
concept in ``postdyn/contrastive_datasets.py``, and alters no checkpoint config.

What it produces
----------------

1. ``python_syntax_pairs.json`` -- 50 paired records for the contrastive concept
   *Python syntactic validity*:

       positive = a full, syntactically valid Python program (the official
                  HumanEval-X ``prompt + canonical_solution`` for a task id),
                  verified to ``compile(..., mode='exec')`` without error.
       negative = the *same* program with exactly one deterministic syntax
                  mutation, verified to raise ``SyntaxError`` or
                  ``IndentationError`` under ``compile(..., mode='exec')``.

   Mutations are purely syntactic; no semantic / runtime bug is ever
   introduced. Six balanced mutation kinds are used so the 50 records are not
   50 identical deletions. Target task ids are drawn deterministically from the
   HumanEval-X ids that are **disjoint** from (a) the pinned downstream
   ``humaneval_x_task_ids`` in ``data/shared_item_ids.json`` and (b) the
   legacy ``0..49`` validator report.

2. ``downstream.json`` -- downstream evaluation items aligned to the same
   experiment, with **no overlap** against the target ids:

   * HumanEval-X: the existing 50 pinned ``humaneval_x_task_ids`` with both the
     Python and C++ entries (``prompt``, ``canonical_solution``, full ``code``,
     and official ``test``) copied verbatim from the pinned source revision, so
     a later pass@1 can run official tests.
   * MMLU: 50 questions from ``cais/mmlu`` config ``all`` / split ``test`` at
     the pinned revision, sampled deterministically (seed 42) with one question
     per distinct subject across 50 of the 57 subjects.

Reproducibility & safety
------------------------

* Every upstream source is pinned to an explicit revision.
* Canonical / benchmark code is **never executed on the host**. Positives are
  checked with ``compile(..., mode='exec')`` (parse + bytecode, no execution);
  negatives are checked the same way and must raise ``SyntaxError`` /
  ``IndentationError``. We do not claim official test execution.
* Target ids and MMLU selection use ``random.Random(42)`` over sorted pools so
  reruns are byte-identical.
* Writes are atomic (temp file + ``os.replace``).

Usage
-----

::

    uv run python scripts/build_rl_zero_syntax_concept.py
    uv run python scripts/build_rl_zero_syntax_concept.py --force
    uv run python scripts/build_rl_zero_syntax_concept.py --only pairs
    uv run python scripts/build_rl_zero_syntax_concept.py --only downstream

Network access to ``huggingface.co`` is required (HumanEval-X test fields and
MMLU). The local ``data/humaneval_x.json`` (code fields, tests dropped) is
reused for the python positives and cpp downstream code, but the official
``test`` field is always re-fetched from the pinned revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter
from typing import Any, Callable

# Make ``src`` importable when run as a script.

from postdyn.dataset_store import (  # noqa: E402
    DATASETS_DIR,
    PROJECT_ROOT,
    load_json,
)

# =============================================================================
# Pinned sources
# =============================================================================

#: Directory that already holds pairs.json + README.md for the Dolci concept set.
CONCEPT_DIR: str = str(PROJECT_ROOT / "data" / "allenai" / "Dolci-RL-Zero-Code-7B")
PAIRS_OUT: str = os.path.join(CONCEPT_DIR, "python_syntax_pairs.json")
DOWNSTREAM_OUT: str = os.path.join(CONCEPT_DIR, "downstream.json")

#: HumanEval-X (code concept source + downstream).
HUMANEVAL_X_DATASET: str = "zai-org/humaneval-x"
HUMANEVAL_X_REVISION: str = "62c78627f3072a1454fa0cb0184737cafe5e4198"
HUMANEVAL_X_LANGS: tuple[str, ...] = ("python", "cpp")

#: MMLU (downstream).
MMLU_DATASET: str = "cais/mmlu"
MMLU_REVISION: str = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_CONFIG: str = "all"
MMLU_SPLIT: str = "test"

#: Local HumanEval-X materialization (code fields; tests dropped locally).
LOCAL_HUMANEVAL_X: str = str(DATASETS_DIR / "humaneval_x.json")

#: Path to the existing legacy validator report (first 50 aligned ids = 0..49).
LEGACY_VALIDATION_REPORT: str = str(
    PROJECT_ROOT / "logs" / "artifacts" / "humaneval-x-validation.jsonl"
)

# =============================================================================
# Determinism knobs
# =============================================================================

SAMPLE_SEED: int = 42
N_PAIRS: int = 50
N_MMLU: int = 50

#: Final curated, build-verified mutation kinds (order is the rotation order).
MUTATION_KINDS: tuple[str, ...] = (
    "drop_def_colon",
    "drop_open_paren_def",
    "drop_close_paren_def",
    "drop_def_keyword",
    "unindent_body",
    "indent_def_header",
)

ANSWER_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")


# =============================================================================
# Small helpers
# =============================================================================


def _log(msg: str) -> None:
    print(msg, flush=True)


def sha256_hex(text: str) -> str:
    """SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("```")
    )


def _humaneval_numeric_id(task_id: str) -> int:
    suffix = str(task_id).rsplit("/", maxsplit=1)[-1]
    suffix = suffix.replace("HumanEval", "").replace("/", "")
    return int(suffix)


def compile_check(code: str) -> tuple[bool, str | None]:
    """Return ``(compiled_ok, error_type_or_none)`` without executing ``code``."""
    try:
        compile(code, "<humaneval-x-syntax-check>", "exec")
        return True, None
    except (SyntaxError, IndentationError) as exc:
        return False, type(exc).__name__


def _atomic_write_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# =============================================================================
# Disjoint target-id selection
# =============================================================================


def _load_shared_downstream_ids() -> list[int]:
    """The pinned 50 HumanEval-X ids used as the downstream set."""
    shared = load_json(DATASETS_DIR / "shared_item_ids.json")
    ids = shared.get("humaneval_x_task_ids", [])
    return [int(x) for x in ids]


def _load_legacy_report_ids() -> set[int]:
    """Task ids covered by the legacy validator report (0..49 of aligned set)."""
    path = LEGACY_VALIDATION_REPORT
    ids: set[int] = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.add(int(row["task_id"]))
    return ids


def select_target_ids(python_pool: list[int]) -> list[int]:
    """Pick ``N_PAIRS`` HumanEval-X python ids disjoint from downstream + legacy.

    The exclusion set is the union of the pinned downstream
    ``humaneval_x_task_ids`` and the legacy ``0..49`` validator-report ids.
    Selection is deterministic over the sorted remaining pool with
    :data:`SAMPLE_SEED`.
    """
    downstream = set(_load_shared_downstream_ids())
    legacy = _load_legacy_report_ids()
    excluded = downstream | legacy
    available = sorted(tid for tid in python_pool if tid not in excluded)
    missing = N_PAIRS - len(available)
    if missing > 0:
        raise RuntimeError(
            f"Only {len(available)} HumanEval-X ids are disjoint from the "
            f"downstream ({len(downstream)}) and legacy-report ({len(legacy)}) "
            f"sets; need {N_PAIRS}."
        )
    rng = random.Random(SAMPLE_SEED)
    selected = sorted(rng.sample(available, N_PAIRS))
    # Hard guarantee: disjoint from downstream.
    overlap = set(selected) & downstream
    if overlap:
        raise RuntimeError(f"Target/downstream id overlap detected: {sorted(overlap)}")
    return selected


# =============================================================================
# Syntax mutation kinds
# =============================================================================
#
# Every function below returns ``None`` when it cannot apply to ``code`` and a
# ``str`` (the mutated program) otherwise. The builder verifies, for each
# record, that the chosen mutation actually raises SyntaxError / IndentationError
# under ``compile(..., mode='exec')`` and falls back to the next kind in rotation
# otherwise, so every emitted negative is guaranteed to be a syntax error.


_DEF_HEADER_RE = re.compile(r"^[ \t]*def[ \t]+\w+")
#: Like _DEF_HEADER_RE but stops before the function name (used by drop_def_keyword).
_DEF_KEYWORD_RE = re.compile(r"^(?P<indent>[ \t]*)def[ \t]+")


def _mut_drop_def_colon(code: str) -> str | None:
    """Remove the trailing ``:`` of the first ``def`` header line."""
    for line in code.splitlines(keepends=True):
        if _DEF_HEADER_RE.match(line):
            stripped = line.rstrip("\n")
            tail = stripped.rstrip()
            if tail.endswith(":"):
                return code.replace(line, tail[:-1] + "\n", 1)
    return None


def _mut_drop_open_paren_def(code: str) -> str | None:
    """Remove the opening ``(`` of the first ``def`` header's parameter list.

    Returns the **entire** program with that single character removed from the
    first ``def`` line; every other line is preserved verbatim.
    """
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if _DEF_HEADER_RE.match(line) and "(" in line:
            idx = line.index("(")
            lines[i] = line[:idx] + line[idx + 1 :]
            return "".join(lines)
    return None


def _mut_drop_close_paren_def(code: str) -> str | None:
    """Remove the last ``)`` of the first ``def`` header's parameter list.

    Returns the **entire** program with that single character removed from the
    first ``def`` line; every other line is preserved verbatim.
    """
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if _DEF_HEADER_RE.match(line) and ")" in line:
            idx = line.rindex(")")
            lines[i] = line[:idx] + line[idx + 1 :]
            return "".join(lines)
    return None


def _mut_drop_def_keyword(code: str) -> str | None:
    """Remove only the ``def`` keyword (and its trailing space) from the first header.

    The leading indentation and the function name are preserved. Returns the
    **entire** program with that single localized edit; every other line is
    preserved verbatim.
    """
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = _DEF_KEYWORD_RE.match(line)
        if m:
            lines[i] = m.group("indent") + line[m.end() :]
            return "".join(lines)
    return None


def _mut_unindent_body(code: str) -> str | None:
    """Dedent the first body line of the first ``def`` to column zero.

    Removing the indentation of the function's first statement yields either an
    "expected an indented block" or an inconsistent-block IndentationError.
    """
    lines = code.splitlines(keepends=True)
    def_index = None
    for i, line in enumerate(lines):
        if _DEF_HEADER_RE.match(line):
            def_index = i
            break
    if def_index is None:
        return None
    for j in range(def_index + 1, len(lines)):
        target = lines[j]
        if target.strip() == "":
            continue
        if not target.endswith("\n"):
            return None
        lines[j] = target.lstrip()
        return "".join(lines)
    return None


def _mut_indent_def_header(code: str) -> str | None:
    """Indent a top-level (column-0) ``def`` header to create an unexpected indent."""
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        m = _DEF_HEADER_RE.match(line)
        if m and not line[:1].isspace():
            lines[i] = "    " + line
            return "".join(lines)
    return None


_MUTATIONS: dict[str, Callable[[str], str | None]] = {
    "drop_def_colon": _mut_drop_def_colon,
    "drop_open_paren_def": _mut_drop_open_paren_def,
    "drop_close_paren_def": _mut_drop_close_paren_def,
    "drop_def_keyword": _mut_drop_def_keyword,
    "unindent_body": _mut_unindent_body,
    "indent_def_header": _mut_indent_def_header,
}


def _rotation_for(record_index: int) -> list[str]:
    """Return the mutation-kind priority order for the ``record_index``-th pair.

    Rotating the starting point balances the kinds across the 50 records while
    keeping the selection fully deterministic.
    """
    n = len(MUTATION_KINDS)
    start = record_index % n
    return [MUTATION_KINDS[(start + k) % n] for k in range(n)]


def apply_mutation(code: str, record_index: int) -> tuple[str, str]:
    """Apply the first verified syntax-breaking mutation for this record.

    Returns ``(mutated_code, applied_kind)``. Raises ``RuntimeError`` only if
    no mutation kind breaks the code -- in practice every kind breaks every
    HumanEval-X python program, but the verification is the source of truth.
    """
    for kind in _rotation_for(record_index):
        mutated = _MUTATIONS[kind](code)
        if mutated is None or mutated == code:
            continue
        ok, err = compile_check(mutated)
        if not ok and err in ("SyntaxError", "IndentationError"):
            return mutated, kind
    raise RuntimeError(
        "No syntax mutation produced a SyntaxError/IndentationError for a "
        "target program; this should be impossible for HumanEval-X python."
    )


# =============================================================================
# Target pairs (python_syntax_pairs.json)
# =============================================================================


def build_pairs(target_ids: list[int]) -> dict[str, Any]:
    """Build the 50-record python-syntax-validity concept payload."""
    local = load_json(LOCAL_HUMANEVAL_X)
    py_items = {int(it["numeric_id"]): it for it in local["languages"]["python"]}

    items: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    for idx, tid in enumerate(target_ids):
        if tid not in py_items:
            raise RuntimeError(f"HumanEval-X python id {tid} missing locally")
        prompt = str(py_items[tid].get("prompt", ""))
        canonical = str(py_items[tid].get("canonical_solution", ""))
        code = _strip_code_fences(prompt + canonical)
        if not code.strip():
            raise RuntimeError(f"HumanEval-X python id {tid} has empty code")

        pos_ok, pos_err = compile_check(code)
        if not pos_ok:
            raise RuntimeError(
                f"HumanEval-X python id {tid} positive does not compile: {pos_err}"
            )

        negative, kind = apply_mutation(code, idx)
        _, neg_err = compile_check(negative)
        kind_counts[kind] += 1

        record = {
            "id": sha256_hex(f"humaneval-x-python|{tid}|{HUMANEVAL_X_REVISION}|{kind}"),
            "source_task_id": f"Python/{tid}",
            "numeric_id": tid,
            "mutation_kind": kind,
            "positive": code,
            "negative": negative,
            "positive_sha256": sha256_hex(code),
            "negative_sha256": sha256_hex(negative),
            "provenance": {
                "source_dataset": HUMANEVAL_X_DATASET,
                "source_revision": HUMANEVAL_X_REVISION,
                "positive": (
                    "Official HumanEval-X Python prompt + canonical_solution "
                    "(full valid program). Syntactically valid by construction."
                ),
                "negative": (
                    "Constructed in this project by applying exactly one "
                    "deterministic, purely-syntactic mutation to the positive. "
                    "No semantic or runtime change is intended or claimed."
                ),
            },
            "validation": {
                "positive_compiles": True,
                "positive_compile_error": None,
                "negative_compiles": False,
                "negative_compile_error_type": neg_err,
                "validation_method": (
                    "compile(..., mode='exec') on the host. No benchmark code "
                    "was executed; only parse/bytecode compilation was run."
                ),
                "positive_tests_executed": False,
                "negative_tests_executed": False,
            },
        }
        items.append(record)

    payload = {
        "dataset": "allenai/Dolci-RL-Zero-Code-7B",
        "artifact": "python_syntax_pairs",
        "concept": {
            "name": "python_valid_vs_syntax_error",
            "target": "Python syntactic validity",
            "polarity": (
                "positive = syntactically valid full Python program "
                "(compiles under compile(..., 'exec')); "
                "negative = the same program with one deterministic syntax "
                "mutation that raises SyntaxError or IndentationError."
            ),
            "positive_class": "syntax_valid",
            "negative_class": "syntax_error",
            "dim_extraction": (
                "direction = mean_i(activation(positive_i) - activation(negative_i))"
            ),
        },
        "source": {
            "dataset": HUMANEVAL_X_DATASET,
            "revision": HUMANEVAL_X_REVISION,
            "language": "python",
            "local_file": "data/humaneval_x.json",
        },
        "selection": {
            "n_pairs": N_PAIRS,
            "seed": SAMPLE_SEED,
            "target_ids": target_ids,
            "excluded_downstream_ids": _load_shared_downstream_ids(),
            "excluded_legacy_report_ids": sorted(_load_legacy_report_ids()),
            "disjoint_from": [
                "data/shared_item_ids.json::humaneval_x_task_ids",
                "scripts/artifacts/humaneval-x-validation.jsonl (legacy 0..49)",
            ],
            "disjointness_verified": set(target_ids).isdisjoint(
                set(_load_shared_downstream_ids())
            ),
        },
        "mutation_kinds": list(MUTATION_KINDS),
        "mutation_kind_counts": dict(sorted(kind_counts.items())),
        "build_note": (
            "Data-only diagnostic concept set. The constructed syntax-error "
            "negatives are NOT part of the upstream Dolci RL training data and "
            "do not imply that syntax errors appeared during RL. Positives are "
            "verbatim HumanEval-X programs; validity is compile-verified only "
            "(official tests were not executed in this phase)."
        ),
        "items": items,
    }
    return payload


# =============================================================================
# Downstream HumanEval-X (python + cpp, with official tests)
# =============================================================================


def _humaneval_lang_url(lang: str) -> str:
    return (
        f"https://huggingface.co/data/{HUMANEVAL_X_DATASET}/resolve/"
        f"{HUMANEVAL_X_REVISION}/data/{lang}/data/humaneval.jsonl"
    )


def _fetch_humaneval_lang(lang: str) -> dict[int, dict[str, str]]:
    """Fetch one HumanEval-X language JSONL and index by numeric id."""
    url = _humaneval_lang_url(lang)
    _log(f"  fetching {lang}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "PostDyn-builder/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        raw = response.read().decode("utf-8")
    indexed: dict[int, dict[str, str]] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        nid = _humaneval_numeric_id(str(row["task_id"]))
        prompt = str(row.get("prompt", ""))
        canonical = str(row.get("canonical_solution", ""))
        test = str(row.get("test", ""))
        code = _strip_code_fences(prompt + canonical)
        indexed[nid] = {
            "task_id": str(row["task_id"]),
            "prompt": prompt,
            "canonical_solution": canonical,
            "code": code,
            "test": test,
        }
    _log(f"    {lang}: {len(indexed)} rows indexed")
    return indexed


def build_downstream_humaneval(
    downstream_ids: list[int],
) -> dict[str, Any]:
    """Build the HumanEval-X python+cpp downstream block with official tests."""
    python = _fetch_humaneval_lang("python")
    cpp = _fetch_humaneval_lang("cpp")
    items: list[dict[str, Any]] = []
    for tid in downstream_ids:
        if tid not in python or tid not in cpp:
            raise RuntimeError(
                f"HumanEval-X downstream id {tid} missing in python/cpp fetch"
            )
        py = python[tid]
        cc = cpp[tid]
        py_ok, py_err = compile_check(py["code"])
        items.append(
            {
                "numeric_id": tid,
                "task_id_python": py["task_id"],
                "task_id_cpp": cc["task_id"],
                "python": {
                    "prompt": py["prompt"],
                    "canonical_solution": py["canonical_solution"],
                    "code": py["code"],
                    "test": py["test"],
                    "code_sha256": sha256_hex(py["code"]),
                    "compiles": py_ok,
                    "compile_error": py_err,
                },
                "cpp": {
                    "prompt": cc["prompt"],
                    "canonical_solution": cc["canonical_solution"],
                    "code": cc["code"],
                    "test": cc["test"],
                    "code_sha256": sha256_hex(cc["code"]),
                },
            }
        )
    return {
        "source_dataset": HUMANEVAL_X_DATASET,
        "source_revision": HUMANEVAL_X_REVISION,
        "task_ids": downstream_ids,
        "n_items": len(items),
        "note": (
            "Verbatim from the pinned HumanEval-X revision so that later "
            "pass@1 can assemble + execute the official CodeGeeX programs. "
            "Python compile status is recorded; official tests are NOT run here."
        ),
        "items": items,
    }


# =============================================================================
# Downstream MMLU (50 questions, 50 distinct subjects, seed 42)
# =============================================================================


def build_downstream_mmlu() -> dict[str, Any]:
    """Build the MMLU downstream block: one question per 50 distinct subjects."""
    from datasets import load_dataset  # local import: keep --help offline

    _log(f"  streaming MMLU {MMLU_CONFIG}/{MMLU_SPLIT} @ {MMLU_REVISION[:10]}")
    stream = load_dataset(
        MMLU_DATASET,
        MMLU_CONFIG,
        split=MMLU_SPLIT,
        revision=MMLU_REVISION,
        streaming=True,
    )
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in stream:
        subject = str(row["subject"])
        if subject not in grouped:
            grouped[subject] = []
            order.append(subject)
        grouped[subject].append(
            {
                "question": str(row["question"]),
                "choices": [str(c) for c in row["choices"]],
                "answer": int(row["answer"]),
            }
        )
    all_subjects = sorted(grouped.keys())
    if len(all_subjects) < N_MMLU:
        raise RuntimeError(
            f"MMLU has only {len(all_subjects)} subjects; need {N_MMLU}."
        )
    _log(f"  MMLU subjects available: {len(all_subjects)}")

    rng = random.Random(SAMPLE_SEED)
    selected_subjects = sorted(rng.sample(all_subjects, N_MMLU))
    items: list[dict[str, Any]] = []
    for subject in selected_subjects:
        rows = grouped[subject]
        idx = rng.randrange(len(rows))
        row = rows[idx]
        answer = int(row["answer"])
        answer_letter = (
            ANSWER_LETTERS[answer] if 0 <= answer < len(ANSWER_LETTERS) else "?"
        )
        items.append(
            {
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": answer,
                "answer_letter": answer_letter,
                "question_sha256": sha256_hex(row["question"]),
                "source_revision": MMLU_REVISION,
            }
        )
    return {
        "source_dataset": MMLU_DATASET,
        "source_revision": MMLU_REVISION,
        "config": MMLU_CONFIG,
        "split": MMLU_SPLIT,
        "n_subjects_total": len(all_subjects),
        "n_subjects_selected": len(selected_subjects),
        "n_questions": len(items),
        "seed": SAMPLE_SEED,
        "selection": (
            "50 distinct subjects sampled (seed 42) from the 57 MMLU subjects, "
            "one question per subject (seed 42). Choices kept in original order."
        ),
        "items": items,
    }


def build_downstream(downstream_ids: list[int]) -> dict[str, Any]:
    """Assemble the full downstream payload (HumanEval-X + MMLU)."""
    return {
        "dataset": "allenai/Dolci-RL-Zero-Code-7B",
        "artifact": "downstream",
        "purpose": (
            "Downstream evaluation set aligned to the python_valid_vs_syntax_error "
            "concept: HumanEval-X python/cpp pass@1 and MMLU accuracy, to be "
            "correlated with representation/readout metrics across checkpoints."
        ),
        "humaneval_x": build_downstream_humaneval(downstream_ids),
        "mmlu": build_downstream_mmlu(),
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the data-only RL-Zero-Code Python syntax-validity concept "
            "(python_syntax_pairs.json + downstream.json)."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--only",
        choices=["pairs", "downstream"],
        default=None,
        help="Build only one artifact (default: both).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args(argv)


def _exists_any(paths: list[str]) -> list[str]:
    return [p for p in paths if os.path.exists(p)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.makedirs(CONCEPT_DIR, exist_ok=True)

    build_pairs_flag = args.only in (None, "pairs")
    build_downstream_flag = args.only in (None, "downstream")

    if not args.force:
        blocking: list[str] = []
        if build_pairs_flag and os.path.exists(PAIRS_OUT):
            blocking.append(PAIRS_OUT)
        if build_downstream_flag and os.path.exists(DOWNSTREAM_OUT):
            blocking.append(DOWNSTREAM_OUT)
        if blocking:
            _log(
                "Existing artifact(s) found; pass --force to overwrite:\n  "
                + "\n  ".join(blocking)
            )
            return 0

    # ---- target pairs -----------------------------------------------------
    if build_pairs_flag:
        _log("\n=== python_syntax_pairs.json ===")
        local = load_json(LOCAL_HUMANEVAL_X)
        python_pool = sorted(
            {int(it["numeric_id"]) for it in local["languages"]["python"]}
        )
        _log(f"  local python pool: {len(python_pool)} ids")
        target_ids = select_target_ids(python_pool)
        _log(f"  selected {len(target_ids)} disjoint target ids (seed {SAMPLE_SEED})")
        _log(f"  target ids: {target_ids}")
        payload = build_pairs(target_ids)
        _atomic_write_json(PAIRS_OUT, payload)
        _log(f"  mutation kind counts: {payload['mutation_kind_counts']}")
        _log(f"  wrote {PAIRS_OUT} ({len(payload['items'])} records)")

    # ---- downstream -------------------------------------------------------
    if build_downstream_flag:
        _log("\n=== downstream.json ===")
        downstream_ids = _load_shared_downstream_ids()
        if len(downstream_ids) != N_PAIRS:
            raise RuntimeError(
                f"Expected {N_PAIRS} pinned downstream ids, got {len(downstream_ids)}."
            )
        _log(f"  downstream HumanEval-X ids ({len(downstream_ids)}): {downstream_ids}")
        payload = build_downstream(downstream_ids)
        _atomic_write_json(DOWNSTREAM_OUT, payload)
        _log(
            f"  wrote {DOWNSTREAM_OUT} "
            f"({payload['humaneval_x']['n_items']} HumanEval-X pairs, "
            f"{payload['mmlu']['n_questions']} MMLU questions)"
        )

    _log("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
