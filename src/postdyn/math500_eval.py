"""Deterministic, resumable first-50 evaluator for MATH-500.

The benchmark contract is intentionally separate from HumanEval/MMLU:
``datasets/math500.json`` is read in stored order, the raw ``problem`` string
is the complete prompt, and the raw completion is preserved verbatim. Answers
are scored with math-verify without fallback parsing.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, cast

from src.downstream_eval import (
    BatchCompletionGenerator,
    CompletionGenerator,
    TokenizerLike,
    generate_completion,
    generate_completions_batch,
)

MATH500_COUNT = 50
TASK = "math500_first50"
PROMPT_TEMPLATE = "{problem}"
GENERATION_CONTRACT = "raw-prompt-greedy-v1"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_DTYPE = "bfloat16"
DEFAULT_QUANTIZATION = "none"
SUMMARY_FILENAME = "summary.json"


class MathVerify(Protocol):
    def parse(self, expression: str, *, fallback_mode: str) -> list[object]: ...

    def verify(self, gold: list[object], target: list[object]) -> bool: ...


@dataclass(frozen=True)
class MathItem:
    index: int
    unique_id: str
    problem: str
    answer: str


@dataclass(frozen=True)
class MathIdentity:
    model_key: str
    model: str
    revision: str
    prompt_sha256: str
    ordered_dataset_sha256: str
    max_new_tokens: int
    dtype: str
    quantization: str
    generation_contract: str = GENERATION_CONTRACT
    experiment_identity: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "model": self.model,
            "revision": self.revision,
            "task": TASK,
            "prompt_sha256": self.prompt_sha256,
            "ordered_dataset_sha256": self.ordered_dataset_sha256,
            "max_new_tokens": self.max_new_tokens,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "generation_contract": self.generation_contract,
            "experiment_identity": dict(self.experiment_identity),
        }


@dataclass(frozen=True)
class MathItemResult:
    identity: MathIdentity
    index: int
    unique_id: str
    prompt: str
    completion: str
    answer: str
    completion_sha256: str
    parsed: bool
    correct: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "index": self.index,
            "unique_id": self.unique_id,
            "prompt": self.prompt,
            "prompt_template": PROMPT_TEMPLATE,
            "completion": self.completion,
            "answer": self.answer,
            "completion_sha256": self.completion_sha256,
            "parsed": self.parsed,
            "correct": self.correct,
            "error": self.error,
        }


@dataclass(frozen=True)
class MathSummary:
    model_key: str
    model: str
    revision: str
    n_expected: int
    n_processed: int
    n_correct: int
    n_parsed: int
    errors: int
    accuracy: float
    ordered_dataset_sha256: str
    max_new_tokens: int
    dtype: str
    quantization: str
    generation_contract: str
    prompt_template: str = PROMPT_TEMPLATE

    def to_dict(self) -> dict[str, object]:
        return {
            "task": TASK,
            "model_key": self.model_key,
            "model": self.model,
            "revision": self.revision,
            "n_expected": self.n_expected,
            "n_processed": self.n_processed,
            "n_correct": self.n_correct,
            "n_parsed": self.n_parsed,
            "errors": self.errors,
            "accuracy": self.accuracy,
            "ordered_dataset_sha256": self.ordered_dataset_sha256,
            "max_new_tokens": self.max_new_tokens,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "generation_contract": self.generation_contract,
            "prompt_template": self.prompt_template,
        }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_hash(rows: Sequence[Mapping[str, object]]) -> str:
    encoded = json.dumps(
        list(rows), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(encoded)


def load_first50(path: Path) -> tuple[list[MathItem], str]:
    """Load exactly the stored first 50 rows; never sort or shuffle them."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("dataset") != "math500":
        raise ValueError("MATH-500 file must contain dataset='math500'")
    rows = raw.get("items")
    if not isinstance(rows, list) or len(rows) < MATH500_COUNT:
        raise ValueError(f"MATH-500 file must contain at least {MATH500_COUNT} items")
    selected: list[MathItem] = []
    for index, row in enumerate(rows[:MATH500_COUNT]):
        if not isinstance(row, Mapping):
            raise ValueError(f"items[{index}] must be an object")
        unique_id = row.get("unique_id")
        problem = row.get("problem")
        answer = row.get("answer")
        if not all(isinstance(value, str) for value in (unique_id, problem, answer)):
            raise ValueError(
                f"items[{index}] requires string unique_id, problem, answer"
            )
        selected.append(
            MathItem(index, cast(str, unique_id), cast(str, problem), cast(str, answer))
        )
    selected_rows = [dict(rows[item.index]) for item in selected]
    return selected, _ordered_hash(selected_rows)


def item_filename(index: int) -> str:
    return f"math500_{index:02d}.json"


def write_atomically(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def score_answer(
    gold_answer: str, completion: str, verifier: MathVerify | None = None
) -> tuple[bool, bool, str | None]:
    """Return ``(parsed, correct, error)``; malformed candidates are incorrect."""
    if verifier is None:
        import importlib

        verifier = cast(
            MathVerify, cast(object, importlib.import_module("math_verify"))
        )
    try:
        checker = verifier
        gold = checker.parse(gold_answer, fallback_mode="no_fallback")
        target = checker.parse(completion, fallback_mode="no_fallback")
        if not gold or not target:
            return False, False, "invalid_parse"
        return True, bool(checker.verify(gold, target)), None
    except Exception as exc:  # parser failures are benchmark-incorrect, not skips
        return False, False, f"verification_error: {type(exc).__name__}: {exc}"


def _identity(
    item: MathItem,
    *,
    model_key: str,
    model: str,
    revision: str,
    dataset_hash: str,
    max_new_tokens: int,
    dtype: str,
    quantization: str,
    experiment_identity: Mapping[str, object] | None,
) -> MathIdentity:
    return MathIdentity(
        model_key,
        model,
        revision,
        sha256_text(item.problem),
        dataset_hash,
        max_new_tokens,
        dtype,
        quantization,
        experiment_identity=experiment_identity or {},
    )


def _cached(
    path: Path,
    expected: MathIdentity,
    item: MathItem,
    verifier: MathVerify | None,
    *,
    verify_score: bool = True,
) -> MathItemResult | None:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or raw.get("identity") != expected.to_dict():
            return None
        if (
            raw.get("index") != item.index
            or raw.get("unique_id") != item.unique_id
            or raw.get("prompt") != item.problem
        ):
            return None
        if not isinstance(raw.get("completion"), str) or not isinstance(
            raw.get("answer"), str
        ):
            return None
        if not isinstance(raw.get("parsed"), bool) or not isinstance(
            raw.get("correct"), bool
        ):
            return None
        error = raw.get("error")
        if error is not None and not isinstance(error, str):
            return None
        if isinstance(error, str) and error.startswith("generation_error:"):
            return None
        completion = cast(str, raw["completion"])
        answer = cast(str, raw["answer"])
        if answer != item.answer or raw.get("prompt_template") != PROMPT_TEMPLATE:
            return None
        if raw.get("completion_sha256") != sha256_text(completion):
            return None
        parsed = cast(bool, raw["parsed"])
        correct = cast(bool, raw["correct"])
        if verify_score and (
            error is None
            or error == "invalid_parse"
            or not error.startswith("generation_error:")
        ):
            checked = score_answer(answer, completion, verifier)
            if (parsed, correct, error) != checked:
                return None
        return MathItemResult(
            expected,
            item.index,
            item.unique_id,
            item.problem,
            completion,
            answer,
            sha256_text(completion),
            parsed,
            correct,
            error,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return None


def load_authoritative_summary(
    *,
    output_dir: Path,
    model: str,
    model_key: str,
    revision: str,
    dataset_path: Path,
    max_new_tokens: int,
    dtype: str,
    quantization: str,
    experiment_identity: Mapping[str, object],
    verifier: MathVerify | None = None,
) -> dict[str, object] | None:
    """Return a summary only when all 50 item files are valid and consistent."""
    items, dataset_hash = load_first50(dataset_path)
    score_verifier = verifier
    if score_verifier is None:
        import importlib

        try:
            score_verifier = cast(
                MathVerify, cast(object, importlib.import_module("math_verify"))
            )
        except ModuleNotFoundError:
            return None
    results: list[MathItemResult] = []
    for item in items:
        expected = _identity(
            item,
            model_key=model_key,
            model=model,
            revision=revision,
            dataset_hash=dataset_hash,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            quantization=quantization,
            experiment_identity=experiment_identity,
        )
        result = _cached(
            output_dir / item_filename(item.index),
            expected,
            item,
            score_verifier,
            verify_score=True,
        )
        if result is None:
            return None
        if result.error is not None and result.error != "invalid_parse":
            return None
        results.append(result)
    summary = MathSummary(
        model_key,
        model,
        revision,
        MATH500_COUNT,
        len(results),
        sum(r.correct for r in results),
        sum(r.parsed for r in results),
        sum(r.error is not None for r in results),
        sum(r.correct for r in results) / MATH500_COUNT,
        dataset_hash,
        max_new_tokens,
        dtype,
        quantization,
        GENERATION_CONTRACT,
    ).to_dict()
    try:
        cached_summary = json.loads(
            (output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return summary if cached_summary == summary else None


def evaluate_first50(
    *,
    model: str,
    model_key: str = "",
    revision: str,
    dataset_path: Path,
    output_dir: Path,
    tokenizer: TokenizerLike,
    generator: CompletionGenerator,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 1,
    dtype: str = DEFAULT_DTYPE,
    quantization: str = DEFAULT_QUANTIZATION,
    force: bool = False,
    verifier: MathVerify | None = None,
    progress: Callable[[str], None] | None = None,
    experiment_identity: Mapping[str, object] | None = None,
) -> MathSummary:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size not in (1, 2)
    ):
        raise ValueError("batch_size must be either 1 or 2")
    items, dataset_hash = load_first50(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[MathItemResult] = []
    misses: list[tuple[MathItem, MathIdentity, Path]] = []
    for item in items:
        expected = _identity(
            item,
            model_key=model_key,
            model=model,
            revision=revision,
            dataset_hash=dataset_hash,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            quantization=quantization,
            experiment_identity=experiment_identity,
        )
        path = output_dir / item_filename(item.index)
        result = None if force else _cached(path, expected, item, verifier)
        if result is None:
            (output_dir / SUMMARY_FILENAME).unlink(missing_ok=True)
            misses.append((item, expected, path))
        else:
            results.append(result)
            if progress is not None:
                progress(f"[{item.index + 1}/{MATH500_COUNT}] {item.unique_id}")

    def persist_generated(
        item: MathItem,
        expected: MathIdentity,
        path: Path,
        completion: str,
    ) -> MathItemResult:
        try:
            parsed, correct, error = score_answer(item.answer, completion, verifier)
        except Exception as exc:
            parsed, correct = False, False
            error = f"verification_error: {type(exc).__name__}: {exc}"
        result = MathItemResult(
            expected,
            item.index,
            item.unique_id,
            item.problem,
            completion,
            item.answer,
            sha256_text(completion),
            parsed,
            correct,
            error,
        )
        write_atomically(path, result.to_dict())
        if progress is not None:
            progress(f"[{item.index + 1}/{MATH500_COUNT}] {item.unique_id}")
        return result

    batching_enabled = batch_size == 2
    for start in range(0, len(misses), batch_size):
        chunk = misses[start : start + batch_size]
        completions: list[str] | None = None
        if batching_enabled and len(chunk) > 1:
            batch_method = getattr(generator, "generate_batch", None)
            if callable(batch_method):
                try:
                    completions = generate_completions_batch(
                        tokenizer,
                        cast(BatchCompletionGenerator, cast(object, generator)),
                        [item.problem for item, _, _ in chunk],
                        max_new_tokens=max_new_tokens,
                    )
                    if len(completions) != len(chunk):
                        raise ValueError(
                            "batch generation returned the wrong number of completions"
                        )
                except Exception:
                    batching_enabled = False
                    completions = None
            else:
                batching_enabled = False
        if completions is None:
            for item, expected, path in chunk:
                completion = generate_completion(
                    tokenizer, generator, item.problem, max_new_tokens=max_new_tokens
                )
                results.append(persist_generated(item, expected, path, completion))
            continue
        for (item, expected, path), completion in zip(chunk, completions):
            results.append(persist_generated(item, expected, path, completion))
    summary = MathSummary(
        model_key,
        model,
        revision,
        MATH500_COUNT,
        len(results),
        sum(r.correct for r in results),
        sum(r.parsed for r in results),
        sum(r.error is not None for r in results),
        sum(r.correct for r in results) / MATH500_COUNT,
        dataset_hash,
        max_new_tokens,
        dtype,
        quantization,
        GENERATION_CONTRACT,
    )
    write_atomically(output_dir / SUMMARY_FILENAME, summary.to_dict())
    return summary
