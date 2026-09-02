"""Raw greedy generation + safe downstream scoring for HumanEval-X and MMLU.

This module is the model-side companion to
``experiments/validate_rl_zero_downstream.py``. Where that CLI validates the
*official canonical solutions* in a bubblewrap sandbox as a hard preflight
gate, this module scores the *model's own raw completions* against the same
official tests, plus MMLU multiple-choice accuracy, across the RL-Zero-Code
checkpoint series.

Design rules (enforced here and by the tests):

* **Raw direct tokenization only.** No chat template, no system prompt. The
  prompt fed to the model is the verbatim HumanEval prompt (or the built MMLU
  "Question + A-D + Answer:" prompt).
* **Deterministic greedy decoding.** ``do_sample=False``, ``num_beams=1`` and
  an explicit ``max_new_tokens`` are always passed to generation. Only the
  *generated* tokens (the slice past the prompt) are decoded; the prompt is
  never echoed into the completion.
* **Raw HumanEval completion preserved exactly.** The model's decoded
  completion is stored verbatim and assembled into the program unchanged --
  no ``<think>`` stripping, no markdown-fence removal, and never substituted
  with the canonical solution. The canonical solution is loaded only as
  provenance and to guard against accidental substitution.
* **Assembly reuses the official CodeGeeX helpers.** Python programs are
  assembled with :func:`src.humaneval_x_validator.assemble_python_program`
  (``prompt + completion + test``) and C++ with
  :func:`assemble_cpp_program`, then executed exclusively through the
  injected/existing :class:`SandboxRunner`. No program code is ever executed
  in the host Python process.
* **Atomic, resumable per-item outputs.** Every item is written to its own
  JSON file via a temp file + ``os.replace`` swap, and identified by a
  ``(model, revision, task, language, prompt_sha256)`` tuple so a stale or
  mismatched file is detected and regenerated before scoring is reused.
* **No ``Any`` / ``# type: ignore``.** Public APIs are fully typed; HF model
  and tokenizer surfaces are captured by narrow ``Protocol`` types and torch
  tensors by the concrete ``torch.Tensor`` type (imported only under
  ``TYPE_CHECKING`` so the module imports cheaply).

Tests inject a fake ``CompletionGenerator`` and a fake ``SandboxRunner`` and
never load a real model or execute downloaded code on the host.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    cast,
)

from src.humaneval_x_validator import (
    OUTCOME_COMPILE_ERROR,
    OUTCOME_ERROR,
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_TIMEOUT,
    ProgramOutcome,
    SandboxRunner,
    assemble_cpp_program,
    assemble_python_program,
    run_cpp_program,
    run_python_program,
    sha256_hex,
)
from src.rl_zero_experiment import (
    DOWNSTREAM_HUMANEVAL_X_ITEMS,
    DOWNSTREAM_MMLU_ITEMS,
    validate_downstream_counts,
)

if TYPE_CHECKING:
    import torch  # noqa: F401  (annotation-only; avoids runtime import cost)


# =============================================================================
# Constants
# =============================================================================

#: Default max generated tokens for HumanEval-X code completion. Always passed
#: explicitly to generation so the slice length is bounded and deterministic.
DEFAULT_MAX_NEW_TOKENS_CODE: int = 512

#: Default max generated tokens for an MMLU answer (a single letter is enough,
#: but a short budget tolerates "The answer is B." preamble).
DEFAULT_MAX_NEW_TOKENS_MMLU: int = 32

#: Default per-program subprocess timeout forwarded to the sandbox runner.
DEFAULT_TIMEOUT_SECONDS: float = 10.0

#: Historical default timeout used by the EOS-corrected (``eos-truncate-1``)
#: downstream runs. The legacy adoption path accepts an ``eos-truncate-1``
#: cached item only when the *expected* timeout equals this value, so a
#: non-default timeout always invalidates a legacy file.
HISTORICAL_DEFAULT_TIMEOUT: float = 10.0

#: The four MMLU answer letters, in choice order.
MMLU_LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

#: Identity "task" namespace for HumanEval-X items.
TASK_HUMANEVAL_X: str = "humaneval_x"
#: Identity "task" namespace for MMLU items.
TASK_MMLU: str = "mmlu"

#: Outcome used when generation itself raises (distinct from sandbox errors).
OUTCOME_GENERATION_ERROR: str = "generation_error"

#: The closed set of outcome strings a cached HumanEval-X body may carry.
#: Used by the cache-body validator to reject corrupted outcome labels.
_VALID_HUMANEVAL_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_PASS,
        OUTCOME_FAIL,
        OUTCOME_TIMEOUT,
        OUTCOME_COMPILE_ERROR,
        OUTCOME_ERROR,
        OUTCOME_GENERATION_ERROR,
    }
)

#: Per-item summary filename written under each checkpoint output dir.
SUMMARY_FILENAME: str = "summary.json"

#: Current generation-contract version. Bumped whenever the completion-
#: decoding or generation-forwarding semantics change (e.g. adding EOS
#: truncation or explicit EOS/pad forwarding to the HF model), so cached
#: items written under a previous contract are automatically invalidated by
#: the identity match and regenerated.
#:
#: ``eos-truncate-2`` extended the identity with ``max_new_tokens`` and a
#: finite positive ``timeout`` so a token-budget or timeout change
#: invalidates the cache instead of silently reusing stale items.
#:
#: ``eos-truncate-3`` extends the identity further with ``task_id`` and
#: ``test_sha256`` (for HumanEval-X) so a manifest test/source change or a
#: task-id swap invalidates the cache at the cheap identity level without
#: waiting for body validation. The narrowly validated ``eos-truncate-2``
#: and ``eos-truncate-1`` adoption paths in :func:`identity_matches` keep
#: existing EOS-corrected files reusable, but only after full body
#: validation succeeds (:func:`validate_humaneval_cached_body` /
#: :func:`validate_mmlu_cached_body`) on every cache hit.
GENERATION_CONTRACT_VERSION: str = "eos-truncate-3"

#: The ``eos-truncate-2`` contract version (EOS-truncation + explicit
#: EOS/pad forwarding + ``max_new_tokens``/``timeout`` in the identity, but
#: no ``task_id``/``test_sha256``). Cached items carrying this version are
#: accepted by :func:`identity_matches` only under the narrow adoption that
#: checks every shared identity field and then requires full body
#: validation before the cached result is reused.
LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_2: str = "eos-truncate-2"

#: The ``eos-truncate-1`` contract version (EOS-truncation + explicit
#: EOS/pad forwarding, but the identity omitted ``max_new_tokens`` and
#: ``timeout``). Cached items carrying this version are accepted by
#: :func:`identity_matches` only under the narrow default-only adoption.
LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1: str = "eos-truncate-1"

#: Version assigned to identity dicts written before this contract existed.
#: Cached items carrying this sentinel (or missing the key entirely) never
#: equal :data:`GENERATION_CONTRACT_VERSION`, so they are always regenerated.
_GENERATION_CONTRACT_VERSION_LEGACY: str = "legacy"


def humaneval_item_filename(language: str, numeric_id: int) -> str:
    """Filename for one scored HumanEval-X item's atomic JSON.

    Single source of truth so the orchestration CLI can scan a checkpoint
    directory for cached items without duplicating the path pattern.
    """
    return f"humaneval_x_{language}_{numeric_id}.json"


def mmlu_item_filename(index: int) -> str:
    """Filename for one scored MMLU item's atomic JSON."""
    return f"mmlu_{index}.json"


# =============================================================================
# Generation contract (tokenizer + generator protocols)
# =============================================================================


class TokenizerLike(Protocol):
    """Minimal tokenizer surface: encode text to ids, decode ids to text.

    Exposes ``eos_token_id`` and ``pad_token_id`` (both ``int | None``) so the
    generator can forward them to the HF model and the completion decoder can
    truncate at EOS. HuggingFace ``PreTrainedTokenizer`` /
    ``PreTrainedTokenizerFast`` satisfy this structurally (``encode`` returns
    ``list[int]``, ``decode`` returns ``str`` and accepts
    ``skip_special_tokens``).
    """

    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = ...
    ) -> str: ...


class CompletionGenerator(Protocol):
    """Deterministic greedy generator returning the full token-id row.

    Implementations MUST return the complete sequence (prompt + generated) so
    the caller can slice off the prompt and decode only the new tokens. The
    signature pins ``do_sample`` / ``num_beams`` so a test fake can assert the
    exact greedy kwargs were forwarded.
    """

    def generate(
        self,
        input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> Sequence[int]: ...


class BatchCompletionGenerator(Protocol):
    """Optional deterministic generator for left-padded prompt batches."""

    def generate_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> Sequence[Sequence[int]]: ...


def generate_completion(
    tokenizer: TokenizerLike,
    generator: CompletionGenerator,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    """Generate one raw greedy completion for ``prompt``.

    Raw direct tokenization (no chat template). Always calls ``generate``
    with ``do_sample=False`` and ``num_beams=1`` and the explicit
    ``max_new_tokens``. Decodes **only the generated tokens** (the slice of
    the output past the prompt length) with ``skip_special_tokens=False`` so
    the completion is preserved verbatim -- markdown fences, ``<think>``
    blocks and any literal text the model emitted are kept untouched.

    **EOS truncation:** the generated token-id slice is defensively truncated
    at the first occurrence of ``tokenizer.eos_token_id`` (when it is not
    ``None``) before decoding, so any tokens the model emitted *after* the
    end-of-text -- which are not meaningful content -- never enter the decoded
    completion. The EOS token itself is excluded from the decode; other
    special tokens in the payload are preserved because
    ``skip_special_tokens`` stays ``False``.

    The returned string is the raw model completion with no post-processing.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    input_ids: list[int] = list(tokenizer.encode(prompt))
    prompt_len: int = len(input_ids)
    full_ids: list[int] = list(
        generator.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    )
    # Decode only the generated tokens (everything past the prompt).
    generated_ids: list[int] = full_ids[prompt_len:]
    # Defensively truncate at the first EOS so post-EOS garbage (tokens the
    # model emitted after the end-of-text, e.g. repeated corpus text) is never
    # decoded into the completion. The EOS token itself is excluded; other
    # special tokens in the payload survive because decode below still uses
    # skip_special_tokens=False. Skipped entirely when eos_token_id is None.
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        for i, token_id in enumerate(generated_ids):
            if token_id == eos_token_id:
                generated_ids = generated_ids[:i]
                break
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def _truncate_generated_ids(
    tokenizer: TokenizerLike, generated_ids: list[int]
) -> list[int]:
    """Apply the canonical EOS truncation used by singleton generation."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        for i, token_id in enumerate(generated_ids):
            if token_id == eos_token_id:
                return generated_ids[:i]
    return generated_ids


def compare_singleton_and_batch_token_ids(
    tokenizer: TokenizerLike,
    generator: CompletionGenerator,
    batch_generator: BatchCompletionGenerator,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
) -> bool:
    """Compare singleton and left-padded batch completion token suffixes."""
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if not prompts:
        return True
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("batch generation requires tokenizer.pad_token_id")
    encoded = [list(tokenizer.encode(prompt)) for prompt in prompts]
    singleton_suffixes = []
    for input_ids in encoded:
        full_ids = list(
            generator.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        )
        singleton_suffixes.append(
            _truncate_generated_ids(tokenizer, full_ids[len(input_ids) :])
        )
    width = max(len(row) for row in encoded)
    padded = [[pad_token_id] * (width - len(row)) + row for row in encoded]
    attention_mask = [[0] * (width - len(row)) + [1] * len(row) for row in encoded]
    rows = list(
        batch_generator.generate_batch(
            padded,
            attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    )
    if len(rows) != len(prompts):
        raise ValueError("batch generation returned the wrong number of rows")
    for row, expected in zip(rows, singleton_suffixes):
        if isinstance(row, (str, bytes)):
            raise ValueError("batch generation returned a non-token row")
        full_ids = list(row)
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in full_ids
        ):
            raise ValueError("batch generation returned a non-integer token")
        if len(full_ids) < width:
            raise ValueError("batch generation returned a short token row")
        actual = _truncate_generated_ids(tokenizer, full_ids[width:])
        if actual != expected:
            return False
    return True


def generate_completions_batch(
    tokenizer: TokenizerLike,
    generator: BatchCompletionGenerator,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    """Generate raw completions for prompts using left padding and masks.

    The generated rows include the padded input prefix.  Each completion is
    therefore sliced from the common padded width, not from its unpadded prompt
    length.  This function deliberately raises for an unavailable/malformed
    batch surface; callers can then fail closed to the exact singleton path.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
    if not prompts:
        return []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("batch generation requires tokenizer.pad_token_id")
    encoded = [list(tokenizer.encode(prompt)) for prompt in prompts]
    width = max(len(row) for row in encoded)
    padded = [[pad_token_id] * (width - len(row)) + row for row in encoded]
    attention_mask = [[0] * (width - len(row)) + [1] * len(row) for row in encoded]
    full_rows = generator.generate_batch(
        padded,
        attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    rows = list(full_rows)
    if len(rows) != len(prompts):
        raise ValueError("batch generation returned the wrong number of rows")
    completions: list[str] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise ValueError("batch generation returned a non-token row")
        full_ids: list[int] = []
        for token_id in row:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ValueError("batch generation returned a non-integer token")
            full_ids.append(token_id)
        if len(full_ids) < width:
            raise ValueError("batch generation returned a short token row")
        generated_ids = _truncate_generated_ids(tokenizer, full_ids[width:])
        completions.append(tokenizer.decode(generated_ids, skip_special_tokens=False))
    return completions


# =============================================================================
# GreedyGenerator: HF-backed adapter (typed, torch only under TYPE_CHECKING)
# =============================================================================


class _TensorGeneratingModel(Protocol):
    """Minimal ``model.generate`` surface returning a token-id tensor.

    ``eos_token_id`` and ``pad_token_id`` are accepted as keyword arguments so
    :class:`GreedyGenerator` can forward the tokenizer's values explicitly
    (overriding a possibly-empty ``generation_config.json``). Both default to
    ``None``; HuggingFace ``generate`` treats ``None`` as "use the model
    config default".
    """

    def generate(
        self,
        input_ids: "torch.Tensor",
        *,
        max_new_tokens: int,
        do_sample: bool = ...,
        num_beams: int = ...,
        eos_token_id: int | None = ...,
        pad_token_id: int | None = ...,
    ) -> "torch.Tensor": ...


class _TensorBatchGeneratingModel(Protocol):
    def generate(
        self,
        input_ids: "torch.Tensor",
        *,
        attention_mask: "torch.Tensor",
        max_new_tokens: int,
        do_sample: bool = ...,
        num_beams: int = ...,
        eos_token_id: int | None = ...,
        pad_token_id: int | None = ...,
    ) -> "torch.Tensor": ...


@dataclass
class GreedyGenerator:
    """``CompletionGenerator`` backed by a HuggingFace model + tokenizer.

    Converts token-id lists <-> ``torch.Tensor`` at the boundary so the
    protocol stays list-typed. The model and tokenizer are typed by narrow
    protocols; real ``transformers`` objects satisfy them structurally.

    This adapter is exercised only by the real run (one RTX 4090); unit tests
    inject a fake ``CompletionGenerator`` and never touch torch.
    """

    model: _TensorGeneratingModel
    tokenizer: TokenizerLike
    device: str = "cuda"

    def generate(
        self,
        input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> list[int]:
        import torch

        tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        # Forward the tokenizer's EOS and pad IDs explicitly so generation
        # stops at the real end-of-text even when
        # ``model.generation_config.eos_token_id`` is None (OLMo ships an
        # effectively empty generation_config.json). ``pad_token_id`` may be
        # None when the tokenizer has no pad token; HF generate accepts None
        # and falls back to the model config.
        output = self.model.generate(
            tensor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # First (only) row -> list[int]. Coerce element-wise so the element
        # type is a concrete ``int`` regardless of tensor dtype.
        row: list[int] = [int(token) for token in output[0].tolist()]
        return row

    def generate_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        attention_mask: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> list[list[int]]:
        import torch

        if self.tokenizer.pad_token_id is None:
            raise ValueError("batch generation requires tokenizer.pad_token_id")
        tensor = torch.tensor(list(input_ids), dtype=torch.long, device=self.device)
        mask = torch.tensor(list(attention_mask), dtype=torch.long, device=self.device)
        batch_model = cast(_TensorBatchGeneratingModel, cast(object, self.model))
        output = batch_model.generate(
            tensor,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if output.ndim != 2:
            raise ValueError("batch generation must return a rank-two tensor")
        return [[int(token) for token in row.tolist()] for row in output]


# =============================================================================
# MMLU prompt + deterministic letter parser
# =============================================================================


def build_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    """Build the raw MMLU prompt: question, A-D choices, then ``Answer:``.

    No chat template, no instructions -- just the question and labelled
    choices terminated by a bare ``Answer:`` that the model continues. The
    format is deterministic so the prompt hash is stable for resume.
    """
    if len(choices) != len(MMLU_LETTERS):
        raise ValueError(
            f"MMLU item must have {len(MMLU_LETTERS)} choices, got {len(choices)}"
        )
    lines: list[str] = [question, ""]
    for letter, choice in zip(MMLU_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


# A "standalone" A-D letter: preceded by start-of-text or whitespace and
# followed by whitespace, common answer punctuation, or end-of-text. This
# rejects letters embedded inside words (e.g. "AB", "CAT") while accepting
# "B", "B.", "B)", "B:", " B, ", "The answer is B".
_MMLU_LETTER_RE: re.Pattern[str] = re.compile(r"(?:^|\s)([A-D])(?:[\s.,):}]|$)")


def parse_mmlu_letter(completion: str) -> str:
    """Deterministically extract the first standalone A-D letter.

    Scans the completion for the first occurrence of an ``A``-``D`` letter
    that is *standalone* -- not part of a larger word. Returns the uppercase
    letter, or ``""`` if no standalone answer letter is found.
    """
    if not completion:
        return ""
    match = _MMLU_LETTER_RE.search(completion)
    if match is None:
        return ""
    return match.group(1)


# =============================================================================
# Typed downstream item loaders (raw dict -> dataclass with validation)
# =============================================================================


@dataclass(frozen=True)
class HumanevalLangFields:
    """Per-language fields of one HumanEval-X item needed for scoring."""

    prompt: str
    canonical_solution: str
    test: str


@dataclass(frozen=True)
class DownstreamHumanevalItem:
    """One aligned python+cpp HumanEval-X item from ``downstream.json``."""

    numeric_id: int
    python: HumanevalLangFields
    cpp: HumanevalLangFields


@dataclass(frozen=True)
class DownstreamMMLUItem:
    """One MMLU question from ``downstream.json``."""

    index: int
    subject: str
    question: str
    choices: tuple[str, ...]
    answer: int
    answer_letter: str
    question_sha256: str


def _require_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string, got {type(value).__name__}")
    return value


def _parse_humaneval_lang(
    block: object, language: str, task_id: int
) -> HumanevalLangFields:
    if not isinstance(block, Mapping):
        raise ValueError(
            f"HumanEval-X item {task_id} '{language}' block is not an object"
        )
    prompt = _require_str(block.get("prompt"), f"humaneval {language} prompt")
    canonical_solution = _require_str(
        block.get("canonical_solution"),
        f"humaneval {language} canonical_solution",
    )
    test = _require_str(block.get("test"), f"humaneval {language} test")
    return HumanevalLangFields(
        prompt=prompt, canonical_solution=canonical_solution, test=test
    )


def parse_humaneval_item(
    raw: Mapping[str, object], index: int
) -> DownstreamHumanevalItem:
    """Parse one ``humaneval_x.items`` entry into a typed record.

    ``canonical_solution`` is captured for provenance and to guard against
    accidental canonical substitution; it is never used as the model answer.
    """
    numeric = raw.get("numeric_id")
    if isinstance(numeric, bool) or not isinstance(numeric, int):
        raise ValueError(
            f"HumanEval-X item at index {index} has non-int numeric_id: {numeric!r}"
        )
    python = _parse_humaneval_lang(raw.get("python"), "python", numeric)
    cpp = _parse_humaneval_lang(raw.get("cpp"), "cpp", numeric)
    return DownstreamHumanevalItem(numeric_id=int(numeric), python=python, cpp=cpp)


def parse_mmlu_item(raw: Mapping[str, object], index: int) -> DownstreamMMLUItem:
    """Parse one ``mmlu.items`` entry into a typed record.

    Validates that ``answer`` (0-3) and ``answer_letter`` (A-D) are
    mutually consistent and that ``question_sha256`` matches a freshly
    recomputed ``sha256_hex(question)`` so a corrupt or hand-edited manifest
    is caught at load time rather than silently propagated into cached
    results.
    """
    subject = _require_str(raw.get("subject"), f"mmlu[{index}] subject")
    question = _require_str(raw.get("question"), f"mmlu[{index}] question")
    raw_choices = raw.get("choices")
    if not isinstance(raw_choices, list) or len(raw_choices) != len(MMLU_LETTERS):
        raise ValueError(
            f"mmlu[{index}] choices must be a list of {len(MMLU_LETTERS)} strings"
        )
    choices = tuple(_require_str(c, f"mmlu[{index}] choice") for c in raw_choices)
    answer = raw.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, int):
        raise ValueError(f"mmlu[{index}] answer must be an int 0-3, got {answer!r}")
    if not (0 <= int(answer) < len(MMLU_LETTERS)):
        raise ValueError(f"mmlu[{index}] answer out of range: {answer}")
    answer_letter = _require_str(
        raw.get("answer_letter"), f"mmlu[{index}] answer_letter"
    )
    if answer_letter not in MMLU_LETTERS:
        raise ValueError(f"mmlu[{index}] answer_letter not A-D: {answer_letter!r}")
    if MMLU_LETTERS[int(answer)] != answer_letter:
        raise ValueError(
            f"mmlu[{index}] answer {answer} (={MMLU_LETTERS[int(answer)]!r}) "
            f"inconsistent with answer_letter {answer_letter!r}"
        )
    question_sha256 = _require_str(
        raw.get("question_sha256"), f"mmlu[{index}] question_sha256"
    )
    recomputed = sha256_hex(question)
    if recomputed != question_sha256:
        raise ValueError(
            f"mmlu[{index}] question_sha256 does not match a recomputed "
            f"hash of the question text"
        )
    return DownstreamMMLUItem(
        index=index,
        subject=subject,
        question=question,
        choices=choices,
        answer=int(answer),
        answer_letter=answer_letter,
        question_sha256=question_sha256,
    )


def load_downstream_items(
    downstream: Mapping[str, object],
    *,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
) -> tuple[list[DownstreamHumanevalItem], list[DownstreamMMLUItem]]:
    """Validate the downstream schema and return typed item lists.

    Delegates count validation to
    :func:`src.rl_zero_experiment.validate_downstream_counts` (so the
    orchestration reuses the single source of truth for the 50/50 contract),
    then parses every item into a typed record with per-field validation.
    """
    validate_downstream_counts(
        dict(downstream),
        expected_humaneval_x=expected_humaneval,
        expected_mmlu=expected_mmlu,
    )
    hx = downstream.get("humaneval_x")
    mmlu = downstream.get("mmlu")
    if not isinstance(hx, Mapping) or not isinstance(mmlu, Mapping):
        # validate_downstream_counts already guards this; keep for the type
        # checker.
        raise ValueError("downstream humaneval_x/mmlu blocks missing")
    raw_humaneval = hx.get("items")
    raw_mmlu = mmlu.get("items")
    if not isinstance(raw_humaneval, list) or not isinstance(raw_mmlu, list):
        raise ValueError("downstream items must be lists")

    humaneval_items: list[DownstreamHumanevalItem] = []
    seen_humaneval_ids: set[int] = set()
    for i, raw in enumerate(raw_humaneval):
        if not isinstance(raw, Mapping):
            raise ValueError(f"humaneval_x.items[{i}] is not an object")
        item = parse_humaneval_item(raw, i)
        if item.numeric_id in seen_humaneval_ids:
            raise ValueError(
                f"humaneval_x.items[{i}] has duplicate numeric_id {item.numeric_id}"
            )
        seen_humaneval_ids.add(item.numeric_id)
        humaneval_items.append(item)

    mmlu_items: list[DownstreamMMLUItem] = []
    seen_mmlu_indices: set[int] = set()
    seen_mmlu_hashes: set[str] = set()
    for i, raw in enumerate(raw_mmlu):
        if not isinstance(raw, Mapping):
            raise ValueError(f"mmlu.items[{i}] is not an object")
        item = parse_mmlu_item(raw, i)
        if item.index in seen_mmlu_indices:
            raise ValueError(f"mmlu.items[{i}] has duplicate index {item.index}")
        if item.question_sha256 in seen_mmlu_hashes:
            raise ValueError(
                f"mmlu.items[{i}] has duplicate question_sha256 {item.question_sha256}"
            )
        seen_mmlu_indices.add(item.index)
        seen_mmlu_hashes.add(item.question_sha256)
        mmlu_items.append(item)

    return humaneval_items, mmlu_items


# =============================================================================
# Identity + result records
# =============================================================================


@dataclass(frozen=True)
class DownstreamIdentity:
    """Binds a per-item result to (model, revision, task, language, prompt).

    A saved item file is reused on resume only when its identity equals the
    expected identity for the current run. Any mismatch (different model /
    revision / task / language, a prompt whose SHA-256 changed, a different
    ``max_new_tokens`` budget, a different ``timeout``, a ``task_id`` or
    ``test_sha256`` change, **or a generation-contract version change**)
    invalidates the cached file and forces regeneration.

    ``task_id`` and ``test_sha256`` were added under ``eos-truncate-3`` so a
    manifest test/source change or a task-id swap invalidates the cache at
    the identity level. For MMLU items ``test_sha256`` is the empty string
    (MMLU has no official test), and ``task_id`` is the item index.

    ``max_new_tokens`` and ``timeout`` were added under ``eos-truncate-2``;
    ``generation_contract_version`` encodes the completion-decoding and
    generation-forwarding semantics and is bumped whenever those change.
    """

    model: str
    revision: str
    task: str
    language: str
    prompt_sha256: str
    max_new_tokens: int
    timeout: float
    task_id: int = 0
    test_sha256: str = ""
    generation_contract_version: str = GENERATION_CONTRACT_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "revision": self.revision,
            "task": self.task,
            "language": self.language,
            "prompt_sha256": self.prompt_sha256,
            "max_new_tokens": self.max_new_tokens,
            "timeout": self.timeout,
            "task_id": self.task_id,
            "test_sha256": self.test_sha256,
            "generation_contract_version": self.generation_contract_version,
        }


@dataclass(frozen=True)
class HumanEvalResult:
    """One scored HumanEval-X completion (python or cpp)."""

    identity: DownstreamIdentity
    task_id: int
    language: str
    prompt: str
    completion: str
    completion_sha256: str
    assembled_sha256: str
    outcome: str
    exit_code: int | None
    diagnostics: str
    max_new_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "task_id": self.task_id,
            "language": self.language,
            "prompt": self.prompt,
            "completion": self.completion,
            "completion_sha256": self.completion_sha256,
            "assembled_sha256": self.assembled_sha256,
            "outcome": self.outcome,
            "exit_code": self.exit_code,
            "diagnostics": self.diagnostics,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class MMLUResult:
    """One scored MMLU completion."""

    identity: DownstreamIdentity
    index: int
    subject: str
    question_sha256: str
    prompt: str
    completion: str
    predicted_letter: str
    correct_letter: str
    is_correct: bool
    max_new_tokens: int

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "index": self.index,
            "subject": self.subject,
            "question_sha256": self.question_sha256,
            "prompt": self.prompt,
            "completion": self.completion,
            "predicted_letter": self.predicted_letter,
            "correct_letter": self.correct_letter,
            "is_correct": self.is_correct,
            "max_new_tokens": self.max_new_tokens,
        }


# =============================================================================
# Atomic per-item JSON write + resume cache
# =============================================================================


def write_item_atomically(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write a single JSON object to ``path``.

    Temp file + ``os.replace``: the destination is created or fully replaced
    only when the write succeeds. A pre-existing file is left untouched if
    the write raises. Mirrors the validator's report-writer contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


_IDENTITY_CORE_FIELDS: tuple[str, ...] = (
    "model",
    "revision",
    "task",
    "language",
    "prompt_sha256",
)


def identity_matches(raw: object, expected: DownstreamIdentity) -> bool:
    """True iff ``raw`` is a cached item whose identity equals ``expected``.

    Used for resume: a file whose identity drifted (different model /
    revision / task / language, a prompt hash mismatch, a token-budget or
    timeout change, or a generation-contract change) is treated as absent
    and regenerated.

    Three acceptance paths:

    * **Current contract** (:data:`GENERATION_CONTRACT_VERSION`,
      ``eos-truncate-3``): the cached identity dict must equal
      ``expected.to_dict()`` exactly -- including ``task_id``,
      ``test_sha256``, ``max_new_tokens`` and ``timeout``.

    * **Legacy ``eos-truncate-2``** adoption: identities that carry
      ``max_new_tokens``/``timeout`` but **not** ``task_id``/``test_sha256``
      remain potentially reusable when every shared core + budget + timeout
      field matches. The missing fields are verified later by full body
      validation (:func:`validate_humaneval_cached_body` /
      :func:`validate_mmlu_cached_body`) on every cache hit, so a test/task
      drift that the identity cannot detect is still caught before reuse.

    * **Legacy ``eos-truncate-1``** adoption: EOS-corrected files whose
      identities omit ``max_new_tokens`` / ``timeout`` remain reusable, but
      only when *all* of the narrow conditions hold: every core identity
      field matches, the cached item *body* carries an integer
      ``max_new_tokens`` equal to ``expected.max_new_tokens``, and
      ``expected.timeout`` equals :data:`HISTORICAL_DEFAULT_TIMEOUT`.

    Any other legacy shape (missing version, pre-version sentinel, missing
    or malformed body budget, budget mismatch, or a non-default timeout on
    an ``eos-truncate-1`` file) is rejected so the item is regenerated.
    """
    if not isinstance(raw, Mapping):
        return False
    ident = raw.get("identity")
    if not isinstance(ident, Mapping):
        return False
    version_raw = ident.get("generation_contract_version")
    if not isinstance(version_raw, str):
        version = _GENERATION_CONTRACT_VERSION_LEGACY
    else:
        version = version_raw

    if version == GENERATION_CONTRACT_VERSION:
        return dict(ident) == expected.to_dict()

    if version == LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_2:
        for field in _IDENTITY_CORE_FIELDS:
            if ident.get(field) != getattr(expected, field):
                return False
        if ident.get("max_new_tokens") != expected.max_new_tokens:
            return False
        if ident.get("timeout") != expected.timeout:
            return False
        return True

    if version == LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1:
        if expected.timeout != HISTORICAL_DEFAULT_TIMEOUT:
            return False
        for field in _IDENTITY_CORE_FIELDS:
            if ident.get(field) != getattr(expected, field):
                return False
        body_mnt = raw.get("max_new_tokens")
        if isinstance(body_mnt, bool) or not isinstance(body_mnt, int):
            return False
        return int(body_mnt) == expected.max_new_tokens

    return False


def load_cached_item(
    path: Path, expected: DownstreamIdentity
) -> dict[str, object] | None:
    """Return the cached item dict if ``path`` exists and its identity matches.

    A missing, corrupt (invalid JSON / OSError), or identity-mismatched file
    returns ``None`` so the caller regenerates it.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            raw: object = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if not identity_matches(raw, expected):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


# =============================================================================
# Finite-positive timeout / positive budget validation
# =============================================================================


def _validate_timeout(timeout: float) -> None:
    """Reject NaN, infinity, and nonpositive timeouts.

    A timeout gates sandbox execution; a NaN/inf/nonpositive value either
    never fires or is silently dropped by ``subprocess``, so it must be
    caught at the library boundary (and the CLI boundary) rather than
    allowing a malformed budget to reach the runner.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError(f"timeout must be a finite positive number, got {timeout!r}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"timeout must be a finite positive number, got {timeout!r}")


def _validate_max_new_tokens(max_new_tokens: int) -> None:
    """Reject nonpositive or non-integer token budgets at the eval boundary."""
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError(
            f"max_new_tokens must be a positive integer, got {max_new_tokens!r}"
        )
    if max_new_tokens <= 0:
        raise ValueError(
            f"max_new_tokens must be a positive integer, got {max_new_tokens!r}"
        )


# =============================================================================
# Scratch directory helper
# =============================================================================


@contextmanager
def scratch_dir(prefix: str) -> Iterator[Path]:
    """Create a fresh temp scratch dir, yield it, remove it on exit.

    Distinct from the validator's per-task scratch so downstream per-item
    isolation is self-contained. Resource isolation inside the sandbox is
    unchanged (the validator's ulimits are applied by ``run_*_program``).
    """
    base = Path(tempfile.gettempdir()) / "downstream_eval"
    base.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=str(base)))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# =============================================================================
# Per-item scoring
# =============================================================================


def score_humaneval_completion(
    *,
    prompt: str,
    completion: str,
    test: str,
    language: str,
    task_id: int,
    runner: SandboxRunner,
    timeout: float,
) -> tuple[str, ProgramOutcome]:
    """Assemble ``prompt + completion + test`` and score it in the sandbox.

    The model completion is assembled in the canonical-solution slot via the
    official CodeGeeX helpers -- verbatim, with no stripping. Returns the
    assembled program source and the :class:`ProgramOutcome` from execution.
    """
    _validate_timeout(timeout)
    if language == "python":
        program = assemble_python_program(prompt, completion, test)
        with scratch_dir(f"he_python_{task_id}") as scratch:
            outcome = run_python_program(program, task_id, scratch, runner, timeout)
        return program, outcome
    if language == "cpp":
        program = assemble_cpp_program(prompt, completion, test)
        with scratch_dir(f"he_cpp_{task_id}") as scratch:
            outcome = run_cpp_program(program, task_id, scratch, runner, timeout)
        return program, outcome
    raise ValueError(f"unsupported HumanEval-X language: {language!r}")


def evaluate_humaneval_item(
    item: DownstreamHumanevalItem,
    language: str,
    *,
    model: str,
    revision: str,
    tokenizer: TokenizerLike,
    generator: CompletionGenerator,
    runner: SandboxRunner,
    output_dir: Path,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_CODE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
) -> HumanEvalResult:
    """Generate, score, and persist one HumanEval-X completion.

    Resume-aware: if a valid cached file exists for this item's identity and
    ``force`` is ``False``, the cached result is returned without generation
    or sandbox work. Every cache hit is body-validated
    (:func:`validate_humaneval_cached_body`); a corrupt or stale body is
    treated as a miss and regenerated.
    """
    _validate_timeout(timeout)
    _validate_max_new_tokens(max_new_tokens)
    fields = item.python if language == "python" else item.cpp
    identity = DownstreamIdentity(
        model=model,
        revision=revision,
        task=TASK_HUMANEVAL_X,
        language=language,
        prompt_sha256=sha256_hex(fields.prompt),
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        task_id=item.numeric_id,
        test_sha256=sha256_hex(fields.test),
    )
    path = output_dir / humaneval_item_filename(language, item.numeric_id)

    raw_cached = None if force else load_cached_item(path, identity)
    if raw_cached is not None:
        try:
            validate_humaneval_cached_body(
                raw_cached,
                expected_identity=identity,
                expected_prompt=fields.prompt,
                expected_test=fields.test,
                expected_task_id=item.numeric_id,
                expected_language=language,
            )
        except ValueError:
            raw_cached = None
    if raw_cached is not None:
        return _human_result_from_dict(raw_cached)

    completion = generate_completion(
        tokenizer, generator, fields.prompt, max_new_tokens=max_new_tokens
    )
    program, outcome = score_humaneval_completion(
        prompt=fields.prompt,
        completion=completion,
        test=fields.test,
        language=language,
        task_id=item.numeric_id,
        runner=runner,
        timeout=timeout,
    )
    result = HumanEvalResult(
        identity=identity,
        task_id=item.numeric_id,
        language=language,
        prompt=fields.prompt,
        completion=completion,
        completion_sha256=sha256_hex(completion),
        assembled_sha256=sha256_hex(program),
        outcome=outcome.status,
        exit_code=outcome.exit_code,
        diagnostics=outcome.diagnostics,
        max_new_tokens=max_new_tokens,
    )
    write_item_atomically(path, result.to_dict())
    return result


def evaluate_mmlu_item(
    item: DownstreamMMLUItem,
    *,
    model: str,
    revision: str,
    tokenizer: TokenizerLike,
    generator: CompletionGenerator,
    output_dir: Path,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
) -> MMLUResult:
    """Generate, parse, and persist one MMLU answer. No sandbox needed.

    ``timeout`` is recorded in the item identity (even though MMLU never
    reaches the sandbox) so a timeout change at the run level still
    invalidates the cache consistently with the HumanEval items. Every cache
    hit is body-validated (:func:`validate_mmlu_cached_body`); a corrupt or
    stale body is treated as a miss and regenerated.
    """
    _validate_timeout(timeout)
    _validate_max_new_tokens(max_new_tokens)
    prompt = build_mmlu_prompt(item.question, item.choices)
    identity = DownstreamIdentity(
        model=model,
        revision=revision,
        task=TASK_MMLU,
        language=TASK_MMLU,
        prompt_sha256=sha256_hex(prompt),
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        task_id=item.index,
        test_sha256="",
    )
    path = output_dir / mmlu_item_filename(item.index)

    raw_cached = None if force else load_cached_item(path, identity)
    if raw_cached is not None:
        try:
            validate_mmlu_cached_body(
                raw_cached, expected_identity=identity, expected_item=item
            )
        except ValueError:
            raw_cached = None
    if raw_cached is not None:
        return _mmlu_result_from_dict(raw_cached)

    completion = generate_completion(
        tokenizer, generator, prompt, max_new_tokens=max_new_tokens
    )
    predicted = parse_mmlu_letter(completion)
    result = MMLUResult(
        identity=identity,
        index=item.index,
        subject=item.subject,
        question_sha256=item.question_sha256,
        prompt=prompt,
        completion=completion,
        predicted_letter=predicted,
        correct_letter=item.answer_letter,
        is_correct=(predicted == item.answer_letter),
        max_new_tokens=max_new_tokens,
    )
    write_item_atomically(path, result.to_dict())
    return result


# =============================================================================
# Cached-result reconstruction (typed)
# =============================================================================


def _human_result_from_dict(raw: Mapping[str, object]) -> HumanEvalResult:
    ident_raw = raw.get("identity")
    if not isinstance(ident_raw, Mapping):
        raise ValueError("cached HumanEval result missing identity")
    identity = _identity_from_dict(ident_raw, raw)
    exit_raw = raw.get("exit_code")
    exit_code: int | None
    if exit_raw is None:
        exit_code = None
    elif isinstance(exit_raw, bool) or not isinstance(exit_raw, int):
        raise ValueError(f"cached exit_code has wrong type: {exit_raw!r}")
    else:
        exit_code = int(exit_raw)
    return HumanEvalResult(
        identity=identity,
        task_id=_as_int(raw.get("task_id"), "task_id"),
        language=_as_str(raw.get("language"), "language"),
        prompt=_as_str(raw.get("prompt"), "prompt"),
        completion=_as_str(raw.get("completion"), "completion"),
        completion_sha256=_as_str(raw.get("completion_sha256"), "completion_sha256"),
        assembled_sha256=_as_str(raw.get("assembled_sha256"), "assembled_sha256"),
        outcome=_as_str(raw.get("outcome"), "outcome"),
        exit_code=exit_code,
        diagnostics=_as_str(raw.get("diagnostics"), "diagnostics"),
        max_new_tokens=_as_int(raw.get("max_new_tokens"), "max_new_tokens"),
    )


def _mmlu_result_from_dict(raw: Mapping[str, object]) -> MMLUResult:
    ident_raw = raw.get("identity")
    if not isinstance(ident_raw, Mapping):
        raise ValueError("cached MMLU result missing identity")
    identity = _identity_from_dict(ident_raw, raw)
    is_correct = raw.get("is_correct")
    if not isinstance(is_correct, bool):
        raise ValueError(f"cached is_correct not bool: {is_correct!r}")
    return MMLUResult(
        identity=identity,
        index=_as_int(raw.get("index"), "index"),
        subject=_as_str(raw.get("subject"), "subject"),
        question_sha256=_as_str(raw.get("question_sha256"), "question_sha256"),
        prompt=_as_str(raw.get("prompt"), "prompt"),
        completion=_as_str(raw.get("completion"), "completion"),
        predicted_letter=_as_str(raw.get("predicted_letter"), "predicted_letter"),
        correct_letter=_as_str(raw.get("correct_letter"), "correct_letter"),
        is_correct=is_correct,
        max_new_tokens=_as_int(raw.get("max_new_tokens"), "max_new_tokens"),
    )


def _identity_from_dict(
    ident_raw: Mapping[str, object], body: Mapping[str, object]
) -> DownstreamIdentity:
    version_raw = ident_raw.get("generation_contract_version")
    version = (
        _as_str(version_raw, "generation_contract_version")
        if isinstance(version_raw, str)
        else _GENERATION_CONTRACT_VERSION_LEGACY
    )
    if version == LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1:
        body_mnt = body.get("max_new_tokens")
        if isinstance(body_mnt, bool) or not isinstance(body_mnt, int):
            raise ValueError(
                f"legacy identity body max_new_tokens not an int: {body_mnt!r}"
            )
        max_new_tokens = int(body_mnt)
        timeout = HISTORICAL_DEFAULT_TIMEOUT
    else:
        max_new_tokens = _as_int(ident_raw.get("max_new_tokens"), "max_new_tokens")
        timeout = _as_float(ident_raw.get("timeout"), "timeout")

    if version == GENERATION_CONTRACT_VERSION:
        task_id = _as_int(ident_raw.get("task_id"), "task_id")
        test_sha256 = _as_str(ident_raw.get("test_sha256"), "test_sha256")
    else:
        raw_id = body.get("task_id")
        if raw_id is None:
            raw_id = body.get("index")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            task_id = 0
        else:
            task_id = int(raw_id)
        test_sha256 = ""

    return DownstreamIdentity(
        model=_as_str(ident_raw.get("model"), "model"),
        revision=_as_str(ident_raw.get("revision"), "revision"),
        task=_as_str(ident_raw.get("task"), "task"),
        language=_as_str(ident_raw.get("language"), "language"),
        prompt_sha256=_as_str(ident_raw.get("prompt_sha256"), "prompt_sha256"),
        max_new_tokens=max_new_tokens,
        timeout=timeout,
        task_id=task_id,
        test_sha256=test_sha256,
        generation_contract_version=version,
    )


def _as_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"cached field {field!r} not a string: {value!r}")
    return value


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"cached field {field!r} not an int: {value!r}")
    return int(value)


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"cached field {field!r} not a number: {value!r}")
    return float(value)


# =============================================================================
# Checkpoint summary
# =============================================================================


@dataclass(frozen=True)
class ScoringConfig:
    """The scoring configuration that produced a checkpoint's items.

    Captured in every checkpoint summary (and the aggregate) so a stale
    summary written under a different token budget, timeout, or generation
    contract is detected and rejected rather than silently reused. The
    fields mirror the run-level parameters of :func:`evaluate_checkpoint`.
    """

    max_new_tokens_code: int
    max_new_tokens_mmlu: int
    timeout: float
    generation_contract_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "max_new_tokens_code": self.max_new_tokens_code,
            "max_new_tokens_mmlu": self.max_new_tokens_mmlu,
            "timeout": self.timeout,
            "generation_contract_version": self.generation_contract_version,
        }


@dataclass(frozen=True)
class CheckpointSummary:
    """Aggregate outcome of one checkpoint's downstream evaluation."""

    model: str
    revision: str
    n_humaneval_python: int
    n_humaneval_cpp: int
    n_mmlu: int
    python_pass_at_1: float
    cpp_pass_at_1: float
    mmlu_accuracy: float
    python_counts: Mapping[str, int]
    cpp_counts: Mapping[str, int]
    mmlu_correct: int
    mmlu_parsed: int
    errors: int
    scoring_config: ScoringConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "revision": self.revision,
            "n_humaneval_python": self.n_humaneval_python,
            "n_humaneval_cpp": self.n_humaneval_cpp,
            "n_mmlu": self.n_mmlu,
            "python_pass_at_1": self.python_pass_at_1,
            "cpp_pass_at_1": self.cpp_pass_at_1,
            "mmlu_accuracy": self.mmlu_accuracy,
            "python_counts": dict(self.python_counts),
            "cpp_counts": dict(self.cpp_counts),
            "mmlu_correct": self.mmlu_correct,
            "mmlu_parsed": self.mmlu_parsed,
            "errors": self.errors,
            "scoring_config": dict(self.scoring_config.to_dict()),
        }


def _empty_counts() -> dict[str, int]:
    return {
        o: 0
        for o in (
            OUTCOME_PASS,
            OUTCOME_FAIL,
            OUTCOME_TIMEOUT,
            OUTCOME_COMPILE_ERROR,
            OUTCOME_ERROR,
            OUTCOME_GENERATION_ERROR,
        )
    }


def is_checkpoint_complete(
    summary: CheckpointSummary,
    *,
    expected_per_language: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
) -> bool:
    """True iff the checkpoint has zero errors and exact expected item counts.

    Completeness is defined as ``errors == 0`` plus exactly
    ``expected_per_language`` Python items, ``expected_per_language`` C++
    items, and ``expected_mmlu`` parsed MMLU items. A checkpoint with any
    generation/scoring error or a short item count is incomplete and must be
    excluded from aggregate coverage.
    """
    return (
        summary.errors == 0
        and summary.n_humaneval_python == expected_per_language
        and summary.n_humaneval_cpp == expected_per_language
        and summary.n_mmlu == expected_mmlu
    )


# =============================================================================
# Orchestration: 50 python + 50 cpp + 50 MMLU per checkpoint
# =============================================================================


def evaluate_checkpoint(
    *,
    model: str,
    revision: str,
    downstream: Mapping[str, object],
    tokenizer: TokenizerLike,
    generator: CompletionGenerator,
    runner: SandboxRunner,
    output_dir: Path,
    max_new_tokens_code: int = DEFAULT_MAX_NEW_TOKENS_CODE,
    max_new_tokens_mmlu: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
    progress: Callable[[str], None] | None = None,
) -> CheckpointSummary:
    """Run one checkpoint: 50 HumanEval python + 50 cpp + 50 MMLU items.

    Validates the downstream schema and exact 50/50 item counts up front
    (delegated to :func:`load_downstream_items`), then generates, scores and
    atomically persists each item. Resume-aware: per-item JSON files whose
    identity matches this run are reused without generation or sandbox work
    (unless ``force``). A :class:`CheckpointSummary` is written to
    ``output_dir/summary.json`` and returned.

    Per-item failures are isolated: a generation or scoring exception is
    recorded as an ``OUTCOME_GENERATION_ERROR`` HumanEval row (or an incorrect
    MMLU row) and counted in ``summary.errors``, so one bad item never aborts
    the whole checkpoint.
    """
    _validate_timeout(timeout)
    _validate_max_new_tokens(max_new_tokens_code)
    _validate_max_new_tokens(max_new_tokens_mmlu)
    humaneval_items, mmlu_items = load_downstream_items(
        downstream,
        expected_humaneval=expected_humaneval,
        expected_mmlu=expected_mmlu,
    )
    if len(humaneval_items) != expected_humaneval:
        raise ValueError(
            f"expected {expected_humaneval} HumanEval-X items, "
            f"got {len(humaneval_items)}"
        )
    if len(mmlu_items) != expected_mmlu:
        raise ValueError(f"expected {expected_mmlu} MMLU items, got {len(mmlu_items)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    python_counts = _empty_counts()
    cpp_counts = _empty_counts()
    errors = 0
    mmlu_correct = 0
    mmlu_parsed = 0

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    for item in humaneval_items:
        for language, counts in (("python", python_counts), ("cpp", cpp_counts)):
            try:
                result = evaluate_humaneval_item(
                    item,
                    language,
                    model=model,
                    revision=revision,
                    tokenizer=tokenizer,
                    generator=generator,
                    runner=runner,
                    output_dir=output_dir,
                    max_new_tokens=max_new_tokens_code,
                    timeout=timeout,
                    force=force,
                )
                counts[result.outcome] = counts.get(result.outcome, 0) + 1
            except Exception as exc:  # noqa: BLE001 - isolate per-item failure
                errors += 1
                counts[OUTCOME_GENERATION_ERROR] = (
                    counts.get(OUTCOME_GENERATION_ERROR, 0) + 1
                )
                _log(f"  ERROR humaneval {language} task {item.numeric_id}: {exc}")

    for item in mmlu_items:
        try:
            result = evaluate_mmlu_item(
                item,
                model=model,
                revision=revision,
                tokenizer=tokenizer,
                generator=generator,
                output_dir=output_dir,
                max_new_tokens=max_new_tokens_mmlu,
                timeout=timeout,
                force=force,
            )
            if result.predicted_letter in MMLU_LETTERS:
                mmlu_parsed += 1
            if result.is_correct:
                mmlu_correct += 1
        except Exception as exc:  # noqa: BLE001 - isolate per-item failure
            errors += 1
            _log(f"  ERROR mmlu[{item.index}] ({item.subject}): {exc}")

    n_python = expected_humaneval
    n_cpp = expected_humaneval
    n_mmlu = expected_mmlu
    summary = CheckpointSummary(
        model=model,
        revision=revision,
        n_humaneval_python=n_python,
        n_humaneval_cpp=n_cpp,
        n_mmlu=n_mmlu,
        python_pass_at_1=python_counts.get(OUTCOME_PASS, 0) / n_python
        if n_python
        else 0.0,
        cpp_pass_at_1=cpp_counts.get(OUTCOME_PASS, 0) / n_cpp if n_cpp else 0.0,
        mmlu_accuracy=mmlu_correct / n_mmlu if n_mmlu else 0.0,
        python_counts=dict(python_counts),
        cpp_counts=dict(cpp_counts),
        mmlu_correct=mmlu_correct,
        mmlu_parsed=mmlu_parsed,
        errors=errors,
        scoring_config=ScoringConfig(
            max_new_tokens_code=max_new_tokens_code,
            max_new_tokens_mmlu=max_new_tokens_mmlu,
            timeout=timeout,
            generation_contract_version=GENERATION_CONTRACT_VERSION,
        ),
    )
    write_item_atomically(output_dir / SUMMARY_FILENAME, summary.to_dict())
    return summary


# =============================================================================
# No-model summary rebuild from existing per-item JSON
# =============================================================================


def _read_required_item(path: Path) -> dict[str, object]:
    """Read one per-item JSON, raising on missing or corrupt files.

    The summary rebuild path reads exactly the expected item files; a missing
    file is a hard error (the checkpoint is incomplete), not a silent skip.
    """
    if not path.exists():
        raise FileNotFoundError(f"required item file missing: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            raw: object = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"item file {path} is corrupt: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"item file {path} is not a JSON object")
    return raw


# =============================================================================
# Cache-body validators: recompute derived fields, never trust editable labels
# =============================================================================
#
# These validators are the integrity layer between a cached per-item JSON and
# the summary rebuild. They recompute every body-derived field (hashes, parsed
# letters, correctness) from the stored completion and the pinned manifest
# inputs, and reject a body whose stored fields drift from the recomputation.
# This is corruption/staleness detection, NOT authenticated tamper resistance:
# a same-user attacker who can rewrite the cache can still flip a stored
# outcome label between valid enum values (only the completion/program hashes
# bind it to the bytes that ran). Re-deriving the outcome label itself
# requires sandbox re-execution -- see ``rescore_cached`` below.


@dataclass(frozen=True)
class ValidatedHumanevalBody:
    """A HumanEval-X cache body after recomputation-based validation.

    ``assembled_program`` is the freshly reassembled source (reused by the
    rescore path to re-execute the completion). ``outcome`` is the *stored*
    label, accepted only behind the recomputed completion/assembled hashes
    and the pinned prompt/task/language/id checks.
    """

    identity: DownstreamIdentity
    task_id: int
    language: str
    prompt: str
    completion: str
    completion_sha256: str
    assembled_program: str
    assembled_sha256: str
    outcome: str
    exit_code: int | None
    diagnostics: str
    max_new_tokens: int


@dataclass(frozen=True)
class ValidatedMMLUBody:
    """A cached MMLU body after deterministic re-derivation of correctness.

    ``predicted_letter`` is re-parsed from the stored completion and
    ``is_correct`` is recomputed against the pinned correct letter; the
    stored editable ``predicted_letter``/``is_correct`` fields are ignored.
    """

    identity: DownstreamIdentity
    index: int
    subject: str
    question_sha256: str
    prompt: str
    completion: str
    predicted_letter: str
    correct_letter: str
    is_correct: bool
    max_new_tokens: int


def validate_humaneval_cached_body(
    raw: Mapping[str, object],
    *,
    expected_identity: DownstreamIdentity,
    expected_prompt: str,
    expected_test: str,
    expected_task_id: int,
    expected_language: str,
) -> ValidatedHumanevalBody:
    """Validate a cached HumanEval-X body by recomputing its derived fields.

    Acceptance requires, in order: identity match (current or legacy
    contract); ``task_id``/``language``/``prompt`` equal the pinned manifest
    inputs; ``sha256_hex(completion)`` equals the stored
    ``completion_sha256``; the freshly reassembled
    ``prompt + completion + expected_test`` program hashes to the stored
    ``assembled_sha256`` (detects completion, prompt, and manifest-test
    drift alike); ``outcome`` is a recognized outcome constant; and
    ``exit_code``/``diagnostics``/``max_new_tokens`` have valid types.

    The stored ``outcome`` label is NOT recomputed here: re-deriving it
    requires sandbox execution (``rescore_cached`` in
    :func:`rebuild_checkpoint_summary`). The label is only accepted behind
    the hash + pinned-input checks, which bind it to the exact completion
    bytes that produced it.
    """
    if not identity_matches(raw, expected_identity):
        raise ValueError(
            "cached HumanEval item identity does not match the expected scoring config"
        )
    ident_raw = raw.get("identity")
    if not isinstance(ident_raw, Mapping):
        raise ValueError("cached HumanEval result missing identity")
    identity = _identity_from_dict(ident_raw, raw)
    task_id = _as_int(raw.get("task_id"), "task_id")
    language = _as_str(raw.get("language"), "language")
    prompt = _as_str(raw.get("prompt"), "prompt")
    completion = _as_str(raw.get("completion"), "completion")
    completion_sha256 = _as_str(raw.get("completion_sha256"), "completion_sha256")
    assembled_sha256_stored = _as_str(raw.get("assembled_sha256"), "assembled_sha256")
    outcome = _as_str(raw.get("outcome"), "outcome")
    exit_raw = raw.get("exit_code")
    if exit_raw is None:
        exit_code: int | None = None
    elif isinstance(exit_raw, bool) or not isinstance(exit_raw, int):
        raise ValueError(f"cached exit_code has wrong type: {exit_raw!r}")
    else:
        exit_code = int(exit_raw)
    diagnostics = _as_str(raw.get("diagnostics"), "diagnostics")
    max_new_tokens = _as_int(raw.get("max_new_tokens"), "max_new_tokens")

    if task_id != expected_task_id:
        raise ValueError(
            f"cached HumanEval task_id {task_id} != expected {expected_task_id}"
        )
    if language != expected_language:
        raise ValueError(
            f"cached HumanEval language {language!r} != expected {expected_language!r}"
        )
    if prompt != expected_prompt:
        raise ValueError(
            "cached HumanEval prompt does not match the pinned manifest prompt"
        )
    if sha256_hex(completion) != completion_sha256:
        raise ValueError(
            "cached HumanEval completion_sha256 does not match a recomputed "
            "hash of the stored completion"
        )
    if language == "python":
        program = assemble_python_program(prompt, completion, expected_test)
    elif language == "cpp":
        program = assemble_cpp_program(prompt, completion, expected_test)
    else:
        raise ValueError(f"unsupported HumanEval-X language: {language!r}")
    if sha256_hex(program) != assembled_sha256_stored:
        raise ValueError(
            "cached HumanEval assembled_sha256 does not match a recomputed "
            "hash of the reassembled program"
        )
    if outcome not in _VALID_HUMANEVAL_OUTCOMES:
        raise ValueError(
            f"cached HumanEval outcome {outcome!r} is not a recognized outcome"
        )
    return ValidatedHumanevalBody(
        identity=identity,
        task_id=task_id,
        language=language,
        prompt=prompt,
        completion=completion,
        completion_sha256=completion_sha256,
        assembled_program=program,
        assembled_sha256=assembled_sha256_stored,
        outcome=outcome,
        exit_code=exit_code,
        diagnostics=diagnostics,
        max_new_tokens=max_new_tokens,
    )


def validate_mmlu_cached_body(
    raw: Mapping[str, object],
    *,
    expected_identity: DownstreamIdentity,
    expected_item: DownstreamMMLUItem,
) -> ValidatedMMLUBody:
    """Validate a cached MMLU body and recompute its correctness in full.

    ``predicted_letter`` is re-parsed from the stored completion and
    ``is_correct`` is recomputed against the pinned correct letter; the
    stored editable ``predicted_letter``/``is_correct`` fields are never
    trusted (MMLU scoring is a deterministic letter parse, so no sandbox is
    needed to re-derive correctness). Acceptance additionally requires
    ``index``/``subject``/``question_sha256``/``correct_letter``/``prompt``
    to match the pinned manifest item.
    """
    if not identity_matches(raw, expected_identity):
        raise ValueError(
            "cached MMLU item identity does not match the expected scoring config"
        )
    ident_raw = raw.get("identity")
    if not isinstance(ident_raw, Mapping):
        raise ValueError("cached MMLU result missing identity")
    identity = _identity_from_dict(ident_raw, raw)
    index = _as_int(raw.get("index"), "index")
    subject = _as_str(raw.get("subject"), "subject")
    question_sha256 = _as_str(raw.get("question_sha256"), "question_sha256")
    prompt = _as_str(raw.get("prompt"), "prompt")
    completion = _as_str(raw.get("completion"), "completion")
    correct_letter = _as_str(raw.get("correct_letter"), "correct_letter")
    max_new_tokens = _as_int(raw.get("max_new_tokens"), "max_new_tokens")

    expected_prompt = build_mmlu_prompt(expected_item.question, expected_item.choices)
    if index != expected_item.index:
        raise ValueError(f"cached MMLU index {index} != expected {expected_item.index}")
    if subject != expected_item.subject:
        raise ValueError(
            f"cached MMLU subject {subject!r} != expected {expected_item.subject!r}"
        )
    if question_sha256 != expected_item.question_sha256:
        raise ValueError(
            "cached MMLU question_sha256 does not match the pinned manifest"
        )
    if correct_letter != expected_item.answer_letter:
        raise ValueError(
            f"cached MMLU correct_letter {correct_letter!r} != expected "
            f"{expected_item.answer_letter!r}"
        )
    if prompt != expected_prompt:
        raise ValueError("cached MMLU prompt does not match the pinned manifest prompt")
    predicted = parse_mmlu_letter(completion)
    is_correct = predicted == expected_item.answer_letter
    return ValidatedMMLUBody(
        identity=identity,
        index=index,
        subject=subject,
        question_sha256=question_sha256,
        prompt=prompt,
        completion=completion,
        predicted_letter=predicted,
        correct_letter=expected_item.answer_letter,
        is_correct=is_correct,
        max_new_tokens=max_new_tokens,
    )


def rebuild_checkpoint_summary(
    *,
    model: str,
    revision: str,
    downstream: Mapping[str, object],
    output_dir: Path,
    max_new_tokens_code: int = DEFAULT_MAX_NEW_TOKENS_CODE,
    max_new_tokens_mmlu: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
    rescore_cached: bool = False,
    runner: SandboxRunner | None = None,
) -> CheckpointSummary:
    """Rebuild a checkpoint summary from existing per-item JSON (no model).

    Reads exactly the expected per-item files and validates each one through
    :func:`validate_humaneval_cached_body` / :func:`validate_mmlu_cached_body`,
    which recompute every body-derived field (hashes, parsed letters,
    correctness) from the stored completion and the pinned manifest inputs.
    MMLU correctness is fully re-derived (a deterministic letter parse); the
    HumanEval outcome label is accepted behind the recomputed completion and
    assembled-program hashes plus the pinned prompt/task/language/id checks.

    When ``rescore_cached`` is ``True`` each cached Python/C++ completion is
    re-executed in the sandbox through ``runner`` with the current ``timeout``
    and the fresh outcome is compared to the stored label; any drift raises
    ``ValueError`` so a stale or tampered outcome cannot silently stand. This
    is the only path that re-derives the HumanEval outcome from actual program
    behavior. ``runner`` MUST be provided when ``rescore_cached`` is ``True``.

    Never loads a model or generates. Raises ``FileNotFoundError`` when any
    expected item file is missing (incomplete checkpoint) and ``ValueError``
    when a file fails body/identity validation or (under rescore) drifts.
    """
    _validate_timeout(timeout)
    _validate_max_new_tokens(max_new_tokens_code)
    _validate_max_new_tokens(max_new_tokens_mmlu)
    if rescore_cached and runner is None:
        raise ValueError(
            "rescore_cached=True requires a sandbox runner to re-execute "
            "cached completions"
        )
    rescore_runner: SandboxRunner | None = runner
    humaneval_items, mmlu_items = load_downstream_items(
        downstream,
        expected_humaneval=expected_humaneval,
        expected_mmlu=expected_mmlu,
    )
    if len(humaneval_items) != expected_humaneval:
        raise ValueError(
            f"expected {expected_humaneval} HumanEval-X items, "
            f"got {len(humaneval_items)}"
        )
    if len(mmlu_items) != expected_mmlu:
        raise ValueError(f"expected {expected_mmlu} MMLU items, got {len(mmlu_items)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    python_counts = _empty_counts()
    cpp_counts = _empty_counts()
    mmlu_correct = 0
    mmlu_parsed = 0

    for item in humaneval_items:
        for language, counts in (("python", python_counts), ("cpp", cpp_counts)):
            fields = item.python if language == "python" else item.cpp
            identity = DownstreamIdentity(
                model=model,
                revision=revision,
                task=TASK_HUMANEVAL_X,
                language=language,
                prompt_sha256=sha256_hex(fields.prompt),
                max_new_tokens=max_new_tokens_code,
                timeout=timeout,
                task_id=item.numeric_id,
                test_sha256=sha256_hex(fields.test),
            )
            path = output_dir / humaneval_item_filename(language, item.numeric_id)
            raw = _read_required_item(path)
            validated = validate_humaneval_cached_body(
                raw,
                expected_identity=identity,
                expected_prompt=fields.prompt,
                expected_test=fields.test,
                expected_task_id=item.numeric_id,
                expected_language=language,
            )
            outcome = validated.outcome
            if rescore_cached and rescore_runner is not None:
                fresh = score_humaneval_completion(
                    prompt=validated.prompt,
                    completion=validated.completion,
                    test=fields.test,
                    language=language,
                    task_id=item.numeric_id,
                    runner=rescore_runner,
                    timeout=timeout,
                )[1]
                if fresh.status != outcome:
                    raise ValueError(
                        f"cached {language} item {item.numeric_id} outcome "
                        f"drift: stored={outcome} rescored={fresh.status}"
                    )
            counts[outcome] = counts.get(outcome, 0) + 1

    for item in mmlu_items:
        prompt = build_mmlu_prompt(item.question, item.choices)
        identity = DownstreamIdentity(
            model=model,
            revision=revision,
            task=TASK_MMLU,
            language=TASK_MMLU,
            prompt_sha256=sha256_hex(prompt),
            max_new_tokens=max_new_tokens_mmlu,
            timeout=timeout,
            task_id=item.index,
            test_sha256="",
        )
        path = output_dir / mmlu_item_filename(item.index)
        raw = _read_required_item(path)
        validated = validate_mmlu_cached_body(
            raw, expected_identity=identity, expected_item=item
        )
        if validated.predicted_letter in MMLU_LETTERS:
            mmlu_parsed += 1
        if validated.is_correct:
            mmlu_correct += 1

    n_python = expected_humaneval
    n_cpp = expected_humaneval
    n_mmlu = expected_mmlu
    summary = CheckpointSummary(
        model=model,
        revision=revision,
        n_humaneval_python=n_python,
        n_humaneval_cpp=n_cpp,
        n_mmlu=n_mmlu,
        python_pass_at_1=python_counts.get(OUTCOME_PASS, 0) / n_python
        if n_python
        else 0.0,
        cpp_pass_at_1=cpp_counts.get(OUTCOME_PASS, 0) / n_cpp if n_cpp else 0.0,
        mmlu_accuracy=mmlu_correct / n_mmlu if n_mmlu else 0.0,
        python_counts=dict(python_counts),
        cpp_counts=dict(cpp_counts),
        mmlu_correct=mmlu_correct,
        mmlu_parsed=mmlu_parsed,
        errors=0,
        scoring_config=ScoringConfig(
            max_new_tokens_code=max_new_tokens_code,
            max_new_tokens_mmlu=max_new_tokens_mmlu,
            timeout=timeout,
            generation_contract_version=GENERATION_CONTRACT_VERSION,
        ),
    )
    write_item_atomically(output_dir / SUMMARY_FILENAME, summary.to_dict())
    return summary


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS_CODE",
    "DEFAULT_MAX_NEW_TOKENS_MMLU",
    "DEFAULT_TIMEOUT_SECONDS",
    "DownstreamHumanevalItem",
    "DownstreamIdentity",
    "DownstreamMMLUItem",
    "GENERATION_CONTRACT_VERSION",
    "GreedyGenerator",
    "HISTORICAL_DEFAULT_TIMEOUT",
    "HumanevalLangFields",
    "HumanEvalResult",
    "LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_1",
    "LEGACY_CONTRACT_VERSION_EOS_TRUNCATE_2",
    "MMLU_LETTERS",
    "MMLUResult",
    "OUTCOME_GENERATION_ERROR",
    "SUMMARY_FILENAME",
    "ScoringConfig",
    "TASK_HUMANEVAL_X",
    "TASK_MMLU",
    "CheckpointSummary",
    "BatchCompletionGenerator",
    "TokenizerLike",
    "CompletionGenerator",
    "ValidatedHumanevalBody",
    "ValidatedMMLUBody",
    "build_mmlu_prompt",
    "evaluate_checkpoint",
    "evaluate_humaneval_item",
    "evaluate_mmlu_item",
    "generate_completion",
    "generate_completions_batch",
    "compare_singleton_and_batch_token_ids",
    "humaneval_item_filename",
    "identity_matches",
    "is_checkpoint_complete",
    "load_cached_item",
    "load_downstream_items",
    "mmlu_item_filename",
    "parse_humaneval_item",
    "parse_mmlu_item",
    "parse_mmlu_letter",
    "rebuild_checkpoint_summary",
    "score_humaneval_completion",
    "scratch_dir",
    "validate_humaneval_cached_body",
    "validate_mmlu_cached_body",
    "write_item_atomically",
]
