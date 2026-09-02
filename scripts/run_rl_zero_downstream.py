#!/usr/bin/env python3
"""Resumable 11-checkpoint downstream evaluation CLI for RL-Zero-Code.

Loads the base ``main`` checkpoint and the ten RL-Zero-Code step checkpoints
one at a time, runs **raw greedy** generation (no chat template, no system
prompt) over the 50 downstream HumanEval-X items (python + cpp) and the 50
downstream MMLU items, and writes **isolated per-checkpoint** results plus an
**aggregate summary** reporting coverage across the canonical 11 checkpoints.

Hard preflight gate
    The CLI refuses to run any model work unless the exact canonical
    preflight report produced by ``experiments/validate_rl_zero_downstream.py``
    validates the pinned 50 downstream HumanEval-X ids. The gate reuses
    :func:`experiments.validate_rl_zero_downstream.report_matches_ids`, which
    verifies -- in order -- that the report:

    1. parses,
    2. covers the exact ordered downstream id set (line-for-line),
    3. carries the pinned dataset + revision on every row,
    4. marks every python/cpp canonical outcome ``pass``, and
    5. whose SHA-256 of every freshly assembled program matches the stored
       hash (re-derived from the current dataset rows).

    The gate cannot be bypassed from this CLI: there is no ``--skip-gate``
    flag. Regenerate the report with ``experiments/validate_rl_zero_downstream.py``
    when it is missing or stale.

Protocol (enforced here and by :mod:`src.downstream_eval`)
    * **Raw direct tokenization.** No chat template; the prompt fed to the
      model is the verbatim HumanEval prompt or the built MMLU prompt.
    * **Deterministic greedy decoding.** ``do_sample=False``, ``num_beams=1``
      via :class:`src.downstream_eval.GreedyGenerator`.
    * **Same revision for model + tokenizer.** Each checkpoint loads its model
      and tokenizer from the same repo at the same revision, ``bfloat16``,
      ``device_map="auto"``. The generator resolves the input device from the
      loaded model's embedding layer so ``device_map="auto"`` split models
      receive ``input_ids`` on the correct device.
    * **Model unloaded between checkpoints.** After each checkpoint the model
      and tokenizer are deleted, ``gc.collect()`` runs, and the CUDA cache is
      emptied, so only one 7B model is resident at a time.
    * **Resume per item.** Every item is an identity-matched JSON file; a
      re-run skips generation and sandbox work for cached items. A fully
      cached checkpoint skips the model load entirely.
    * **Aggregate coverage.** A single atomic ``aggregate_summary.json``
      records which of the canonical 11 checkpoints have a valid per-checkpoint
      summary, regardless of which subset was selected for the current run.

Usage:
    uv run python experiments/run_rl_zero_downstream.py [OPTIONS]

Options:
    --results-root DIR     Root for per-checkpoint + aggregate outputs
                           (default: results/rl_zero_code_syntax/downstream)
    --report-path PATH     Preflight report path
                           (default: results/rl_zero_code_syntax/preflight/
                            humaneval-x-downstream.jsonl)
    --checkpoints SPEC     Comma-separated subset of the 11 checkpoints
                           (default: all 11, i.e. main + the ten RL steps)
    --timeout SECS         Per-program subprocess timeout (default: 10)
    --max-new-tokens-code  Max generated tokens for HumanEval-X (default: 512)
    --max-new-tokens-mmlu  Max generated tokens for MMLU (default: 32)
    --force                Regenerate every per-item result (ignore cache)
    --rebuild-summaries-only
                           CPU-only rebuild: revalidate every cached body
                           (recomputed hashes/parsing + pinned inputs) and
                           recount summaries without loading a model
    --rescore-cached       CPU-only rebuild that ALSO re-executes every cached
                           Python/C++ completion in the sandbox and fails on
                           outcome drift vs the stored label. Requires the
                           bwrap/g++ tool check + smoke. Compatible with
                           --rebuild-summaries-only.
    --skip-tool-check      Skip the bwrap/g++ presence check (testing only)
    --help                 Show this message and exit

This script never writes into any paired-concept or legacy results directory,
never executes model-generated or canonical code in the host Python process
(all HumanEval programs run through the injected :class:`SandboxRunner`), and
never strips ``<think>`` blocks or markdown fences from completions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Mapping,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    cast,
)

# Make ``src`` importable when run directly via ``python experiments/...``.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OLMO3_VARIANTS
from src.downstream_eval import (
    DEFAULT_MAX_NEW_TOKENS_CODE,
    DEFAULT_MAX_NEW_TOKENS_MMLU,
    DEFAULT_TIMEOUT_SECONDS as DE_DEFAULT_TIMEOUT,
    GENERATION_CONTRACT_VERSION,
    CheckpointSummary,
    CompletionGenerator,
    DownstreamIdentity,
    DownstreamMMLUItem,
    GreedyGenerator,
    ScoringConfig,
    SUMMARY_FILENAME,
    TASK_HUMANEVAL_X,
    TASK_MMLU,
    TokenizerLike,
    build_mmlu_prompt,
    evaluate_checkpoint,
    humaneval_item_filename,
    is_checkpoint_complete,
    load_cached_item,
    load_downstream_items,
    mmlu_item_filename,
    rebuild_checkpoint_summary,
    validate_humaneval_cached_body,
    validate_mmlu_cached_body,
    write_item_atomically,
)
from src.humaneval_x_validator import (
    SandboxRunner,
    check_sandbox_tools_available,
    sha256_hex,
)
from src.rl_zero_experiment import (
    BASE_MODEL_KEY,
    EXPERIMENT_CHECKPOINTS,
    RL_ZERO_CODE_RESULTS_ROOT,
    TARGET_MODEL_KEY,
    DOWNSTREAM_HUMANEVAL_X_ITEMS,
    DOWNSTREAM_MMLU_ITEMS,
    is_base_checkpoint,
    is_rl_checkpoint,
    load_downstream,
)

# Re-used preflight gate + id loader from the companion preflight CLI. These
# are imported (not duplicated) so the hard gate stays the single source of
# truth for "is the canonical report valid for the pinned downstream ids?".
from experiments.validate_rl_zero_downstream import (
    DEFAULT_REPORT_PATH,
    load_downstream_humaneval_ids,
    report_matches_ids,
)

if TYPE_CHECKING:
    import torch  # noqa: F401 (annotation-only; keeps the module cheap to import)


# =============================================================================
# Constants
# =============================================================================

#: Default root for per-checkpoint directories + the aggregate summary. Lives
#: under the isolated RL-Zero-Code syntax results root and is deliberately
#: distinct from the preflight report directory and from any paired-concept
#: results path.
DEFAULT_RESULTS_ROOT: str = os.path.join(RL_ZERO_CODE_RESULTS_ROOT, "downstream")

#: Aggregate summary filename written at ``<results-root>/aggregate_summary.json``.
AGGREGATE_FILENAME: str = "aggregate_summary.json"


# =============================================================================
# Typed model surface (annotation-only torch)
# =============================================================================


class _DeviceCarrier(Protocol):
    """Anything exposing a read-only ``.device`` (e.g. ``torch.Tensor``).

    Declared as a property so Protocol matching is covariant: a fake with
    ``device: str`` satisfies it just as well as ``torch.Tensor`` whose
    ``device`` is ``torch.device`` (both are subtypes of ``object``).
    """

    @property
    def device(self) -> object: ...


class _EmbeddingWithWeight(Protocol):
    """Embedding layer whose weight exposes a device (for input placement)."""

    @property
    def weight(self) -> _DeviceCarrier: ...


class _EmbeddingSource(Protocol):
    """Model surface needed only to resolve the input device."""

    def get_input_embeddings(self) -> _EmbeddingWithWeight: ...


class _ModelForDownstream(Protocol):
    """HuggingFace model surface needed for raw greedy downstream eval.

    Combines :class:`src.downstream_eval.CompletionGenerator`'s underlying
    ``generate`` (via :class:`GreedyGenerator`) with the embedding lookup used
    to resolve the input device for ``device_map="auto"`` models. The
    ``generate`` signature mirrors :class:`GreedyGenerator`'s
    ``_TensorGeneratingModel`` -- including the ``eos_token_id`` /
    ``pad_token_id`` kwargs that :class:`GreedyGenerator` forwards -- so the
    two protocols stay structurally compatible. Real ``transformers`` causal
    LMs satisfy this structurally; tests inject a fake loader returning a
    fake model.
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

    def get_input_embeddings(self) -> _EmbeddingWithWeight: ...


# A loaded (model, tokenizer) pair. ``Any`` is intentionally absent: the model
# is bounded by :class:`_ModelForDownstream` and the tokenizer by
# :class:`TokenizerLike` (re-exported from :mod:`src.downstream_eval`).
LoadedModel = tuple[_ModelForDownstream, TokenizerLike]


# =============================================================================
# Errors
# =============================================================================


class PreflightGateError(RuntimeError):
    """Raised when the downstream preflight report is missing or invalid.

    The CLI treats this as a hard, non-recoverable failure: no model work
    runs until the report is (re)generated by
    ``experiments/validate_rl_zero_downstream.py``.
    """


# =============================================================================
# Checkpoint identity: name -> (model_key, hf_id, revision)
# =============================================================================


def checkpoint_model_key(checkpoint: str) -> str:
    """Return the OLMO3_VARIANTS key for a checkpoint name.

    The base checkpoint (``main``) maps to :data:`BASE_MODEL_KEY`; every RL
    step maps to :data:`TARGET_MODEL_KEY`. The revision is always the
    checkpoint name itself (``main`` ships as the base repo's ``main``
    revision; RL steps ship as ``step_*`` revisions of the target repo).
    """
    if is_base_checkpoint(checkpoint):
        return BASE_MODEL_KEY
    if is_rl_checkpoint(checkpoint):
        return TARGET_MODEL_KEY
    raise ValueError(
        f"unknown checkpoint {checkpoint!r}; expected one of {EXPERIMENT_CHECKPOINTS}"
    )


def checkpoint_identity(checkpoint: str) -> tuple[str, str]:
    """Return ``(hf_id, revision)`` for the per-item downstream identity.

    The ``hf_id`` is the HuggingFace repo (so the identity distinguishes the
    base repo from the RL repo); the ``revision`` is the checkpoint name, so
    every RL step gets a distinct, traceable identity.
    """
    model_key = checkpoint_model_key(checkpoint)
    hf_id = OLMO3_VARIANTS[model_key].hf_id
    return hf_id, checkpoint


# =============================================================================
# Model loading + device resolution + unloading (real-run path)
# =============================================================================


def resolve_input_device(model: _EmbeddingSource) -> str:
    """Return the device where ``input_ids`` must live for ``device_map="auto"``.

    With ``device_map="auto"`` the model may be split across GPUs/CPU; the
    input embeddings always live on the first device, so the token-id tensor
    must be created there. Real ``transformers`` models expose
    ``get_input_embeddings().weight.device`` for exactly this purpose.
    """
    return str(model.get_input_embeddings().weight.device)


def load_model_and_tokenizer(model_key: str, revision: str) -> LoadedModel:
    """Load HF model (bfloat16, device_map="auto") + tokenizer at one revision.

    The model and tokenizer are loaded from the **same repo at the same
    revision** so a checkpoint and its tokenizer cannot drift apart.
    Mirrors the load pattern in :mod:`src.concept_dynamics` (bfloat16,
    ``device_map="auto"``, ``low_cpu_mem_usage=True``).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = OLMO3_VARIANTS[model_key].hf_id

    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Double cast through object: transformers' stubs bind generate() to a
    # narrower self type than our Protocol, so a direct cast is rejected for
    # insufficient overlap. The runtime object satisfies it structurally.
    return cast(_ModelForDownstream, cast(object, model)), tokenizer


def build_greedy_generator(
    model: _ModelForDownstream, tokenizer: TokenizerLike
) -> CompletionGenerator:
    """Build the default :class:`GreedyGenerator` with the resolved input device.

    The device is resolved from the loaded model's embedding layer so a
    ``device_map="auto"`` split model receives ``input_ids`` on the correct
    device. Tests inject a fake factory returning a fake generator.
    """
    device = resolve_input_device(model)
    return GreedyGenerator(model=model, tokenizer=tokenizer, device=device)


def unload_model(model: _ModelForDownstream, tokenizer: TokenizerLike) -> None:
    """Delete the model + tokenizer, collect garbage, and empty the CUDA cache.

    Called between checkpoints so only one 7B model is resident at a time.
    """
    del model
    del tokenizer
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Checkpoint-level resume: skip the model load when every item is cached
# =============================================================================


def checkpoint_complete(
    *,
    model: str,
    revision: str,
    downstream: Mapping[str, object],
    output_dir: Path,
    max_new_tokens_code: int = DEFAULT_MAX_NEW_TOKENS_CODE,
    max_new_tokens_mmlu: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    timeout: float = DE_DEFAULT_TIMEOUT,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
) -> bool:
    """True iff every expected per-item result is cached with a matching identity.

    Reuses :func:`load_downstream_items` to parse the downstream manifest and
    :func:`load_cached_item` to verify each item file exists *and* its
    identity matches this checkpoint's scoring config
    (``model``/``revision``/``task``/``language``/``prompt_sha256`` plus
    ``max_new_tokens``/``timeout``/``task_id``/``test_sha256``, with the
    narrow legacy adoption). Every identity-matched file is additionally
    body-validated (:func:`validate_humaneval_cached_body` /
    :func:`validate_mmlu_cached_body`) so a corrupt or stale body prevents
    the checkpoint-level resume from firing (forcing a model load and
    regeneration of the bad item). A single missing, identity-drifted, or
    body-corrupt file returns ``False``.
    """
    humaneval_items, mmlu_items = load_downstream_items(
        downstream,
        expected_humaneval=expected_humaneval,
        expected_mmlu=expected_mmlu,
    )
    for item in humaneval_items:
        for language, fields in (("python", item.python), ("cpp", item.cpp)):
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
            raw = load_cached_item(path, identity)
            if raw is None:
                return False
            try:
                validate_humaneval_cached_body(
                    raw,
                    expected_identity=identity,
                    expected_prompt=fields.prompt,
                    expected_test=fields.test,
                    expected_task_id=item.numeric_id,
                    expected_language=language,
                )
            except ValueError:
                return False
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
        raw = load_cached_item(path, identity)
        if raw is None:
            return False
        try:
            validate_mmlu_cached_body(
                raw, expected_identity=identity, expected_item=item
            )
        except ValueError:
            return False
    return True


def _summary_from_dict(raw: Mapping[str, object]) -> CheckpointSummary:
    """Reconstruct a :class:`CheckpointSummary` from its JSON dict.

    The ``scoring_config`` block is required: a summary written without one
    (pre-eos-truncate-2) cannot be validated against a current run and is
    treated as unparseable by the caller.
    """
    python_counts_raw = raw.get("python_counts")
    cpp_counts_raw = raw.get("cpp_counts")
    if not isinstance(python_counts_raw, Mapping) or not isinstance(
        cpp_counts_raw, Mapping
    ):
        raise ValueError("cached summary missing python_counts/cpp_counts")
    python_counts: dict[str, int] = {
        str(k): _count_value(v) for k, v in python_counts_raw.items()
    }
    cpp_counts: dict[str, int] = {
        str(k): _count_value(v) for k, v in cpp_counts_raw.items()
    }
    return CheckpointSummary(
        model=_field_str(raw, "model"),
        revision=_field_str(raw, "revision"),
        n_humaneval_python=_field_int(raw, "n_humaneval_python"),
        n_humaneval_cpp=_field_int(raw, "n_humaneval_cpp"),
        n_mmlu=_field_int(raw, "n_mmlu"),
        python_pass_at_1=_field_float(raw, "python_pass_at_1"),
        cpp_pass_at_1=_field_float(raw, "cpp_pass_at_1"),
        mmlu_accuracy=_field_float(raw, "mmlu_accuracy"),
        python_counts=python_counts,
        cpp_counts=cpp_counts,
        mmlu_correct=_field_int(raw, "mmlu_correct"),
        mmlu_parsed=_field_int(raw, "mmlu_parsed"),
        errors=_field_int(raw, "errors"),
        scoring_config=_scoring_config_from_dict(raw.get("scoring_config")),
    )


def _scoring_config_from_dict(raw: object) -> ScoringConfig:
    """Reconstruct a :class:`ScoringConfig` from its JSON dict.

    Raises ``ValueError`` when the block is missing or malformed so the
    caller (summary / aggregate loaders) treats the summary as invalid.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("cached summary missing scoring_config")
    return ScoringConfig(
        max_new_tokens_code=_field_int(raw, "max_new_tokens_code"),
        max_new_tokens_mmlu=_field_int(raw, "max_new_tokens_mmlu"),
        timeout=_field_float(raw, "timeout"),
        generation_contract_version=_field_str(raw, "generation_contract_version"),
    )


def _count_value(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"count value not an int: {value!r}")
    return int(value)


def _field_str(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"summary field {key!r} not a string: {value!r}")
    return value


def _field_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"summary field {key!r} not an int: {value!r}")
    return int(value)


def _field_float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"summary field {key!r} not a number: {value!r}")
    return float(value)


def load_cached_summary(
    checkpoint_dir: Path,
    model: str,
    revision: str,
    scoring_config: ScoringConfig | None = None,
) -> CheckpointSummary | None:
    """Return the cached per-checkpoint summary iff it matches this identity.

    Used by the checkpoint-level resume path: when every per-item file is
    already cached (:func:`checkpoint_complete``), the previously written
    ``summary.json`` is reused without reloading the model.

    When ``scoring_config`` is provided, the cached summary's own scoring
    config must match it exactly; a summary written under a different token
    budget, timeout, or contract is treated as stale and returns ``None`` so
    the caller regenerates it.
    """
    path = checkpoint_dir / SUMMARY_FILENAME
    if not path.exists():
        return None
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("model") != model or raw.get("revision") != revision:
        return None
    try:
        summary = _summary_from_dict(raw)
    except (KeyError, ValueError, TypeError):
        return None
    if scoring_config is not None and summary.scoring_config != scoring_config:
        return None
    return summary


# =============================================================================
# Per-checkpoint run
# =============================================================================


def run_checkpoint(
    checkpoint: str,
    *,
    downstream: Mapping[str, object],
    runner: SandboxRunner,
    results_root: Path,
    model_loader: Callable[[str, str], LoadedModel] = load_model_and_tokenizer,
    generator_factory: Callable[
        [_ModelForDownstream, TokenizerLike], CompletionGenerator
    ] = build_greedy_generator,
    max_new_tokens_code: int = DEFAULT_MAX_NEW_TOKENS_CODE,
    max_new_tokens_mmlu: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    timeout: float = DE_DEFAULT_TIMEOUT,
    force: bool = False,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
    progress: Callable[[str], None] | None = None,
) -> CheckpointSummary:
    """Run one checkpoint end-to-end: load, evaluate, persist, unload.

    Resume contract:

    * If ``force`` is ``False`` and every expected per-item result is already
      cached with a matching identity (:func:`checkpoint_complete``), the
      model is **not** loaded; the cached ``summary.json`` is reused.
    * Otherwise the model + tokenizer are loaded, :func:`evaluate_checkpoint`
      runs (it itself skips generation/sandbox work for cached items), and
      the model is unloaded in a ``finally`` so the next checkpoint starts
      from a clean GPU.
    """
    hf_id, revision = checkpoint_identity(checkpoint)
    checkpoint_dir = results_root / checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_scoring_config = ScoringConfig(
        max_new_tokens_code=max_new_tokens_code,
        max_new_tokens_mmlu=max_new_tokens_mmlu,
        timeout=timeout,
        generation_contract_version=GENERATION_CONTRACT_VERSION,
    )

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)

    # Checkpoint-level resume: avoid the (slow) model load when complete.
    if not force and checkpoint_complete(
        model=hf_id,
        revision=revision,
        downstream=downstream,
        output_dir=checkpoint_dir,
        max_new_tokens_code=max_new_tokens_code,
        max_new_tokens_mmlu=max_new_tokens_mmlu,
        timeout=timeout,
        expected_humaneval=expected_humaneval,
        expected_mmlu=expected_mmlu,
    ):
        cached = load_cached_summary(
            checkpoint_dir, hf_id, revision, run_scoring_config
        )
        if cached is not None:
            _log(f"[{checkpoint}] fully cached; skipping model load")
            return cached
        # Items complete but summary missing/corrupt: fall through and rebuild.

    model_key = checkpoint_model_key(checkpoint)
    _log(f"[{checkpoint}] loading {OLMO3_VARIANTS[model_key].hf_id} @ {revision}")
    model, tokenizer = model_loader(model_key, revision)
    try:
        generator = generator_factory(model, tokenizer)
        summary = evaluate_checkpoint(
            model=hf_id,
            revision=revision,
            downstream=downstream,
            tokenizer=tokenizer,
            generator=generator,
            runner=runner,
            output_dir=checkpoint_dir,
            max_new_tokens_code=max_new_tokens_code,
            max_new_tokens_mmlu=max_new_tokens_mmlu,
            timeout=timeout,
            force=force,
            expected_humaneval=expected_humaneval,
            expected_mmlu=expected_mmlu,
            progress=progress,
        )
    finally:
        unload_model(model, tokenizer)
    return summary


# =============================================================================
# Aggregate summary (atomic, 11-checkpoint coverage)
# =============================================================================


def _summary_matches_checkpoint(raw: Mapping[str, object], ckpt: str) -> bool:
    """True iff a summary's model+revision match the canonical mapping for ckpt.

    The revision of every checkpoint IS its name (``main`` -> ``main``,
    ``step_100`` -> ``step_100``), and the model is the HF repo for that
    branch (base repo for ``main``, RL-Zero-Code repo for every step). A
    summary whose model or revision disagrees -- e.g. a ``step_100``
    summary copied verbatim into the ``main`` directory -- is rejected so a
    misplaced or copied summary can never inflate coverage.
    """
    try:
        expected_model, expected_revision = checkpoint_identity(ckpt)
    except ValueError:
        return False
    return (
        raw.get("model") == expected_model and raw.get("revision") == expected_revision
    )


def build_aggregate(
    results_root: Path,
    *,
    expected_checkpoints: Sequence[str],
    selected_checkpoints: Sequence[str],
    scoring_config: ScoringConfig | None = None,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
) -> dict[str, object]:
    """Build the aggregate summary dict from the per-checkpoint summaries.

    Coverage is reported across the **canonical** checkpoints regardless of
    which subset was selected for the current run, so partial runs
    accumulate toward full coverage. ``selected_checkpoints`` records what
    ran this invocation.

    A per-checkpoint summary is **complete** (counted toward coverage and
    listed under ``checkpoints``) only if it parses, its model and revision
    match the canonical checkpoint mapping (:func:`checkpoint_identity`),
    its ``scoring_config`` matches (when provided), AND it passes
    :func:`is_checkpoint_complete` (``errors == 0`` plus exact expected
    Python/C++/MMLU item counts). Any checkpoint that exists but fails one
    of these checks is listed under ``incomplete`` with a reason instead of
    silently inflating coverage.

    ``n_present`` (alias of ``n_complete``) preserves backward
    compatibility with prior aggregate consumers.
    """
    complete: dict[str, object] = {}
    incomplete: dict[str, dict[str, object]] = {}

    for ckpt in expected_checkpoints:
        path = results_root / ckpt / SUMMARY_FILENAME
        if not path.exists():
            incomplete[ckpt] = {"reason": "missing"}
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            incomplete[ckpt] = {"reason": "corrupt"}
            continue
        if not isinstance(raw, dict):
            incomplete[ckpt] = {"reason": "corrupt"}
            continue
        if not _summary_matches_checkpoint(raw, ckpt):
            incomplete[ckpt] = {"reason": "model_or_revision_mismatch"}
            continue
        if scoring_config is not None:
            try:
                cfg = _scoring_config_from_dict(raw.get("scoring_config"))
            except (KeyError, ValueError, TypeError):
                incomplete[ckpt] = {"reason": "scoring_config_unparseable"}
                continue
            if cfg != scoring_config:
                incomplete[ckpt] = {"reason": "scoring_config_mismatch"}
                continue
        try:
            summary = _summary_from_dict(raw)
        except (KeyError, ValueError, TypeError):
            incomplete[ckpt] = {"reason": "summary_unparseable"}
            continue
        if is_checkpoint_complete(
            summary,
            expected_per_language=expected_humaneval,
            expected_mmlu=expected_mmlu,
        ):
            complete[ckpt] = raw
        else:
            incomplete[ckpt] = {
                "reason": "errors_or_counts",
                "errors": summary.errors,
                "n_humaneval_python": summary.n_humaneval_python,
                "n_humaneval_cpp": summary.n_humaneval_cpp,
                "n_mmlu": summary.n_mmlu,
                "mmlu_parsed": summary.mmlu_parsed,
            }

    aggregate: dict[str, object] = {
        "expected_checkpoints": list(expected_checkpoints),
        "n_expected": len(expected_checkpoints),
        "n_present": len(complete),
        "n_complete": len(complete),
        "n_incomplete": len(incomplete),
        "selected_checkpoints": list(selected_checkpoints),
        "checkpoints": complete,
        "incomplete": incomplete,
    }
    if scoring_config is not None:
        aggregate["scoring_config"] = dict(scoring_config.to_dict())
    return aggregate


def write_aggregate(
    results_root: Path,
    aggregate: Mapping[str, object],
) -> Path:
    """Atomically write the aggregate summary and return its path.

    Delegates to :func:`write_item_atomically` (temp file + ``os.replace``)
    so a crash midway through the write leaves any pre-existing aggregate
    untouched.
    """
    results_root.mkdir(parents=True, exist_ok=True)
    path = results_root / AGGREGATE_FILENAME
    write_item_atomically(path, aggregate)
    return path


# =============================================================================
# Checkpoint selection parsing
# =============================================================================


def parse_checkpoint_selection(
    spec: str | None,
    *,
    valid: Sequence[str] = EXPERIMENT_CHECKPOINTS,
) -> list[str]:
    """Parse a comma-separated checkpoint spec into an ordered, validated list.

    ``None`` or an empty string selects the full canonical schedule. Unknown
    names and duplicates are rejected with ``ValueError`` so a typo cannot
    silently skip a checkpoint or run it twice.
    """
    valid_set = set(valid)
    if spec is None or not spec.strip():
        return list(valid)
    requested = [token.strip() for token in spec.split(",") if token.strip()]
    unknown = [name for name in requested if name not in valid_set]
    if unknown:
        raise ValueError(f"unknown checkpoint(s) {unknown}; valid: {list(valid)}")
    if len(requested) != len(set(requested)):
        raise ValueError(f"duplicate checkpoint(s) in selection: {requested}")
    return requested


# =============================================================================
# Core orchestration (testable: inject gate, model_loader, runner)
# =============================================================================


@dataclass(frozen=True)
class DownstreamRunResult:
    """Outcome of a full downstream evaluation run.

    ``aggregate_path`` is the atomically-written aggregate summary path;
    ``aggregate`` is its dict; ``checkpoint_summaries`` maps each selected
    checkpoint to its :class:`CheckpointSummary`.
    """

    aggregate_path: Path
    aggregate: Mapping[str, object]
    checkpoint_summaries: Mapping[str, CheckpointSummary]


def run_downstream_eval(
    *,
    report_path: Path,
    results_root: Path,
    selected_checkpoints: Sequence[str],
    downstream: Mapping[str, object],
    humaneval_ids: Sequence[int],
    runner: SandboxRunner,
    gate: Callable[[Path, Sequence[int]], bool] = report_matches_ids,
    model_loader: Callable[[str, str], LoadedModel] = load_model_and_tokenizer,
    generator_factory: Callable[
        [_ModelForDownstream, TokenizerLike], CompletionGenerator
    ] = build_greedy_generator,
    max_new_tokens_code: int = DEFAULT_MAX_NEW_TOKENS_CODE,
    max_new_tokens_mmlu: int = DEFAULT_MAX_NEW_TOKENS_MMLU,
    timeout: float = DE_DEFAULT_TIMEOUT,
    force: bool = False,
    rebuild_summaries_only: bool = False,
    rescore_cached: bool = False,
    expected_humaneval: int = DOWNSTREAM_HUMANEVAL_X_ITEMS,
    expected_mmlu: int = DOWNSTREAM_MMLU_ITEMS,
    progress: Callable[[str], None] | None = None,
) -> DownstreamRunResult:
    """Run the resumable 11-checkpoint downstream evaluation.

    Hard gate: ``gate(report_path, humaneval_ids)`` MUST return ``True`` or a
    :class:`PreflightGateError` is raised before any model work. The default
    ``gate`` is :func:`report_matches_ids` so the exact canonical preflight
    report (IDs / revision / all-pass / hashes) is enforced.

    Each checkpoint in ``selected_checkpoints`` is run via
    :func:`run_checkpoint` (load, evaluate, unload). The aggregate summary is
    rebuilt from the per-checkpoint summaries and written atomically.

    When ``rebuild_summaries_only`` is ``True`` (or ``rescore_cached`` is
    ``True``, which implies the rebuild path), no model is loaded: each
    checkpoint's summary is rebuilt from its existing per-item JSON via
    :func:`src.downstream_eval.rebuild_checkpoint_summary`. The hard preflight
    gate still runs first. This is the CPU-only path for refreshing summaries
    (e.g. after a scoring-config change) without regenerating completions.

    When ``rescore_cached`` is ``True`` the rebuild additionally re-executes
    every cached Python/C++ completion in the sandbox through ``runner`` with
    the current timeout and raises on any outcome drift, so a stale or tampered
    outcome label cannot silently stand. The caller MUST have verified the
    sandbox tooling (bwrap/g++ + smoke) before enabling rescore.
    """
    if not gate(report_path, humaneval_ids):
        raise PreflightGateError(
            f"preflight report {report_path} is missing or invalid for the "
            f"{len(humaneval_ids)} downstream HumanEval-X ids; regenerate it "
            f"with experiments/validate_rl_zero_downstream.py"
        )

    results_root.mkdir(parents=True, exist_ok=True)

    run_scoring_config = ScoringConfig(
        max_new_tokens_code=max_new_tokens_code,
        max_new_tokens_mmlu=max_new_tokens_mmlu,
        timeout=timeout,
        generation_contract_version=GENERATION_CONTRACT_VERSION,
    )

    use_rebuild = rebuild_summaries_only or rescore_cached
    summaries: dict[str, CheckpointSummary] = {}
    for checkpoint in selected_checkpoints:
        if use_rebuild:
            hf_id, revision = checkpoint_identity(checkpoint)
            summary = rebuild_checkpoint_summary(
                model=hf_id,
                revision=revision,
                downstream=downstream,
                output_dir=results_root / checkpoint,
                max_new_tokens_code=max_new_tokens_code,
                max_new_tokens_mmlu=max_new_tokens_mmlu,
                timeout=timeout,
                expected_humaneval=expected_humaneval,
                expected_mmlu=expected_mmlu,
                rescore_cached=rescore_cached,
                runner=runner,
            )
        else:
            summary = run_checkpoint(
                checkpoint,
                downstream=downstream,
                runner=runner,
                results_root=results_root,
                model_loader=model_loader,
                generator_factory=generator_factory,
                max_new_tokens_code=max_new_tokens_code,
                max_new_tokens_mmlu=max_new_tokens_mmlu,
                timeout=timeout,
                force=force,
                expected_humaneval=expected_humaneval,
                expected_mmlu=expected_mmlu,
                progress=progress,
            )
        summaries[checkpoint] = summary

    aggregate = build_aggregate(
        results_root,
        expected_checkpoints=EXPERIMENT_CHECKPOINTS,
        selected_checkpoints=selected_checkpoints,
        scoring_config=run_scoring_config,
        expected_humaneval=expected_humaneval,
        expected_mmlu=expected_mmlu,
    )
    aggregate_path = write_aggregate(results_root, aggregate)
    return DownstreamRunResult(
        aggregate_path=aggregate_path,
        aggregate=aggregate,
        checkpoint_summaries=summaries,
    )


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable raw-greedy downstream eval (50 python + 50 cpp + "
            "50 MMLU) across the 11 RL-Zero-Code checkpoints (base main + "
            "ten RL steps). Refuses to run unless the canonical preflight "
            "report validates the pinned downstream ids."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=DEFAULT_RESULTS_ROOT,
        help=(
            "Root for per-checkpoint directories + aggregate summary.\n"
            f"(default: {DEFAULT_RESULTS_ROOT})"
        ),
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help=(
            "Preflight JSONL report path. The report MUST validate the\n"
            "pinned downstream ids or this CLI refuses to run.\n"
            f"(default: {DEFAULT_REPORT_PATH})"
        ),
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of the 11 checkpoints to run this\n"
            "invocation (default: all 11). Unknown names and duplicates\n"
            f"are rejected. Valid: {','.join(EXPERIMENT_CHECKPOINTS)}"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DE_DEFAULT_TIMEOUT,
        help=(
            "Per-program subprocess timeout in seconds, forwarded to the\n"
            f"sandbox runner. (default: {DE_DEFAULT_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--max-new-tokens-code",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS_CODE,
        help=(
            "Max generated tokens for HumanEval-X code completion.\n"
            f"(default: {DEFAULT_MAX_NEW_TOKENS_CODE})"
        ),
    )
    parser.add_argument(
        "--max-new-tokens-mmlu",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS_MMLU,
        help=(
            "Max generated tokens for an MMLU answer letter.\n"
            f"(default: {DEFAULT_MAX_NEW_TOKENS_MMLU})"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate every per-item result even if a valid cache exists.",
    )
    parser.add_argument(
        "--rebuild-summaries-only",
        action="store_true",
        help=(
            "Do not load any model or invoke the sandbox: rebuild every\n"
            "selected checkpoint's summary.json (and the aggregate) from\n"
            "the existing per-item JSON, validating bodies (recomputed\n"
            "hashes/parsing + pinned inputs) and recounting outcomes. The\n"
            "hard preflight gate still runs first."
        ),
    )
    parser.add_argument(
        "--rescore-cached",
        action="store_true",
        help=(
            "CPU-only rebuild that additionally re-executes every cached\n"
            "Python/C++ completion in the sandbox with the current timeout\n"
            "and FAILS on any outcome drift vs the stored label. Implies the\n"
            "rebuild path (no model load). Compatible with\n"
            "--rebuild-summaries-only. Requires bwrap/g++ + smoke (the tool\n"
            "check is NOT skippable under rescore, unlike a plain rebuild)."
        ),
    )
    parser.add_argument(
        "--skip-tool-check",
        action="store_true",
        help="Do not verify bwrap/g++ presence before running (testing only).",
    )
    return parser.parse_args(argv)


def _is_finite_positive_timeout(timeout: float) -> bool:
    """True iff ``timeout`` is a finite, strictly positive real number."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return False
    return math.isfinite(timeout) and timeout > 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not _is_finite_positive_timeout(args.timeout):
        print("ERROR: --timeout must be a finite positive number", file=sys.stderr)
        return 2
    if args.max_new_tokens_code <= 0 or args.max_new_tokens_mmlu <= 0:
        print(
            "ERROR: --max-new-tokens-* must be positive integers",
            file=sys.stderr,
        )
        return 2

    try:
        selected = parse_checkpoint_selection(args.checkpoints)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report_path = Path(args.report_path)
    results_root = Path(args.results_root)

    print("=" * 60)
    print("RL-Zero-Code downstream eval (raw greedy, 11 checkpoints)")
    print("=" * 60)
    print(f"  Checkpoints:   {len(selected)} -> {selected}")
    print(
        f"  Items/ckpt:    {DOWNSTREAM_HUMANEVAL_X_ITEMS} python + "
        f"{DOWNSTREAM_HUMANEVAL_X_ITEMS} cpp + {DOWNSTREAM_MMLU_ITEMS} MMLU"
    )
    print(f"  Preflight:     {report_path}")
    print(f"  Results root:  {results_root}")
    print(f"  Force rerun:   {bool(args.force)}")
    print(f"  Rebuild only:  {bool(args.rebuild_summaries_only)}")
    print(f"  Rescore cache: {bool(args.rescore_cached)}")
    print()

    try:
        downstream = load_downstream()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        humaneval_ids = load_downstream_humaneval_ids()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # ---- HARD PREFLIGHT GATE -------------------------------------------------
    # Enforced before any model work. report_matches_ids re-derives the
    # SHA-256 of every canonical program from the current dataset rows, so a
    # stale, partial, reordered, non-pass, or hash-drifted report is refused.
    print(f"Checking preflight gate ({len(humaneval_ids)} downstream ids)...")
    try:
        gate_ok = report_matches_ids(report_path, humaneval_ids)
    except Exception as exc:  # noqa: BLE001 - any gate failure is fatal here
        print(f"ERROR: preflight gate raised: {exc}", file=sys.stderr)
        return 1
    if not gate_ok:
        print(
            f"REFUSE: preflight report {report_path} is missing or invalid.\n"
            "        Regenerate it with:\n"
            "          uv run python experiments/validate_rl_zero_downstream.py",
            file=sys.stderr,
        )
        return 1
    print("OK: preflight report valid; proceeding to model evaluation.")
    print()

    # ---- Per-checkpoint run --------------------------------------------------
    # The sandbox is invoked under --rescore-cached (it re-executes cached
    # completions) and under the normal model path. A plain
    # --rebuild-summaries-only never touches the sandbox, so the tool check is
    # skipped there; --rescore-cached requires bwrap/g++ + smoke and is NOT
    # skippable via --skip-tool-check.
    rescore = bool(args.rescore_cached)
    plain_rebuild = args.rebuild_summaries_only and not rescore
    requires_tools = rescore or (not plain_rebuild and not args.skip_tool_check)
    from src.humaneval_x_validator import BwrapRunner

    if requires_tools:
        try:
            check_sandbox_tools_available()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    runner: SandboxRunner = BwrapRunner()

    def _cli_progress(msg: str) -> None:
        print(msg)

    try:
        result = run_downstream_eval(
            report_path=report_path,
            results_root=results_root,
            selected_checkpoints=selected,
            downstream=downstream,
            humaneval_ids=humaneval_ids,
            runner=runner,
            progress=_cli_progress,
            max_new_tokens_code=args.max_new_tokens_code,
            max_new_tokens_mmlu=args.max_new_tokens_mmlu,
            timeout=args.timeout,
            force=args.force,
            rebuild_summaries_only=args.rebuild_summaries_only,
            rescore_cached=rescore,
        )
    except PreflightGateError as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1

    n_present = _field_int(result.aggregate, "n_present")
    n_expected = _field_int(result.aggregate, "n_expected")
    n_incomplete_raw = result.aggregate.get("n_incomplete", 0)
    if isinstance(n_incomplete_raw, bool) or not isinstance(n_incomplete_raw, int):
        n_incomplete = 0
    else:
        n_incomplete = int(n_incomplete_raw)
    print()
    print(
        f"DONE: ran {len(selected)} checkpoint(s); "
        f"aggregate coverage {n_present}/{n_expected}."
    )
    if n_incomplete > 0:
        print(f"  Incomplete checkpoints ({n_incomplete}):")
        incomplete = result.aggregate.get("incomplete")
        if isinstance(incomplete, dict):
            for ckpt_name, info in sorted(incomplete.items()):
                reason = (
                    info.get("reason", "unknown")
                    if isinstance(info, dict)
                    else "unknown"
                )
                print(f"    {ckpt_name}: {reason}")
    print(f"Aggregate: {result.aggregate_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
