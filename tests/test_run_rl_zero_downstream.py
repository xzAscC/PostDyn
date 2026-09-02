"""Tests for scripts/run_rl_zero_downstream.py.

No real model, no real benchmark code, no host execution. Every test injects
a fake model loader, a fake generator factory, a fake sandbox runner, and a
fake gate so the preflight check, 11-checkpoint orchestration, resume,
isolation, and aggregate coverage are all exercised without torch/CUDA/bwrap.

Coverage matrix:
  * Checkpoint identity -- base ``main`` -> olmo3-base, RL steps -> target.
  * Checkpoint selection -- defaults to all 11, subset honored, unknown and
    duplicate names rejected.
  * Preflight hard gate -- refuses (PreflightGateError) when the gate returns
    False (missing/stale report); proceeds when True; no model loaded on fail.
  * Orchestration -- model loaded per checkpoint with the correct
    (model_key, revision); per-checkpoint isolated output dirs; aggregate
    reports 11-checkpoint coverage; raw greedy kwargs forwarded
    (do_sample=False, num_beams=1); a fresh model object per checkpoint.
  * Resume -- checkpoint-level skip (no model load) when every item is cached;
    partial cache regenerates only missing items; --force regenerates all.
  * Output isolation -- two checkpoints never share item files; aggregate is
    atomic; partial runs accumulate coverage.
  * CLI wiring -- parse_args defaults/overrides; main() refuses (rc=1) when
    the gate fails or raises; bad selection / timeout / token-budget rejected.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence, cast

import pytest

import scripts.run_rl_zero_downstream as cli
from scripts.run_rl_zero_downstream import (
    AGGREGATE_FILENAME,
    DEFAULT_MAX_NEW_TOKENS_CODE,
    DEFAULT_MAX_NEW_TOKENS_MMLU,
    DEFAULT_REPORT_PATH,
    DEFAULT_RESULTS_ROOT,
    DE_DEFAULT_TIMEOUT,
    DownstreamRunResult,
    LoadedModel,
    PreflightGateError,
    SUMMARY_FILENAME,
    build_aggregate,
    checkpoint_complete,
    checkpoint_identity,
    checkpoint_model_key,
    load_cached_summary,
    main,
    parse_args,
    parse_checkpoint_selection,
    resolve_input_device,
    run_downstream_eval,
)
from postdyn.config import OLMO3_VARIANTS
from postdyn.downstream_eval import (
    CheckpointSummary,
    DEFAULT_MAX_NEW_TOKENS_CODE as DE_CODE,
    DEFAULT_MAX_NEW_TOKENS_MMLU as DE_MMLU,
    DEFAULT_TIMEOUT_SECONDS as DE_TIMEOUT,
    GENERATION_CONTRACT_VERSION,
    MMLU_LETTERS,
    ScoringConfig,
    humaneval_item_filename,
    mmlu_item_filename,
    sha256_hex,
    write_item_atomically,
)
from postdyn.humaneval_x_validator import assemble_python_program
from postdyn.rl_zero_experiment import (
    BASE_CHECKPOINT,
    BASE_MODEL_KEY,
    EXPERIMENT_CHECKPOINTS,
    RL_CHECKPOINTS,
    TARGET_MODEL_KEY,
)


# =============================================================================
# Fakes & helpers
# =============================================================================


class _FakeTensor:
    """Stand-in for a torch tensor exposing only ``.device``."""

    def __init__(self, device: object = "cpu") -> None:
        self.device = device


class _FakeEmbedding:
    def __init__(self, device: str = "cpu") -> None:
        self.weight = _FakeTensor(device)


class _FakeModel:
    """Minimal model surface: get_input_embeddings + a generate stub.

    ``generate`` is never called when a fake ``generator_factory`` is injected
    (every orchestration test injects one); it exists only so the object can
    be cast across the typed ``LoadedModel`` boundary.
    """

    def __init__(self, device: str = "cpu") -> None:
        self._embeds = _FakeEmbedding(device)

    def get_input_embeddings(self) -> _FakeEmbedding:
        return self._embeds


class _FakeTokenizer:
    """Encodes one stable id per char; decode returns a fixed completion.

    ``eos_token_id`` / ``pad_token_id`` are ``None`` so the EOS-truncation
    path in ``generate_completion`` is skipped (existing tests rely on the
    full generated slice reaching decode unchanged).
    """

    def __init__(self, decoded: str = "A") -> None:
        self.decoded = decoded
        self.eos_token_id: int | None = None
        self.pad_token_id: int | None = None

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 1000 for ch in text]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self.decoded


class _RecordingGenerator:
    """CompletionGenerator fake: returns input_ids + extra, records kwargs."""

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
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "num_beams": num_beams,
            }
        )
        return list(input_ids) + self.extra_ids


class _PassRunner:
    """SandboxRunner fake: every invocation returns rc=0 (pass)."""

    def __init__(self) -> None:
        self.call_count = 0

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        self.call_count += 1
        return subprocess.CompletedProcess(
            args=list(command), returncode=0, stdout="", stderr=""
        )


class _RecordingModelLoader:
    """Records (model_key, revision) per call; returns fresh fakes each time.

    ``decoded`` controls what the fake tokenizer's ``decode`` returns -- the
    raw completion stored per-item comes from ``tokenizer.decode``, not from
    the generator, so this is the seam that sets the canned completion text.
    """

    def __init__(self, decoded: str = "A") -> None:
        self.calls: list[tuple[str, str]] = []
        self.models: list[_FakeModel] = []
        self.decoded = decoded

    def __call__(self, model_key: str, revision: str) -> LoadedModel:
        self.calls.append((model_key, revision))
        model = _FakeModel()
        self.models.append(model)
        # Test double crossing the typed LoadedModel boundary: the real model
        # satisfies _ModelForDownstream structurally; the fake's generate is
        # never invoked because every test injects a fake generator_factory.
        return cast(LoadedModel, cast(object, (model, _FakeTokenizer(self.decoded))))


def _generator_factory() -> tuple[
    list[_RecordingGenerator], Callable[[object, object], _RecordingGenerator]
]:
    """Return (generators list, factory) so tests can inspect greedy kwargs.

    The factory's parameters are typed ``object`` (a supertype of the declared
    ``_ModelForDownstream`` / ``TokenizerLike``), so by parameter
    contravariance the factory satisfies the injected ``generator_factory``
    signature without a cast.
    """
    generators: list[_RecordingGenerator] = []

    def factory(model: object, tokenizer: object) -> _RecordingGenerator:
        gen = _RecordingGenerator([1])
        generators.append(gen)
        return gen

    return generators, factory


def _passing_gate() -> Callable[[Path, Sequence[int]], bool]:
    def gate(report_path: Path, ids: Sequence[int]) -> bool:
        return True

    return gate


def _failing_gate() -> Callable[[Path, Sequence[int]], bool]:
    def gate(report_path: Path, ids: Sequence[int]) -> bool:
        return False

    return gate


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


def _make_downstream(n_he: int = 2, n_mmlu: int = 2) -> dict[str, object]:
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


def _run_eval(
    tmp_path: Path,
    *,
    selected: Sequence[str] = (BASE_CHECKPOINT,),
    n_he: int = 2,
    n_mmlu: int = 2,
    gate: Callable[[Path, Sequence[int]], bool] | None = None,
    loader: _RecordingModelLoader | None = None,
    decoded_completion: str = "A",
    force: bool = False,
) -> tuple[DownstreamRunResult, _RecordingModelLoader, list[_RecordingGenerator]]:
    """Centralized wrapper: builds fakes, runs eval, returns all records."""
    loader = loader if loader is not None else _RecordingModelLoader(decoded_completion)
    gens, factory = _generator_factory()
    result = run_downstream_eval(
        report_path=tmp_path / "ok.jsonl",
        results_root=tmp_path / "out",
        selected_checkpoints=list(selected),
        downstream=_make_downstream(n_he, n_mmlu),
        humaneval_ids=list(range(1, n_he + 1)),
        runner=_PassRunner(),
        gate=gate if gate is not None else _passing_gate(),
        model_loader=loader,
        generator_factory=factory,
        expected_humaneval=n_he,
        expected_mmlu=n_mmlu,
        force=force,
    )
    return result, loader, gens


# =============================================================================
# Checkpoint identity
# =============================================================================


class TestCheckpointIdentity:
    def test_base_main_maps_to_base_model_key(self):
        assert checkpoint_model_key(BASE_CHECKPOINT) == BASE_MODEL_KEY

    def test_each_rl_step_maps_to_target_model_key(self):
        for ckpt in RL_CHECKPOINTS:
            assert checkpoint_model_key(ckpt) == TARGET_MODEL_KEY

    def test_base_identity_uses_base_hf_id_and_main_revision(self):
        hf_id, revision = checkpoint_identity(BASE_CHECKPOINT)
        assert hf_id == OLMO3_VARIANTS[BASE_MODEL_KEY].hf_id
        assert revision == BASE_CHECKPOINT

    def test_rl_identity_uses_target_hf_id_and_step_revision(self):
        hf_id, revision = checkpoint_identity(RL_CHECKPOINTS[0])
        assert hf_id == OLMO3_VARIANTS[TARGET_MODEL_KEY].hf_id
        assert revision == RL_CHECKPOINTS[0]

    def test_unknown_checkpoint_rejected(self):
        with pytest.raises(ValueError, match="unknown checkpoint"):
            checkpoint_model_key("not_a_checkpoint")


# =============================================================================
# Checkpoint selection parsing
# =============================================================================


class TestParseCheckpointSelection:
    def test_none_selects_all_eleven_in_order(self):
        selected = parse_checkpoint_selection(None)
        assert selected == EXPERIMENT_CHECKPOINTS
        assert len(selected) == 11

    def test_empty_string_selects_all_eleven(self):
        assert parse_checkpoint_selection("") == EXPERIMENT_CHECKPOINTS
        assert parse_checkpoint_selection("  ") == EXPERIMENT_CHECKPOINTS

    def test_explicit_subset_preserves_caller_order(self):
        valid = ["main", "step_100", "step_2900"]
        selected = parse_checkpoint_selection("step_2900,main", valid=valid)
        assert selected == ["step_2900", "main"]

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="unknown checkpoint"):
            parse_checkpoint_selection("main,bogus", valid=EXPERIMENT_CHECKPOINTS)

    def test_duplicates_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            parse_checkpoint_selection("main,main", valid=EXPERIMENT_CHECKPOINTS)

    def test_whitespace_tolerated(self):
        selected = parse_checkpoint_selection(
            " main , step_100 ", valid=["main", "step_100"]
        )
        assert selected == ["main", "step_100"]


# =============================================================================
# resolve_input_device (device_map="auto" input placement)
# =============================================================================


class TestResolveInputDevice:
    def test_returns_embedding_weight_device_string(self):
        model = _FakeModel(device="cuda:0")
        assert resolve_input_device(model) == "cuda:0"

    def test_cpu_device(self):
        model = _FakeModel(device="cpu")
        assert resolve_input_device(model) == "cpu"


# =============================================================================
# Preflight hard gate
# =============================================================================


class TestPreflightGate:
    def test_refuses_when_gate_returns_false(self, tmp_path):
        with pytest.raises(PreflightGateError, match="preflight report"):
            _run_eval(tmp_path, gate=_failing_gate(), n_he=1, n_mmlu=1)

    def test_no_model_loaded_when_gate_fails(self, tmp_path):
        loader = _RecordingModelLoader()
        with pytest.raises(PreflightGateError):
            _run_eval(tmp_path, gate=_failing_gate(), n_he=1, n_mmlu=1, loader=loader)
        # Hard gate: the model loader must never be called.
        assert loader.calls == []

    def test_proceeds_when_gate_returns_true(self, tmp_path):
        result, _, _ = _run_eval(tmp_path, gate=_passing_gate(), n_he=1, n_mmlu=1)
        assert isinstance(result, DownstreamRunResult)
        assert BASE_CHECKPOINT in result.checkpoint_summaries

    def test_default_gate_is_report_matches_ids(self):
        # The default gate callable must be the canonical matcher imported
        # from the preflight CLI (exact IDs / revision / all-pass / hashes).
        from scripts.validate_rl_zero_downstream import report_matches_ids

        # run_downstream_eval is keyword-only (* prefix), so its defaults
        # live in __kwdefaults__, not __defaults__.
        kwdefs = cli.run_downstream_eval.__kwdefaults__
        assert kwdefs is not None
        assert kwdefs["gate"] is report_matches_ids


# =============================================================================
# Orchestration: model loaded per checkpoint, correct (key, revision)
# =============================================================================


class TestModelLoadingPerCheckpoint:
    def test_model_loaded_once_per_selected_checkpoint(self, tmp_path):
        selected = [BASE_CHECKPOINT, RL_CHECKPOINTS[0]]
        _, loader, _ = _run_eval(tmp_path, selected=selected, n_he=1, n_mmlu=1)
        assert len(loader.calls) == 2
        assert loader.calls[0] == (BASE_MODEL_KEY, BASE_CHECKPOINT)
        assert loader.calls[1] == (TARGET_MODEL_KEY, RL_CHECKPOINTS[0])

    def test_model_key_revision_match_checkpoint_identity(self, tmp_path):
        _, loader, _ = _run_eval(
            tmp_path, selected=[RL_CHECKPOINTS[3]], n_he=1, n_mmlu=1
        )
        _, expected_rev = checkpoint_identity(RL_CHECKPOINTS[3])
        assert loader.calls[0][0] == TARGET_MODEL_KEY
        assert loader.calls[0][1] == expected_rev


class TestUnloadBetweenCheckpoints:
    def test_each_checkpoint_gets_a_fresh_model_object(self, tmp_path):
        selected = [BASE_CHECKPOINT, RL_CHECKPOINTS[0], RL_CHECKPOINTS[1]]
        _, loader, _ = _run_eval(tmp_path, selected=selected, n_he=1, n_mmlu=1)
        # 3 distinct model objects (one per checkpoint); no reuse.
        assert len(loader.models) == 3
        assert len({id(m) for m in loader.models}) == 3


# =============================================================================
# Output isolation + aggregate coverage
# =============================================================================


class TestOutputIsolation:
    def test_each_checkpoint_writes_to_its_own_subdir(self, tmp_path):
        selected = [BASE_CHECKPOINT, RL_CHECKPOINTS[0]]
        _run_eval(tmp_path, selected=selected, n_he=2, n_mmlu=2)
        base_dir = tmp_path / "out" / BASE_CHECKPOINT
        rl_dir = tmp_path / "out" / RL_CHECKPOINTS[0]
        assert base_dir.exists() and rl_dir.exists()
        assert (base_dir / SUMMARY_FILENAME).exists()
        assert (rl_dir / SUMMARY_FILENAME).exists()
        # Identical item filenames live under separate parents (no bleed).
        assert (base_dir / humaneval_item_filename("python", 0)).exists()
        assert (rl_dir / humaneval_item_filename("python", 0)).exists()

    def test_two_checkpoints_item_files_have_distinct_identities(self, tmp_path):
        selected = [BASE_CHECKPOINT, RL_CHECKPOINTS[0]]
        _run_eval(tmp_path, selected=selected, n_he=1, n_mmlu=1)
        base_item = json.loads(
            (
                tmp_path
                / "out"
                / BASE_CHECKPOINT
                / humaneval_item_filename("python", 0)
            ).read_text(encoding="utf-8")
        )
        rl_item = json.loads(
            (
                tmp_path
                / "out"
                / RL_CHECKPOINTS[0]
                / humaneval_item_filename("python", 0)
            ).read_text(encoding="utf-8")
        )
        # Same prompt -> same prompt_sha256, but model/revision differ.
        assert base_item["identity"]["model"] != rl_item["identity"]["model"]
        assert base_item["identity"]["revision"] != rl_item["identity"]["revision"]


class TestAggregateCoverage:
    def test_aggregate_reports_eleven_expected_regardless_of_selection(self, tmp_path):
        result, _, _ = _run_eval(tmp_path, selected=[BASE_CHECKPOINT], n_he=1, n_mmlu=1)
        assert result.aggregate["n_expected"] == 11
        assert result.aggregate["n_present"] == 1
        assert result.aggregate["expected_checkpoints"] == EXPERIMENT_CHECKPOINTS
        assert result.aggregate["selected_checkpoints"] == [BASE_CHECKPOINT]

    def test_aggregate_written_atomically_to_root(self, tmp_path):
        _run_eval(tmp_path, selected=[BASE_CHECKPOINT], n_he=1, n_mmlu=1)
        path = tmp_path / "out" / AGGREGATE_FILENAME
        assert path.exists()
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        assert aggregate["n_expected"] == 11
        assert BASE_CHECKPOINT in aggregate["checkpoints"]
        assert not list((tmp_path / "out").glob(f".{AGGREGATE_FILENAME}.*.tmp"))

    def test_partial_runs_accumulate_coverage(self, tmp_path):
        _run_eval(tmp_path, selected=[BASE_CHECKPOINT], n_he=1, n_mmlu=1)
        result, _, _ = _run_eval(
            tmp_path, selected=[RL_CHECKPOINTS[0]], n_he=1, n_mmlu=1
        )
        assert result.aggregate["n_present"] == 2
        checkpoints = result.aggregate["checkpoints"]
        assert isinstance(checkpoints, dict)
        assert BASE_CHECKPOINT in checkpoints
        assert RL_CHECKPOINTS[0] in checkpoints

    def test_build_aggregate_skips_corrupt_summary(self, tmp_path):
        root = tmp_path / "out"
        (root / BASE_CHECKPOINT).mkdir(parents=True)
        (root / BASE_CHECKPOINT / SUMMARY_FILENAME).write_text(
            "{not json", encoding="utf-8"
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
        )
        assert agg["n_present"] == 0


# =============================================================================
# Raw greedy + completion preservation
# =============================================================================


class TestRawGreedyPreservation:
    def test_generator_receives_do_sample_false_and_num_beams_one(self, tmp_path):
        _, _, gens = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        # 2 python + 2 cpp + 2 mmlu = 6 generation calls.
        assert len(gens) == 1
        assert len(gens[0].calls) == 6
        for call in gens[0].calls:
            assert call["do_sample"] is False
            assert call["num_beams"] == 1

    def test_completion_preserved_verbatim_in_item_json(self, tmp_path):
        _run_eval(
            tmp_path,
            n_he=1,
            n_mmlu=0,
            decoded_completion="    return 42\n",
        )
        item = json.loads(
            (
                tmp_path
                / "out"
                / BASE_CHECKPOINT
                / humaneval_item_filename("python", 0)
            ).read_text(encoding="utf-8")
        )
        # The raw model completion is stored unchanged.
        assert item["completion"] == "    return 42\n"

    def test_assembled_program_uses_completion_not_canonical(self, tmp_path):
        _run_eval(
            tmp_path,
            n_he=1,
            n_mmlu=0,
            decoded_completion="MODEL_ANSWER\n",
        )
        item = json.loads(
            (
                tmp_path
                / "out"
                / BASE_CHECKPOINT
                / humaneval_item_filename("python", 0)
            ).read_text(encoding="utf-8")
        )
        prompt = item["prompt"]
        test = "assert f0(0)==0\n"
        expected_assembled = assemble_python_program(prompt, "MODEL_ANSWER\n", test)
        assert item["assembled_sha256"] == sha256_hex(expected_assembled)


# =============================================================================
# Resume: checkpoint-level skip + per-item resume + force
# =============================================================================


class TestResume:
    def test_checkpoint_level_skip_avoids_model_load_when_complete(self, tmp_path):
        _, loader1, gens1 = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        assert len(loader1.calls) == 1
        assert len(gens1[0].calls) > 0

        # Second run: everything cached -> no model load, no generation.
        loader2 = _RecordingModelLoader()
        _, loader2, gens2 = _run_eval(tmp_path, n_he=2, n_mmlu=2, loader=loader2)
        assert loader2.calls == []
        assert all(g.calls == [] for g in gens2)

    def test_force_regenerates_everything(self, tmp_path):
        _, _, gens1 = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        gen_calls_first = len(gens1[0].calls)

        loader2 = _RecordingModelLoader()
        _, loader2, gens2 = _run_eval(
            tmp_path, n_he=2, n_mmlu=2, loader=loader2, force=True
        )
        # force=True bypasses checkpoint-level skip -> model loads.
        assert len(loader2.calls) == 1
        # And per-item cache is ignored -> every item regenerates.
        assert len(gens2[0].calls) == gen_calls_first

    def test_partial_cache_regenerates_missing_items_only(self, tmp_path):
        _run_eval(tmp_path, n_he=2, n_mmlu=2)
        # Delete one MMLU item so the checkpoint is no longer complete.
        (tmp_path / "out" / BASE_CHECKPOINT / mmlu_item_filename(0)).unlink()

        loader2 = _RecordingModelLoader()
        _, loader2, gens2 = _run_eval(tmp_path, n_he=2, n_mmlu=2, loader=loader2)
        # Cache incomplete -> model loads.
        assert len(loader2.calls) == 1
        # Only the deleted MMLU item regenerated; the rest were cached.
        assert len(gens2[0].calls) == 1


# =============================================================================
# checkpoint_complete + load_cached_summary unit tests
# =============================================================================


class TestCheckpointComplete:
    def test_true_when_all_items_cached_with_matching_identity(self, tmp_path):
        _run_eval(tmp_path, n_he=2, n_mmlu=2)
        hf_id, revision = checkpoint_identity(BASE_CHECKPOINT)
        assert (
            checkpoint_complete(
                model=hf_id,
                revision=revision,
                downstream=_make_downstream(2, 2),
                output_dir=tmp_path / "out" / BASE_CHECKPOINT,
                expected_humaneval=2,
                expected_mmlu=2,
            )
            is True
        )

    def test_false_when_an_item_is_missing(self, tmp_path):
        _run_eval(tmp_path, n_he=2, n_mmlu=2)
        (
            tmp_path / "out" / BASE_CHECKPOINT / humaneval_item_filename("python", 0)
        ).unlink()
        hf_id, revision = checkpoint_identity(BASE_CHECKPOINT)
        assert (
            checkpoint_complete(
                model=hf_id,
                revision=revision,
                downstream=_make_downstream(2, 2),
                output_dir=tmp_path / "out" / BASE_CHECKPOINT,
                expected_humaneval=2,
                expected_mmlu=2,
            )
            is False
        )

    def test_false_when_identity_drifts(self, tmp_path):
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        _, revision = checkpoint_identity(BASE_CHECKPOINT)
        assert (
            checkpoint_complete(
                model="wrong/repo",
                revision=revision,
                downstream=_make_downstream(1, 1),
                output_dir=tmp_path / "out" / BASE_CHECKPOINT,
                expected_humaneval=1,
                expected_mmlu=1,
            )
            is False
        )


class TestLoadCachedSummary:
    def test_returns_none_when_summary_missing(self, tmp_path):
        assert load_cached_summary(tmp_path, "m", "r") is None

    def test_returns_none_when_model_revision_mismatch(self, tmp_path):
        summary = CheckpointSummary(
            model="other",
            revision="other",
            n_humaneval_python=1,
            n_humaneval_cpp=1,
            n_mmlu=1,
            python_pass_at_1=1.0,
            cpp_pass_at_1=1.0,
            mmlu_accuracy=1.0,
            python_counts={"pass": 1},
            cpp_counts={"pass": 1},
            mmlu_correct=1,
            mmlu_parsed=1,
            errors=0,
            scoring_config=_expected_scoring_config(),
        )
        write_item_atomically(tmp_path / SUMMARY_FILENAME, summary.to_dict())
        assert load_cached_summary(tmp_path, "m", "r") is None

    def test_round_trips_a_valid_summary(self, tmp_path):
        summary = CheckpointSummary(
            model="allenai/Olmo-3-1025-7B",
            revision="main",
            n_humaneval_python=50,
            n_humaneval_cpp=50,
            n_mmlu=50,
            python_pass_at_1=0.5,
            cpp_pass_at_1=0.3,
            mmlu_accuracy=0.25,
            python_counts={"pass": 25, "fail": 25},
            cpp_counts={"pass": 15, "compile_error": 35},
            mmlu_correct=12,
            mmlu_parsed=40,
            errors=0,
            scoring_config=_expected_scoring_config(),
        )
        write_item_atomically(tmp_path / SUMMARY_FILENAME, summary.to_dict())
        loaded = load_cached_summary(tmp_path, "allenai/Olmo-3-1025-7B", "main")
        assert loaded is not None
        assert loaded.python_pass_at_1 == 0.5
        assert loaded.mmlu_correct == 12
        assert loaded.python_counts["pass"] == 25


# =============================================================================
# CLI: parse_args + main gate refusal
# =============================================================================


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.results_root == DEFAULT_RESULTS_ROOT
        assert args.report_path == DEFAULT_REPORT_PATH
        assert args.checkpoints is None
        assert args.timeout == DE_DEFAULT_TIMEOUT
        assert args.max_new_tokens_code == DEFAULT_MAX_NEW_TOKENS_CODE
        assert args.max_new_tokens_mmlu == DEFAULT_MAX_NEW_TOKENS_MMLU
        assert args.force is False
        assert args.skip_tool_check is False

    def test_explicit_overrides(self):
        args = parse_args(
            [
                "--results-root",
                "/tmp/r",
                "--report-path",
                "/tmp/r.jsonl",
                "--checkpoints",
                "main,step_100",
                "--timeout",
                "5.5",
                "--max-new-tokens-code",
                "256",
                "--max-new-tokens-mmlu",
                "16",
                "--force",
                "--skip-tool-check",
            ]
        )
        assert args.results_root == "/tmp/r"
        assert args.report_path == "/tmp/r.jsonl"
        assert args.checkpoints == "main,step_100"
        assert args.timeout == 5.5
        assert args.max_new_tokens_code == 256
        assert args.max_new_tokens_mmlu == 16
        assert args.force is True
        assert args.skip_tool_check is True


class TestMainGateRefusal:
    def test_refuses_with_rc1_when_gate_fails(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_downstream", lambda: _make_downstream(1, 1))
        monkeypatch.setattr(cli, "load_downstream_humaneval_ids", lambda: [1])
        monkeypatch.setattr(cli, "report_matches_ids", lambda *a, **k: False)
        rc = main(
            [
                "--results-root",
                str(tmp_path / "out"),
                "--report-path",
                str(tmp_path / "absent.jsonl"),
                "--checkpoints",
                "main",
                "--skip-tool-check",
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert "REFUSE" in err
        assert "preflight" in err.lower()

    def test_refuses_with_rc1_when_gate_raises(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_downstream", lambda: _make_downstream(1, 1))
        monkeypatch.setattr(cli, "load_downstream_humaneval_ids", lambda: [1])

        def boom(report_path: Path, ids: Sequence[int]) -> bool:
            raise RuntimeError("dataset unreachable")

        monkeypatch.setattr(cli, "report_matches_ids", boom)
        rc = main(
            [
                "--results-root",
                str(tmp_path / "out"),
                "--report-path",
                str(tmp_path / "absent.jsonl"),
                "--checkpoints",
                "main",
                "--skip-tool-check",
            ]
        )
        assert rc == 1
        assert "preflight gate raised" in capsys.readouterr().err

    def test_rejects_bad_checkpoint_selection(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "load_downstream", lambda: _make_downstream(1, 1))
        monkeypatch.setattr(cli, "load_downstream_humaneval_ids", lambda: [1])
        rc = main(["--checkpoints", "bogus", "--skip-tool-check"])
        assert rc == 2
        assert "unknown checkpoint" in capsys.readouterr().err

    def test_rejects_non_positive_timeout(self, capsys):
        rc = main(["--timeout", "0", "--skip-tool-check"])
        assert rc == 2
        assert "timeout" in capsys.readouterr().err.lower()

    def test_rejects_non_positive_max_new_tokens(self, capsys):
        rc = main(["--max-new-tokens-code", "0", "--skip-tool-check"])
        assert rc == 2
        assert "max-new-tokens" in capsys.readouterr().err.lower()

    def test_rejects_inf_timeout(self, capsys):
        rc = main(["--timeout", "inf", "--skip-tool-check"])
        assert rc == 2
        assert "timeout" in capsys.readouterr().err.lower()

    def test_rejects_nan_timeout(self, capsys):
        rc = main(["--timeout", "nan", "--skip-tool-check"])
        assert rc == 2
        assert "timeout" in capsys.readouterr().err.lower()

    def test_rejects_negative_timeout(self, capsys):
        rc = main(["--timeout", "-1", "--skip-tool-check"])
        assert rc == 2
        assert "timeout" in capsys.readouterr().err.lower()


# =============================================================================
# Checkpoint resume: scoring_config-aware identity + cache completeness
# =============================================================================


class TestCheckpointCompleteBudget:
    def test_checkpoint_complete_uses_budget_and_timeout(self, tmp_path):
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        hf_id, revision = checkpoint_identity(BASE_CHECKPOINT)
        assert (
            checkpoint_complete(
                model=hf_id,
                revision=revision,
                downstream=_make_downstream(1, 1),
                output_dir=tmp_path / "out" / BASE_CHECKPOINT,
                expected_humaneval=1,
                expected_mmlu=1,
                max_new_tokens_code=DE_CODE,
                max_new_tokens_mmlu=DE_MMLU,
                timeout=DE_TIMEOUT,
            )
            is True
        )

    def test_checkpoint_complete_false_when_timeout_differs(self, tmp_path):
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        hf_id, revision = checkpoint_identity(BASE_CHECKPOINT)
        assert (
            checkpoint_complete(
                model=hf_id,
                revision=revision,
                downstream=_make_downstream(1, 1),
                output_dir=tmp_path / "out" / BASE_CHECKPOINT,
                expected_humaneval=1,
                expected_mmlu=1,
                max_new_tokens_code=DE_CODE,
                max_new_tokens_mmlu=DE_MMLU,
                timeout=3.0,
            )
            is False
        )


# =============================================================================
# Per-item identity now records max_new_tokens + timeout
# =============================================================================


class TestPerItemIdentityBudget:
    def test_item_identity_records_budget_and_timeout(self, tmp_path):
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        item = json.loads(
            (
                tmp_path
                / "out"
                / BASE_CHECKPOINT
                / humaneval_item_filename("python", 0)
            ).read_text(encoding="utf-8")
        )
        assert (
            item["identity"]["generation_contract_version"]
            == GENERATION_CONTRACT_VERSION
        )
        assert item["identity"]["max_new_tokens"] == DE_CODE
        assert item["identity"]["timeout"] == DE_TIMEOUT


# =============================================================================
# Summary scoring_config + aggregate validation
# =============================================================================


def _expected_scoring_config() -> ScoringConfig:
    return ScoringConfig(
        max_new_tokens_code=DE_CODE,
        max_new_tokens_mmlu=DE_MMLU,
        timeout=DE_TIMEOUT,
        generation_contract_version=GENERATION_CONTRACT_VERSION,
    )


class TestSummaryScoringConfig:
    def test_checkpoint_summary_carries_scoring_config(self, tmp_path):
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        loaded = json.loads(
            (tmp_path / "out" / BASE_CHECKPOINT / SUMMARY_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        assert loaded["scoring_config"]["max_new_tokens_code"] == DE_CODE
        assert loaded["scoring_config"]["timeout"] == DE_TIMEOUT
        assert (
            loaded["scoring_config"]["generation_contract_version"]
            == GENERATION_CONTRACT_VERSION
        )

    def test_resume_reuses_summary_only_when_scoring_config_matches(self, tmp_path):
        _, loader1, _ = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        assert len(loader1.calls) == 1
        loader2 = _RecordingModelLoader()
        _, loader2, _ = _run_eval(tmp_path, n_he=2, n_mmlu=2, loader=loader2)
        # Fully cached (matching scoring_config) -> no model load.
        assert loader2.calls == []

    def test_resume_reloads_when_scoring_config_differs(self, tmp_path):
        _, loader1, _ = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        assert len(loader1.calls) == 1
        loader2 = _RecordingModelLoader()
        gens, factory = _generator_factory()
        run_downstream_eval(
            report_path=tmp_path / "ok.jsonl",
            results_root=tmp_path / "out",
            selected_checkpoints=[BASE_CHECKPOINT],
            downstream=_make_downstream(2, 2),
            humaneval_ids=[1, 2],
            runner=_PassRunner(),
            gate=_passing_gate(),
            model_loader=loader2,
            generator_factory=factory,
            expected_humaneval=2,
            expected_mmlu=2,
            max_new_tokens_code=DE_CODE,
            max_new_tokens_mmlu=DE_MMLU,
            timeout=4.0,  # different timeout -> scoring_config differs
        )
        # Per-item identity timed out -> cache miss -> model reloaded.
        assert len(loader2.calls) == 1


class TestAggregateScoringConfig:
    def test_aggregate_includes_scoring_config(self, tmp_path):
        result, _, _ = _run_eval(tmp_path, n_he=1, n_mmlu=1)
        cfg = result.aggregate["scoring_config"]
        assert isinstance(cfg, dict)
        assert cfg["max_new_tokens_code"] == DE_CODE
        assert cfg["timeout"] == DE_TIMEOUT
        assert cfg["generation_contract_version"] == GENERATION_CONTRACT_VERSION

    def test_build_aggregate_excludes_mismatched_scoring_config(self, tmp_path):
        root = tmp_path / "out"
        (root / BASE_CHECKPOINT).mkdir(parents=True)
        mismatched = CheckpointSummary(
            model=OLMO3_VARIANTS[BASE_MODEL_KEY].hf_id,
            revision=BASE_CHECKPOINT,
            n_humaneval_python=1,
            n_humaneval_cpp=1,
            n_mmlu=1,
            python_pass_at_1=1.0,
            cpp_pass_at_1=1.0,
            mmlu_accuracy=1.0,
            python_counts={"pass": 1},
            cpp_counts={"pass": 1},
            mmlu_correct=1,
            mmlu_parsed=1,
            errors=0,
            scoring_config=ScoringConfig(
                max_new_tokens_code=DE_CODE,
                max_new_tokens_mmlu=DE_MMLU,
                timeout=1.0,  # mismatched
                generation_contract_version=GENERATION_CONTRACT_VERSION,
            ),
        )
        write_item_atomically(
            root / BASE_CHECKPOINT / SUMMARY_FILENAME, mismatched.to_dict()
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
            scoring_config=_expected_scoring_config(),
        )
        # Mismatched scoring_config -> checkpoint excluded from coverage.
        assert agg["n_present"] == 0
        agg_cfg = agg["scoring_config"]
        assert isinstance(agg_cfg, dict)
        assert agg_cfg["timeout"] == DE_TIMEOUT


# =============================================================================
# --rebuild-summaries-only: no-model summary rebuild
# =============================================================================


class TestRebuildSummariesOnly:
    def test_no_model_loaded_when_rebuilding_summaries(self, tmp_path):
        _, loader1, _ = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        assert len(loader1.calls) == 1
        loader2 = _RecordingModelLoader()
        gens, factory = _generator_factory()
        result = run_downstream_eval(
            report_path=tmp_path / "ok.jsonl",
            results_root=tmp_path / "out",
            selected_checkpoints=[BASE_CHECKPOINT],
            downstream=_make_downstream(2, 2),
            humaneval_ids=[1, 2],
            runner=_PassRunner(),
            gate=_passing_gate(),
            model_loader=loader2,
            generator_factory=factory,
            expected_humaneval=2,
            expected_mmlu=2,
            rebuild_summaries_only=True,
        )
        # No model load, no generation during a summaries-only rebuild.
        assert loader2.calls == []
        assert all(g.calls == [] for g in gens)
        # Summary + aggregate rebuilt and present.
        assert (tmp_path / "out" / BASE_CHECKPOINT / SUMMARY_FILENAME).exists()
        assert (tmp_path / "out" / AGGREGATE_FILENAME).exists()
        assert result.aggregate["n_present"] == 1

    def test_rebuild_recounts_after_summary_deleted(self, tmp_path):
        _, loader1, _ = _run_eval(tmp_path, n_he=2, n_mmlu=2)
        (tmp_path / "out" / BASE_CHECKPOINT / SUMMARY_FILENAME).unlink()
        loader2 = _RecordingModelLoader()
        gens, factory = _generator_factory()
        run_downstream_eval(
            report_path=tmp_path / "ok.jsonl",
            results_root=tmp_path / "out",
            selected_checkpoints=[BASE_CHECKPOINT],
            downstream=_make_downstream(2, 2),
            humaneval_ids=[1, 2],
            runner=_PassRunner(),
            gate=_passing_gate(),
            model_loader=loader2,
            generator_factory=factory,
            expected_humaneval=2,
            expected_mmlu=2,
            rebuild_summaries_only=True,
        )
        assert (tmp_path / "out" / BASE_CHECKPOINT / SUMMARY_FILENAME).exists()

    def test_rebuild_refuses_when_items_missing(self, tmp_path):
        _run_eval(tmp_path, n_he=2, n_mmlu=2)
        (tmp_path / "out" / BASE_CHECKPOINT / mmlu_item_filename(0)).unlink()
        loader2 = _RecordingModelLoader()
        gens, factory = _generator_factory()
        with pytest.raises((ValueError, FileNotFoundError)):
            run_downstream_eval(
                report_path=tmp_path / "ok.jsonl",
                results_root=tmp_path / "out",
                selected_checkpoints=[BASE_CHECKPOINT],
                downstream=_make_downstream(2, 2),
                humaneval_ids=[1, 2],
                runner=_PassRunner(),
                gate=_passing_gate(),
                model_loader=loader2,
                generator_factory=factory,
                expected_humaneval=2,
                expected_mmlu=2,
                rebuild_summaries_only=True,
            )


# =============================================================================
# build_aggregate: checkpoint-identity validation (reject copied summaries)
# =============================================================================
#


def _matching_summary(checkpoint: str) -> CheckpointSummary:
    """A summary whose model/revision match the canonical mapping for ckpt."""
    hf_id, revision = checkpoint_identity(checkpoint)
    return CheckpointSummary(
        model=hf_id,
        revision=revision,
        n_humaneval_python=1,
        n_humaneval_cpp=1,
        n_mmlu=1,
        python_pass_at_1=1.0,
        cpp_pass_at_1=1.0,
        mmlu_accuracy=1.0,
        python_counts={"pass": 1},
        cpp_counts={"pass": 1},
        mmlu_correct=1,
        mmlu_parsed=1,
        errors=0,
        scoring_config=_expected_scoring_config(),
    )


class TestAggregateCheckpointIdentity:
    def test_build_aggregate_includes_identity_matching_summary(self, tmp_path):
        root = tmp_path / "out"
        (root / BASE_CHECKPOINT).mkdir(parents=True)
        write_item_atomically(
            root / BASE_CHECKPOINT / SUMMARY_FILENAME,
            _matching_summary(BASE_CHECKPOINT).to_dict(),
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
            scoring_config=_expected_scoring_config(),
            expected_humaneval=1,
            expected_mmlu=1,
        )
        assert agg["n_present"] == 1
        checkpoints = agg["checkpoints"]
        assert isinstance(checkpoints, dict)
        assert BASE_CHECKPOINT in checkpoints

    def test_build_aggregate_excludes_summary_with_wrong_revision(self, tmp_path):
        root = tmp_path / "out"
        (root / BASE_CHECKPOINT).mkdir(parents=True)
        # Correct model for "main" but a revision that is NOT the dir name.
        wrong = _matching_summary(BASE_CHECKPOINT)
        wrong_summary = replace(wrong, revision="step_100")
        write_item_atomically(
            root / BASE_CHECKPOINT / SUMMARY_FILENAME, wrong_summary.to_dict()
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
            scoring_config=_expected_scoring_config(),
            expected_humaneval=1,
            expected_mmlu=1,
        )
        # revision "step_100" in the "main" dir -> excluded.
        assert agg["n_present"] == 0
        checkpoints = agg["checkpoints"]
        assert isinstance(checkpoints, dict)
        assert BASE_CHECKPOINT not in checkpoints

    def test_build_aggregate_excludes_summary_with_wrong_model(self, tmp_path):
        root = tmp_path / "out"
        rl_ckpt = RL_CHECKPOINTS[0]
        (root / rl_ckpt).mkdir(parents=True)
        # Correct revision (== dir name) but the BASE model instead of TARGET.
        wrong = _matching_summary(rl_ckpt)
        wrong_summary = replace(wrong, model=OLMO3_VARIANTS[BASE_MODEL_KEY].hf_id)
        write_item_atomically(
            root / rl_ckpt / SUMMARY_FILENAME, wrong_summary.to_dict()
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
            scoring_config=_expected_scoring_config(),
        )
        # An RL-step dir must carry the TARGET model; base model -> excluded.
        assert agg["n_present"] == 0

    def test_build_aggregate_excludes_copied_summary(self, tmp_path):
        root = tmp_path / "out"
        (root / BASE_CHECKPOINT).mkdir(parents=True)
        # Copy a step_100 summary (TARGET model + step_100 revision) verbatim
        # into the "main" dir -> both model and revision are wrong for "main".
        write_item_atomically(
            root / BASE_CHECKPOINT / SUMMARY_FILENAME,
            _matching_summary(RL_CHECKPOINTS[0]).to_dict(),
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
            scoring_config=_expected_scoring_config(),
        )
        assert agg["n_present"] == 0
        checkpoints = agg["checkpoints"]
        assert isinstance(checkpoints, dict)
        assert BASE_CHECKPOINT not in checkpoints

    def test_build_aggregate_no_scoring_config_still_validates_identity(self, tmp_path):
        root = tmp_path / "out"
        (root / BASE_CHECKPOINT).mkdir(parents=True)
        # Identity-validation must apply even when scoring_config is not passed.
        wrong = _matching_summary(BASE_CHECKPOINT)
        wrong_summary = replace(wrong, revision="step_100")
        write_item_atomically(
            root / BASE_CHECKPOINT / SUMMARY_FILENAME, wrong_summary.to_dict()
        )
        agg = build_aggregate(
            root,
            expected_checkpoints=EXPERIMENT_CHECKPOINTS,
            selected_checkpoints=[],
            scoring_config=None,
        )
        assert agg["n_present"] == 0


# =============================================================================
# --rescore-cached: CPU-only rebuild + sandbox re-execution of cached items
# =============================================================================
#


class _RecordingPassRunner:
    """SandboxRunner fake returning rc=0 and recording every invocation."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Path, float]] = []

    def run_in_sandbox(
        self,
        command: Sequence[str],
        scratch_dir: Path,
        timeout: float,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((list(command), scratch_dir, timeout))
        return subprocess.CompletedProcess(
            args=list(command), returncode=0, stdout="", stderr=""
        )


class TestRescoreCached:
    def test_rescore_cached_implies_rebuild_and_invokes_sandbox(self, tmp_path):
        # First pass: produce cached items with a model.
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        loader2 = _RecordingModelLoader()
        runner = _RecordingPassRunner()
        result = run_downstream_eval(
            report_path=tmp_path / "ok.jsonl",
            results_root=tmp_path / "out",
            selected_checkpoints=[BASE_CHECKPOINT],
            downstream=_make_downstream(1, 1),
            humaneval_ids=[1],
            runner=runner,
            gate=_passing_gate(),
            model_loader=loader2,
            generator_factory=_generator_factory()[1],
            expected_humaneval=1,
            expected_mmlu=1,
            rescore_cached=True,
        )
        # No model loaded; sandbox re-executed python (1) + cpp compile+run (2).
        assert loader2.calls == []
        assert len(runner.calls) == 3
        assert result.aggregate["n_present"] == 1

    def test_rescore_cached_compatible_with_rebuild_summaries_only(self, tmp_path):
        _run_eval(tmp_path, n_he=1, n_mmlu=1)
        loader2 = _RecordingModelLoader()
        runner = _RecordingPassRunner()
        # Both flags together: rescore still re-executes; no model load.
        run_downstream_eval(
            report_path=tmp_path / "ok.jsonl",
            results_root=tmp_path / "out",
            selected_checkpoints=[BASE_CHECKPOINT],
            downstream=_make_downstream(1, 1),
            humaneval_ids=[1],
            runner=runner,
            gate=_passing_gate(),
            model_loader=loader2,
            generator_factory=_generator_factory()[1],
            expected_humaneval=1,
            expected_mmlu=1,
            rebuild_summaries_only=True,
            rescore_cached=True,
        )
        assert loader2.calls == []
        assert len(runner.calls) == 3

    def test_parse_args_rescore_cached_default_false(self):
        args = parse_args([])
        assert args.rescore_cached is False

    def test_parse_args_rescore_cached_set(self):
        args = parse_args(["--rescore-cached"])
        assert args.rescore_cached is True

    def test_main_runs_tool_check_when_rescore_cached(self, tmp_path, monkeypatch):
        # main() must run the sandbox tool check + smoke under --rescore-cached
        # (unlike plain --rebuild-summaries-only, which skips it). The data
        # loaders, gate, and eval are stubbed so the test isolates the tool
        # check requirement from real dataset/sandbox state.
        checks: list[str] = []

        def fake_check() -> None:
            checks.append("called")

        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.check_sandbox_tools_available",
            fake_check,
        )
        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.load_downstream",
            lambda: _make_downstream(1, 1),
        )
        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.load_downstream_humaneval_ids",
            lambda: [1],
        )
        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.report_matches_ids",
            lambda report_path, ids: True,
        )

        def fake_eval(**kwargs):
            return DownstreamRunResult(
                aggregate_path=tmp_path / "out" / AGGREGATE_FILENAME,
                aggregate={"n_present": 1, "n_expected": 11},
                checkpoint_summaries={},
            )

        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.run_downstream_eval", fake_eval
        )
        rc = main(
            [
                "--results-root",
                str(tmp_path / "out"),
                "--report-path",
                str(tmp_path / "ok.jsonl"),
                "--checkpoints",
                BASE_CHECKPOINT,
                "--rebuild-summaries-only",
                "--rescore-cached",
            ]
        )
        assert rc == 0
        assert checks == ["called"]

    def test_main_skips_tool_check_for_plain_rebuild(self, tmp_path, monkeypatch):
        # A plain --rebuild-summaries-only (no --rescore-cached) must NOT run
        # the tool check, since the sandbox is never invoked.
        checks: list[str] = []

        def fake_check() -> None:
            checks.append("called")

        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.check_sandbox_tools_available",
            fake_check,
        )
        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.load_downstream",
            lambda: _make_downstream(1, 1),
        )
        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.load_downstream_humaneval_ids",
            lambda: [1],
        )
        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.report_matches_ids",
            lambda report_path, ids: True,
        )

        def fake_eval(**kwargs):
            return DownstreamRunResult(
                aggregate_path=tmp_path / "out" / AGGREGATE_FILENAME,
                aggregate={"n_present": 1, "n_expected": 11},
                checkpoint_summaries={},
            )

        monkeypatch.setattr(
            "experiments.run_rl_zero_downstream.run_downstream_eval", fake_eval
        )
        rc = main(
            [
                "--results-root",
                str(tmp_path / "out"),
                "--report-path",
                str(tmp_path / "ok.jsonl"),
                "--checkpoints",
                BASE_CHECKPOINT,
                "--rebuild-summaries-only",
            ]
        )
        assert rc == 0
        assert checks == []
