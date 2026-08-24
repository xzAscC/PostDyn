"""Tests for the RL-Zero-Code syntax experiment configuration module.

Covers the structural invariants encoded in ``src.rl_zero_experiment.self_check``
plus the downstream-data helpers (50 HumanEval-X items, 50 MMLU questions,
disjoint target ids) against the real builder artifacts when present.

These tests are data/config only: they run no model and write no results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src import rl_zero_experiment as exp
from src.config import (
    EXPERIMENT_LAYERS_7B,
    MODEL_CHECKPOINTS,
    OLMO3_VARIANTS,
    RESULTS_DIR,
)
from src.rl_zero_experiment import (
    BASE_CHECKPOINT,
    BASE_MODEL,
    BASE_MODEL_KEY,
    CONCEPT_PROBE_CLASSES,
    CONTROL_CONCEPT,
    DOWNSTREAM_FILE,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_CONCEPTS,
    EXPERIMENT_LAYERS,
    N_SAMPLES,
    PAIRS_FILE,
    PRIMARY_USE_CHAT_TEMPLATE,
    PROBE_CLASSES,
    RELATED_CONCEPTS,
    RL_CHECKPOINTS,
    RL_ZERO_CODE_RESULTS_ROOT,
    SENSITIVITY_STEPS,
    TARGET_CONCEPT,
    TARGET_MODEL,
    TARGET_MODEL_KEY,
    ConceptSpec,
    DownstreamDataError,
    concept_specs,
    downstream_humaneval_ids,
    humaneval_x_item_count,
    is_base_checkpoint,
    is_rl_checkpoint,
    load_downstream,
    load_pairs,
    mmlu_question_count,
    probe_classes_used,
    results_root,
    self_check,
    target_ids,
    validate_downstream,
    validate_downstream_counts,
    validate_target_downstream_disjoint,
)

# Shared-registry imports used by the membership tests.
from src.contrastive_datasets import CONCEPTS as REGISTRY


# =============================================================================
# self_check
# =============================================================================


class TestSelfCheck:
    def test_self_check_passes(self) -> None:
        # Should raise nothing.
        self_check()

    def test_import_time_assertions_already_ran(self) -> None:
        # If the module imported at all, the cheap invariants held.
        assert exp.EXPERIMENT_CHECKPOINTS is not None


# =============================================================================
# Results isolation
# =============================================================================


class TestResultsIsolation:
    def test_root_is_under_results_dir(self) -> None:
        assert RL_ZERO_CODE_RESULTS_ROOT.startswith(RESULTS_DIR)

    def test_root_is_not_concept_dynamics_multi(self) -> None:
        assert RL_ZERO_CODE_RESULTS_ROOT != RESULTS_DIR + "/concept_dynamics_multi"

    def test_root_does_not_collide_with_any_paired_root(self) -> None:
        forbidden = {
            exp.PAIRED_CONCEPT_RESULTS_ROOT,
            exp.PAIRED_CONCEPT_RESULTS_ROOT_QUICK,
            RESULTS_DIR + "/concept_dynamics",
            RESULTS_DIR + "/concept_dynamics_paired",
        }
        assert RL_ZERO_CODE_RESULTS_ROOT not in forbidden

    def test_quick_root_is_distinct_from_paired_quick(self) -> None:
        assert (
            exp.RL_ZERO_CODE_RESULTS_ROOT_QUICK != exp.PAIRED_CONCEPT_RESULTS_ROOT_QUICK
        )

    @pytest.mark.parametrize("quick", [False, True])
    def test_results_root_default(self, quick: bool) -> None:
        expected = (
            exp.RL_ZERO_CODE_RESULTS_ROOT_QUICK if quick else RL_ZERO_CODE_RESULTS_ROOT
        )
        assert results_root(quick=quick) == expected

    def test_results_root_explicit_override_wins(self) -> None:
        assert results_root(override="results/custom") == "results/custom"
        assert results_root(quick=True, override="results/custom") == "results/custom"

    def test_sensitivity_input_root_is_the_paired_run(self) -> None:
        assert exp.SENSITIVITY_INPUT_RESULTS_ROOT == exp.PAIRED_CONCEPT_RESULTS_ROOT


# =============================================================================
# Checkpoint schedule (11 = main + 10 RL)
# =============================================================================


class TestCheckpointSchedule:
    def test_base_model_is_olmo3_base(self) -> None:
        assert BASE_MODEL_KEY == "olmo3-base"
        assert BASE_MODEL is OLMO3_VARIANTS[BASE_MODEL_KEY]
        assert BASE_MODEL.revision == "main"

    def test_target_model_is_olmo3_rl_zero_code(self) -> None:
        assert TARGET_MODEL_KEY == "olmo3-rl-zero-code"
        assert TARGET_MODEL is OLMO3_VARIANTS[TARGET_MODEL_KEY]
        assert TARGET_MODEL.pathway == "rl-zero"

    def test_base_checkpoint_is_main(self) -> None:
        assert BASE_CHECKPOINT == "main"

    def test_rl_checkpoints_reused_from_model_checkpoints(self) -> None:
        # Must be exactly the src.config schedule -- never re-derived here.
        assert RL_CHECKPOINTS == list(MODEL_CHECKPOINTS[TARGET_MODEL_KEY])

    def test_ten_rl_checkpoints(self) -> None:
        assert len(RL_CHECKPOINTS) == 10

    def test_eleven_total_checkpoints(self) -> None:
        assert len(EXPERIMENT_CHECKPOINTS) == 11

    def test_schedule_starts_with_base_then_rl_in_order(self) -> None:
        assert EXPERIMENT_CHECKPOINTS[0] == BASE_CHECKPOINT
        assert EXPERIMENT_CHECKPOINTS[1:] == RL_CHECKPOINTS

    def test_checkpoints_are_unique(self) -> None:
        assert len(set(EXPERIMENT_CHECKPOINTS)) == 11

    def test_base_not_in_model_checkpoints(self) -> None:
        # olmo3-base has no step schedule; only 'main' is used.
        assert BASE_MODEL_KEY not in MODEL_CHECKPOINTS

    def test_predicates(self) -> None:
        assert is_base_checkpoint("main") is True
        assert is_base_checkpoint("step_100") is False
        assert is_rl_checkpoint("step_100") is True
        assert is_rl_checkpoint("main") is False
        assert is_rl_checkpoint("step_9999") is False


# =============================================================================
# Sensitivity sub-steps
# =============================================================================


class TestSensitivitySteps:
    def test_exactly_three(self) -> None:
        assert SENSITIVITY_STEPS == ("step_100", "step_1700", "step_2900")

    def test_are_subset_of_rl_checkpoints(self) -> None:
        assert set(SENSITIVITY_STEPS) <= set(RL_CHECKPOINTS)

    def test_are_first_middle_last(self) -> None:
        positions = [RL_CHECKPOINTS.index(s) for s in SENSITIVITY_STEPS]
        assert positions == [
            0,
            len(RL_CHECKPOINTS) // 2 - 1 + 1,
            len(RL_CHECKPOINTS) - 1,
        ]
        assert positions[0] == 0
        assert positions[-1] == len(RL_CHECKPOINTS) - 1
        # Strictly increasing.
        assert positions == sorted(positions)


# =============================================================================
# Layers & samples
# =============================================================================


class TestLayersAndSamples:
    def test_ten_layers_reused_from_config(self) -> None:
        assert EXPERIMENT_LAYERS == list(EXPERIMENT_LAYERS_7B)
        assert EXPERIMENT_LAYERS == [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]

    def test_layers_are_unique_and_in_range(self) -> None:
        assert len(set(EXPERIMENT_LAYERS)) == 10
        assert all(0 <= i < 32 for i in EXPERIMENT_LAYERS)

    def test_sample_count_is_fifty(self) -> None:
        assert N_SAMPLES == 50

    def test_downstream_counts_track_samples(self) -> None:
        assert exp.DOWNSTREAM_HUMANEVAL_X_ITEMS == N_SAMPLES
        assert exp.DOWNSTREAM_MMLU_ITEMS == N_SAMPLES


# =============================================================================
# Protocol
# =============================================================================


class TestProtocol:
    def test_primary_is_raw_text(self) -> None:
        assert PRIMARY_USE_CHAT_TEMPLATE is False


# =============================================================================
# Probe classes & concepts
# =============================================================================


class TestProbeClasses:
    def test_exactly_eight_classes(self) -> None:
        assert PROBE_CLASSES == (
            "python_valid",
            "python_syntax_error",
            "cpp",
            "js",
            "java",
            "go",
            "she",
            "he",
        )

    def test_unique(self) -> None:
        assert len(set(PROBE_CLASSES)) == 8


class TestConcepts:
    def test_six_concepts(self) -> None:
        assert len(EXPERIMENT_CONCEPTS) == 6

    def test_structure_target_related_control(self) -> None:
        assert EXPERIMENT_CONCEPTS[0] == TARGET_CONCEPT
        assert EXPERIMENT_CONCEPTS[-1] == CONTROL_CONCEPT
        assert tuple(EXPERIMENT_CONCEPTS[1:-1]) == RELATED_CONCEPTS

    def test_target_concept_key(self) -> None:
        assert TARGET_CONCEPT == "python_valid_vs_syntax_error"

    def test_four_related_concepts(self) -> None:
        assert RELATED_CONCEPTS == (
            "code_python_vs_cpp",
            "code_python_vs_js",
            "code_python_vs_java",
            "code_python_vs_go",
        )

    def test_control_concept_key(self) -> None:
        assert CONTROL_CONCEPT == "gender_she_vs_he"

    def test_concepts_unique(self) -> None:
        assert len(set(EXPERIMENT_CONCEPTS)) == 6

    def test_every_concept_declares_probe_classes(self) -> None:
        assert set(CONCEPT_PROBE_CLASSES) == set(EXPERIMENT_CONCEPTS)

    def test_probe_class_coverage_matches_exactly(self) -> None:
        assert probe_classes_used() == set(PROBE_CLASSES)

    def test_positive_and_negative_distinct_and_known(self) -> None:
        known = set(PROBE_CLASSES)
        for key, (pos, neg) in CONCEPT_PROBE_CLASSES.items():
            assert pos != neg, key
            assert pos in known, (key, pos)
            assert neg in known, (key, neg)

    def test_target_polarity(self) -> None:
        assert CONCEPT_PROBE_CLASSES[TARGET_CONCEPT] == (
            "python_valid",
            "python_syntax_error",
        )

    def test_related_polarity_python_is_negative(self) -> None:
        for key in RELATED_CONCEPTS:
            pos, neg = CONCEPT_PROBE_CLASSES[key]
            assert neg == "python_valid", key
            assert pos in {"cpp", "js", "java", "go"}, key

    def test_control_polarity(self) -> None:
        assert CONCEPT_PROBE_CLASSES[CONTROL_CONCEPT] == ("he", "she")


class TestRegistryMembership:
    def test_related_concepts_registered_and_contrast_python(self) -> None:
        for key in RELATED_CONCEPTS:
            assert key in REGISTRY, key
            assert REGISTRY[key]["negative"] == "python", key
            assert REGISTRY[key]["domain"] == "code", key

    def test_control_concept_registered(self) -> None:
        assert CONTROL_CONCEPT in REGISTRY
        assert REGISTRY[CONTROL_CONCEPT]["domain"] == "general"

    def test_target_concept_registered_as_syntax(self) -> None:
        assert TARGET_CONCEPT in REGISTRY
        assert REGISTRY[TARGET_CONCEPT]["domain"] == "syntax"
        assert REGISTRY[TARGET_CONCEPT]["positive"] == "syntax_valid"
        assert REGISTRY[TARGET_CONCEPT]["negative"] == "syntax_error"


class TestConceptSpecs:
    def test_specs_cover_all_concepts(self) -> None:
        specs = concept_specs()
        assert set(specs) == set(EXPERIMENT_CONCEPTS)

    def test_specs_are_frozen_dataclass(self) -> None:
        import dataclasses

        spec = concept_specs()[TARGET_CONCEPT]
        assert isinstance(spec, ConceptSpec)
        assert dataclasses.is_dataclass(ConceptSpec)
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(spec, "key", "mutated")

    def test_target_spec_registered(self) -> None:
        spec = concept_specs()[TARGET_CONCEPT]
        assert spec.role == "target"
        assert spec.registered is True
        assert spec.positive_class == "python_valid"
        assert spec.negative_class == "python_syntax_error"
        assert spec.domain == "syntax"

    def test_related_specs_registered_with_python_negative(self) -> None:
        specs = concept_specs()
        for key in RELATED_CONCEPTS:
            s = specs[key]
            assert s.role == "related"
            assert s.registered is True
            assert s.negative_class == "python_valid"
            assert s.domain == "code"

    def test_control_spec_registered(self) -> None:
        s = concept_specs()[CONTROL_CONCEPT]
        assert s.role == "control"
        assert s.registered is True
        assert s.domain == "general"
        assert (s.positive_class, s.negative_class) == ("he", "she")


# =============================================================================
# Downstream-data helpers (against real artifacts, skip-friendly)
# =============================================================================


def _skip_if_no_artifact(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"{path} not built; run experiments/build_rl_zero_syntax_concept.py"
        )


@pytest.fixture(scope="module")
def pairs_data() -> dict[str, Any]:
    _skip_if_no_artifact(PAIRS_FILE)
    return load_pairs()


@pytest.fixture(scope="module")
def downstream_data() -> dict[str, Any]:
    _skip_if_no_artifact(DOWNSTREAM_FILE)
    return load_downstream()


class TestArtifactsPresent:
    def test_pairs_file_path_points_into_dolci_dir(self) -> None:
        assert PAIRS_FILE.name == "python_syntax_pairs.json"
        assert "Dolci-RL-Zero-Code-7B" in str(PAIRS_FILE)

    def test_downstream_file_path_points_into_dolci_dir(self) -> None:
        assert DOWNSTREAM_FILE.name == "downstream.json"
        assert "Dolci-RL-Zero-Code-7B" in str(DOWNSTREAM_FILE)


class TestDownstreamCounts:
    def test_fifty_humaneval_x_items(self, downstream_data: dict[str, Any]) -> None:
        assert humaneval_x_item_count(downstream_data) == 50
        assert len(downstream_data["humaneval_x"]["items"]) == 50

    def test_fifty_mmlu_questions(self, downstream_data: dict[str, Any]) -> None:
        assert mmlu_question_count(downstream_data) == 50
        assert len(downstream_data["mmlu"]["items"]) == 50

    def test_validate_downstream_counts_passes(
        self, downstream_data: dict[str, Any]
    ) -> None:
        validate_downstream_counts(downstream_data)

    def test_validate_downstream_counts_rejects_wrong_humaneval_x(
        self, downstream_data: dict[str, Any]
    ) -> None:
        bad = json.loads(json.dumps(downstream_data))
        bad["humaneval_x"]["n_items"] = 49
        with pytest.raises(DownstreamDataError):
            validate_downstream_counts(bad)

    def test_validate_downstream_counts_rejects_wrong_list_length(
        self, downstream_data: dict[str, Any]
    ) -> None:
        bad = json.loads(json.dumps(downstream_data))
        bad["humaneval_x"]["items"].append(bad["humaneval_x"]["items"][0])
        with pytest.raises(DownstreamDataError):
            validate_downstream_counts(bad)

    def test_validate_downstream_counts_rejects_missing_block(
        self, downstream_data: dict[str, Any]
    ) -> None:
        bad = json.loads(json.dumps(downstream_data))
        del bad["humaneval_x"]
        with pytest.raises(DownstreamDataError):
            validate_downstream_counts(bad)


class TestTargetDisjointness:
    def test_target_and_downstream_ids_disjoint(
        self,
        pairs_data: dict[str, Any],
        downstream_data: dict[str, Any],
    ) -> None:
        target = target_ids(pairs_data)
        down = downstream_humaneval_ids(downstream_data)
        assert target.isdisjoint(down)

    def test_builder_disjointness_flag_is_true(
        self, pairs_data: dict[str, Any]
    ) -> None:
        assert pairs_data["selection"]["disjointness_verified"] is True

    def test_excluded_downstream_matches_pinned_ids(
        self,
        pairs_data: dict[str, Any],
        downstream_data: dict[str, Any],
    ) -> None:
        excluded = set(pairs_data["selection"]["excluded_downstream_ids"])
        assert excluded == downstream_humaneval_ids(downstream_data)

    def test_validate_disjointness_passes(
        self,
        pairs_data: dict[str, Any],
        downstream_data: dict[str, Any],
    ) -> None:
        validate_target_downstream_disjoint(pairs_data, downstream_data)

    def test_validate_disjointness_detects_overlap(
        self,
        pairs_data: dict[str, Any],
        downstream_data: dict[str, Any],
    ) -> None:
        bad_pairs = json.loads(json.dumps(pairs_data))
        # Inject one downstream id into the target set to force overlap.
        down_id = next(iter(downstream_humaneval_ids(downstream_data)))
        bad_pairs["selection"]["target_ids"].append(down_id)
        with pytest.raises(DownstreamDataError):
            validate_target_downstream_disjoint(bad_pairs, downstream_data)

    def test_validate_disjointness_detects_false_flag(
        self,
        pairs_data: dict[str, Any],
        downstream_data: dict[str, Any],
    ) -> None:
        bad_pairs = json.loads(json.dumps(pairs_data))
        bad_pairs["selection"]["disjointness_verified"] = False
        with pytest.raises(DownstreamDataError):
            validate_target_downstream_disjoint(bad_pairs, downstream_data)


class TestValidateDownstreamOrchestrator:
    def test_end_to_end_passes_with_loaded_artifacts(
        self,
        pairs_data: dict[str, Any],
        downstream_data: dict[str, Any],
    ) -> None:
        validate_downstream(pairs=pairs_data, downstream=downstream_data)

    def test_loads_artifacts_when_omitted(self) -> None:
        # When artifacts exist, this exercises the lazy-loading path.
        _skip_if_no_artifact(PAIRS_FILE)
        _skip_if_no_artifact(DOWNSTREAM_FILE)
        validate_downstream()  # should not raise

    def test_missing_pairs_file_raises(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(exp, "PAIRS_FILE", tmp_path / "absent_pairs.json")
        monkeypatch.setattr(exp, "DOWNSTREAM_FILE", tmp_path / "absent_down.json")
        with pytest.raises(FileNotFoundError):
            validate_downstream()

    def test_missing_downstream_file_raises(
        self, tmp_path: Path, monkeypatch, pairs_data: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(exp, "DOWNSTREAM_FILE", tmp_path / "absent_down.json")
        with pytest.raises(FileNotFoundError):
            validate_downstream(pairs=pairs_data)


# =============================================================================
# Consistency with src.config (anti-duplication guard)
# =============================================================================


class TestNoDuplicationFromConfig:
    def test_rl_checkpoints_identity_to_config(self) -> None:
        # Same values, derived from the same source of truth.
        assert RL_CHECKPOINTS == list(MODEL_CHECKPOINTS[TARGET_MODEL_KEY])

    def test_layers_identity_to_config(self) -> None:
        assert EXPERIMENT_LAYERS == list(EXPERIMENT_LAYERS_7B)

    def test_models_are_the_same_objects_as_config(self) -> None:
        assert BASE_MODEL is OLMO3_VARIANTS[BASE_MODEL_KEY]
        assert TARGET_MODEL is OLMO3_VARIANTS[TARGET_MODEL_KEY]
