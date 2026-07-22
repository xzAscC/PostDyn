"""Tests for experiment-specific configuration constants.

References the experimental setup described in TrainingDynamic.tex
(adapted in docs/experiment_setup.md):
  - PaCE concepts, first 100
  - OLMo-3 7B post-training, bfloat16
  - Think chain (base -> SFT -> DPO -> RL) + RL-Zero family
  - Focus on Think, ignore Instruct
"""

import importlib.util
import os
import subprocess
import sys

import pytest

from src.config import (
    THINK_CHAIN,
    RL_ZERO_FAMILY,
    EXPERIMENT_MODELS,
    EXPERIMENT_NUM_CONCEPTS,
    EXPERIMENT_DTYPE,
    EXPERIMENT_CONCEPT_SOURCE_URL,
    EXPERIMENT_MODEL_COLLECTION_URL,
    EXPERIMENT_LAYERS_7B,
    compute_experiment_layers,
    MODEL_CHECKPOINTS,
    OLMO3_VARIANTS,
)


class TestExperimentMetadata:
    """Experiment-level scalars from TrainingDynamic.tex."""

    def test_num_concepts_is_first_100(self):
        assert EXPERIMENT_NUM_CONCEPTS == 100

    def test_dtype_is_bfloat16(self):
        assert EXPERIMENT_DTYPE == "bfloat16"

    def test_concept_source_url_is_pace(self):
        assert (
            "peterljq/Parsimonious-Concept-Engineering" in EXPERIMENT_CONCEPT_SOURCE_URL
        )

    def test_model_collection_url_is_olmo3_post_training(self):
        assert "allenai/olmo-3-post-training" in EXPERIMENT_MODEL_COLLECTION_URL


class TestThinkChain:
    """Ordered Think chain: base -> SFT -> DPO -> RL (focus on Think)."""

    def test_chain_ordered_base_to_rl(self):
        assert THINK_CHAIN == [
            "olmo3-base",
            "olmo3-think-sft",
            "olmo3-think-dpo",
            "olmo3-think-rlvr",
        ]

    def test_chain_starts_at_base(self):
        assert THINK_CHAIN[0] == "olmo3-base"

    def test_chain_excludes_instruct(self):
        assert not any("instruct" in k for k in THINK_CHAIN)

    def test_all_chain_keys_in_olmo3_variants(self):
        for key in THINK_CHAIN:
            assert key in OLMO3_VARIANTS, f"{key} not in OLMO3_VARIANTS"


class TestRLZeroFamily:
    """RL-Zero family: RL directly from base, no SFT/DPO."""

    def test_family_has_five_models(self):
        assert len(RL_ZERO_FAMILY) == 5

    def test_family_members(self):
        assert RL_ZERO_FAMILY == [
            "olmo3-rl-zero-math",
            "olmo3-rl-zero-code",
            "olmo3-rl-zero-if",
            "olmo3-rl-zero-general",
            "olmo3-rl-zero-mix",
        ]

    def test_all_family_keys_in_olmo3_variants(self):
        for key in RL_ZERO_FAMILY:
            assert key in OLMO3_VARIANTS, f"{key} not in OLMO3_VARIANTS"


class TestExperimentModels:
    """Combined experiment model list (Think chain + RL-Zero)."""

    def test_combined_is_think_chain_plus_rl_zero(self):
        assert EXPERIMENT_MODELS == THINK_CHAIN + RL_ZERO_FAMILY

    def test_base_appears_once(self):
        assert EXPERIMENT_MODELS.count("olmo3-base") == 1

    def test_all_keys_valid_olmo3_variants(self):
        for key in EXPERIMENT_MODELS:
            assert key in OLMO3_VARIANTS, f"{key} not in OLMO3_VARIANTS"

    def test_no_instruct_models(self):
        assert not any("instruct" in k for k in EXPERIMENT_MODELS)

    def test_checkpoint_count_is_nine(self):
        # Early-stage: 9 unique checkpoints, room to expand to 10.
        assert len(EXPERIMENT_MODELS) == 9


class TestLayerSelection:
    """Slide formula layer indices for j=0..n-1.

    ell_j = round[(0.1 + 0.8 * j / (n-1)) * (n_layers - 1)]

    For OLMo-3 7B (n_layers=32, n=10) this yields
    ``[3, 6, 9, 11, 14, 17, 20, 22, 25, 28]``.
    """

    def test_compute_layers_for_32_layer_model(self):
        layers = compute_experiment_layers(32)
        assert layers == [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]

    def test_compute_layers_returns_ten(self):
        assert len(compute_experiment_layers(32)) == 10

    def test_compute_layers_all_in_range(self):
        n = 32
        layers = compute_experiment_layers(n)
        assert all(0 <= i < n for i in layers)

    def test_compute_layers_first_near_ten_percent(self):
        # ell_0 = round(0.1 * 31) = round(3.1) = 3
        assert compute_experiment_layers(32)[0] == 3

    def test_compute_layers_last_below_final(self):
        # ell_9 = round(0.9 * 31) = round(27.9) = 28 (not 31, by design)
        assert compute_experiment_layers(32)[-1] == 28

    def test_compute_layers_n_one_returns_middle(self):
        assert compute_experiment_layers(32, n=1) == [16]

    def test_compute_layers_n_zero_returns_empty(self):
        assert compute_experiment_layers(32, n=0) == []

    def test_precomputed_7b_matches_function(self):
        assert EXPERIMENT_LAYERS_7B == compute_experiment_layers(32)


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)
_RUNNER_PATH = os.path.join(_PROJECT_ROOT, "experiments", "run_concept_dynamics.py")


def _load_runner():
    import sys

    prior = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location(
            "_concept_dynamics_runner", _RUNNER_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load concept dynamics runner")
        loader = spec.loader
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    finally:
        sys.path[:] = prior


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


class TestConceptDynamicsCheckpointWiring:
    _SPLIT = [
        ("olmo3-think-sft", 10),
        ("olmo3-rl-zero-math", 10),
        ("olmo3-rl-zero-code", 10),
        ("olmo3-rl-zero-if", 10),
        ("olmo3-rl-zero-general", 8),
        ("olmo3-rl-zero-mix", 10),
    ]

    def test_total_selected_checkpoints_is_58(self):
        total = sum(len(MODEL_CHECKPOINTS[m]) for m, _ in self._SPLIT)
        assert total == 58

    def test_checkpoint_split_in_runner_model_order(self, runner):
        split = [len(MODEL_CHECKPOINTS[m]) for m in runner.DEFAULT_MODELS]
        assert split == [10, 10, 10, 10, 8, 10]

    def test_runner_family_is_six_models_covering_selected_checkpoints(self, runner):
        assert len(runner.DEFAULT_MODELS) == 6
        assert set(runner.DEFAULT_MODELS) == {m for m, _ in self._SPLIT}
        for m in runner.DEFAULT_MODELS:
            assert m in MODEL_CHECKPOINTS


class TestConceptDynamicsLayerWiring:
    _LAYERS = [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]

    def test_exactly_ten_layers(self):
        assert len(self._LAYERS) == 10

    def test_runner_default_layers_are_the_ten_exact(self):
        assert list(EXPERIMENT_LAYERS_7B) == self._LAYERS


class TestConceptDynamicsConceptWiring:
    def test_default_concepts_are_full_catalogue(self, runner):
        from src.contrastive_datasets import all_concept_keys

        assert runner.DEFAULT_CONCEPTS == all_concept_keys()

    def test_exactly_forty_six_concepts(self, runner):
        assert len(runner.DEFAULT_CONCEPTS) == 46


class TestConceptDynamicsOutputWiring:
    def test_default_output_is_distinct_from_results_concept_dynamics(
        self, runner, monkeypatch
    ):
        monkeypatch.setattr("sys.argv", ["run_concept_dynamics.py"])
        args = runner.parse_args()
        assert args.output != "results/concept_dynamics"

    def test_shell_wrapper_uses_the_fresh_output_directory(self):
        wrapper_path = os.path.join(
            _PROJECT_ROOT, "experiments", "run_concept_dynamics.sh"
        )
        with open(wrapper_path, encoding="utf-8") as handle:
            wrapper = handle.read()
        assert "results/concept_dynamics_multi" in wrapper
        assert (
            'OUTPUT_DIR="${OUTPUT_DIR:-results/concept_dynamics_paired}"' not in wrapper
        )

    def test_quick_and_full_modes_have_distinct_default_outputs(self, runner):
        assert runner.resolve_output_directory(quick=False, output=None) == (
            "results/concept_dynamics_multi"
        )
        assert runner.resolve_output_directory(quick=True, output=None) == (
            "results/concept_dynamics_multi_quick"
        )

    def test_explicit_output_overrides_both_mode_defaults(self, runner):
        for quick in (False, True):
            assert (
                runner.resolve_output_directory(quick=quick, output="results/custom")
                == "results/custom"
            )

    def test_shell_wrapper_declares_separate_quick_default(self):
        wrapper_path = os.path.join(
            _PROJECT_ROOT, "experiments", "run_concept_dynamics.sh"
        )
        with open(wrapper_path, encoding="utf-8") as handle:
            wrapper = handle.read()
        assert "results/concept_dynamics_multi_quick" in wrapper


class TestConceptDynamicsCliHelp:
    def test_help_exits_zero_and_mentions_full_catalogue(self):
        result = subprocess.run(
            [sys.executable, _RUNNER_PATH, "--help"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "46" in result.stdout
        assert "all_concept_keys" in result.stdout

    def test_shell_help_does_not_claim_results_were_saved(self):
        wrapper_path = os.path.join(
            _PROJECT_ROOT, "experiments", "run_concept_dynamics.sh"
        )
        result = subprocess.run(
            [wrapper_path, "--help"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Results saved" not in result.stdout
