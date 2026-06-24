"""Tests for experiment-specific configuration constants.

References the experimental setup described in TrainingDynamic.tex
(adapted in docs/experiment_setup.md):
  - PaCE concepts, first 100
  - OLMo-3 7B post-training, bfloat16
  - Think chain (base -> SFT -> DPO -> RL) + RL-Zero family
  - Focus on Think, ignore Instruct
"""

import pytest

from src.config import (
    THINK_CHAIN,
    RL_ZERO_FAMILY,
    EXPERIMENT_MODELS,
    EXPERIMENT_NUM_CONCEPTS,
    EXPERIMENT_DTYPE,
    EXPERIMENT_CONCEPT_SOURCE_URL,
    EXPERIMENT_MODEL_COLLECTION_URL,
    EXPERIMENT_LAYER_PERCENTAGES,
    EXPERIMENT_LAYERS_7B,
    compute_experiment_layers,
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
    """10 layers at 10%, 20%, ..., 100% of model depth (tex line 8)."""

    def test_percentages_are_ten_even_steps(self):
        assert EXPERIMENT_LAYER_PERCENTAGES == [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

    def test_compute_layers_for_32_layer_model(self):
        layers = compute_experiment_layers(32)
        assert layers == [3, 6, 9, 12, 16, 19, 22, 25, 28, 31]

    def test_compute_layers_returns_ten(self):
        assert len(compute_experiment_layers(32)) == 10

    def test_compute_layers_all_in_range(self):
        n = 32
        layers = compute_experiment_layers(n)
        assert all(0 <= i < n for i in layers)

    def test_compute_layers_last_is_final_layer(self):
        assert compute_experiment_layers(32)[-1] == 31

    def test_precomputed_7b_matches_function(self):
        assert EXPERIMENT_LAYERS_7B == compute_experiment_layers(32)
