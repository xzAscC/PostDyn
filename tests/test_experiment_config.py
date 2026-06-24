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
