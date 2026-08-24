"""Math-trajectory differential-subspace experiment configuration.

Scope (locked):
  * Model trajectory: 10 RL-Zero-Math checkpoints
  * Pair: ``math_vs_text`` (Dolci Math vs Dolci General)
  * 1,000 prompts / domain, last-token hidden states, 10 layers
  * Save U per (checkpoint, layer, concept) and the five PostDyn metrics
"""

from __future__ import annotations

from pathlib import Path

from src.config import (
    EXPERIMENT_LAYERS_7B,
    MODEL_CHECKPOINTS,
    OLMO3_VARIANTS,
    ModelConfig,
)
from src.dataset_store import PROJECT_ROOT, SHARED_SAMPLE_SEED
from src.domain_datasets import DEFAULT_CONCEPT_PAIRS
from src.differential_subspace import DEFAULT_TAU

# =============================================================================
# Models & checkpoints
# =============================================================================

BASE_MODEL_KEY: str = "olmo3-base"
TARGET_MODEL_KEY: str = "olmo3-rl-zero-math"

BASE_MODEL: ModelConfig = OLMO3_VARIANTS[BASE_MODEL_KEY]
TARGET_MODEL: ModelConfig = OLMO3_VARIANTS[TARGET_MODEL_KEY]

RL_CHECKPOINTS: list[str] = list(MODEL_CHECKPOINTS[TARGET_MODEL_KEY])
EXPERIMENT_CHECKPOINTS: list[str] = RL_CHECKPOINTS

EXPERIMENT_LAYERS: list[int] = list(EXPERIMENT_LAYERS_7B)

# =============================================================================
# Sampling & geometry
# =============================================================================

N_SAMPLES: int = 1000
SAMPLE_SEED: int = SHARED_SAMPLE_SEED
TAU: float = DEFAULT_TAU
MAX_SEQ_LEN: int = 2048
USE_CHAT_TEMPLATE: bool = False
EXTRACTION_CONTRACT: str = "raw_prompt_final_attention_token_v1"

CONCEPT_PAIRS: tuple[tuple[str, str, str], ...] = (DEFAULT_CONCEPT_PAIRS[1],)

# =============================================================================
# Output layout
# =============================================================================

RESULTS_ROOT: Path = (
    PROJECT_ROOT / "results" / "math_differential_subspace_setup_raw_prompt"
)
RESULTS_ROOT_QUICK: Path = (
    PROJECT_ROOT / "results" / "math_differential_subspace_setup_raw_prompt_quick"
)

CHECKPOINT_REVISIONS: dict[str, str] = {
    "step_100": "3315e80ceb281ae2e6a20bd09e8594ba52d4f312",
    "step_300": "528d76d90a93f8498801534bc72a346cf886a115",
    "step_500": "e5280adc8e0de0719e336ec50e095ccbd2577ab4",
    "step_700": "c23e8ebda0c59b21db2cb9747739998ffbd71430",
    "step_900": "d0bb0760e45e5bba36410285ffff80ac9b8fcabf",
    "step_1100": "4b024587d5af90918b0d97da34f649e145936459",
    "step_1300": "526ba5a33e775c2fb5780636274e1e38c6fbeea2",
    "step_1500": "968604e73e5ff2b027567270194750bdf5288ccb",
    "step_1700": "5fce4e571a11d60aed9eccaeef72cc603e7e260c",
    "step_1900": "8182367150cef52ddf00dd5259ea94eaa330918e",
}

U_SUBDIR: str = "U"
METRICS_SUBDIR: str = "metrics"
MANIFESTS_SUBDIR: str = "manifests"
PROMPTS_SUBDIR: str = "prompts"


def model_for_checkpoint(checkpoint: str) -> ModelConfig:
    return TARGET_MODEL


def revision_for_checkpoint(checkpoint: str) -> str:
    """HF revision string for a checkpoint name."""
    try:
        return CHECKPOINT_REVISIONS[checkpoint]
    except KeyError as exc:
        raise ValueError(f"Unknown immutable RL checkpoint: {checkpoint}") from exc


def is_base_checkpoint(checkpoint: str) -> bool:
    return False
