"""
Configuration module for PostDyn.

Defines all model configurations, checkpoint schedules, weight matrix patterns,
and result paths used throughout the analysis pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")


# =============================================================================
# Model Configuration Data Classes
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a single model variant."""

    name: str
    hf_id: str
    revision: str = "main"
    architecture: str = "gpt_neox"  # or "olmo3"
    layers: int = 0
    d_model: int = 0
    intermediate_size: int = 0
    n_heads: int = 0
    n_kv_heads: Optional[int] = None
    total_params: str = ""
    pathway: str = ""  # "base", "think", "instruct", "rl-zero"
    stage: str = ""  # "pretrain", "sft", "dpo", "rlvr", "base"


# =============================================================================
# Pythia Model Configurations
# =============================================================================

PYTHIA_CONFIGS: dict[str, ModelConfig] = {
    "pythia-70m": ModelConfig(
        name="pythia-70m",
        hf_id="EleutherAI/pythia-70m",
        architecture="gpt_neox",
        layers=6,
        d_model=512,
        intermediate_size=2048,
        n_heads=8,
        n_kv_heads=8,
        total_params="70M",
    ),
    "pythia-160m": ModelConfig(
        name="pythia-160m",
        hf_id="EleutherAI/pythia-160m",
        architecture="gpt_neox",
        layers=12,
        d_model=768,
        intermediate_size=3072,
        n_heads=12,
        n_kv_heads=12,
        total_params="160M",
    ),
    "pythia-410m": ModelConfig(
        name="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        architecture="gpt_neox",
        layers=24,
        d_model=1024,
        intermediate_size=4096,
        n_heads=16,
        n_kv_heads=16,
        total_params="410M",
    ),
    "pythia-1b": ModelConfig(
        name="pythia-1b",
        hf_id="EleutherAI/pythia-1b",
        architecture="gpt_neox",
        layers=16,
        d_model=2048,
        intermediate_size=8192,
        n_heads=8,
        n_kv_heads=8,
        total_params="1B",
    ),
    "pythia-1.4b": ModelConfig(
        name="pythia-1.4b",
        hf_id="EleutherAI/pythia-1.4b",
        architecture="gpt_neox",
        layers=24,
        d_model=2048,
        intermediate_size=8192,
        n_heads=16,
        n_kv_heads=16,
        total_params="1.4B",
    ),
    "pythia-2.8b": ModelConfig(
        name="pythia-2.8b",
        hf_id="EleutherAI/pythia-2.8b",
        architecture="gpt_neox",
        layers=32,
        d_model=2560,
        intermediate_size=10240,
        n_heads=32,
        n_kv_heads=32,
        total_params="2.8B",
    ),
    "pythia-6.9b": ModelConfig(
        name="pythia-6.9b",
        hf_id="EleutherAI/pythia-6.9b",
        architecture="gpt_neox",
        layers=32,
        d_model=4096,
        intermediate_size=16384,
        n_heads=32,
        n_kv_heads=32,
        total_params="6.9B",
    ),
    "pythia-12b": ModelConfig(
        name="pythia-12b",
        hf_id="EleutherAI/pythia-12b",
        architecture="gpt_neox",
        layers=36,
        d_model=5120,
        intermediate_size=20480,
        n_heads=40,
        n_kv_heads=40,
        total_params="12B",
    ),
}

# Deduped variants (same architecture, trained on deduplicated data)
PYTHIA_DEDUPED_CONFIGS: dict[str, ModelConfig] = {}
for name, cfg in PYTHIA_CONFIGS.items():
    dedup_name = f"{name}-deduped"
    dedup_id = cfg.hf_id.replace("pythia-", "pythia-") + "-deduped"
    PYTHIA_DEDUPED_CONFIGS[dedup_name] = ModelConfig(
        name=dedup_name,
        hf_id=dedup_id,
        architecture=cfg.architecture,
        layers=cfg.layers,
        d_model=cfg.d_model,
        intermediate_size=cfg.intermediate_size,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        total_params=cfg.total_params,
    )

# =============================================================================
# Pythia Training Checkpoints
# =============================================================================

# Curated list of checkpoints covering the full training trajectory
# ~25 checkpoints: log-spaced early + linear-spaced later
PYTHIA_CHECKPOINTS: list[str] = [
    # Log-spaced early checkpoints (warmup phase)
    "step0",
    "step1",
    "step2",
    "step4",
    "step8",
    "step16",
    "step32",
    "step64",
    "step128",
    "step256",
    "step512",
    # Transition phase
    "step1000",
    "step2000",
    "step4000",
    "step8000",
    # Mid-training
    "step10000",
    "step20000",
    "step40000",
    "step60000",
    # Late training
    "step80000",
    "step100000",
    "step120000",
    "step140000",
    # Final
    "step143000",
]

# Shorter list for quick testing
PYTHIA_CHECKPOINTS_QUICK: list[str] = [
    "step0",
    "step1",
    "step16",
    "step128",
    "step1000",
    "step10000",
    "step50000",
    "step100000",
    "step143000",
]


# =============================================================================
# OLMo-3 Model Configurations
# =============================================================================

OLMO3_BASE_CONFIG = ModelConfig(
    name="olmo3-base",
    hf_id="allenai/Olmo-3-1025-7B",
    revision="main",
    architecture="olmo3",
    layers=32,
    d_model=4096,
    intermediate_size=11008,
    n_heads=32,
    n_kv_heads=32,
    total_params="7B",
    pathway="base",
    stage="base",
)

# OLMo-3 post-training variants
OLMO3_VARIANTS: dict[str, ModelConfig] = {
    # Base model
    "olmo3-base": OLMO3_BASE_CONFIG,
    # Think pathway (reasoning with chain-of-thought)
    "olmo3-think-sft": ModelConfig(
        name="olmo3-think-sft",
        hf_id="allenai/Olmo-3-7B-Think-SFT",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="think",
        stage="sft",
    ),
    "olmo3-think-dpo": ModelConfig(
        name="olmo3-think-dpo",
        hf_id="allenai/Olmo-3-7B-Think-DPO",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="think",
        stage="dpo",
    ),
    "olmo3-think-rlvr": ModelConfig(
        name="olmo3-think-rlvr",
        hf_id="allenai/Olmo-3-7B-Think",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="think",
        stage="rlvr",
    ),
    # Instruct pathway (chat/tool-use)
    "olmo3-instruct-sft": ModelConfig(
        name="olmo3-instruct-sft",
        hf_id="allenai/Olmo-3-7B-Instruct-SFT",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="instruct",
        stage="sft",
    ),
    "olmo3-instruct-dpo": ModelConfig(
        name="olmo3-instruct-dpo",
        hf_id="allenai/Olmo-3-7B-Instruct-DPO",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="instruct",
        stage="dpo",
    ),
    "olmo3-instruct-rlvr": ModelConfig(
        name="olmo3-instruct-rlvr",
        hf_id="allenai/Olmo-3-7B-Instruct",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="instruct",
        stage="rlvr",
    ),
    # RL-Zero pathway (RL directly from base, no SFT/DPO)
    "olmo3-rl-zero-math": ModelConfig(
        name="olmo3-rl-zero-math",
        hf_id="allenai/Olmo-3-7B-RL-Zero-Math",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="rl-zero",
        stage="rlvr-math",
    ),
    "olmo3-rl-zero-code": ModelConfig(
        name="olmo3-rl-zero-code",
        hf_id="allenai/Olmo-3-7B-RL-Zero-Code",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="rl-zero",
        stage="rlvr-code",
    ),
    "olmo3-rl-zero-if": ModelConfig(
        name="olmo3-rl-zero-if",
        hf_id="allenai/Olmo-3-7B-RL-Zero-IF",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="rl-zero",
        stage="rlvr-if",
    ),
    "olmo3-rl-zero-general": ModelConfig(
        name="olmo3-rl-zero-general",
        hf_id="allenai/Olmo-3-7B-RL-Zero-General",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="rl-zero",
        stage="rlvr-general",
    ),
    "olmo3-rl-zero-mix": ModelConfig(
        name="olmo3-rl-zero-mix",
        hf_id="allenai/Olmo-3-7B-RL-Zero-Mix",
        architecture="olmo3",
        layers=32,
        d_model=4096,
        intermediate_size=11008,
        n_heads=32,
        n_kv_heads=32,
        total_params="7B",
        pathway="rl-zero",
        stage="rlvr-mix",
    ),
}

# =============================================================================
# EXPERIMENT CONFIGURATION — PaCE concepts × OLMo-3 (Think focus)
# Setup adapted from TrainingDynamic.tex. See docs/experiment_setup.md.
# =============================================================================

EXPERIMENT_CONCEPT_SOURCE_URL = (
    "https://github.com/peterljq/Parsimonious-Concept-Engineering"
)
EXPERIMENT_NUM_CONCEPTS = 100

EXPERIMENT_MODEL_COLLECTION_URL = (
    "https://huggingface.co/collections/allenai/olmo-3-post-training"
)
EXPERIMENT_DTYPE = "bfloat16"

# Think chain (ordered): base -> SFT -> DPO -> RL
# Keys index into OLMO3_VARIANTS. The three families SFT/DPO/RLVR
# are the last three entries; olmo3-base is the shared starting point.
THINK_CHAIN: list[str] = [
    "olmo3-base",
    "olmo3-think-sft",
    "olmo3-think-dpo",
    "olmo3-think-rlvr",
]

# RL-Zero family: RL directly from base (no SFT/DPO)
RL_ZERO_FAMILY: list[str] = [
    "olmo3-rl-zero-math",
    "olmo3-rl-zero-code",
    "olmo3-rl-zero-if",
    "olmo3-rl-zero-general",
    "olmo3-rl-zero-mix",
]

# Combined experiment checkpoint list (9 unique; base shared once).
# Early stage; will expand toward ~10 per TrainingDynamic.tex.
EXPERIMENT_MODELS: list[str] = THINK_CHAIN + RL_ZERO_FAMILY

# Layer selection: 10 layers at 10%, 20%, ..., 100% of model depth.
# Per TrainingDynamic.tex line 8.
EXPERIMENT_LAYER_PERCENTAGES: list[float] = [
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


def compute_experiment_layers(n_layers: int) -> list[int]:
    """Map depth percentages to 0-indexed layer indices."""
    return [min(int(p * n_layers), n_layers - 1) for p in EXPERIMENT_LAYER_PERCENTAGES]


# Pre-computed for OLMo-3 7B (32 transformer layers)
EXPERIMENT_LAYERS_7B: list[int] = compute_experiment_layers(32)


# =============================================================================
# Training Trajectory Checkpoints (10 uniform per model)
# =============================================================================


def select_uniform_checkpoints(
    all_steps: list[str],
    n: int = 10,
) -> list[str]:
    """Select n uniformly-spaced checkpoints from step branch names.

    Parses numeric step values from names like 'step5000' or 'step_300',
    sorts by step number, then picks n at uniform intervals spanning
    first to last (inclusive of the final step).
    """
    import re

    parsed: list[tuple[int, str]] = []
    for name in all_steps:
        m = re.search(r"step_?(\d+)", name)
        if m:
            parsed.append((int(m.group(1)), name))
    parsed.sort()

    if not parsed:
        return list(all_steps)

    if len(parsed) <= n:
        return [name for _, name in parsed]

    total = len(parsed)
    if n == 1:
        indices = [total - 1]
    else:
        indices = [
            min(int(round(i * (total - 1) / (n - 1))), total - 1) for i in range(n)
        ]
    seen: set[int] = set()
    result: list[str] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(parsed[idx][1])
    return result


_ALL_STEPS: dict[str, list[str]] = {
    "olmo3-think-sft": [f"step{s}" for s in range(1000, 44000, 1000)],
    "olmo3-instruct-sft": ["main"],
    "olmo3-rl-zero-math": [f"step_{s}" for s in range(100, 2000, 100)],
    "olmo3-rl-zero-code": [f"step_{s}" for s in range(100, 3000, 100)],
    "olmo3-rl-zero-if": [f"step_{s}" for s in range(100, 2000, 100)],
    "olmo3-rl-zero-general": [f"step_{s}" for s in range(100, 900, 100)],
    "olmo3-rl-zero-mix": [f"step_{s}" for s in range(50, 1000, 50)],
}

MODEL_CHECKPOINTS: dict[str, list[str]] = {
    model: select_uniform_checkpoints(steps, n=10)
    for model, steps in _ALL_STEPS.items()
}


# OLMo-3 pretraining stage checkpoints (if available as revisions)
# These may not all exist - the pipeline should handle missing gracefully
OLMO3_PRETRAIN_CHECKPOINTS: list[str] = [
    # Stage 1 (pretraining, ~1.4M steps, 5.93T tokens)
    "stage1-step0",
    "stage1-step100000",
    "stage1-step1000000",
    # Stage 2 (mid-training, ~47K steps, 100B tokens)
    "stage2-step10000",
    # Stage 3 (long context, ~12K steps, 50B tokens)
    "stage3-step10000",
]


# =============================================================================
# Weight Matrix Extraction Patterns
# =============================================================================


def get_weight_patterns(architecture: str) -> dict[str, list[str]]:
    """
    Get weight matrix name patterns grouped by type for a given architecture.

    Returns dict of {group_name: [list of substring patterns to match]}
    """
    if architecture == "gpt_neox":
        return {
            "attention_qkv": [
                "attention.query_key_value",
            ],
            "attention_dense": [
                "attention.dense",
            ],
            "mlp_dense_h_to_4h": [
                "mlp.dense_h_to_4h",
            ],
            "mlp_dense_4h_to_h": [
                "mlp.dense_4h_to_h",
            ],
        }
    elif architecture == "olmo3":
        return {
            "attention_q_proj": ["self_attn.q_proj"],
            "attention_k_proj": ["self_attn.k_proj"],
            "attention_v_proj": ["self_attn.v_proj"],
            "attention_o_proj": ["self_attn.o_proj"],
            "mlp_gate_proj": ["mlp.gate_proj"],
            "mlp_up_proj": ["mlp.up_proj"],
            "mlp_down_proj": ["mlp.down_proj"],
        }
    else:
        return {}


def categorize_weight(name: str, architecture: str) -> str:
    """
    Categorize a weight matrix by its layer type.

    Returns one of: 'attention_qkv', 'attention_output', 'mlp', 'embed', 'norm', 'other'
    """
    patterns = get_weight_patterns(architecture)

    for group, substrings in patterns.items():
        for substr in substrings:
            if substr in name:
                if (
                    "qkv" in group
                    or "q_proj" in group
                    or "k_proj" in group
                    or "v_proj" in group
                ):
                    return "attention_qkv"
                elif "dense" in group and "h_to_4h" not in group or "o_proj" in group:
                    return "attention_output"
                elif "mlp" in group:
                    return "mlp"

    if "embed" in name.lower():
        return "embed"
    if "norm" in name.lower():
        return "norm"
    return "other"


def get_layer_index(name: str) -> Optional[int]:
    """Extract layer index from weight name, or None if not a layer weight."""
    import re

    # Match patterns like "layers.0.", "layer.0.", "h.0.", "blocks.0."
    match = re.search(r"(?:layers?|h|blocks?)\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# Analysis Configuration
# =============================================================================

# Default analysis settings
DEFAULT_SETTINGS = {
    "svd_eps": 1e-10,
    "max_dim_for_svd": 8192,  # Skip matrices larger than this
    "use_float64": True,
    "batch_size": 1,  # Process one model at a time (memory)
}

# Color palettes for visualization
COLOR_PALETTES = {
    "layer_types": {
        "attention_qkv": "#e74c3c",
        "attention_output": "#3498db",
        "mlp": "#2ecc71",
        "embed": "#f39c12",
        "other": "#95a5a6",
    },
    "training_stages": {
        "pretrain": "#3498db",
        "midtrain": "#2ecc71",
        "longcontext": "#9b59b6",
        "sft": "#e74c3c",
        "dpo": "#f39c12",
        "rlvr": "#1abc9c",
    },
    "pathways": {
        "base": "#3498db",
        "think": "#e74c3c",
        "instruct": "#2ecc71",
        "rl-zero": "#f39c12",
    },
}
