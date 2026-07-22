# Experiment Setup — PaCE Concept Steering on OLMo-3 (Think Focus)

## Overview

This experiment applies difference-in-means (DIM) concept steering to OLMo-3 7B post-training checkpoints, using 100 concepts from the PaCE dictionary. The setup is adapted from `TrainingDynamic.tex` and represents an early-stage investigation focused on the Think pathway.

## Concepts

**Source**: PaCE (Parsimonious Concept Engineering), https://github.com/peterljq/Parsimonious-Concept-Engineering

**Count**: First 100 of PaCE's ~1 million frequency-ranked concepts. The full index (40,000 concepts in the public release) is stored in `data/concept_index.txt`. Selection uses `select_concepts()` from `src/concept_steering.py` with the default `strategy="first"`.

## Models

**Collection**: https://huggingface.co/collections/allenai/olmo-3-post-training

**Precision**: bfloat16 (`EXPERIMENT_DTYPE`)

All models are OLMo-3 7B variants sharing the same architecture (32 layers, d_model=4096). Configurations are defined in `src/config.py` under `OLMO3_VARIANTS`.

| Key | HuggingFace ID | Pathway | Stage |
|-----|----------------|---------|-------|
| olmo3-base | allenai/Olmo-3-1025-7B | base | base |
| olmo3-think-sft | allenai/Olmo-3-7B-Think-SFT | think | sft |
| olmo3-think-dpo | allenai/Olmo-3-7B-Think-DPO | think | dpo |
| olmo3-think-rlvr | allenai/Olmo-3-7B-Think | think | rlvr |
| olmo3-rl-zero-math | allenai/Olmo-3-7B-RL-Zero-Math | rl-zero | rlvr-math |
| olmo3-rl-zero-code | allenai/Olmo-3-7B-RL-Zero-Code | rl-zero | rlvr-code |
| olmo3-rl-zero-if | allenai/Olmo-3-7B-RL-Zero-IF | rl-zero | rlvr-if |
| olmo3-rl-zero-general | allenai/Olmo-3-7B-RL-Zero-General | rl-zero | rlvr-general |
| olmo3-rl-zero-mix | allenai/Olmo-3-7B-RL-Zero-Mix | rl-zero | rlvr-mix |

### Think Chain (reasoning / chain-of-thought)

The Think chain traces a clean three-stage post-training pipeline, each stage released as a separate checkpoint:

```
base → olmo3-think-sft → olmo3-think-dpo → olmo3-think-rlvr
```

The three post-training families map to Think chain stages:
- **SFT**: `olmo3-think-sft` (supervised fine-tuning)
- **DPO**: `olmo3-think-dpo` (direct preference optimization)
- **RLVR**: `olmo3-think-rlvr` (reinforcement learning with verifiable rewards)

### RL-Zero Family (RL directly from base)

RL-Zero models perform RL directly from the base model, skipping SFT and DPO entirely:

```
base → {olmo3-rl-zero-math, olmo3-rl-zero-code, olmo3-rl-zero-if,
        olmo3-rl-zero-general, olmo3-rl-zero-mix}
```

Each variant trains on a different reward domain (math, code, instruction-following, general, or a mixture). These are distinct from the Think chain's RLVR stage — they isolate the effect of RL without prior SFT/DPO.

### Families

The experiment covers three post-training method families:
- **SFT** — represented by `olmo3-think-sft`
- **DPO** — represented by `olmo3-think-dpo`
- **RLVR** — represented by `olmo3-think-rlvr` (Think chain) and the five RL-Zero variants (RL from base)

`olmo3-base` serves as the shared pre-training root for both the Think chain and the RL-Zero family.

## Scope

- **Focus**: Think pathway variants (base + Think chain + RL-Zero family = 9 checkpoints)
- **Deferred**: Instruct pathway variants (`olmo3-instruct-sft`, `olmo3-instruct-dpo`, `olmo3-instruct-rlvr`) are excluded from the current experiment
- **Checkpoint count**: 9 unique models currently; the experimental design targets ~10, with room to expand

## Layer Selection

Per the experimental setup, the DiM pipeline is applied at 10 depth positions
selected via the slide formula `ℓ_j = round[(0.1 + 0.8·j/9)·(L-1)]` for
`j = 0..9`. For OLMo-3 7B (32 layers, `L-1 = 31`) this yields layer indices
`[3, 6, 9, 11, 14, 17, 20, 22, 25, 28]`.

The formula is implemented in `src/config.py`:

- `compute_experiment_layers(n_layers, n=10)` — applies the slide formula
- `EXPERIMENT_LAYERS_7B` — pre-computed for the 32-layer architecture
- `src.concept_dynamics.select_uniform_layers(n_layers, n=10)` — the same
  formula used inside the extraction pipeline

## Method: Difference-in-Means (DIM)

For each concept, the steering vector is computed as the difference between the mean activations of positive and negative stimulus sets:

```
steering_vector = mean(positive_activations) − mean(negative_activations)
```

**Saved per concept** (via `compute_steering_vector()` in `src/concept_steering.py`):
- **Steering vector** — DIM direction in activation space (d_model)
- **Positive mean** — mean of positive-class activations
- **Negative mean** — mean of negative-class activations
- **Positive std** — spread of positive-class activations (Bessel's correction)
- **Negative std** — spread of negative-class activations (Bessel's correction)

Results are stored as safetensors + JSON via `save_steering_vectors()`, using indexed keys to handle concept names with special characters.

## Limitations

1. **Early stage**: Only 9 of the target ~10 checkpoints are configured; Instruct pathway is deferred
2. **Layer selection**: 10 depth positions specified (10%–100%); the activation extraction pipeline to probe these layers is not yet implemented
3. **Activation extraction**: The model-specific activation extraction pipeline (Phase 2) is not yet implemented — the DIM computation module is model-agnostic and tested with synthetic data
4. **Single precision**: All models use bfloat16; no mixed-precision or quantization comparison
5. **Concept coverage**: 100 concepts is a small subset of PaCE's full dictionary; results may not generalize to the complete concept space
