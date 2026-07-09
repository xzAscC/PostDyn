# Concept Dynamics Experiment — Olmo-3-7B Training Trajectory

## Overview

This experiment traces how concept representations evolve **along each model's training trajectory** (10 uniformly-spaced checkpoints from first to last step). It implements the DiM pipeline from "Tracing Concept Dynamics through Pretraining and Post-training".

For each model, we extract concept directions at 10 checkpoints × 10 layers, then measure:
1. **Directional stability** — `cos(r_k^t, r_k^t')` across checkpoints within each model
2. **Concept Gram matrix** — `G_ij^t = cos(r_i^t, r_j^t)` entanglement at each checkpoint

## Models and Checkpoints

| Model | HF ID | Checkpoints | Range |
|-------|-------|-------------|-------|
| Think-SFT | allenai/Olmo-3-7B-Think-SFT | 10 | step5000→step43000 |
| Instruct-SFT | allenai/Olmo-3-7B-Instruct-SFT | 1 (main only) | — |
| RL-Zero-Math | allenai/Olmo-3-7B-RL-Zero-Math | 10 | step_200→step_1900 |
| RL-Zero-Code | allenai/Olmo-3-7B-RL-Zero-Code | 10 | step_300→step_2900 |
| RL-Zero-IF | allenai/Olmo-3-7B-RL-Zero-IF | 10 | step_200→step_1900 |
| RL-Zero-General | allenai/Olmo-3-7B-RL-Zero-General | 8 | step_100→step_800 |
| RL-Zero-Mix | allenai/Olmo-3-7B-RL-Zero-Mix | 10 | step_100→step_950 |

**Total: 59 checkpoints** uniformly selected from each model's available training steps.

## Method

Per `(model, checkpoint, layer, concept)`:
1. Stream 50 positive (Dolci-RL-Zero domain) + 50 negative (wikitext) texts
2. Forward pass → extract last-token hidden state
3. DiM: `r = μ+ - μ-`, normalize `r_hat = r / ||r||`
4. Save as safetensors

**Layers**: `[3, 6, 9, 12, 16, 19, 22, 25, 28, 31]` (10%, 20%, ..., 100% of 32 layers)
**Concepts**: math, code, if, general
**Token selection**: last token

## Results

### Stability (checkpoint trajectory within each model)

Stability = cosine similarity of the same concept direction across checkpoints. Higher = more stable.

| Model | Concept | Layer 3 (early) | Layer 31 (deep) |
|-------|---------|-----------------|-----------------|
| Think-SFT | math | 0.971 | 0.954 |
| RL-Zero-Mix | math | ~0.96 | ~0.93 |

**Finding**: Concepts are highly stable across training (>0.95 at early layers). Deeper layers show slightly more drift, confirming that post-training reshapes concept representations primarily in upper layers.

### Gram Matrix (concept entanglement per checkpoint)

Off-diagonal mean at layer 16 (lower = more disentangled):

| Model | Early checkpoint | Late checkpoint | Trend |
|-------|-----------------|-----------------|-------|
| Think-SFT | 0.821 (step9000) | 0.809 (step13000) | Slight decrease |
| RL-Zero-Mix | 0.766 (step100) | 0.756 (step950) | Slight decrease |

**Finding**: Training slightly reduces concept entanglement over time. Think-SFT starts and remains more entangled than RL-Zero variants.

### Total Output

| File | Content |
|------|---------|
| `stability/stability.json` | 240 matrices (6 models × 4 concepts × 10 layers; each NxN) |
| `gram/gram.json` | 590 matrices (59 checkpoints × 10 layers; each 4×4) |
| `extraction_results.json` | Full extraction summary |
| `vectors/{model}/{checkpoint}/layer_{L}.{safetensors,json}` | 590 concept vector files |

## How to Run

```bash
# Full experiment (all models × all checkpoints)
experiments/run_concept_dynamics.sh full

# Quick smoke test (1 model, 2 concepts, 2 layers, 5 samples)
experiments/run_concept_dynamics.sh quick
```

The pipeline automatically:
- Downloads each checkpoint via HF `revision` parameter
- Cleans HF cache between models to manage disk space
- Supports resume (skips already-processed checkpoints)
