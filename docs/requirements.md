# Project Requirements — PaCE Concept Steering for OLMo-3

## Requirement 1: Concept Source & Steering Pipeline

**Source**: [PaCE: Parsimonious Concept Engineering](https://github.com/peterljq/Parsimonious-Concept-Engineering) (NeurIPS 2024)

- **Concept dictionary**: `data/concept_index.txt` — 40,000 frequency-ranked concepts from PaCE-1M
- **Selection**: 100 concepts per run (default: top-100 by frequency, or random sampling)
- **Steering method**: Difference-in-Means (DIM)
  ```
  steering_vector = mean(positive_activations) - mean(negative_activations)
  ```
- **Statistics recorded per concept**:
  - `steering_vector` — DIM direction in activation space
  - `positive_mean` — mean of positive-class activations
  - `negative_mean` — mean of negative-class activations
  - `positive_std` — spread of positive-class activations
  - `negative_std` — spread of negative-class activations
- **Storage**: safetensors (tensors) + JSON (metadata), model-agnostic format

**Current status**: Phase 1 complete — model-agnostic DIM pipeline built with TDD (30 tests).
Activation extraction (model-specific) is Phase 2.

---

## Requirement 2: Model Selection (Phase 2)

**Models**: [OLMo-3 7B Post-Training Collection](https://huggingface.co/collections/allenai/olmo-3-post-training)

All models use **bfloat16** precision. Three post-training chains share the same 7B base:

### Think Chain (reasoning / chain-of-thought)
```
base → Olmo-3-7B-Think-SFT → Olmo-3-7B-Think-DPO → Olmo-3-7B-Think (RL)
```
Clean SFT→DPO→RL three-stage pipeline, each stage is a separate checkpoint.

### Instruct Chain (chat / tool-use)
```
base → Olmo-3-7B-Instruct-SFT → Olmo-3-7B-Instruct-DPO → Olmo-3-7B-Instruct (RL)
```

### RL-Zero Chain (RL directly from base, no SFT/DPO)
```
base → RL-Zero-{Math, Code, IF, General, Mix}
```

All model configs are defined in `src/config.py` under `OLMO3_VARIANTS`.

---

## Architecture

```
Phase 1 (DONE):
  data/concept_index.txt          40K concepts (PaCE)
  src/concept_steering.py         DIM pipeline (model-agnostic)
  tests/test_concept_steering.py  30 TDD tests

Phase 2 (TODO):
  src/activation_extractor.py     Extract activations from OLMo-3 models (bfloat16)
  → feeds into concept_steering.compute_steering_vector()
```
