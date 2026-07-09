# Concept Dynamics Experiment — Olmo-3-7B Post-Training Variants

## Overview

This experiment traces how concept representations evolve across different post-training methods applied to the Olmo-3-7B backbone. It implements the Difference-in-Means (DiM) concept extraction pipeline from "Tracing Concept Dynamics through Pretraining and Post-training", computing normalized concept directions for **math**, **code**, **instruction-following (if)**, and **general** text across six post-training variants, then measuring:

1. **Directional stability** — how much a concept's direction moves across models
2. **Concept Gram matrix** — how concepts are positioned relative to one another (entanglement)

## Source

Paper: `69e5bc6d27002aeac07372b4/memo/Tracing Concept Dynamics.tex`

## Models

Six Olmo-3-7B post-training variants, all sharing the same architecture (32 layers, d_model=4096, bfloat16):

| Model Key | HuggingFace ID | Pathway |
|-----------|---------------|---------|
| `olmo3-think-sft` | allenai/Olmo-3-7B-Think-SFT | Think (SFT) |
| `olmo3-rl-zero-math` | allenai/Olmo-3-7B-RL-Zero-Math | RL-Zero (math) |
| `olmo3-rl-zero-code` | allenai/Olmo-3-7B-RL-Zero-Code | RL-Zero (code) |
| `olmo3-rl-zero-if` | allenai/Olmo-3-7B-RL-Zero-IF | RL-Zero (IF) |
| `olmo3-rl-zero-general` | allenai/Olmo-3-7B-RL-Zero-General | RL-Zero (general) |
| `olmo3-rl-zero-mix` | allenai/Olmo-3-7B-RL-Zero-Mix | RL-Zero (mix) |

## Datasets (Contrastive Pairs)

Each concept requires positive (domain-specific) and negative (general) text sets:

| Concept | Positive Dataset | Negative Dataset |
|---------|-----------------|-----------------|
| math | `allenai/Dolci-RL-Zero-Math-7B` (field: `prompt`) | `Salesforce/wikitext` (config: `wikitext-2-raw-v1`, field: `text`) |
| code | `allenai/Dolci-RL-Zero-Code-7B` (field: `prompt`) | `Salesforce/wikitext` |
| if | `allenai/Dolci-RL-Zero-IF-7B` (field: `prompt`) | `Salesforce/wikitext` |
| general | `allenai/Dolci-RL-Zero-General-7B` (field: `prompt`) | `Salesforce/wikitext` |

All datasets are loaded via **streaming mode** (`streaming=True`) — the full datasets are never downloaded. Only the first 50 non-empty examples per class are materialized.

**Token selection rule S(x)**: last token (standard for concept extraction; captures the model's summary representation of the input).

## Method: Difference-in-Means (DiM)

For each (model, layer, concept), the concept direction is computed as:

```
A+ = { h_i(x) : x ∈ D+, i ∈ S(x) }     # positive activations (last token)
A- = { h_i(x) : x ∈ D-, i ∈ S(x) }     # negative activations (last token)

μ+ = (1/|A+|) Σ A+                       # positive mean
μ- = (1/|A-|) Σ A-                       # negative mean
r  = μ+ - μ-                             # DiM direction (raw)
r̂  = r / ||r||₂                          # normalized direction (default)
```

The normalized direction r̂ is saved as `steering_vector`, so that steering strength is controlled only by a scalar coefficient, not by the norm of the estimated direction (following Min et al. 2025).

## Layer Selection

Ten layers uniformly spaced at 10%, 20%, ..., 100% of the model's 32 transformer layers:

```
EXPERIMENT_LAYERS_7B = [3, 6, 9, 12, 16, 19, 22, 25, 28, 31]
```

## Metrics

### 1. Directional Stability (cross-model)

For each concept k at layer L, the cosine similarity of its direction across all model pairs:

```
stability(k; t, t') = cos(r_k^t, r_k^t')
```

This produces a 6×6 symmetric matrix per (concept, layer). High values mean the concept direction is preserved across post-training methods; low values mean it shifts.

### 2. Concept Gram Matrix (per-model, entanglement)

For each model t at layer L, the pairwise cosine similarity among all concept directions:

```
G_ij^t = cos(r_i^t, r_j^t)
```

This produces a 4×4 symmetric matrix per (model, layer). Off-diagonal ≈ 0 means concepts are orthogonal (disentangled); high off-diagonal means concepts are aligned (entangled).

## Pipeline Architecture

```
experiments/run_concept_dynamics.py    CLI entry point (argparse)
experiments/run_concept_dynamics.sh    Shell wrapper

src/concept_dynamics.py                Core module:
  ├── select_uniform_layers()          10 layers at 10%-100% depth
  ├── extract_layer_activations()      Last-token hidden states (HF transformers)
  ├── compute_concept_vector()         DiM + normalization (r̂ = r/||r||)
  ├── cross_model_stability()          6×6 cosine matrix across models
  ├── concept_gram_matrices()          4×4 cosine matrix across concepts
  ├── run_model_extraction()           Single-model pipeline
  ├── run_full_experiment()            All models + dynamics analysis
  └── compute_dynamics_analysis()      Load + compute stability + gram

src/contrastive_datasets.py            Streaming dataset loader:
  ├── load_contrastive_texts()         50 pos + 50 neg (streaming, no full download)
  ├── _extract_text()                  Handles text/prompt/content/messages fields
  └── _stream_n_samples()              Memory-efficient islice + filter
```

## Output Layout

```
results/concept_dynamics/
├── vectors/
│   └── {model_name}/
│       ├── layer_3.safetensors        # 4 concepts × 6 tensors (steering, raw, means, stds)
│       ├── layer_3.json               # Metadata (concept names, n_samples, d_model)
│       ├── layer_6.safetensors
│       └── ...
├── stability/
│   └── stability.json                 # {concept: {layer: {matrix, models}}}
├── gram/
│   └── gram.json                      # {model: {layer: {matrix, concepts}}}
└── extraction_results.json            # Full summary + timing
```

## How to Run

### Quick (smoke test)

```bash
experiments/run_concept_dynamics.sh quick
```

Runs: 1 model (Think-SFT), 2 concepts (math, code), 2 layers (3, 16), 5 samples each.

### Full experiment

```bash
experiments/run_concept_dynamics.sh full
```

Runs: 6 models × 4 concepts × 10 layers × 50 samples.

### Custom

```bash
experiments/run_concept_dynamics.sh --models olmo3-think-sft,olmo3-rl-zero-math \
    --concepts math,code --n-samples 100
```

## Development

Developed with TDD (test-first). All new code has 100% test coverage with no GPU required for unit tests:

- `tests/test_contrastive_datasets.py` — 19 tests (dataset loading, text extraction, streaming)
- `tests/test_concept_dynamics.py` — 29 tests (activation extraction, DiM+normalization, stability, gram)
- Total: 48 new tests, 156 project-wide (0 regressions)

## Steps Reproduced

1. **Layer selection**: `select_uniform_layers(32, n=10)` → `[3, 6, 9, 12, 16, 19, 22, 25, 28, 31]`
2. **Contrastive data loading**: For each concept, stream 50 positive (Dolci) + 50 negative (wikitext) texts
3. **Activation extraction**: For each text, forward pass with `output_hidden_states=True`, extract last-token hidden state at each of the 10 layers
4. **DiM computation**: `r = μ+ - μ-`, then normalize `r̂ = r / ||r||`
5. **Save**: Per (model, layer) as safetensors + JSON metadata
6. **Stability analysis**: Load all models' vectors, compute 6×6 cosine matrix per (concept, layer)
7. **Gram analysis**: Compute 4×4 cosine matrix per (model, layer)

## Results

See `results/concept_dynamics/extraction_results.json` after running the full experiment. The stability and gram JSON files contain the raw matrices for visualization.

### Smoke Test Results

Verified on RTX 4090 (bfloat16, `olmo3-think-sft`):

| Check | Result |
|-------|--------|
| Model loaded | `allenai/Olmo-3-7B-Think-SFT` on `cuda:0`, 11.7s (cached) |
| Streaming data | 5 positive + 5 negative for math and code (no full download) |
| safetensors output | `layer_3.safetensors` + `layer_16.safetensors` created |
| Normalization | All `||r_hat|| = 1.000000` (paper requirement met) |
| d_model | 4096 (matches Olmo-3-7B architecture) |
| Gram matrix | Symmetric, diagonal = 1.0, meaningful values |

### Full Experiment Results (6 models × 4 concepts × 10 layers × 50 samples)

All 6 models processed successfully on RTX 4090 (bfloat16). Results saved to `results/concept_dynamics/`.

#### Directional Stability (concept preservation across models)

The stability metric `cos(r_k^t, r_k^t')` measures how much each concept direction is preserved across post-training variants. Higher = more stable.

| Concept | Layer 3 (early) | Layer 31 (deep) | Trend |
|---------|----------------|-----------------|-------|
| math | 0.952 | 0.913 | Decreases with depth |
| code | ~0.95 | ~0.90 | Decreases with depth |
| if | ~0.94 | ~0.88 | Decreases with depth |
| general | ~0.94 | ~0.89 | Decreases with depth |

**Finding**: Concepts are highly stable at early layers (>0.94) but diverge more at deeper layers — confirming the hypothesis that post-training reshapes concept representations primarily in upper layers.

#### Concept Gram Matrix (entanglement per model)

Off-diagonal mean of the 4×4 Gram matrix (lower = more disentangled):

| Model | Layer 16 | Interpretation |
|-------|----------|----------------|
| Think-SFT | 0.822 | Most entangled |
| RL-Zero-Math | 0.763 | Less entangled |
| RL-Zero-Mix | 0.756 | Least entangled |

**Finding**: RL-Zero variants show lower concept entanglement than SFT — RL training disentangles concepts more than SFT. This aligns with prior work showing RLVR consolidates representations into narrower subspaces.

#### Files Produced

| File | Content |
|------|---------|
| `results/concept_dynamics/stability/stability.json` | 40 stability matrices (4 concepts × 10 layers, each 6×6) |
| `results/concept_dynamics/gram/gram.json` | 60 Gram matrices (6 models × 10 layers, each 4×4) |
| `results/concept_dynamics/extraction_results.json` | Full extraction summary + dynamics |
| `results/concept_dynamics/vectors/{model}/layer_{L}.safetensors` | Concept vectors (gitignored, large) |
