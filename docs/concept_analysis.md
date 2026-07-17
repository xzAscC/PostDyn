# Concept Analysis: Training Dynamics in Activation Space

## Overview

This document describes four model-agnostic analysis metrics for studying how concepts evolve across training checkpoints and post-training methods. All metrics operate on `ConceptSteeringVector` data produced by the DIM pipeline in `src/concept_steering.py`.

The core question: **How are different concepts shaped across pre-training and post-training?** We treat a concept as a direction in activation space (diff-of-means), and track how that direction — and the geometry of a set of directions — evolves.

**Source**: `src/concept_analysis.py` (4 functions, 28 TDD tests)

## Metric 1: Directional Stability (Trajectory Analysis)

**Function**: `directional_stability(trajectory, eps=1e-10)`

**Purpose**: Measures how much a concept's steering vector moves across checkpoints. High stability means the concept direction is preserved; low stability means it shifts.

**Formula**:
```
stability(k; t, t') = cos(v_k^{(t)}, v_k^{(t')})
```

Where `v_k^{(t)}` is the steering vector of concept `k` at checkpoint `t`.

**Input**: `dict[str, dict[str, ConceptSteeringVector]]` — maps checkpoint IDs to per-concept steering vectors.

**Output**: `dict[str, Tensor]` — for each concept, a `(T, T)` cosine similarity matrix where `T` = number of checkpoints. Axes follow sorted checkpoint names. Diagonal = 1.0 (self-similarity).

**Caveat**: Comparing raw direction coordinates across checkpoints/models is confounded by basis drift. Consider Procrustes alignment or intrinsic quantities such as inter-concept angles for cross-model comparisons.

## Metric 2: Separability Margin (Cohen's d)

**Function**: `separability_margin(vec, eps=1e-10)`

**Purpose**: Quantifies how separable the positive and negative classes are along each activation dimension. Tracks whether post-training sharpens or blurs concept boundaries.

**Per-dimension formula**:
```
d_k^{(t)}[i] = (μ_p[i] - μ_n[i]) / sqrt(0.5 * (σ_p[i]² + σ_n[i]²))
```

The numerator is exactly the steering vector `v = μ_p - μ_n`.

**Scalar multivariate summary**:
```
D_k^{(t)} = ||v||₂ / sqrt(0.5 * (||σ_p||₂² + ||σ_n||₂²))
```

**Input**: A single `ConceptSteeringVector` (has `positive_mean`, `negative_mean`, `positive_std`, `negative_std`).

**Output**: `SeparabilityMargin` dataclass with:
- `margin_vector` — per-dimension Cohen's d, shape `(d_model,)`
- `scalar_summary` — multivariate Cohen's d as a float

**Edge case**: When both classes have zero variance (σ = 0), the eps guard prevents division by zero. The margin will be large but finite.

## Metric 3: Concept Gram Matrix

**Function**: `concept_gram_matrix(vectors, eps=1e-10)`

**Purpose**: Measures pairwise alignment (entanglement) among all concept steering vectors at a single checkpoint. Reveals whether post-training disentangles or compresses concepts.

**Formula**:
```
G_{ij}^{(t)} = cos(v_i^{(t)}, v_j^{(t)})
```

**Input**: `dict[str, ConceptSteeringVector]` — all concepts at one checkpoint.

**Output**: `(n_concepts, n_concepts)` symmetric tensor. Diagonal = 1.0 (self-alignment). Off-diagonal values in [-1, 1]. Axes follow sorted concept names.

**Interpretation**:
- Off-diagonal ≈ 0 → concepts are orthogonal (disentangled)
- High off-diagonal → concepts are aligned (entangled/compressed)
- Negative off-diagonal → concepts are anti-correlated

## Metric 4: Anisotropy Spectrum

**Function**: `anisotropy_spectrum(vectors, eps=1e-10)`

**Purpose**: Characterizes how isotropically concept vectors fill activation space. Post-training often pushes representations toward anisotropy — a dominant "rogue dimension" — which may be a by-product of certain concepts being amplified.

**Procedure**:
1. Stack all `n` steering vectors into matrix `V` of shape `(n, d_model)`
2. Center columns: `V_c = V - mean(V, dim=0)`
3. Form covariance: `Σ = V_c^T V_c / (n - 1)`
4. Compute eigenvalues: `λ₁ ≥ λ₂ ≥ ... ≥ λᵣ` where `r = min(n-1, d)`
5. Explained variance ratio: `ρᵢ = λᵢ / Σⱼ λⱼ`

**Input**: `dict[str, ConceptSteeringVector]` — all concepts at one checkpoint.

**Output**: `AnisotropySpectrum` dataclass with:
- `eigenvalues` — sorted descending over the activation/feature dimension, all ≥ 0
- `explained_variance_ratio` — same length as `eigenvalues`, sums to 1.0

**Interpretation**:
- Flat spectrum (all ρᵢ ≈ 1/r) → isotropic, concepts spread evenly
- Dominant first eigenvalue (ρ₁ >> 1/r) → anisotropic, a "rogue dimension" dominates
- Evolution across checkpoints: if a concept being amplified coincides with emergence of a new dominant direction, that is a trace of "concept shaping" in the global geometry

## Usage

```python
from src.concept_steering import load_steering_vectors
from src.concept_analysis import (
    directional_stability,
    separability_margin,
    concept_gram_matrix,
    anisotropy_spectrum,
)

# Load steering vectors for one checkpoint
vectors = load_steering_vectors("results/checkpoint_sft")

# Metric 2: Per-concept margin
for name, vec in vectors.items():
    margin = separability_margin(vec)
    print(f"{name}: Cohen's d = {margin.scalar_summary:.3f}")

# Metric 3: Gram matrix (concept entanglement)
gram = concept_gram_matrix(vectors)

# Metric 4: Anisotropy spectrum
spectrum = anisotropy_spectrum(vectors)
print(f"Top-5 explained variance: {spectrum.explained_variance_ratio[:5]}")

# Metric 1: Directional stability across checkpoints (requires multiple checkpoints)
trajectory = {
    "base": load_steering_vectors("results/checkpoint_base"),
    "sft": load_steering_vectors("results/checkpoint_sft"),
    "dpo": load_steering_vectors("results/checkpoint_dpo"),
}
stability = directional_stability(trajectory)
```

## Test Coverage

28 TDD tests in `tests/test_concept_analysis.py`:

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestDirectionalStability` | 6 | Diagonal=1, orthogonality, symmetry, shape, per-concept output, sorted order |
| `TestSeparabilityMargin` | 8 | Formula match, norm formula, positivity, zero-std eps guard, shape, name preservation, hand-computed values, multivariate Cohen's d |
| `TestConceptGramMatrix` | 6 | Diagonal=1, symmetry, shape, value range [-1,1], orthogonality, sorted order |
| `TestAnisotropySpectrum` | 8 | Descending order, ratio sums to 1, shape match, non-negativity, isotropic flat, rank-1 dominant, type check, centering |

These 28 tests live in `tests/test_concept_analysis.py`. The full suite currently has 300+ tests (`uv run pytest`).

## Limitations

1. **No Procrustes alignment**: Cross-model directional stability comparisons may be confounded by basis drift
2. **Separate from concept-dynamics extraction**: These metrics operate on pre-computed steering vectors. Model-side DiM extraction and checkpoint trajectories are implemented in `src/concept_dynamics.py` / `experiments/run_concept_dynamics.py`
3. **Cohen's d projection**: The per-dimension margin treats each axis independently; the scalar summary aggregates via L2 norms
4. **Covariance estimation**: With only 100 concepts, the covariance estimate may be noisy for d_model=4096
