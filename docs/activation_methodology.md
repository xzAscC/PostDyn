# Activation-Level Effective Rank Methodology

## Overview

This document describes the activation-level effective rank analysis, which complements the weight-level analysis described in `methodology.md`. While the weight analysis examines the spectral properties of model parameters, the activation analysis examines the spectral properties of model representations (hidden states) when processing real data.

This analysis follows the methodology of:

1. **Li et al. (2025)** — "Tracing the Representation Geometry of Language Models from Pretraining to Post-training" (arXiv:2509.23024)
   - Introduced RankMe and α-ReQ as metrics for representation geometry
   - Discovered three universal phases in pretraining: warmup → entropy-seeking → compression-seeking
   - Analyzed OLMo (1B-7B) and Pythia (160M-12B) using FineWeb dataset

2. **Roy & Vetterli (2007)** — Original effective rank definition via Von Neumann entropy

## Key Difference from Weight Analysis

| Aspect | Weight Analysis | Activation Analysis |
|--------|----------------|---------------------|
| **Input to SVD** | Weight matrix W ∈ ℝ^{m×n} | Feature matrix F ∈ ℝ^{M×d} |
| **Data dependency** | None (static parameters) | Depends on input data distribution |
| **What it measures** | Parameter space utilization | Representation space utilization |
| **Computation** | SVD of W directly | SVD of centered F, then σ² for covariance eigenvalues |
| **Formula** | p_i = σ_i / Σσ_j | p_i = λ_i / Σλ_j where λ_i = σ_i² |

## Methodology

### 1. Dataset

We use **MMLU Pro** (TIGER-Lab/MMLU-Pro) instead of the FineWeb dataset used in the original paper. MMLU Pro contains 12,032 expert-level multiple-choice questions across 14 academic domains (math, physics, chemistry, law, engineering, etc.).

**Rationale**: MMLU Pro provides diverse, high-quality text that probes the model's representation geometry across different knowledge domains. While the original paper uses web text (FineWeb), our choice enables future per-category analysis.

**Prompt format**: Raw question text only (`example["question"]`), without options or answer prefixes. This matches the paper's approach of probing natural representations rather than task-specific behavior.

### 2. Feature Matrix Construction

Following Li et al. (Section 2.1):

Given M input sequences, we form a feature matrix **F** ∈ ℝ^{M×d}:
- Each row is the hidden state of the **last token** at a given layer
- M = number of MMLU Pro samples (configurable, default 500)
- d = model hidden dimension (d_model)

**Extraction**:
```
For each question:
  1. Tokenize (truncate to max_seq_len=512)
  2. Forward pass through model with output_hidden_states=True
  3. For each layer L:
     Extract y_N^(L) = hidden_state at last token position
  4. Stack into per-layer feature matrices
```

**Preprocessing**: Features are centered before SVD computation:
```
F_centered = F - mean(F, axis=0)
```

This is critical — the paper explicitly centers features before computing the eigenspectrum, following the standard PCA convention.

### 3. RankMe (Effective Rank of Activations)

**Formula** (Li et al., Equation 1):

```
RankMe := exp(S(Σ̂)) = exp(-Σ_{i=1}^{d} p_i · ln(p_i))
```

Where:
- Σ̂ = (1/M) · F_centered^T · F_centered is the empirical covariance matrix
- {σ_i} are eigenvalues of Σ̂ (equivalently, squared singular values of F_centered / √M)
- p_i = σ_i / Σ_j σ_j is the proportion of variance along the i-th principal axis
- S(Σ̂) is the Von Neumann entropy of the normalized eigenvalue distribution

**Implementation**: We compute SVD of F_centered, then square the singular values to obtain the covariance eigenspectrum (avoiding explicit covariance matrix computation):
```python
s = torch.linalg.svdvals(F_centered)    # Singular values
eigenvalues = s ** 2                     # Covariance eigenspectrum (λ_i = σ_i²)
p = eigenvalues / eigenvalues.sum()      # Normalize to probability distribution
H = -torch.sum(p * torch.log(p))         # Shannon entropy
RankMe = torch.exp(H)
```

**Range**: (0, d] where d is the hidden dimension

### 4. RankMe Ratio

```
ratio = RankMe / d ∈ (0, 1]
```

This normalizes for different hidden dimensions across model sizes, enabling cross-model comparison. A ratio of 1.0 means isotropic (all singular values equal); lower values indicate increasing anisotropy.

### 5. α-ReQ (Power-Law Decay Rate)

Following Agrawal et al. (2022), we estimate the power-law decay exponent of the eigenspectrum:

```
σ_i ∝ i^{-α_ReQ}
```

Estimated via weighted least-squares regression on log(λ_i) vs log(i), typically over the range i ∈ [10, 100):
```python
eigenvalues = singular_values ** 2      # Covariance eigenspectrum
log_λ = log(eigenvalues[fit_range])
log_i = log(indices[fit_range])
α_ReQ = -slope(log_i vs log_λ)
```

**Interpretation**:
- Higher α → faster eigenvalue decay → more anisotropic → lower effective dimensionality
- Lower α → slower decay → more isotropic → higher effective dimensionality

### 6. Per-Layer Analysis

For each model, we compute RankMe and α-ReQ at **every transformer layer** (not just the last layer). We extract `hidden_states[1..num_layers]`, skipping the embedding output (`hidden_states[0]`). Layer indices 0 through `num_layers-1` in the results correspond to transformer layers 1 through `num_layers`. This reveals how representation geometry evolves through network depth:

- **Early layers** (0–⅓ depth): Local syntactic features (expected: lower RankMe)
- **Middle layers** (⅓–⅔ depth): Semantic composition (expected: RankMe transition)
- **Late layers** (⅔–full depth): Task-relevant representations (expected: varies by model capability)

### 7. Analysis Modes

#### Cross-Model Comparison
Compare activation RankMe across 8 Pythia model sizes (70M → 12B). Tests whether representation geometry is consistent across scales.

#### Training Dynamics
Track activation RankMe over Pythia training checkpoints (step0 → step143000). Tests for the three-phase pattern discovered by Li et al.

## Computational Considerations

### Memory Management
- **Model loading**: `device_map="auto"` with `torch_dtype=bfloat16` for automatic GPU/CPU distribution
- **Hidden state extraction**: Batch size 1, immediate CPU offload (`hidden.cpu().float()`)
- **Feature matrix**: (M × d × num_layers × 4 bytes) in float32 on CPU
  - For Pythia-12B (d=5120, 36 layers, 500 samples): ~370MB
  - For Pythia-70M (d=512, 6 layers, 500 samples): ~6MB
- **GPU memory**: Peak usage depends on model size; `device_map="auto"` handles offloading automatically

### Runtime Estimates (500 samples)
| Model | Layers | d_model | Forward time |
|-------|--------|---------|-------------|
| Pythia-70m | 6 | 512 | ~1s |
| Pythia-160m | 12 | 768 | ~1s |
| Pythia-410m | 24 | 1024 | ~3s |
| Pythia-1b | 16 | 2048 | ~3s |
| Pythia-1.4b | 24 | 2048 | ~4s |
| Pythia-2.8b | 32 | 2560 | ~7s |
| Pythia-6.9b | 32 | 4096 | ~12s |
| Pythia-12b | 36 | 5120 | ~70s (CPU offloaded) |

## Limitations

1. **Sample count**: 500 samples is fewer than the paper's 15,000 — may not capture full representation geometry
2. **Dataset difference**: MMLU Pro (questions) vs FineWeb (web text) — different text distributions may yield different RankMe values
3. **No per-category analysis yet**: MMLU Pro has 14 categories that could show domain-dependent rank structure
4. **Single-seed**: No variance estimation from multiple random subsets
5. **Weight vs activation comparison**: Direct comparison between weight RankMe and activation RankMe requires careful interpretation — they measure fundamentally different things
