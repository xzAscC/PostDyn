# Methodology: Effective Rank Analysis of LLM Weight Matrices

## Background

This research follows the methodology established by two key works:

1. **苏剑林 (2026)** - "矩阵参数的奇异值熵越高越好吗？" (kexue.fm/archives/11767)
   - Introduced the question: is there an optimal singular value entropy?
   - Used in the Moonlight technical report (Muon optimizer) to compare training methods
   - Computed SVD entropy of weight matrices directly (not activations)

2. **Li et al. (2025)** - "Tracing the Representation Geometry of Language Models from Pretraining to Post-training" (arXiv:2509.23024)
   - Analyzed hidden state representations using RankMe (effective rank)
   - Discovered three universal phases in pretraining geometry
   - Found post-training transforms: SFT/DPO = entropy-seeking, RLVR = compression-seeking

## Our Approach: Weight-Level Analysis

Unlike Li et al. who analyzed representation geometry (hidden states), we analyze the **weight matrices** directly, following the Moonlight paper approach. This reveals how the model's parameter space is utilized.

### Effective Rank Definition (Roy & Vetterli, 2007)

For a weight matrix W ∈ ℝ^{m×n} with singular values σ₁ ≥ σ₂ ≥ ... ≥ σₖ > 0 (k = min(m,n)):

**Step 1**: Normalize to probability distribution:
```
pᵢ = σᵢ / Σⱼ σⱼ
```

**Step 2**: Compute Shannon entropy:
```
H = -Σᵢ pᵢ · ln(pᵢ)
```

**Step 3**: Effective rank:
```
erank(W) = exp(H)
```

**Properties**:
- Range: [1, min(m,n)]
- Scale invariant: erank(cW) = erank(W)
- Unitary invariant: erank(UW) = erank(W)
- erank = 1 ⟺ rank-1 matrix (one dominant singular value)
- erank = min(m,n) ⟺ all singular values equal (isotropic)

### Effective Rank Ratio

To compare across layers of different dimensions:
```
ratio = erank(W) / min(m, n) ∈ (0, 1]
```

This normalized metric tells us what fraction of the theoretical rank capacity is actually utilized.

### Normalized SVD Entropy (Moonlight Style)

```
H_norm = -(1/log n) · Σᵢ qᵢ · log(qᵢ)  where  qᵢ = σᵢ² / Σσⱼ²
```

Range: [0, 1] where 0 = rank-1, 1 = isotropic.

## Analysis Dimensions

### 1. Cross-Model-Size (Scaling Laws)

**Hypothesis**: The effective rank ratio is approximately constant across model scales.

**Method**: Load final checkpoints of Pythia models from 70M to 12B parameters. Compute erank ratio for every linear weight matrix. Compare mean ratios across scales.

**Expected**: If the hypothesis holds, models of all sizes would use approximately the same fraction of their parameter space, suggesting a fundamental constraint on information utilization.

### 2. Training Dynamics (Temporal Evolution)

**Hypothesis**: Weight-level effective rank shows characteristic phases during training, analogous to the representation-level phases found by Li et al.

**Method**: Load Pythia-70m checkpoints from step0 to step143000 (25 checkpoints). Track per-layer-type erank ratio over training.

**Expected**: Initial randomness → low utilization (warmup), then increasing utilization (entropy-seeking), then consolidation (compression-seeking).

### 3. Training Stages (OLMo-3 Pretraining)

**Hypothesis**: Different pretraining stages (broad data → targeted data → long context) produce distinct effective rank signatures.

**Method**: Load OLMo-3 base model checkpoints from stages 1, 2, and 3 (if available as HuggingFace revisions).

### 4. Post-Training Methods (OLMo-3 Pathways)

**Hypothesis**: Different post-training methods produce different effective rank patterns:
- SFT: entropy-seeking (expansion)
- DPO: further entropy-seeking
- RLVR: compression-seeking (consolidation)
- RL-Zero: unknown (RL directly from base)

**Method**: Load all OLMo-3 variants (Think, Instruct, RL-Zero) at each post-training stage.

### 5. Fixed Ratio Hypothesis

**Hypothesis**: There exists a consistent effective rank ratio "space" that weight matrices converge to, regardless of model size, training stage, or method.

**Method**: Aggregate all observed ratios from analyses 1-4. Compute coefficient of variation (CV). If CV < 0.2, the ratio is approximately constant.

**Possible outcomes**:
- **Constant ratio**: Suggests a fundamental constraint on neural network parameter utilization
- **Variable by layer type**: Suggests different components have different capacity requirements
- **Variable by training stage**: Suggests effective rank is a useful training progress indicator
- **Variable by method**: Suggests different training methods use parameter space differently

## Computational Considerations

- **Memory**: Load models to CPU with float32. Extract weights, delete model, run SVD.
- **SVD stability**: Use float64 for SVD computation. Filter singular values > ε.
- **Large matrices**: For dimensions > 8192, consider truncated SVD (not needed for Pythia/OLMo-3).
- **Checkpoint loading**: Pythia uses Git branches (`revision="step1000"`).

## Limitations

1. Weight-level analysis doesn't capture activation-level dynamics
2. Entropy-based effective rank is sensitive to near-zero singular values
3. OLMo-3 stage checkpoints may not all be publicly available
4. Large models (6.9B, 12B) require significant RAM for full SVD
5. Results may not generalize to other architectures (e.g., MoE, attention-free)
