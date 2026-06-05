# Design Document: Effective Rank Analysis of Open-Source LLMs

## Project Overview

This project analyzes the effective rank of weight matrices in open-source large language models (LLMs) to investigate whether a consistent effective rank ratio exists across different model scales, training stages, and post-training methods.

## Research Questions

1. **Cross-model-size**: Does the effective rank ratio `erank(W) / min(m,n)` remain constant as model scale increases?
2. **Training dynamics**: How does effective rank evolve during pretraining? Do we observe the three-phase pattern (warmup → entropy-seeking → compression-seeking) at the weight level?
3. **Training stages**: How does effective rank change across pretraining stages (initial pretraining → mid-training → long context extension)?
4. **Post-training methods**: How do different post-training methods (SFT, DPO, RLVR, RL-Zero) affect the effective rank of weight matrices?
5. **Fixed ratio hypothesis**: Is there a theoretical "ratio space" that effective rank consistently occupies?

## Architecture

```
RankAnalysis/
├── main.py                      # CLI entry point
├── src/
│   ├── effective_rank.py        # Core SVD entropy computation (weight-level)
│   ├── config.py                # Model configs, checkpoints, patterns
│   ├── model_loader.py          # HuggingFace model loading + weight extraction
│   ├── analysis.py              # Five weight-level analysis pipelines
│   ├── activation_analysis.py   # Activation-level RankMe analysis
│   └── visualization.py         # Matplotlib/Seaborn plots
├── tests/
│   ├── test_effective_rank.py       # Unit tests for weight-level computation
│   └── test_activation_analysis.py  # Unit tests for activation RankMe
├── results/                     # JSON results + figures
└── docs/                        # Design + methodology documentation
```

## Data Flow

```
HuggingFace Hub
    ↓ (download model)
model_loader.py → state_dict → extract_linear_weights()
    ↓ (2D weight tensors)
effective_rank.py → SVD → singular values → entropy → erank/ratio
    ↓ (RankMetrics per layer)
analysis.py → aggregate by group/layer/model → JSON results
    ↓ (structured data)
visualization.py → matplotlib/seaborn → PNG/PDF figures
```

## Models Analyzed

### Pythia Suite (EleutherAI)
- 8 model sizes: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B
- 154 checkpoints per model (step0 to step143000)
- GPT-NeoX architecture with fused QKV attention

### OLMo-3 (Allen AI)
- 7B base model with 3 pretraining stages
- Post-training pathways: Think, Instruct, RL-Zero
- Each pathway has SFT → DPO → RLVR stages

## Key Metrics

| Metric | Formula | Range | Purpose |
|--------|---------|-------|---------|
| Effective rank | `exp(-Σ p_i ln p_i)` where `p_i = σ_i/Σσ_j` | [1, min(m,n)] | Main metric |
| Ratio | `erank / min(m,n)` | (0, 1] | Cross-model comparison |
| SVD entropy | `-(1/log n) Σ q_i log q_i` where `q_i = σ_i²/Σσ_j²` | [0, 1] | Moonlight paper metric |
| Stable rank | `‖W‖²_F / ‖W‖²_2` | [1, min(m,n)] | Simpler alternative |
| α-ReQ | Power-law decay rate | [0, ∞) | Spectrum shape |

## Output Format

All results are saved as JSON files in `results/`:

**Weight-level analysis:**
- `cross_model_size.json` - Per-model ratios and group statistics
- `training_dynamics_{model}.json` - Per-checkpoint metrics
- `training_stages.json` - Per-stage OLMo-3 metrics
- `post_training_methods.json` - Per-variant OLMo-3 metrics
- `fixed_ratio_hypothesis.json` - Aggregated hypothesis test

**Activation-level analysis:**
- `activation_cross_model.json` - Per-model RankMe across layers
- `activation_training_dynamics_{model}.json` - Per-checkpoint activation RankMe
- `activation_post_training.json` - Per-variant OLMo-3 activation RankMe
- `activation_fixed_ratio_hypothesis.json` - Aggregated activation hypothesis test

Plots saved as PNG + PDF in `results/figures/`.

## Running

```bash
# Install dependencies
uv sync

# Run all analyses (quick mode - fewer checkpoints)
uv run python main.py --analysis all --quick

# Run specific analysis
uv run python main.py --analysis cross-model-size
uv run python main.py --analysis training-dynamics --model pythia-70m

# Validate configs without downloading
uv run python main.py --dry-run

# Regenerate plots from saved results
uv run python main.py --plot-only

# Run tests
uv run pytest
```
