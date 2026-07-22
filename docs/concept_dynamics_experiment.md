# Concept Dynamics Experiment — Olmo-3-7B Training Trajectory

## Overview

This experiment traces how concept representations evolve **along each model's training trajectory** (up to 10 uniformly-spaced checkpoints spanning the first and final available steps). It implements the DiM pipeline from "Tracing Concept Dynamics through Pretraining and Post-training".

For each model, we extract concept directions at each selected checkpoint (up to 10 per model) × 10 layers, then measure:
1. **Directional stability** — `cos(r_k^t, r_k^t')` across checkpoints within each model
2. **Concept Gram matrix** — `G_ij^t = cos(r_i^t, r_j^t)` entanglement at each checkpoint

## Models and Checkpoints

| Model | HF ID | Checkpoints | Range |
|-------|-------|-------------|-------|
| Think-SFT | allenai/Olmo-3-7B-Think-SFT | 10 | first→last available steps |
| RL-Zero-Math | allenai/Olmo-3-7B-RL-Zero-Math | 10 | step_100→step_1900 |
| RL-Zero-Code | allenai/Olmo-3-7B-RL-Zero-Code | 10 | step_100→step_2900 |
| RL-Zero-IF | allenai/Olmo-3-7B-RL-Zero-IF | 10 | step_100→step_1900 |
| RL-Zero-General | allenai/Olmo-3-7B-RL-Zero-General | 8 | step_100→step_800 |
| RL-Zero-Mix | allenai/Olmo-3-7B-RL-Zero-Mix | 10 | step_50→step_950 |

**Total: 58 checkpoints** uniformly selected from each model's available training steps across **6 trajectories**. (`olmo3-instruct-sft` was dropped from the default set because it only exposes a single `main` revision; the 6 trajectories above all expose real checkpoint series.)

## Method

Per `(model, checkpoint, layer, concept)`:
1. Load 50 aligned positive/negative text pairs from a pinned source
2. Optionally wrap each text with `tokenizer.apply_chat_template` (default: on)
3. Forward pass → extract last-token hidden state
4. DiM: `r = μ+ - μ-`, normalize `r_hat = r / ||r||`
5. Save as safetensors

**Layers**: `[3, 6, 9, 11, 14, 17, 20, 22, 25, 28]` via the slide formula
`ℓ_j = round[(0.1 + 0.8·j/9) · (L-1)]` for `j = 0..9` and `L = 32`. The range
slides from ~10% to ~90% of `(L-1)` so every layer has both a non-trivial
receptive field over the input and a non-degenerate downstream head.
**Token selection**: the last token of each complete paired text after chat
templating. Code and math texts include the problem followed by the solution,
so this token lies in the response; FLORES+/Belebele and WinoGender are
sentence-level pairs, so it is the sentence-final token. Chat templating is
on by default to match the post-training distribution of the variants.

## Paired Concepts (multi-domain, 46 total)

The catalogue lives in `src/contrastive_datasets.py` and is exposed via
`all_concept_keys()`. It spans the four PaCE domains:

| Domain | Concept count | Representative key | Direction |
|--------|---------------|--------------------|-----------|
| Code (HumanEval-X) | 20 | `code_python_vs_cpp` | Python → C++ |
| Math (MATH-500, miniF2F, BeyondX) | 3 | `math_cot_vs_direct` | CoT → direct |
| Instruction-following (Belebele) | 20 | `if_eng_vs_fra` | English → French |
| Gender / Social (WinoGender) | 1 | `gender_she_vs_he` | she → he |
| Sentiment (SST-2) | 1 | `sentiment_label0_vs_label1` | label0 → label1 |
| Safety / Refusal (LLM-Latent) | 1 | `refusal_harmful_vs_benign` | harmful → benign |

The 4 concept keys chosen as defaults for the 6-trajectory slide-deck
("representative per domain") are `code_python_vs_cpp`,
`math_cot_vs_direct`, `if_eng_vs_fra`, `gender_she_vs_he`. Legacy aliases
(`python_vs_cpp`, `concise_math_reasoning_vs_verbose_math_reasoning`,
`french_vs_english_language`, `female_vs_male_gender`) are still accepted
by `load_contrastive_texts` for backward compatibility.

### HumanEval-X preflight

Only the legacy `python_vs_cpp` concept has a strict sandbox preflight
(`experiments/validate_humaneval_x.py`). For all other `code_*` concepts,
the runner emits a relaxed warning instead of failing; place a snapshot at
`datasets/humaneval_x.json` to silence it once a multi-language validator
lands.

### Gender surface-token control

After extracting `gender_she_vs_he`, run:

```bash
uv run python experiments/analyze_gender_surface_control.py \
  --model olmo3-rl-zero-math --checkpoint step_1900
```

The control uses only the gendered pronouns, weighted exactly like the
default WinoGender set. It extracts their last-token DiM direction through
the same model and layers, then records cosine alignment with the
full-sentence WinoGender direction. High absolute cosine means the
measured direction is dominated by surface pronoun identity; low absolute
cosine indicates that sentence context contributes substantially.
Results are saved under `results/concept_dynamics_multi/`.

## Output

Stability is the cosine similarity of the same concept direction across
checkpoints. Gram matrices measure pairwise concept-direction entanglement
at each checkpoint and layer.

### Total Output

| File | Content |
|------|---------|
| `stability/stability.json` | Per-model, per-concept checkpoint stability matrices |
| `gram/gram.json` | One matrix per (model, checkpoint, layer) across concepts |
| `extraction_results.json` | Full extraction summary |
| `vectors/{model}/{checkpoint}/layer_{L}.{safetensors,json}` | Concept vector files |

## Visualization

`experiments/plot_concept_dynamics.py` renders Gram and stability heatmaps
from a concept-dynamics output directory:

```bash
# Default summary plot: mid-layer Gram at the last checkpoint per model
uv run python experiments/plot_concept_dynamics.py

# Single Gram heatmap
uv run python experiments/plot_concept_dynamics.py \
  --input results/concept_dynamics_multi \
  --model olmo3-think-sft --checkpoint step_500 --layer 14

# Single stability heatmap
uv run python experiments/plot_concept_dynamics.py \
  --input results/concept_dynamics_multi \
  --model olmo3-think-sft --concept code_python_vs_cpp --layer 14
```

All heatmaps use `vmin=-1, vmax=1, center=0, cmap=RdBu_r`. Matrices with
more than 20 concepts hide tick labels for readability.

## How to Run

Belebele / FLORES+ / WinoGender / HumanEval-X may require Hugging Face
authentication. Before running extraction, request access to any gated
dataset and authenticate with either `uv run hf auth login` or an
`HF_TOKEN` environment variable.

```bash
# Validate 50 aligned HumanEval-X canonical pairs (python/cpp only)
uv run python experiments/validate_humaneval_x.py

# Full experiment (6 trajectories × 46 concepts × 10 layers × 50 samples)
experiments/run_concept_dynamics.sh full

# Run the four representative concepts only
uv run python experiments/run_concept_dynamics.py \
  --concepts code_python_vs_cpp,math_cot_vs_direct,if_eng_vs_fra,gender_she_vs_he

# Quick smoke test (1 model, 2 concepts, 2 layers, 5 samples)
experiments/run_concept_dynamics.sh quick
```

The pipeline automatically:
- Downloads each checkpoint via HF `revision` parameter
- Cleans HF cache between models to manage disk space
- Supports resume (skips already-processed checkpoints)
- Writes to `results/concept_dynamics_multi` by default so older
  single-domain runs cannot collide with the 46-concept expansion
- Writes quick-mode smoke results to `results/concept_dynamics_multi_quick`
  so partial quick checkpoints cannot be mistaken for completed full
  checkpoints
