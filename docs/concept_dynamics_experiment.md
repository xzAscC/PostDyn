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
| Instruct-SFT | allenai/Olmo-3-7B-Instruct-SFT | 1 (main only) | — |
| RL-Zero-Math | allenai/Olmo-3-7B-RL-Zero-Math | 10 | step_100→step_1900 |
| RL-Zero-Code | allenai/Olmo-3-7B-RL-Zero-Code | 10 | step_100→step_2900 |
| RL-Zero-IF | allenai/Olmo-3-7B-RL-Zero-IF | 10 | step_100→step_1900 |
| RL-Zero-General | allenai/Olmo-3-7B-RL-Zero-General | 8 | step_100→step_800 |
| RL-Zero-Mix | allenai/Olmo-3-7B-RL-Zero-Mix | 10 | step_50→step_950 |

**Total: 59 checkpoints** uniformly selected from each model's available training steps.

## Method

Per `(model, checkpoint, layer, concept)`:
1. Load 50 aligned positive/negative text pairs from a pinned source
2. Forward pass → extract last-token hidden state
3. DiM: `r = μ+ - μ-`, normalize `r_hat = r / ||r||`
4. Save as safetensors

**Layers**: `[3, 6, 9, 12, 16, 19, 22, 25, 28, 31]` (10%, 20%, ..., 100% of 32 layers)
**Token selection**: the last token of each complete raw paired text. Code and
math texts include the problem followed by the solution, so this token lies in
the response; FLORES+ and WinoGender are sentence-level pairs, so it is the
sentence-final token. The same rule and no chat wrapper are used at every
checkpoint to avoid introducing model-specific formatting as a confound.

## Paired Concepts

| Concept | Pinned source | Positive | Negative | Direction |
|---------|---------------|----------|----------|-----------|
| `python_vs_cpp` | HumanEval-X | Python | C++ | Python − C++ |
| `concise_math_reasoning_vs_verbose_math_reasoning` | MATH-500 | Concise reasoning | Verbose reasoning | Concise − Verbose |
| `french_vs_english_language` | FLORES+ | French | English | French − English |
| `female_vs_male_gender` | WinoGender | Female pronoun | Male pronoun | Female − Male |

HumanEval-X, FLORES+, and WinoGender are loaded from immutable revisions in
`src/contrastive_datasets.py`. MATH-500 pairs are generated once with the final
RL-Zero-Math checkpoint; verbose reasoning is generated first, concise reasoning
uses that trajectory as its reference, and both final answers must pass
`math-verify` against the gold answer.

The default 50 WinoGender pairs cover all three canonical pronoun slots while
remaining balanced between `answer=0` and `answer=1` within every slot: 36
nominative `she/he`, 10 possessive-determiner `her/his`, and all 4 available
accusative `her/him` templates. Equal three-way balancing is impossible because
the pinned source contains only four accusative templates.

### Gender surface-token control

After extracting `female_vs_male_gender`, run:

```bash
uv run python experiments/analyze_gender_surface_control.py \
  --model olmo3-rl-zero-math --checkpoint step_1900
```

The control uses only the gendered pronouns, weighted exactly like the default
WinoGender set (36 `she/he`, 10 `her/his`, 4 `her/him`). It extracts their
last-token DiM direction through the same model and layers, then records cosine
alignment with the full-sentence WinoGender direction. High absolute cosine
means the measured direction is dominated by surface pronoun identity; low
absolute cosine indicates that sentence context contributes substantially.
Results are saved under `results/concept_dynamics_paired/`.

## Output

Stability is the cosine similarity of the same concept direction across
checkpoints. Gram matrices measure pairwise concept-direction entanglement at
each checkpoint and layer.

### Total Output

| File | Content |
|------|---------|
| `stability/stability.json` | Per-model, per-concept checkpoint stability matrices |
| `gram/gram.json` | 590 matrices (59 checkpoints × 10 layers; 3×3 for the public run, 4×4 after FLORES+) |
| `extraction_results.json` | Full extraction summary |
| `vectors/{model}/{checkpoint}/layer_{L}.{safetensors,json}` | 590 concept vector files |

## How to Run

FLORES+ is gated on Hugging Face. Before running extraction, request access to
`openlanguagedata/flores_plus`, accept its terms, and authenticate with either
`uv run hf auth login` or an `HF_TOKEN` environment variable that can read the
dataset.

```bash
# Validate exactly 50 aligned HumanEval-X canonical pairs
uv run python experiments/validate_humaneval_x.py

# Prepare exactly 50 verified MATH-500 pairs
uv run python experiments/prepare_math_pairs.py

# Full experiment (all models × all checkpoints)
experiments/run_concept_dynamics.sh full

# Run the three public/non-gated concepts while FLORES+ access is unavailable
uv run python experiments/run_concept_dynamics.py \
  --concepts python_vs_cpp,concise_math_reasoning_vs_verbose_math_reasoning,female_vs_male_gender

# Quick smoke test (1 model, 2 concepts, 2 layers, 5 samples)
experiments/run_concept_dynamics.sh quick
```

The pipeline automatically:
- Downloads each checkpoint via HF `revision` parameter
- Cleans HF cache between models to manage disk space
- Supports resume (skips already-processed checkpoints)
- Writes to `results/concept_dynamics_paired` by default so older generic
  concept vectors and resume state cannot collide with this experiment
- Writes quick-mode smoke results to `results/concept_dynamics_paired_quick`
  so partial quick checkpoints cannot be mistaken for completed full checkpoints
