# PostDyn

**Post-training dynamics** analysis for open-source LLMs.

Tools and experiments for studying how models evolve under post-training (SFT,
RL-Zero, DPO, etc.) — including effective-rank structure and concept-direction
trajectories along training checkpoints.

Python package name: `postdyn` (see `pyproject.toml`). Requires Python ≥ 3.13.

## Setup

```bash
uv sync --group dev
```

## Concept dynamics (Olmo-3-7B)

Trace DiM concept directions across **58 checkpoints × 10 layers × 6 trajectories × 46 paired concepts**.

The 6 default trajectories (Think-SFT + the five RL-Zero variants) cover every
post-training branch of Olmo-3-7B that ships a real checkpoint series. The
46-concept catalogue (`src/contrastive_datasets.all_concept_keys()`) spans the
four PaCE domains: code, math, instruction-following, and social/gender, plus
sentiment and refusal add-ons.

Arrow polarity is always A→B with +B − A. Representative keys:

| Concept | Domain | Direction |
|---------|--------|-----------|
| `code_python_vs_cpp` | HumanEval-X (20 directed pairs) | Python → C++ |
| `math_cot_vs_direct` | MATH-500 | CoT → direct answer |
| `math_informal_vs_formal` | MiniF2F | informal → Lean |
| `math_nl_vs_equations` | BeyondX | NL → equations |
| `if_eng_vs_fra` | Belebele (20 directed pairs) | Eng → Fr |
| `gender_she_vs_he` | WinoGender | she → he |
| `sentiment_label0_vs_label1` | SST-2 | label0 → label1 |
| `refusal_harmful_vs_benign` | LLM-LAT | harmful → benign |

Details: [`docs/concept_dynamics_experiment.md`](docs/concept_dynamics_experiment.md).

### Preflight / data prep

```bash
# Stream-download all concept sources into datasets/*.json
uv run python experiments/download_datasets.py

# Optional: sandbox-validate HumanEval-X python/cpp pairs
uv run python experiments/validate_humaneval_x.py
```

Some HF datasets may require `HF_TOKEN` (`uv run hf auth login` or export it).

### Run extraction + dynamics

```bash
# Full run: 6 trajectories × 46 concepts × 10 layers × 50 samples
# Output: results/concept_dynamics_multi
experiments/run_concept_dynamics.sh full

# Quick smoke test → results/concept_dynamics_multi_quick
experiments/run_concept_dynamics.sh quick

# Subset
uv run python experiments/run_concept_dynamics.py \
  --concepts code_python_vs_cpp,math_cot_vs_direct,if_eng_vs_fra,gender_she_vs_he

# Gram + stability heatmaps
uv run python experiments/plot_concept_dynamics.py \
  --input results/concept_dynamics_multi
```

Optional controls / pipelines:

```bash
# Gender surface-pronoun control vs full WinoGender direction
uv run python experiments/analyze_gender_surface_control.py \
  --model olmo3-rl-zero-math --checkpoint step_1900

# Prefetch-overlapped FLORES+ extraction
uv run python experiments/run_flores_pipeline.py
```

## Effective-rank pipelines

```bash
# Validate configs (no downloads)
uv run python main.py --dry-run

# Weight / activation rank analyses (see --analysis choices)
uv run python main.py --analysis all
```

## Layout

```
PostDyn/
├── main.py                 # effective-rank CLI
├── src/
│   ├── concept_dynamics.py # DiM extraction + stability / Gram analysis
│   ├── contrastive_datasets.py  # 46-concept loaders (local datasets/)
│   ├── dataset_store.py
│   └── ...
├── experiments/
│   ├── download_datasets.py
│   ├── run_concept_dynamics.{py,sh}
│   ├── plot_concept_dynamics.py
│   └── ...
├── datasets/               # materialized JSONs (gitignored; download_datasets.py)
├── docs/
├── tests/
└── results/                # generated outputs (gitignored)
```

## Tests

```bash
uv run pytest
```

## Docs

| Doc | Topic |
|-----|--------|
| [`docs/concept_dynamics_experiment.md`](docs/concept_dynamics_experiment.md) | Paired-concept trajectory experiment |
| [`docs/humaneval_x_validation.md`](docs/humaneval_x_validation.md) | HumanEval-X sandbox preflight |
| [`docs/downstream_cache_integrity.md`](docs/downstream_cache_integrity.md) | Downstream cache integrity & threat model |
| [`docs/design.md`](docs/design.md) | Project design |
| [`docs/methodology.md`](docs/methodology.md) | Effective-rank methodology |
