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

Trace DiM concept directions across **59 checkpoints × 10 layers × 7 models**,
using four **aligned paired** steering concepts:

| Concept key | Source | Direction |
|-------------|--------|-----------|
| `python_vs_cpp` | HumanEval-X | Python − C++ |
| `concise_math_reasoning_vs_verbose_math_reasoning` | MATH-500 | Concise − Verbose |
| `french_vs_english_language` | FLORES+ | French − English |
| `female_vs_male_gender` | WinoGender | Female − Male |

Details: [`docs/concept_dynamics_experiment.md`](docs/concept_dynamics_experiment.md).

### Preflight / data prep

```bash
# Validate 50 aligned HumanEval-X canonical pairs (sandbox + JSONL report)
uv run python experiments/validate_humaneval_x.py

# Prepare 50 verified MATH-500 concise/verbose pairs (math-verify gate)
uv run python experiments/prepare_math_pairs.py
```

FLORES+ is gated on Hugging Face: accept terms for
`openlanguagedata/flores_plus`, then `uv run hf auth login` or set `HF_TOKEN`.

### Run extraction + dynamics

```bash
# Full run (default output: results/concept_dynamics_paired)
experiments/run_concept_dynamics.sh full

# Quick smoke test → results/concept_dynamics_paired_quick
experiments/run_concept_dynamics.sh quick

# Skip gated FLORES+ while access is pending
uv run python experiments/run_concept_dynamics.py \
  --concepts python_vs_cpp,concise_math_reasoning_vs_verbose_math_reasoning,female_vs_male_gender
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
│   ├── contrastive_datasets.py
│   ├── humaneval_x_validator.py
│   ├── math_pairs.py
│   ├── gender_surface_analysis.py
│   └── ...                 # rank / activation / steering modules
├── experiments/
│   ├── run_concept_dynamics.{py,sh}
│   ├── validate_humaneval_x.py
│   ├── prepare_math_pairs.py
│   ├── analyze_gender_surface_control.py
│   └── run_flores_pipeline.py
├── docs/                   # design, methodology, experiment notes
├── tests/
├── notebook/
├── data/                   # local pair artifacts (e.g. MATH-500 JSONL)
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
| [`docs/design.md`](docs/design.md) | Project design |
| [`docs/methodology.md`](docs/methodology.md) | Effective-rank methodology |
