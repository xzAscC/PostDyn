# PostDyn

PostDyn studies post-training dynamics specified by the project slides:

- **Q1:** domain-covariance spectra across OLMo-3 7B and 32B Think post-training
  trajectories.
- **Q2:** variance-direction ablations and their downstream effects.

## Setup

```bash
uv sync --group dev
uv run pytest
```

## Layout

```text
PostDyn/
├── src/postdyn/
│   ├── config.py          # experiment and schedule contracts
│   ├── data.py            # domain prompt loading and materialization
│   ├── extract.py         # hidden-state extraction
│   ├── spectra.py         # covariance and eigenspectrum analysis
│   ├── models.py          # model and checkpoint access
│   ├── persistence.py     # checkpointed result storage
│   ├── intervention.py    # variance-direction ablations
│   ├── bench.py           # downstream benchmark runners
│   └── verifiers.py       # integrity and output checks
├── scripts/
│   ├── enumerate_domain_sources.py # enumerate source datasets
│   ├── run_q1.py                  # Q1 spectrum pipeline
│   ├── run_q1_robustness.py       # Q1 robustness checks
│   ├── run_q2_exp1.py             # Q2 experiment 1
│   ├── run_q2_exp2.py             # Q2 experiment 2
│   └── run_q2_exp3.py             # Q2 experiment 3
├── configs/domain_sources.json
├── tests/
└── data/domain_prompts/           # materialized local prompt pools
```

## Domain mapping

| Domain | Source role |
|---|---|
| Math | filtered mathematical reasoning post-training sources |
| Code | Python and tool-free coding post-training sources |
| Instruction following | verified instruction-following post-training sources |
| General reasoning | science and general reasoning post-training sources |

The source mapping is maintained in `configs/domain_sources.json`.

## Checkpoint schedule

All Q1/Q2 runs use the same 22-checkpoint post-training schedule:

| Stage | Checkpoints | Count |
|---|---|---:|
| Base | `main` | 1 |
| SFT | 9 uniformly selected revisions + `main` | 10 |
| DPO | `main` | 1 |
| RLVR | 9 uniformly selected revisions + `main` | 10 |
| **Total** | | **22** |

## Quickstart

First enumerate and materialize the prompt pools:

```bash
uv run python scripts/enumerate_domain_sources.py
uv run python - <<'PY'
from postdyn.data import materialize_pools

materialize_pools("configs/domain_sources.json", "data/domain_prompts")
PY
```

Run Q1 on the 7B GPU queue or the 32B H100 in bf16:

```bash
uv run python scripts/run_q1.py --model olmo3-7b-think --device cuda
uv run python scripts/run_q1.py --model olmo3-32b-think --device cuda --dtype bfloat16
```

Run the Q2 ablations:

```bash
uv run python scripts/run_q2_exp1.py --model olmo3-7b-think --device cuda
uv run python scripts/run_q2_exp2.py --model olmo3-32b-think --device cuda --dtype bfloat16
uv run python scripts/run_q2_exp3.py --model olmo3-32b-think --device cuda --dtype bfloat16
```

## Benchmarks

| Benchmark | Purpose |
|---|---|
| MATH-500 | mathematical reasoning |
| LiveCodeBench | coding ability |
| IFEval | instruction following |
| MMLU-Pro | broad academic reasoning |

Benchmark outputs are validated before being used in Q1/Q2 comparisons.
