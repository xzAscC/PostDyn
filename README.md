# PostDyn

**Post-training dynamics** analysis for open-source LLMs.

Tools and experiments for studying how models evolve under post-training (SFT,
RL-Zero, DPO, etc.) — including effective-rank structure and concept-direction
trajectories along training checkpoints.

Requires Python 3.13+. Use [uv](https://docs.astral.sh/uv/) for the virtual
environment.

```bash
uv sync --group dev
```

## Files Architecture

```text
src/: source code (postdyn package)
tests/: test code
notebooks/: jupyter notebooks
scripts/: py/sh files to run the code
logs/: log and JSON results for experiments
figs/: figs for experiments (PDF)
data/: store different datasets and other data
configs/: yaml files for default configs
checkpoints/: store different checkpoints if has
README.md: this file
AGENTS.md: agent rules
pyproject.toml: project metadata and build config
.python-version: Python version pin
.gitignore: git ignore rules
.ignore: ! un-ignore gitignored paths so agents can find them
LICENSE: MIT license text
```

## Concept dynamics (Olmo-3-7B)

Trace DiM concept directions across **58 checkpoints × 10 layers × 6 trajectories × 46 paired concepts**.

The 6 default trajectories (Think-SFT + the five RL-Zero variants) cover every
post-training branch of Olmo-3-7B that ships a real checkpoint series. The
46-concept catalogue (`postdyn.contrastive_datasets.all_concept_keys()`) spans
the four PaCE domains: code, math, instruction-following, and social/gender,
plus sentiment and refusal add-ons.

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

### Preflight / data prep

```bash
# Stream-download all concept sources into data/*.json
uv run python scripts/download_datasets.py

# Optional: sandbox-validate HumanEval-X python/cpp pairs
uv run python scripts/validate_humaneval_x.py
```

Some HF datasets may require `HF_TOKEN` (`uv run hf auth login` or export it).

### Run extraction + dynamics

```bash
# Full run: 6 trajectories × 46 concepts × 10 layers × 50 samples
# Output: logs/concept_dynamics_multi
bash scripts/run_concept_dynamics.sh full

# Quick smoke test → logs/concept_dynamics_multi_quick
bash scripts/run_concept_dynamics.sh quick

# Subset
uv run python scripts/run_concept_dynamics.py \
  --concepts code_python_vs_cpp,math_cot_vs_direct,if_eng_vs_fra,gender_she_vs_he

# Gram + stability heatmaps (PDF → figs/)
uv run python scripts/plot_concept_dynamics.py \
  --input logs/concept_dynamics_multi
```

Optional controls / pipelines:

```bash
# Gender surface-pronoun control vs full WinoGender direction
uv run python scripts/analyze_gender_surface_control.py \
  --model olmo3-rl-zero-math --checkpoint step_1900

# Prefetch-overlapped FLORES+ extraction
uv run python scripts/run_flores_pipeline.py
```

## Effective-rank pipelines

```bash
# Validate configs (no downloads)
uv run python -m postdyn.cli --dry-run

# Weight / activation rank analyses (see --analysis choices)
uv run python -m postdyn.cli --analysis all
```

Methodology (absorbed from the former `docs/`):

- **Weight level** — SVD entropy of weight matrices following the Moonlight
  approach: normalize singular values to a probability distribution, compute
  Shannon entropy. Background: 苏剑林 (2026) "矩阵参数的奇异值熵越高越好吗？"
  (kexue.fm/archives/11767); Li et al. (2025) "Tracing the Representation
  Geometry of Language Models from Pretraining to Post-training"
  (arXiv:2509.23024).
- **Activation level** — RankMe effective rank of hidden-state matrices on
  MMLU Pro text (diverse, high-quality probing of representation geometry),
  with α-ReQ power-law fits over a configurable singular-value range.
- **Findings context** — SFT/DPO behave entropy-seeking, RLVR
  compression-seeking (Li et al.); weight-level entropy distinguishes
  post-training methods analogously to the Muon/Moonlight report.

## RL-Zero-Code syntax-validity concept

Diagnostic concept for the Olmo-3-7B RL-Zero-Code trajectory: does a Python
syntactic-validity direction (`python_valid_vs_syntax_error`) emerge, stay
stable / separable across checkpoints, and correlate with downstream
capability?

- **Data**: `data/allenai/Dolci-RL-Zero-Code-7B/` — 50 paired HumanEval-X
  Python programs (valid vs one deterministic syntax mutation; compile-only
  validation, benchmark code never executed on host) + 50 pinned downstream
  HumanEval-X pairs and 50 MMLU questions. Built by
  `scripts/build_rl_zero_syntax_concept.py` (deterministic, seed 42).
- **Four readout metrics**: checkpoint cosine (directional stability), Δcosine
  against the four language-identity concepts, raw-direction norm change, and
  leakage-safe one-vs-rest linear probes (balanced accuracy / AUROC).
- **Downstream correlates**: Python pass@1, C++ pass@1, MMLU accuracy.
- **Run**: `scripts/run_rl_zero_syntax_extraction.py` →
  `scripts/run_rl_zero_syntax_metrics.py`; durable run notes in
  `logs/rl_zero_code_syntax/run_notes.md`.

## Other pipelines

| Pipeline | Entry | Output |
|----------|-------|--------|
| Think-SFT differential subspace | `scripts/run_think_sft_differential_subspace.py` | `logs/think_sft_differential_subspace` |
| MATH differential subspace | `scripts/run_math_differential_subspace.py` | `logs/math_differential_subspace` |
| MATH-500 ablation (7B/32B) | `scripts/run_math500_ablation.py` | `logs/math500_ablation_first50*` |
| RL-Zero downstream eval | `scripts/run_rl_zero_downstream.py` | `logs/rl_zero_code_syntax/downstream` |
| Trajectory sensitivity | `scripts/run_sensitivity_analysis.py` | `logs/sensitivity` |
| Analysis summaries | `scripts/build_analysis_summary.py` | `logs/rl_zero_code_syntax/analysis_summary.json` |

Downstream caches are integrity-checked (SHA-256 binding, corruption/staleness
detection) via `postdyn.cross_pipeline_integrity` and the validators under
`scripts/validate_*.py`.

## Tests

```bash
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
