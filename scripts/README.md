# Runbook

All commands run from the repository root with `uv`. GPU jobs on the local
RTX 4090 must go through `gpu-queue add <name> <cmd...>` (FIFO, gated on free
GPU memory). The 32B family runs on an H100 with bf16 weights.

## One-time data preparation

```bash
# Enumerate distinct Dolci dataset_source values (sanity check, offline)
uv run python scripts/enumerate_domain_sources.py

# Materialize the four fixed domain pools (n = 3 x d_model of 32B = 15,360;
# 7B consumes the deterministic 12,288 prefix). Requires the cached
# allenai/Dolci-Think-SFT-7B snapshot (~34 GB).
uv run python - <<'PY'
import sys
sys.path.insert(0, "src")
from postdyn.data import materialize_pools
materialize_pools("configs/domain_sources.json", "data/domain_prompts", n=15360)
PY
```

Benchmark caches (MATH-500, LiveCodeBench `release_v6`, IFEval, MMLU-Pro)
are pulled automatically from the Hugging Face cache on first use.

## 7B on the local queue (RTX 4090, bf16)

```bash
# Smoke: finals only, first 2 layers, 64 prompts per domain (~minutes)
gpu-queue add postdyn-q1-7b-smoke \
    uv run python scripts/run_q1.py --family 7b --scale smoke --output logs/q1/7b-smoke

# Full overnight pipeline: Q1 (22 checkpoints x 10 layers x 4 domains)
# -> robustness (R=5) -> Q2 exp1 -> exp2 -> exp3. Resumable per unit.
REPEATS=5 gpu-queue add postdyn-7b-overnight bash scripts/run_7b_overnight.sh
```

Progress: `gpu-queue list`, `tail -f logs/overnight_7b.log`,
`ls logs/q1/7b/eigensystems/*/`. Each stage writes incremental JSON(L) after
every checkpoint/layer/domain (Q1) or condition/item (Q2), so interrupted
runs resume where they stopped.

## 32B on the H100 (bf16)

```bash
uv sync --group dev

# Q1 full trajectory (22 checkpoints, d=5120, n=15,360 per domain)
uv run python scripts/run_q1.py --family 32b --scale full \
    --dtype bfloat16 --device cuda --output logs/q1/32b

# Q1 robustness on the final RLVR checkpoint (Math, R=5)
uv run python scripts/run_q1_robustness.py --family 32b --repeats 5 \
    --dtype bfloat16 --device cuda --output logs/q1_robustness/32b

# Q2 experiments (require the 32B Q1 final bases above)
uv run python scripts/run_q2_exp1.py --family 32b --q1-root logs/q1/32b \
    --dtype bfloat16 --output logs/q2/32b/exp1
uv run python scripts/run_q2_exp2.py --family 32b --q1-root logs/q1/32b \
    --exp1-output logs/q2/32b/exp1 --dtype bfloat16 --output logs/q2/32b/exp2
uv run python scripts/run_q2_exp3.py --family 32b --q1-root logs/q1/32b \
    --dtype bfloat16 --sft-lr 1e-4 --output logs/q2/32b/exp3
```

The 32B SFT trajectory defaults to the `1e-4-` learning-rate branch
(README-official example); pass `--sft-lr 5e-5` anywhere to switch.

Local 32B fallback (no H100): append `--quantization nf4` to any 32B command;
the loader checks the measured free-VRAM budget before loading.

## Verification gates

```bash
uv run pytest                                   # full suite (fast, CPU)
uv run python -m compileall -q src/postdyn scripts
uv run python scripts/run_q1.py --family 7b --scale tiny --output /tmp/q1_tiny
uv run python scripts/run_q2_exp1.py --family 7b --scale tiny --q1-root /tmp/x --output /tmp/q2e1
```
