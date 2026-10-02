# Runbook

All commands run from the repository root with `uv`. GPU jobs on the local
RTX 4090 must go through `gpu-queue add <name> <cmd...>` (FIFO, gated on free
GPU memory). The 32B family runs on an H100 with bf16 weights.

## One-time data preparation

```bash
# Enumerate distinct Dolci dataset_source values (sanity check, offline)
uv run python scripts/enumerate_domain_sources.py

# Materialize the four fixed domain pools (data-collection step; default
# n = 2 x max(3d) = 30,720; 7B consumes the deterministic 12,288 prefix. The
# extra headroom enables genuine independent robustness resampling. Requires
# the cached allenai/Dolci-Think-SFT-7B snapshot (~34 GB).)
uv run python scripts/materialize_pools.py
```

Benchmark caches (MATH-500, LiveCodeBench `release_v6`, IFEval, MMLU-Pro)
are pulled automatically from the Hugging Face cache on first use.

## Sandbox scope (LiveCodeBench execution)

Generated solutions run via `python -I` in a temp directory with a scrubbed
environment (PATH only), CPU/address/process/file-size rlimits, a fresh
process group (killed on timeout), and capped output capture. This is a
research-box mitigation, NOT full isolation: generated code still shares the
host filesystem and network. Do not run untrusted-scale generation on a
machine with secrets or multi-tenant exposure; use container/VM isolation
there.

## 7B on the local queue (RTX 4090, bf16)

```bash
# Smoke: finals only, first 2 layers, 64 prompts per domain (~minutes)
gpu-queue add postdyn-q1-7b-smoke \
    uv run python scripts/run_q1.py --family 7b --scale smoke --output logs/q1/7b-smoke

# Full overnight pipeline: Q1 (22 checkpoints x 10 layers x 4 domains).
# Intermediate checkpoints stream (download -> extract -> prune) with a
# one-deep prefetch: the next checkpoint downloads while the current one
# extracts (transient disk bound: two checkpoints; --prefetch none to
# serialize).
# -> robustness (R=5) -> Q2 per-model orchestrator (run_q2_model.py: exp1 -> exp2 -> exp3 per single model load, once for rlvr and once for sft). Resumable per unit.
REPEATS=5 gpu-queue add postdyn-7b-overnight bash scripts/run_7b_overnight.sh
```
# GR pool: 6210 unique < n=3d — explicit deviation, see PR #7

Progress: `gpu-queue list`, `tail -f logs/overnight_7b.log`,
`ls logs/q1/7b/eigensystems/*/`. Each stage writes incremental JSON(L) after
every checkpoint/layer/domain (Q1) or condition/item (Q2), so interrupted
runs resume where they stopped.

## 32B on the H100 (bf16)

```bash
uv sync --group dev

# Q1 full trajectory (22 checkpoints, d=5120, n=30,720 materialized per domain).
# The GR domain has only 6,210 unique prompts (< n=3d): Q1 fails closed unless
# you explicitly accept the reduced GR sample by appending --allow-short-pool
# (deviation recorded per-domain in manifests/metrics; see PR #7 open decision).
uv run python scripts/run_q1.py --family 32b --scale full \
    --dtype bfloat16 --device cuda --output logs/q1/32b

# Q1 robustness on the final RLVR checkpoint (Math, R=5)
uv run python scripts/run_q1_robustness.py --family 32b --repeats 5 \
    --dtype bfloat16 --device cuda --output logs/q1_robustness/32b

# Q2 experiments (require the 32B Q1 final bases above); exp3 replaces the
# SFT low-variance component with the Procrustes-aligned RLVR counterpart
# (h - U_S U_S'h + U_R R* U_S'h), with sft_only removal as control
uv run python scripts/run_q2_exp1.py --family 32b --q1-root logs/q1/32b \
    --dtype bfloat16 --output logs/q2/32b/exp1
uv run python scripts/run_q2_exp2.py --family 32b --q1-root logs/q1/32b \
    --exp1-output logs/q2/32b/exp1 --dtype bfloat16 --output logs/q2/32b/exp2
uv run python scripts/run_q2_exp3.py --family 32b --q1-root logs/q1/32b \
    --dtype bfloat16 --sft-lr 1e-4 --output logs/q2/32b/exp3
```

The 32B SFT trajectory defaults to the `1e-4-` learning-rate branch
(README-official example); pass `--sft-lr 5e-5` anywhere to switch.

Pools are materialized at `n=30,720`; the general-reasoning pool remains at
6,210 records (16x duplication in OT3-science). This is below `n=3d`, so the
explicit deviation is recorded in the manifest/metrics for each run. Other
pools should retain the 2x headroom needed by robustness resampling.

Local 32B fallback (no H100): append `--quantization nf4` to any 32B command;
the loader checks the measured free-VRAM budget before loading.

## Verification gates

```bash
uv run pytest                                   # full suite (fast, CPU)
uv run python -m compileall -q src/postdyn scripts
uv run python scripts/run_q1.py --family 7b --scale tiny --output /tmp/q1_tiny
uv run python scripts/run_q2_exp1.py --family 7b --scale tiny --q1-root /tmp/x --output /tmp/q2e1
```

## Artifact uploads (HF Hub)

Local `data/` + `logs/` trees can live in a Hub dataset repo; uploads stream
in the background and never block the runners.

```bash
# Batch upload / resume existing artifacts (state file skips completed files)
POSTDYN_UPLOAD_TO=<user>/postdyn-artifacts uv run python scripts/upload_artifacts.py

# Streaming during runs: pass --upload-to (or export POSTDYN_UPLOAD_TO) to
# run_q1 / run_q1_robustness / run_q2_exp{1,2,3}; immutable eigensystems are
# uploaded as they land, append-only files (metrics/logs) at run end.
ALLOW_SHORT_POOL=1 POSTDYN_UPLOAD_TO=<user>/postdyn-artifacts \
    gpu-queue add postdyn-7b-overnight bash scripts/run_7b_overnight.sh

# Reclaim disk after a family's analysis is complete: deletes local
# intermediate-checkpoint eigensystems confirmed uploaded (finals, manifests,
# analysis, and pools are always kept)
uv run python scripts/upload_artifacts.py --prune-uploaded
```

Uploads retry three times per file, record failures in `.upload_state.json`,
and resume where they left off. The repo is created private on first use.

## Remote H100 (self-contained)

`scripts/run_7b_h100.sh` / `scripts/run_32b_h100.sh` run a full family from scratch on a
remote H100: `uv sync`, local `HF_HOME` (`$PWD/hf_cache`, override with
`POSTDYN_HF_HOME`), explicit Dolci snapshot download + pool materialization
(skipped when the four pools exist), Q1 full -> Q1 robustness (`REPEATS`, default 5)
-> Q2 orchestrator for both models. `--allow-short-pool` defaults ON for Q1
(`ALLOW_SHORT_POOL=0` disables); `SFT_LR` (32B, default 1e-4) and
`POSTDYN_UPLOAD_TO` pass through. `DRY_RUN=1` prints the pipeline without
executing. No gpu-queue (local 4090 keeps run_7b_overnight.sh via gpu-queue).
