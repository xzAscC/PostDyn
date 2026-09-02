# RL-Zero-Code Concept Experiment — Python Syntax Validity

## Overview

This document specifies a **diagnostic concept experiment** for the Olmo-3-7B
RL-Zero-Code trajectory. The target concept is **Python syntactic validity**:
does a syntactic-validity direction emerge in the model's representations, and
is it stable / separable across post-training checkpoints, and do its dynamics
correlate with downstream code & general capability?

**Status:** the data-only artifacts were built first; the approved model phase
has since been executed under the **raw / no-chat** primary protocol. Durable
run notes live in `results/rl_zero_code_syntax/run_notes.md`. Historical
data-only builder details remain below for auditability.

## Target concept

| Field | Value |
|-------|-------|
| Concept key (proposed) | `python_valid_vs_syntax_error` *(documentation only; not registered in this phase)* |
| Target | Python syntactic validity |
| Positive class | `syntax_valid` — a full Python program that compiles under `compile(..., mode='exec')` |
| Negative class | `syntax_error` — the *same* program with exactly one deterministic, purely-syntactic mutation that raises `SyntaxError` or `IndentationError` |
| Polarity / DiM | `direction = mean_i(activation(positive_i) − activation(negative_i))`, i.e. the direction points **toward syntactically valid Python** |

The negatives are **syntactic only**. A missing colon, a dropped parenthesis,
a dedented block — never a semantic or runtime bug. Each negative is verified
at build time to raise `SyntaxError`/`IndentationError`, so the contrast is
purely about *whether the program parses*, not what it does.

### Why this concept

The existing 46-concept catalogue measures *cross-language* code directions
(`code_python_vs_cpp`, etc.) and implementation style (`list_comprehension_return`).
None of them isolate **syntactic well-formedness** as a within-language,
same-program contrast. A syntax-validity direction is a clean probe for
whether RL-Zero-Code sharpens the model's internal notion of "this is valid
Python," which is plausibly correlated with — but distinct from — pass@1.

## Related existing concepts (for cross-concept analysis)

The target is analyzed alongside the already-extracted concepts in
`src/contrastive_datasets.py`. The four directly related code concepts are:

| Concept | Direction | Relation to target |
|---------|-----------|--------------------|
| `code_python_vs_cpp` | Python → C++ | Same source (HumanEval-X), language identity |
| `code_python_vs_js` | Python → JavaScript | Same source, language identity |
| `code_python_vs_java` | Python → Java | Same source, language identity |
| `code_python_vs_go` | Python → Go | Same source, language identity |

These four are the natural neighbors for the cross-concept cosine metric below.
Their existing concept vectors (already extracted across the 6 trajectories × 10
layers) will be reused directly where the checkpoint/layer/token-selection rule
matches.

## Control concept

| Concept | Direction | Role |
|---------|-----------|------|
| `gender_she_vs_he` | she → he (WinoGender) | Out-of-domain control: a direction the RL-Zero-Code trajectory should **not** move systematically. Drift here indicates a generic post-training effect rather than a code-specific one. |

The existing `gender_she_vs_he` vectors are reused unchanged.

## Checkpoints

The experiment uses checkpoints that **already exist** in this repo's config
(`src/config.py`). No checkpoint config is added or modified in this phase.

* **Base / pretraining anchor:** `allenai/Olmo-3-1025-7B`, revision `main`
  (`OLMO3_BASE_CONFIG`). Used as the pre-RL reference point only.
* **RL-Zero-Code trajectory:** the existing ten `olmo3-rl-zero-code`
  checkpoints, uniformly selected from `step_100 … step_2900`:

  ```
  step_100, step_400, step_700, step_1000, step_1300,
  step_1700, step_2000, step_2300, step_2600, step_2900
  ```

Layers and token selection reuse the existing concept-dynamics rule:
`EXPERIMENT_LAYERS_7B = [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]`, last-token of
the **raw (no chat template)** paired text. Chat-template vectors appear only
in the secondary sensitivity sweep where available.

## Data-only deliverables (this phase)

All artifacts live in
[`datasets/allenai/Dolci-RL-Zero-Code-7B/`](../datasets/allenai/Dolci-RL-Zero-Code-7B/)
and are built by the single reproducible script
[`experiments/build_rl_zero_syntax_concept.py`](../experiments/build_rl_zero_syntax_concept.py):

| File | Contents | Count |
|------|----------|-------|
| `python_syntax_pairs.json` | Paired (valid, syntax-broken) HumanEval-X Python programs | exactly 50 records |
| `downstream.json` | HumanEval-X python/cpp (prompt, canonical_solution, code, official test) + MMLU questions | exactly 50 HE-X pairs + 50 MMLU questions |

### Target pairs (`python_syntax_pairs.json`)

* **Positives** are the official HumanEval-X Python `prompt + canonical_solution`
  for 50 task ids, copied verbatim from the pinned source revision
  `zai-org/humaneval-x@62c78627f3072a1454fa0cb0184737cafe5e4198`. Each is
  verified to compile.
* **Negatives** are produced by one of **six deterministic, balanced syntax
  mutation kinds** (~8-9 records each): `drop_def_colon`,
  `drop_open_paren_def`, `drop_close_paren_def`, `drop_def_keyword`,
  `unindent_body`, `indent_def_header`. Each is verified to raise
  `SyntaxError`/`IndentationError`. No two of the 50 are identical deletions.
* **Target ids** are sampled deterministically (seed 42) from HumanEval-X
  python ids **disjoint from both** the pinned downstream
  `humaneval_x_task_ids` (`datasets/shared_item_ids.json`) and the legacy
  `0..49` validator report
  (`experiments/artifacts/humaneval-x-validation.jsonl`).
* **Validation is compile-only.** Official HumanEval-X tests are **not** run in
  this phase and the records say so honestly. Benchmark code is never executed
  on the host — only `compile(..., mode='exec')` (parse + bytecode, no run).

### Downstream (`downstream.json`)

* **HumanEval-X:** the existing 50 pinned `humaneval_x_task_ids`, each with the
  Python **and** C++ `prompt`, `canonical_solution`, full `code`, and the
  official `test` field — all verbatim from the pinned revision, so a later
  pass@1 can assemble and execute the official CodeGeeX programs. Downstream
  answers are always valid code; the target and downstream id sets are
  disjoint (verified).
* **MMLU:** 50 questions from `cais/mmlu`, config `all`, split `test`, pinned
  revision `c30699e8356da336a370243923dbaf21066bb9fe`, sampled deterministically
  (seed 42) with one question per distinct subject across 50 of the 57
  subjects. Each item preserves `question`, `choices` (original order), integer
  `answer`, `answer_letter`, `subject`, a stable `question_sha256`, and the
  source revision.

### Provenance, license, and caveats

* HumanEval-X is licensed under the [CodeGeeX license](https://huggingface.co/datasets/zai-org/humaneval-x)
  and is reproduced here only at the pinned revision for reproducibility.
* MMLU is released under the [MIT license](https://huggingface.co/datasets/cais/mmlu).
* The constructed syntax-error negatives are **not** part of the upstream Dolci
  RL training data. This dataset does **not** claim that syntax errors appeared
  during Dolci RL training or that RLVR rewarded broken code — it is a
  diagnostic contrast set sourced from HumanEval-X.
* Constructing this data does **not** prove the model learned the concept; that
  is precisely what the later extraction phase would test.

## Representation / readout metrics (exactly four)

The target concept direction is extracted per `(checkpoint, layer)` with the
existing DiM pipeline and evaluated with **exactly four** metrics:

1. **Checkpoint cosine similarity (directional stability).**
   `cos(r_k^t, r_k^{t'})` between the target direction at the current and a
   reference checkpoint (base `main`, or the first RL step). High intra-trajectory
   cosine means the syntactic-validity direction is stable across RL.

2. **Cross-concept cosine-similarity change.**
   `Δcos = cos(r_target^t, r_related^t) − cos(r_target^{t0}, r_related^{t0})`
   against the four related code concepts (`code_python_vs_cpp/js/java/go`).
   Tracks whether the syntax direction entangles/disentangles from language-identity
   directions during RL.

3. **Raw-direction magnitude change.**
   `‖r_k^t‖ − ‖r_k^{t0}‖` (the un-normalized DiM mean-difference norm). A growing
   norm indicates the model is separating valid from invalid Python more strongly,
   independent of orientation.

4. **One-vs-rest category separability (linear probe).**
   Define eight explicit classes — `python_valid`, `python_syntax_error`
   (the target), and the six related/control categories `cpp`, `js`, `java`,
   `go`, `she`, `he` (from the HumanEval-X code directions and the
   `gender_she_vs_he` control). Extract hidden states for all texts, then train
   **one binary linear probe (logistic regression) per class** against the union
   of all remaining classes (one-vs-rest). Splits are **task/template-grouped**:
   all contrastive texts derived from the same HumanEval-X task id (or WinoGender
   template) stay in the same fold, so the probe cannot memorize item identity.
   Report **balanced accuracy** and **AUROC** per class. The `python_valid` vs
   rest and `python_syntax_error` vs rest probes are the primary readout for
   whether the syntax-validity signal is linearly decodable; the language and
   gender probes provide the control comparison and feed the cross-category
   view.

## Downstream performance measures (correlated with the four metrics)

Three downstream quantities, measured per checkpoint, to correlate against the
four representation/readout metrics above:

| Measure | Source | Scoring input |
|---------|--------|--------------|
| **Python pass@1** | HumanEval-X Python | the model-generated completion combined with the benchmark `prompt` + official `test` |
| **C++ pass@1** | HumanEval-X C++ | the model-generated completion combined with the benchmark `prompt` + official `test` |
| **MMLU accuracy** | MMLU `all`/`test` | model prediction over `question` + `choices`, scored against `answer_letter` |

`canonical_solution` and the assembled full `code` are retained in
`downstream.json` only as **provenance and preflight reference** (e.g. to
confirm a task is solvable / assemble the official CodeGeeX harness). They are
**not** used as the model's answer during scoring — pass@1 scores the model's
own completion against the official tests.

The correlation questions: does the syntax-direction's stability (metric 1),
disentanglement from language identity (metric 2), growing norm (metric 3), or
probe separability (metric 4) track Python pass@1 / C++ pass@1 / MMLU accuracy
along the RL-Zero-Code trajectory?

All downstream code execution happens in a later, sandboxed phase — **not** in
this data-only phase. Official tests are never run on the host.

## Scope: what this phase does NOT do

Per the experiment brief, this data-only phase explicitly:

* Does **not** register `python_valid_vs_syntax_error` in
  `src/contrastive_datasets.py` or its `_LOADERS` dispatcher.
* Does **not** alter `src/config.py` checkpoint config, model lists, or layer
  selection.
* Does **not** run any model forward pass, extraction, or steering.
* Does **not** modify or delete the existing `pairs.json` (the
  `list_comprehension_return` concept) — that file is untouched.
* Does **not** alter the existing pinned HumanEval-X shared ids, existing
  concept vectors, or extraction outputs.
* Does **not** execute any downloaded benchmark code on the host.
* Uses **no** type suppressions and **no** empty/bare exception handlers.

## User-approval gate

After the artifacts in this phase are built, the parent agent inspects every
changed file and reports them. **No model experiment begins until the user
explicitly approves.** The next phase (extraction + metric computation +
downstream pass@1/MMLU evaluation) is a separate, gated work item.

## Reproduction

```bash
# Fresh clone: materialize the pinned HumanEval-X data and shared task ids first.
uv run python experiments/download_datasets.py --only humaneval_x

# Rebuild both data artifacts (network access to huggingface.co required).
uv run python experiments/build_rl_zero_syntax_concept.py --force

# Validate the artifacts + mutation logic.
uv run pytest tests/test_rl_zero_syntax_concept.py -v
```

The builder is deterministic: reruns with the same pinned revisions and seed 42
produce byte-identical JSON (modulo any upstream revision drift, which the
pinned revisions prevent). It expects `datasets/humaneval_x.json` and
`datasets/shared_item_ids.json` from the download step above.
