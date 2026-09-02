# Dolci RL-Zero Code concept pairs

`pairs.json` contains a small, auditable contrast set derived from the
`train` split of [`allenai/Dolci-RL-Zero-Code-7B`](https://huggingface.co/datasets/allenai/Dolci-RL-Zero-Code-7B).
It is not a mirror of the full upstream dataset.

## Concept

The concept is `list_comprehension_return`, directed from an explicit
`for`/`append` implementation to a direct list-comprehension return:

```text
explicit loop + append -> return [expression for item in iterable]
```

The positive side is detected syntactically: a parsed Python solution must
contain an `ast.Return` node whose value is an `ast.ListComp`.

## Provenance

For every item:

- `id`, `source_row`, `prompt`, and `positive` come from the upstream dataset.
- `ground_truth` contains the upstream assertions decoded from its serialized
  JSON string into a directly usable array.
- `negative` was constructed locally by rewriting the list comprehension as
  an explicit loop while preserving the function signature and behavior.
- Both implementations were executed against all 20 assertions per item and
  passed. The result is recorded in each item's `validation` object.

The constructed negatives are not original AllenAI training records and must
not be described as such.

## Intended use

For paired Difference-in-Means extraction, concatenate the same `prompt` with
the positive and negative completions, extract activations at the same model
layer and token-selection rule, and average the within-item differences:

```text
direction = mean_i(activation(positive_i) - activation(negative_i))
```

This is an implementation-style concept. Its occurrence in an upstream
reference solution demonstrates data presence, not that RLVR explicitly
rewarded list comprehensions; the verifier can also reward an equivalent loop.

---

# Python syntax-validity concept (data-only phase)

In addition to `pairs.json`, this directory holds a second, independent
diagnostic concept set built for the RL-Zero-Code experiment plan in
[`docs/rl_zero_code_concept_experiment_plan.md`](../../../docs/rl_zero_code_concept_experiment_plan.md):

- `python_syntax_pairs.json` — 50 paired records for the concept
  `python_valid_vs_syntax_error`.
- `downstream.json` — downstream evaluation items (HumanEval-X python/cpp and
  MMLU) aligned to the same experiment.

These files are **not** derived from the upstream Dolci training rows and are
**not** a mirror of any AllenAI dataset. They are constructed from public
benchmarks (HumanEval-X, MMLU) as a diagnostic contrast set.

## Concept: `python_valid_vs_syntax_error`

The target concept is **Python syntactic validity**:

```text
positive = a full, syntactically valid Python program (compiles)
negative = the same program with one deterministic syntax mutation
           that raises SyntaxError or IndentationError
```

Polarity is explicit: the positive class is `syntax_valid` and the negative
class is `syntax_error`. Paired Difference-in-Means extraction uses

```text
direction = mean_i(activation(positive_i) - activation(negative_i))
```

so the direction points toward "syntactically valid Python". The negatives are
**purely syntactic** mutations — a missing colon, a dropped parenthesis, a
dedented block, etc. No semantic or runtime bug is ever introduced or claimed.

### Mutation kinds

Six deterministic mutation kinds are used and balanced across the 50 records
(~8-9 each) so the set is not 50 identical deletions. Each emitted negative is
the **entire** positive program with exactly one localized edit (the rest of
the program is preserved verbatim), and is verified at build time to raise
`SyntaxError` or `IndentationError` under `compile(..., mode='exec')`:

| Kind | Operation | Typical error |
|------|-----------|---------------|
| `drop_def_colon` | remove trailing `:` of first `def` header | SyntaxError |
| `drop_open_paren_def` | remove `(` in first `def` header | SyntaxError |
| `drop_close_paren_def` | remove `)` in first `def` header | SyntaxError |
| `drop_def_keyword` | strip leading `def ` from first header | SyntaxError |
| `unindent_body` | dedent the first function-body line to column 0 | IndentationError |
| `indent_def_header` | indent a top-level `def` header (unexpected indent) | IndentationError |

## Provenance (syntax pairs)

- `positive` is the official HumanEval-X Python `prompt + canonical_solution`
  for a task id, copied verbatim from the pinned source revision
  `zai-org/humaneval-x@62c78627f3072a1454fa0cb0184737cafe5e4198`.
- `negative` is constructed in this project by applying exactly one
  deterministic syntax mutation to the positive. It is **not** an upstream
  HumanEval-X record.
- The 50 target task ids are sampled deterministically (seed 42) from the
  HumanEval-X python ids that are **disjoint** from both the existing pinned
  downstream `humaneval_x_task_ids` (in `datasets/shared_item_ids.json`) and
  the legacy `0..49` validator report
  (`experiments/artifacts/humaneval-x-validation.jsonl`).
- Validity is checked with `compile(..., mode='exec')` on the host only —
  **parse/bytecode compilation, no execution**. Official HumanEval-X tests are
  **not** run in this data-only phase; this is recorded honestly in each
  record's `validation` object.

> The constructed syntax-error negatives do **not** imply that syntax errors
> appeared in the Dolci RL training data or that RLVR rewarded broken code.
> This is a diagnostic concept set sourced from HumanEval-X for probing whether
> a syntactic-validity direction is learnable and stable across checkpoints.

## `downstream.json`

Two aligned downstream benchmarks for correlating representation/readout
metrics with capability across checkpoints:

- **HumanEval-X** — the existing 50 pinned `humaneval_x_task_ids`, each with
  both the Python and C++ entries (`prompt`, `canonical_solution`, full `code`,
  and the official `test` field) copied verbatim from the pinned revision. A
  later pass@1 scores the model's **own** completion against the official
  `test`; `canonical_solution`/`code` are retained only as provenance and
  preflight reference (to confirm a task is solvable / assemble the official
  harness), not as the model answer. The target and downstream id sets are
  disjoint (verified).
- **MMLU** — 50 questions from `cais/mmlu`, config `all`, split `test`, pinned
  revision `c30699e8356da336a370243923dbaf21066bb9fe`, sampled deterministically
  (seed 42) with one question per distinct subject across 50 of the 57 subjects.
  Each item preserves `question`, `choices` in original order, integer `answer`,
  `answer_letter`, `subject`, a stable `question_sha256`, and the source
  revision.

## Reproduction

Both artifacts are rebuilt by a single reproducible script (network access to
`huggingface.co` is required; no model is loaded):

```bash
uv run python experiments/download_datasets.py --only humaneval_x
uv run python experiments/build_rl_zero_syntax_concept.py --force
```

The first command materializes the pinned HumanEval-X source and shared task
IDs required by the builder.

This is a **data-only** phase: it does not register the concept in
`src/contrastive_datasets.py`, does not change checkpoint config, and runs no
extraction. See the experiment plan doc for the full checkpoint/metric design.
