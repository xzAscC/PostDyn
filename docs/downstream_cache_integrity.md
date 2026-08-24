# Downstream cache integrity & threat model

This document describes the integrity controls on the RL-Zero-Code downstream
evaluation cache (`results/rl_zero_code_syntax/downstream/`) and, importantly,
what those controls do **not** claim to provide.

## Scope

The cache stores three kinds of files per checkpoint directory:

- **Per-item bodies** — `humaneval_x_<lang>_<id>.json`, `mmlu_<index>.json`: the
  raw model completion plus derived fields (hashes, parsed letters, outcomes).
- **Per-checkpoint summaries** — `summary.json`: recounted pass@1 / accuracy +
  outcome histograms for one checkpoint.
- **Aggregate** — `aggregate_summary.json`: coverage + the per-checkpoint
  summaries inlined, across the canonical 11 checkpoints.

The integrity layer sits between these files and any consumer that reuses
them (resume, summary rebuild, aggregate coverage).

## What is detected (corruption / staleness)

1. **Identity drift (staleness).** Every per-item file carries an `identity`
   sub-record binding it to `(model, revision, task, language, prompt_sha256,
   max_new_tokens, timeout, generation_contract_version)`. A file whose
   identity does not equal the expected one for the current run is treated as
   absent and regenerated (`identity_matches`,
   `src/downstream_eval.py`). The narrow `eos-truncate-1` legacy adoption
   reuses EOS-corrected files only at the default timeout and a matching body
   token budget.

2. **Body field corruption (recomputed hashes / parsing).** The cache-body
   validators (`validate_humaneval_cached_body`, `validate_mmlu_cached_body`)
   recompute every body-derived field and reject drift:
   - HumanEval: `completion_sha256` is recomputed from the stored completion;
     the program is reassembled with the official CodeGeeX helpers from
     `prompt + completion + <manifest test>` and its SHA-256 must equal the
     stored `assembled_sha256`; `task_id`/`language`/`prompt` must equal the
     pinned manifest inputs; `outcome` must be a recognized outcome constant;
     `exit_code`/`diagnostics`/`max_new_tokens` must have valid types.
   - MMLU: `predicted_letter` is **re-parsed** from the stored completion and
     `is_correct` is **recomputed** against the pinned correct letter; the
     stored editable `predicted_letter`/`is_correct` fields are never trusted.
     `index`/`subject`/`question_sha256`/`correct_letter`/`prompt` must match
     the pinned manifest item.

3. **Summary misplacement (copied / wrong-checkpoint summaries).**
   `build_aggregate` cross-checks each per-checkpoint summary's `model` and
   `revision` against the canonical checkpoint mapping
   (`checkpoint_identity`: revision always equals the checkpoint name; model
   is the base repo for `main`, the RL-Zero-Code repo for every step). A
   summary copied into the wrong directory — or one whose revision disagrees
   with its directory name — is excluded from coverage rather than counted.

4. **Scoring-config drift.** A summary whose `scoring_config` (token budgets,
   timeout, generation contract) differs from the active run is excluded from
   aggregate coverage, so a summary written under an old contract cannot
   inflate coverage.

5. **Outcome-label drift (opt-in).** `--rescore-cached` re-executes every
   cached Python/C++ completion in the bubblewrap sandbox with the current
   timeout and compares the fresh outcome to the stored label; any mismatch
   raises. This is the only path that re-derives the HumanEval outcome from
   actual program behavior.

## What is NOT claimed (non-goals)

This is **corruption and staleness detection, not authenticated tamper
resistance.** In particular:

- **No cryptographic / same-user tamper proofing.** The cache files live on
  the same filesystem the user owns. A user who can rewrite the cache can
  edit any stored field. The controls above detect *inconsistent* edits
  (a completion change without updating its hashes; a copied summary; a
  stale identity) but cannot prevent a determined same-user attacker from
  rewriting a body and all its derived fields consistently.

- **The HumanEval outcome label is hash-bound, not behavior-verified, in the
  plain rebuild.** `rebuild_checkpoint_summary` (without `--rescore-cached`)
  accepts the stored `outcome` label behind the recomputed completion and
  assembled-program hashes: you cannot change *what ran* (the completion)
  without breaking those hashes, but flipping the outcome *label* alone
  between valid enum values is not detected without re-execution. Use
  `--rescore-cached` to re-derive the outcome from actual program behavior.

- **No attestation of the model that produced the completion.** The identity
  records which model/revision a file claims to be from; it does not prove
  the claim. Verification that a completion truly came from a given checkpoint
  requires regenerating it from the model (the `--force` path), not cache
  inspection.

- **MMLU correctness is fully recomputed** (it is a deterministic letter
  parse, no sandbox needed), so the MMLU `is_correct` label is never trusted
  from the cache.

## Modes

| Mode | Loads model? | Runs sandbox? | Re-derives outcome? |
|------|--------------|---------------|---------------------|
| Normal run | yes | yes (first generation) | yes (on generation) |
| `--force` | yes | yes (every item) | yes |
| `--rebuild-summaries-only` | no | no | HumanEval: hash-bound label; MMLU: recomputed |
| `--rescore-cached` | no | yes (re-execute cached) | yes — drift raises |

`--rescore-cached` requires the bubblewrap/g++ tool check and sandbox smoke
(it is not skippable via `--skip-tool-check`), because it re-executes cached
code. It is compatible with `--rebuild-summaries-only` and implies the
no-model rebuild path.

## Preflight gate (orthogonal)

The hard preflight gate (`experiments/validate_rl_zero_downstream.py`,
`report_matches_ids`) validates the **official canonical solutions**, not the
model completions: it re-derives the SHA-256 of every canonical program from
the current dataset rows and requires every python/cpp canonical outcome to
be `pass`. It runs before any model or rebuild work, including
`--rescore-cached`. See [`humaneval_x_validation.md`](humaneval_x_validation.md).

## Related integrity controls (same experiment)

- **Normal resume validates cache bodies.** Every HumanEval/MMLU cache hit runs
  `validate_humaneval_cached_body` / `validate_mmlu_cached_body` before reuse.
  Identity includes `task_id` and `test_sha256` under the current generation
  contract, with a narrow legacy path only when full body validation succeeds.
- **Aggregate completeness.** A checkpoint counts as present only when
  `errors == 0` and the item counts are exactly 50 Python / 50 C++ / 50 MMLU.
  Incomplete checkpoints are listed under `incomplete` rather than counted in
  `n_present`.
- **Concept-vector sidecars (v1).** Concept vectors bind checkpoint, revision,
  raw protocol, `max_seq_len`, `d_model`, and ordered per-concept source-text
  fingerprints. Legacy v0 sidecars are rejected; migrate with
  `experiments/migrate_concept_sidecars.py` (JSON only; tensors unchanged) or
  re-extract.
- **Analysis summary.** `experiments/build_analysis_summary.py` regenerates
  `analysis_summary.json` deterministically from `metrics.json` +
  `aggregate_summary.json`, binds source SHA-256 values, and validates a build
  fingerprint.
