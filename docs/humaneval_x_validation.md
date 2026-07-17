# HumanEval-X Canonical-Solution Validation

## Why this exists

The `python_vs_cpp` concept in `experiments/run_concept_dynamics.py` feeds the
DiM extraction pipeline with paired Python and C++ source code from the pinned
[`zai-org/humaneval-x`](https://huggingface.co/datasets/zai-org/humaneval-x)
dataset (revision `62c78627f3072a1454fa0cb0184737cafe5e4198`). Before any model
activations are computed, we must prove each canonical solution actually
compiles, runs, and passes the upstream tests — otherwise a single broken row
would silently steer concept extraction with bogus text. The validator
implements that proof and the runner refuses to extract `python_vs_cpp` until a
matching report exists.

The assembly rules match the official CodeGeeX evaluator at
`CodeGeeX SHA 2838420b7b4492cf3d16bce5320e26e65960c9e2`.

## Files

| File | Purpose |
|------|---------|
| `src/humaneval_x_validator.py` | Assembly, bubblewrap runner, atomic report writer, preflight |
| `experiments/validate_humaneval_x.py` | CLI that validates the first N aligned pairs |
| `experiments/run_concept_dynamics.py` | Concept-dynamics runner with the preflight gate wired into `main()` |
| `tests/test_humaneval_x_validator.py` | Focused tests (no bwrap/network required) |
| `experiments/artifacts/humaneval-x-validation.jsonl` | Default report path (created on first successful validation) |

## Run the validator

The validator needs `bubblewrap` (`/usr/bin/bwrap`), `g++` (`/usr/bin/g++`),
and `python3` (`/usr/bin/python3`) on the host. On Arch/Debian/Fedora these
are already present or one `pacman -S bubblewrap gcc python` away.
The first 50 C++ tasks also require Boost headers (`boost/any.hpp`). If they
are installed outside the system include path, set
`HUMANEVAL_X_CPP_INCLUDE_DIR` to the include root containing `boost/`; the
validator mounts that directory read-only and passes it to g++ with `-I`.

```bash
uv run python experiments/validate_humaneval_x.py
```

Options:

| Flag | Default | Effect |
|------|---------|--------|
| `--n N` | `50` | Aligned pairs to validate |
| `--report-path P` | `experiments/artifacts/humaneval-x-validation.jsonl` | Output JSONL report |
| `--timeout SECS` | `10` | Per-program subprocess timeout |
| `--skip-tool-check` | off | Skip bwrap/g++ presence check (testing only) |

The script:

1. Pulls the pinned raw JSONL for Python and C++ via streaming `load_dataset`.
2. Aligns the first `--n` shared numeric task ids.
3. For each task, assembles the official Python and C++ programs, runs both
   inside a bubblewrap sandbox, and records the outcome.
4. Writes the report **atomically** (temp file + `os.replace`) only if every
   requested pair passes. Any failure exits non-zero and leaves the previous
   report (if any) untouched.

`--help` is fully offline — no network or dataset library is touched.

## How canonical code is executed

* **Assembly** is byte-exact and deterministic. Python programs begin with the
  official CodeGeeX import header (`import math`, `import re`, …,
  `from collections import *`) followed by `prompt + canonical_solution + test`.
  C++ programs prepend the system includes (`stdlib.h`, `algorithm`, `math.h`,
  `stdio.h`, `vector`, `string`, `climits`, `cstring`, `iostream`) — de-duplicating
  any include already declared in the prompt — followed by
  `prompt + canonical_solution + test`.
* **Compilation** uses `/usr/bin/g++ -std=c++11`. Task `CPP/162` additionally
  links OpenSSL (`-lcrypto -lssl`), matching the CodeGeeX harness.
* **Execution** happens only inside `bubblewrap` with `--unshare-all`
  (network, IPC, PID, mount, user namespaces), `--die-with-parent`, read-only
  `/usr` + `/etc`, a per-task writable scratch directory bind-mounted from the
  system temp dir, and a per-program `subprocess` timeout. Canonical code is
  **never** imported or executed in the host Python process.
* **Diagnostics** (stdout + stderr) are captured and truncated to 4 KiB so a
  single failing task cannot bloat the report.

## Report format

One JSON object per line. Rows are sorted by `task_id` (ascending), matching
the order returned by `load_humaneval_x_raw_pairs`. Every row binds the
outcome to the pinned dataset and the exact bytes that ran:

```json
{
  "task_id": 1,
  "revision": "62c78627f3072a1454fa0cb0184737cafe5e4198",
  "dataset": "zai-org/humaneval-x",
  "python_code_sha256": "...",
  "cpp_code_sha256": "...",
  "python_outcome": "pass",
  "cpp_outcome": "pass",
  "python_exit_code": 0,
  "cpp_exit_code": 0,
  "python_diagnostics": "",
  "cpp_diagnostics": ""
}
```

Outcomes are one of `pass`, `fail`, `timeout`, `compile_error`, `error`.

## Preflight integration

`experiments/run_concept_dynamics.py` enforces the report before extracting
the `python_vs_cpp` concept. When `python_vs_cpp` is among the selected
concepts, `main()` calls `run_humaneval_preflight(report_path, n_samples)`
which:

1. Parses the report and checks at least `n_samples` successful rows.
2. Verifies every row's `revision` and `dataset` match the pinned constants.
3. Verifies every row marks both Python and C++ as `pass`.
4. Verifies task ids are unique within the report.
5. Re-loads the first `n_samples` pinned pairs and recomputes the SHA-256 of
   each assembled program, comparing against the stored hashes. Any mismatch
   (stale report, manual edit, dataset drift) aborts the run with exit code 2
   and prints the exact failure reason.

A missing report, a partial report, any failed row, or a report for different
task IDs will block extraction. There is no bypass flag because both canonical
solutions passing their tests is part of the experiment's data contract.

## Coverage and HuggingFace checkpoint publication limits

A single HumanEval-X report gates the **entire** `python_vs_cpp` extraction
schedule. The report only needs to be regenerated when the pinned dataset
revision moves, when `--n-samples` increases, or when a pair starts failing
on a refreshed toolchain. It is independent of which model checkpoints HF
publishes.

The experiment schedule is bounded by what HuggingFace actually publishes per
model. The verified publication counts and selected counts today are:

| Model | HF ID | Selected / published |
|-------|-------|----------------------|
| Think-SFT | `allenai/Olmo-3-7B-Think-SFT` | 10 / 43 |
| Instruct-SFT | `allenai/Olmo-3-7B-Instruct-SFT` | 1 / 1 (`main` only) |
| RL-Zero-Math | `allenai/Olmo-3-7B-RL-Zero-Math` | 10 / 19 |
| RL-Zero-Code | `allenai/Olmo-3-7B-RL-Zero-Code` | 10 / 29 |
| RL-Zero-IF | `allenai/Olmo-3-7B-RL-Zero-IF` | 10 / 19 |
| RL-Zero-General | `allenai/Olmo-3-7B-RL-Zero-General` | 8 / 8 |
| RL-Zero-Mix | `allenai/Olmo-3-7B-RL-Zero-Mix` | 10 / 19 |
| **Total selected** | | **59** |

These caps live in `src/config.py` (`select_uniform_checkpoints`) and are
re-asserted by `tests/test_experiment_config.py`. The validator does **not**
re-fetch model checkpoints — it only streams the pinned HumanEval-X JSONL —
so a freshly generated report at the default `--n 50` is reusable across all
59 checkpoint extractions. If HF later publishes more checkpoints, the static
schedule in `src/config.py` must be audited and updated explicitly; the existing
report still applies because it binds the dataset (`revision` + SHA-256 hashes),
not the downstream model schedule.

## Workflow

```bash
# 1. Validate (network needed to stream the pinned JSONL)
uv run python experiments/validate_humaneval_x.py

# 2. Run extraction (preflight consumes the report above)
experiments/run_concept_dynamics.sh quick   # python_vs_cpp enabled by default
experiments/run_concept_dynamics.sh full    # python_vs_cpp enabled by default
```

For tests, the validator's runner and dataset loader are injectable seams
(`SandboxRunner` Protocol and the `dataset_loader` parameter), so the test
suite never spawns bwrap or touches the network. See
`tests/test_humaneval_x_validator.py` for the full coverage matrix.

## Outcomes other concepts do not need

The preflight gate fires only when `python_vs_cpp` is in the selected concept
list. Running the other three concepts (MATH, FLORES+, WinoGender) requires no
HumanEval-X report.
