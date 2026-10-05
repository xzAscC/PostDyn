# Local Claude instructions

Read and follow AGENTS.md and AGENTS.local.md before starting work.

Claude writes code, tests, scripts, and experiments. Codex edits LaTeX manuscripts and their bibliography. Neither agent may do the other's work; ask the user to hand off tasks that cross this boundary.

To show formulas, render a temporary PDF under /tmp/ and open it with Zathura. Do not use terminal/chat formula rendering as the presentation.

## Experiment validation order

Before submitting any experiment to OSC A100 or H100 GPUs, Claude must first pass the relevant local tests and complete a small end-to-end run on the local RTX 4090. Queue the local GPU run with `gpu-queue`; do not run it directly. Verify that the run completes and writes the expected results and logs before remote submission. Repeat this check after changes to the experiment code or configuration; if the local check cannot run or fails, resolve it or ask the user before submitting remotely.
