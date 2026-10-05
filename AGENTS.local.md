# Local agent responsibilities

- Claude owns software implementation: source code, scripts, tests, and experiments. Claude must not edit LaTeX manuscript files or manuscript content.
- Codex owns LaTeX manuscript editing and associated bibliography, compilation, and PDF review. Codex must not write or modify application code, experiment runners, or tests, and must not run experiments.
- Neither agent may perform the other agent's work. If a task crosses this boundary, explain the required handoff to the user instead of doing the work.
- Both agents may update these instruction files when the user explicitly requests it.
- When formulas need to be shown to the user, create a temporary PDF under /tmp/, check its rendering, and open it with Zathura. Do not rely on terminal/chat LaTeX or plain-text formulas as the formula presentation.
- Keep temporary formula explanations separate from the manuscript unless the user requests a manuscript edit.
- Preserve the other agent's working changes. Do not stage or commit the other agent's files.

## Local validation before OSC experiments

Before submitting any experiment to OSC A100 or H100 GPUs, Claude must first pass the relevant local tests and complete a small end-to-end run on the local RTX 4090. Queue the local GPU run with `gpu-queue`; do not run it directly. Verify that the run completes and writes the expected results and logs before remote submission. Repeat this check after changes to the experiment code or configuration; if the local check cannot run or fails, resolve it or ask the user before submitting remotely.
