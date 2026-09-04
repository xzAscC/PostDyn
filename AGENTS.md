# AGENTS

1. Use TDD: write tests before writing source code.
2. Start work on a new branch. When the work is finished, commit all changes and open a PR. Keep the PR concise and clear; do not add fluff.
3. Use only the folders already listed. Put each file in its corresponding folder. Do not create new folders unless I am unavailable and you cannot ask me.
4. Use uv to manage virtual environments.
5. If you are unsure, search the website or documentation. Do not guess.
6. Save figures as PDF first. Do not save as PNG.
7. Keep code clean and concise.
8. Experiments must log incrementally: persist partial results as JSON after each completed method or condition, and tee every print to both stdout and a log file under `logs/`. Run artifacts (incremental JSON/JSONL/safetensors and their `run.log`) live inside per-run subdirectories such as `logs/q1/` and `logs/q2/`; keep the `logs/` root itself for shared log files only.
9. Before using a GPU, queue the job with `gpu-queue` (path command). Machine-level FIFO: one job at a time, gated on free GPU memory. `gpu-queue add <name> <command...>` to enqueue; `list` / `remove` / `status` / `start` / `stop` as needed. Do not run GPU jobs directly.

## Repository conventions

- Python package code lives in `src/postdyn/`; experiment runners live in `scripts/`.
- Run tests with `uv run pytest`; `pyproject.toml` configures `pythonpath = ["src"]`.
- Use incremental JSON/JSONL resume artifacts for long runs. Materialized `data/domain_prompts/` files are gitignored artifacts.
- Keep generated run artifacts under the selected output directory; `logs/` is for log files only.
- Use a new branch and open a concise PR when work is complete.
- Add or update tests before implementation and keep the full suite green.
