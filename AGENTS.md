# AGENTS

1. Use TDD: write tests before writing source code.
2. Start work on a new branch. When the work is finished, commit all changes and open a PR. Keep the PR concise and clear; do not add fluff.
3. Use only the folders already listed. Put each file in its corresponding folder. Do not create new folders unless I am unavailable and you cannot ask me.
4. Use uv to manage virtual environments.
5. If you are unsure, ask me first. If you cannot ask me, search the website or documentation. Do not guess.
6. Save figures as PDF first. Do not save as PNG.
7. Keep code clean and concise.
8. experiments must log incrementally: persist partial results (json) after each completed method or condition, and tee every print to both stdout and a log file under `logs/`. never buffer all output until the end. `logs/` stores log files only; do not put other files there.
9. before using a gpu, queue the job with `gpu-queue` (path command). machine-level fifo: one job at a time, gated on free gpu memory. `gpu-queue add <name> <command...>` to enqueue; `list` / `remove` / `status` / `start` / `stop` as needed. do not run gpu jobs directly.
