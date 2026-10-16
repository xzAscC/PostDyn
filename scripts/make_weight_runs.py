"""Write configs/weight_runs.json: start, final and 10 intermediate checkpoints per run.

Intermediate checkpoints are Hugging Face branches (OLMo 3) or one repository per step
(third-party Qwen runs). Each stage starts from main of the previous stage, never from its
own first checkpoint. Run on a machine with Hugging Face API access (metadata only).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from postdyn.weight_diff import select_uniform

RUNS: list[dict[str, Any]] = [
    {"name": "olmo3-7b_sft", "family": "olmo3-7b", "row": "SFT",
     "start": "allenai/Olmo-3-1025-7B", "final": "allenai/Olmo-3-7B-Think-SFT",
     "branches": r"step(\d+)"},
    {"name": "olmo3-7b_rl", "family": "olmo3-7b", "row": "RL",
     "start": "allenai/Olmo-3-1025-7B", "final": "allenai/Olmo-3-7B-RL-Zero-Mix",
     "branches": r"step_(\d+)"},
    {"name": "qwen-scaleml_sft", "family": "qwen-math-7b-scaleml", "row": "SFT",
     "start": "Qwen/Qwen2.5-Math-7B", "final": "ScaleML-RLHF/qwmathbase_raw_raft_step320",
     "repos": r"qwmathbase_raw_raft_step(\d+)"},
    {"name": "qwen-scaleml_rl", "family": "qwen-math-7b-scaleml", "row": "RL",
     "start": "Qwen/Qwen2.5-Math-7B", "final": "ScaleML-RLHF/qwmathbase_grpo_n4_bz512_step180",
     "repos": r"qwmathbase_grpo_n4_bz512_step(\d+)"},
    {"name": "qwen-cutelemonlili_sft", "family": "qwen-math-7b-cutelemonlili", "row": "SFT",
     "start": "Qwen/Qwen2.5-Math-7B-Instruct",
     "final": "cutelemonlili/Qwen2.5-Math-7B-Instruct_QwQ_SFT_step_1818_deepscaler",
     "repos": r"Qwen2\.5-Math-7B-Instruct_QwQ_SFT_step_(\d+)_deepscaler"},
    {"name": "qwen-cutelemonlili_rl", "family": "qwen-math-7b-cutelemonlili", "row": "RL",
     "start": "Qwen/Qwen2.5-Math-7B-Instruct", "final": "cutelemonlili/Math-7B-Instruct-step620-deepscaler",
     "repos": r"Math-7B-Instruct-step(\d+)-deepscaler"},
    # 32B Think-SFT has two learning-rate runs; main is their merge. Use the 1e-4 run as in configs.
    {"name": "olmo3-32b_sft", "family": "olmo3-32b", "row": "SFT",
     "start": "allenai/Olmo-3-1125-32B", "final": "allenai/Olmo-3-32B-Think-SFT",
     "branches": r"1e-4-step(\d+)"},
    {"name": "olmo3-32b_rl", "family": "olmo3-32b", "row": "RL",
     "start": "allenai/Olmo-3-32B-Think-DPO", "final": "allenai/Olmo-3-32B-Think",
     "branches": r"step_(\d+)"},
]


def pick_steps(names: list[str], pattern: str, final: str | None = None, n: int = 10) -> list[tuple[int, str]]:
    """(step, name) of names that fully match pattern, final excluded, n evenly spaced by step."""
    found = []
    for name in names:
        match = re.fullmatch(pattern, name)
        if match and name != final:
            found.append((int(match.group(1)), name))
    return select_uniform(sorted(found), n)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="configs/weight_runs.json")
    args = parser.parse_args(argv)
    api = HfApi()
    runs = []
    for spec in RUNS:
        time.sleep(3)  # polite: one metadata request at a time
        if "branches" in spec:
            names = [b.name for b in api.list_repo_refs(spec["final"]).branches]
            picked = pick_steps(names, spec["branches"])
            mids = [{"repo": spec["final"], "revision": name, "step": step} for step, name in picked]
        else:
            author, final_name = spec["final"].split("/")
            names = [m.id.split("/")[1] for m in api.list_models(author=author, limit=5000)]
            picked = pick_steps(names, spec["repos"], final=final_name)
            mids = [{"repo": f"{author}/{name}", "revision": "main", "step": step} for step, name in picked]
        runs.append({"name": spec["name"], "family": spec["family"], "row": spec["row"],
                     "start": {"repo": spec["start"], "revision": "main"},
                     "final": {"repo": spec["final"], "revision": "main"},
                     "intermediates": mids})
        print(spec["name"], [m["step"] for m in mids], flush=True)
    Path(args.output).write_text(json.dumps({"runs": runs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
