"""Download the checkpoints listed in configs/weight_runs.json into the Hugging Face cache.

One checkpoint at a time, one file at a time, 3 s between checkpoints, in config order so
that GPU jobs (run_weight_diff.py --wait-minutes) can start on early runs while later
runs download. Already cached files are skipped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

# Metadata first: a checkpoint counts as complete only once its index and every shard exist,
# so the index must not be the last file to arrive.
META_PATTERNS = ["*.json", "*.txt", "*.jinja", "*.model", "*.py"]


def checkpoint_list(runs: list[dict[str, Any]], which: str) -> list[tuple[str, str]]:
    """(repo, revision) pairs in run order: each run's start, then its final or intermediates."""
    seen: list[tuple[str, str]] = []
    for run in runs:
        ends = [run["final"]] if which == "final" else run["intermediates"]
        for spec in [run["start"], *ends]:
            key = (spec["repo"], spec["revision"])
            if key not in seen:
                seen.append(key)
    return seen


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weight_runs.json")
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--which", choices=["final", "intermediate"], default="intermediate")
    args = parser.parse_args(argv)
    runs = json.loads(Path(args.config).read_text())["runs"]
    if args.runs:
        runs = [r for r in runs if r["name"] in args.runs]
    for repo, revision in checkpoint_list(runs, args.which):
        t0 = time.time()
        try:
            snapshot_download(repo, revision=revision, allow_patterns=META_PATTERNS, max_workers=1)
            path = snapshot_download(repo, revision=revision, allow_patterns=["*.safetensors"], max_workers=1)
            print(f"OK {repo}@{revision} {time.time() - t0:.0f}s {path}", flush=True)
        except Exception as exc:
            print(f"FAIL {repo}@{revision}: {exc}", flush=True)
        time.sleep(3)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
