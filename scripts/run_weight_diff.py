"""Compare each run's start checkpoint with its final or intermediate checkpoints, tensor by tensor.

Rows go to <output>/<run>/<label>.jsonl (label = final or stepNNNNNN), one summary JSON per
checkpoint next to it. Checkpoints are read from the local Hugging Face cache only; with
--wait-minutes a job polls until the downloader has finished a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from postdyn.persistence import RunDir, atomic_write_json, tee_log
from postdyn.weight_diff import compare, tensor_files

MEAN_KEYS = ("one_minus_cos", "gini", "rank_ratio", "erank_ratio", "overlap")


@dataclass(frozen=True)
class Task:
    run: dict[str, Any]
    label: str
    end: dict[str, Any]


def build_tasks(runs: list[dict[str, Any]], which: str, shard: int = 0, num_shards: int = 1) -> list[Task]:
    """Flatten runs into (run, checkpoint) tasks in config order, then keep every num_shards-th."""
    tasks = []
    for run in runs:
        if which == "final":
            tasks.append(Task(run, "final", run["final"]))
        else:
            tasks += [Task(run, f"step{c['step']:06d}", c) for c in run["intermediates"]]
    return tasks[shard::num_shards]


def resolve(spec: dict[str, Any], wait_minutes: float) -> Path:
    """Local directory of a checkpoint; wait until every safetensors shard is present."""
    deadline = time.time() + 60 * wait_minutes
    while True:
        try:
            if "path" in spec:
                path = Path(spec["path"])
            else:
                path = Path(snapshot_download(spec["repo"], revision=spec["revision"], local_files_only=True))
            files = tensor_files(path)
            if files and all(f.is_file() for f in set(files.values())):
                return path
        except Exception as exc:  # not downloaded yet
            if time.time() >= deadline:
                raise
            print(f"waiting for {spec}: {type(exc).__name__}", flush=True)
        else:
            if time.time() >= deadline:
                raise FileNotFoundError(f"incomplete checkpoint {spec}")
        time.sleep(60)


def summarize(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    numel = sum(r["numel"] for r in rows)
    summary = {"tensors": len(rows), "sparsity": 1 - sum(r["changed_tol"] for r in rows) / numel}
    for key in MEAN_KEYS:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weight_runs.json")
    parser.add_argument("--runs", nargs="*", help="run names (default: all)")
    parser.add_argument("--which", choices=["final", "intermediate"], default="final")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", default="logs/weights")
    parser.add_argument("--basis-root", default=None, help="cache of W0 principal bases, one folder per start")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wait-minutes", type=float, default=0)
    args = parser.parse_args(argv)

    runs = json.loads(Path(args.config).read_text())["runs"]
    if args.runs:
        unknown = set(args.runs) - {r["name"] for r in runs}
        if unknown:
            parser.error(f"unknown runs: {sorted(unknown)}")
        runs = [r for r in runs if r["name"] in args.runs]
    tasks = build_tasks(runs, args.which, args.shard, args.num_shards)
    output = Path(args.output)

    with tee_log(RunDir(output / f"{args.which}-s{args.shard}of{args.num_shards}")):
        print(f"{len(tasks)} tasks on {args.device}: {[(t.run['name'], t.label) for t in tasks]}", flush=True)
        for task in tasks:
            t0 = time.time()
            start = resolve(task.run["start"], args.wait_minutes)
            end = resolve(task.end, args.wait_minutes)
            print(f"== {task.run['name']} {task.label}: {start} -> {end}", flush=True)
            run_dir = RunDir(output / task.run["name"])
            out = run_dir.path(f"{task.label}.jsonl")
            basis = None
            if args.basis_root:
                key = task.run["start"].get("repo", task.run["start"].get("path", "")).replace("/", "__")
                basis = Path(args.basis_root) / key
            compare(start, end, out, basis_dir=basis, device=args.device,
                    log=lambda m: print(m, flush=True))
            summary = {"run": task.run["name"], "family": task.run["family"], "row": task.run["row"],
                       "label": task.label, "start": str(start), "end": str(end), **summarize(out)}
            atomic_write_json(run_dir.path(f"{task.label}.summary.json"), summary)
            print(f"   done in {time.time() - t0:.0f}s: {summary}", flush=True)


if __name__ == "__main__":
    main()
