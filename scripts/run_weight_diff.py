"""Compare each run's start checkpoint with its final or intermediate checkpoints, tensor by tensor.

Rows go to <output>/<run>/<label>.jsonl (label = final or stepNNNNNN), one summary JSON per
checkpoint next to it. Checkpoints are read from the local Hugging Face cache only.

Every job is a worker over the same task list: it skips tasks that are done (summary exists),
held by another worker (file lock), or not downloaded yet, and with --wait-minutes it keeps
coming back to the skipped ones. Jobs in different queues can therefore share the work.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from postdyn.persistence import RunDir, atomic_write_json, tee_log
from postdyn.weight_diff import compare, tensor_files

MEAN_KEYS = ("one_minus_cos", "gini", "rank_ratio", "erank_ratio", "overlap")


@dataclass(frozen=True)
class Task:
    run: dict[str, Any]
    label: str
    end: dict[str, Any]


def build_tasks(runs: list[dict[str, Any]], which: str, shard: int = 0, num_shards: int = 1) -> list[Task]:
    """Flatten runs into (run, checkpoint) tasks: finals first, then intermediates in config order."""
    tasks = []
    if which in ("final", "all"):
        tasks += [Task(run, "final", run["final"]) for run in runs]
    if which in ("intermediate", "all"):
        for run in runs:
            tasks += [Task(run, f"step{c['step']:06d}", c) for c in run["intermediates"]]
    return tasks[shard::num_shards]


def ready(spec: dict[str, Any]) -> Path | None:
    """Local directory of a fully downloaded checkpoint, or None."""
    try:
        if "path" in spec:
            path = Path(spec["path"])
        else:
            path = Path(snapshot_download(spec["repo"], revision=spec["revision"], local_files_only=True))
        if not (path / "model.safetensors.index.json").is_file() and any(path.glob("*-of-*.safetensors")):
            return None  # sharded checkpoint whose index has not arrived yet (shards download first)
        files = tensor_files(path)
    except Exception:
        return None
    return path if files and all(f.is_file() for f in set(files.values())) else None


def summarize(path: Path) -> dict[str, float]:
    rows = {r["name"]: r for r in map(json.loads, filter(str.strip, path.read_text().splitlines()))}
    numel = sum(r["numel"] for r in rows.values())
    summary = {"tensors": len(rows), "sparsity": 1 - sum(r["changed_tol"] for r in rows.values()) / numel}
    for key in MEAN_KEYS:
        values = [r[key] for r in rows.values() if r.get(key) is not None]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


def summarize_angles(path: Path) -> dict[str, float]:
    """Mean over matrices (tensors with a row std) of the mean row angle and its bf16 floor."""
    rows = {r["name"]: r for r in map(json.loads, filter(str.strip, path.read_text().splitlines()))}
    mats = [r for r in rows.values() if r.get("theta_std") is not None] or list(rows.values())
    return {"tensors": len(rows), "theta_mean": sum(r["theta_mean"] for r in mats) / len(mats),
            "theta_ulp": sum(r["theta_ulp"] for r in mats) / len(mats)}


def summarize_spectral(path: Path) -> dict[str, float]:
    """Mean over matrices of the max top-k principal angles, spectrum drift and relative update norm."""
    rows = {r["name"]: r for r in map(json.loads, filter(str.strip, path.read_text().splitlines()))}
    mats = [r for r in rows.values() if "nss" in r]
    summary: dict[str, float] = {"tensors": len(rows)}
    for key in ("pa_u_max", "pa_v_max", "nss", "rel_update_norm"):
        values = [r[key] for r in mats if r.get(key) is not None]
        if values:
            summary[key] = sum(values) / len(values)
    return summary


SUMMARIZE = {"metrics": summarize, "angles": summarize_angles, "spectral": summarize_spectral}


def run_task(task: Task, output: Path, basis_root: str | None, device: str, mode: str = "metrics") -> str:
    """Return done, unready, busy, failed or computed."""
    run_dir = RunDir(output / task.run["name"])
    stem = task.label if mode == "metrics" else f"{task.label}.{mode}"
    summary_path = run_dir.path(f"{stem}.summary.json")
    if summary_path.is_file():
        return "done"
    start, end = ready(task.run["start"]), ready(task.end)
    if start is None or end is None:
        return "unready"
    lock_path = run_dir.path(f"{stem}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy"
        if summary_path.is_file():  # another worker finished it meanwhile
            return "done"
        t0 = time.time()
        print(f"== {task.run['name']} {task.label}: {start} -> {end}", flush=True)
        basis = None
        if basis_root:
            key = task.run["start"].get("repo", task.run["start"].get("path", "")).replace("/", "__")
            basis = Path(basis_root) / key
        out = run_dir.path(f"{stem}.jsonl")
        try:
            compare(start, end, out, basis_dir=basis, device=device, log=lambda m: print(m, flush=True),
                    mode=mode, keep_curves=task.label == "final")
        except Exception:  # record and move on; one bad checkpoint must not stop the worker
            run_dir.path(f"{stem}.failed").write_text(traceback.format_exc())
            print(f"   FAILED {task.run['name']} {task.label}:\n{traceback.format_exc()}", flush=True)
            return "failed"
        summary = {"run": task.run["name"], "family": task.run["family"], "row": task.run["row"],
                   "label": task.label, "start": str(start), "end": str(end),
                   **SUMMARIZE[mode](out)}
        atomic_write_json(summary_path, summary)
        print(f"   done in {time.time() - t0:.0f}s: {summary}", flush=True)
    return "computed"


def main(argv: list[str] | None = None) -> list[tuple[str, str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/weight_runs.json")
    parser.add_argument("--runs", nargs="*", help="run names (default: all)")
    parser.add_argument("--which", choices=["final", "intermediate", "all"], default="final")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", default="logs/weights")
    parser.add_argument("--basis-root", default=None, help="cache of W0 principal bases, one folder per start")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wait-minutes", type=float, default=0, help="keep retrying unready or busy tasks")
    parser.add_argument("--angles", action="store_true", help="only per-row rotation angles (no SVD)")
    parser.add_argument("--spectral", action="store_true",
                        help="update norm, top-10%% principal angles and spectrum drift")
    args = parser.parse_args(argv)

    runs = json.loads(Path(args.config).read_text())["runs"]
    if args.runs:
        unknown = set(args.runs) - {r["name"] for r in runs}
        if unknown:
            parser.error(f"unknown runs: {sorted(unknown)}")
        runs = [r for r in runs if r["name"] in args.runs]
    pending = build_tasks(runs, args.which, args.shard, args.num_shards)
    output = Path(args.output)
    deadline = time.time() + 60 * args.wait_minutes

    mode = "angles" if args.angles else "spectral" if args.spectral else "metrics"
    prefix = "" if mode == "metrics" else f"{mode}-"
    with tee_log(RunDir(output / f"{prefix}{args.which}-s{args.shard}of{args.num_shards}")):
        print(f"{len(pending)} tasks on {args.device}", flush=True)
        failed = []
        while True:
            skipped, progressed, unready = [], False, False
            for task in pending:
                status = run_task(task, output, args.basis_root, args.device, mode)
                if status == "failed":
                    failed.append(task)
                if status in ("unready", "busy"):
                    skipped.append(task)
                progressed |= status == "computed"
                unready |= status == "unready"
            pending = skipped
            # tasks held by other workers are theirs; only wait for downloads
            if not unready or time.time() >= deadline:
                break
            if not progressed:
                print(f"{len(pending)} tasks waiting (unready or busy); retry in 60 s", flush=True)
                time.sleep(60)
        undone = [(t.run["name"], t.label) for t in [*failed, *pending]]
        print(f"worker exits; not done here: {undone}", flush=True)
    return undone


if __name__ == "__main__":
    main()
