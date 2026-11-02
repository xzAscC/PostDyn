"""Sample every problem many times, then score the samples and report pass@k.

generate (GPU): one gzip file per chunk of problems under <out-dir>/<task>/; a finished chunk is
never redone, so a job cut off by its time limit resumes with the same command.
score (CPU): writes <out-dir>/scores/<task>.jsonl and <out-dir>/summary.json. Separate from
generate because LiveCodeBench runs model code and must not hold a GPU meanwhile.
compare: run b minus run a, paired by problem.

Usage:
  python scripts/run_passk.py generate --model M --tokenizer T --prompt-format chat \\
      --tasks math500 livecodebench --samples 64 --out-dir DIR
  python scripts/run_passk.py score --out-dir DIR --workers 40
  python scripts/run_passk.py compare --a DIR_A --b DIR_B --out compare.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from postdyn import passk  # noqa: E402
from postdyn.bench import BenchItem, load_benchmark  # noqa: E402

KINDS = {"math500": "math", "livecodebench": "code"}
STOP_TOKENS = ("<|endoftext|>", "<|im_end|>")
RUN_KEYS = ("model", "prompt_format", "samples", "temperature", "top_p", "max_tokens", "limit", "seed")


def log(out_dir: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out_dir / "run.log", "a") as f:
        f.write(line + "\n")


def load_items(task: str, limit: int, source: str | None) -> list[BenchItem]:
    """All problems, or a fixed pseudo-random ``limit`` of them (load_benchmark's hash order)."""
    held_back, rest = load_benchmark(task, n_val=limit, source=source if task != "math500" else None)
    return held_back if limit else rest


def render(tokenizer: Any, kind: str, question: str, prompt_format: str) -> str:
    text = passk.user_text(kind, question, prompt_format)
    if prompt_format == "simple":
        return text
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def check_settings(out_dir: Path, args: argparse.Namespace) -> None:
    """Keep one directory to one setting, so resumed chunks stay comparable."""
    settings = {key: getattr(args, key) for key in RUN_KEYS}
    path = out_dir / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError(f"{out_dir} was started with other settings: {path.read_text()}")
    path.write_text(json.dumps(settings, indent=2))


def write_chunk(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomic, so a file that exists is a finished chunk."""
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    os.replace(temporary, path)


def read_chunks(directory: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(directory.glob("chunk_*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def generate(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_settings(out_dir, args)
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS],
        # </think> marks where the final response starts.
        skip_special_tokens=False,
    )
    for task in args.tasks:
        kind = KINDS[task]
        items = load_items(task, args.limit, args.lcb_source)
        directory = out_dir / task
        directory.mkdir(exist_ok=True)
        for begin in range(0, len(items), args.chunk_problems):
            path = directory / f"chunk_{begin:06d}.jsonl.gz"
            if path.exists():
                continue
            chunk = items[begin : begin + args.chunk_problems]
            started = time.time()
            requests = llm.generate(
                [render(tokenizer, kind, item.prompt, args.prompt_format) for item in chunk], params
            )
            rows = [
                {
                    "id": item.id,
                    "samples": [
                        {
                            "text": output.text,
                            "finish_reason": output.finish_reason,
                            "tokens": len(output.token_ids),
                        }
                        for output in request.outputs
                    ],
                }
                for item, request in zip(chunk, requests, strict=True)
            ]
            write_chunk(path, rows)
            tokens = sum(sample["tokens"] for row in rows for sample in row["samples"])
            seconds = time.time() - started
            log(
                out_dir,
                f"{task} {begin + len(chunk)}/{len(items)}: {tokens} tokens, {seconds:.0f} s, "
                f"{tokens / max(seconds, 1e-9):.0f} tokens/s",
            )


def _score_row(job: tuple[str, dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    kind, row, reference = job
    return {
        "id": row["id"],
        "outcomes": [passk.classify(kind, sample, reference) for sample in row["samples"]],
        "tokens": [sample["tokens"] for sample in row["samples"]],
    }


def score(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    settings = json.loads((out_dir / "settings.json").read_text())
    (out_dir / "scores").mkdir(exist_ok=True)
    summary = {}
    for directory in sorted(path for path in out_dir.iterdir() if path.name in KINDS):
        task = directory.name
        rows = read_chunks(directory)
        references = {
            item.id: item.reference for item in load_items(task, settings["limit"], args.lcb_source)
        }
        jobs = [(KINDS[task], row, references[row["id"]]) for row in rows]
        if args.workers > 1:
            with multiprocessing.Pool(args.workers, maxtasksperchild=20) as pool:
                scored = pool.map(_score_row, jobs, chunksize=1)
        else:
            scored = [_score_row(job) for job in jobs]
        with open(out_dir / "scores" / f"{task}.jsonl", "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in scored)
        summary[task] = passk.task_summary(scored, args.ks)
        log(out_dir, f"{task}: {summary[task]}")
    (out_dir / "summary.json").write_text(json.dumps({**summary, "settings": settings}, indent=2))


def compare_runs(args: argparse.Namespace) -> None:
    a, b = Path(args.a) / "scores", Path(args.b) / "scores"
    result = {}
    for path in sorted(a.glob("*.jsonl")):
        rows = [[json.loads(line) for line in open(run / path.name)] for run in (a, b)]
        result[path.stem] = passk.compare(rows[0], rows[1], args.ks)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("generate")
    run.add_argument("--model", required=True)
    run.add_argument("--tokenizer", required=True, help="holds the chat template; the base has none")
    run.add_argument("--prompt-format", choices=("chat", "simple"), required=True)
    run.add_argument("--tasks", nargs="+", choices=tuple(KINDS), default=list(KINDS))
    run.add_argument("--samples", type=int, default=64)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--max-tokens", type=int, default=16384)
    run.add_argument("--chunk-problems", type=int, default=4)
    run.add_argument("--limit", type=int, default=0, help="a fixed random subset of N problems")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--lcb-source", help="LiveCodeBench test6.jsonl (or its directory)")
    run.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    run.add_argument("--max-num-seqs", type=int, default=64)
    run.add_argument("--out-dir", required=True)
    run.set_defaults(handler=generate)

    scoring = commands.add_parser("score")
    scoring.add_argument("--out-dir", required=True)
    scoring.add_argument("--workers", type=int, default=8)
    scoring.add_argument("--ks", type=int, nargs="+", default=list(passk.KS))
    scoring.add_argument("--lcb-source")
    scoring.set_defaults(handler=score)

    comparison = commands.add_parser("compare")
    comparison.add_argument("--a", required=True)
    comparison.add_argument("--b", required=True)
    comparison.add_argument("--ks", type=int, nargs="+", default=list(passk.KS))
    comparison.add_argument("--out", required=True)
    comparison.set_defaults(handler=compare_runs)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
