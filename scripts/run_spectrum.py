"""Can the hidden-state covariance of a post-trained model explain what it gained over its base?

Reads the answers of scripts/run_passk.py. Every model sees the exact answer tokens under its own
prompt, so a gap between two models on one text is the models' gap, not a gap in the text.

moments (GPU): for each model and answer text, the within-problem residual covariance of every
layer over the rows that pick the first 10% of each answer (from the last prompt token on), one
file per fold under <out>/moments/.
report (CPU): per layer, how the covariances compare, and the k two halves of the data agree on.
select (CPU): the problems base gets wrong and post gets right, from the pass@k scores.
ablate (GPU): sample with a map from the other folds applied to every layer for the first tokens
only, then let the model run free. Writes run_passk chunks for the selected problems only, so
score them with run_passk.py score --partial.

Usage:
  python scripts/run_spectrum.py moments --pass-root R --base-model B --post-model P --out O
  python scripts/run_spectrum.py report --out O
  python scripts/run_spectrum.py select --pass-root R --base-max 0.25 --post-min 0.75 --out O
  python scripts/run_spectrum.py ablate --pass-root R --base-model B --post-model P \\
      --kind reorder --out O
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from postdyn import spectrum  # noqa: E402
from postdyn.persistence import atomic_write_json  # noqa: E402
from postdyn.spectra import eigensystem  # noqa: E402
from scripts.run_passk import (  # noqa: E402
    KINDS,
    STOP_TOKENS,
    check_settings,
    check_transformers,
    claim,
    load_items,
    log,
    read_chunks,
    render,
    write_chunk,
)

TASK = "math500"
FOLDS = 5
LIMIT = 500
# Names of the pass@k runs, each with the prompt its model was sampled under.
RUNS = {"base": ("base-plain", "plain"), "post": ("sft-chat", "chat")}
K_GRID = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
K_AGREEMENT = 0.9  # squared subspace overlap two halves must reach
ABLATIONS = (*spectrum.KINDS, "continue")
RANKED = ("random", "spectrum", "reorder", "full")  # the maps that act on the top k only
MAX_PROMPT_TOKENS = 2048


def complete_texts(row: dict[str, Any], samples: int) -> list[str]:
    """The first answers that ended by themselves: a truncated one is often an endless loop."""
    return [s["text"] for s in row["samples"] if s["finish_reason"] == "stop"][:samples]


def common_ids(order: Sequence[str], rows: dict[str, dict[str, Any]], problems: int) -> list[str]:
    """The first ``problems`` of the fixed order that every run has finished."""
    ids = [problem for problem in order if all(problem in held for held in rows.values())]
    return ids[:problems]


def prefix_batches(
    prompt: list[int], answers: Sequence[list[int]], fraction: float, batch: int, pad: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Prompt + prefix, padded on the right, and each token's weight (0 outside the prefix).

    Weight 1/length gives every answer the same total, so long answers do not rule the covariance.
    """
    for start in range(0, len(answers), batch):
        part = answers[start : start + batch]
        prefixes = [a[: spectrum.prefix_length(len(a), fraction)] for a in part if a]
        if not prefixes:
            continue
        width = len(prompt) + max(map(len, prefixes))
        ids = torch.full((len(prefixes), width), pad, dtype=torch.long)
        mask = torch.zeros((len(prefixes), width), dtype=torch.long)
        weights = torch.zeros((len(prefixes), width), dtype=torch.float64)
        for row, prefix in enumerate(prefixes):
            end = len(prompt) + len(prefix)
            ids[row, :end] = torch.tensor(prompt + prefix)
            mask[row, :end] = 1
            # The rows that pick the prefix tokens: from the last prompt token, as ablate steers.
            weights[row, len(prompt) - 1 : end - 1] = 1.0 / len(prefix)
        yield ids, mask, weights


def load(path: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()


def rows_of(root: Path, condition: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in read_chunks(root / condition / TASK)}


def keep_knobs(directory: Path, knobs: dict[str, Any]) -> None:
    """As check_settings, for the knobs of this script that run_passk does not know."""
    path = directory / "knobs.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(knobs)):
        raise ValueError(f"{directory} was started with other knobs: {path.read_text()}")
    atomic_write_json(path, knobs)


def moment_path(out: Path, model: str, text: str, fold: int) -> Path:
    return out / "moments" / f"{model}__on__{text}" / f"fold_{fold}.pt"


def frozen_ids(args: argparse.Namespace, rows: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """One list per output directory: later answers must not mix into finished folds."""
    path = Path(args.out) / "ids.json"
    if path.exists():
        return json.loads(path.read_text())
    order = [item.id for item in load_items(TASK, LIMIT, None)]
    ids = common_ids(order, rows, args.problems)
    atomic_write_json(path, ids)
    return ids


def moments(args: argparse.Namespace) -> None:
    check_transformers(__import__("transformers").__version__)
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.pass_root)
    keep_knobs(out, {"samples": args.samples, "fraction": args.fraction})
    rows = {name: rows_of(root, run) for name, (run, _) in RUNS.items()}
    ids = frozen_ids(args, rows)
    items = {item.id: item for item in load_items(TASK, LIMIT, None)}
    tokenizer = AutoTokenizer.from_pretrained(args.post_model)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    paths = {"base": args.base_model, "post": args.post_model}

    for name, (_, prompt_format) in RUNS.items():
        todo = [
            (text, fold)
            for text in RUNS
            for fold in range(FOLDS)
            if not moment_path(out, name, text, fold).exists()
        ]
        if not todo:
            continue
        model = load(paths[name])
        layers = len(model.model.layers)
        widths = {layer: model.config.hidden_size for layer in range(layers)}
        device = next(model.parameters()).device
        for text, fold in todo:
            held = spectrum.FoldMoments(widths, FOLDS)
            members = [problem for problem in ids if spectrum.fold_of(problem, FOLDS) == fold]
            for problem in members:
                prompt = tokenizer(render(tokenizer, KINDS[TASK], items[problem].prompt, prompt_format))
                answers = [
                    tokenizer(answer, add_special_tokens=False)["input_ids"]
                    for answer in complete_texts(rows[text][problem], args.samples)
                ]
                running = spectrum.ProblemMoments(widths, device)
                batches = prefix_batches(prompt["input_ids"], answers, args.fraction, args.batch, pad)
                for input_ids, mask, weights in batches:
                    use = weights.to(device) > 0
                    with torch.no_grad(), spectrum.residual_capture(model) as states:
                        # The decoder alone: the hooks need no logits.
                        model.model(input_ids=input_ids.to(device), attention_mask=mask.to(device))
                        running.update(
                            {layer: state[use] for layer, state in states.items()}, weights.to(device)[use]
                        )
                if running.weight > 0:
                    held.add(problem, fold, *running.finish())
            path = moment_path(out, name, text, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(
                {"scatter": held.scatter.get(fold, {}), "weight": held.weight.get(fold, 0.0),
                 "problem_means": held.problem_means, "problem_fold": held.problem_fold},
                temporary,
            )
            temporary.replace(path)
            log(out, f"moments {name} on {text} fold {fold}: {len(members)} problems")
        del model
        torch.cuda.empty_cache()


def read_moments(out: Path, model: str, text: str) -> spectrum.FoldMoments:
    held = spectrum.FoldMoments({}, FOLDS)
    for fold in range(FOLDS):
        # mmap: a scatter of 4 GB is read only when a layer of it is used.
        state = torch.load(moment_path(out, model, text, fold), weights_only=True, mmap=True)
        if state["scatter"]:  # a small run can leave a fold empty
            held.scatter[fold] = state["scatter"]
            held.weight[fold] = state["weight"]
        held.problem_means.update(state["problem_means"])
        held.problem_fold.update(state["problem_fold"])
    return held


def report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sets = {(m, t): read_moments(out, m, t) for m in RUNS for t in RUNS}
    folds = sorted(sets["base", "post"].scatter)
    layers = sorted(sets["base", "post"].scatter[folds[0]])
    table = []
    for layer in layers:
        row: dict[str, Any] = {"layer": layer}
        ks = {}
        for model in RUNS:
            halves = [
                eigensystem(sets[model, "post"].stats(layer, only=part).sigma.to(device))[1]
                for part in (folds[: len(folds) // 2], folds[len(folds) // 2 :])
            ]
            ks[model] = spectrum.stable_k(*halves, K_GRID, K_AGREEMENT)
        row["stable_k"] = ks
        row["k"] = min(ks.values())
        pair = lambda a, b: (  # noqa: E731
            sets[a].stats(layer).sigma.to(device),
            sets[b].stats(layer).sigma.to(device),
        )
        size = max(row["k"], 1)
        row["model_gap_on_post_text"] = spectrum.spectrum_summary(*pair(("base", "post"), ("post", "post")), size)
        row["model_gap_on_base_text"] = spectrum.spectrum_summary(*pair(("base", "base"), ("post", "base")), size)
        row["text_gap_in_base"] = spectrum.spectrum_summary(*pair(("base", "base"), ("base", "post")), size)
        table.append(row)
        atomic_write_json(out / "report.json", table)
        log(out, f"report layer {layer}: k {row['k']}")


def select_ids(
    base: dict[str, Sequence[str]],
    post: dict[str, Sequence[str]],
    base_max: float,
    post_min: float,
    allowed: Collection[str],
) -> list[str]:
    """Problems by their correct rate over all samples, so one lucky sample decides nothing.

    Only ``allowed`` ones: an ablation needs each model's own mean on the problem.
    """

    def rate(outcomes: Sequence[str]) -> float:
        return sum(outcome == "correct" for outcome in outcomes) / len(outcomes)

    return [
        problem
        for problem, outcomes in post.items()
        if problem in base
        and problem in allowed
        and rate(base[problem]) <= base_max
        and rate(outcomes) >= post_min
    ]


def read_scores(root: Path, condition: str) -> dict[str, list[str]]:
    with open(root / condition / "scores" / f"{TASK}.jsonl") as f:
        return {row["id"]: row["outcomes"] for row in map(json.loads, f)}


def select(args: argparse.Namespace) -> None:
    root, out = Path(args.pass_root), Path(args.out)
    base, post = (read_scores(root, RUNS[name][0]) for name in ("base", "post"))
    # The moments' problems that each model's own text covers: ablate reads these means.
    allowed = set.intersection(
        *(set(read_moments(out, name, name).problem_means) for name in RUNS)
    )
    ids = select_ids(base, post, args.base_max, args.post_min, allowed)
    atomic_write_json(out / "selected.json", ids)
    log(out, f"selected {len(ids)} of {len(set(base) & set(post) & allowed)}")


def steps_of(texts: Sequence[str], tokenizer: Any, fraction: float) -> int:
    """How many tokens to steer: the median prefix of the donor's answers to this problem."""
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    return int(statistics.median(spectrum.prefix_length(n, fraction) for n in lengths))


def maps_for(
    kind: str,
    fit: tuple[spectrum.FoldMoments, spectrum.FoldMoments],
    fold: int,
    ks: dict[int, int],
    seed: int,
    device: str = "cpu",
) -> dict[int, spectrum.StateMap]:
    """One map per layer, from every fold but ``fold``."""

    def stats(held: spectrum.FoldMoments, layer: int) -> spectrum.LayerStats:
        part = held.stats(layer, leave_out=fold)
        return spectrum.LayerStats(part.sigma.to(device), part.mean.to(device))

    maps = {}
    for layer, k in ks.items():
        made = spectrum.build_map(kind, stats(fit[0], layer), stats(fit[1], layer), k, seed)
        maps[layer] = spectrum.StateMap(made.left.float(), made.right.float(), made.shift.float())
    return maps


def ablate(args: argparse.Namespace) -> None:
    check_transformers(__import__("transformers").__version__)
    from transformers import AutoTokenizer

    from postdyn.vllm_steering import PrefixMaps  # sets vLLM's environment before it loads

    from vllm import LLM, SamplingParams  # noqa: I001

    out = Path(args.out)
    root = Path(args.pass_root)
    # Forward: base gets post's statistics. Reverse: post gets base's, to test necessity.
    source, donor = ("post", "base") if args.reverse else ("base", "post")
    run_dir = out / "ablate" / f"{args.kind}{'-reverse' if args.reverse else ''}"
    (run_dir / TASK).mkdir(parents=True, exist_ok=True)
    prompt_format = RUNS[source][1]
    args.tasks, args.limit, args.lcb_source = [TASK], LIMIT, None
    args.model, args.tokenizer, args.prompt_format = source, args.post_model, prompt_format
    args.max_tokens, args.chunk_problems = args.max_new_tokens, 1
    check_settings(run_dir, args)
    table = {
        row["layer"]: args.k if args.k is not None else row["k"]
        for row in json.loads((out / "report.json").read_text())
    }
    ids = json.loads((out / "selected.json").read_text())
    # Chunks are named by index in selected.json, so the list is part of the run.
    keep_knobs(run_dir, {"k": table, "fraction": args.fraction, "selected": ids})
    if unsteered := [layer for layer, k in table.items() if k < 1 and args.kind in RANKED]:
        log(run_dir, f"k is 0, so {args.kind} leaves layers {unsteered} unchanged")

    tokenizer = AutoTokenizer.from_pretrained(args.post_model)
    items = {item.id: item for item in load_items(TASK, LIMIT, None)}
    donors = rows_of(root, RUNS[donor][0])
    outcomes = read_scores(root, RUNS[donor][0])
    means = read_moments(out, source, source).problem_means
    llm = LLM(
        model=args.post_model if source == "post" else args.base_model,
        tokenizer=args.post_model,
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_new_tokens + MAX_PROMPT_TOKENS,  # the default fills an A100
        # What the rewrite needs: see postdyn/vllm_steering.py.
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    stop = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]

    def params(samples: int) -> Any:
        return SamplingParams(
            n=samples, temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_new_tokens, stop_token_ids=stop, skip_special_tokens=False,
        )

    fit, steering = None, None
    if args.kind != "continue":
        fit = (read_moments(out, source, "post"), read_moments(out, donor, "post"))
        steering = PrefixMaps(llm, sorted(table))
    held_fold, maps = None, {}

    # By fold, so only one fold's maps sit in GPU memory.
    order = sorted(enumerate(ids), key=lambda pair: spectrum.fold_of(pair[1], FOLDS))
    for index, problem in order:
        path = run_dir / TASK / f"chunk_{index:06d}.jsonl.gz"
        if path.exists() or not claim(path):
            continue
        started = time.time()
        try:
            correct = [
                sample["text"]
                for sample, outcome in zip(donors[problem]["samples"], outcomes[problem], strict=True)
                if outcome == "correct"
            ] or complete_texts(donors[problem], args.samples)
            steps = steps_of(correct, tokenizer, args.fraction)
            rendered = render(tokenizer, KINDS[TASK], items[problem].prompt, prompt_format)
            prompt = tokenizer(rendered)["input_ids"]
            if steering is None:
                leads = [
                    tokenizer(text, add_special_tokens=False)["input_ids"]
                    for text in (correct * args.samples)[: args.samples]
                ]
                leads = [a[: spectrum.prefix_length(len(a), args.fraction)] for a in leads]
                requests = llm.generate(
                    [{"prompt_token_ids": prompt + lead} for lead in leads], params(1)
                )
                made = [
                    (tokenizer.decode(lead, skip_special_tokens=False), len(lead), output)
                    for lead, request in zip(leads, requests, strict=True)
                    for output in request.outputs
                ]
            else:
                if held_fold != spectrum.fold_of(problem, FOLDS):
                    held_fold = spectrum.fold_of(problem, FOLDS)
                    maps = maps_for(args.kind, fit, held_fold, table, args.seed, "cuda")
                steering.set(maps, means[problem], len(prompt), steps)
                requests = llm.generate([{"prompt_token_ids": prompt}], params(args.samples))
                made = [("", 0, output) for output in requests[0].outputs]
        except BaseException:
            path.with_suffix(".claim").unlink(missing_ok=True)
            raise
        samples = [
            {
                "text": lead + output.text,
                "finish_reason": output.finish_reason,
                "tokens": lead_tokens + len(output.token_ids),
            }
            for lead, lead_tokens, output in made
        ]
        write_chunk(path, [{"id": problem, "samples": samples, "steered_tokens": steps}])
        path.with_suffix(".claim").unlink(missing_ok=True)
        log(run_dir, f"{problem}: {steps} steered, {time.time() - started:.0f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("moments")
    run.add_argument("--pass-root", required=True)
    run.add_argument("--base-model", required=True)
    run.add_argument("--post-model", required=True)
    run.add_argument("--problems", type=int, default=200)
    run.add_argument("--samples", type=int, default=64)
    run.add_argument("--fraction", type=float, default=0.1)
    run.add_argument("--batch", type=int, default=16)
    run.add_argument("--out", required=True)
    run.set_defaults(handler=moments)

    summary = commands.add_parser("report")
    summary.add_argument("--out", required=True)
    summary.set_defaults(handler=report)

    choosing = commands.add_parser("select")
    choosing.add_argument("--pass-root", required=True)
    choosing.add_argument("--base-max", type=float, required=True)
    choosing.add_argument("--post-min", type=float, required=True)
    choosing.add_argument("--out", required=True)
    choosing.set_defaults(handler=select)

    sampling = commands.add_parser("ablate")
    sampling.add_argument("--pass-root", required=True)
    sampling.add_argument("--base-model", required=True)
    sampling.add_argument("--post-model", required=True)
    sampling.add_argument("--kind", choices=ABLATIONS, required=True)
    sampling.add_argument("--reverse", action="store_true", help="post with base's statistics")
    sampling.add_argument("--samples", type=int, default=64)
    sampling.add_argument("--fraction", type=float, default=0.1)
    sampling.add_argument("--k", type=int, help="the same k in every layer, not report.json's")
    sampling.add_argument("--max-num-seqs", type=int, default=12)
    sampling.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    sampling.add_argument("--temperature", type=float, default=0.6)
    sampling.add_argument("--top-p", type=float, default=0.95)
    sampling.add_argument("--max-new-tokens", type=int, default=32768)
    sampling.add_argument("--seed", type=int, default=0)
    sampling.add_argument("--out", required=True)
    sampling.set_defaults(handler=ablate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
