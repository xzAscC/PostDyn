from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import scripts.q2_common as common
from postdyn.config import BENCHMARKS
from postdyn.persistence import RunDir, tee_log, atomic_write_json
from postdyn.spectra import eigensystem, subsim
from postdyn.verifiers import split_sentences, verify


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument("--layer", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--exp1-output", type=Path)
    return parser.parse_args(argv)


def sentence_final_states(
    tokenizer: Any,
    token_ids: list[int],
    text: str,
    captured: dict[int, torch.Tensor],
    prompt_token_len: int = 0,
) -> torch.Tensor:
    del token_ids
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    if torch.is_tensor(offsets):
        offsets = offsets.tolist()
    offsets = (
        offsets[0]
        if offsets
        and isinstance(offsets[0], list)
        and offsets[0]
        and isinstance(offsets[0][0], (list, tuple))
        else offsets
    )
    ends = []
    cursor = 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        cursor = start + len(sentence)
        ends.append(cursor)
    indices = []
    for end in ends:
        candidates = [
            i for i, pair in enumerate(offsets) if pair[0] < pair[1] and pair[1] <= end
        ]
        if candidates:
            indices.append(candidates[-1])
    if not captured:
        return torch.empty((0, 0), dtype=torch.float32)
    state = next(iter(captured.values()))
    state = state[0] if state.ndim == 3 else state
    if not indices:
        return state[:0]
    offset = prompt_token_len if state.shape[0] > len(offsets) else 0
    positions = torch.tensor([offset + index for index in indices], dtype=torch.long)
    if int(positions.max()) >= state.shape[0]:
        raise ValueError("captured states do not cover generated sentence-final tokens")
    return state[positions]


def solution_eigensystem(states: torch.Tensor, K: int):
    values = states.to(torch.float64)
    if values.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.float64),
            torch.empty((values.shape[1], 0), dtype=torch.float64),
            0,
        )
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / values.shape[0]
    vals, vecs = eigensystem(covariance)
    k = min(K, max(0, int(values.shape[0]) - 1))
    return vals, vecs[:, :k], k


def run(args: argparse.Namespace) -> None:
    cfg = common.family_config(args.family, args.scale)
    root = common.output_root(args, "exp2")
    q1 = args.q1_root
    selected_path = (
        args.exp1_output
        if args.exp1_output
        else Path("logs") / "q2" / args.family / "exp1"
    ) / "selected.json"
    if not selected_path.is_file() and args.scale != "tiny":
        raise SystemExit(f"Q2 exp1 selected.json missing: {selected_path}")
    selected = json.loads(selected_path.read_text()) if selected_path.is_file() else {}
    if (
        args.layer is not None
        and args.layer not in common.family_config(args.family, args.scale).layers
    ):
        raise SystemExit(f"layer must be one of configured layers: {args.layer}")
    if args.scale != "tiny":
        for domain in args.domains:
            layer = (
                args.layer
                if args.layer is not None
                else int(selected.get(domain, {"layer": 0})["layer"])
            )
            common.require_bases(q1, domain, (layer,), "rlvr")
    model, tokenizer = common.load_runtime(args, "rlvr")
    root.mkdir(parents=True, exist_ok=True)
    common.write_manifest(root, args, "exp2")
    summaries: dict[str, Any] = {}
    with tee_log(RunDir(root)):
        for domain in args.domains:
            choice = selected.get(domain, {"layer": 0, "alpha": 1.0})
            layer = args.layer if args.layer is not None else int(choice["layer"])
            if args.alpha is not None:
                choice["alpha"] = args.alpha
            bases = (
                common.require_bases(q1, domain, (layer,), "rlvr")
                if args.scale != "tiny"
                else (
                    common.tiny_bases(q1, [domain], (layer,), ("rlvr",))
                    or common.require_bases(q1, domain, (layer,), "rlvr")
                )
            )
            eig = bases[layer]
            k = cfg.d_model // 3
            high, low = eig[1][:, :k], eig[1][:, -k:]
            _, items = common.load_items(domain, args.limit, args.scale == "tiny")
            from postdyn import bench

            generations = bench.generate(
                model,
                tokenizer,
                items,
                chat_template=True,
                greedy=True,
                max_new_tokens=common.CAPS[BENCHMARKS[domain]][1],
                batch_size=args.batch_size,
                capture_layers=[layer],
            )
            path = root / f"solutions_{domain}.jsonl"
            done = (
                {json.loads(x)["item_id"] for x in path.read_text().splitlines()}
                if path.is_file()
                else set()
            )
            for item, generation in zip(items, generations):
                if item.id in done:
                    continue
                states = sentence_final_states(
                    tokenizer,
                    [],
                    generation.text,
                    generation.captured or {},
                    generation.prompt_token_len,
                )
                vals, vectors, k = solution_eigensystem(states, k)
                row = {
                    "item_id": item.id,
                    "correct": bool(
                        verify(BENCHMARKS[domain], generation.text, item.reference)
                    ),
                    "V_i": float(vals.sum().item()) if vals.numel() else 0.0,
                    "subsim_high": subsim(vectors, high[:, :k]) if k else 0.0,
                    "subsim_low": subsim(vectors, low[:, :k]) if k else 0.0,
                    "n_sentences": int(states.shape[0]),
                }
                common.append(path, row)
            rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            correct = [x for x in rows if x["correct"]]
            incorrect = [x for x in rows if not x["correct"]]
            mean = lambda group, key: (
                sum(x[key] for x in group) / len(group) if group else 0.0
            )
            summaries[domain] = {
                "n": len(rows),
                "mean_V_correct": mean(correct, "V_i"),
                "mean_V_incorrect": mean(incorrect, "V_i"),
                "mean_subsim_high": mean(rows, "subsim_high"),
                "mean_subsim_low": mean(rows, "subsim_low"),
            }
        atomic_write_json(root / "summary.json", summaries)


if __name__ == "__main__":
    run(parse_args())
