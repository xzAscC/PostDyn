"""Measure Q1 eigensystem stability across independent prompt subsets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from postdyn.config import MODEL_FAMILIES, ROBUSTNESS_DOMAIN
from postdyn.data import DomainPool, load_pool
from postdyn.extract import OnlineCovariance, extract_layer_hiddens
from postdyn.persistence import (
    atomic_write_json,
    load_eigensystem,
    save_eigensystem,
    tee_log,
)
from postdyn.spectra import (
    eigensystem,
    effective_rank,
    match_eigenvectors,
    rank_displacement,
    subsim,
)

import scripts.run_q1 as q1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(MODEL_FAMILIES), required=True)
    parser.add_argument("--checkpoint", default="rlvr")
    parser.add_argument("--domain", default=ROBUSTNESS_DOMAIN)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--scale", choices=("tiny", "smoke", "full"), default="full")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--attention-budget", type=int, default=8_388_608)
    parser.add_argument("--allow-short-pool", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.domain not in q1.DOMAINS:
        parser.error(f"unknown domain: {args.domain}")
    return args


def _pool(args: argparse.Namespace, n: int) -> DomainPool:
    if args.scale == "tiny":
        return q1._tiny_pool(args.domain, max(n + 1, 2 * n))
    path = ROOT / "data" / "domain_prompts" / f"{args.domain}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing domain pool: {path}")
    return load_pool(path)


def _checkpoint(args: argparse.Namespace):
    family = MODEL_FAMILIES[args.family]
    refs = {ref.name: ref for ref in family.checkpoints(args.sft_lr)}
    if args.checkpoint not in refs:
        raise ValueError(f"unknown checkpoint: {args.checkpoint}")
    return refs[args.checkpoint], q1._checkpoint_model(args, refs[args.checkpoint])


def run(args: argparse.Namespace) -> int:
    family = MODEL_FAMILIES[args.family]
    layers = [0, 1] if args.scale == "tiny" else list(family.layers)
    requested_n = args.limit or (32 if args.scale == "tiny" else family.n_samples())
    pool = _pool(args, requested_n)
    n = min(requested_n, pool.actual_n)
    if n <= 0:
        raise ValueError("selected domain pool contains no records")
    if args.repeats > 1 and pool.actual_n <= n and not args.allow_short_pool:
        raise SystemExit(
            "robustness pools must exceed 3d for genuine resampling; "
            "rematerialize with the runbook's 2x pool size"
        )
    checkpoint, (model, tokenizer) = _checkpoint(args)
    output = Path(args.output) if args.output else ROOT / "logs" / "q1_robustness"
    root = output / args.family / checkpoint.name / args.domain
    root.mkdir(parents=True, exist_ok=True)
    with tee_log(root):
        first: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        selected_ids: list[set[str]] = []
        for repeat in range(args.repeats):
            path = root / f"repeat_{repeat}.json"
            if path.is_file():
                payload = json.loads(path.read_text())
                selected_ids.append(set(payload.get("record_ids", [])))
                for layer in layers:
                    base = root / "eigensystems" / str(repeat) / str(layer)
                    if base.with_suffix(".json").is_file():
                        first.setdefault(layer, load_eigensystem(base))
                continue
            shuffled = list(pool.records)
            random.Random(42 + repeat).shuffle(shuffled)
            chosen = shuffled[:n]
            chosen_ids = {record.id for record in chosen}
            if chosen_ids in selected_ids:
                raise SystemExit(
                    "robustness resampling produced an identical subset; "
                    "pools must contain more than 3d records"
                )
            selected_ids.append(chosen_ids)
            records = [record.prompt for record in chosen]
            hidden_by_layer = extract_layer_hiddens(
                model,
                tokenizer,
                records,
                layers,
                args.batch_size,
                args.max_length,
                token_budget=args.token_budget,
                attention_budget=args.attention_budget,
            )
            values_by_layer: dict[str, list[float]] = {}
            ranks: dict[str, float] = {}
            stability: dict[str, dict[str, float]] = {}
            for layer in layers:
                hidden = hidden_by_layer[layer]
                covariance = OnlineCovariance()
                covariance.update(hidden)
                del hidden
                values, vectors = eigensystem(covariance.covariance)
                base = root / "eigensystems" / str(repeat) / str(layer)
                save_eigensystem(base, values, vectors)
                values_by_layer[str(layer)] = [float(value) for value in values]
                ranks[str(layer)] = effective_rank(values)
                if repeat == 0:
                    first[layer] = (values, vectors)
                else:
                    first_values, first_vectors = first[layer]
                    displacement = (
                        rank_displacement(match_eigenvectors(first_vectors, vectors))
                        .float()
                        .tolist()
                    )
                    stability[str(layer)] = {
                        "mean": mean(displacement),
                        "p90": sorted(displacement)[int(0.9 * (len(displacement) - 1))],
                        "subsim_high_vs_first": subsim(
                            first_vectors[:, q1.band_slices(vectors.shape[0])[0]],
                            vectors[:, q1.band_slices(vectors.shape[0])[0]],
                        ),
                        "subsim_low_vs_first": subsim(
                            first_vectors[:, q1.band_slices(vectors.shape[0])[2]],
                            vectors[:, q1.band_slices(vectors.shape[0])[2]],
                        ),
                    }
                    del first_values, first_vectors, values, vectors
            del hidden_by_layer
            payload = {
                "repeat": repeat,
                "n": n,
                "record_ids": [record.id for record in chosen],
                "eigenvalues": values_by_layer,
                "effective_rank": ranks,
                "subsim_high_vs_first": {
                    key: value.get("subsim_high_vs_first", 1.0)
                    for key, value in stability.items()
                },
                "subsim_low_vs_first": {
                    key: value.get("subsim_low_vs_first", 1.0)
                    for key, value in stability.items()
                },
                "rank_stability": stability,
            }
            atomic_write_json(path, payload)
            print(f"repeat={repeat} checkpoint={checkpoint.name} domain={args.domain}")
        repeat_payloads = [
            json.loads((root / f"repeat_{r}.json").read_text())
            for r in range(args.repeats)
        ]
        spreads = {
            str(layer): pstdev(
                [
                    float(payload["effective_rank"][str(layer)])
                    for payload in repeat_payloads
                ]
            )
            for layer in layers
        }
        pair_values = []
        low_pair_values = []
        for left in range(args.repeats):
            for right in range(left + 1, args.repeats):
                for layer in layers:
                    left_vectors = load_eigensystem(
                        root / "eigensystems" / str(left) / str(layer)
                    )[1]
                    right_vectors = load_eigensystem(
                        root / "eigensystems" / str(right) / str(layer)
                    )[1]
                    band = q1.band_slices(left_vectors.shape[0])[0]
                    low_band = q1.band_slices(left_vectors.shape[0])[2]
                    pair_values.append(
                        subsim(left_vectors[:, band], right_vectors[:, band])
                    )
                    low_pair_values.append(
                        subsim(left_vectors[:, low_band], right_vectors[:, low_band])
                    )
        atomic_write_json(
            root / "summary.json",
            {
                "repeats": args.repeats,
                "n": n,
                "layers": layers,
                "spread": {
                    "std_effective_rank": spreads,
                    "mean_subsim_high_across_pairs": mean(pair_values)
                    if pair_values
                    else 0.0,
                    "mean_subsim_low_across_pairs": mean(low_pair_values)
                    if low_pair_values
                    else 0.0,
                },
            },
        )
    if args.scale != "tiny":
        q1.release_model(model)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
