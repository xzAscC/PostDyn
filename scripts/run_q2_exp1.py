from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.q2_common as common
from postdyn.config import ALPHAS, BENCHMARKS
from postdyn.intervention import (
    matched_random_basis,
    mean_hidden_norm,
    register_ablation_hook,
)
from postdyn.persistence import RunDir, tee_log, atomic_write_json
from postdyn.verifiers import verify

completed_keys = common.completed_keys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return common.add_common_args(argparse.ArgumentParser()).parse_args(argv)


def require_bases(
    root: Path,
    family: str,
    domain: str,
    layers: list[int] | tuple[int, ...],
    k: int,
    stage: str = "rlvr",
):
    del family, k
    return common.require_bases(root, domain, tuple(layers), stage)


def random_basis(d: int, k: int, domain: str, layer: int) -> torch.Tensor:
    seed = int(hashlib.sha256(f"42|{domain}|{layer}".encode()).hexdigest()[:8], 16)
    return matched_random_basis(d, k, seed)


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(
        rows,
        key=lambda row: (row["accuracy"], -int(row["layer"]), -float(row["alpha"])),
    )
    return {"layer": int(best["layer"]), "alpha": float(best["alpha"])}


def _evaluate(
    model: Any,
    tokenizer: Any,
    items: Any,
    benchmark: str,
    layer: int,
    alpha: float,
    basis: torch.Tensor | None,
    batch_size: int,
    max_new_tokens: int,
    condition: str,
) -> list[dict[str, Any]]:
    from postdyn import bench

    handle = (
        register_ablation_hook(model, layer, basis, alpha, "dimensionless")
        if basis is not None
        else None
    )
    try:
        generations = bench.generate(
            model,
            tokenizer,
            items,
            chat_template=True,
            greedy=True,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
        )
    finally:
        if handle is not None:
            handle.remove()
    return [
        {
            "item_id": g.item_id,
            "correct": bool(
                verify(
                    benchmark,
                    g.text,
                    next(x.reference for x in items if x.id == g.item_id),
                )
            ),
            "condition": condition,
        }
        for g in generations
    ]


def run(args: argparse.Namespace) -> None:
    cfg = common.family_config(args.family, args.scale)
    output = common.output_root(args, "exp1")
    q1_root = args.q1_root
    if args.scale == "tiny":
        common.tiny_bases(q1_root, args.domains, cfg.layers, ("rlvr",))
    output.mkdir(parents=True, exist_ok=True)
    run_dir = RunDir(output)
    common.write_manifest(output, args, "exp1")
    if args.scale != "tiny":
        for domain in args.domains:
            require_bases(q1_root, args.family, domain, cfg.layers, cfg.d_model // 3)
    with tee_log(run_dir):
        model, tokenizer = common.load_runtime(args, "rlvr")
        prior_selected = (
            json.loads((output / "selected.json").read_text())
            if (output / "selected.json").is_file()
            else {}
        )
        selected: dict[str, dict[str, Any]] = {}
        for domain in args.domains:
            benchmark = BENCHMARKS[domain]
            val, test = common.load_items(domain, args.limit, args.scale == "tiny")
            eig = require_bases(
                q1_root, args.family, domain, cfg.layers, cfg.d_model // 3
            )
            k = cfg.d_model // 3
            bases = {
                "high": {l: v[1][:, :k] for l, v in eig.items()},
                "low": {l: v[1][:, -k:] for l, v in eig.items()},
                "random": {
                    l: random_basis(cfg.d_model, k, domain, l) for l in cfg.layers
                },
            }
            val_summary: list[dict[str, Any]] = []
            completed = common.completed_item_keys(output / "validation.jsonl")
            for layer in cfg.layers:
                for alpha in ALPHAS:
                    for condition in ("high", "low", "random"):
                        key = (domain, layer, alpha, condition)
                        if all(
                            (domain, layer, alpha, condition, item.id) in completed
                            for item in val
                        ):
                            continue
                        rows = _evaluate(
                            model,
                            tokenizer,
                            val,
                            benchmark,
                            layer,
                            alpha,
                            bases[condition][layer],
                            args.batch_size,
                            common.CAPS[benchmark][0],
                            condition,
                        )
                        accuracy = sum(1 for row in rows if bool(row["correct"])) / max(
                            1, len(rows)
                        )
                        for row in rows:
                            if (
                                domain,
                                layer,
                                alpha,
                                condition,
                                row["item_id"],
                            ) not in completed:
                                common.append(
                                    output / "validation.jsonl",
                                    {
                                        "domain": domain,
                                        "layer": layer,
                                        "alpha": alpha,
                                        "condition": condition,
                                        "item_id": row["item_id"],
                                        "correct": row["correct"],
                                        "accuracy": accuracy,
                                    },
                                )
                        val_summary.append(
                            {
                                "domain": domain,
                                "layer": layer,
                                "alpha": alpha,
                                "condition": condition,
                                "accuracy": accuracy,
                            }
                        )
            existing = []
            validation_path = output / "validation.jsonl"
            if validation_path.is_file():
                grouped: dict[tuple[Any, ...], list[bool]] = {}
                for line in validation_path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("domain") == domain:
                        grouped.setdefault(
                            (row["layer"], row["alpha"], row["condition"]), []
                        ).append(bool(row["correct"]))
                existing = [
                    {
                        "layer": key[0],
                        "alpha": key[1],
                        "condition": key[2],
                        "accuracy": sum(values) / len(values),
                    }
                    for key, values in grouped.items()
                ]
            val_summary.extend(existing)
            choice = prior_selected.get(domain, select_best(val_summary))
            selected[domain] = choice
            if "r_bar" not in choice:
                choice["r_bar"] = mean_hidden_norm(
                    model,
                    tokenizer,
                    [x.prompt for x in val],
                    choice["layer"],
                    args.batch_size,
                )
            for condition in ("high", "low", "random"):
                rows = _evaluate(
                    model,
                    tokenizer,
                    test,
                    benchmark,
                    choice["layer"],
                    choice["alpha"],
                    bases[condition][choice["layer"]],
                    args.batch_size,
                    common.CAPS[benchmark][1],
                    condition,
                )
                path = output / f"eval_{domain}_{condition}.jsonl"
                done = (
                    {json.loads(x)["item_id"] for x in path.read_text().splitlines()}
                    if path.is_file()
                    else set()
                )
                for row in rows:
                    if row["item_id"] not in done:
                        common.append(
                            path,
                            {
                                **row,
                                "domain": domain,
                                "layer": choice["layer"],
                                "alpha": choice["alpha"],
                            },
                        )
        atomic_write_json(output / "selected.json", selected)
        atomic_write_json(
            output / "summary.json",
            {
                domain: {
                    "selected": selected[domain],
                    "n": len(
                        common.load_items(domain, args.limit, args.scale == "tiny")[1]
                    ),
                }
                for domain in args.domains
            },
        )


if __name__ == "__main__":
    run(parse_args())
