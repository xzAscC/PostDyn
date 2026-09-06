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
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument(
        "--model",
        choices=("rlvr", "sft"),
        default="rlvr",
        help="checkpoint to intervene on (covariance bases match it)",
    )
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
    return parser.parse_args(argv)


def require_bases(
    root: Path,
    domain: str,
    layers: list[int] | tuple[int, ...],
    stage: str = "rlvr",
):
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
    replacement: tuple[torch.Tensor, torch.Tensor] | None = None,
    done_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    from postdyn import bench
    from postdyn.intervention import register_replacement_hook

    references: dict[str, Any] = {}
    for item in items:
        references.setdefault(item.id, item.reference)
    generations = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        if done_ids is not None and all(item.id in done_ids for item in batch):
            continue
        if replacement is not None:
            handle: Any = register_replacement_hook(
                model, layer, replacement[0], replacement[1], alpha
            )
        else:
            handle = (
                register_ablation_hook(model, layer, basis, alpha, "dimensionless")
                if basis is not None
                else None
            )
        try:
            generations.extend(
                bench.generate(
                    model,
                    tokenizer,
                    batch,
                    chat_template=True,
                    greedy=True,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                )
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
                    references[g.item_id],
                )
            ),
            "condition": condition,
        }
        for g in generations
    ]


def identity_for(args: argparse.Namespace) -> dict[str, Any]:
    cfg = common.family_config(args.family, args.scale)
    return {
        "family": args.family,
        "q1_root": str(args.q1_root.resolve()),
        "sft_lr": getattr(args, "sft_lr", "1e-4"),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "device": args.device,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "model": args.model,
        "checkpoints": common.checkpoint_pairs(
            args.family, (args.model,), getattr(args, "sft_lr", "1e-4")
        ),
        "domains": args.domains,
        "k": cfg.d_model // 3,
        "alphas": list(ALPHAS),
        "alpha_mode": "dimensionless",
        "scale": args.scale,
    }


def run_with(args: argparse.Namespace, load_runtime) -> None:
    cfg = common.family_config(args.family, args.scale)
    output = common.output_root(args, f"exp1_{args.model}")
    q1_root = args.q1_root
    output.mkdir(parents=True, exist_ok=True)
    common.write_identity_manifest(
        output,
        identity_for(args),
    )
    uploader = common.start_uploader(args, ROOT)
    if args.scale == "tiny":
        common.tiny_bases(q1_root, args.domains, cfg.layers, (args.model,))
    run_dir = RunDir(output)
    if args.scale != "tiny":
        for domain in args.domains:
            require_bases(q1_root, domain, cfg.layers, args.model)
    with tee_log(run_dir):
        model, tokenizer = load_runtime()
        prior_selected = (
            json.loads((output / "selected.json").read_text())
            if (output / "selected.json").is_file()
            else {}
        )
        selected: dict[str, dict[str, Any]] = {}
        test_lengths: dict[str, int] = {}
        for domain in args.domains:
            benchmark = BENCHMARKS[domain]
            val, test = common.load_items(domain, args.limit, args.scale == "tiny")
            test_lengths[domain] = len(test)
            eig = require_bases(q1_root, domain, cfg.layers, args.model)
            k = cfg.d_model // 3
            bases = {
                "high": {l: v[1][:, :k] for l, v in eig.items()},
                "low": {l: v[1][:, -k:] for l, v in eig.items()},
                "random": {
                    l: random_basis(cfg.d_model, k, domain, l) for l in cfg.layers
                },
            }
            val_summary: list[dict[str, Any]] = []
            validation_path = output / "validation.jsonl"
            completed = common.completed_item_keys(validation_path)
            persisted_correct: dict[tuple[Any, ...], dict[str, bool]] = {}
            if validation_path.is_file():
                for line in validation_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("domain") == domain and "correct" in row:
                        config = (
                            row["domain"],
                            row["layer"],
                            row["alpha"],
                            row["condition"],
                        )
                        persisted_correct.setdefault(config, {})[row["item_id"]] = bool(
                            row["correct"]
                        )
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
                            done_ids={
                                key[4]
                                for key in completed
                                if key[:4] == (domain, layer, alpha, condition)
                            },
                        )
                        done_correct = persisted_correct.get(key, {})
                        merged = {
                            **done_correct,
                            **{row["item_id"]: bool(row["correct"]) for row in rows},
                        }
                        accuracy = sum(merged.values()) / max(1, len(merged))
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
            expected = len(cfg.layers) * len(ALPHAS) * 3
            if len(val_summary) < expected:
                val_summary = common.validation_scores(validation_path, domain)
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
                path = output / f"eval_{domain}_{condition}.jsonl"
                done = (
                    {json.loads(x)["item_id"] for x in path.read_text().splitlines()}
                    if path.is_file()
                    else set()
                )
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
                    done_ids=done,
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
                    "n": test_lengths[domain],
                }
                for domain in args.domains
            },
        )
        common.finish_uploader(uploader, output)


def run(args: argparse.Namespace) -> None:
    run_with(args, lambda: common.load_runtime(args, args.model))


if __name__ == "__main__":
    run(parse_args())
