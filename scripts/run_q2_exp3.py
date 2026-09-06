from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
import scripts.q2_common as common
import scripts.run_q2_exp1 as exp1
from postdyn.config import BENCHMARKS
from postdyn.intervention import procrustes_align
from postdyn.persistence import RunDir, tee_log, atomic_write_json
from postdyn.spectra import subsim

# Slide spec: replace the SFT low-variance component by its Procrustes-aligned
# RLVR counterpart (h - U_S U_S^T h + U_R R* U_S^T h). "sft_only" keeps the
# removal half as the control isolating what the RLVR re-expression adds.
CONDITIONS = ("baseline", "own_only", "replace")
ALPHA = 1.0  # slide formula is applied exactly; strength is not searched


def select_layer(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the best layer at the fixed alpha (ties prefer smaller layer)."""
    best = max(rows, key=lambda row: (row["accuracy"], -int(row["layer"])))
    return {"layer": int(best["layer"]), "alpha": ALPHA}


SELECTION_CONDITION = "replace"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
    parser.add_argument(
        "--model",
        choices=("sft", "rlvr"),
        default="sft",
        help="fixed checkpoint receiving the other stage's aligned low basis",
    )
    return parser.parse_args(argv)


def identity_for(args: argparse.Namespace) -> dict[str, Any]:
    cfg = common.family_config(args.family, args.scale)
    return {
        "family": args.family,
        "q1_root": str(args.q1_root.resolve()),
        "sft_lr": args.sft_lr,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "device": args.device,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "model": args.model,
        "other": "rlvr" if args.model == "sft" else "sft",
        "checkpoints": common.checkpoint_pairs(
            args.family, ("sft", "rlvr"), args.sft_lr
        ),
        "domains": args.domains,
        "k": cfg.d_model // 3,
        "alpha": 1.0,
        "alpha_mode": "dimensionless",
        "scale": args.scale,
    }


def run_with(args: argparse.Namespace, load_runtime) -> None:
    cfg = common.family_config(args.family, args.scale)
    output = common.output_root(args, f"exp3_{args.model}")
    uploader = common.start_uploader(args, common.ROOT)
    output.mkdir(parents=True, exist_ok=True)
    common.write_identity_manifest(
        output,
        identity_for(args),
    )
    other = "rlvr" if args.model == "sft" else "sft"
    if args.scale == "tiny":
        common.tiny_bases(args.q1_root, args.domains, cfg.layers, (args.model, other))
    if args.scale != "tiny":
        for domain in args.domains:
            common.require_bases(args.q1_root, domain, cfg.layers, args.model)
            common.require_bases(args.q1_root, domain, cfg.layers, other)
    with tee_log(RunDir(output)):
        model, tokenizer = load_runtime()
        selected: dict[str, dict[str, Any]] = {}
        alignment: dict[str, dict[int, float]] = {}
        for domain in args.domains:
            benchmark = BENCHMARKS[domain]
            val, test = common.load_items(domain, args.limit, args.scale == "tiny")
            own = common.require_bases(args.q1_root, domain, cfg.layers, args.model)
            others = common.require_bases(args.q1_root, domain, cfg.layers, other)
            k = cfg.d_model // 3
            u_own = {l: own[l][1][:, -k:] for l in cfg.layers}
            u_other = {l: others[l][1][:, -k:] for l in cfg.layers}
            u_aligned = {
                l: u_other[l] @ procrustes_align(u_other[l], u_own[l])
                for l in cfg.layers
            }
            alignment[domain] = {l: subsim(u_own[l], u_other[l]) for l in cfg.layers}
            scores = []
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
                if all(
                    (domain, layer, ALPHA, SELECTION_CONDITION, item.id) in completed
                    for item in val
                ):
                    continue
                rows = exp1._evaluate(
                    model,
                    tokenizer,
                    val,
                    benchmark,
                    layer,
                    ALPHA,
                    None,
                    args.batch_size,
                    common.CAPS[benchmark][0],
                    SELECTION_CONDITION,
                    replacement=(u_own[layer], u_aligned[layer]),
                    done_ids={
                        key[4]
                        for key in completed
                        if key[:4] == (domain, layer, ALPHA, SELECTION_CONDITION)
                    },
                )
                key = (domain, layer, ALPHA, SELECTION_CONDITION)
                merged = {
                    **persisted_correct.get(key, {}),
                    **{row["item_id"]: bool(row["correct"]) for row in rows},
                }
                scores.append(
                    {
                        "layer": layer,
                        "alpha": ALPHA,
                        "accuracy": sum(merged.values()) / max(1, len(merged)),
                    }
                )
                for row in rows:
                    if (
                        domain,
                        layer,
                        ALPHA,
                        SELECTION_CONDITION,
                        row["item_id"],
                    ) not in completed:
                        common.append(
                            output / "validation.jsonl",
                            {
                                "domain": domain,
                                "layer": layer,
                                "alpha": ALPHA,
                                "condition": SELECTION_CONDITION,
                                "item_id": row["item_id"],
                                "correct": row["correct"],
                                "accuracy": scores[-1]["accuracy"],
                            },
                        )
            expected = len(cfg.layers)
            if len(scores) < expected:
                scores = common.validation_scores(
                    validation_path, domain, {SELECTION_CONDITION}
                )
            selected[domain] = select_layer(scores)
            choice = selected[domain]
            for condition in CONDITIONS:
                replacement = (
                    (u_own[choice["layer"]], u_aligned[choice["layer"]])
                    if condition == "replace"
                    else None
                )
                basis = u_own[choice["layer"]] if condition == "own_only" else None
                path = output / f"eval_{domain}_{condition}.jsonl"
                done = (
                    {
                        json.loads(line)["item_id"]
                        for line in path.read_text().splitlines()
                    }
                    if path.is_file()
                    else set()
                )
                rows = exp1._evaluate(
                    model,
                    tokenizer,
                    test,
                    benchmark,
                    choice["layer"],
                    choice["alpha"],
                    basis,
                    args.batch_size,
                    common.CAPS[benchmark][1],
                    condition,
                    replacement=replacement,
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
            output / "summary.json", {"selected": selected, "alignment": alignment}
        )
        common.finish_uploader(uploader, output)


def run(args: argparse.Namespace) -> None:
    run_with(args, lambda: common.load_runtime(args, args.model))


if __name__ == "__main__":
    run(parse_args())
