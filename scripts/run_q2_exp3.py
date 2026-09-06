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
from postdyn.config import ALPHAS, BENCHMARKS
from postdyn.intervention import procrustes_align
from postdyn.persistence import RunDir, tee_log, atomic_write_json
from postdyn.spectra import subsim

# Slide spec: replace the SFT low-variance component by its Procrustes-aligned
# RLVR counterpart (h - U_S U_S^T h + U_R R* U_S^T h). "sft_only" keeps the
# removal half as the control isolating what the RLVR re-expression adds.
CONDITIONS = ("baseline", "sft_only", "replace")
SELECTION_CONDITION = "replace"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
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
        "checkpoints": common.checkpoint_pairs(
            args.family, ("sft", "rlvr"), args.sft_lr
        ),
        "domains": args.domains,
        "k": cfg.d_model // 3,
        "alphas": list(ALPHAS),
        "alpha_mode": "dimensionless",
        "scale": args.scale,
    }


def run(args: argparse.Namespace) -> None:
    cfg = common.family_config(args.family, args.scale)
    output = common.output_root(args, "exp3")
    uploader = common.start_uploader(args, common.ROOT)
    output.mkdir(parents=True, exist_ok=True)
    common.write_identity_manifest(
        output,
        identity_for(args),
    )
    if args.scale == "tiny":
        common.tiny_bases(args.q1_root, args.domains, cfg.layers, ("sft", "rlvr"))
    if args.scale != "tiny":
        for domain in args.domains:
            common.require_bases(args.q1_root, domain, cfg.layers, "sft")
            common.require_bases(args.q1_root, domain, cfg.layers, "rlvr")
    with tee_log(RunDir(output)):
        model, tokenizer = common.load_runtime(args, "sft")
        selected: dict[str, dict[str, Any]] = {}
        alignment: dict[str, dict[int, float]] = {}
        for domain in args.domains:
            benchmark = BENCHMARKS[domain]
            val, test = common.load_items(domain, args.limit, args.scale == "tiny")
            sft = common.require_bases(args.q1_root, domain, cfg.layers, "sft")
            rlvr = common.require_bases(args.q1_root, domain, cfg.layers, "rlvr")
            k = cfg.d_model // 3
            u_s = {l: sft[l][1][:, -k:] for l in cfg.layers}
            u_r = {l: rlvr[l][1][:, -k:] for l in cfg.layers}
            u_aligned = {
                l: u_r[l] @ procrustes_align(u_r[l], u_s[l]) for l in cfg.layers
            }
            alignment[domain] = {l: subsim(u_s[l], u_r[l]) for l in cfg.layers}
            scores = []
            validation_path = output / "validation.jsonl"
            completed = common.completed_item_keys(validation_path)
            for layer in cfg.layers:
                for alpha in ALPHAS:
                    if all(
                        (domain, layer, alpha, SELECTION_CONDITION, item.id)
                        in completed
                        for item in val
                    ):
                        continue
                    rows = exp1._evaluate(
                        model,
                        tokenizer,
                        val,
                        benchmark,
                        layer,
                        alpha,
                        None,
                        args.batch_size,
                        common.CAPS[benchmark][0],
                        SELECTION_CONDITION,
                        replacement=(u_s[layer], u_aligned[layer]),
                    )
                    scores.append(
                        {
                            "layer": layer,
                            "alpha": alpha,
                            "accuracy": sum(1 for x in rows if bool(x["correct"]))
                            / max(1, len(rows)),
                        }
                    )
                    for row in rows:
                        if (
                            domain,
                            layer,
                            alpha,
                            SELECTION_CONDITION,
                            row["item_id"],
                        ) not in completed:
                            common.append(
                                output / "validation.jsonl",
                                {
                                    "domain": domain,
                                    "layer": layer,
                                    "alpha": alpha,
                                    "condition": SELECTION_CONDITION,
                                    "item_id": row["item_id"],
                                    "correct": row["correct"],
                                    "accuracy": scores[-1]["accuracy"],
                                },
                            )
            expected = len(cfg.layers) * len(ALPHAS)
            if len(scores) < expected:
                scores = common.validation_scores(
                    validation_path, domain, {SELECTION_CONDITION}
                )
            selected[domain] = exp1.select_best(scores)
            choice = selected[domain]
            for condition in CONDITIONS:
                replacement = (
                    (u_s[choice["layer"]], u_aligned[choice["layer"]])
                    if condition == "replace"
                    else None
                )
                basis = u_s[choice["layer"]] if condition == "sft_only" else None
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
                )
                path = output / f"eval_{domain}_{condition}.jsonl"
                done = (
                    {
                        json.loads(line)["item_id"]
                        for line in path.read_text().splitlines()
                    }
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
            output / "summary.json", {"selected": selected, "alignment": alignment}
        )
        common.finish_uploader(uploader, output)


if __name__ == "__main__":
    run(parse_args())
