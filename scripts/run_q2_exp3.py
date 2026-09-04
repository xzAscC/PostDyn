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
from postdyn.persistence import RunDir, tee_log, atomic_write_json

CONDITIONS = ("baseline", "sft_low", "rlvr_low")


def basis_stage(condition: str) -> str:
    if condition == "sft_low":
        return "sft"
    if condition == "rlvr_low":
        return "rlvr"
    raise ValueError(f"baseline has no basis stage: {condition}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    cfg = common.family_config(args.family, args.scale)
    output = common.output_root(args, "exp3")
    if args.scale == "tiny":
        common.tiny_bases(args.q1_root, args.domains, cfg.layers, ("sft", "rlvr"))
    output.mkdir(parents=True, exist_ok=True)
    common.write_manifest(output, args, "exp3")
    if args.scale != "tiny":
        for domain in args.domains:
            common.require_bases(args.q1_root, domain, cfg.layers, "sft")
            common.require_bases(args.q1_root, domain, cfg.layers, "rlvr")
    with tee_log(RunDir(output)):
        model, tokenizer = common.load_runtime(args, "sft")
        selected: dict[str, dict[str, Any]] = {}
        for domain in args.domains:
            benchmark = BENCHMARKS[domain]
            val, test = common.load_items(domain, args.limit, args.scale == "tiny")
            sft = common.require_bases(args.q1_root, domain, cfg.layers, "sft")
            rlvr = common.require_bases(args.q1_root, domain, cfg.layers, "rlvr")
            low = {
                "sft_low": {l: sft[l][1][:, -(cfg.d_model // 3) :] for l in cfg.layers},
                "rlvr_low": {
                    l: rlvr[l][1][:, -(cfg.d_model // 3) :] for l in cfg.layers
                },
            }
            scores = []
            validation_path = output / "validation.jsonl"
            completed = common.completed_item_keys(validation_path)
            for layer in cfg.layers:
                for alpha in ALPHAS:
                    if all(
                        (domain, layer, alpha, "sft_low", item.id) in completed
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
                        low["sft_low"][layer],
                        args.batch_size,
                        common.CAPS[benchmark][0],
                        "sft_low",
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
                            "sft_low",
                            row["item_id"],
                        ) not in completed:
                            common.append(
                                output / "validation.jsonl",
                                {
                                    "domain": domain,
                                    "layer": layer,
                                    "alpha": alpha,
                                    "condition": "sft_low",
                                    "item_id": row["item_id"],
                                    "correct": row["correct"],
                                    "accuracy": scores[-1]["accuracy"],
                                },
                            )
            expected = len(cfg.layers) * len(ALPHAS)
            if len(scores) < expected:
                scores = common.validation_scores(validation_path, domain, {"sft_low"})
            selected[domain] = exp1.select_best(scores)
            choice = selected[domain]
            for condition in CONDITIONS:
                basis = (
                    None if condition == "baseline" else low[condition][choice["layer"]]
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
        atomic_write_json(output / "summary.json", selected)


if __name__ == "__main__":
    run(parse_args())
