from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import scripts.q2_common as common
import scripts.run_q2_exp1 as exp1
import scripts.run_q2_exp2 as exp2
import scripts.run_q2_exp3 as exp3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = common.add_common_args(argparse.ArgumentParser())
    parser.add_argument("--model", choices=("sft", "rlvr"), default="rlvr")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
    parser.add_argument(
        "--engine",
        choices=("hf", "vllm"),
        default="hf",
        help="generation backend forwarded to exp1/exp3 (exp2 stays hf)",
    )
    return parser.parse_args(argv)


def _shared_argv(args: argparse.Namespace) -> list[str]:
    values = [
        "--family",
        args.family,
        "--q1-root",
        str(args.q1_root),
        "--scale",
        args.scale,
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--sft-lr",
        args.sft_lr,
        "--batch-size",
        str(args.batch_size),
        "--domains",
        *args.domains,
    ]
    if args.quantization is not None:
        values.extend(("--quantization", args.quantization))
    if args.upload_to is not None:
        values.extend(("--upload-to", args.upload_to))
    if args.limit is not None:
        values.extend(("--limit", str(args.limit)))
    return values


def build_experiment_args(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, argparse.Namespace, argparse.Namespace]:
    root = args.output_root or Path("logs") / "q2" / args.family
    shared = _shared_argv(args)
    model = args.model
    engine_argv = (
        ["--engine", args.engine] if getattr(args, "engine", "hf") == "vllm" else []
    )
    a1 = exp1.parse_args(
        [
            *shared,
            "--model",
            model,
            "--output",
            str(root / f"exp1_{model}"),
            *engine_argv,
        ]
    )
    exp1_output = root / f"exp1_{model}"
    a2 = exp2.parse_args(
        [
            *shared,
            "--model",
            model,
            "--exp1-output",
            str(exp1_output),
            "--output",
            str(root / f"exp2_{model}"),
        ]
    )
    a3 = exp3.parse_args(
        [
            *shared,
            "--model",
            model,
            "--output",
            str(root / f"exp3_{model}"),
            *engine_argv,
        ]
    )
    return a1, a2, a3


def run(args: argparse.Namespace) -> None:
    a1, a2, a3 = build_experiment_args(args)
    if getattr(args, "engine", "hf") == "vllm":
        # exp1/exp3 build their own vLLM engines; only exp2 needs the HF
        # runtime, and VRAM must be returned between engine switches.
        def _release_vram() -> None:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        exp1.run_with(a1, lambda: (_ for _ in ()).throw(RuntimeError("unused")))
        _release_vram()
        exp2.run_with(a2, lambda: common.load_runtime(a2, args.model))
        _release_vram()
        exp3.run_with(a3, lambda: (_ for _ in ()).throw(RuntimeError("unused")))
        return
    runtime = common.load_runtime(a1, args.model)
    loader = lambda: runtime
    exp1.run_with(a1, loader)
    exp2.run_with(a2, loader)
    exp3.run_with(a3, loader)


if __name__ == "__main__":
    run(parse_args())
