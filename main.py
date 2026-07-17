"""
PostDyn - Post-training dynamics analysis for open-source LLMs

Analyzes training dynamics across post-training methods (SFT, RL, etc.),
including effective rank and concept-direction trajectories on Pythia / OLMo-3.

Usage:
    python main.py --analysis all                  # Run all analyses
    python main.py --analysis cross-model-size     # Only cross-model comparison
    python main.py --analysis training-dynamics --model pythia-70m
    python main.py --dry-run                       # Validate configs
    python main.py --plot-only                     # Regenerate plots from saved results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from src.config import (
    PYTHIA_CONFIGS,
    PYTHIA_DEDUPED_CONFIGS,
    PYTHIA_CHECKPOINTS,
    PYTHIA_CHECKPOINTS_QUICK,
    OLMO3_VARIANTS,
    RESULTS_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="PostDyn - Post-training dynamics analysis for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--analysis",
        choices=[
            "all",
            "cross-model-size",
            "training-dynamics",
            "training-stages",
            "post-training-methods",
            "fixed-ratio",
            "activation-cross-model",
            "activation-training-dynamics",
            "activation-single",
            "activation-post-training",
            "activation-all",
        ],
        default="all",
        help="Which analysis to run (default: all)",
    )

    parser.add_argument(
        "--model",
        default="pythia-70m",
        help="Model name for training dynamics analysis (default: pythia-70m)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced checkpoint lists for faster runs",
    )

    parser.add_argument(
        "--max-dim",
        type=int,
        default=None,
        help="Maximum matrix dimension for SVD computation (skip larger)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs without downloading models",
    )

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate plots from existing results",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output directory (default: results/)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of MMLU Pro samples for activation analysis (default: all 12032)",
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for tokenization in activation analysis (default: 512)",
    )

    parser.add_argument(
        "--fit-range",
        default="10,100",
        help="Fit range for alpha-ReQ, comma-separated (default: 10,100)",
    )

    return parser.parse_args()


def dry_run_validation():
    """Validate all model configs without downloading."""
    print("=" * 60)
    print("DRY RUN - Validating configurations")
    print("=" * 60)

    print(f"\nPythia models: {len(PYTHIA_CONFIGS)}")
    for name, cfg in PYTHIA_CONFIGS.items():
        print(f"  {name}: {cfg.hf_id} (layers={cfg.layers}, d_model={cfg.d_model})")

    print(f"\nPythia deduped: {len(PYTHIA_DEDUPED_CONFIGS)}")

    checkpoints = PYTHIA_CHECKPOINTS_QUICK if True else PYTHIA_CHECKPOINTS
    print(f"\nCheckpoints (quick): {len(checkpoints)}")
    print(f"  From {checkpoints[0]} to {checkpoints[-1]}")

    print(f"\nOLMo-3 variants: {len(OLMO3_VARIANTS)}")
    for name, cfg in OLMO3_VARIANTS.items():
        print(f"  {name}: {cfg.hf_id} (pathway={cfg.pathway}, stage={cfg.stage})")

    print("\n✓ All configs validated successfully")
    return True


def main():
    args = parse_args()

    if args.output_dir:
        import src.config as cfg

        cfg.RESULTS_DIR = args.output_dir
        cfg.FIGURES_DIR = os.path.join(args.output_dir, "figures")

    if args.dry_run:
        return dry_run_validation()

    if args.plot_only:
        from src.visualization import generate_all_plots

        generate_all_plots()
        return

    from src.analysis import (
        analyze_cross_model_size,
        analyze_training_dynamics,
        analyze_training_stages,
        analyze_post_training_methods,
        analyze_fixed_ratio_hypothesis,
        run_all_analyses,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    analysis = args.analysis
    results = {}

    if analysis == "all":
        results = run_all_analyses(quick=args.quick, max_dim=args.max_dim)
    elif analysis == "cross-model-size":
        results = analyze_cross_model_size(max_dim=args.max_dim)
    elif analysis == "training-dynamics":
        ckpts = PYTHIA_CHECKPOINTS_QUICK if args.quick else PYTHIA_CHECKPOINTS
        results = analyze_training_dynamics(
            args.model, checkpoints=ckpts, max_dim=args.max_dim
        )
    elif analysis == "training-stages":
        results = analyze_training_stages(max_dim=args.max_dim)
    elif analysis == "post-training-methods":
        results = analyze_post_training_methods(max_dim=args.max_dim)
    elif analysis == "fixed-ratio":
        results = analyze_fixed_ratio_hypothesis()

    elif analysis in (
        "activation-cross-model",
        "activation-training-dynamics",
        "activation-single",
        "activation-post-training",
        "activation-all",
    ):
        from src.activation_analysis import (
            load_mmlu_questions,
            extract_and_analyze_activations,
            analyze_activation_cross_model,
            analyze_activation_training_dynamics,
            analyze_activation_post_training,
        )

        questions = load_mmlu_questions(num_samples=args.num_samples)
        fit_range = tuple(int(x) for x in args.fit_range.split(","))

        if analysis == "activation-cross-model" or analysis == "activation-all":
            results["activation_cross_model"] = analyze_activation_cross_model(
                PYTHIA_CONFIGS,
                num_samples=args.num_samples,
                max_seq_len=args.max_seq_len,
                fit_range=fit_range,
            )

        if analysis == "activation-training-dynamics" or analysis == "activation-all":
            ckpts = PYTHIA_CHECKPOINTS_QUICK if args.quick else PYTHIA_CHECKPOINTS
            results[f"activation_training_dynamics_{args.model}"] = (
                analyze_activation_training_dynamics(
                    args.model,
                    checkpoints=ckpts,
                    num_samples=args.num_samples,
                    max_seq_len=args.max_seq_len,
                    fit_range=fit_range,
                )
            )

        if analysis == "activation-single":
            if args.model not in PYTHIA_CONFIGS:
                print(
                    f"Model {args.model} not found. Available: {list(PYTHIA_CONFIGS.keys())}"
                )
                return
            single_result = extract_and_analyze_activations(
                PYTHIA_CONFIGS[args.model],
                questions,
                max_seq_len=args.max_seq_len,
                fit_range=fit_range,
            )
            results["activation_single"] = {
                "analysis": "activation_single",
                "model": single_result.model_name,
                "layer_results": {
                    str(k): v for k, v in single_result.layer_results.items()
                },
                "num_questions": single_result.num_questions,
                "elapsed_seconds": single_result.elapsed_seconds,
            }

        if analysis == "activation-post-training" or analysis == "activation-all":
            results["activation_post_training"] = analyze_activation_post_training(
                num_samples=args.num_samples,
                max_seq_len=args.max_seq_len,
                fit_range=fit_range,
            )

    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    from src.visualization import (
        plot_cross_model_size,
        plot_training_dynamics,
        plot_training_stages,
        plot_post_training_methods,
        plot_fixed_ratio_distribution,
    )

    if analysis in ("all", "cross-model-size"):
        if "models" in results or "cross_model_size" in results:
            data = results.get("cross_model_size", results)
            if isinstance(data, dict) and "models" in data:
                plot_cross_model_size(data=data)

    if analysis in ("all", "training-dynamics"):
        data = results.get("training_dynamics", results)
        if isinstance(data, dict) and "results" in data:
            plot_training_dynamics(data=data)

    if analysis in ("all", "training-stages"):
        data = results.get("training_stages", results)
        if isinstance(data, dict) and "results" in data:
            plot_training_stages(data=data)

    if analysis in ("all", "post-training-methods"):
        data = results.get("post_training_methods", results)
        if isinstance(data, dict) and "variants" in data:
            plot_post_training_methods(data=data)

    if analysis in ("all", "fixed-ratio"):
        data = results.get("fixed_ratio_hypothesis", results)
        if isinstance(data, dict) and "overall_stats" in data:
            plot_fixed_ratio_distribution(data=data)

    if analysis in ("activation-cross-model", "activation-all"):
        from src.visualization import plot_activation_analysis

        acm_data = results.get("activation_cross_model", {})
        if isinstance(acm_data, dict) and "models" in acm_data:
            plot_activation_analysis(data=acm_data)

    if analysis in ("activation-training-dynamics", "activation-all"):
        from src.visualization import plot_activation_analysis

        for key, val in results.items():
            if key.startswith("activation_training_dynamics") and isinstance(val, dict):
                plot_activation_analysis(data=val)

    if analysis == "activation-single":
        from src.visualization import plot_activation_analysis

        single_data = results.get("activation_single", {})
        if isinstance(single_data, dict) and "layer_results" in single_data:
            plot_activation_analysis(data=single_data)

    if analysis in ("activation-post-training", "activation-all"):
        from src.visualization import plot_activation_analysis

        apt_data = results.get("activation_post_training", {})
        if isinstance(apt_data, dict) and "variants" in apt_data:
            plot_activation_analysis(data=apt_data)

    print("\n" + "=" * 60)
    print("DONE! Results saved to:", RESULTS_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
