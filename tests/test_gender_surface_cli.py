from __future__ import annotations

import importlib.util
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_CLI_PATH = _ROOT / "experiments" / "analyze_gender_surface_control.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("_gender_surface_cli", _CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load gender surface CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_target_paired_vectors_and_final_math_checkpoint():
    cli = _load_cli()
    args = cli.parse_args([])
    assert args.model == "olmo3-rl-zero-math"
    assert args.checkpoint == "step_1900"
    assert args.vectors_dir == "results/concept_dynamics_paired/vectors"
    assert args.output.startswith("results/concept_dynamics_paired/")
