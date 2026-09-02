"""Think-SFT vs Think differential-subspace experiment configuration.

Same math-vs-text protocol as the RL-Zero-Math analysis:
  * Dolci Math vs Dolci General, 10 x d_model prompts / domain
    (full-rank covariance; avoids the mass of zero eigenvalues that a
    sample-starved Gram matrix produces)
  * last prompt token (raw, no chat template)
  * 10 slide-formula layers
  * both positive (math-dominant) and negative (text-dominant) eigenspaces

Models (7B first; 32B is configured but not the default run):
  * allenai/Olmo-3-7B-Think-SFT
  * allenai/Olmo-3-7B-Think
  * allenai/Olmo-3-32B-Think-SFT
  * allenai/Olmo-3-32B-Think
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from postdyn.config import (
    EXPERIMENT_LAYERS_7B,
    EXPERIMENT_LAYERS_32B,
    MODEL_CHECKPOINTS,
    OLMO3_VARIANTS,
    THINK_7B_RLVR_CHECKPOINTS,
    THINK_7B_RLVR_REVISIONS,
    THINK_32B_RLVR_CHECKPOINTS,
    THINK_32B_RLVR_REVISIONS,
    THINK_32B_SFT_CHECKPOINTS,
    THINK_32B_SFT_REVISIONS,
    ModelConfig,
)
from postdyn.dataset_store import PROJECT_ROOT, SHARED_SAMPLE_SEED
from postdyn.differential_subspace import DEFAULT_TAU
from postdyn.domain_datasets import DEFAULT_CONCEPT_PAIRS

SCALE_7B: str = "7b"
SCALE_32B: str = "32b"

MODEL_KEYS_BY_SCALE: dict[str, tuple[str, str]] = {
    SCALE_7B: ("olmo3-think-sft", "olmo3-think-rlvr"),
    SCALE_32B: ("olmo3-32b-think-sft", "olmo3-32b-think-rlvr"),
}

COVARIANCE_SAMPLES_PER_DIM: int = 10
SAMPLE_SEED: int = SHARED_SAMPLE_SEED
TAU: float = DEFAULT_TAU
MAX_SEQ_LEN: int = 2048
USE_CHAT_TEMPLATE: bool = False
EXTRACTION_CONTRACT: str = "raw_prompt_final_attention_token_v1"
DTYPE: str = "bfloat16"


def covariance_n_samples(scale: str) -> int:
    """Prompts per domain: 10 x d_model, so covariance can reach full rank."""
    keys = MODEL_KEYS_BY_SCALE.get(scale)
    if keys is None:
        raise ValueError(f"unknown scale: {scale!r}")
    return COVARIANCE_SAMPLES_PER_DIM * OLMO3_VARIANTS[keys[0]].d_model


def extraction_protocol_payload(
    *,
    n_samples: int,
    tau: float,
    max_seq_len: int,
    use_chat_template: bool,
    extraction_contract: str,
    dtype: str,
    signed: bool,
) -> dict[str, object]:
    """Return the complete extraction protocol as a stable JSON payload."""
    return {
        "n_samples": n_samples,
        "tau": tau,
        "max_seq_len": max_seq_len,
        "use_chat_template": use_chat_template,
        "extraction_contract": extraction_contract,
        "dtype": dtype,
        "signed": signed,
    }


def canonical_extraction_protocol(scale: str) -> dict[str, object]:
    return extraction_protocol_payload(
        n_samples=covariance_n_samples(scale),
        tau=TAU,
        max_seq_len=MAX_SEQ_LEN,
        use_chat_template=USE_CHAT_TEMPLATE,
        extraction_contract=EXTRACTION_CONTRACT,
        dtype=DTYPE,
        signed=True,
    )


def validate_extraction_protocol(
    protocol: object, *, canonical: bool = True, scale: str = SCALE_7B
) -> None:
    """Reject missing, mistyped, or noncanonical extraction protocol fields."""
    if not isinstance(protocol, dict):
        raise ValueError("extraction protocol must be an object")
    expected = canonical_extraction_protocol(scale)
    if not canonical:
        return
    if set(protocol) != set(expected):
        raise ValueError(
            f"canonical extraction protocol fields must be exactly {sorted(expected)}"
        )
    for key, expected_value in expected.items():
        actual = protocol.get(key)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(
                f"canonical extraction protocol {key}={actual!r}, "
                f"expected {expected_value!r}"
            )


def extraction_protocols_equal(actual: object, expected: dict[str, object]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    return all(
        type(actual[key]) is type(value) and actual[key] == value
        for key, value in expected.items()
    )


FAMILY_THINK: str = "think"
TRAJECTORY_SFT: str = "sft"
TRAJECTORY_RLVR: str = "rlvr"
TRAJECTORY_SFT_LR_1E4: str = "sft_lr_1e-4"
TRAJECTORY_SFT_LR_5E5: str = "sft_lr_5e-5"

CONCEPT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("math_vs_wikitext", "math", "wikitext"),
    ("code_vs_wikitext", "code", "wikitext"),
    ("instruction_following_vs_wikitext", "instruction_following", "wikitext"),
    ("general_reasoning_vs_wikitext", "general_reasoning", "wikitext"),
    ("math_vs_code", "math", "code"),
    ("math_vs_instruction_following", "math", "instruction_following"),
    ("math_vs_general_reasoning", "math", "general_reasoning"),
)

FIXED_POINT_MODEL_KEYS_BY_SCALE: dict[str, dict[str, str]] = {
    SCALE_7B: {
        "base": "olmo3-base",
        "dpo": "olmo3-think-dpo",
    },
    SCALE_32B: {},
}

RESULTS_ROOT: Path = PROJECT_ROOT / "logs" / "think_sft_differential_subspace"
RESULTS_ROOT_QUICK: Path = (
    PROJECT_ROOT / "logs" / "think_sft_differential_subspace_quick"
)

U_SUBDIR: str = "U"
METRICS_SUBDIR: str = "metrics"
MANIFESTS_SUBDIR: str = "manifests"
PROMPTS_SUBDIR: str = "prompts"
FIGURES_SUBDIR: str = "figures"


def layers_for_scale(scale: str) -> list[int]:
    if scale == SCALE_7B:
        return list(EXPERIMENT_LAYERS_7B)
    if scale == SCALE_32B:
        return list(EXPERIMENT_LAYERS_32B)
    raise ValueError(f"Unknown scale {scale!r}; expected {SCALE_7B} or {SCALE_32B}")


def model_keys_for_scale(scale: str) -> tuple[str, str]:
    try:
        return MODEL_KEYS_BY_SCALE[scale]
    except KeyError as exc:
        raise ValueError(f"Unknown scale {scale!r}") from exc


def fixed_point_configs(scale: str) -> dict[str, tuple[str, str]]:
    try:
        return {
            label: (model_key, model_config(model_key).revision)
            for label, model_key in FIXED_POINT_MODEL_KEYS_BY_SCALE[scale].items()
        }
    except KeyError as exc:
        raise ValueError(f"Unknown scale {scale!r}") from exc


def model_config(model_key: str) -> ModelConfig:
    try:
        return OLMO3_VARIANTS[model_key]
    except KeyError as exc:
        raise ValueError(f"Unknown model key {model_key!r}") from exc


def sft_model_key(scale: str) -> str:
    return model_keys_for_scale(scale)[0]


def think_model_key(scale: str) -> str:
    return model_keys_for_scale(scale)[1]


def checkpoints_for_scale(scale: str) -> list[str]:
    return checkpoints_for_trajectory(FAMILY_THINK, scale, TRAJECTORY_SFT)


@dataclass(frozen=True)
class TrajectoryConfig:
    family: str
    scale: str
    trajectory: str
    model_key: str
    checkpoints: tuple[str, ...]
    revisions: dict[str, str]
    root_name: str


def trajectory_config(
    family: str = FAMILY_THINK,
    scale: str = SCALE_7B,
    trajectory: str = TRAJECTORY_SFT,
) -> TrajectoryConfig:
    if family != FAMILY_THINK:
        raise ValueError(f"Unknown family {family!r}")
    if scale not in MODEL_KEYS_BY_SCALE:
        raise ValueError(f"Unknown scale {scale!r}")
    if trajectory == TRAJECTORY_SFT:
        if scale == SCALE_7B:
            checkpoints = tuple(MODEL_CHECKPOINTS["olmo3-think-sft"])
            revisions = {checkpoint: checkpoint for checkpoint in checkpoints}
            root_name = "think_sft_differential_subspace"
        else:
            raise ValueError(
                "No checkpoint schedule: 32b SFT requires an explicit learning-rate trajectory"
            )
        model_key = sft_model_key(scale)
    elif trajectory == TRAJECTORY_RLVR:
        model_key = think_model_key(scale)
        if scale == SCALE_7B:
            checkpoints, revisions = THINK_7B_RLVR_CHECKPOINTS, THINK_7B_RLVR_REVISIONS
        else:
            checkpoints, revisions = (
                THINK_32B_RLVR_CHECKPOINTS,
                THINK_32B_RLVR_REVISIONS,
            )
        root_name = f"think_{scale}_rlvr_differential_subspace"
    elif scale == SCALE_32B and trajectory in {
        TRAJECTORY_SFT_LR_1E4,
        TRAJECTORY_SFT_LR_5E5,
    }:
        lr = trajectory.removeprefix("sft_lr_")
        model_key = sft_model_key(scale)
        checkpoints = THINK_32B_SFT_CHECKPOINTS
        revisions = THINK_32B_SFT_REVISIONS[lr]
        root_name = f"think_32b_{trajectory}_differential_subspace"
    else:
        raise ValueError(f"Unsupported trajectory {trajectory!r} for scale {scale!r}")
    if len(checkpoints) != 10 or set(checkpoints) != set(revisions):
        raise ValueError(f"Trajectory {trajectory!r} must have ten pinned checkpoints")
    return TrajectoryConfig(
        family, scale, trajectory, model_key, checkpoints, revisions, root_name
    )


def checkpoints_for_trajectory(family: str, scale: str, trajectory: str) -> list[str]:
    return list(trajectory_config(family, scale, trajectory).checkpoints)


def revision_for_checkpoint(
    family: str, scale: str, trajectory: str, checkpoint: str
) -> str:
    config = trajectory_config(family, scale, trajectory)
    try:
        return config.revisions[checkpoint]
    except KeyError as exc:
        raise ValueError(
            f"Unknown checkpoint {checkpoint!r} for {trajectory!r}"
        ) from exc


def available_trajectories(
    family: str = FAMILY_THINK, scale: str = SCALE_7B
) -> tuple[str, ...]:
    if family != FAMILY_THINK:
        raise ValueError(f"Unknown family {family!r}")
    if scale == SCALE_7B:
        return (TRAJECTORY_SFT, TRAJECTORY_RLVR)
    if scale == SCALE_32B:
        return (TRAJECTORY_SFT_LR_1E4, TRAJECTORY_SFT_LR_5E5, TRAJECTORY_RLVR)
    raise ValueError(f"Unknown scale {scale!r}")


def root_for_trajectory(
    family: str = FAMILY_THINK,
    scale: str = SCALE_7B,
    trajectory: str = TRAJECTORY_SFT,
    quick: bool = False,
    project_root: Path | None = None,
) -> Path:
    config = trajectory_config(family, scale, trajectory)
    base = PROJECT_ROOT if project_root is None else Path(project_root)
    if config.root_name == RESULTS_ROOT.name and quick:
        return base / "logs" / RESULTS_ROOT_QUICK.name
    root = base / "logs" / config.root_name
    return Path(f"{root}_quick") if quick else root


def validate_extraction_root_not_other_trajectory(
    root: Path,
    *,
    family: str,
    scale: str,
    trajectory: str,
    project_root: Path | None = None,
    quick: bool = False,
) -> None:
    for other in available_trajectories(family, scale):
        if other == trajectory:
            continue
        canonical = root_for_trajectory(
            family, scale, other, quick=quick, project_root=project_root
        )
        if Path(root).resolve() == canonical.resolve():
            raise ValueError(
                f"extraction root {root} belongs to {other}, not {trajectory}"
            )


def ownership_payload(
    *,
    scale: str,
    trajectory: str,
    model_key: str,
    checkpoints: list[str],
    revisions: dict[str, str],
    purpose: str,
) -> dict[str, object]:
    if purpose not in {"extraction", "math500"}:
        raise ValueError(f"unknown ownership purpose {purpose!r}")
    return {
        "scale": scale,
        "trajectory": trajectory,
        "model_key": model_key,
        "checkpoints": list(checkpoints),
        "revisions": [
            [checkpoint, revisions[checkpoint]] for checkpoint in checkpoints
        ],
        "purpose": purpose,
    }


def _ownership_identity(
    *,
    scale: str,
    trajectory: str,
    model_key: str,
    checkpoints: list[str],
    revisions: dict[str, str],
    purpose: str,
) -> dict[str, object]:
    return ownership_payload(
        scale=scale,
        trajectory=trajectory,
        model_key=model_key,
        checkpoints=checkpoints,
        revisions=revisions,
        purpose=purpose,
    )


def validate_root_ownership(
    root: Path,
    *,
    family: str,
    scale: str,
    trajectory: str,
    model_key: str,
    checkpoints: list[str],
    revisions: dict[str, str],
    purpose: str,
    canonical: bool = False,
) -> None:
    marker = root / ".trajectory_identity.json"
    identity = _ownership_identity(
        scale=scale,
        trajectory=trajectory,
        model_key=model_key,
        checkpoints=checkpoints,
        revisions=revisions,
        purpose=purpose,
    )
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("output root has an unreadable ownership marker") from exc
        if existing != identity:
            raise ValueError("output root has different ownership")
        return
    if canonical:
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError("output root is non-empty and unmarked")


def claim_root_ownership(
    root: Path,
    *,
    family: str,
    scale: str,
    trajectory: str,
    model_key: str,
    checkpoints: list[str],
    revisions: dict[str, str],
    purpose: str,
) -> None:
    validate_root_ownership(
        root,
        family=family,
        scale=scale,
        trajectory=trajectory,
        model_key=model_key,
        checkpoints=checkpoints,
        revisions=revisions,
        purpose=purpose,
    )
    marker = root / ".trajectory_identity.json"
    if marker.exists():
        return
    identity = _ownership_identity(
        scale=scale,
        trajectory=trajectory,
        model_key=model_key,
        checkpoints=checkpoints,
        revisions=revisions,
        purpose=purpose,
    )
    root.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(identity, indent=2, ensure_ascii=False) + "\n"
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError("output root has different ownership")
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        marker.unlink(missing_ok=True)
        raise


def ensure_root_ownership(
    root: Path,
    *,
    family: str,
    scale: str,
    trajectory: str,
    model_key: str,
    checkpoints: list[str],
    revisions: dict[str, str],
    purpose: str,
    canonical: bool = False,
) -> None:
    validate_root_ownership(
        root,
        family=family,
        scale=scale,
        trajectory=trajectory,
        model_key=model_key,
        checkpoints=checkpoints,
        revisions=revisions,
        purpose=purpose,
        canonical=canonical,
    )
    if not canonical:
        claim_root_ownership(
            root,
            family=family,
            scale=scale,
            trajectory=trajectory,
            model_key=model_key,
            checkpoints=checkpoints,
            revisions=revisions,
            purpose=purpose,
        )
