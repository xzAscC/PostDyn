"""Read-only gates shared by downstream 32B pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from postdyn.math500_ablation_validator import validate_result_tree as validate_math
from postdyn.think_sft_differential_experiment import (
    FAMILY_THINK,
    SCALE_7B,
    root_for_trajectory,
)
from postdyn.think_sft_differential_validator import (
    validate_result_tree as validate_extraction,
)


@dataclass(frozen=True)
class Canonical7BPreflightReport:
    """Results from validating every canonical 7B upstream tree."""

    errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors


class Canonical7BPreflightError(RuntimeError):
    """Raised when a 32B pipeline is started before 7B work is complete."""


def preflight_canonical_7b(
    *, project_root: Path | None = None
) -> Canonical7BPreflightReport:
    """Validate both signed-extraction and MATH 7B trees without writing.

    The validators are deliberately called with their canonical roots and
    complete schedules.  This function does not create directories, rebuild
    aggregates, load models, or access the network.
    """
    base = Path(project_root) if project_root is not None else Path(__file__).parents[1]
    extraction_report = preflight_canonical_7b_extraction(project_root=base)
    extraction_errors = list(extraction_report.errors)
    math_errors: list[str] = []
    dataset_path = base / "data" / "math500.json"
    for trajectory, name in (
        ("sft", "math500_ablation_first50"),
        ("rlvr", "math500_ablation_first50_rlvr"),
    ):
        artifact_root = root_for_trajectory(
            FAMILY_THINK, SCALE_7B, trajectory, project_root=base
        )
        report = validate_math(
            base / "logs" / name,
            trajectory=trajectory,
            dataset_path=dataset_path,
            max_new_tokens=2048,
            dtype="bfloat16",
            quantization="none",
            scale=SCALE_7B,
            artifact_root=artifact_root,
            project_root=base,
        )
        if not report.ok:
            math_errors.extend(f"math/{trajectory}: {error}" for error in report.errors)
    return Canonical7BPreflightReport(tuple(extraction_errors + math_errors))


def preflight_canonical_7b_extraction(
    *, project_root: Path | None = None
) -> Canonical7BPreflightReport:
    """Validate only the canonical 7B SFT+RLVR extraction trees."""
    base = Path(project_root) if project_root is not None else Path(__file__).parents[1]
    extraction_errors: list[str] = []
    for trajectory in ("sft", "rlvr"):
        root = root_for_trajectory(
            FAMILY_THINK, SCALE_7B, trajectory, project_root=base
        )
        report = validate_extraction(root, trajectory)
        if not report.ok:
            extraction_errors.extend(
                f"extraction/{trajectory}: {error}" for error in report.errors
            )
    return Canonical7BPreflightReport(tuple(extraction_errors))


def require_canonical_7b(*, project_root: Path | None = None) -> None:
    """Raise before any 32B loader construction or checkpoint access."""
    report = preflight_canonical_7b(project_root=project_root)
    if not report.ok:
        raise Canonical7BPreflightError(
            "canonical 7B SFT+RLVR extraction and MATH preflight failed:\n"
            + "\n".join(f"- {error}" for error in report.errors)
        )


def require_canonical_7b_extraction(*, project_root: Path | None = None) -> None:
    """Raise unless canonical 7B SFT+RLVR extraction trees are complete."""
    report = preflight_canonical_7b_extraction(project_root=project_root)
    if not report.ok:
        raise Canonical7BPreflightError(
            "canonical 7B SFT+RLVR extraction preflight failed:\n"
            + "\n".join(f"- {error}" for error in report.errors)
        )
