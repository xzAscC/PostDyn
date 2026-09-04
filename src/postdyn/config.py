"""Frozen configuration for the PostDyn rewrite."""

from __future__ import annotations

from dataclasses import dataclass

DOMAINS: tuple[str, ...] = (
    "math",
    "code",
    "instruction_following",
    "general_reasoning",
)
BENCHMARKS: dict[str, str] = {
    "math": "math500",
    "code": "livecodebench",
    "instruction_following": "ifeval",
    "general_reasoning": "mmlu_pro",
}
SAMPLING_SEED = 42
ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0)
K_FRACTION = 1 / 3
VAL_N = 30
ROBUSTNESS_DOMAIN = "math"


@dataclass(frozen=True)
class CheckpointRef:
    """An immutable model checkpoint reference."""

    name: str
    repo: str
    revision: str
    stage: str


@dataclass(frozen=True)
class FamilyConfig:
    """Model dimensions, repositories, and checkpoint schedule metadata."""

    key: str
    d_model: int
    n_layers: int
    layers: tuple[int, ...]
    base_repo: str
    sft_repo: str
    dpo_repo: str
    rlvr_repo: str
    sft_branch_prefix: str = ""

    def checkpoints(self, sft_lr: str = "1e-4") -> tuple[CheckpointRef, ...]:
        """Return 22 deterministic checkpoints spanning all four stages."""
        if self.key == "7b":
            sft_steps = [f"step{1000 + 5250 * index}" for index in range(9)]
            rlvr_steps = [f"step_{step:04d}" for step in range(25, 1376, 25)]
        else:
            if sft_lr not in {"1e-4", "5e-5"}:
                raise ValueError("sft_lr must be '1e-4' or '5e-5'")
            sft_steps = [f"step{step}" for step in range(1000, 10000, 1000)] + [
                "step10790"
            ]
            rlvr_steps = [f"step_{step:03d}" for step in range(50, 751, 50)]

        sft_revisions = _uniform_with_final(sft_steps)
        rlvr_revisions = _uniform_with_final(rlvr_steps)
        sft_prefix = self.sft_branch_prefix if self.key == "7b" else sft_lr + "-"
        refs = [CheckpointRef("base", self.base_repo, "main", "base")]
        refs.extend(
            CheckpointRef(
                "sft" if revision == "main" else f"sft_{revision}",
                self.sft_repo,
                sft_prefix + revision if revision != "main" else "main",
                "sft",
            )
            for revision in sft_revisions
        )
        refs.append(CheckpointRef("dpo", self.dpo_repo, "main", "dpo"))
        refs.extend(
            CheckpointRef(
                "rlvr" if revision == "main" else f"rlvr_{revision}",
                self.rlvr_repo,
                revision,
                "rlvr",
            )
            for revision in rlvr_revisions
        )
        return tuple(refs)

    def n_samples(self) -> int:
        """Return the three-dimensional-model sample count."""
        return 3 * self.d_model


def _uniform_with_final(steps: list[str]) -> list[str]:
    """Select nine uniform steps and append the stage's main branch."""
    if len(steps) <= 9:
        selected = steps
    else:
        last = len(steps) - 1
        indices = [round(i * last / 8) for i in range(9)]
        selected = [steps[index] for index in dict.fromkeys(indices)]
    return selected + ["main"]


MODEL_FAMILIES: dict[str, FamilyConfig] = {
    "7b": FamilyConfig(
        "7b",
        4096,
        32,
        (3, 6, 9, 11, 14, 17, 20, 22, 25, 28),
        "allenai/Olmo-3-1025-7B",
        "allenai/Olmo-3-7B-Think-SFT",
        "allenai/Olmo-3-7B-Think-DPO",
        "allenai/Olmo-3-7B-Think",
        "",
    ),
    "32b": FamilyConfig(
        "32b",
        5120,
        64,
        (6, 12, 18, 23, 29, 34, 40, 46, 51, 57),
        "allenai/Olmo-3-1125-32B",
        "allenai/Olmo-3-32B-Think-SFT",
        "allenai/Olmo-3-32B-Think-DPO",
        "allenai/Olmo-3-32B-Think",
        "1e-4-",
    ),
}
