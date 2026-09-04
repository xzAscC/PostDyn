from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from postdyn.config import (
    ALPHAS,
    BENCHMARKS,
    DOMAINS,
    K_FRACTION,
    MODEL_FAMILIES,
    ROBUSTNESS_DOMAIN,
    VAL_N,
    CheckpointRef,
)


def test_domains_and_benchmarks_are_frozen() -> None:
    assert DOMAINS == ("math", "code", "instruction_following", "general_reasoning")
    assert BENCHMARKS == {
        "math": "math500",
        "code": "livecodebench",
        "instruction_following": "ifeval",
        "general_reasoning": "mmlu_pro",
    }
    assert ALPHAS == (0.1, 1.0, 10.0)
    assert K_FRACTION == 1 / 3
    assert VAL_N == 30
    assert ROBUSTNESS_DOMAIN == "math"


def test_model_families_have_exact_dimensions_and_layers() -> None:
    assert set(MODEL_FAMILIES) == {"7b", "32b"}
    expected = {
        "7b": (4096, 32, (3, 6, 9, 11, 14, 17, 20, 22, 25, 28)),
        "32b": (5120, 64, (6, 12, 18, 23, 29, 34, 40, 46, 51, 57)),
    }
    for key, (d_model, n_layers, layers) in expected.items():
        family = MODEL_FAMILIES[key]
        assert family.d_model == d_model
        assert family.n_layers == n_layers
        assert family.layers == layers
        assert len(family.layers) == 10
        assert tuple(sorted(family.layers)) == family.layers
        assert all(0 <= layer < family.n_layers for layer in family.layers)


@pytest.mark.parametrize(
    ("key", "repos"),
    [
        (
            "7b",
            (
                "allenai/Olmo-3-1025-7B",
                "allenai/Olmo-3-7B-Think-SFT",
                "allenai/Olmo-3-7B-Think-DPO",
                "allenai/Olmo-3-7B-Think",
            ),
        ),
        (
            "32b",
            (
                "allenai/Olmo-3-1125-32B",
                "allenai/Olmo-3-32B-Think-SFT",
                "allenai/Olmo-3-32B-Think-DPO",
                "allenai/Olmo-3-32B-Think",
            ),
        ),
    ],
)
def test_model_family_repositories(key: str, repos: tuple[str, ...]) -> None:
    family = MODEL_FAMILIES[key]
    assert (
        family.base_repo,
        family.sft_repo,
        family.dpo_repo,
        family.rlvr_repo,
    ) == repos


def test_checkpoint_counts_stages_and_final_revisions() -> None:
    for key, family in MODEL_FAMILIES.items():
        checkpoints = family.checkpoints()
        assert len(checkpoints) == 22
        assert [checkpoint.stage for checkpoint in checkpoints].count("base") == 1
        assert [checkpoint.stage for checkpoint in checkpoints].count("sft") == 10
        assert [checkpoint.stage for checkpoint in checkpoints].count("dpo") == 1
        assert [checkpoint.stage for checkpoint in checkpoints].count("rlvr") == 10

        sft = [checkpoint for checkpoint in checkpoints if checkpoint.stage == "sft"]
        rlvr = [checkpoint for checkpoint in checkpoints if checkpoint.stage == "rlvr"]
        assert (sft[-1].name, sft[-1].repo, sft[-1].revision) == (
            "sft",
            family.sft_repo,
            "main",
        )
        assert (rlvr[-1].name, rlvr[-1].repo, rlvr[-1].revision) == (
            "rlvr",
            family.rlvr_repo,
            "main",
        )

        if key == "7b":
            revisions = [checkpoint.revision for checkpoint in sft[:-1]]
            steps = [int(revision.removeprefix("step")) for revision in revisions]
            assert all(re.fullmatch(r"step\d+", revision) for revision in revisions)
            assert steps[0] == 1000
            assert 1000 <= steps[-1] <= 43000
            assert len({right - left for left, right in zip(steps, steps[1:])}) == 1
            assert all(
                re.fullmatch(r"step_\d{4}", checkpoint.revision)
                and 25 <= int(checkpoint.revision.removeprefix("step_")) <= 1375
                for checkpoint in rlvr[:-1]
            )
        else:
            assert all(
                checkpoint.revision.startswith("1e-4-") for checkpoint in sft[:-1]
            )
            assert all(
                re.fullmatch(r"step_\d{3}", checkpoint.revision)
                and 50 <= int(checkpoint.revision.removeprefix("step_")) <= 750
                for checkpoint in rlvr[:-1]
            )


def test_32b_sft_learning_rate_prefix_switches() -> None:
    family = MODEL_FAMILIES["32b"]
    assert all(
        checkpoint.revision.startswith("1e-4-")
        for checkpoint in family.checkpoints()
        if checkpoint.stage == "sft" and checkpoint.revision != "main"
    )
    assert all(
        checkpoint.revision.startswith("5e-5-")
        for checkpoint in family.checkpoints(sft_lr="5e-5")
        if checkpoint.stage == "sft" and checkpoint.revision != "main"
    )


def test_checkpoint_selection_is_deterministic_and_uniform() -> None:
    family = MODEL_FAMILIES["7b"]
    first = family.checkpoints()
    second = family.checkpoints()
    assert first == second

    sft_steps = [
        int(checkpoint.revision.removeprefix("step"))
        for checkpoint in first
        if checkpoint.stage == "sft" and checkpoint.revision != "main"
    ]
    assert sft_steps[0] == 1000
    assert sft_steps[-1] == 43000
    assert len(sft_steps) == 9


def test_sample_counts_and_checkpoint_ref_is_frozen() -> None:
    assert MODEL_FAMILIES["7b"].n_samples() == 12288
    assert MODEL_FAMILIES["32b"].n_samples() == 15360
    checkpoint = CheckpointRef("name", "repo", "revision", "base")
    with pytest.raises(FrozenInstanceError):
        setattr(checkpoint, "name", "changed")
