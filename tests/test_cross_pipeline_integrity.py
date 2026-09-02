from __future__ import annotations

from pathlib import Path

import pytest

from postdyn import cross_pipeline_integrity as integrity


def test_preflight_checks_both_extraction_and_math_7b_roots(monkeypatch) -> None:
    extraction: list[tuple[Path, str]] = []
    math: list[tuple[Path, str]] = []

    class Good:
        ok = True
        errors: list[str] = []

    monkeypatch.setattr(
        integrity,
        "validate_extraction",
        lambda root, trajectory: extraction.append((root, trajectory)) or Good(),
    )
    monkeypatch.setattr(
        integrity,
        "validate_math",
        lambda root, **kwargs: math.append((root, kwargs["trajectory"])) or Good(),
    )

    report = integrity.preflight_canonical_7b(project_root=Path("/tmp/project"))

    assert report.ok
    assert [trajectory for _, trajectory in extraction] == ["sft", "rlvr"]
    assert [trajectory for _, trajectory in math] == ["sft", "rlvr"]
    assert math[0][0].name == "math500_ablation_first50"
    assert math[1][0].name == "math500_ablation_first50_rlvr"


def test_preflight_scopes_extraction_roots_to_alternate_project(monkeypatch) -> None:
    extraction: list[Path] = []
    math: list[tuple[Path, Path, Path]] = []

    class Good:
        ok = True
        errors: list[str] = []

    monkeypatch.setattr(
        integrity,
        "validate_extraction",
        lambda root, trajectory: extraction.append(root) or Good(),
    )
    monkeypatch.setattr(
        integrity,
        "validate_math",
        lambda root, **kwargs: (
            math.append((root, kwargs["artifact_root"], kwargs["project_root"]))
            or Good()
        ),
    )

    integrity.preflight_canonical_7b(project_root=Path("/tmp/alternate"))

    assert all(str(root).startswith("/tmp/alternate/logs/") for root in extraction)
    assert all(str(root).startswith("/tmp/alternate/logs/") for root, _, _ in math)
    assert [artifact_root for _, artifact_root, _ in math] == extraction
    assert all(project_root == Path("/tmp/alternate") for _, _, project_root in math)


def test_extraction_preflight_ignores_failing_math_validator(
    monkeypatch, tmp_path
) -> None:
    extraction_roots: list[Path] = []

    class Good:
        ok = True
        errors: list[str] = []

    monkeypatch.setattr(
        integrity,
        "validate_extraction",
        lambda root, trajectory: extraction_roots.append(root) or Good(),
    )

    def fail_math(*args, **kwargs):
        raise AssertionError("MATH validation is not part of the extraction gate")

    monkeypatch.setattr(integrity, "validate_math", fail_math)

    from scripts import run_think_sft_differential_subspace as runner

    assert (
        runner.main(
            [
                "--scale",
                "32b",
                "--trajectory",
                "rlvr",
                "--allow-32b",
                "--dry-run",
                "--no-fixed-points",
                "--project-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert all(
        str(root).startswith(str(tmp_path / "logs")) for root in extraction_roots
    )


def test_extraction_preflight_blocks_before_loader_when_extraction_is_incomplete(
    monkeypatch, tmp_path
) -> None:
    class Incomplete:
        ok = False
        errors = ["missing canonical extraction artifact"]

    monkeypatch.setattr(integrity, "validate_extraction", lambda *args: Incomplete())
    monkeypatch.setattr(
        integrity,
        "validate_math",
        lambda *args, **kwargs: pytest.fail("MATH validation must not run"),
    )

    from scripts import run_think_sft_differential_subspace as runner

    monkeypatch.setattr(
        runner,
        "build_model_loader",
        lambda *args, **kwargs: pytest.fail("loader must not be constructed"),
    )

    with pytest.raises(
        integrity.Canonical7BPreflightError,
        match="missing canonical extraction artifact",
    ):
        runner.main(
            [
                "--scale",
                "32b",
                "--trajectory",
                "rlvr",
                "--allow-32b",
                "--dry-run",
                "--no-fixed-points",
                "--project-root",
                str(tmp_path),
            ]
        )


def test_validator_accepts_alternate_project_root_for_default_artifacts(
    monkeypatch, tmp_path
):
    from postdyn import math500_ablation_validator as validator

    monkeypatch.setattr(
        validator, "collect_valid_conditions", lambda *args, **kwargs: ([], [])
    )
    report = validator.validate_result_tree(
        tmp_path,
        trajectory="sft",
        dataset_path=tmp_path / "math500.json",
        max_new_tokens=2048,
        dtype="bfloat16",
        quantization="none",
        selected_checkpoints=(),
        selected_layers=(),
        project_root=Path("/tmp/alternate"),
    )
    assert report.root == str(tmp_path)


def test_preflight_failure_blocks_extraction_loader(monkeypatch) -> None:
    from scripts import run_think_sft_differential_subspace as runner

    loader_called = False
    preflight_roots: list[Path | None] = []

    def fail_preflight(**kwargs) -> None:
        preflight_roots.append(kwargs.get("project_root"))
        raise integrity.Canonical7BPreflightError("incomplete upstream")

    def fail_loader(*args, **kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("32B loader must not be constructed")

    monkeypatch.setattr(integrity, "require_canonical_7b_extraction", fail_preflight)
    monkeypatch.setattr(runner, "build_model_loader", fail_loader)

    with pytest.raises(
        integrity.Canonical7BPreflightError, match="incomplete upstream"
    ):
        runner.main(
            [
                "--scale",
                "32b",
                "--trajectory",
                "rlvr",
                "--allow-32b",
                "--project-root",
                "/tmp/alternate",
            ]
        )

    assert loader_called is False
    assert preflight_roots == [Path("/tmp/alternate")]


def test_extraction_preflight_failure_leaves_custom_output_absent(
    monkeypatch, tmp_path
) -> None:
    from scripts import run_think_sft_differential_subspace as runner

    monkeypatch.setattr(
        integrity,
        "require_canonical_7b_extraction",
        lambda **kwargs: (_ for _ in ()).throw(
            integrity.Canonical7BPreflightError("incomplete upstream")
        ),
    )
    output = tmp_path / "custom-output"

    with pytest.raises(integrity.Canonical7BPreflightError):
        runner.main(
            [
                "--scale",
                "32b",
                "--trajectory",
                "rlvr",
                "--allow-32b",
                "--output",
                str(output),
            ]
        )

    assert not output.exists()


def test_preflight_failure_blocks_math_loader(monkeypatch) -> None:
    from scripts import run_math500_ablation as runner

    loader_called = False

    def fail_preflight() -> None:
        raise integrity.Canonical7BPreflightError("incomplete upstream")

    def fail_loader(*args, **kwargs):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("32B loader must not be called")

    monkeypatch.setattr(integrity, "require_canonical_7b", fail_preflight)
    monkeypatch.setattr(runner, "load_model_for_run", fail_loader)

    args = runner.parse_args(["--scale", "32b", "--trajectory", "rlvr"])
    assert runner.run(args) == 2
    assert loader_called is False


def test_32b_preflight_failure_leaves_custom_roots_absent(
    monkeypatch, tmp_path
) -> None:
    from scripts import run_math500_ablation as runner

    monkeypatch.setattr(
        integrity,
        "require_canonical_7b",
        lambda **kwargs: (_ for _ in ()).throw(
            integrity.Canonical7BPreflightError("incomplete upstream")
        ),
    )
    artifact_root = tmp_path / "artifacts"
    result_root = tmp_path / "logs"
    args = runner.parse_args(
        [
            "--scale",
            "32b",
            "--trajectory",
            "rlvr",
            "--artifact-root",
            str(artifact_root),
            "--result-root",
            str(result_root),
        ]
    )

    assert runner.run(args) == 2
    assert not artifact_root.exists()
    assert not result_root.exists()
