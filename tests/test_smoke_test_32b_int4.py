from __future__ import annotations

from unittest.mock import Mock
from pathlib import Path

from experiments import smoke_test_32b_int4 as smoke
from src.think_sft_differential_experiment import trajectory_config


def test_load_requires_canonical_7b_preflight_before_dependencies(monkeypatch, capsys):
    preflight = Mock(side_effect=RuntimeError("preflight failed"))
    dependencies = Mock()
    loader = Mock()
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)
    monkeypatch.setattr(smoke, "check_quantization_dependencies", dependencies)
    monkeypatch.setattr(smoke, "load_olmo3_32b_think", loader)

    assert (
        smoke.main(
            [
                "--load",
                "--trajectory",
                "sft_lr_1e-4",
                "--checkpoint",
                "step1000",
            ]
        )
        == 2
    )

    preflight.assert_called_once_with()
    dependencies.assert_not_called()
    loader.assert_not_called()
    assert "preflight failed" in capsys.readouterr().err


def test_load_rejects_missing_request_before_preflight_or_dependencies(
    monkeypatch, capsys
):
    preflight = Mock()
    dependencies = Mock()
    loader = Mock()
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)
    monkeypatch.setattr(smoke, "check_quantization_dependencies", dependencies)
    monkeypatch.setattr(smoke, "load_olmo3_32b_think", loader)

    assert smoke.main(["--load"]) == 2

    preflight.assert_not_called()
    dependencies.assert_not_called()
    loader.assert_not_called()
    assert "--checkpoint" in capsys.readouterr().err


def test_load_rejects_invalid_request_before_preflight_or_dependencies(
    monkeypatch, capsys
):
    preflight = Mock()
    dependencies = Mock()
    loader = Mock()
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)
    monkeypatch.setattr(smoke, "check_quantization_dependencies", dependencies)
    monkeypatch.setattr(smoke, "load_olmo3_32b_think", loader)

    assert (
        smoke.main(["--load", "--model-id", "arbitrary/model", "--revision", "main"])
        == 2
    )

    preflight.assert_not_called()
    dependencies.assert_not_called()
    loader.assert_not_called()


def test_load_forwards_project_root_to_preflight(monkeypatch, capsys):
    preflight = Mock()
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)
    monkeypatch.setattr(smoke, "check_quantization_dependencies", Mock(return_value={}))
    monkeypatch.setattr(smoke, "load_olmo3_32b_think", Mock(return_value=Mock()))

    assert (
        smoke.main(
            [
                "--load",
                "--trajectory",
                "sft_lr_1e-4",
                "--checkpoint",
                "step1000",
                "--project-root",
                "/tmp/alternate",
            ]
        )
        == 0
    )
    preflight.assert_called_once_with(project_root=Path("/tmp/alternate"))


def test_without_load_keeps_dependency_only_behavior(monkeypatch, capsys):
    preflight = Mock()
    dependencies = Mock(return_value={"transformers": "4.57.1"})
    loader = Mock()
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)
    monkeypatch.setattr(smoke, "check_quantization_dependencies", dependencies)
    monkeypatch.setattr(smoke, "load_olmo3_32b_think", loader)

    assert smoke.main([]) == 0

    preflight.assert_not_called()
    dependencies.assert_called_once_with()
    loader.assert_not_called()
    assert "Dependency check passed" in capsys.readouterr().out


def test_canonical_smoke_resolves_model_and_exact_configured_revision(monkeypatch):
    config = trajectory_config("think", "32b", "sft_lr_1e-4")
    checkpoint = config.checkpoints[0]
    dependencies = Mock(return_value={})
    loader = Mock(return_value=Mock(diagnostics=Mock(as_dict=lambda: {})))
    preflight = Mock()
    monkeypatch.setattr(smoke, "check_quantization_dependencies", dependencies)
    monkeypatch.setattr(smoke, "load_olmo3_32b_think", loader)
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)

    assert (
        smoke.main(
            ["--load", "--trajectory", "sft_lr_1e-4", "--checkpoint", checkpoint]
        )
        == 0
    )
    loader.assert_called_once_with(
        "allenai/Olmo-3-32B-Think-SFT", revision=config.revisions[checkpoint]
    )


def test_canonical_smoke_rejects_arbitrary_sha_before_preflight_or_dependencies(
    monkeypatch,
):
    preflight = Mock()
    dependencies = Mock()
    monkeypatch.setattr(smoke, "require_canonical_7b", preflight)
    monkeypatch.setattr(smoke, "check_quantization_dependencies", dependencies)

    assert (
        smoke.main(
            [
                "--load",
                "--trajectory",
                "rlvr",
                "--checkpoint",
                "step_0025",
                "--revision",
                "a" * 40,
            ]
        )
        == 2
    )
    preflight.assert_not_called()
    dependencies.assert_not_called()
