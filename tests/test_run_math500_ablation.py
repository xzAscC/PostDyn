from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import json
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import weakref

import pytest

import scripts.run_math500_ablation as cli
from postdyn.config import OLMO3_VARIANTS
from postdyn.think_sft_differential_experiment import (
    FAMILY_THINK,
    SCALE_7B,
    THINK_7B_RLVR_REVISIONS,
    root_for_trajectory,
    trajectory_config,
)


def test_rlvr_resolves_all_canonical_checkpoints_and_sha_revisions() -> None:
    args = cli.parse_args(["--trajectory", "rlvr"])
    resolved = cli.resolve_run_config(args)
    expected = trajectory_config(FAMILY_THINK, SCALE_7B, "rlvr")

    assert resolved.model_key == "olmo3-think-rlvr"
    assert resolved.checkpoints == expected.checkpoints
    assert len(resolved.checkpoints) == 10
    assert dict(resolved.revisions) == dict(THINK_7B_RLVR_REVISIONS)
    assert resolved.artifact_root == root_for_trajectory(FAMILY_THINK, SCALE_7B, "rlvr")
    assert resolved.result_root == Path("logs/math500_ablation_first50_rlvr")


def test_alternate_project_root_scopes_default_artifact_and_result_roots(tmp_path):
    args = cli.parse_args(["--project-root", str(tmp_path), "--trajectory", "rlvr"])
    resolved = cli.resolve_run_config(args)
    assert (
        resolved.artifact_root
        == tmp_path / "logs" / "think_7b_rlvr_differential_subspace"
    )
    assert (
        resolved.result_root == tmp_path / "logs" / "math500_ablation_first50_rlvr"
    )


def test_default_result_roots_are_scale_isolated() -> None:
    roots_7b = {
        trajectory: cli._default_result_root(trajectory, SCALE_7B)
        for trajectory in ("sft", "rlvr")
    }
    roots_32b = {
        trajectory: cli._default_result_root(trajectory, "32b")
        for trajectory in ("sft_lr_1e-4", "sft_lr_5e-5", "rlvr")
    }

    assert roots_7b == {
        "sft": Path("logs/math500_ablation_first50"),
        "rlvr": Path("logs/math500_ablation_first50_rlvr"),
    }
    assert roots_32b == {
        trajectory: Path(f"logs/math500_ablation_first50_32b_{trajectory}")
        for trajectory in ("sft_lr_1e-4", "sft_lr_5e-5", "rlvr")
    }
    assert len(set(roots_7b.values()) | set(roots_32b.values())) == 5


def test_32b_rejects_a_7b_default_result_root() -> None:
    args = cli.parse_args(
        [
            "--scale",
            "32b",
            "--trajectory",
            "rlvr",
            "--result-root",
            str(cli._default_result_root("rlvr", SCALE_7B)),
        ]
    )

    with pytest.raises(ValueError, match="belongs to 7b/rlvr"):
        cli.resolve_run_config(args)


def test_custom_result_root_rejects_normalized_cross_scale_collision() -> None:
    args = cli.parse_args(
        [
            "--scale",
            "32b",
            "--trajectory",
            "rlvr",
            "--result-root",
            "logs/other/../math500_ablation_first50_rlvr",
        ]
    )

    with pytest.raises(ValueError, match="belongs to 7b/rlvr"):
        cli.resolve_run_config(args)


def test_artifact_and_result_roots_must_differ(tmp_path: Path) -> None:
    args = cli.parse_args(
        [
            "--artifact-root",
            str(tmp_path),
            "--result-root",
            str(tmp_path),
        ]
    )

    with pytest.raises(ValueError, match="must differ"):
        cli.resolve_run_config(args)


def test_project_root_scopes_artifact_collision_checks(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    foreign = project_root / "logs" / "think_sft_differential_subspace"
    args = cli.parse_args(
        [
            "--project-root",
            str(project_root),
            "--trajectory",
            "rlvr",
            "--artifact-root",
            str(foreign),
        ]
    )

    with pytest.raises(ValueError, match="belongs to sft"):
        cli.resolve_run_config(args)


def test_cached_validation_errors_are_fatal_before_summary_publication(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        ["--checkpoints", "step1000", "--layers", "3", "--result-root", str(tmp_path)]
    )
    config = cli.resolve_run_config(args)
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar",
        "tensor",
        {"U_pos": object(), "U_neg": object()},
        {
            "model": OLMO3_VARIANTS[config.model_key].name,
            "checkpoint": "step1000",
            "revision": config.revision_for("step1000"),
        },
    )
    monkeypatch.setattr(cli, "load_and_validate_basis", lambda *a, **k: basis)
    monkeypatch.setattr(
        cli, "load_authoritative_summary", lambda **kwargs: {"n_processed": 50}
    )
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *a, **k: ([{"stale": True}], ["bad cached condition"]),
    )

    with pytest.raises(ValueError, match="bad cached condition"):
        cli.run_checkpoint(
            args=args,
            run_config=replace(config, result_root=tmp_path),
            checkpoint="step1000",
        )
    assert not (tmp_path / "checkpoints" / "step1000" / "summary.json").exists()


def test_fresh_validation_errors_are_fatal_and_release_loaded_references(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        [
            "--checkpoints",
            "step1000",
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
            "--force",
        ]
    )
    config = cli.resolve_run_config(args)
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar",
        "tensor",
        {"U_pos": object(), "U_neg": object()},
        {
            "model": OLMO3_VARIANTS[config.model_key].name,
            "checkpoint": "step1000",
            "revision": config.revision_for("step1000"),
        },
    )

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

    model = Model()
    tokenizer = object()
    model_ref = weakref.ref(model)
    tokenizer_ref = (
        weakref.ref(tokenizer) if hasattr(tokenizer, "__weakref__") else None
    )
    monkeypatch.setattr(cli, "load_and_validate_basis", lambda *a, **k: basis)
    monkeypatch.setattr(cli, "GreedyGenerator", lambda **kwargs: object())
    monkeypatch.setattr(cli, "residual_stream_ablation", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        cli,
        "evaluate_first50",
        lambda **kwargs: SimpleNamespace(to_dict=lambda: {"n_processed": 50}),
    )
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *a, **k: ([], ["post-generation validation failed"]),
    )
    loaded_holder = [(model, tokenizer)]

    with pytest.raises(ValueError, match="post-generation validation failed"):
        cli.run_checkpoint(
            args=args,
            run_config=replace(config, result_root=tmp_path),
            checkpoint="step1000",
            load_model=lambda *a: loaded_holder.pop(),
        )
    del model, tokenizer
    assert model_ref() is None
    if tokenizer_ref is not None:
        assert tokenizer_ref() is None
    assert not (tmp_path / "checkpoints" / "step1000" / "summary.json").exists()


def test_run_and_rebuild_fail_before_publishing_on_validation_errors(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        ["--checkpoints", "step1000", "--layers", "3", "--result-root", str(tmp_path)]
    )
    monkeypatch.setattr(
        cli, "run_checkpoint", lambda **kwargs: [{"checkpoint": "step1000"}]
    )
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *a, **k: ([{"checkpoint": "step1000"}], ["aggregate validation failed"]),
    )
    config = cli.resolve_run_config(args)
    cli.ensure_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="sft",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="math500",
    )
    (tmp_path / "aggregate.json").write_text("old")
    assert cli.run(args) == 2
    assert not (tmp_path / "aggregate.json").exists()

    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *a, **k: ([{"checkpoint": "step1000"}], ["rebuild validation failed"]),
    )
    with pytest.raises(ValueError, match="rebuild validation failed"):
        cli.rebuild_aggregate(
            tmp_path,
            trajectory="sft",
            model_key="olmo3-think-sft",
            selected_checkpoints=("step1000",),
            selected_layers=(),
        )


def test_fresh_multi_checkpoint_run_validates_prefix_then_full_set(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoints = trajectory_config(FAMILY_THINK, SCALE_7B, "sft").checkpoints[:2]
    args = cli.parse_args(
        [
            "--checkpoints",
            *checkpoints,
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
        ]
    )
    config = cli.resolve_run_config(args)
    seen: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda **kwargs: [{"checkpoint": kwargs["checkpoint"]}],
    )

    def fake_collect(*args, **kwargs):
        selected = tuple(kwargs["selected_checkpoints"])
        seen.append(selected)
        return (
            [
                {"checkpoint": checkpoint, "condition": "baseline"}
                for checkpoint in selected
            ],
            [],
        )

    monkeypatch.setattr(cli, "collect_valid_conditions", fake_collect)

    assert cli.run(args) == 0
    assert seen == [checkpoints[:1], checkpoints, checkpoints]
    assert json.loads((tmp_path / "aggregate.json").read_text())["conditions"] == [
        {"checkpoint": checkpoints[0], "condition": "baseline"},
        {"checkpoint": checkpoints[1], "condition": "baseline"},
    ]


def test_multi_checkpoint_final_validation_error_does_not_publish_aggregate(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoints = trajectory_config(FAMILY_THINK, SCALE_7B, "sft").checkpoints[:2]
    args = cli.parse_args(
        [
            "--checkpoints",
            *checkpoints,
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
        ]
    )
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda **kwargs: [{"checkpoint": kwargs["checkpoint"]}],
    )
    calls = 0

    def fake_collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        selected = tuple(kwargs["selected_checkpoints"])
        if calls == 3:
            return [], ["final aggregate validation failed"]
        return [{"checkpoint": selected[0], "condition": "baseline"}], []

    monkeypatch.setattr(cli, "collect_valid_conditions", fake_collect)
    config = cli.resolve_run_config(args)
    cli.ensure_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="sft",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="math500",
    )
    (tmp_path / "aggregate.json").write_text("old")

    assert cli.run(args) == 2
    assert calls == 3
    assert not (tmp_path / "aggregate.json").exists()


@pytest.mark.parametrize("bad_tokens", [1, 2047, 2049])
def test_32b_runner_rejects_noncanonical_max_new_tokens(bad_tokens: int) -> None:
    args = cli.parse_args(
        ["--scale", "32b", "--trajectory", "rlvr", "--max-new-tokens", str(bad_tokens)]
    )

    with pytest.raises(ValueError, match="max_new_tokens=2048"):
        cli.resolve_run_config(args)


def test_7b_runner_keeps_custom_max_new_tokens() -> None:
    args = cli.parse_args(["--max-new-tokens", "17"])
    resolved = cli.resolve_run_config(args)

    assert resolved.scale == SCALE_7B
    assert args.max_new_tokens == 17


def test_32b_runner_cli_rejects_noncanonical_max_new_tokens() -> None:
    args = cli.parse_args(
        ["--scale", "32b", "--trajectory", "rlvr", "--max-new-tokens", "2049"]
    )

    assert cli.run(args) == 2


def test_rlvr_accepts_sha_override_only_for_one_matching_checkpoint() -> None:
    checkpoint = "step_0025"
    revision = THINK_7B_RLVR_REVISIONS[checkpoint]
    args = cli.parse_args(
        [
            "--trajectory",
            "rlvr",
            "--checkpoints",
            checkpoint,
            "--revision",
            revision,
        ]
    )

    resolved = cli.resolve_run_config(args)
    assert resolved.checkpoints == (checkpoint,)
    assert resolved.revision_for(checkpoint) == revision


@pytest.mark.parametrize("trajectory", ("sft_lr_1e-4", "sft_lr_5e-5", "rlvr"))
def test_32b_resolves_canonical_trajectory_layers_and_nf4(trajectory: str) -> None:
    args = cli.parse_args(["--scale", "32b", "--trajectory", trajectory])
    resolved = cli.resolve_run_config(args)
    expected = trajectory_config(FAMILY_THINK, "32b", trajectory)

    assert resolved.scale == "32b"
    assert resolved.checkpoints == expected.checkpoints
    assert dict(resolved.revisions) == dict(expected.revisions)
    assert tuple(args.layers) == tuple(cli.layers_for_scale("32b"))
    assert args.quantization == "nf4"
    assert resolved.artifact_root == root_for_trajectory(
        FAMILY_THINK, "32b", trajectory
    )
    assert resolved.result_root == Path(
        f"logs/math500_ablation_first50_32b_{trajectory}"
    )


def test_32b_loader_routes_to_nf4_without_moving_dispatched_model(monkeypatch) -> None:
    import postdyn.quantized_model_loader as loader

    model = SimpleNamespace()
    called: dict[str, object] = {}
    gate_called: dict[str, object] = {}

    def fake_loader(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(
            model=model,
            tokenizer="tokenizer",
            diagnostics=SimpleNamespace(
                as_dict=lambda: {"placement": "gpu-only", "device_map": {"": "cuda:0"}}
            ),
        )

    monkeypatch.setattr(loader, "load_olmo3_32b_think", fake_loader)
    monkeypatch.setattr(
        cli, "_require_canonical_7b", lambda **kwargs: gate_called.update(kwargs)
    )
    loaded_model, tokenizer, provenance = cli.load_model_for_run(
        "olmo3-32b-think-sft",
        "sha",
        "bfloat16",
        "nf4",
        project_root=Path("/tmp/project"),
    )

    assert loaded_model is model
    assert tokenizer == "tokenizer"
    assert called == {
        "model_id": "allenai/Olmo-3-32B-Think-SFT",
        "revision": "sha",
    }
    assert provenance is not None
    assert provenance["loader"] == "load_olmo3_32b_think"
    assert provenance["nf4_config"] == dict(cli.NF4_CONFIG)
    assert not hasattr(model, "to")
    assert gate_called == {"project_root": Path("/tmp/project")}


def test_model_device_never_returns_meta_or_nonconcrete_fallback() -> None:
    class Model:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="meta"))

        hf_device_map = {"model.embed_tokens": "meta", "model.layers.0": "meta"}

        def parameters(self):
            yield SimpleNamespace(device="meta")

        def generate(self, *args, **kwargs):
            raise NotImplementedError

    with pytest.raises(ValueError, match="concrete execution device"):
        cli._model_device(Model())


def test_32b_basis_loader_rejects_missing_static_nf4_provenance(tmp_path: Path) -> None:
    model = "olmo3-32b-think-sft"
    checkpoint = "step1000"
    layer = 6
    base = tmp_path / "U" / model / checkpoint / f"layer_{layer}" / "math_vs_text"
    base.parent.mkdir(parents=True)
    base.with_suffix(".json").write_text(
        json.dumps(
            {
                "model": model,
                "checkpoint": checkpoint,
                "revision": checkpoint,
                "layer": layer,
                "setup_signature": "setup-v1",
                "d_model": 5120,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="noncanonical static NF4 provenance"):
        cli.load_and_validate_basis(
            tmp_path, model, checkpoint, checkpoint, layer, expected_d_model=5120
        )


def test_32b_incomplete_extraction_publication_blocks_basis_loading(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        [
            "--scale",
            "32b",
            "--trajectory",
            "sft_lr_1e-4",
            "--checkpoints",
            "step1000",
            "--layers",
            "6",
            "--result-root",
            str(tmp_path / "logs"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--max-new-tokens",
            "2048",
            "--quantization",
            "nf4",
        ]
    )
    config = cli.resolve_run_config(args)
    config = replace(config, artifact_root=tmp_path / "artifacts")
    monkeypatch.setattr(
        "postdyn.think_32b_differential_validator.validate_full_canonical_publication",
        lambda *a, **k: SimpleNamespace(ok=False, errors=["incomplete publication"]),
    )
    monkeypatch.setattr(
        cli,
        "run_checkpoint",
        lambda **kwargs: pytest.fail("checkpoint execution must not occur"),
    )
    import postdyn.cross_pipeline_integrity as integrity

    monkeypatch.setattr(integrity, "require_canonical_7b", lambda **kwargs: None)

    assert cli.run(args) == 2


def test_direct_32b_checkpoint_validates_publication_before_unlink_or_basis(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        ["--scale", "32b", "--trajectory", "rlvr", "--checkpoints", "step_050"]
    )
    config = cli.resolve_run_config(args)
    result_root = tmp_path / "logs"
    summary = result_root / "checkpoints" / "step_050" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("old", encoding="utf-8")
    events: list[str] = []
    monkeypatch.setattr(cli, "_require_canonical_7b", lambda **kwargs: None)
    monkeypatch.setattr(
        "postdyn.think_32b_differential_validator.validate_full_canonical_publication",
        lambda *a, **k: (
            events.append("publication")
            or SimpleNamespace(ok=False, errors=["incomplete"])
        ),
    )
    monkeypatch.setattr(
        cli, "load_and_validate_basis", lambda *a, **k: events.append("basis")
    )

    with pytest.raises(ValueError, match="publication"):
        cli.run_checkpoint(
            args=args,
            run_config=replace(config, result_root=result_root),
            checkpoint="step_050",
        )
    assert events == ["publication"]
    assert summary.read_text(encoding="utf-8") == "old"


def test_direct_32b_checkpoint_does_not_accept_publication_verified_bypass() -> None:
    assert (
        "publication_verified" not in inspect.signature(cli.run_checkpoint).parameters
    )


@pytest.mark.parametrize(
    ("dtype", "quantization"),
    [("float16", "nf4"), ("bfloat16", "none")],
)
def test_direct_32b_loader_rejects_noncanonical_identity(
    dtype: str, quantization: str, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_require_canonical_7b", lambda **kwargs: None)
    with pytest.raises(ValueError, match="32b requires bfloat16 dtype and NF4"):
        cli.load_model_for_run("olmo3-32b-think-sft", "sha", dtype, quantization)


def test_direct_32b_loader_gate_precedes_nf4_import(monkeypatch) -> None:
    import builtins

    imports: list[str] = []
    original_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        imports.append(name)
        return original_import(name, *args, **kwargs)

    def fail_gate(**kwargs):
        raise RuntimeError("7b incomplete")

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    monkeypatch.setattr(cli, "_require_canonical_7b", fail_gate)

    with pytest.raises(RuntimeError, match="7b incomplete"):
        cli.load_model_for_run("olmo3-32b-think-sft", "sha", "bfloat16", "nf4")

    assert "postdyn.quantized_model_loader" not in imports


def test_direct_32b_checkpoint_gate_precedes_publication_and_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        ["--scale", "32b", "--trajectory", "rlvr", "--checkpoints", "step_050"]
    )
    config = cli.resolve_run_config(args)
    result_root = tmp_path / "logs"
    summary = result_root / "checkpoints" / "step_050" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("old", encoding="utf-8")
    events: list[str] = []

    def fail_gate(**kwargs):
        events.append("preflight")
        raise RuntimeError("7b incomplete")

    monkeypatch.setattr(cli, "_require_canonical_7b", fail_gate)
    monkeypatch.setattr(
        "postdyn.think_32b_differential_validator.validate_full_canonical_publication",
        lambda *a, **k: events.append("publication"),
    )
    monkeypatch.setattr(
        cli, "load_and_validate_basis", lambda *a, **k: events.append("basis")
    )

    def loader(*args: str) -> tuple[object, ...]:
        events.append("loader")
        return ()

    with pytest.raises(RuntimeError, match="7b incomplete"):
        cli.run_checkpoint(
            args=args,
            run_config=replace(config, result_root=result_root),
            checkpoint="step_050",
            load_model=loader,
        )

    assert events == ["preflight"]
    assert summary.read_text(encoding="utf-8") == "old"


def test_generic_math_loader_rejects_32b_before_transformers_access(
    monkeypatch,
) -> None:
    transformers = __import__("sys").modules.get("transformers")
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    try:
        with pytest.raises(ValueError, match="generic MATH loader.*32B"):
            cli.load_model_and_tokenizer(
                "olmo3-32b-think-sft", "a" * 40, "bfloat16", "none"
            )
    finally:
        if transformers is not None:
            monkeypatch.setitem(__import__("sys").modules, "transformers", transformers)


def test_32b_condition_identity_binds_runtime_and_basis_hashes() -> None:
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar-hash",
        "tensor-hash",
        {},
        {"model": "model", "checkpoint": "step", "revision": "sha"},
    )
    identity = cli._condition_identity(
        model="model",
        checkpoint="step",
        revision="sha",
        condition="layer_3_U_pos",
        basis=basis,
        dataset=Path("data/math500.json"),
        max_new_tokens=2048,
        dtype="bfloat16",
        quantization="nf4",
        runtime_provenance={"loader": "load_olmo3_32b_think"},
    )

    generation = cast(dict[str, object], identity["generation"])
    runtime = cast(dict[str, object], identity["runtime_provenance"])
    basis_identity = cast(dict[str, object], identity["basis"])
    assert generation["quantization"] == "nf4"
    assert runtime == {"loader": "load_olmo3_32b_think"}
    assert basis_identity["sidecar_sha256"] == "sidecar-hash"
    assert basis_identity["tensor_sha256"] == "tensor-hash"


def test_display_tag_revision_fails_before_model_loading(monkeypatch) -> None:
    args = cli.parse_args(
        [
            "--trajectory",
            "rlvr",
            "--checkpoints",
            "step_0025",
            "--revision",
            "step_0025",
        ]
    )
    loaded = False

    def fail_if_loaded(*args: object, **kwargs: object):
        nonlocal loaded
        loaded = True
        raise AssertionError("model loading must not occur")

    monkeypatch.setattr(cli, "load_model_and_tokenizer", fail_if_loaded)
    assert cli.run(args) == 2
    assert loaded is False


def test_mixed_sft_artifact_root_fails_before_model_loading() -> None:
    sft_root = root_for_trajectory(FAMILY_THINK, SCALE_7B, "sft")
    args = cli.parse_args(
        [
            "--trajectory",
            "rlvr",
            "--artifact-root",
            str(sft_root),
            "--checkpoints",
            "step_0025",
        ]
    )

    with pytest.raises(ValueError, match="belongs to sft"):
        cli.resolve_run_config(args)


def test_revision_override_cannot_apply_to_multiple_checkpoints() -> None:
    args = cli.parse_args(
        [
            "--trajectory",
            "rlvr",
            "--checkpoints",
            "step_0025",
            "step_0175",
            "--revision",
            THINK_7B_RLVR_REVISIONS["step_0025"],
        ]
    )

    with pytest.raises(ValueError, match="only valid with one selected checkpoint"):
        cli.resolve_run_config(args)


@pytest.mark.parametrize(
    "option, values",
    [
        ("--checkpoints", ["step1000", "step1000"]),
        ("--layers", ["3", "3"]),
    ],
)
def test_duplicate_checkpoint_or_layer_arguments_are_rejected(option, values) -> None:
    args = cli.parse_args([option, *values])

    with pytest.raises(ValueError, match="duplicate"):
        cli.resolve_run_config(args)


def test_aggregate_rebuild_reads_all_authoritative_condition_directories(
    tmp_path: Path, monkeypatch
) -> None:
    expected = {"checkpoint": "step1000", "condition": "baseline", "n_processed": 50}
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *args, **kwargs: ([expected], []),
    )

    rebuilt = cli.rebuild_aggregate(
        tmp_path,
        trajectory="sft",
        model_key="olmo3-think-sft",
        selected_checkpoints=("step1000",),
        selected_layers=(),
    )

    assert rebuilt["conditions"] == [expected]


def test_rebuild_rejects_model_key_not_owned_by_trajectory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model_key"):
        cli.rebuild_aggregate(
            tmp_path,
            trajectory="sft",
            model_key="olmo3-think-rlvr",
            selected_checkpoints=("step1000",),
            selected_layers=(),
        )


def test_rebuild_claims_result_root_ownership_before_aggregate_write(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "collect_valid_conditions", lambda *a, **k: ([], []))
    original = cli.ensure_root_ownership

    def tracked(*args, **kwargs):
        calls.append("ownership")
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "ensure_root_ownership", tracked)
    cli.rebuild_aggregate(
        tmp_path,
        trajectory="sft",
        model_key="olmo3-think-sft",
        selected_checkpoints=("step1000",),
        selected_layers=(),
    )
    assert calls == ["ownership"]
    assert (tmp_path / ".trajectory_identity.json").is_file()


def test_32b_rebuild_passes_scale_to_model_free_validator(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, object] = {}

    def fake_collect(*args, **kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(cli, "collect_valid_conditions", fake_collect)
    monkeypatch.setattr(
        "postdyn.think_32b_differential_validator.validate_full_canonical_publication",
        lambda *args, **kwargs: SimpleNamespace(ok=True, errors=[]),
    )
    monkeypatch.setattr(
        "postdyn.cross_pipeline_integrity.require_canonical_7b",
        lambda **kwargs: None,
    )
    cli.rebuild_aggregate(
        tmp_path,
        trajectory="rlvr",
        model_key="olmo3-32b-think-rlvr",
        scale="32b",
        selected_checkpoints=("step_050",),
        selected_layers=(6,),
        quantization="nf4",
    )

    assert seen["scale"] == "32b"
    assert seen["selected_checkpoints"] == ("step_050",)
    assert seen["selected_layers"] == (6,)


def test_32b_rebuild_preflights_before_artifact_validation_or_side_effects(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    project_root = tmp_path / "project"

    def fail_preflight(**kwargs):
        events.append(f"preflight:{kwargs['project_root']}")
        raise RuntimeError("7b incomplete")

    monkeypatch.setattr(
        "postdyn.cross_pipeline_integrity.require_canonical_7b", fail_preflight
    )
    monkeypatch.setattr(
        "postdyn.think_32b_differential_validator.validate_full_canonical_publication",
        lambda *args, **kwargs: events.append("artifact-validation"),
    )
    monkeypatch.setattr(
        cli, "ensure_root_ownership", lambda *args, **kwargs: events.append("ownership")
    )
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *args, **kwargs: events.append("collect"),
    )

    with pytest.raises(RuntimeError, match="7b incomplete"):
        cli.rebuild_aggregate(
            tmp_path / "logs",
            trajectory="rlvr",
            model_key="olmo3-32b-think-rlvr",
            scale="32b",
            project_root=project_root,
            selected_checkpoints=("step_050",),
            selected_layers=(6,),
        )

    assert events == [f"preflight:{project_root}"]
    assert not (tmp_path / "logs").exists()


def test_rebuild_aggregate_forwards_explicit_artifact_and_project_roots(
    tmp_path: Path, monkeypatch
) -> None:
    seen: dict[str, object] = {}
    artifact_root = tmp_path / "artifacts"
    project_root = tmp_path / "project"

    def fake_collect(*args, **kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(cli, "collect_valid_conditions", fake_collect)
    cli.rebuild_aggregate(
        tmp_path / "logs",
        trajectory="rlvr",
        model_key="olmo3-think-rlvr",
        artifact_root=artifact_root,
        project_root=project_root,
        selected_checkpoints=("step_0025",),
        selected_layers=(),
    )

    assert seen["artifact_root"] == artifact_root
    assert seen["project_root"] == project_root

    seen.clear()
    cli.rebuild_aggregate(
        tmp_path / "results-defaulted",
        trajectory="rlvr",
        model_key="olmo3-think-rlvr",
        project_root=project_root,
        selected_checkpoints=("step_0025",),
        selected_layers=(),
    )

    assert seen["artifact_root"] == (
        project_root / "logs" / "think_7b_rlvr_differential_subspace"
    )
    assert seen["project_root"] == project_root


def test_fully_cached_subset_uses_hf_id_and_skips_model_loading(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        [
            "--trajectory",
            "sft",
            "--checkpoints",
            "step1000",
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
        ]
    )
    config = cli.resolve_run_config(args)
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar",
        "tensor",
        {"U_pos": object(), "U_neg": object()},
        {
            "model": OLMO3_VARIANTS[config.model_key].name,
            "checkpoint": "step1000",
            "revision": config.revision_for("step1000"),
        },
    )
    monkeypatch.setattr(cli, "load_and_validate_basis", lambda *a, **k: basis)
    seen_models: list[str] = []

    def cached_summary(*, model, **kwargs):
        seen_models.append(model)
        return {"n_processed": 50}

    monkeypatch.setattr(cli, "load_authoritative_summary", cached_summary)
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *args, **kwargs: (
            [
                {"checkpoint": "step1000", "condition": condition, "n_processed": 50}
                for condition in ("baseline", "layer_3_U_pos", "layer_3_U_neg")
            ],
            [],
        ),
    )

    def fail_if_loaded(*args: object, **kwargs: object):
        raise AssertionError("fully cached conditions must not load a model")

    records = cli.run_checkpoint(
        args=args,
        run_config=replace(config, result_root=tmp_path),
        checkpoint="step1000",
        load_model=fail_if_loaded,
    )

    assert len(records) == 3
    assert seen_models == [OLMO3_VARIANTS[config.model_key].hf_id] * 3


def test_generation_batch_size_canary_hooks_representative_ablation_and_forwards_size(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        [
            "--checkpoints",
            "step1000",
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
            "--force",
            "--generation-batch-size",
            "2",
        ]
    )
    config = cli.resolve_run_config(args)
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar",
        "tensor",
        {"U_pos": object(), "U_neg": object()},
        {
            "model": OLMO3_VARIANTS[config.model_key].name,
            "checkpoint": "step1000",
            "revision": config.revision_for("step1000"),
        },
    )
    monkeypatch.setattr(cli, "load_and_validate_basis", lambda *a, **k: basis)
    monkeypatch.setattr(
        cli,
        "load_first50",
        lambda path: (
            [
                SimpleNamespace(problem="prompt one"),
                SimpleNamespace(problem="prompt two"),
            ],
            "dataset-hash",
        ),
    )

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

    model = Model()
    tokenizer = object()
    generator = object()
    monkeypatch.setattr(cli, "GreedyGenerator", lambda **kwargs: generator)
    canary_calls = []
    monkeypatch.setattr(
        cli,
        "compare_singleton_and_batch_token_ids",
        lambda *args, **kwargs: canary_calls.append(args[3]) or True,
    )
    hook_calls = []
    monkeypatch.setattr(
        cli,
        "residual_stream_ablation",
        lambda model, *, layer, basis: (
            hook_calls.append((layer, basis)) or nullcontext()
        ),
    )
    evaluations = []
    monkeypatch.setattr(
        cli,
        "evaluate_first50",
        lambda **kwargs: (
            evaluations.append(kwargs)
            or SimpleNamespace(to_dict=lambda: {"n_processed": 50})
        ),
    )
    records = [
        {"checkpoint": "step1000", "condition": condition}
        for condition in ("baseline", "layer_3_U_pos", "layer_3_U_neg")
    ]
    monkeypatch.setattr(cli, "collect_valid_conditions", lambda *a, **k: (records, []))

    result = cli.run_checkpoint(
        args=args,
        run_config=replace(config, result_root=tmp_path),
        checkpoint="step1000",
        load_model=lambda *a: (model, tokenizer),
    )

    assert result == records
    assert canary_calls == [["prompt one", "prompt two"]] * 2
    assert hook_calls[0] == (3, basis.tensors["U_pos"])
    assert [call["batch_size"] for call in evaluations] == [2, 2, 2]
    assert "generation_batch_size" not in cli._condition_identity(
        model="model",
        checkpoint="step1000",
        revision="revision",
        condition="baseline",
        basis=None,
        dataset=Path("dataset"),
        max_new_tokens=2048,
        dtype="bfloat16",
        quantization="none",
    )


def test_generation_batch_size_canary_mismatch_falls_back_to_singletons(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        [
            "--checkpoints",
            "step1000",
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
            "--force",
            "--generation-batch-size",
            "2",
        ]
    )
    config = cli.resolve_run_config(args)
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar",
        "tensor",
        {"U_pos": object(), "U_neg": object()},
        {
            "model": OLMO3_VARIANTS[config.model_key].name,
            "checkpoint": "step1000",
            "revision": config.revision_for("step1000"),
        },
    )
    monkeypatch.setattr(cli, "load_and_validate_basis", lambda *a, **k: basis)
    monkeypatch.setattr(
        cli,
        "load_first50",
        lambda path: (
            [SimpleNamespace(problem="one"), SimpleNamespace(problem="two")],
            "hash",
        ),
    )
    model = SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cpu")
        )
    )
    monkeypatch.setattr(cli, "GreedyGenerator", lambda **kwargs: object())
    monkeypatch.setattr(
        cli, "compare_singleton_and_batch_token_ids", lambda *a, **k: False
    )
    evaluations = []
    monkeypatch.setattr(cli, "residual_stream_ablation", lambda *a, **k: nullcontext())
    monkeypatch.setattr(
        cli,
        "evaluate_first50",
        lambda **kwargs: (
            evaluations.append(kwargs) or SimpleNamespace(to_dict=lambda: {})
        ),
    )
    monkeypatch.setattr(
        cli,
        "collect_valid_conditions",
        lambda *a, **k: ([{"checkpoint": "step1000"}], []),
    )
    cli.run_checkpoint(
        args=args,
        run_config=replace(config, result_root=tmp_path),
        checkpoint="step1000",
        load_model=lambda *a: (model, object()),
    )
    assert [call["batch_size"] for call in evaluations] == [1, 1, 1]


@pytest.mark.parametrize("batch_size", [0, 3])
def test_generation_batch_size_rejects_unsupported_value_before_model_loading(
    batch_size, monkeypatch
):
    with pytest.raises(SystemExit):
        cli.parse_args(["--generation-batch-size", str(batch_size)])


def test_cached_subset_writes_all_authoritative_checkpoint_records(
    tmp_path: Path, monkeypatch
) -> None:
    args = cli.parse_args(
        [
            "--trajectory",
            "sft",
            "--checkpoints",
            "step1000",
            "--layers",
            "3",
            "--result-root",
            str(tmp_path),
        ]
    )
    config = cli.resolve_run_config(args)
    basis = cli.BasisArtifact(
        3,
        "setup",
        "sidecar",
        "tensor",
        {"U_pos": object(), "U_neg": object()},
        {
            "model": OLMO3_VARIANTS[config.model_key].name,
            "checkpoint": "step1000",
            "revision": config.revision_for("step1000"),
        },
    )
    monkeypatch.setattr(cli, "load_and_validate_basis", lambda *a, **k: basis)
    monkeypatch.setattr(
        cli,
        "load_authoritative_summary",
        lambda **kwargs: {"n_processed": 50},
    )
    authoritative = [
        {"checkpoint": "step1000", "condition": condition, "n_processed": 50}
        for condition in (
            "baseline",
            "layer_1_U_pos",
            "layer_1_U_neg",
            "layer_3_U_pos",
            "layer_3_U_neg",
        )
    ]
    monkeypatch.setattr(
        cli, "collect_valid_conditions", lambda *a, **k: (authoritative, [])
    )

    records = cli.run_checkpoint(
        args=args,
        run_config=replace(config, result_root=tmp_path),
        checkpoint="step1000",
        load_model=lambda *a: (_ for _ in ()).throw(
            AssertionError("fully cached conditions must not load a model")
        ),
    )

    assert records == authoritative
    assert (
        json.loads(
            (tmp_path / "checkpoints" / "step1000" / "summary.json").read_text()
        )["conditions"]
        == authoritative
    )
