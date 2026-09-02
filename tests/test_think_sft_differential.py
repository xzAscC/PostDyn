"""Tests for signed Think-SFT differential subspaces."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from postdyn.differential_subspace import (
    compute_differential_subspace,
    compute_signed_differential_subspace,
    signed_subspace_stability,
)
from postdyn.think_sft_differential_experiment import (
    CONCEPT_PAIRS,
    FAMILY_THINK,
    SCALE_7B,
    SCALE_32B,
    checkpoints_for_scale,
    covariance_n_samples,
    layers_for_scale,
    model_config,
    sft_model_key,
    available_trajectories,
    revision_for_checkpoint,
    root_for_trajectory,
    trajectory_config,
    ensure_root_ownership,
    fixed_point_configs,
    canonical_extraction_protocol,
    extraction_protocol_payload,
    validate_extraction_protocol,
    validate_extraction_root_not_other_trajectory,
)


def test_custom_empty_root_claim_has_exact_ownership(tmp_path):
    config = trajectory_config(trajectory="rlvr")
    ensure_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="rlvr",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="extraction",
    )
    assert json.loads((tmp_path / ".trajectory_identity.json").read_text()) == {
        "scale": "7b",
        "trajectory": "rlvr",
        "model_key": "olmo3-think-rlvr",
        "checkpoints": list(config.checkpoints),
        "revisions": [
            [checkpoint, config.revisions[checkpoint]]
            for checkpoint in config.checkpoints
        ],
        "purpose": "extraction",
    }


def test_custom_nonempty_unmarked_root_is_rejected_before_claim(tmp_path):
    (tmp_path / "existing").write_text("sentinel")
    config = trajectory_config(trajectory="sft")
    with pytest.raises(ValueError, match="non-empty.*unmarked"):
        ensure_root_ownership(
            tmp_path,
            family=FAMILY_THINK,
            scale=SCALE_7B,
            trajectory="sft",
            model_key=config.model_key,
            checkpoints=list(config.checkpoints),
            revisions=dict(config.revisions),
            purpose="extraction",
        )
    assert not (tmp_path / ".trajectory_identity.json").exists()


def test_custom_mismatched_marker_is_rejected(tmp_path):
    config = trajectory_config(trajectory="sft")
    ensure_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="sft",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="extraction",
    )
    marker = tmp_path / ".trajectory_identity.json"
    marker.write_text(marker.read_text().replace('"sft"', '"rlvr"'))
    with pytest.raises(ValueError, match="different ownership"):
        ensure_root_ownership(
            tmp_path,
            family=FAMILY_THINK,
            scale=SCALE_7B,
            trajectory="sft",
            model_key=config.model_key,
            checkpoints=list(config.checkpoints),
            revisions=dict(config.revisions),
            purpose="extraction",
        )


def test_custom_matching_marker_is_valid_resume(tmp_path):
    config = trajectory_config(trajectory="sft")
    ensure_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="sft",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="extraction",
    )
    marker = tmp_path / ".trajectory_identity.json"
    before = marker.read_bytes()
    ensure_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="sft",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="extraction",
    )
    assert marker.read_bytes() == before


def test_extraction_cannot_claim_other_trajectory_canonical_root(tmp_path):
    project_root = tmp_path / "project"
    foreign = root_for_trajectory(
        FAMILY_THINK, SCALE_32B, "rlvr", project_root=project_root
    )
    with pytest.raises(ValueError, match="belongs to rlvr"):
        validate_extraction_root_not_other_trajectory(
            foreign,
            family=FAMILY_THINK,
            scale=SCALE_32B,
            trajectory="sft_lr_1e-4",
            project_root=project_root,
        )


def test_read_only_root_validation_does_not_claim_empty_custom_root(tmp_path):
    from postdyn.think_sft_differential_experiment import validate_root_ownership

    config = trajectory_config(trajectory="sft")
    validate_root_ownership(
        tmp_path,
        family=FAMILY_THINK,
        scale=SCALE_7B,
        trajectory="sft",
        model_key=config.model_key,
        checkpoints=list(config.checkpoints),
        revisions=dict(config.revisions),
        purpose="extraction",
    )
    assert not (tmp_path / ".trajectory_identity.json").exists()


def test_experiment_config_7b():
    sft = sft_model_key(SCALE_7B)
    assert model_config(sft).hf_id == "allenai/Olmo-3-7B-Think-SFT"
    assert CONCEPT_PAIRS == (
        ("math_vs_wikitext", "math", "wikitext"),
        ("code_vs_wikitext", "code", "wikitext"),
        ("instruction_following_vs_wikitext", "instruction_following", "wikitext"),
        ("general_reasoning_vs_wikitext", "general_reasoning", "wikitext"),
        ("math_vs_code", "math", "code"),
        ("math_vs_instruction_following", "math", "instruction_following"),
        ("math_vs_general_reasoning", "math", "general_reasoning"),
    )
    assert covariance_n_samples(SCALE_7B) == 40960
    assert covariance_n_samples(SCALE_32B) == 51200
    with pytest.raises(ValueError, match="unknown scale"):
        covariance_n_samples("9b")
    assert layers_for_scale(SCALE_7B) == [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]
    assert checkpoints_for_scale(SCALE_7B) == [
        "step1000",
        "step6000",
        "step10000",
        "step15000",
        "step20000",
        "step24000",
        "step29000",
        "step34000",
        "step38000",
        "step43000",
    ]


def test_experiment_config_32b():
    assert model_config("olmo3-32b-think-sft").hf_id == "allenai/Olmo-3-32B-Think-SFT"
    assert model_config("olmo3-32b-think-sft").layers == 64
    assert len(layers_for_scale(SCALE_32B)) == 10
    with pytest.raises(ValueError, match="checkpoint schedule"):
        checkpoints_for_scale(SCALE_32B)


def test_memo_grid_and_fixed_point_configuration():
    from postdyn.think_sft_differential_experiment import CONCEPT_PAIRS

    assert len(CONCEPT_PAIRS) == 7
    assert {pair[2] for pair in CONCEPT_PAIRS if pair[0].endswith("wikitext")} == {
        "wikitext"
    }
    assert fixed_point_configs(SCALE_7B) == {
        "base": ("olmo3-base", "main"),
        "dpo": ("olmo3-think-dpo", "main"),
    }
    assert fixed_point_configs(SCALE_32B) == {}


def test_revision_resolution_is_immutable_and_fail_closed():
    from scripts import run_think_sft_differential_subspace as runner

    class Info:
        sha = "A" * 40

    class Api:
        def model_info(self, model_id, *, revision):
            assert model_id == "model"
            assert revision == "main"
            return Info()

    assert runner.resolve_model_revision("model", "main", api_factory=Api) == "a" * 40
    with pytest.raises(ValueError, match="no immutable commit SHA"):
        runner.resolve_model_revision(
            "model",
            "main",
            api_factory=lambda: type(
                "Api", (), {"model_info": lambda self, *_args, **_kwargs: object()}
            )(),
        )


def test_prepare_domain_prompts_uses_streaming_memo_selection(monkeypatch, tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    calls = []
    monkeypatch.setattr(
        runner,
        "resolve_hub_dataset_revision",
        lambda repo_id, revision: "a" * 40,
    )

    def fake_selection(domain, n_samples, *, seed, prefer_local, allow_short=False):
        calls.append((domain, n_samples, seed, prefer_local, allow_short))
        source = runner._expected_prompt_source(domain)
        return SimpleNamespace(
            source=source,
            as_list=lambda: [f"{domain}-{index}" for index in range(n_samples)],
        )

    monkeypatch.setattr(runner, "load_domain_prompt_selection", fake_selection)
    prompts = runner.prepare_domain_prompts(
        tmp_path,
        n_samples=2,
        seed=7,
        allow_hf=True,
        max_seq_len=2048,
        use_chat_template=False,
    )
    assert set(prompts) == {
        "math",
        "code",
        "instruction_following",
        "general_reasoning",
        "wikitext",
    }
    assert len(calls) == 5
    assert all(call[3] is False for call in calls)
    assert all(call[4] is True for call in calls)
    wiki = json.loads((tmp_path / "prompts/wikitext.json").read_text())
    assert wiki["source"]["config"] == "wikitext-103-raw-v1"


def test_exact_pinned_rlvr_and_32b_sft_trajectories():
    seven = trajectory_config(trajectory="rlvr")
    assert list(seven.checkpoints) == [
        "step_0025",
        "step_0175",
        "step_0325",
        "step_0475",
        "step_0625",
        "step_0775",
        "step_0925",
        "step_1075",
        "step_1225",
        "step_1375",
    ]
    assert revision_for_checkpoint("think", "7b", "rlvr", "step_0025") == (
        "817b9d38d9cf462c90fcf56cad08563abd0054a0"
    )
    assert revision_for_checkpoint("think", "32b", "sft_lr_1e-4", "step9000") == (
        "9807a1f8c641785bfc1e2e5aa718efc6801a4363"
    )
    assert available_trajectories("think", "32b") == (
        "sft_lr_1e-4",
        "sft_lr_5e-5",
        "rlvr",
    )
    for trajectory in available_trajectories("think", "32b"):
        config = trajectory_config("think", "32b", trajectory)
        assert len(config.checkpoints) == 10
        assert len(config.revisions) == 10
        assert len(set(config.revisions.values())) == 10
    assert root_for_trajectory("think", "32b", "sft_lr_1e-4") != root_for_trajectory(
        "think", "32b", "sft_lr_5e-5"
    )
    assert root_for_trajectory("think", "7b", "sft") != root_for_trajectory(
        "think", "7b", "rlvr"
    )


def test_checkpoint_worker_is_private_and_routes_pinned_model_and_revision(
    monkeypatch, tmp_path
):
    from scripts import run_think_sft_differential_subspace as runner

    assert not hasattr(runner, "run_checkpoint")
    assert callable(runner._run_checkpoint)

    calls = []
    fake_sub = SimpleNamespace(
        concept="math_vs_text", k_pos=1, k_neg=1, d_eff_pos=1.0, d_eff_neg=1.0
    )
    monkeypatch.setattr(runner, "model_complete", lambda *args, **kwargs: False)
    monkeypatch.setattr(runner, "preflight_tokenizer_prompts", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "extract_raw_layer_activations",
        lambda *args, **kwargs: {3: torch.ones(2, 2)},
    )
    monkeypatch.setattr(
        runner, "compute_signed_differential_subspace", lambda *args, **kwargs: fake_sub
    )
    monkeypatch.setattr(runner, "signed_subspace_to_serializable", lambda sub: {})
    monkeypatch.setattr(runner, "save_signed_subspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_atomic_write_json", lambda *args, **kwargs: None)

    def fake_loader(cfg, revision):
        calls.append((cfg.name, cfg.hf_id, revision))
        return object(), object()

    runner._run_checkpoint(
        "olmo3-think-rlvr",
        "step_0025",
        revision_for_checkpoint("think", "7b", "rlvr", "step_0025"),
        root=tmp_path,
        scale="7b",
        layers=[3],
        n_samples=2,
        max_seq_len=16,
        tau=0.95,
        domain_prompts={
            domain: [f"{domain}1", f"{domain}2"]
            for domain in (
                "math",
                "code",
                "instruction_following",
                "general_reasoning",
                "wikitext",
            )
        },
        setup_sig="fake",
        model_loader=fake_loader,
    )
    assert calls == [
        (
            "olmo3-think-rlvr",
            "allenai/Olmo-3-7B-Think",
            "817b9d38d9cf462c90fcf56cad08563abd0054a0",
        )
    ]


def test_signed_subspace_recovers_both_axes():
    torch.manual_seed(1)
    n, d = 40, 8
    base = torch.randn(n, d) * 0.1
    h_c = base.clone()
    h_c[:, 0] += torch.randn(n) * 3.0
    h_r = base.clone()
    h_r[:, 1] += torch.randn(n) * 3.0

    signed = compute_signed_differential_subspace(
        h_c, h_r, concept="math_vs_text", tau=0.95
    )
    pos_only = compute_differential_subspace(h_c, h_r, concept="math_vs_text", tau=0.95)
    assert signed.k_pos == pos_only.k
    assert torch.allclose(signed.u_pos, pos_only.u)
    assert signed.k_pos >= 1 and signed.k_neg >= 1
    lead_pos = signed.u_pos[:, 0].abs()
    lead_neg = signed.u_neg[:, 0].abs()
    assert lead_pos[0] > lead_pos[1]
    assert lead_neg[1] > lead_neg[0]
    assert float(signed.eigenvalues_neg.min()) > 0


def test_signed_stability_identical_is_one():
    torch.manual_seed(2)
    h_c = torch.randn(20, 6)
    h_r = torch.randn(20, 6)
    a = compute_signed_differential_subspace(h_c, h_r, concept="math_vs_text")
    stab = signed_subspace_stability(a, a)
    assert stab["pos"]["subsim"] == pytest.approx(1.0, abs=1e-5)
    assert stab["neg"]["subsim"] == pytest.approx(1.0, abs=1e-5)


def test_32b_refused_without_flag():
    from scripts.run_think_sft_differential_subspace import validate_scale

    with pytest.raises(ValueError, match="32B"):
        validate_scale("32b", allow_32b=False)
    validate_scale("7b", allow_32b=False)


def test_32b_fixed_points_are_reported_unavailable_without_substitution(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    with pytest.raises(ValueError, match="fixed points are unavailable"):
        runner.main(
            [
                "--dry-run",
                "--scale",
                "32b",
                "--allow-32b",
                "--trajectory",
                "rlvr",
                "--output",
                str(tmp_path),
                "--checkpoints",
                "step_050",
                "--layers",
                "6",
                "--fixed-points",
                "base",
            ]
        )


def test_canonical_extraction_protocol_payload_is_exact():
    assert canonical_extraction_protocol(SCALE_7B) == extraction_protocol_payload(
        n_samples=40960,
        tau=0.95,
        max_seq_len=2048,
        use_chat_template=False,
        extraction_contract="raw_prompt_final_attention_token_v1",
        dtype="bfloat16",
        signed=True,
    )
    assert canonical_extraction_protocol(SCALE_32B) == extraction_protocol_payload(
        n_samples=51200,
        tau=0.95,
        max_seq_len=2048,
        use_chat_template=False,
        extraction_contract="raw_prompt_final_attention_token_v1",
        dtype="bfloat16",
        signed=True,
    )
    validate_extraction_protocol(
        canonical_extraction_protocol(SCALE_7B), scale=SCALE_7B
    )
    validate_extraction_protocol(
        canonical_extraction_protocol(SCALE_32B), scale=SCALE_32B
    )


@pytest.mark.parametrize(
    "field",
    [
        "n_samples",
        "tau",
        "max_seq_len",
        "use_chat_template",
        "extraction_contract",
        "dtype",
        "signed",
    ],
)
def test_canonical_extraction_protocol_rejects_mutated_or_missing_field(field):
    payload = canonical_extraction_protocol(SCALE_7B)
    payload.pop(field)
    with pytest.raises(ValueError, match="canonical extraction protocol"):
        validate_extraction_protocol(payload)

    payload = canonical_extraction_protocol(SCALE_7B)
    payload[field] = {
        "n_samples": 999,
        "tau": 0.9,
        "max_seq_len": 1024,
        "use_chat_template": True,
        "extraction_contract": "tampered",
        "dtype": "float16",
        "signed": False,
    }[field]
    with pytest.raises(ValueError, match="canonical extraction protocol"):
        validate_extraction_protocol(payload)


def test_32b_invalid_extraction_protocol_rejects_before_loader(monkeypatch, tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    called = []
    monkeypatch.setattr(runner, "_require_canonical_7b", lambda **kwargs: None)
    monkeypatch.setattr(runner, "model_complete", lambda *args, **kwargs: False)

    def loader(*args, **kwargs):
        called.append((args, kwargs))
        return object(), object()

    with pytest.raises(ValueError, match="canonical extraction protocol"):
        runner._run_checkpoint(
            "olmo3-32b-think-sft",
            "step1000",
            "a" * 40,
            root=tmp_path,
            scale="32b",
            layers=[6],
            n_samples=999,
            max_seq_len=2048,
            tau=0.95,
            domain_prompts={
                domain: []
                for domain in (
                    "math",
                    "code",
                    "instruction_following",
                    "general_reasoning",
                    "wikitext",
                )
            },
            setup_sig="fake",
            model_loader=loader,
        )
    assert called == []


@pytest.mark.parametrize(
    ("model_key", "scale"),
    [
        ("olmo3-32b-think-sft", SCALE_7B),
        ("olmo3-think-sft", SCALE_32B),
    ],
)
def test_checkpoint_rejects_model_scale_mismatch_before_side_effects(
    monkeypatch, tmp_path, model_key, scale
):
    from scripts import run_think_sft_differential_subspace as runner

    calls = []
    monkeypatch.setattr(
        runner,
        "model_config",
        lambda *args, **kwargs: pytest.fail("config lookup must be gated"),
    )
    monkeypatch.setattr(
        runner,
        "_require_canonical_7b",
        lambda **kwargs: calls.append(("preflight", kwargs)),
    )
    monkeypatch.setattr(
        runner,
        "model_complete",
        lambda *args, **kwargs: pytest.fail("completion checks must be gated"),
    )
    monkeypatch.setattr(
        runner,
        "validate_extraction_protocol",
        lambda *args, **kwargs: pytest.fail("validators must be gated"),
    )

    def loader(*args, **kwargs):
        calls.append(("loader", args, kwargs))
        raise AssertionError("injected loader must be gated")

    output = tmp_path / "uncreated-output"
    with pytest.raises(ValueError, match="does not belong to scale"):
        runner._run_checkpoint(
            model_key,
            "step1000",
            "a" * 40,
            root=output,
            scale=scale,
            layers=[6],
            n_samples=1,
            max_seq_len=16,
            tau=0.95,
            domain_prompts={"math": [], "text": []},
            setup_sig="fake",
            model_loader=loader,
            project_root=tmp_path,
        )

    assert calls == []
    assert not output.exists()


def test_direct_32b_checkpoint_preflight_precedes_skip_filesystem_and_loader(
    monkeypatch, tmp_path
):
    from scripts import run_think_sft_differential_subspace as runner
    from postdyn import cross_pipeline_integrity as integrity

    calls = []

    def fail_preflight(**kwargs):
        calls.append(("preflight", kwargs))
        raise integrity.Canonical7BPreflightError("incomplete upstream")

    monkeypatch.setattr(integrity, "require_canonical_7b_extraction", fail_preflight)
    monkeypatch.setattr(
        runner,
        "model_complete",
        lambda *args, **kwargs: pytest.fail("completion checks must be gated"),
    )
    monkeypatch.setattr(
        runner,
        "validate_extraction_protocol",
        lambda *args, **kwargs: pytest.fail("validators must be gated"),
    )

    def loader(*args, **kwargs):
        calls.append(("loader", args, kwargs))
        raise AssertionError("injected loader must be gated")

    output = tmp_path / "uncreated-output"
    with pytest.raises(
        integrity.Canonical7BPreflightError, match="incomplete upstream"
    ):
        runner._run_checkpoint(
            "olmo3-32b-think-sft",
            "step1000",
            "a" * 40,
            root=output,
            scale="32b",
            layers=[6],
            n_samples=1,
            max_seq_len=16,
            tau=0.95,
            domain_prompts={"math": [], "text": []},
            setup_sig="fake",
            model_loader=loader,
            project_root=tmp_path,
        )

    assert calls == [("preflight", {"project_root": tmp_path})]
    assert not output.exists()


def test_32b_allow_flag_selects_nf4_loader_without_gpu_size_gate(monkeypatch):
    from scripts import run_think_sft_differential_subspace as runner
    from postdyn import quantized_model_loader

    assert not hasattr(runner, "load_olmo3_32b_think")
    monkeypatch.setattr(
        quantized_model_loader,
        "load_olmo3_32b_think",
        lambda *args, **kwargs: None,
    )
    runner.validate_scale("32b", allow_32b=True)
    assert runner.build_model_loader("7b") is runner._load_model_and_tokenizer


def test_32b_loader_passes_immutable_revision_and_records_diagnostics(monkeypatch):
    from scripts import run_think_sft_differential_subspace as runner
    from postdyn.quantized_model_loader import LoadDiagnostics, LoadedQuantizedModel

    monkeypatch.setattr(runner, "_require_canonical_7b", lambda **kwargs: None)
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>")
    diagnostics = LoadDiagnostics(placement="gpu-only", device_map={"": "cuda:0"})
    calls = []

    def fake_load(model_id, *, revision):
        calls.append((model_id, revision))
        return LoadedQuantizedModel(object(), tokenizer, diagnostics)

    monkeypatch.setattr(
        "postdyn.quantized_model_loader.load_olmo3_32b_think", fake_load
    )
    runtime = {}
    load = runner.build_model_loader("32b", runtime_provenance=runtime)
    cfg = runner.model_config("olmo3-32b-think-sft")
    model, loaded_tokenizer = load(cfg, "a" * 40)
    assert model is not None
    assert loaded_tokenizer.pad_token == "<eos>"
    assert calls == [(cfg.hf_id, "a" * 40)]
    assert runtime["placement"] == "gpu-only"


def test_32b_loader_preflight_blocks_nf4_import_and_call(monkeypatch, tmp_path):
    from scripts import run_think_sft_differential_subspace as runner
    from postdyn import cross_pipeline_integrity as integrity

    gate_calls = []

    def fail_preflight(**kwargs):
        gate_calls.append(kwargs)
        raise integrity.Canonical7BPreflightError("incomplete upstream")

    monkeypatch.setattr(integrity, "require_canonical_7b_extraction", fail_preflight)
    monkeypatch.setattr(
        "postdyn.quantized_model_loader.load_olmo3_32b_think",
        lambda *args, **kwargs: pytest.fail(
            "NF4 loader must not be imported or called"
        ),
    )

    loader = runner.build_model_loader("32b", project_root=tmp_path)
    cfg = runner.model_config("olmo3-32b-think-sft")
    with pytest.raises(
        integrity.Canonical7BPreflightError, match="incomplete upstream"
    ):
        loader(cfg, "a" * 40)

    assert gate_calls == [{"project_root": tmp_path}]


def test_32b_loader_rejects_missing_trajectory_revision(monkeypatch):
    from scripts import run_think_sft_differential_subspace as runner

    monkeypatch.setattr(runner, "_require_canonical_7b", lambda **kwargs: None)
    loader = runner.build_model_loader("32b")
    cfg = runner.model_config("olmo3-32b-think-sft")
    called = []
    monkeypatch.setattr(
        "postdyn.quantized_model_loader.load_olmo3_32b_think",
        lambda *args, **kwargs: called.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="pinned revision"):
        loader(cfg, None)
    assert called == []


def test_32b_setup_signature_binds_loader_provenance():
    from scripts import run_think_sft_differential_subspace as runner

    common: dict[str, Any] = dict(
        pairs=[("math_vs_text", "math", "text")],
        model_keys=["olmo3-32b-think-sft"],
        checkpoints=["step1000"],
        layers=[6],
        model_ids={"olmo3-32b-think-sft": "model"},
        dataset_sources={},
        prompt_fingerprints={},
        n_samples=1,
        tau=0.95,
        max_seq_len=16,
        use_chat_template=False,
        seed=1,
        extraction_contract="raw_prompt_final_attention_token_v1",
        dtype="bfloat16",
        signed=True,
        scale="32b",
    )
    plain = runner.setup_signature(**common)
    nf4 = runner.setup_signature(**common, loader_provenance=runner.NF4_PROVENANCE)
    assert plain != nf4


def test_activation_inputs_follow_dispatched_model_parameter_device():
    from scripts.run_math_differential_subspace import extract_raw_layer_activations

    moved = []

    class Input:
        def __init__(self, value):
            self.value = value

        @property
        def shape(self):
            return self.value.shape

        def to(self, device):
            moved.append(device)
            return self.value

    class Model:
        hf_device_map = {"model.embed_tokens": "cuda:0"}

        def parameters(self):
            yield SimpleNamespace(device=torch.device("cpu"))

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("meta")))

        def __call__(self, **kwargs):
            assert all(
                isinstance(value, torch.Tensor)
                for key, value in kwargs.items()
                if key != "output_hidden_states"
            )
            return SimpleNamespace(
                hidden_states=[torch.ones(1, 2, 3) for _ in range(2)]
            )

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": Input(torch.tensor([[4, 5]])),
                "attention_mask": Input(torch.tensor([[1, 1]])),
            }

    result = extract_raw_layer_activations(Model(), Tokenizer(), ["raw"], [0])
    assert moved == ["cuda:0", "cuda:0"]
    assert result[0].shape == (1, 3)


class _PaddedTokenizer:
    """Tokenizes batches with right padding; token value encodes the text id."""

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        seqs = []
        for text in texts:
            text_id = int(text.split("-")[0])
            length = int(text.split("-")[1])
            seqs.append([1000 + text_id] * length)
        max_len = max(len(seq) for seq in seqs)
        input_ids = torch.zeros(len(seqs), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for row, seq in enumerate(seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq)
            attention_mask[row, : len(seq)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _PositionalModel:
    """Hidden state at (row, position, layer) depends only on token id and layer,
    so batched and single-text forwards must agree exactly."""

    def __init__(self, n_layers, calls=None):
        self.n_layers = n_layers
        self.calls = calls if calls is not None else []

    def parameters(self):
        yield SimpleNamespace(device=torch.device("cpu"))

    def __call__(self, input_ids=None, attention_mask=None, **_kwargs):
        self.calls.append(tuple(input_ids.shape))
        hidden = []
        for layer in range(self.n_layers + 1):
            per_pos = (input_ids.float() + 0.5 * layer) / 1000.0
            hidden.append(per_pos.unsqueeze(-1).expand(-1, -1, 4))
        return SimpleNamespace(hidden_states=hidden)


def test_batched_extraction_matches_single_text_results():
    from scripts.run_math_differential_subspace import extract_raw_layer_activations

    texts = [f"{i}-{length}" for i, length in enumerate([7, 3, 9, 5, 4])]
    layers = [0, 2]
    single_model = _PositionalModel(n_layers=3)
    reference = extract_raw_layer_activations(
        single_model, _PaddedTokenizer(), list(texts), layers, token_budget=1
    )
    batched_model = _PositionalModel(n_layers=3)
    batched = extract_raw_layer_activations(
        batched_model, _PaddedTokenizer(), texts, layers, token_budget=8
    )
    assert len(batched_model.calls) < len(single_model.calls)
    for layer in layers:
        assert torch.allclose(reference[layer], batched[layer], atol=1e-6)


def test_batched_extraction_keeps_original_order():
    from scripts.run_math_differential_subspace import extract_raw_layer_activations

    texts = [f"{i}-{length}" for i, length in enumerate([9, 2, 6, 3])]
    result = extract_raw_layer_activations(
        _PositionalModel(n_layers=2), _PaddedTokenizer(), texts, [1], token_budget=3
    )
    for index, text in enumerate(texts):
        text_id = int(text.split("-")[0])
        expected = (1000 + text_id + 0.5 * 2) / 1000.0
        assert torch.allclose(result[1][index], torch.full((4,), expected))


def test_token_budget_caps_batch_shape():
    from scripts.run_math_differential_subspace import extract_raw_layer_activations

    texts = [f"{i}-5" for i in range(6)]
    model = _PositionalModel(n_layers=1)
    extract_raw_layer_activations(
        model, _PaddedTokenizer(), texts, [0], token_budget=10
    )
    assert model.calls
    assert all(rows <= 2 for rows, _seq_len in model.calls)


class _HookedLayer:
    def __init__(self, layer_index):
        self.layer_index = layer_index
        self._hooks = []

    def register_forward_hook(self, hook):
        self._hooks.append(hook)
        return SimpleNamespace(remove=lambda: None)

    def fire(self, value):
        for hook in self._hooks:
            hook(self, (), value)


class _HookedModel:
    """Layer output at (row, position) depends only on the token id and layer."""

    def __init__(self, n_layers=3, calls=None):
        self.n_layers = n_layers
        self.calls = calls if calls is not None else []
        self.layers = [_HookedLayer(i) for i in range(n_layers)]
        self.config = SimpleNamespace(use_cache=True)

    def named_modules(self):
        for index, layer in enumerate(self.layers):
            yield f"model.layers.{index}", layer

    def parameters(self):
        yield SimpleNamespace(device=torch.device("cpu"))

    def __call__(self, input_ids=None, **_kwargs):
        self.calls.append(tuple(input_ids.shape))
        for index, layer in enumerate(self.layers):
            per_pos = (input_ids.float() + 0.5 * index) / 1000.0
            layer.fire(per_pos.unsqueeze(-1).expand(-1, -1, 4))


def test_hooked_batched_extraction_matches_single_text_results():
    from scripts import run_think_sft_differential_subspace as runner

    texts = [f"{i}-{length}" for i, length in enumerate([7, 3, 9, 5, 4])]
    layers = [0, 2]
    single = runner.extract_raw_layer_activations(
        _HookedModel(), _PaddedTokenizer(), list(texts), layers, token_budget=1
    )
    batched_model = _HookedModel()
    batched = runner.extract_raw_layer_activations(
        batched_model, _PaddedTokenizer(), texts, layers, token_budget=8
    )
    assert len(batched_model.calls) < 5
    for layer in layers:
        assert torch.allclose(single[layer], batched[layer], atol=1e-6)


def test_hooked_batched_extraction_keeps_original_order():
    from scripts import run_think_sft_differential_subspace as runner

    texts = [f"{i}-{length}" for i, length in enumerate([9, 2, 6, 3])]
    result = runner.extract_raw_layer_activations(
        _HookedModel(), _PaddedTokenizer(), texts, [1], token_budget=8
    )
    for index, text in enumerate(texts):
        text_id = int(text.split("-")[0])
        expected = (1000 + text_id + 0.5 * 1) / 1000.0
        assert torch.allclose(result[1][index], torch.full((4,), expected))


def test_concepts_flag_filters_global_pairs(monkeypatch):
    from scripts import run_think_sft_differential_subspace as runner

    monkeypatch.setattr(
        runner,
        "CONCEPT_PAIRS",
        (("math_vs_wikitext", "math", "wikitext"), ("math_vs_code", "math", "code")),
    )
    runner.apply_concept_filter("math_vs_wikitext")
    assert runner.CONCEPT_PAIRS == (("math_vs_wikitext", "math", "wikitext"),)


def test_concepts_flag_rejects_unknown_names(monkeypatch):
    from scripts import run_think_sft_differential_subspace as runner

    monkeypatch.setattr(
        runner,
        "CONCEPT_PAIRS",
        (("math_vs_wikitext", "math", "wikitext"),),
    )
    with pytest.raises(ValueError, match="unknown concept"):
        runner.apply_concept_filter("math_vs_greek")


def test_skip_7b_gate_flag_bypasses_canonical_preflight(monkeypatch):
    from postdyn import cross_pipeline_integrity as integrity
    from scripts import run_think_sft_differential_subspace as runner

    def fail(**_kwargs):
        raise AssertionError("canonical 7B gate must not run when skipped")

    monkeypatch.setattr(integrity, "require_canonical_7b_extraction", fail)
    runner.set_skip_7b_gate(True)
    try:
        runner._require_canonical_7b(project_root=None)
    finally:
        runner.set_skip_7b_gate(False)
    with pytest.raises(AssertionError, match="must not run"):
        runner._require_canonical_7b(project_root=None)


def test_think_subspace_complete_rejects_checkpoint_sidecar_mismatch(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    sub = compute_signed_differential_subspace(
        torch.randn(4, 3), torch.randn(4, 3), concept="math_vs_text"
    )
    runner.save_signed_subspace(
        tmp_path, "model", "step1000", 3, sub, "sig", "rev", save_tensors=True
    )
    path = runner._u_paths(tmp_path, "model", "step1000", 3, "math_vs_text")[1]
    data = json.loads(path.read_text())
    data["checkpoint"] = "step6000"
    path.write_text(json.dumps(data))
    assert not runner.subspace_complete(
        tmp_path, "model", "step1000", 3, "math_vs_text", "sig", "rev"
    )


def test_save_load_signed_roundtrip(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    torch.manual_seed(0)
    sub = compute_signed_differential_subspace(
        torch.randn(8, 5), torch.randn(8, 5), concept="math_vs_text"
    )
    runner.save_signed_subspace(
        tmp_path,
        "olmo3-think-sft",
        "step1000",
        3,
        sub,
        "sig",
        "step1000",
        save_tensors=True,
    )
    assert runner.subspace_complete(
        tmp_path, "olmo3-think-sft", "step1000", 3, "math_vs_text", "sig", "step1000"
    )
    loaded = runner.load_signed_subspace(
        tmp_path, "olmo3-think-sft", "step1000", 3, "math_vs_text"
    )
    assert loaded.k_pos == sub.k_pos
    assert loaded.k_neg == sub.k_neg
    assert torch.allclose(loaded.u_pos, sub.u_pos)
    assert torch.allclose(loaded.u_neg, sub.u_neg)
    assert loaded.eigenvalues_signed is not None
    assert loaded.eigenvectors_signed is not None
    assert loaded.u_pos_full is not None
    assert torch.equal(
        cast(torch.Tensor, loaded.eigenvalues_signed),
        cast(torch.Tensor, sub.eigenvalues_signed),
    )
    assert torch.equal(
        cast(torch.Tensor, loaded.eigenvectors_signed),
        cast(torch.Tensor, sub.eigenvectors_signed),
    )
    assert torch.equal(
        cast(torch.Tensor, loaded.u_pos_full), cast(torch.Tensor, sub.u_pos_full)
    )
    assert loaded.energy_pos == pytest.approx(sub.energy_pos)
    assert loaded.energy_neg == pytest.approx(sub.energy_neg)
    assert loaded.frobenius_strength_pos == pytest.approx(sub.frobenius_strength_pos)
    assert loaded.frobenius_strength_neg == pytest.approx(sub.frobenius_strength_neg)
    assert loaded.r_pos == pytest.approx(sub.r_pos)
    assert loaded.emergence_pos == pytest.approx(sub.emergence_pos)
    assert loaded.emergence_neg == pytest.approx(sub.emergence_neg)
    sidecar = json.loads(
        runner._u_paths(tmp_path, "olmo3-think-sft", "step1000", 3, "math_vs_text")[
            1
        ].read_text()
    )
    assert "eigenvectors_signed" not in sidecar


def test_save_signed_subspace_defaults_to_json_only(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    sub = compute_signed_differential_subspace(
        torch.randn(8, 5), torch.randn(8, 5), concept="math_vs_text"
    )
    runner.save_signed_subspace(tmp_path, "m", "step1000", 3, sub, "sig", "rev")
    st_path, js_path = runner._u_paths(tmp_path, "m", "step1000", 3, "math_vs_text")
    assert not st_path.exists()
    meta = json.loads(js_path.read_text(encoding="utf-8"))
    assert meta["tensors_saved"] is False
    assert "k_pos" in meta and "d_eff_pos" in meta and "k_neg" in meta


def test_json_only_sidecar_skips_tensor_validation(tmp_path):
    from postdyn import think_sft_differential_validator as validator
    from scripts import run_think_sft_differential_subspace as runner

    sub = compute_signed_differential_subspace(
        torch.randn(8, 5), torch.randn(8, 5), concept="math_vs_text"
    )
    runner.save_signed_subspace(tmp_path, "m", "step1000", 3, sub, "sig", "rev")
    st_path, js_path = runner._u_paths(tmp_path, "m", "step1000", 3, "math_vs_text")
    assert (
        validator._basis_error(st_path, js_path, "m", "step1000", "step1000", "sig", 3)
        is None
    )


def test_load_signed_subspace_without_tensors_raises_clear_error(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    sub = compute_signed_differential_subspace(
        torch.randn(8, 5), torch.randn(8, 5), concept="math_vs_text"
    )
    runner.save_signed_subspace(tmp_path, "m", "step1000", 3, sub, "sig", "rev")
    with pytest.raises(FileNotFoundError, match="--save-tensors"):
        runner.load_signed_subspace(tmp_path, "m", "step1000", 3, "math_vs_text")


def test_finalize_stability_writes_both_signs(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    torch.manual_seed(4)
    sub = compute_signed_differential_subspace(
        torch.randn(10, 6), torch.randn(10, 6), concept="math_vs_text"
    )
    for ck in ("step1000", "step6000"):
        runner.save_signed_subspace(
            tmp_path, "olmo3-think-sft", ck, 3, sub, "sig", ck, save_tensors=True
        )
    final_root = tmp_path / "final_points"
    runner.save_signed_subspace(
        final_root,
        "olmo3-think-sft",
        "sft_main",
        3,
        sub,
        "final-sig",
        "final-rev",
        save_tensors=True,
    )
    out = runner.finalize_stability(
        tmp_path,
        "7b",
        ["step1000", "step6000"],
        [3],
        ["math_vs_text"],
        "sig",
        final_root=final_root,
        final_checkpoint="sft_main",
        final_setup_sig="final-sig",
        final_revision="final-rev",
    )
    vs = out["layers"]["3"]["pos"]["vs_reference"]["math_vs_text"]
    assert vs[0]["subsim"] == pytest.approx(1.0, abs=1e-5)
    saved = json.loads((tmp_path / "metrics" / "stability.json").read_text())
    assert saved["checkpoint_order"] == ["step1000", "step6000"]
    assert saved["reference"] == "step1000"


def test_residual_to_final_reports_observed_chance_and_excess_only():
    from scripts import run_think_sft_differential_subspace as runner
    from postdyn.differential_subspace import residual_to_later_subspace_overlap

    sub = compute_signed_differential_subspace(
        torch.randn(12, 5), torch.randn(12, 5), concept="math_vs_wikitext"
    )
    result = runner.residual_to_final_analysis(sub, sub)
    assert result == residual_to_later_subspace_overlap(sub, sub)
    assert set(result["pos"]) >= {
        "defined",
        "k_final",
        "d_res",
        "observed",
        "chance",
        "excess",
    }
    assert "l2" not in result["pos"]
    assert 0.0 <= result["pos"]["observed"] <= 1.0


def test_reference_robustness_and_histograms_are_per_concept(tmp_path):
    from scripts import run_think_sft_differential_subspace as runner

    concepts = [
        "math_vs_wikitext",
        "math_vs_code",
        "math_vs_instruction_following",
        "math_vs_general_reasoning",
    ]
    for checkpoint in ("step1000", "step6000"):
        for concept in concepts:
            sub = compute_signed_differential_subspace(
                torch.randn(10, 6), torch.randn(10, 6), concept=concept
            )
            runner.save_signed_subspace(
                tmp_path,
                "olmo3-think-sft",
                checkpoint,
                3,
                sub,
                "sig",
                checkpoint,
                save_tensors=True,
            )
            runner.save_signed_subspace(
                tmp_path / "final_points",
                "olmo3-think-sft",
                "sft_main",
                3,
                sub,
                "final-sig",
                "final-rev",
                save_tensors=True,
            )
    result = runner.finalize_stability(
        tmp_path,
        "7b",
        ["step1000", "step6000"],
        [3],
        concepts,
        "sig",
        final_root=tmp_path / "final_points",
        final_checkpoint="sft_main",
        final_setup_sig="final-sig",
        final_revision="final-rev",
    )
    assert set(result["histogram"]["3"]) == set(concepts)
    for metadata in result["histogram"]["3"].values():
        assert len(metadata["edges"]) == metadata["bins"] + 1
    assert set(result["layers"]["3"]["reference_robustness"]["step1000"]) == set(
        concepts[1:]
    )


def test_summary_core_metrics_exclude_spectra():
    from scripts import run_think_sft_differential_subspace as runner

    result = runner._summary_core_metrics(
        {
            "math_vs_wikitext": {
                "energy_pos": 1.0,
                "eigenvalues_signed": [1.0, -1.0],
                "eigenvectors_signed": [[1.0]],
            }
        }
    )
    assert result == {"math_vs_wikitext": {"energy_pos": 1.0}}
