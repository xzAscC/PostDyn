"""Unit tests for differential covariance subspaces and domain loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from postdyn.differential_subspace import (
    DifferentialSubspace,
    compute_differential_subspace,
    compute_signed_differential_subspace,
    compute_pair_metrics_at_checkpoint,
    compute_stability_trajectory,
    empirical_covariance,
    inter_subspace_relation,
    signed_subspace_to_serializable,
    participation_ratio,
    residual_basis,
    residual_to_later_subspace_overlap,
    select_k_from_positive_spectrum,
    subspace_stability,
)
from postdyn.domain_datasets import (
    DEFAULT_CONCEPT_PAIRS,
    DOLCI_HF_IDS,
    DOLCI_HF_REVISIONS,
    load_all_default_pairs,
    load_dolci_domain_prompts,
    load_domain_prompts,
)
from postdyn.math_differential_experiment import (
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_LAYERS,
    N_SAMPLES,
    CONCEPT_PAIRS,
    RESULTS_ROOT,
    TARGET_MODEL_KEY,
    model_for_checkpoint,
    revision_for_checkpoint,
)


def test_empirical_covariance_divisor_n():
    torch.manual_seed(0)
    h = torch.randn(5, 4)
    cov = empirical_covariance(h)
    x = h - h.mean(0, keepdim=True)
    expected = (x.T @ x) / 5.0
    assert torch.allclose(cov, expected, atol=1e-5)
    assert cov.shape == (4, 4)


def test_select_k_threshold():
    evals = torch.tensor([4.0, 3.0, 2.0, 1.0])
    # sq = 16,9,4,1 total=30; 0.95*30=28.5 → need 16+9+4=29 → K=3
    assert select_k_from_positive_spectrum(evals, tau=0.95) == 3
    assert select_k_from_positive_spectrum(evals, tau=1.0) == 4
    assert select_k_from_positive_spectrum(torch.tensor([]), tau=0.95) == 0


def test_compute_differential_subspace_positive_only():
    torch.manual_seed(1)
    # Concept group stretched along e0; ref stretched along e1.
    n, d = 40, 8
    base = torch.randn(n, d) * 0.1
    h_c = base.clone()
    h_c[:, 0] += torch.randn(n) * 3.0
    h_r = base.clone()
    h_r[:, 1] += torch.randn(n) * 3.0

    sub = compute_differential_subspace(h_c, h_r, concept="math_vs_text", tau=0.95)
    assert sub.concept == "math_vs_text"
    assert sub.u.shape[0] == d
    assert sub.k == sub.u.shape[1]
    assert sub.k >= 1
    assert sub.eigenvalues_pos.numel() >= sub.k
    assert float(sub.eigenvalues_pos.min()) > 0
    # Leading direction should align with e0 more than e1.
    lead = sub.u[:, 0].abs()
    assert lead[0] > lead[1]
    assert 0.0 <= sub.geometry_strength <= 1.0 + 1e-5
    assert sub.d_eff >= 1.0 - 1e-5


def _diagonal_covariance_samples(variances: list[float]) -> torch.Tensor:
    n = len(variances) + 1
    centered_identity = torch.eye(n) - torch.ones(n, n) / n
    basis, _ = torch.linalg.qr(centered_identity)
    return basis[:, : len(variances)] * torch.tensor(variances).sqrt() * n**0.5


def test_signed_metrics_use_all_spectrum_and_persist_full_eigenbasis():
    h_concept = _diagonal_covariance_samples([5.0, 2.0, 1.3, 1.0, 1.0])
    h_ref = _diagonal_covariance_samples([1.0, 1.0, 1.0, 1.0, 4.0])
    sub = compute_signed_differential_subspace(h_concept, h_ref, tau=0.95)

    positive = torch.tensor([4.0, 1.0, 0.3])
    negative = torch.tensor([3.0])
    assert sub.eigenvalues_pos.tolist() == pytest.approx(positive.tolist())
    assert sub.k_pos == 2
    assert sub.eigenvalues_neg.tolist() == pytest.approx(negative.tolist())
    assert sub.d_eff_pos == pytest.approx(participation_ratio(positive, 3))
    assert sub.d_eff_pos != pytest.approx(participation_ratio(positive, sub.k_pos))
    assert sub.energy_pos == pytest.approx(float(positive.square().sum()))
    assert sub.frobenius_strength_pos == pytest.approx(
        float(positive.square().sum().sqrt())
    )
    assert sub.r_pos == pytest.approx(positive.square().sum().item() / 26.09)
    assert sub.u_pos.shape == (5, 2)
    assert sub.u_pos_full is not None and sub.u_pos_full.shape == (5, 3)
    assert residual_basis(sub).shape == (5, 2)
    assert sub.eigenvalues_signed is not None
    assert sub.eigenvalues_signed.tolist() == pytest.approx([4.0, 1.0, 0.3, 0.0, -3.0])
    metadata = signed_subspace_to_serializable(sub)
    assert metadata["u_pos_full_shape"] == [5, 3]
    assert "eigenvectors_signed" not in metadata
    assert "u_pos_full" not in metadata
    assert "u_neg_full" not in metadata


def test_strict_sign_metrics_keep_tiny_nonzero_all_sign_spectra():
    tiny = 1e-8
    h_zero = torch.zeros(4, 3)
    h_positive = _diagonal_covariance_samples([tiny, tiny, tiny])

    positive = compute_signed_differential_subspace(h_positive, h_zero, tau=0.95)
    assert positive.eigenvalues_pos.numel() == 3
    assert positive.k_pos == 3
    assert positive.d_eff_pos == pytest.approx(3.0, rel=1e-5)
    assert positive.energy_pos == pytest.approx(3 * tiny**2, rel=1e-5)
    assert positive.frobenius_strength_pos == pytest.approx(3**0.5 * tiny, rel=1e-5)
    assert positive.geometry_strength_pos == pytest.approx(1.0)
    assert positive.k_neg == 0

    negative = compute_signed_differential_subspace(h_zero, h_positive, tau=0.95)
    assert negative.eigenvalues_neg.numel() == 3
    assert negative.k_neg == 3
    assert negative.d_eff_neg == pytest.approx(3.0, rel=1e-5)
    assert negative.energy_neg == pytest.approx(3 * tiny**2, rel=1e-5)
    assert negative.frobenius_strength_neg == pytest.approx(3**0.5 * tiny, rel=1e-5)
    assert negative.geometry_strength_neg == pytest.approx(1.0)
    assert negative.k_pos == 0


def test_residual_to_later_subspace_overlap_exact_formula():
    current = compute_signed_differential_subspace(
        _diagonal_covariance_samples([5.0, 2.0, 1.3, 1.0, 1.0]),
        _diagonal_covariance_samples([1.0, 1.0, 1.0, 1.0, 4.0]),
        tau=0.95,
    )
    residual = residual_basis(current)
    assert residual.shape == (5, 2)

    final = compute_signed_differential_subspace(
        _diagonal_covariance_samples([2.0, 1.0, 2.0, 1.0, 1.0]),
        _diagonal_covariance_samples([1.0, 2.0, 1.0, 1.0, 1.0]),
        tau=1.0,
    )
    metrics = residual_to_later_subspace_overlap(current, final)
    assert metrics["pos"]["observed"] == pytest.approx(0.5)
    assert metrics["pos"]["chance"] == pytest.approx(2 / 5)
    assert metrics["pos"]["excess"] == pytest.approx(0.1, abs=1e-6)
    assert metrics["neg"]["observed"] == pytest.approx(0.0)
    assert metrics["neg"]["chance"] == pytest.approx(2 / 5)
    assert metrics["neg"]["excess"] == pytest.approx(-0.4)
    explicit = residual_to_later_subspace_overlap(current, final.u_pos, final.u_neg)
    assert explicit == metrics


def test_residual_overlap_marks_zero_final_side_undefined():
    current = compute_signed_differential_subspace(torch.randn(8, 4), torch.randn(8, 4))
    final = compute_signed_differential_subspace(torch.ones(8, 4), torch.ones(8, 4))
    metrics = residual_to_later_subspace_overlap(current, final)
    assert metrics["pos"]["defined"] is False
    assert metrics["pos"]["observed"] is None
    assert metrics["neg"]["defined"] is False


def test_signed_zero_energy_metrics_are_defined():
    h = torch.ones(4, 3)
    sub = compute_signed_differential_subspace(h, h)
    assert sub.k_pos == sub.k_neg == 0
    assert sub.d_eff_pos == sub.d_eff_neg == 0.0
    assert sub.energy_pos == sub.energy_neg == 0.0
    assert sub.frobenius_strength_pos == sub.frobenius_strength_neg == 0.0
    assert sub.r_pos == 0.0
    assert sub.emergence_pos is None
    assert sub.emergence_neg is None


def test_subspace_stability_identical_is_one():
    torch.manual_seed(2)
    q, _ = torch.linalg.qr(torch.randn(16, 4))
    u = q[:, :3]
    assert subspace_stability(u, u) == pytest.approx(1.0, abs=1e-5)
    assert inter_subspace_relation(u, u) == pytest.approx(1.0, abs=1e-5)


def test_subspace_stability_is_bounded():
    u_a = torch.ones(4, 1)
    u_b = torch.ones(4, 1)
    assert 0.0 <= subspace_stability(u_a, u_b) <= 1.0


def test_subspace_stability_uses_full_unequal_bases():
    u_a = torch.eye(3)
    u_b = torch.tensor(
        [[1.0, 0.0], [0.0, 2**-0.5], [0.0, 2**-0.5]],
    )

    assert subspace_stability(u_a, u_b) == pytest.approx(1.0)
    assert subspace_stability(u_a[:, :2], u_b) == pytest.approx(0.75)


def test_pair_metrics_and_stability_trajectory():
    torch.manual_seed(3)
    d = 12
    n = 30

    def make_pair(scale_c: float, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
        h_c = torch.randn(n, d) * 0.2
        h_c[:, axis] += torch.randn(n) * scale_c
        h_r = torch.randn(n, d) * 0.2
        h_r[:, (axis + 1) % d] += torch.randn(n) * 2.0
        return h_c, h_r

    subs_a = {}
    for name, axis in (("math_vs_code", 0), ("math_vs_text", 1)):
        hc, hr = make_pair(3.0, axis)
        subs_a[name] = compute_differential_subspace(hc, hr, concept=name)

    metrics = compute_pair_metrics_at_checkpoint(subs_a)
    assert "math_vs_code" in metrics["per_concept"]
    assert "math_vs_text" in metrics["per_concept"]
    g = metrics["inter_subspace_relation"]
    assert 0.0 <= g["math_vs_code"]["math_vs_text"] <= 1.0 + 1e-6
    assert g["math_vs_code"]["math_vs_code"] == pytest.approx(1.0, abs=1e-5)

    # Second checkpoint: similar geometry
    subs_b = {}
    for name, axis in (("math_vs_code", 0), ("math_vs_text", 1)):
        hc, hr = make_pair(3.2, axis)
        subs_b[name] = compute_differential_subspace(hc, hr, concept=name)

    traj = compute_stability_trajectory(
        {"main": subs_a, "step_100": subs_b},
        ["main", "step_100"],
        reference_checkpoint="main",
    )
    assert traj["reference"] == "main"
    assert len(traj["consecutive"]["math_vs_code"]) == 1
    assert 0.0 <= traj["consecutive"]["math_vs_code"][0]["subsim"] <= 1.0 + 1e-6
    assert len(traj["vs_reference"]["math_vs_text"]) == 2


def test_experiment_config_ten_checkpoints():
    assert TARGET_MODEL_KEY == "olmo3-rl-zero-math"
    assert len(EXPERIMENT_CHECKPOINTS) == 10
    assert EXPERIMENT_CHECKPOINTS[0] == "step_100"
    assert model_for_checkpoint("step_100").name == "olmo3-rl-zero-math"
    assert len(EXPERIMENT_LAYERS) == 10
    assert N_SAMPLES == 1000
    assert CONCEPT_PAIRS == (("math_vs_text", "math", "text"),)
    assert "setup_raw_prompt" in str(RESULTS_ROOT)
    assert (
        revision_for_checkpoint("step_100")
        == "3315e80ceb281ae2e6a20bd09e8594ba52d4f312"
    )
    assert (
        revision_for_checkpoint("step_1900")
        == "8182367150cef52ddf00dd5259ea94eaa330918e"
    )
    assert [
        revision_for_checkpoint(checkpoint) for checkpoint in EXPERIMENT_CHECKPOINTS
    ] == [
        "3315e80ceb281ae2e6a20bd09e8594ba52d4f312",
        "528d76d90a93f8498801534bc72a346cf886a115",
        "e5280adc8e0de0719e336ec50e095ccbd2577ab4",
        "c23e8ebda0c59b21db2cb9747739998ffbd71430",
        "d0bb0760e45e5bba36410285ffff80ac9b8fcabf",
        "4b024587d5af90918b0d97da34f649e145936459",
        "526ba5a33e775c2fb5780636274e1e38c6fbeea2",
        "968604e73e5ff2b027567270194750bdf5288ccb",
        "5fce4e571a11d60aed9eccaeef72cc603e7e260c",
        "8182367150cef52ddf00dd5259ea94eaa330918e",
    ]


def test_raw_extraction_configuration():
    from postdyn.math_differential_experiment import MAX_SEQ_LEN, USE_CHAT_TEMPLATE

    assert MAX_SEQ_LEN == 2048
    assert USE_CHAT_TEMPLATE is False


def test_tokenizer_preflight_passes_and_rejects_overlong():
    from scripts import run_math_differential_subspace as runner

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            size = len(texts[0])
            return {
                "input_ids": torch.zeros(1, size, dtype=torch.long),
                "attention_mask": torch.ones(1, size, dtype=torch.long),
            }

    assert runner.preflight_tokenizer_prompts(FakeTokenizer(), ["abc"], 3) == [3]
    with pytest.raises(ValueError, match="exceeds"):
        runner.preflight_tokenizer_prompts(FakeTokenizer(), ["abcd"], 3)


def test_raw_extraction_uses_untruncated_final_attention_token():
    from types import SimpleNamespace

    from scripts import run_math_differential_subspace as runner

    calls: list[bool] = []

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            calls.append(kwargs["truncation"])
            size = len(texts[0])
            return {
                "input_ids": torch.zeros(1, size, dtype=torch.long),
                "attention_mask": torch.ones(1, size, dtype=torch.long),
            }

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            size = kwargs["input_ids"].shape[1]
            states = tuple(
                torch.arange(size * 2, dtype=torch.float32).reshape(1, size, 2)
                for _ in range(3)
            )
            return SimpleNamespace(hidden_states=states)

    result = runner.extract_raw_layer_activations(
        FakeModel(), FakeTokenizer(), ["abc"], [0]
    )
    assert calls == [False, False]
    assert result[0].tolist() == [[4.0, 5.0]]


def test_raw_extraction_prefers_concrete_embedding_over_parameters():
    from types import SimpleNamespace

    from scripts import run_math_differential_subspace as runner

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
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("cpu")))

        def parameters(self):
            yield SimpleNamespace(device=torch.device("cuda:0"))

        def __call__(self, **kwargs):
            return SimpleNamespace(
                hidden_states=[torch.ones(1, 2, 3) for _ in range(2)]
            )

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": Input(torch.tensor([[4, 5]])),
                "attention_mask": Input(torch.tensor([[1, 1]])),
            }

    runner.extract_raw_layer_activations(Model(), Tokenizer(), ["raw"], [0])
    assert moved == [torch.device("cpu"), torch.device("cpu")]


def test_raw_extraction_uses_concrete_embedding_map_when_embedding_is_meta():
    from types import SimpleNamespace

    from scripts import run_math_differential_subspace as runner

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
        hf_device_map = {"model.embed_tokens": "cuda:0", "model.layers.0": "cuda:1"}

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("meta")))

        def __call__(self, **kwargs):
            return SimpleNamespace(
                hidden_states=[torch.ones(1, 2, 3) for _ in range(2)]
            )

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {
                "input_ids": Input(torch.tensor([[4, 5]])),
                "attention_mask": Input(torch.tensor([[1, 1]])),
            }

    runner.extract_raw_layer_activations(Model(), Tokenizer(), ["raw"], [0])
    assert moved == ["cuda:0", "cuda:0"]


def test_raw_extraction_rejects_all_meta_devices_before_input_movement():
    from types import SimpleNamespace

    from scripts import run_math_differential_subspace as runner

    moved = []

    class Input:
        def to(self, device):
            moved.append(device)
            return torch.tensor([[4, 5]])

    class Model:
        device = torch.device("meta")
        hf_device_map = {"model.embed_tokens": "meta", "model.layers.0": "meta"}

        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device=torch.device("meta")))

        def parameters(self):
            yield SimpleNamespace(device=torch.device("meta"))

        def __call__(self, **kwargs):
            raise AssertionError("model must not run without a concrete input device")

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": Input(), "attention_mask": Input()}

    with pytest.raises(ValueError, match="concrete execution device"):
        runner.extract_raw_layer_activations(Model(), Tokenizer(), ["raw"], [0])
    assert moved == []


def test_quick_sample_count_is_capped_at_sixteen():
    from scripts.run_math_differential_subspace import quick_sample_count

    assert quick_sample_count(N_SAMPLES) == 16
    assert quick_sample_count(8) == 8


def test_load_domain_prompts_local():
    for domain in ("math", "code", "text"):
        prompts = load_domain_prompts(domain, n_samples=20, allow_hf=False)
        assert len(prompts) == 20
        assert all(isinstance(p, str) and p.strip() for p in prompts)


def test_load_all_default_pairs_shares_math():
    pairs = load_all_default_pairs(n_samples=15, allow_hf=False)
    assert set(pairs) == {"math_vs_code", "math_vs_text"}
    math_a = pairs["math_vs_code"][0]
    math_b = pairs["math_vs_text"][0]
    assert math_a == math_b


def test_runner_importable():
    from scripts import run_math_differential_subspace as runner

    assert hasattr(runner, "main")
    assert hasattr(runner, "run_checkpoint")


def test_strict_dolci_loader_uses_pinned_sources(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def fake_loader(domain: str, *, revision: str | None = None) -> list[str]:
        calls.append((domain, revision))
        return [f"{domain}-{i}" for i in range(4)]

    import postdyn.domain_datasets as domain_datasets

    monkeypatch.setattr(domain_datasets, "_load_dolci_hf", fake_loader)
    assert load_dolci_domain_prompts("math", n_samples=2, seed=1)
    assert calls == [("math", DOLCI_HF_REVISIONS["math"])]
    assert DOLCI_HF_IDS["text"] == "allenai/Dolci-RL-Zero-General-7B"


def test_domain_sampling_is_stable(monkeypatch):
    import postdyn.domain_datasets as domain_datasets

    monkeypatch.setattr(
        domain_datasets,
        "_load_cached_domain",
        lambda domain: [f"{i}" for i in range(20)],
    )
    first = load_domain_prompts("math", n_samples=10, allow_hf=False)
    second = load_domain_prompts("math", n_samples=10, allow_hf=False)
    assert first == second


def test_prompt_cache_rejects_incompatible_metadata(tmp_path, monkeypatch):
    from scripts import run_math_differential_subspace as runner

    calls: list[str] = []

    def fake_loader(domain: str, *, n_samples: int, seed: int) -> list[str]:
        calls.append(domain)
        return [f"{domain}-{i}" for i in range(n_samples)]

    monkeypatch.setattr(runner, "load_dolci_domain_prompts", fake_loader)
    runner.prepare_domain_prompts(tmp_path, 2, 3)
    with pytest.raises(ValueError, match="cache"):
        runner.prepare_domain_prompts(tmp_path, 2, 4)
    assert calls == ["math", "text"]


def test_malformed_prompt_cache_is_rejected(tmp_path):
    from scripts import run_math_differential_subspace as runner

    cache = tmp_path / "prompts" / "math.json"
    cache.parent.mkdir()
    cache.write_text("not json")
    with pytest.raises(ValueError, match="Malformed prompt cache"):
        runner.prepare_domain_prompts(tmp_path, 2, 3)


def test_setup_signature_changes_with_prompts_and_revision():
    from scripts import run_math_differential_subspace as runner

    def make_signature(prompt: str, revision: str) -> str:
        return runner.setup_signature(
            pairs=[("math_vs_text", "math", "text")],
            checkpoints=["step_100"],
            layers=[3],
            model_id="model",
            checkpoint_revisions={"step_100": revision},
            dataset_sources={"math": {"id": "m", "revision": "1"}},
            prompt_fingerprints={"math": prompt, "text": "p-b"},
            n_samples=2,
            seed=1,
            tau=0.95,
            max_seq_len=2048,
            use_chat_template=False,
            extraction_contract="raw_prompt_final_attention_token_v1",
        )

    first = make_signature("p-a", "rev-a")
    assert first != make_signature("p-c", "rev-a")
    assert first != make_signature("p-a", "rev-b")


def test_stability_has_all_unordered_checkpoint_pairs():
    basis = torch.eye(4, 2)
    sub = DifferentialSubspace(
        concept="math_vs_text",
        u=basis,
        eigenvalues_pos=torch.ones(2),
        k=2,
        tau=0.95,
        n_concept=2,
        n_ref=2,
        d_model=4,
        tr_concept=1.0,
        tr_ref=1.0,
        geometry_strength=0.5,
        d_eff=2.0,
    )
    checkpoints = [f"step_{i}" for i in range(10)]
    result = compute_stability_trajectory(
        {checkpoint: {"math_vs_text": sub} for checkpoint in checkpoints},
        checkpoints,
        reference_checkpoint=checkpoints[0],
    )
    assert len(result["pairwise"]["math_vs_text"]) == 45


def test_tensor_sidecar_mismatch_is_not_complete(tmp_path):
    from scripts import run_math_differential_subspace as runner

    sub = compute_differential_subspace(torch.randn(4, 3), torch.randn(4, 3))
    runner.save_subspace(tmp_path, "model", "step_100", 3, sub, "sig")
    sidecar = tmp_path / "U" / "model" / "step_100" / "layer_3" / "concept.json"
    data = json.loads(sidecar.read_text())
    data["u_shape"] = [999, 999]
    sidecar.write_text(json.dumps(data))
    assert not runner.subspace_complete(
        tmp_path, "model", "step_100", 3, "concept", "sig"
    )


def test_structurally_malformed_completion_metadata_returns_false(
    tmp_path, monkeypatch
):
    from scripts import run_math_differential_subspace as runner

    sub = compute_differential_subspace(torch.randn(4, 3), torch.randn(4, 3))
    runner.save_subspace(tmp_path, "model", "step_100", 3, sub, "sig", "rev")
    sidecar = tmp_path / "U" / "model" / "step_100" / "layer_3" / "concept.json"

    sidecar.write_text("[]")
    assert not runner.subspace_complete(
        tmp_path, "model", "step_100", 3, "concept", "sig", "rev"
    )

    sidecar.write_text(json.dumps({"layer": "not-an-int"}))
    assert not runner.subspace_complete(
        tmp_path, "model", "step_100", 3, "concept", "sig", "rev"
    )

    monkeypatch.setattr(runner, "subspace_complete", lambda *args: True)
    metrics = tmp_path / "metrics" / "model" / "step_100" / "layer_3.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("[]")
    manifest = tmp_path / "manifests" / "model__step_100.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[]")
    assert not runner.checkpoint_complete(
        tmp_path, "model", "step_100", [3], ["concept"], "sig", "rev"
    )


def test_incomplete_stability_aggregation_fails(tmp_path):
    from scripts import run_math_differential_subspace as runner

    with pytest.raises(ValueError, match="incomplete"):
        runner.finalize_stability(
            tmp_path, ["step_100", "step_300"], [3], ["math_vs_text"]
        )


def test_empty_checkpoint_selection_is_rejected():
    from scripts import run_math_differential_subspace as runner

    with pytest.raises(ValueError, match="checkpoint"):
        runner.validate_selection([], [3], 1, None)


@pytest.mark.parametrize(
    "checkpoints,layers",
    [
        (["not-a-checkpoint"], [3]),
        (["step_100", "step_100"], [3]),
        (["step_100"], [999]),
    ],
)
def test_noncanonical_selection_is_rejected(checkpoints, layers):
    from scripts import run_math_differential_subspace as runner

    with pytest.raises(ValueError, match="selection"):
        runner.validate_selection(checkpoints, layers, 1, None)


def test_math_checkpoint_complete_rejects_wrong_metrics_identity(tmp_path, monkeypatch):
    from scripts import run_math_differential_subspace as runner

    monkeypatch.setattr(runner, "subspace_complete", lambda *args: True)
    metrics = tmp_path / "metrics/model/step_100/layer_3.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({"setup_signature": "sig", "checkpoint": "wrong"}))
    manifest = tmp_path / "manifests/model__step_100.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "status": "ok",
                "model": "model",
                "checkpoint": "step_100",
                "revision": "rev",
                "setup_signature": "sig",
            }
        )
    )
    assert not runner.checkpoint_complete(
        tmp_path, "model", "step_100", [3], ["concept"], "sig", "rev"
    )


def test_math_run_checkpoint_cleans_hf_cache_on_extraction_failure(
    monkeypatch, tmp_path
):
    from scripts import run_math_differential_subspace as runner

    monkeypatch.setattr(runner, "checkpoint_complete", lambda *args: False)
    monkeypatch.setattr(runner, "preflight_tokenizer_prompts", lambda *args: None)
    monkeypatch.setattr(
        runner,
        "extract_raw_layer_activations",
        lambda *args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    cleaned = []
    monkeypatch.setattr(
        runner, "_clean_hf_cache", lambda model_id: cleaned.append(model_id)
    )
    with pytest.raises(RuntimeError, match="boom"):
        runner.run_checkpoint(
            "step_100",
            root=tmp_path,
            layers=[3],
            n_samples=1,
            max_seq_len=8,
            tau=0.95,
            use_chat_template=False,
            domain_prompts={"math": ["m"], "text": ["t"]},
            keep_hf_cache=False,
            model_loader=lambda cfg, rev: (object(), object()),
        )
    assert cleaned


def test_stability_uses_first_provided_checkpoint(monkeypatch, tmp_path):
    from scripts import run_math_differential_subspace as runner

    observed: dict[str, str | None] = {}
    monkeypatch.setattr(runner, "subspace_complete", lambda *args: True)
    monkeypatch.setattr(runner, "load_saved_subspace", lambda *args: object())

    def fake_stability(_blocks, _order, *, reference_checkpoint=None):
        observed["reference"] = reference_checkpoint
        return {"reference": reference_checkpoint}

    monkeypatch.setattr(runner, "compute_stability_trajectory", fake_stability)
    runner.finalize_stability(tmp_path, ["step_100", "step_200"], [3], ["math_vs_text"])
    assert observed["reference"] == "step_100"
