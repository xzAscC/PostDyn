"""Tests for concept_dynamics module (TDD).

Tests the core concept-dynamics pipeline:
  1. Layer-specific activation extraction (last-token, mock model)
  2. DiM concept vector with normalization (r_hat = r / ||r||)
  3. Cross-model directional stability (7 models → 7×7 cosine matrix)
  4. Per-model concept Gram matrix (4 concepts → 4×4 cosine matrix)

No GPU or network required — model forward pass is mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch
import pytest

from postdyn.concept_dynamics import (
    ConceptVector,
    extract_layer_activations,
    compute_concept_vector,
    cross_model_stability,
    concept_gram_matrices,
    select_uniform_layers,
    _load_model_and_tokenizer,
    load_model_and_tokenizer,
    compute_dynamics_analysis,
    run_full_experiment,
    save_concept_vectors,
    load_concept_vectors,
)


# =============================================================================
# Mock model for activation extraction
# =============================================================================


class _MockModelOutput:
    """Mimics transformers model output with .hidden_states tuple."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class MockModel:
    """Mock transformer that returns deterministic hidden states.

    hidden_states is a tuple of (n_layers + 1) tensors, each (1, seq_len, d_model),
    matching the HF transformers convention where index 0 = embedding layer.
    """

    def __init__(self, n_layers: int = 4, d_model: int = 8, seq_len: int = 5):
        self.config = type(
            "Config", (), {"num_hidden_layers": n_layers, "hidden_size": d_model}
        )()
        self.device = torch.device("cpu")
        self._n_layers = n_layers
        self._d_model = d_model
        self._seq_len = seq_len

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        bs = input_ids.shape[0] if input_ids is not None else 1
        seq = input_ids.shape[1] if input_ids is not None else self._seq_len
        # hidden_states[0] = embedding, [1..n_layers] = transformer layers
        torch.manual_seed(hash((bs, seq)) & 0xFFFF)
        hidden_states = tuple(
            torch.randn(bs, seq, self._d_model) for _ in range(self._n_layers + 1)
        )
        return _MockModelOutput(hidden_states)


class MockTokenizer:
    """Mock tokenizer that returns fixed-size input_ids."""

    def __init__(self, seq_len: int = 5):
        self._seq_len = seq_len
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=True,
        max_length=None,
        padding=False,
    ):
        batch = 1 if isinstance(text, str) else len(text)
        input_ids = torch.randint(0, 1000, (batch, self._seq_len))
        attention_mask = torch.ones(batch, self._seq_len)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestModelLoading:
    @pytest.mark.parametrize(
        "model_name",
        ["olmo3-32b-think-sft", "olmo3-32b-think-rlvr"],
    )
    @pytest.mark.parametrize(
        "loader", [_load_model_and_tokenizer, load_model_and_tokenizer]
    )
    def test_generic_32b_loading_rejected_before_dependency_imports(
        self, model_name, loader, monkeypatch
    ):
        from postdyn.config import OLMO3_VARIANTS

        imported = []
        real_import = __import__

        def tracking_import(name, *args, **kwargs):
            if name in {"torch", "transformers"}:
                imported.append(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", tracking_import)

        with pytest.raises(
            ValueError,
            match=r"src\.quantized_model_loader\.load_olmo3_32b_think",
        ):
            loader(OLMO3_VARIANTS[model_name], "configured-revision")

        assert imported == []

    def test_7b_generic_loading_remains_supported(self, monkeypatch):
        calls = {}

        class TokenizerFactory:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls["tokenizer"] = (model_id, kwargs)
                return SimpleNamespace(pad_token="<pad>", eos_token="<eos>")

        class ModelFactory:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls["model"] = (model_id, kwargs)
                return SimpleNamespace(eval=lambda: calls.__setitem__("eval", True))

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoModelForCausalLM=ModelFactory,
                AutoTokenizer=TokenizerFactory,
            ),
        )
        from postdyn.config import OLMO3_VARIANTS

        _load_model_and_tokenizer(OLMO3_VARIANTS["olmo3-think-sft"], "step_1000")

        assert calls["model"][1]["dtype"] is torch.bfloat16
        assert calls["model"][1]["device_map"] == "auto"
        assert calls["eval"] is True

    def test_native_olmo3_uses_dtype_without_remote_code(self, monkeypatch):
        calls = {}

        class TokenizerFactory:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls["tokenizer"] = kwargs
                return SimpleNamespace(pad_token=None, eos_token="<eos>")

        class LoadedModel:
            def eval(self):
                calls["eval"] = True

        class ModelFactory:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                calls["model"] = kwargs
                return LoadedModel()

        fake_transformers = SimpleNamespace(
            AutoModelForCausalLM=ModelFactory,
            AutoTokenizer=TokenizerFactory,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        config = SimpleNamespace(
            hf_id="allenai/Olmo-3-7B-RL-Zero-Math",
            revision="main",
            architecture="olmo3",
        )

        _load_model_and_tokenizer(config, "step_1900")

        assert calls["model"]["dtype"] is torch.bfloat16
        assert "torch_dtype" not in calls["model"]
        assert "trust_remote_code" not in calls["model"]
        assert "trust_remote_code" not in calls["tokenizer"]
        assert calls["eval"] is True

    def test_tokenizer_fallback_uses_pinned_revision(self, monkeypatch):
        tokenizer_calls = []

        class TokenizerFactory:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                tokenizer_calls.append((model_id, kwargs))
                if len(tokenizer_calls) == 1:
                    raise KeyError("missing tokenizer")
                return SimpleNamespace(pad_token=None, eos_token="<eos>")

        class ModelFactory:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                return SimpleNamespace(eval=lambda: None)

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoModelForCausalLM=ModelFactory,
                AutoTokenizer=TokenizerFactory,
            ),
        )
        config = SimpleNamespace(
            hf_id="allenai/Olmo-3-7B-RL-Zero-Math",
            revision="main",
            architecture="olmo3",
        )

        _load_model_and_tokenizer(config, "step_1900")

        assert tokenizer_calls[1] == (
            "allenai/Olmo-3-1025-7B",
            {"revision": "a81bae42db3975be1671e27b9c9a56da1a9f980f"},
        )


class TestExperimentResume:
    def test_failed_checkpoint_remains_retryable_and_keeps_cache(
        self, tmp_path, monkeypatch
    ):
        from postdyn import config as config_module
        from postdyn import concept_dynamics as dynamics_module

        model_config = SimpleNamespace(name="model", hf_id="org/model")
        monkeypatch.setattr(config_module, "OLMO3_VARIANTS", {"model": model_config})
        monkeypatch.setattr(config_module, "MODEL_CHECKPOINTS", {"model": ["step_1"]})
        monkeypatch.setattr(
            dynamics_module,
            "run_model_extraction",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("failed")),
        )
        monkeypatch.setattr(
            dynamics_module,
            "compute_dynamics_analysis",
            lambda *args, **kwargs: {},
        )
        cleaned = []
        monkeypatch.setattr(dynamics_module, "_clean_hf_cache", cleaned.append)

        results = run_full_experiment(
            ["model"],
            ["python_vs_cpp"],
            [3],
            1,
            str(tmp_path),
        )

        assert results["checkpoints_done"] == []
        assert results["extraction"]["model/step_1"] == {"error": "failed"}
        assert cleaned == []

    def test_dynamics_excludes_partial_failed_checkpoints(self, tmp_path, monkeypatch):
        from postdyn import config as config_module
        from postdyn import concept_dynamics as dynamics_module

        checkpoints = ["step_1", "step_2", "step_failed"]
        monkeypatch.setattr(config_module, "MODEL_CHECKPOINTS", {"model": checkpoints})
        for checkpoint in checkpoints:
            (tmp_path / "vectors" / "model" / checkpoint).mkdir(parents=True)

        results = {
            "checkpoints_done": ["model/step_1", "model/step_2"],
            "extraction": {"model/step_failed": {"error": "failed"}},
        }
        (tmp_path / "extraction_results.json").write_text(
            json.dumps(results), encoding="utf-8"
        )

        def load_vectors(*args, **kwargs):
            vector = ConceptVector(
                concept_name="python_vs_cpp",
                model_name="model",
                layer_idx=3,
                steering_vector=torch.tensor([1.0, 0.0]),
                raw_direction=torch.tensor([1.0, 0.0]),
                positive_mean=torch.tensor([1.0, 0.0]),
                negative_mean=torch.tensor([0.0, 0.0]),
                positive_std=torch.tensor([0.0, 0.0]),
                negative_std=torch.tensor([0.0, 0.0]),
                n_positive=1,
                n_negative=1,
                d_model=2,
            )
            return {"python_vs_cpp": vector, "french_vs_english_language": vector}

        monkeypatch.setattr(dynamics_module, "load_concept_vectors", load_vectors)

        dynamics = compute_dynamics_analysis(
            str(tmp_path),
            ["model"],
            ["python_vs_cpp", "french_vs_english_language"],
            [3],
        )

        stability = dynamics["stability"]["model"]["python_vs_cpp"][3]
        assert stability["checkpoints"] == ["step_1", "step_2"]
        assert set(dynamics["gram"]["model"]) == {"step_1", "step_2"}


# =============================================================================
# ConceptVector dataclass
# =============================================================================


class TestConceptVectorDataclass:
    """Verify the ConceptVector container."""

    def test_has_required_fields(self):
        v = ConceptVector(
            concept_name="math",
            model_name="olmo3-think-sft",
            layer_idx=3,
            steering_vector=torch.randn(8),
            raw_direction=torch.randn(8),
            positive_mean=torch.randn(8),
            negative_mean=torch.randn(8),
            positive_std=torch.randn(8),
            negative_std=torch.randn(8),
            n_positive=50,
            n_negative=50,
            d_model=8,
        )
        assert v.concept_name == "math"
        assert v.model_name == "olmo3-think-sft"
        assert v.layer_idx == 3
        assert v.steering_vector.shape == (8,)


# =============================================================================
# extract_layer_activations
# =============================================================================


class TestExtractLayerActivations:
    """Test last-token hidden state extraction at specified layers."""

    def test_returns_dict_of_correct_layers(self):
        model = MockModel(n_layers=4, d_model=8, seq_len=5)
        tokenizer = MockTokenizer(seq_len=5)
        texts = ["hello world", "foo bar", "baz qux"]
        layers = [0, 1, 3]

        result = extract_layer_activations(model, tokenizer, texts, layers)

        assert set(result.keys()) == {0, 1, 3}

    def test_each_layer_has_correct_shape(self):
        """Each layer tensor should be (n_texts, d_model)."""
        model = MockModel(n_layers=4, d_model=8, seq_len=5)
        tokenizer = MockTokenizer(seq_len=5)
        texts = ["a", "b", "c", "d"]
        layers = [0, 2]

        result = extract_layer_activations(model, tokenizer, texts, layers)

        for layer_idx in layers:
            assert result[layer_idx].shape == (4, 8)

    def test_extracts_last_token_position(self):
        """The extracted vector should be from the last token position."""

        # Build a model where layer 1 hidden state encodes position in dim 0
        class PositionModel(MockModel):
            def __call__(self, input_ids=None, attention_mask=None, **kwargs):
                assert input_ids is not None
                seq = input_ids.shape[1]
                bs = input_ids.shape[0]
                # layer 0 (embedding): dim 0 = position index
                hs0 = torch.zeros(bs, seq, 8)
                for s in range(seq):
                    hs0[:, s, 0] = float(s)
                hs1 = hs0.clone()
                hidden_states = (hs0, hs1)
                return _MockModelOutput(hidden_states)

        model = PositionModel(n_layers=1, d_model=8, seq_len=5)
        tokenizer = MockTokenizer(seq_len=5)
        result = extract_layer_activations(model, tokenizer, ["x"], [0])
        # Last token at seq_len=5 → position index 4
        assert result[0][0, 0].item() == 4.0

    def test_empty_texts_returns_empty_per_layer(self):
        model = MockModel(n_layers=2, d_model=4)
        tokenizer = MockTokenizer(seq_len=3)
        result = extract_layer_activations(model, tokenizer, [], [0])
        assert result[0].shape == (0, 4)

    def test_invalid_layer_raises(self):
        model = MockModel(n_layers=2, d_model=4)
        tokenizer = MockTokenizer(seq_len=3)
        with pytest.raises((ValueError, IndexError)):
            extract_layer_activations(model, tokenizer, ["x"], [5])


# =============================================================================
# compute_concept_vector (DiM + normalization)
# =============================================================================


class TestComputeConceptVector:
    """Test DiM concept vector computation with normalization."""

    def test_raw_direction_is_diff_of_means(self):
        """r = mu+ - mu-."""
        pos = torch.tensor([[2.0, 4.0], [4.0, 2.0]])  # mean = [3, 3]
        neg = torch.tensor([[1.0, 1.0], [3.0, 1.0]])  # mean = [2, 1]

        cv = compute_concept_vector(pos, neg, concept_name="test", normalize=False)

        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(cv.raw_direction, expected)

    def test_normalized_direction_has_unit_norm(self):
        """||r_hat|| = 1 when normalize=True."""
        pos = torch.randn(50, 64)
        neg = torch.randn(50, 64)

        cv = compute_concept_vector(pos, neg, normalize=True)

        norm = cv.steering_vector.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_unnormalized_direction_preserves_norm(self):
        """||r|| = ||mu+ - mu-|| when normalize=False."""
        pos = torch.randn(20, 16)
        neg = torch.randn(20, 16)

        cv = compute_concept_vector(pos, neg, normalize=False)

        expected = (pos.mean(0) - neg.mean(0)).norm().item()
        assert abs(cv.steering_vector.norm().item() - expected) < 1e-5

    def test_normalized_direction_parallel_to_raw(self):
        """r_hat should be parallel to r."""
        pos = torch.randn(30, 32)
        neg = torch.randn(30, 32)

        cv = compute_concept_vector(pos, neg, normalize=True)

        # cos(r_hat, r) = 1
        cos = torch.nn.functional.cosine_similarity(
            cv.steering_vector.unsqueeze(0),
            cv.raw_direction.unsqueeze(0),
        )
        assert abs(cos.item() - 1.0) < 1e-5

    def test_means_and_stds_correct(self):
        pos = torch.randn(25, 8)
        neg = torch.randn(25, 8)

        cv = compute_concept_vector(pos, neg)

        assert torch.allclose(cv.positive_mean, pos.mean(0))
        assert torch.allclose(cv.negative_mean, neg.mean(0))
        assert cv.n_positive == 25
        assert cv.n_negative == 25
        assert cv.d_model == 8

    def test_metadata_propagated(self):
        pos = torch.randn(10, 4)
        neg = torch.randn(10, 4)

        cv = compute_concept_vector(
            pos,
            neg,
            concept_name="math",
            model_name="olmo3-think-sft",
            layer_idx=12,
        )

        assert cv.concept_name == "math"
        assert cv.model_name == "olmo3-think-sft"
        assert cv.layer_idx == 12

    def test_default_is_normalized(self):
        """The paper requires normalized directions by default."""
        pos = torch.randn(10, 4)
        neg = torch.randn(10, 4)

        cv = compute_concept_vector(pos, neg)

        assert abs(cv.steering_vector.norm().item() - 1.0) < 1e-5

    def test_zero_direction_handled_gracefully(self):
        pos = torch.ones(10, 4)
        neg = torch.ones(10, 4)

        cv = compute_concept_vector(pos, neg, normalize=True)

        assert not torch.any(torch.isnan(cv.steering_vector))
        assert torch.allclose(cv.steering_vector, torch.zeros_like(cv.steering_vector))

    def test_empty_activations_raise(self):
        pos = torch.zeros(0, 4)
        neg = torch.ones(3, 4)
        with pytest.raises(ValueError, match="at least one"):
            compute_concept_vector(pos, neg)


# =============================================================================
# cross_model_stability
# =============================================================================


class TestCrossModelStability:
    """Test cosine stability matrix across models (single concept, single layer)."""

    def _make_cv(self, vec, model_name="m"):
        return ConceptVector(
            concept_name="k",
            model_name=model_name,
            layer_idx=0,
            steering_vector=vec,
            raw_direction=vec,
            positive_mean=vec,
            negative_mean=torch.zeros_like(vec),
            positive_std=torch.ones_like(vec),
            negative_std=torch.ones_like(vec),
            n_positive=10,
            n_negative=10,
            d_model=vec.shape[0],
        )

    def test_diagonal_is_one(self):
        v = torch.randn(16)
        vectors = {f"model_{i}": self._make_cv(v, f"model_{i}") for i in range(6)}
        matrix = cross_model_stability(vectors)
        assert torch.allclose(torch.diagonal(matrix), torch.ones(6), atol=1e-5)

    def test_symmetric(self):
        vectors = {f"m{i}": self._make_cv(torch.randn(8), f"m{i}") for i in range(5)}
        matrix = cross_model_stability(vectors)
        assert torch.allclose(matrix, matrix.T)

    def test_shape_n_models(self):
        vectors = {f"m{i}": self._make_cv(torch.randn(4), f"m{i}") for i in range(6)}
        matrix = cross_model_stability(vectors)
        assert matrix.shape == (6, 6)

    def test_orthogonal_models_off_diagonal_zero(self):
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        vectors = {"m1": self._make_cv(v1, "m1"), "m2": self._make_cv(v2, "m2")}
        matrix = cross_model_stability(vectors)
        assert abs(matrix[0, 1].item()) < 1e-5

    def test_identical_models_off_diagonal_one(self):
        v = torch.tensor([1.0, 2.0, 3.0])
        vectors = {"m1": self._make_cv(v, "m1"), "m2": self._make_cv(v, "m2")}
        matrix = cross_model_stability(vectors)
        assert abs(matrix[0, 1].item() - 1.0) < 1e-5


# =============================================================================
# concept_gram_matrices
# =============================================================================


class TestConceptGramMatrices:
    """Test Gram matrix of concept vectors (single model, single layer)."""

    def _make_cv(self, vec, concept_name="c"):
        return ConceptVector(
            concept_name=concept_name,
            model_name="m",
            layer_idx=0,
            steering_vector=vec,
            raw_direction=vec,
            positive_mean=vec,
            negative_mean=torch.zeros_like(vec),
            positive_std=torch.ones_like(vec),
            negative_std=torch.ones_like(vec),
            n_positive=10,
            n_negative=10,
            d_model=vec.shape[0],
        )

    def test_diagonal_is_one(self):
        concepts = {
            name: self._make_cv(torch.randn(8), name)
            for name in ["math", "code", "if", "general"]
        }
        gram = concept_gram_matrices(concepts)
        assert torch.allclose(torch.diagonal(gram), torch.ones(4), atol=1e-5)

    def test_symmetric(self):
        concepts = {
            name: self._make_cv(torch.randn(6), name)
            for name in ["math", "code", "if", "general"]
        }
        gram = concept_gram_matrices(concepts)
        assert torch.allclose(gram, gram.T)

    def test_shape_4_concepts(self):
        concepts = {
            name: self._make_cv(torch.randn(4), name)
            for name in ["math", "code", "if", "general"]
        }
        gram = concept_gram_matrices(concepts)
        assert gram.shape == (4, 4)

    def test_orthogonal_concepts_off_diagonal_zero(self):
        concepts = {
            "math": self._make_cv(torch.tensor([1.0, 0.0]), "math"),
            "code": self._make_cv(torch.tensor([0.0, 1.0]), "code"),
        }
        gram = concept_gram_matrices(concepts)
        assert abs(gram[0, 1].item()) < 1e-5


# =============================================================================
# select_uniform_layers
# =============================================================================


class TestSelectUniformLayers:
    """Test uniform layer selection via the slide formula."""

    def test_10_layers_from_32(self):
        layers = select_uniform_layers(32, n=10)
        assert len(layers) == 10
        assert all(0 <= l < 32 for l in layers)

    def test_matches_experiment_layers_7b(self):
        """Should match EXPERIMENT_LAYERS_7B from config."""
        from postdyn.config import EXPERIMENT_LAYERS_7B

        layers = select_uniform_layers(32, n=10)
        assert layers == EXPERIMENT_LAYERS_7B

    def test_slide_range_for_32_layers(self):
        """ell_0 = round(0.1*31) = 3; ell_9 = round(0.9*31) = 28."""
        layers = select_uniform_layers(32, n=10)
        assert layers[0] == 3
        assert layers[-1] == 28
        assert layers == [3, 6, 9, 11, 14, 17, 20, 22, 25, 28]

    def test_n_one_returns_middle(self):
        assert select_uniform_layers(32, n=1) == [16]

    def test_n_zero_returns_empty(self):
        assert select_uniform_layers(32, n=0) == []


# =============================================================================
# save_concept_vectors / load_concept_vectors — crash-safe, symlink-resistant
# =============================================================================


class TestSaveConceptVectorsAtomicity:
    """save_concept_vectors must publish tensor+sidecar crash-safe and
    symlink-resistant, matching the hardened save_layer_activations pattern:

      * both files are written to secure unique temp paths (tempfile.mkstemp)
        inside the destination directory — never a predictable ``.tmp`` name,
      * the safetensors tensor is published (os.replace) BEFORE the JSON sidecar,
        so any reader observing the sidecar is guaranteed the tensor is visible,
      * on any failure before publication, all temp artifacts are removed,
      * file names, JSON schema, tensor keys, and the return value are unchanged.
    """

    def _make_cv(self, vec, name="k", model="m", layer=0):
        return ConceptVector(
            concept_name=name,
            model_name=model,
            layer_idx=layer,
            steering_vector=vec,
            raw_direction=vec.clone(),
            positive_mean=vec.clone(),
            negative_mean=torch.zeros_like(vec),
            positive_std=torch.ones_like(vec),
            negative_std=torch.ones_like(vec),
            n_positive=10,
            n_negative=10,
            d_model=vec.shape[0],
        )

    def _vectors(self):
        return {
            "code_python_vs_cpp": self._make_cv(torch.randn(8), "code_python_vs_cpp"),
            "math_cot_vs_direct": self._make_cv(torch.randn(8), "math_cot_vs_direct"),
        }

    # ------------------------------------------------------------------
    # Success path: schema, names, roundtrip all preserved
    # ------------------------------------------------------------------

    def test_roundtrip_preserves_vectors_and_metadata(self, tmp_path):
        vectors = self._vectors()
        base = save_concept_vectors(
            vectors, str(tmp_path), "olmo3-think-sft", 17, "step_100"
        )

        # File names and return value unchanged.
        assert base == os.path.join(
            str(tmp_path), "olmo3-think-sft", "step_100", "layer_17"
        )
        assert os.path.exists(base + ".safetensors")
        assert os.path.exists(base + ".json")

        loaded = load_concept_vectors(str(tmp_path), "olmo3-think-sft", 17, "step_100")
        assert set(loaded.keys()) == set(vectors.keys())
        for name, cv in vectors.items():
            got = loaded[name]
            assert torch.allclose(got.steering_vector, cv.steering_vector)
            assert torch.allclose(got.raw_direction, cv.raw_direction)
            assert got.n_positive == cv.n_positive
            assert got.n_negative == cv.n_negative
            assert got.d_model == cv.d_model

        # JSON schema unchanged.
        with open(base + ".json") as f:
            meta = json.load(f)
        assert meta["model_name"] == "olmo3-think-sft"
        assert meta["layer_idx"] == 17
        assert [c["name"] for c in meta["concepts"]] == sorted(vectors.keys())

    # ------------------------------------------------------------------
    # Crash-safety: no leftover temp files after a successful write
    # ------------------------------------------------------------------

    def test_no_temp_files_after_save(self, tmp_path):
        save_concept_vectors(self._vectors(), str(tmp_path), "m", 3, "c")
        for _dirpath, _dirnames, filenames in os.walk(str(tmp_path)):
            for fn in filenames:
                assert ".tmp" not in fn, f"leftover temp file: {fn}"

    # ------------------------------------------------------------------
    # Ordering: tensor is published BEFORE the JSON sidecar
    # ------------------------------------------------------------------

    def test_tensor_published_before_sidecar(self, tmp_path, monkeypatch):
        """Interpose between the two os.replace calls and assert the tensor is
        already on disk when the sidecar lands."""
        import postdyn.concept_dynamics as cd

        events: list[str] = []
        real_replace = os.replace

        def tracked_replace(src, dst):
            events.append(f"replace:{os.path.basename(dst)}")
            return real_replace(src, dst)

        monkeypatch.setattr(cd.os, "replace", tracked_replace)

        save_concept_vectors(self._vectors(), str(tmp_path), "m", 3, "c")

        pub = [e for e in events if e.startswith("replace:")]
        assert len(pub) == 2, pub
        assert pub[0].endswith(".safetensors"), pub
        assert pub[1].endswith(".json"), pub

    # ------------------------------------------------------------------
    # Crash BEFORE tensor publish: save_file fails -> no temp, no final files
    # ------------------------------------------------------------------

    def test_failure_before_tensor_publish_cleans_temp(self, tmp_path, monkeypatch):
        import postdyn.concept_dynamics as cd

        def boom(*args, **kwargs):
            raise RuntimeError("simulated tensor write failure")

        monkeypatch.setattr(cd, "save_file", boom)

        with pytest.raises(RuntimeError, match="simulated tensor write failure"):
            save_concept_vectors(self._vectors(), str(tmp_path), "m", 3, "c")

        # No temp residue anywhere.
        for _dirpath, _dirnames, filenames in os.walk(str(tmp_path)):
            for fn in filenames:
                assert ".tmp" not in fn, f"leftover temp after failure: {fn}"
        # Neither final artifact was published.
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        assert not os.path.exists(base + ".safetensors")
        assert not os.path.exists(base + ".json")

    # ------------------------------------------------------------------
    # Crash BETWEEN tensor and JSON publish: sidecar never appears, temp cleaned
    # ------------------------------------------------------------------

    def test_failure_between_tensor_and_json_cleans_temp(self, tmp_path, monkeypatch):
        """The tensor's os.replace succeeds; the sidecar's os.replace raises.
        Outcome: tensor may be on disk, but the sidecar is NOT, so no loader
        can ever observe a sidecar referencing an unpublished tensor. All temp
        files are cleaned."""
        import postdyn.concept_dynamics as cd

        real_replace = os.replace
        call = {"n": 0}

        def fail_on_second_replace(src, dst):
            call["n"] += 1
            if call["n"] == 2:
                raise OSError("simulated sidecar publish failure")
            return real_replace(src, dst)

        monkeypatch.setattr(cd.os, "replace", fail_on_second_replace)

        with pytest.raises(OSError, match="simulated sidecar publish failure"):
            save_concept_vectors(self._vectors(), str(tmp_path), "m", 3, "c")

        # No temp residue.
        for _dirpath, _dirnames, filenames in os.walk(str(tmp_path)):
            for fn in filenames:
                assert ".tmp" not in fn, f"leftover temp after mid-failure: {fn}"
        # Crucially, the sidecar (JSON) was never published, so no consumer can
        # be fooled into thinking a complete layer exists.
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        assert not os.path.exists(base + ".json")

    # ------------------------------------------------------------------
    # Symlink / predictable-.tmp avoidance: temp paths are unique & unpredictable
    # ------------------------------------------------------------------

    def test_uses_unique_unpredictable_temp_paths(self, tmp_path, monkeypatch):
        """A predictable temp name (e.g. ``layer_3.safetensors.tmp``) is a
        symlink-injection vector: an attacker who pre-places a symlink at that
        path can redirect or corrupt the write. The hardened pattern must use
        ``tempfile.mkstemp`` (O_CREAT|O_EXCL, randomized name) so the temp path
        is neither predictable nor pre-createable.

        We capture every path ``save_file`` and ``_write_json_file`` receive and
        assert none of them is the predictable base+extension name.
        """
        import postdyn.concept_dynamics as cd

        seen_paths: list[str] = []
        real_save = cd.save_file
        real_write_json = cd._write_json_file

        def capture_save(tensor_dict, path, *args, **kwargs):
            seen_paths.append(path)
            return real_save(tensor_dict, path, *args, **kwargs)

        def capture_write_json(path, payload, *args, **kwargs):
            seen_paths.append(path)
            return real_write_json(path, payload, *args, **kwargs)

        monkeypatch.setattr(cd, "save_file", capture_save)
        monkeypatch.setattr(cd, "_write_json_file", capture_write_json)

        save_concept_vectors(self._vectors(), str(tmp_path), "m", 3, "c")

        # mkstemp must supply a randomized prefix (``.cd_<8 chars>``) so the
        # temp basename is never the predictable base name a symlink attacker
        # could pre-create. We assert the basename is neither predictable nor
        # lacking the random component.
        predictable = {"layer_3.safetensors.tmp", "layer_3.json.tmp"}
        for p in seen_paths:
            name = os.path.basename(p)
            assert name not in predictable, (
                f"predictable temp basename (symlink risk): {name}"
            )
            assert name.startswith(".cd_"), (
                f"temp path lacks mkstemp random prefix: {name}"
            )
