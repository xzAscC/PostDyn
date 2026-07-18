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
import sys
from types import SimpleNamespace
from typing import Optional

import torch
import pytest

from src.concept_dynamics import (
    ConceptVector,
    extract_layer_activations,
    compute_concept_vector,
    cross_model_stability,
    concept_gram_matrices,
    select_uniform_layers,
    _load_model_and_tokenizer,
    compute_dynamics_analysis,
    run_full_experiment,
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

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None):
        # Return fixed seq_len regardless of input text
        input_ids = torch.randint(0, 1000, (1, self._seq_len))
        attention_mask = torch.ones(1, self._seq_len)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestModelLoading:
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


class TestExperimentResume:
    def test_failed_checkpoint_remains_retryable_and_keeps_cache(
        self, tmp_path, monkeypatch
    ):
        from src import config as config_module
        from src import concept_dynamics as dynamics_module

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
        from src import config as config_module
        from src import concept_dynamics as dynamics_module

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
    """Test uniform layer selection."""

    def test_10_layers_from_32(self):
        layers = select_uniform_layers(32, n=10)
        assert len(layers) == 10
        assert all(0 <= l < 32 for l in layers)

    def test_matches_experiment_layers_7b(self):
        """Should match EXPERIMENT_LAYERS_7B from config."""
        from src.config import EXPERIMENT_LAYERS_7B

        layers = select_uniform_layers(32, n=10)
        assert layers == EXPERIMENT_LAYERS_7B

    def test_uniformly_spaced(self):
        """Layers should cover 10% to 100% of depth."""
        layers = select_uniform_layers(32, n=10)
        # First layer ≈ 10%, last layer = last index
        assert layers[0] == 3  # int(0.1 * 32) = 3
        assert layers[-1] == 31  # min(int(1.0 * 32), 31) = 31
