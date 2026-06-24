"""Tests for concept_steering module (TDD).

Tests the difference-in-means steering vector pipeline using mock activations.
No model dependency — activation extraction is abstracted away.
"""

import torch
import pytest

from src.concept_steering import (
    ConceptSteeringVector,
    load_concept_index,
    select_concepts,
    compute_steering_vector,
    batch_compute_steering_vectors,
    save_steering_vectors,
    load_steering_vectors,
)


# =============================================================================
# Concept Index Loading
# =============================================================================


class TestLoadConceptIndex:
    """Test loading PaCE concept index."""

    def test_load_returns_list_of_strings(self):
        concepts = load_concept_index()
        assert isinstance(concepts, list)
        assert all(isinstance(c, str) for c in concepts)

    def test_load_correct_count(self):
        """PaCE concept_index.txt has 40,000 concepts."""
        concepts = load_concept_index()
        assert len(concepts) == 40000

    def test_load_first_concept_is_said(self):
        """First concept should be 'said' (highest frequency rank)."""
        concepts = load_concept_index()
        assert concepts[0] == "said"

    def test_load_custom_path(self, tmp_path):
        """Load from a custom index file."""
        index_file = tmp_path / "custom_index.txt"
        index_file.write_text("['alpha', 'beta', 'gamma']")
        concepts = load_concept_index(str(index_file))
        assert concepts == ["alpha", "beta", "gamma"]

    def test_load_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_concept_index("/nonexistent/path/index.txt")


# =============================================================================
# Concept Selection
# =============================================================================


class TestSelectConcepts:
    """Test concept selection strategies."""

    def test_select_first_n_default_100(self):
        """Default selection returns 100 concepts."""
        concepts = load_concept_index()
        selected = select_concepts(concepts)
        assert len(selected) == 100

    def test_select_first_strategy_preserves_order(self):
        """'first' strategy returns top-N in ranked order."""
        concepts = ["a", "b", "c", "d", "e"]
        selected = select_concepts(concepts, n=3, strategy="first")
        assert selected == ["a", "b", "c"]

    def test_select_random_reproducible(self):
        """Same seed produces same selection."""
        concepts = [f"c{i}" for i in range(1000)]
        s1 = select_concepts(concepts, n=10, strategy="random", seed=42)
        s2 = select_concepts(concepts, n=10, strategy="random", seed=42)
        assert s1 == s2

    def test_select_random_different_seeds_differ(self):
        concepts = [f"c{i}" for i in range(1000)]
        s1 = select_concepts(concepts, n=10, strategy="random", seed=42)
        s2 = select_concepts(concepts, n=10, strategy="random", seed=99)
        assert s1 != s2

    def test_select_random_no_duplicates(self):
        concepts = [f"c{i}" for i in range(100)]
        selected = select_concepts(concepts, n=50, strategy="random", seed=42)
        assert len(selected) == len(set(selected))

    def test_select_n_greater_than_available(self):
        """Selecting more than available returns all."""
        concepts = ["a", "b", "c"]
        selected = select_concepts(concepts, n=10, strategy="first")
        assert len(selected) == 3

    def test_select_invalid_strategy_raises(self):
        concepts = ["a", "b"]
        with pytest.raises(ValueError):
            select_concepts(concepts, n=1, strategy="invalid")


# =============================================================================
# Difference-in-Means Steering Vector
# =============================================================================


class TestComputeSteeringVector:
    """Test DIM steering vector computation."""

    def test_steering_vector_is_difference_of_means(self):
        """Core: steering = mean(positive) - mean(negative)."""
        positive = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # mean=[2,3]
        negative = torch.tensor([[0.0, 0.0], [2.0, 2.0]])  # mean=[1,1]
        csv = compute_steering_vector(positive, negative, "test")
        expected = torch.tensor([1.0, 2.0])
        assert torch.allclose(csv.steering_vector, expected)

    def test_positive_mean_correct(self):
        positive = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        negative = torch.zeros(2, 2)
        csv = compute_steering_vector(positive, negative, "test")
        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(csv.positive_mean, expected)

    def test_negative_mean_correct(self):
        positive = torch.zeros(2, 2)
        negative = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        csv = compute_steering_vector(positive, negative, "test")
        expected = torch.tensor([2.0, 3.0])
        assert torch.allclose(csv.negative_mean, expected)

    def test_positive_std_correct(self):
        """Sample std (correction=1): std([1,3,5]) = 2.0."""
        positive = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        negative = torch.zeros(2, 2)
        csv = compute_steering_vector(positive, negative, "test")
        expected = torch.tensor([2.0, 2.0])
        assert torch.allclose(csv.positive_std, expected)

    def test_negative_std_correct(self):
        positive = torch.zeros(2, 1)
        negative = torch.tensor([[1.0], [3.0], [5.0]])
        csv = compute_steering_vector(positive, negative, "test")
        assert torch.allclose(csv.negative_std, torch.tensor([2.0]))

    def test_n_positive_n_negative_recorded(self):
        positive = torch.randn(50, 128)
        negative = torch.randn(30, 128)
        csv = compute_steering_vector(positive, negative, "test")
        assert csv.n_positive == 50
        assert csv.n_negative == 30

    def test_d_model_recorded(self):
        positive = torch.randn(10, 256)
        negative = torch.randn(10, 256)
        csv = compute_steering_vector(positive, negative, "test")
        assert csv.d_model == 256

    def test_concept_name_recorded(self):
        positive = torch.randn(5, 10)
        negative = torch.randn(5, 10)
        csv = compute_steering_vector(positive, negative, "happiness")
        assert csv.concept_name == "happiness"

    def test_single_sample_per_group(self):
        """Edge case: one sample each → std should be 0, not NaN."""
        positive = torch.tensor([[1.0, 2.0, 3.0]])
        negative = torch.tensor([[0.0, 0.0, 0.0]])
        csv = compute_steering_vector(positive, negative, "test")
        assert torch.allclose(csv.steering_vector, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(csv.positive_std, torch.tensor([0.0, 0.0, 0.0]))
        assert not torch.isnan(csv.positive_std).any()

    def test_mismatched_d_model_raises(self):
        positive = torch.randn(10, 128)
        negative = torch.randn(10, 256)
        with pytest.raises(ValueError):
            compute_steering_vector(positive, negative, "test")

    def test_steering_vector_shape(self):
        positive = torch.randn(10, 512)
        negative = torch.randn(10, 512)
        csv = compute_steering_vector(positive, negative, "test")
        assert csv.steering_vector.shape == (512,)


# =============================================================================
# Batch Computation
# =============================================================================


class TestBatchCompute:
    """Test batch steering vector computation."""

    def test_batch_multiple_concepts(self):
        activations = {
            "happiness": (torch.randn(10, 64), torch.randn(10, 64)),
            "sadness": (torch.randn(10, 64), torch.randn(10, 64)),
            "anger": (torch.randn(10, 64), torch.randn(10, 64)),
        }
        results = batch_compute_steering_vectors(activations)
        assert len(results) == 3
        assert "happiness" in results
        assert isinstance(results["happiness"], ConceptSteeringVector)

    def test_batch_empty_dict(self):
        results = batch_compute_steering_vectors({})
        assert results == {}


# =============================================================================
# Save / Load
# =============================================================================


class TestSaveLoadSteeringVectors:
    """Test save/load roundtrip."""

    def test_save_load_roundtrip(self, tmp_path):
        """Saved vectors should load back identically."""
        vec1 = compute_steering_vector(
            torch.randn(20, 128), torch.randn(20, 128), "happiness"
        )
        vec2 = compute_steering_vector(
            torch.randn(15, 128), torch.randn(15, 128), "sadness"
        )
        vectors = {"happiness": vec1, "sadness": vec2}

        save_path = str(tmp_path / "steering")
        save_steering_vectors(vectors, save_path)
        loaded = load_steering_vectors(save_path)

        assert set(loaded.keys()) == set(vectors.keys())
        for name in vectors:
            assert loaded[name].concept_name == vectors[name].concept_name
            assert torch.allclose(
                loaded[name].steering_vector, vectors[name].steering_vector
            )
            assert torch.allclose(
                loaded[name].positive_mean, vectors[name].positive_mean
            )
            assert torch.allclose(
                loaded[name].negative_mean, vectors[name].negative_mean
            )
            assert torch.allclose(loaded[name].positive_std, vectors[name].positive_std)
            assert torch.allclose(loaded[name].negative_std, vectors[name].negative_std)
            assert loaded[name].n_positive == vectors[name].n_positive
            assert loaded[name].n_negative == vectors[name].n_negative

    def test_save_creates_files(self, tmp_path):
        """Save should create safetensors + json metadata files."""
        vectors = {
            "test": compute_steering_vector(
                torch.randn(5, 32), torch.randn(5, 32), "test"
            )
        }
        save_path = str(tmp_path / "steering")
        save_steering_vectors(vectors, save_path)
        import os

        assert os.path.exists(save_path + ".safetensors")
        assert os.path.exists(save_path + ".json")

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_steering_vectors(str(tmp_path / "nonexistent"))

    def test_save_concepts_with_special_chars(self, tmp_path):
        """Concept names with apostrophes should roundtrip correctly."""
        vectors = {
            "aujourd'hui": compute_steering_vector(
                torch.randn(5, 32), torch.randn(5, 32), "aujourd'hui"
            ),
        }
        save_path = str(tmp_path / "steering")
        save_steering_vectors(vectors, save_path)
        loaded = load_steering_vectors(save_path)
        assert "aujourd'hui" in loaded

    def test_save_large_batch(self, tmp_path):
        """Save/load 100 concepts (target use case)."""
        vectors = {}
        for i in range(100):
            vectors[f"concept_{i}"] = compute_steering_vector(
                torch.randn(10, 64), torch.randn(10, 64), f"concept_{i}"
            )
        save_path = str(tmp_path / "steering_100")
        save_steering_vectors(vectors, save_path)
        loaded = load_steering_vectors(save_path)
        assert len(loaded) == 100
