"""
Concept Steering Vector Module

Computes steering vectors via difference-in-means (DIM) for concepts
from the PaCE (Parsimonious Concept Engineering) dataset.

Model-agnostic: operates on pre-extracted activation tensors. The
activation extraction step (model-specific) is abstracted away, allowing
this module to be tested independently.

DIM formula:
    steering_vector = mean(positive_activations) - mean(negative_activations)

References:
    PaCE: Luo et al., "PaCE: Parsimonious Concept Engineering for Large
    Language Models", NeurIPS 2024.
    https://github.com/peterljq/Parsimonious-Concept-Engineering
"""

from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# =============================================================================
# Path Configuration
# =============================================================================

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_INDEX_PATH = os.path.join(_PROJECT_ROOT, "data", "concept_index.txt")


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ConceptSteeringVector:
    """Container for a single concept's DIM steering vector and statistics.

    Attributes:
        concept_name: Name of the concept (e.g., "happiness")
        steering_vector: DIM vector = positive_mean - negative_mean,
                         shape (d_model,)
        positive_mean: Mean of positive activations, shape (d_model,)
        negative_mean: Mean of negative activations, shape (d_model,)
        positive_std: Std of positive activations, shape (d_model,)
        negative_std: Std of negative activations, shape (d_model,)
        n_positive: Number of positive samples used
        n_negative: Number of negative samples used
        d_model: Dimensionality of the activation space
    """

    concept_name: str
    steering_vector: torch.Tensor
    positive_mean: torch.Tensor
    negative_mean: torch.Tensor
    positive_std: torch.Tensor
    negative_std: torch.Tensor
    n_positive: int
    n_negative: int
    d_model: int


# =============================================================================
# Concept Index Loading
# =============================================================================


def load_concept_index(path: str = _DEFAULT_INDEX_PATH) -> list[str]:
    """Load PaCE concept index from file.

    The index file contains a Python list literal of concept names,
    ranked by frequency (most frequent first).

    Args:
        path: Path to concept_index.txt

    Returns:
        List of concept name strings, ordered by frequency rank.

    Raises:
        FileNotFoundError: If the index file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept index not found: {path}")
    with open(path, "r") as f:
        concepts = ast.literal_eval(f.read())
    return concepts


# =============================================================================
# Concept Selection
# =============================================================================


def select_concepts(
    concepts: list[str],
    n: int = 100,
    strategy: str = "first",
    seed: Optional[int] = 42,
) -> list[str]:
    """Select N concepts from a list.

    Args:
        concepts: Full list of concept names.
        n: Number of concepts to select (default: 100).
        strategy: Selection strategy:
            - "first": Top-N by frequency rank (deterministic).
            - "random": Random sampling without replacement.
        seed: Random seed for reproducibility (used with "random").

    Returns:
        Selected concept names.

    Raises:
        ValueError: If strategy is not "first" or "random".
    """
    if strategy == "first":
        return concepts[:n]
    elif strategy == "random":
        rng = random.Random(seed)
        return rng.sample(concepts, min(n, len(concepts)))
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Use 'first' or 'random'.")


# =============================================================================
# Difference-in-Means Steering Vector
# =============================================================================


def compute_steering_vector(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    concept_name: str = "",
) -> ConceptSteeringVector:
    """Compute steering vector via difference-in-means (DIM).

        steering_vector = mean(positive) - mean(negative)

    Std is computed with Bessel's correction (correction=1) when n > 1,
    and falls back to population std (correction=0) for single samples
    to avoid NaN.

    Args:
        positive_activations: Positive-class activations, shape
            (n_positive, d_model).
        negative_activations: Negative-class activations, shape
            (n_negative, d_model).
        concept_name: Name to attach to the result.

    Returns:
        ConceptSteeringVector with all statistics.

    Raises:
        ValueError: If tensors are not 2D or d_model dimensions mismatch.
    """
    if positive_activations.dim() != 2 or negative_activations.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors (n_samples, d_model), got "
            f"{positive_activations.dim()}D and {negative_activations.dim()}D"
        )

    n_pos, d_pos = positive_activations.shape
    n_neg, d_neg = negative_activations.shape

    if d_pos != d_neg:
        raise ValueError(
            f"d_model mismatch: positive has {d_pos}, negative has {d_neg}"
        )

    # Means
    positive_mean = positive_activations.mean(dim=0)
    negative_mean = negative_activations.mean(dim=0)

    # Std with Bessel's correction (sample std) when n > 1,
    # population std when n == 1 to avoid division-by-zero NaN.
    pos_correction = 1 if n_pos > 1 else 0
    neg_correction = 1 if n_neg > 1 else 0
    positive_std = positive_activations.std(dim=0, correction=pos_correction)
    negative_std = negative_activations.std(dim=0, correction=neg_correction)

    # Difference-in-means steering vector
    steering_vector = positive_mean - negative_mean

    return ConceptSteeringVector(
        concept_name=concept_name,
        steering_vector=steering_vector,
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        positive_std=positive_std,
        negative_std=negative_std,
        n_positive=n_pos,
        n_negative=n_neg,
        d_model=d_pos,
    )


# =============================================================================
# Batch Computation
# =============================================================================


def batch_compute_steering_vectors(
    concept_activations: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, ConceptSteeringVector]:
    """Compute steering vectors for multiple concepts.

    Args:
        concept_activations: Mapping from concept name to a tuple of
            (positive_activations, negative_activations).

    Returns:
        Mapping from concept name to ConceptSteeringVector.
    """
    results: dict[str, ConceptSteeringVector] = {}
    for name, (pos, neg) in concept_activations.items():
        results[name] = compute_steering_vector(pos, neg, name)
    return results


# =============================================================================
# Storage (Save / Load)
# =============================================================================

_TENSOR_FIELDS = (
    "steering_vector",
    "positive_mean",
    "negative_mean",
    "positive_std",
    "negative_std",
)


def save_steering_vectors(
    vectors: dict[str, ConceptSteeringVector],
    path: str,
) -> None:
    """Save steering vectors to safetensors + JSON metadata.

    Creates two files:
        {path}.safetensors — all tensor data (float32)
        {path}.json        — concept names and scalar metadata

    Tensors are stored with indexed keys (concept_0000, concept_0001, ...)
    to handle concept names with special characters safely.

    Args:
        vectors: Mapping from concept name to ConceptSteeringVector.
        path: Base output path (extensions added automatically).
    """
    tensor_dict: dict[str, torch.Tensor] = {}
    metadata: dict = {
        "concepts": [],
        "n_positive": [],
        "n_negative": [],
        "d_model": [],
    }

    for idx, name in enumerate(sorted(vectors.keys())):
        vec = vectors[name]
        prefix = f"concept_{idx:04d}"
        for field in _TENSOR_FIELDS:
            tensor = getattr(vec, field)
            tensor_dict[f"{prefix}.{field}"] = tensor.contiguous().to(torch.float32)

        metadata["concepts"].append(name)
        metadata["n_positive"].append(vec.n_positive)
        metadata["n_negative"].append(vec.n_negative)
        metadata["d_model"].append(vec.d_model)

    save_file(tensor_dict, path + ".safetensors")
    with open(path + ".json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_steering_vectors(path: str) -> dict[str, ConceptSteeringVector]:
    """Load steering vectors from safetensors + JSON metadata.

    Args:
        path: Base path (extensions added automatically).

    Returns:
        Mapping from concept name to ConceptSteeringVector.

    Raises:
        FileNotFoundError: If safetensors or JSON file is missing.
    """
    safetensors_path = path + ".safetensors"
    json_path = path + ".json"

    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    with open(json_path, "r") as f:
        metadata = json.load(f)

    vectors: dict[str, ConceptSteeringVector] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for idx, name in enumerate(metadata["concepts"]):
            prefix = f"concept_{idx:04d}"
            vectors[name] = ConceptSteeringVector(
                concept_name=name,
                **{
                    field: f.get_tensor(f"{prefix}.{field}") for field in _TENSOR_FIELDS
                },
                n_positive=metadata["n_positive"][idx],
                n_negative=metadata["n_negative"][idx],
                d_model=metadata["d_model"][idx],
            )

    return vectors
