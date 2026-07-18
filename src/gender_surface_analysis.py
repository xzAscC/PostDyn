from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from src.concept_dynamics import (
    ConceptVector,
    compute_concept_vector,
    extract_layer_activations,
)


_SURFACE_FORMS = (("she", "he", 36), ("her", "his", 10), ("her", "him", 4))


def build_surface_pronoun_texts() -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for female, male, count in _SURFACE_FORMS:
        positive.extend([female] * count)
        negative.extend([male] * count)
    return positive, negative


def compute_surface_pronoun_vectors(
    model,
    tokenizer,
    layers: list[int],
    max_seq_len: int = 32,
) -> dict[int, ConceptVector]:
    positive, negative = build_surface_pronoun_texts()
    positive_activations = extract_layer_activations(
        model, tokenizer, positive, layers, max_seq_len=max_seq_len
    )
    negative_activations = extract_layer_activations(
        model, tokenizer, negative, layers, max_seq_len=max_seq_len
    )
    return {
        layer: compute_concept_vector(
            positive_activations[layer],
            negative_activations[layer],
            concept_name="surface_pronoun_control",
            model_name="surface-pronoun-control",
            layer_idx=layer,
        )
        for layer in layers
    }


def compare_gender_surface_vectors(
    gender_vectors: dict[int, ConceptVector],
    surface_vectors: dict[int, ConceptVector],
) -> dict[int, dict[str, float]]:
    if set(gender_vectors) != set(surface_vectors):
        raise ValueError("Gender and surface-control layers must match")
    comparisons: dict[int, dict[str, float]] = {}
    for layer in sorted(gender_vectors):
        gender = gender_vectors[layer].steering_vector.float()
        surface = surface_vectors[layer].steering_vector.float()
        if gender.shape != surface.shape:
            raise ValueError(f"Vector shape mismatch at layer {layer}")
        cosine = torch.nn.functional.cosine_similarity(
            gender.unsqueeze(0), surface.unsqueeze(0), dim=1
        ).item()
        comparisons[layer] = {
            "cosine": cosine,
            "absolute_cosine": abs(cosine),
        }
    return comparisons


def save_surface_analysis(
    output_path: Path,
    *,
    model_name: str,
    checkpoint: str,
    comparisons: dict[int, dict[str, float]],
) -> None:
    payload = {
        "model": model_name,
        "checkpoint": checkpoint,
        "control_pair_counts": {"she_he": 36, "her_his": 10, "her_him": 4},
        "interpretation": (
            "Higher absolute cosine indicates stronger alignment with the "
            "pronoun-only surface-token direction."
        ),
        "layers": {str(layer): values for layer, values in comparisons.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
