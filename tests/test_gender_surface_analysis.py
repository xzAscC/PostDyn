from __future__ import annotations

import json

import pytest
import torch

from src.concept_dynamics import ConceptVector
from src.gender_surface_analysis import (
    build_surface_pronoun_texts,
    compare_gender_surface_vectors,
    compute_surface_pronoun_vectors,
    save_surface_analysis,
)


def _vector(layer: int, values: list[float], concept: str) -> ConceptVector:
    tensor = torch.tensor(values, dtype=torch.float32)
    zeros = torch.zeros_like(tensor)
    return ConceptVector(
        concept_name=concept,
        model_name="model",
        layer_idx=layer,
        steering_vector=tensor,
        raw_direction=tensor,
        positive_mean=tensor,
        negative_mean=zeros,
        positive_std=zeros,
        negative_std=zeros,
        n_positive=50,
        n_negative=50,
        d_model=tensor.numel(),
    )


def test_surface_pronoun_texts_match_winogender_36_10_4_weights():
    positive, negative = build_surface_pronoun_texts()
    assert len(positive) == len(negative) == 50
    assert positive.count("she") == negative.count("he") == 36
    assert positive.count("her") == 14
    assert negative.count("his") == 10
    assert negative.count("him") == 4


def test_compare_gender_surface_vectors_returns_per_layer_cosines():
    gender = {
        3: _vector(3, [1.0, 0.0], "female_vs_male_gender"),
        6: _vector(6, [1.0, 1.0], "female_vs_male_gender"),
    }
    surface = {
        3: _vector(3, [1.0, 0.0], "surface_pronoun_control"),
        6: _vector(6, [1.0, -1.0], "surface_pronoun_control"),
    }

    result = compare_gender_surface_vectors(gender, surface)

    assert result[3]["cosine"] == pytest.approx(1.0)
    assert result[6]["cosine"] == pytest.approx(0.0, abs=1e-7)
    assert result[3]["absolute_cosine"] == pytest.approx(1.0)


def test_compare_rejects_mismatched_layers():
    gender = {3: _vector(3, [1.0, 0.0], "female_vs_male_gender")}
    surface = {6: _vector(6, [1.0, 0.0], "surface_pronoun_control")}
    with pytest.raises(ValueError, match="layers"):
        compare_gender_surface_vectors(gender, surface)


def test_compute_surface_vectors_reuses_last_token_extraction(monkeypatch):
    calls: list[list[str]] = []

    def fake_extract(model, tokenizer, texts, layers, max_seq_len):
        calls.append(texts)
        value = 2.0 if texts[0] == "she" else 1.0
        return {layer: torch.full((len(texts), 2), value) for layer in layers}

    monkeypatch.setattr(
        "src.gender_surface_analysis.extract_layer_activations", fake_extract
    )
    vectors = compute_surface_pronoun_vectors(object(), object(), [3], 128)
    assert len(calls) == 2
    assert vectors[3].raw_direction.tolist() == pytest.approx([1.0, 1.0])
    assert vectors[3].n_positive == vectors[3].n_negative == 50
    assert vectors[3].model_name == "surface-pronoun-control"


def test_save_surface_analysis_writes_atomic_json(tmp_path):
    output = tmp_path / "nested" / "analysis.json"
    save_surface_analysis(
        output,
        model_name="model",
        checkpoint="step_1",
        comparisons={3: {"cosine": 0.25, "absolute_cosine": 0.25}},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["model"] == "model"
    assert payload["checkpoint"] == "step_1"
    assert payload["layers"]["3"]["cosine"] == 0.25
    assert not output.with_suffix(".json.tmp").exists()
