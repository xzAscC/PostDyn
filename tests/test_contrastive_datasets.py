"""Tests for contrastive_datasets module (TDD).

Tests the streaming contrastive dataset loader that provides
positive (domain-specific Dolci) and negative (wikitext) text pairs
for DiM concept extraction. No network access required — all
dataset loading is mocked.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.contrastive_datasets import (
    CONCEPT_DATASETS,
    NEGATIVE_DATASET,
    load_contrastive_texts,
    _extract_text,
    _stream_n_samples,
)


# =============================================================================
# Concept → Dataset mapping
# =============================================================================


class TestConceptDatasetMapping:
    """Verify concept names map to the correct HF dataset IDs."""

    def test_math_maps_to_dolci_math(self):
        assert CONCEPT_DATASETS["math"] == "allenai/Dolci-RL-Zero-Math-7B"

    def test_code_maps_to_dolci_code(self):
        assert CONCEPT_DATASETS["code"] == "allenai/Dolci-RL-Zero-Code-7B"

    def test_if_maps_to_dolci_if(self):
        assert CONCEPT_DATASETS["if"] == "allenai/Dolci-RL-Zero-IF-7B"

    def test_general_maps_to_dolci_general(self):
        assert CONCEPT_DATASETS["general"] == "allenai/Dolci-RL-Zero-General-7B"

    def test_exactly_four_concepts(self):
        assert set(CONCEPT_DATASETS.keys()) == {"math", "code", "if", "general"}

    def test_negative_dataset_is_wikitext(self):
        assert NEGATIVE_DATASET == "Salesforce/wikitext"


# =============================================================================
# Text field extraction
# =============================================================================


class TestExtractText:
    """Verify text extraction from various dataset schemas."""

    def test_extracts_text_field(self):
        example = {"text": "hello world"}
        assert _extract_text(example) == "hello world"

    def test_extracts_prompt_field(self):
        example = {"prompt": "solve x+1=2"}
        assert _extract_text(example) == "solve x+1=2"

    def test_extracts_content_field(self):
        example = {"content": "some content"}
        assert _extract_text(example) == "some content"

    def test_extracts_question_field(self):
        example = {"question": "what is 2+2?"}
        assert _extract_text(example) == "what is 2+2?"

    def test_extracts_from_messages_chat_format(self):
        """Conversational datasets store [{role, content}, ...]."""
        example = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        text = _extract_text(example)
        assert "hello" in text
        assert "hi there" in text

    def test_raises_on_no_text_field(self):
        with pytest.raises(KeyError, match="text"):
            _extract_text({"id": 123, "label": "foo"})

    def test_skips_empty_text(self):
        """Empty strings should not be returned."""
        assert _extract_text({"text": ""}) == ""


# =============================================================================
# Streaming sample collection
# =============================================================================


class TestStreamNSamples:
    """Verify streaming sample collection."""

    def test_returns_exactly_n_samples(self):
        def gen():
            for i in range(1000):
                yield {"text": f"sample {i}"}

        samples = _stream_n_samples(gen(), n=10)
        assert len(samples) == 10
        assert samples[0] == "sample 0"
        assert samples[9] == "sample 9"

    def test_returns_all_if_fewer_than_n(self):
        def gen():
            for i in range(3):
                yield {"text": f"s{i}"}

        samples = _stream_n_samples(gen(), n=10)
        assert len(samples) == 3

    def test_handles_zero_n(self):
        def gen():
            yield {"text": "x"}

        samples = _stream_n_samples(gen(), n=0)
        assert len(samples) == 0

    def test_filters_empty_text(self):
        def gen():
            yield {"text": "good"}
            yield {"text": ""}
            yield {"text": "also good"}

        samples = _stream_n_samples(gen(), n=5)
        assert samples == ["good", "also good"]


# =============================================================================
# load_contrastive_texts (integration with mocked datasets)
# =============================================================================


class TestLoadContrastiveTexts:
    """Verify the public loader API with mocked dataset streaming."""

    @patch("src.contrastive_datasets.load_dataset")
    def test_returns_positive_and_negative_lists(self, mock_load):
        """Loader returns a (positive, negative) tuple of text lists."""

        def fake_stream(name, *args, **kw):
            if "Dolci" in name:
                return iter([{"text": f"math problem {i}"} for i in range(100)])
            else:
                return iter([{"text": f"wiki text {i}"} for i in range(100)])

        mock_load.side_effect = fake_stream

        pos, neg = load_contrastive_texts("math", n_samples=10)
        assert isinstance(pos, list)
        assert isinstance(neg, list)
        assert len(pos) == 10
        assert len(neg) == 10
        assert "math problem" in pos[0]
        assert "wiki text" in neg[0]

    @patch("src.contrastive_datasets.load_dataset")
    def test_uses_streaming_by_default(self, mock_load):
        mock_load.return_value = iter([{"text": "x"}])
        load_contrastive_texts("code", n_samples=1)
        # streaming=True must be passed
        _, kwargs = mock_load.call_args
        assert kwargs.get("streaming") is True

    @patch("src.contrastive_datasets.load_dataset")
    def test_positive_uses_correct_concept_dataset(self, mock_load):
        seen_datasets = []

        def tracking(name, *args, **kw):
            seen_datasets.append(name)
            return iter([{"text": "x"}])

        mock_load.side_effect = tracking
        load_contrastive_texts("if", n_samples=1)
        assert "allenai/Dolci-RL-Zero-IF-7B" in seen_datasets
        assert "Salesforce/wikitext" in seen_datasets

    def test_unknown_concept_raises_valueerror(self):
        with pytest.raises(ValueError, match="concept"):
            load_contrastive_texts("unknown_concept", n_samples=5)

    @patch("src.contrastive_datasets.load_dataset")
    def test_n_samples_50_default(self, mock_load):
        """User requirement: 50 cases default."""
        mock_load.side_effect = lambda name, *args, **kw: iter(
            [{"text": f"s{i}"} for i in range(200)]
        )
        pos, neg = load_contrastive_texts("general")
        assert len(pos) == 50
        assert len(neg) == 50
