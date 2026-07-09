"""
Streaming Contrastive Dataset Loader for Concept Dynamics.

Provides positive (domain-specific Dolci-RL-Zero) and negative (wikitext)
text samples for DiM concept extraction. Uses HuggingFace datasets in
streaming mode so the full datasets are never downloaded.

Concept → Dataset mapping:
    math    → allenai/Dolci-RL-Zero-Math-7B
    code    → allenai/Dolci-RL-Zero-Code-7B
    if      → allenai/Dolci-RL-Zero-IF-7B
    general → allenai/Dolci-RL-Zero-General-7B

Negative class (shared): Salesforce/wikitext (wikitext-2-raw-v1 config).

Token selection rule S(x): last token (handled in concept_dynamics.extract_layer_activations).
"""

from __future__ import annotations

import itertools
from typing import Iterable

from datasets import load_dataset

# =============================================================================
# Dataset Configuration
# =============================================================================

CONCEPT_DATASETS: dict[str, str] = {
    "math": "allenai/Dolci-RL-Zero-Math-7B",
    "code": "allenai/Dolci-RL-Zero-Code-7B",
    "if": "allenai/Dolci-RL-Zero-IF-7B",
    "general": "allenai/Dolci-RL-Zero-General-7B",
}

NEGATIVE_DATASET: str = "Salesforce/wikitext"
_NEGATIVE_CONFIG: str = "wikitext-2-raw-v1"
_NEGATIVE_SPLIT: str = "train"

# Text fields to try, in priority order. Different datasets use different
# schemas; this makes the loader robust without hard-coding per dataset.
_TEXT_FIELDS: tuple[str, ...] = ("text", "prompt", "content", "question")


# =============================================================================
# Text Extraction
# =============================================================================


def _extract_text(example: dict) -> str:
    """Extract a plain-text string from a dataset example.

    Tries common field names in priority order. Falls back to conversational
    ``messages`` format (concatenates all message contents).

    Args:
        example: A single dataset row.

    Returns:
        The extracted text string (may be empty if the field is empty).

    Raises:
        KeyError: If no recognized text field is found.
    """
    for field in _TEXT_FIELDS:
        if field in example:
            return str(example[field])

    # Conversational format: messages = [{role, content}, ...]
    if "messages" in example:
        messages = example["messages"]
        parts = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
        return "\n".join(parts)

    raise KeyError(
        f"No text field found in example. "
        f"Tried {list(_TEXT_FIELDS)} and 'messages'. "
        f"Available keys: {list(example.keys())}"
    )


# =============================================================================
# Streaming Sample Collection
# =============================================================================


def _stream_n_samples(
    dataset: Iterable[dict],
    n: int,
) -> list[str]:
    """Collect exactly n non-empty text samples from a streaming dataset.

    Uses itertools.islice for memory-efficient streaming. Filters out
    empty/whitespace-only texts.

    Args:
        dataset: An iterable (streaming) of dataset examples.
        n: Maximum number of samples to collect.

    Returns:
        List of up to n non-empty text strings.
    """
    if n <= 0:
        return []

    texts: list[str] = []
    for example in dataset:
        text = _extract_text(example)
        if text and text.strip():
            texts.append(text)
            if len(texts) >= n:
                break
    return texts


# =============================================================================
# Public API: Contrastive Text Loading
# =============================================================================


def load_contrastive_texts(
    concept: str,
    n_samples: int = 50,
    streaming: bool = True,
) -> tuple[list[str], list[str]]:
    """Load positive and negative text samples for a concept.

    Positive class: domain-specific Dolci-RL-Zero-{Concept}-7B dataset.
    Negative class: Salesforce/wikitext (general text).

    Both are loaded in streaming mode — the full datasets are never
    downloaded. Only the first ``n_samples`` non-empty examples are
    materialized.

    Args:
        concept: One of "math", "code", "if", "general".
        n_samples: Number of samples per class (default: 50).
        streaming: Whether to use streaming mode (default: True).

    Returns:
        (positive_texts, negative_texts) — two lists of n_samples strings.

    Raises:
        ValueError: If ``concept`` is not one of the supported concepts.
    """
    if concept not in CONCEPT_DATASETS:
        raise ValueError(
            f"Unknown concept '{concept}'. "
            f"Supported concepts: {sorted(CONCEPT_DATASETS.keys())}"
        )

    # --- Positive class (domain-specific) ---
    pos_dataset_id = CONCEPT_DATASETS[concept]
    pos_stream = load_dataset(pos_dataset_id, split="train", streaming=streaming)
    positive = _stream_n_samples(pos_stream, n=n_samples)

    # --- Negative class (wikitext, general text) ---
    neg_stream = load_dataset(
        NEGATIVE_DATASET,
        _NEGATIVE_CONFIG,
        split=_NEGATIVE_SPLIT,
        streaming=streaming,
    )
    negative = _stream_n_samples(neg_stream, n=n_samples)

    return positive, negative
