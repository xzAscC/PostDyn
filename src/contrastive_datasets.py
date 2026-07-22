"""Contrastive dataset loaders for the 46-concept PaCE study.

This module reads materialized JSON files under ``datasets/`` (produced by
``experiments/download_datasets.py``) and exposes the 46 concept directions
from the 20260724 slides:

* **Code (20)**: all 5x4 directed pairs over {python, cpp, java, js, go}
  built on 50 shared HumanEval-X items. For an arrow ``A -> B``, ``B`` is the
  positive class and ``A`` the negative class.
* **Math (3)**: ``math_informal_vs_formal`` (MiniF2F), ``math_nl_vs_equations``
  (BeyondX), ``math_cot_vs_direct`` (MATH-500).
* **IF (20)**: all 5x4 directed pairs over {eng, fra, deu, zho, jpn} built on
  50 shared Belebele items.
* **General (3)**: ``gender_she_vs_he`` (WinoGender), ``sentiment_label0_vs_label1``
  (SST-2), ``refusal_harmful_vs_benign`` (LLM-LAT).

A small number of **legacy aliases** keep older callers working with the new
polarity (callers must be aware that the polarity now follows the slides).

If a required JSON is missing the loader raises ``FileNotFoundError`` with a
clear message pointing the user at ``experiments/download_datasets.py``.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import urllib.request
from collections.abc import Callable
from typing import Any

# ``load_dataset`` is imported lazily inside fallback helpers to keep this
# module importable in fully offline environments.

# =============================================================================
# Constants (legacy / pinned)
# =============================================================================

# These remain import-compatible for callers that still reference them.
HUMANEVAL_X_DATASET: str = "zai-org/humaneval-x"
HUMANEVAL_X_REVISION: str = "62c78627f3072a1454fa0cb0184737cafe5e4198"
MATH_JSONL_PATH: str = "data/math-500.jsonl"

# Dropped from required defaults (kept here for reference / tests).
FLORES_DATASET: str = "openlanguagedata/flores_plus"
FLORES_REVISION: str = "b3a5298db5721c8a682e7ef00a37fcc9ab522757"

WINOGENDER_DATASET: str = "rudinger/winogender-schemas"
WINOGENDER_REVISION: str = "1c7f8b481ad8a234b41e9f76a424d6e856e13f7f"
_WINOGENDER_TEMPLATES_URL: str = (
    f"https://raw.githubusercontent.com/{WINOGENDER_DATASET}/"
    f"{WINOGENDER_REVISION}/data/templates.tsv"
)

# Pinned HumanEval-X JSONL URLs (compat for src.humaneval_x_validator).
_HUMANEVAL_X_BASE_URL: str = (
    f"https://huggingface.co/datasets/{HUMANEVAL_X_DATASET}/resolve/"
    f"{HUMANEVAL_X_REVISION}/data"
)
_HUMANEVAL_X_FILES: dict[str, str] = {
    language: f"{_HUMANEVAL_X_BASE_URL}/{language}/data/humaneval.jsonl"
    for language in ("python", "cpp", "java", "js", "go")
}


def _humaneval_task_id(example: dict) -> int:
    """Extract the numeric suffix from a HumanEval-X ``task_id`` (compat)."""
    suffix = str(example["task_id"]).rsplit("/", maxsplit=1)[-1]
    suffix = suffix.replace("HumanEval", "").replace("/", "")
    try:
        return int(suffix)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid HumanEval-X task_id: {example.get('task_id')!r}"
        ) from exc


# =============================================================================
# Dataset store integration
# =============================================================================

from src.dataset_store import (  # noqa: E402
    BELEBELE_FILE,
    BEYONDX_FILE,
    HUMANEVAL_X_FILE,
    LLM_LAT_BENIGN_FILE,
    LLM_LAT_HARMFUL_FILE,
    MATH500_FILE,
    MINIF2F_FILE,
    SST2_FILE,
    WINOGENDER_FILE,
    dataset_path,
    humaneval_x_shared_ids,
    belebele_shared_keys,
    load_dataset_json,
)

DEFAULT_N_SAMPLES: int = 50
N_SHARED_ITEMS: int = 50
SAMPlE_SEED: int = 42  # local convenience alias (kept private)
SAMPLE_SEED: int = 42

# Language / dialect vocabularies for directed pairs.
CODE_LANGS: tuple[str, ...] = ("python", "cpp", "java", "js", "go")
IF_LANGS: tuple[str, ...] = ("eng", "fra", "deu", "zho", "jpn")
#: Belebele dialect FLORES-code for each IF language short name.
IF_DIALECTS: dict[str, str] = {
    "eng": "eng_Latn",
    "fra": "fra_Latn",
    "deu": "deu_Latn",
    "zho": "zho_Hans",
    "jpn": "jpn_Jpan",
}

# Common text-field names for streaming fallback extraction.
_TEXT_FIELDS: tuple[str, ...] = ("text", "prompt", "content", "question")


# =============================================================================
# Concept registry
# =============================================================================


def _make_code_concept(src: str, tgt: str) -> dict[str, str]:
    return {
        "domain": "code",
        "name": f"HumanEval-X {src}->{tgt}",
        "dataset": "HumanEval-X",
        "direction": f"{src}->{tgt}",
        "positive": tgt,
        "negative": src,
        "src": src,
        "tgt": tgt,
    }


def _make_if_concept(src: str, tgt: str) -> dict[str, str]:
    return {
        "domain": "if",
        "name": f"Belebele {src}->{tgt}",
        "dataset": "Belebele",
        "direction": f"{src}->{tgt}",
        "positive": tgt,
        "negative": src,
        "src": src,
        "tgt": tgt,
    }


def _build_concepts() -> dict[str, dict[str, str]]:
    concepts: dict[str, dict[str, str]] = {}
    # 20 code pairs
    for src in CODE_LANGS:
        for tgt in CODE_LANGS:
            if src == tgt:
                continue
            concepts[f"code_{src}_vs_{tgt}"] = _make_code_concept(src, tgt)
    # 3 math
    concepts["math_informal_vs_formal"] = {
        "domain": "math",
        "name": "MiniF2F informal->formal",
        "dataset": "MiniF2F",
        "direction": "informal->formal",
        "positive": "formal",
        "negative": "informal",
    }
    concepts["math_nl_vs_equations"] = {
        "domain": "math",
        "name": "BeyondX NL->equations",
        "dataset": "BeyondX",
        "direction": "nl->equations",
        "positive": "equations",
        "negative": "nl",
    }
    concepts["math_cot_vs_direct"] = {
        "domain": "math",
        "name": "MATH-500 CoT->direct",
        "dataset": "MATH-500",
        "direction": "cot->direct",
        "positive": "direct",
        "negative": "cot",
    }
    # 20 IF pairs
    for src in IF_LANGS:
        for tgt in IF_LANGS:
            if src == tgt:
                continue
            concepts[f"if_{src}_vs_{tgt}"] = _make_if_concept(src, tgt)
    # 3 general
    concepts["gender_she_vs_he"] = {
        "domain": "general",
        "name": "WinoGender she->he",
        "dataset": "WinoGender",
        "direction": "she->he",
        "positive": "he",
        "negative": "she",
    }
    concepts["sentiment_label0_vs_label1"] = {
        "domain": "general",
        "name": "SST-2 label0->label1",
        "dataset": "SST-2",
        "direction": "label0->label1",
        "positive": "label1",
        "negative": "label0",
    }
    concepts["refusal_harmful_vs_benign"] = {
        "domain": "general",
        "name": "LLM-LAT harmful->benign",
        "dataset": "LLM-LAT",
        "direction": "harmful->benign",
        "positive": "benign",
        "negative": "harmful",
    }
    return concepts


#: Canonical 46 concepts (no aliases).
CONCEPTS: dict[str, dict[str, str]] = _build_concepts()

#: Legacy alias. Older code used ``PAIRED_CONCEPTS``; it now maps to the same
#: 46-key registry. Callers that hard-coded one of the legacy keys must check
#: the new polarity.
PAIRED_CONCEPTS: dict[str, dict[str, str]] = CONCEPTS

#: Legacy concept key -> new canonical key. All aliases now follow the slides'
#: polarity (the alias is only a name change, not a polarity swap).
ALIASES: dict[str, str] = {
    "python_vs_cpp": "code_python_vs_cpp",
    "french_vs_english_language": "if_eng_vs_fra",
    "female_vs_male_gender": "gender_she_vs_he",
}


def _resolve_concept(concept: str) -> str:
    if concept in CONCEPTS:
        return concept
    if concept in ALIASES:
        return ALIASES[concept]
    raise ValueError(
        f"Unknown concept {concept!r}. "
        f"Supported canonical concepts ({len(CONCEPTS)}): {sorted(CONCEPTS)}"
    )


def list_concepts() -> list[str]:
    """Return all supported concept keys (canonical + aliases)."""
    return sorted(set(CONCEPTS) | set(ALIASES))


def all_concept_keys() -> list[str]:
    """Return the 46 canonical concept keys (no aliases)."""
    return sorted(CONCEPTS.keys())


# =============================================================================
# Text extraction helpers (legacy compat + reuse)
# =============================================================================


def _extract_text(example: dict) -> str:
    """Extract a plain-text string from a dataset row (multi-schema)."""
    for field in _TEXT_FIELDS:
        if field in example:
            return str(example[field])
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


def _stream_n_samples(dataset, n: int) -> list[str]:
    """Collect up to ``n`` non-empty text samples from a streaming dataset."""
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


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("```")
    )


# =============================================================================
# JSON helpers
# =============================================================================


def _require_dataset(file_name: str, concept_hint: str) -> dict:
    """Load a materialized dataset JSON or raise a clear error."""
    path = dataset_path(file_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset {file_name} for concept '{concept_hint}'. "
            f"Run `uv run python experiments/download_datasets.py "
            f"--only {file_name.removesuffix('.json')}` first."
        )
    return load_dataset_json(file_name)


# =============================================================================
# HumanEval-X directed pairs (Code domain)
# =============================================================================


def _load_humaneval_x_index() -> dict[str, dict[int, dict]]:
    """Load the materialized HumanEval-X JSON indexed by lang -> numeric_id.

    Returns ``{lang: {numeric_id: {"code": str, "prompt": str, ...}}}``.
    """
    data = _require_dataset(HUMANEVAL_X_FILE, "code_*")
    raw_languages: dict[str, list[dict]] = data.get("languages", {})
    indexed: dict[str, dict[int, dict]] = {}
    for lang, items in raw_languages.items():
        by_id: dict[int, dict] = {}
        for item in items:
            try:
                numeric_id = int(item["numeric_id"])
            except (KeyError, TypeError, ValueError):
                continue
            code = _strip_code_fences(str(item.get("code") or ""))
            if not code.strip():
                continue
            item["code"] = code
            by_id[numeric_id] = item
        indexed[lang] = by_id
    return indexed


def load_humaneval_x_directed_pairs(
    src: str, tgt: str, n_samples: int = DEFAULT_N_SAMPLES
) -> list[tuple[str, str]]:
    """Load aligned ``(positive=tgt, negative=src)`` HumanEval-X code pairs.

    Uses the pinned 50 shared numeric task ids from ``shared_item_ids.json``;
    if that file is empty (download was skipped), falls back to the
    intersection of src and tgt ids and samples with ``SAMPLE_SEED``.
    """
    if n_samples <= 0:
        return []
    if src == tgt:
        raise ValueError(f"HumanEval-X src == tgt ({src})")
    if src not in CODE_LANGS or tgt not in CODE_LANGS:
        raise ValueError(
            f"Unknown HumanEval-X language(s): {src!r}, {tgt!r}. Known: {CODE_LANGS}"
        )
    indexed = _load_humaneval_x_index()
    if src not in indexed or tgt not in indexed:
        missing = [x for x in (src, tgt) if x not in indexed]
        raise ValueError(
            f"HumanEval-X language(s) missing from {HUMANEVAL_X_FILE}: {missing}"
        )
    src_by_id = indexed[src]
    tgt_by_id = indexed[tgt]

    pinned = humaneval_x_shared_ids()
    if pinned and len(pinned) >= n_samples:
        selected_ids = list(pinned)
    else:
        shared_pool = sorted(set(src_by_id) & set(tgt_by_id))
        if len(shared_pool) < n_samples:
            raise ValueError(
                f"HumanEval-X {src}->{tgt}: only {len(shared_pool)} shared ids "
                f"(need {n_samples}). Re-run download_datasets.py --force "
                f"humaneval_x to refresh shared_item_ids.json."
            )
        rng = random.Random(SAMPLE_SEED)
        selected_ids = sorted(rng.sample(shared_pool, n_samples))

    pairs: list[tuple[str, str]] = []
    for tid in selected_ids[:n_samples]:
        if tid not in src_by_id or tid not in tgt_by_id:
            continue
        pairs.append(
            (
                str(tgt_by_id[tid].get("code", "")),
                str(src_by_id[tid].get("code", "")),
            )
        )
    if len(pairs) < n_samples:
        raise ValueError(
            f"HumanEval-X {src}->{tgt}: produced only {len(pairs)} aligned pairs "
            f"(need {n_samples}); shared_item_ids may be stale."
        )
    return pairs


def load_humaneval_x_pairs(n_samples: int = DEFAULT_N_SAMPLES) -> list[tuple[str, str]]:
    """Legacy alias: python -> cpp per slides polarity (+cpp -python)."""
    return load_humaneval_x_directed_pairs("python", "cpp", n_samples)


# =============================================================================
# Math domain
# =============================================================================


def load_minif2f_pairs(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    """Load aligned ``(positive=formal, negative=informal)`` MiniF2F pairs."""
    if n_samples <= 0:
        return []
    data = _require_dataset(MINIF2F_FILE, "math_informal_vs_formal")
    items = data.get("items", [])
    pairs: list[tuple[str, str]] = []
    for item in items:
        formal = str(item.get("formal", "")).strip()
        informal = str(item.get("informal", "")).strip()
        if not formal or not informal:
            continue
        pairs.append((formal, informal))
        if len(pairs) >= n_samples:
            break
    if len(pairs) < n_samples:
        raise ValueError(f"MiniF2F: only {len(pairs)} usable pairs (need {n_samples}).")
    return pairs


def load_beyondx_pairs(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    """Load aligned ``(positive=equations, negative=nl/problem)`` BeyondX pairs."""
    if n_samples <= 0:
        return []
    data = _require_dataset(BEYONDX_FILE, "math_nl_vs_equations")
    items = data.get("items", [])
    pairs: list[tuple[str, str]] = []
    for item in items:
        equations = str(item.get("equations", "")).strip()
        problem = str(item.get("problem", "")).strip()
        if not equations or not problem:
            continue
        pairs.append((equations, problem))
        if len(pairs) >= n_samples:
            break
    if len(pairs) < n_samples:
        raise ValueError(f"BeyondX: only {len(pairs)} usable pairs (need {n_samples}).")
    return pairs


def load_math_cot_vs_direct_pairs(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    """Load aligned ``(positive=direct_text, negative=cot_text)`` MATH-500 pairs.

    CoT is the verbose chain-of-thought solution; the direct side is a short
    ``"The answer is {answer}"`` string. Both share the problem statement.
    """
    if n_samples <= 0:
        return []
    data = _require_dataset(MATH500_FILE, "math_cot_vs_direct")
    items = data.get("items", [])
    pairs: list[tuple[str, str]] = []
    for item in items:
        direct_text = str(item.get("direct_text", "")).strip()
        cot_text = str(item.get("cot_text", "")).strip()
        if not direct_text or not cot_text:
            continue
        pairs.append((direct_text, cot_text))
        if len(pairs) >= n_samples:
            break
    if len(pairs) < n_samples:
        raise ValueError(
            f"MATH-500: only {len(pairs)} usable pairs (need {n_samples})."
        )
    return pairs


def load_math_pairs(n_samples: int = DEFAULT_N_SAMPLES) -> list[tuple[str, str]]:
    """Legacy alias for concise-vs-verbose MATH pairs.

    Old callers expected ``MATH_JSONL_PATH``. New default routes to CoT vs
    direct (the slide's polarity: +direct, -cot).
    """
    # Prefer the materialized MATH-500 JSON when available.
    try:
        return load_math_cot_vs_direct_pairs(n_samples)
    except FileNotFoundError:
        pass
    # Fall back to legacy data/math-500.jsonl if present.
    return _load_math_pairs_legacy_jsonl(n_samples)


def _load_math_pairs_legacy_jsonl(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    if not os.path.exists(MATH_JSONL_PATH):
        raise FileNotFoundError(
            f"Missing dataset {MATH500_FILE}. Run "
            "`uv run python experiments/download_datasets.py --only math500`."
        )
    rows: list[dict] = []
    with open(MATH_JSONL_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) < n_samples:
        raise ValueError(
            f"Requested {n_samples} MATH pairs but only {len(rows)} rows are available."
        )
    pairs: list[tuple[str, str]] = []
    for row in rows[:n_samples]:
        problem = str(row["problem"])
        # In the new polarity, the concise/direct side is positive.
        concise = f"{problem}\n{row['concise_solution']}"
        verbose = f"{problem}\n{row['verbose_solution']}"
        pairs.append((concise, verbose))
    return pairs


# =============================================================================
# Belebele directed pairs (IF domain)
# =============================================================================


def _load_belebele_index() -> dict[str, dict[tuple[str, int], str]]:
    """Load Belebele JSON indexed by dialect -> (link, qnum) -> text."""
    data = _require_dataset(BELEBELE_FILE, "if_*")
    raw_dialects: dict[str, list[dict]] = data.get("dialects", {})
    indexed: dict[str, dict[tuple[str, int], str]] = {}
    for dialect, items in raw_dialects.items():
        by_key: dict[tuple[str, int], str] = {}
        for item in items:
            link = item.get("link")
            qnum = item.get("question_number")
            if link is None or qnum is None:
                continue
            try:
                key = (str(link), int(qnum))
            except (TypeError, ValueError):
                continue
            text = str(item.get("text", "") or "")
            if not text.strip():
                continue
            by_key[key] = text
        indexed[dialect] = by_key
    return indexed


def _belebele_dialect_for(lang: str) -> str:
    if lang not in IF_DIALECTS:
        raise ValueError(f"Unknown IF language {lang!r}. Known: {sorted(IF_DIALECTS)}")
    return IF_DIALECTS[lang]


def load_belebele_directed_pairs(
    src: str, tgt: str, n_samples: int = DEFAULT_N_SAMPLES
) -> list[tuple[str, str]]:
    """Load aligned ``(positive=tgt, negative=src)`` Belebele text pairs."""
    if n_samples <= 0:
        return []
    if src == tgt:
        raise ValueError(f"Belebele src == tgt ({src})")
    src_dialect = _belebele_dialect_for(src)
    tgt_dialect = _belebele_dialect_for(tgt)
    indexed = _load_belebele_index()
    if src_dialect not in indexed or tgt_dialect not in indexed:
        missing = [
            d
            for d, present in (
                (src_dialect, src_dialect in indexed),
                (tgt_dialect, tgt_dialect in indexed),
            )
            if not present
        ]
        raise ValueError(f"Belebele dialect(s) missing from {BELEBELE_FILE}: {missing}")
    src_by_key = indexed[src_dialect]
    tgt_by_key = indexed[tgt_dialect]

    pinned = belebele_shared_keys()
    pinned_keys: list[tuple[str, int]] = []
    if pinned and len(pinned) >= n_samples:
        for entry in pinned:
            if not isinstance(entry, dict):
                continue
            link = entry.get("link")
            qnum = entry.get("question_number")
            if link is None or qnum is None:
                continue
            try:
                pinned_keys.append((str(link), int(qnum)))
            except (TypeError, ValueError):
                continue

    if len(pinned_keys) >= n_samples:
        selected = pinned_keys
    else:
        shared_pool = sorted(set(src_by_key) & set(tgt_by_key))
        if len(shared_pool) < n_samples:
            raise ValueError(
                f"Belebele {src}->{tgt}: only {len(shared_pool)} shared keys "
                f"(need {n_samples})."
            )
        rng = random.Random(SAMPLE_SEED)
        selected = sorted(rng.sample(shared_pool, n_samples))

    pairs: list[tuple[str, str]] = []
    for key in selected[:n_samples]:
        if key not in src_by_key or key not in tgt_by_key:
            continue
        pairs.append((tgt_by_key[key], src_by_key[key]))
    if len(pairs) < n_samples:
        raise ValueError(
            f"Belebele {src}->{tgt}: produced only {len(pairs)} aligned pairs "
            f"(need {n_samples})."
        )
    return pairs


# Legacy FLORES loader removed — Belebele is the new IF source.
def load_flores_pairs(n_samples: int = DEFAULT_N_SAMPLES) -> list[tuple[str, str]]:
    """Deprecated: Belebele is the new IF source. Alias for eng->fra."""
    return load_belebele_directed_pairs("eng", "fra", n_samples)


# =============================================================================
# WinoGender (gender she -> he)
# =============================================================================

_WINOGENDER_PRONOUNS: dict[str, tuple[str, str]] = {
    "$NOM_PRONOUN": ("she", "he"),
    "$POSS_PRONOUN": ("her", "his"),
    "$ACC_PRONOUN": ("her", "him"),
}


def _load_winogender_rows_from_json() -> list[dict] | None:
    try:
        data = _require_dataset(WINOGENDER_FILE, "gender_she_vs_he")
    except FileNotFoundError:
        return None
    return list(data.get("rows", []))


def _parse_winogender_tsv(content: str) -> list[dict]:
    reader = csv.reader(io.StringIO(content), delimiter="\t")
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    parsed: list[dict] = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        parsed.append({str(header[i]): row[i] for i in range(len(header))})
    return parsed


def _build_winogender_nominative_pairs(
    rows: list[dict], n_samples: int
) -> list[tuple[str, str]]:
    """Build nominative-only (he, she) minimal pairs from raw template rows.

    Per slides: both sides are the *same* sentence, differing only in the
    nominative pronoun (``he`` vs ``she``). Returns pairs as ``(he, she)``
    (positive, negative) per the she->he polarity.

    Only ``$NOM_PRONOUN`` templates are used; if fewer than ``n_samples`` are
    available, returns whatever we have (caller decides whether to fall back).
    """
    placeholder = "$NOM_PRONOUN"
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        sentence = row.get("sentence", "")
        if placeholder not in sentence:
            continue
        if sentence.count(placeholder) != 1:
            continue
        occupation = row.get("occupation(0)", "")
        participant = row.get("other-participant(1)", "")
        instantiated = sentence.replace("$OCCUPATION", occupation).replace(
            "$PARTICIPANT", participant
        )
        # Per slide: he = positive, she = negative.
        he_sentence = instantiated.replace(placeholder, "he")
        she_sentence = instantiated.replace(placeholder, "she")
        key = he_sentence + "|" + she_sentence
        if key in seen:
            continue
        seen.add(key)
        pairs.append((he_sentence, she_sentence))
        if len(pairs) >= n_samples:
            break
    return pairs


def load_winogender_pairs(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    """Load aligned ``(positive=he, negative=she)`` WinoGender pairs.

    Prefers nominative-only minimal pairs (per slides). Falls back to the
    multi-form allocation if the nominative pool is too small.

    Memory: reads from the materialized JSON when available; otherwise falls
    back to fetching the pinned ``templates.tsv`` via HTTP (kept for legacy).
    """
    if n_samples <= 0:
        return []

    rows = _load_winogender_rows_from_json()
    if rows is None:
        with urllib.request.urlopen(_WINOGENDER_TEMPLATES_URL, timeout=30) as response:
            content = response.read().decode("utf-8")
        rows = _parse_winogender_tsv(content)

    nom_pairs = _build_winogender_nominative_pairs(rows, n_samples)
    if len(nom_pairs) >= n_samples:
        return nom_pairs[:n_samples]

    # Fallback: legacy multi-form allocation but with new polarity (he, she).
    return _allocate_winogender_multi_form(rows, n_samples)


def _allocate_winogender_multi_form(
    rows: list[dict], n_samples: int
) -> list[tuple[str, str]]:
    if n_samples % 2 != 0:
        raise ValueError(
            "WinoGender fallback requires even n_samples for balanced pairing"
        )

    try:
        occupation_idx_keys = ("occupation(0)",)
        participant_idx_keys = ("other-participant(1)",)
        answer_key = "answer"
        sentence_key = "sentence"
        header_keys = set(rows[0].keys()) if rows else set()
        if not all(
            key in header_keys
            for key in (
                *occupation_idx_keys,
                *participant_idx_keys,
                answer_key,
                sentence_key,
            )
        ):
            raise ValueError("WinoGender templates missing required columns")
    except (IndexError, ValueError) as exc:
        raise ValueError(f"WinoGender rows unusable: {exc}") from exc

    buckets: dict[str, dict[int, list[tuple[str, str]]]] = {
        form: {0: [], 1: []} for form in _WINOGENDER_PRONOUNS
    }
    for row in rows:
        try:
            answer = int(row[answer_key])
        except (KeyError, TypeError, ValueError):
            continue
        if answer not in (0, 1):
            continue
        sentence = row[sentence_key]
        sentence = sentence.replace("$OCCUPATION", row["occupation(0)"])
        sentence = sentence.replace("$PARTICIPANT", row["other-participant(1)"])
        forms = [f for f in _WINOGENDER_PRONOUNS if f in sentence]
        if len(forms) != 1:
            continue
        form = forms[0]
        if sentence.count(form) != 1:
            continue
        female_pron, male_pron = _WINOGENDER_PRONOUNS[form]
        she_sentence = sentence.replace(form, female_pron)
        he_sentence = sentence.replace(form, male_pron)
        # Polarity per slides: positive=he, negative=she.
        buckets[form][answer].append((he_sentence, she_sentence))

    capacities = {
        form: min(len(by_answer[0]), len(by_answer[1]))
        for form, by_answer in buckets.items()
    }
    pairs_per_answer = n_samples // 2
    forms_available = [f for f in _WINOGENDER_PRONOUNS if capacities.get(f, 0) > 0]
    if sum(capacities[f] for f in forms_available) < pairs_per_answer:
        raise ValueError(
            f"Requested {n_samples} WinoGender pairs, but pinned templates "
            f"have insufficient balanced rows (capacities={capacities})."
        )

    allocations = {f: 0 for f in _WINOGENDER_PRONOUNS}
    # Seed one per form, then proportional fill.
    seeded = forms_available[:pairs_per_answer]
    for f in seeded:
        allocations[f] = 1
    remaining = pairs_per_answer - len(seeded)
    if remaining > 0:
        remaining_cap = {f: capacities[f] - allocations[f] for f in forms_available}
        total_cap = sum(remaining_cap.values())
        if total_cap <= 0:
            raise ValueError("WinoGender fallback capacity exhausted")
        raw_shares = {
            f: remaining * remaining_cap[f] / total_cap for f in forms_available
        }
        floor = {f: int(raw_shares[f]) for f in forms_available}
        for f in forms_available:
            allocations[f] += floor[f]
        leftover = remaining - sum(floor.values())
        order = sorted(
            forms_available,
            key=lambda f: (raw_shares[f] - floor[f], capacities[f]),
            reverse=True,
        )
        for f in order:
            if leftover <= 0:
                break
            if allocations[f] < capacities[f]:
                allocations[f] += 1
                leftover -= 1
        if leftover > 0:
            raise ValueError("Could not allocate WinoGender form quotas")

    selected: list[tuple[str, str]] = []
    for form in _WINOGENDER_PRONOUNS:
        per_answer = allocations[form]
        for answer in (0, 1):
            selected.extend(buckets[form][answer][:per_answer])
    return selected


# =============================================================================
# SST-2 (sentiment label0 -> label1)
# =============================================================================


def load_sst2_pairs(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    """Load unpaired ``(positive=label1, negative=label0)`` SST-2 sentences.

    Returns ``n_samples`` (positive, negative) tuples. The two classes are
    *not* item-paired; we pin the indices deterministically with
    ``SAMPLE_SEED`` so reruns are reproducible.
    """
    if n_samples <= 0:
        return []
    data = _require_dataset(SST2_FILE, "sentiment_label0_vs_label1")
    items = data.get("items", [])
    positives = [str(it["sentence"]) for it in items if int(it["label"]) == 1]
    negatives = [str(it["sentence"]) for it in items if int(it["label"]) == 0]
    if len(positives) < n_samples or len(negatives) < n_samples:
        raise ValueError(
            f"SST-2 needs >= {n_samples} per class; got "
            f"pos={len(positives)}, neg={len(negatives)}."
        )
    rng = random.Random(SAMPLE_SEED)
    pos_sel_idx = sorted(rng.sample(range(len(positives)), n_samples))
    neg_sel_idx = sorted(rng.sample(range(len(negatives)), n_samples))
    return [(positives[i], negatives[j]) for i, j in zip(pos_sel_idx, neg_sel_idx)]


# =============================================================================
# LLM-LAT refusal (harmful -> benign)
# =============================================================================


def _load_llm_lat_texts(file_name: str, concept_hint: str) -> list[str]:
    data = _require_dataset(file_name, concept_hint)
    texts = data.get("texts", [])
    out: list[str] = []
    for t in texts:
        s = str(t)
        if s.strip():
            out.append(s)
    return out


def load_refusal_pairs(
    n_samples: int = DEFAULT_N_SAMPLES,
) -> list[tuple[str, str]]:
    """Load unpaired ``(positive=benign, negative=harmful)`` LLM-LAT texts."""
    if n_samples <= 0:
        return []
    benign = _load_llm_lat_texts(LLM_LAT_BENIGN_FILE, "refusal_harmful_vs_benign/+")
    harmful = _load_llm_lat_texts(LLM_LAT_HARMFUL_FILE, "refusal_harmful_vs_benign/-")
    if len(benign) < n_samples or len(harmful) < n_samples:
        raise ValueError(
            f"LLM-LAT needs >= {n_samples} per class; got benign={len(benign)}, "
            f"harmful={len(harmful)}."
        )
    rng = random.Random(SAMPLE_SEED)
    ben_idx = sorted(rng.sample(range(len(benign)), n_samples))
    harm_idx = sorted(rng.sample(range(len(harmful)), n_samples))
    return [(benign[i], harmful[j]) for i, j in zip(ben_idx, harm_idx)]


# =============================================================================
# Concept dispatcher
# =============================================================================


def _code_loader_for(concept: str) -> Callable[[int], list[tuple[str, str]]]:
    entry = CONCEPTS[concept]
    src, tgt = entry["src"], entry["tgt"]

    def _loader(n: int) -> list[tuple[str, str]]:
        return load_humaneval_x_directed_pairs(src, tgt, n)

    return _loader


def _if_loader_for(concept: str) -> Callable[[int], list[tuple[str, str]]]:
    entry = CONCEPTS[concept]
    src, tgt = entry["src"], entry["tgt"]

    def _loader(n: int) -> list[tuple[str, str]]:
        return load_belebele_directed_pairs(src, tgt, n)

    return _loader


#: Loader lookup keyed by resolved canonical concept.
_LOADERS: dict[str, Callable[[int], list[tuple[str, str]]]] = {
    "math_informal_vs_formal": load_minif2f_pairs,
    "math_nl_vs_equations": load_beyondx_pairs,
    "math_cot_vs_direct": load_math_cot_vs_direct_pairs,
    "gender_she_vs_he": load_winogender_pairs,
    "sentiment_label0_vs_label1": load_sst2_pairs,
    "refusal_harmful_vs_benign": load_refusal_pairs,
}
# Add code/IF directed loaders.
for _concept_key in [k for k, v in CONCEPTS.items() if v["domain"] == "code"]:
    _LOADERS[_concept_key] = _code_loader_for(_concept_key)
for _concept_key in [k for k, v in CONCEPTS.items() if v["domain"] == "if"]:
    _LOADERS[_concept_key] = _if_loader_for(_concept_key)


def _get_loader(concept: str) -> Callable[[int], list[tuple[str, str]]]:
    resolved = _resolve_concept(concept)
    return _LOADERS[resolved]


def load_contrastive_texts(
    concept: str,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> tuple[list[str], list[str]]:
    """Load ``(positive, negative)`` text samples for a concept.

    Args:
        concept: One of the 46 canonical keys (or a legacy alias).
        n_samples: Number of aligned/unpaired samples per side (default: 50).

    Returns:
        ``(positive_texts, negative_texts)`` — two lists of ``n_samples``
        strings each. For arrow ``A->B``, the positive class is ``B``.

    Raises:
        ValueError: if ``concept`` is unknown.
        FileNotFoundError: if the backing dataset JSON is missing.
    """
    resolved = _resolve_concept(concept)
    loader = _LOADERS[resolved]
    pairs = loader(n_samples)
    positives = [p for p, _ in pairs]
    negatives = [n for _, n in pairs]
    return positives, negatives


__all__ = [
    "CODE_LANGS",
    "IF_LANGS",
    "IF_DIALECTS",
    "CONCEPTS",
    "PAIRED_CONCEPTS",
    "ALIASES",
    "DEFAULT_N_SAMPLES",
    "HUMANEVAL_X_DATASET",
    "HUMANEVAL_X_REVISION",
    "WINOGENDER_DATASET",
    "WINOGENDER_REVISION",
    "MATH_JSONL_PATH",
    "FLORES_DATASET",
    "FLORES_REVISION",
    "load_contrastive_texts",
    "load_humaneval_x_pairs",
    "load_humaneval_x_directed_pairs",
    "load_belebele_directed_pairs",
    "load_minif2f_pairs",
    "load_beyondx_pairs",
    "load_math_cot_vs_direct_pairs",
    "load_math_pairs",
    "load_winogender_pairs",
    "load_sst2_pairs",
    "load_refusal_pairs",
    "load_flores_pairs",
    "list_concepts",
    "all_concept_keys",
]
