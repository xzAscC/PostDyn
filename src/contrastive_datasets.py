"""Paired contrastive dataset loaders for concept dynamics.

Four paired steering concepts, each returning aligned (positive, negative)
text pairs loaded from external sources:

    HumanEval-X Python-positive   (python_vs_cpp)
    MATH-500    Concise-positive  (concise_math_reasoning_vs_verbose_math_reasoning)
    FLORES+     French-positive   (french_vs_english_language)
    WinoGender  Female-positive   (female_vs_male_gender)

External dataset access (HuggingFace ``load_dataset`` and HTTP) is patched in
the test-suite, so no network access is required at runtime in tests.
"""

from __future__ import annotations

import csv
import io
import json
import os
import urllib.request
from typing import Callable, Iterable, cast

from datasets import load_dataset


# =============================================================================
# Dataset Configuration
# =============================================================================

HUMANEVAL_X_DATASET: str = "zai-org/humaneval-x"
HUMANEVAL_X_REVISION: str = "62c78627f3072a1454fa0cb0184737cafe5e4198"
_HUMANEVAL_X_BASE_URL: str = (
    f"https://huggingface.co/datasets/{HUMANEVAL_X_DATASET}/resolve/"
    f"{HUMANEVAL_X_REVISION}/data"
)
_HUMANEVAL_X_FILES: dict[str, str] = {
    language: f"{_HUMANEVAL_X_BASE_URL}/{language}/data/humaneval.jsonl"
    for language in ("python", "cpp")
}

FLORES_DATASET: str = "openlanguagedata/flores_plus"
FLORES_REVISION: str = "b3a5298db5721c8a682e7ef00a37fcc9ab522757"
_FLORES_SPLIT: str = "devtest"
_FLORES_FRENCH_CONFIG: str = "fra_Latn"
_FLORES_ENGLISH_CONFIG: str = "eng_Latn"

WINOGENDER_DATASET: str = "rudinger/winogender-schemas"
WINOGENDER_REVISION: str = "1c7f8b481ad8a234b41e9f76a424d6e856e13f7f"
_WINOGENDER_TEMPLATES_URL: str = (
    f"https://raw.githubusercontent.com/{WINOGENDER_DATASET}/"
    f"{WINOGENDER_REVISION}/data/templates.tsv"
)

MATH_JSONL_PATH: str = "data/math-500.jsonl"

PAIRED_CONCEPTS: dict[str, dict[str, str]] = {
    "python_vs_cpp": {
        "name": "HumanEval-X Python-positive",
        "dataset": "HumanEval-X",
        "direction": "Python-C++",
        "positive": "Python",
        "negative": "C++",
    },
    "concise_math_reasoning_vs_verbose_math_reasoning": {
        "name": "MATH concise-positive",
        "dataset": "MATH-500",
        "direction": "Concise-Verbose",
        "positive": "Concise",
        "negative": "Verbose",
    },
    "french_vs_english_language": {
        "name": "FLORES+ French-positive",
        "dataset": "FLORES+",
        "direction": "French-English",
        "positive": "French",
        "negative": "English",
    },
    "female_vs_male_gender": {
        "name": "WinoGender Female-positive",
        "dataset": "WinoGender",
        "direction": "Female-Male",
        "positive": "Female",
        "negative": "Male",
    },
}

# Text fields tried in priority order by ``_extract_text``. Different datasets
# use different schemas; this makes the helper robust without hard-coding.
_TEXT_FIELDS: tuple[str, ...] = ("text", "prompt", "content", "question")


# =============================================================================
# Text Extraction Helpers
# =============================================================================


def _extract_text(example: dict) -> str:
    """Extract a plain-text string from a single dataset row.

    Tries common field names in priority order, then falls back to the
    conversational ``messages`` format (concatenating every message content).

    Raises:
        KeyError: If no recognized text field is found.
    """
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


def _stream_n_samples(dataset: Iterable[dict], n: int) -> list[str]:
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
    """Remove Markdown code-fence marker lines (```...```) from a code string.

    Non-fenced text is returned untouched so trailing newlines are preserved
    for already-clean inputs.
    """
    if "```" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("```")
    )


# =============================================================================
# HumanEval-X Python vs C++
# =============================================================================


def _humaneval_task_id(example: dict) -> int:
    try:
        return int(str(example["task_id"]).rsplit("/", maxsplit=1)[-1])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid HumanEval-X task_id: {example.get('task_id')!r}"
        ) from exc


def _index_humaneval_solutions(dataset: Iterable[dict]) -> dict[int, str]:
    solutions: dict[int, str] = {}
    for example in dataset:
        task_id = _humaneval_task_id(example)
        if task_id in solutions:
            raise ValueError(f"Duplicate HumanEval-X task ID: {task_id}")
        try:
            code = str(example["prompt"]) + str(example["canonical_solution"])
        except KeyError as exc:
            raise ValueError(
                f"HumanEval-X task {task_id} is missing {exc.args[0]!r}"
            ) from exc
        code = _strip_code_fences(code)
        if not code.strip():
            raise ValueError(f"HumanEval-X task {task_id} has empty code")
        solutions[task_id] = code
    return solutions


def load_humaneval_x_pairs(n_samples: int = 50) -> list[tuple[str, str]]:
    """Load aligned ``(Python-positive, C++-negative)`` complete solutions.

    Pulls the pinned raw JSONL files for both languages from
    ``HUMANEVAL_X_DATASET`` at ``HUMANEVAL_X_REVISION`` in streaming mode,
    indexes them by numeric task id, and returns the first ``n_samples``
    shared ids as ``(python, cpp)`` pairs.

    Raises:
        ValueError: If fewer than ``n_samples`` shared task ids are available.
    """
    if n_samples <= 0:
        return []

    datasets: dict[str, Iterable[dict]] = {
        language: cast(
            Iterable[dict],
            load_dataset(
                "json",
                data_files=data_file,
                split="train",
                streaming=True,
            ),
        )
        for language, data_file in _HUMANEVAL_X_FILES.items()
    }
    python_by_id = _index_humaneval_solutions(datasets["python"])
    cpp_by_id = _index_humaneval_solutions(datasets["cpp"])
    shared_ids = sorted(set(python_by_id) & set(cpp_by_id))
    if len(shared_ids) < n_samples:
        raise ValueError(
            f"Requested {n_samples} aligned HumanEval-X pairs, "
            f"but only {len(shared_ids)} were aligned"
        )
    return [
        (python_by_id[task_id], cpp_by_id[task_id])
        for task_id in shared_ids[:n_samples]
    ]


# =============================================================================
# MATH-500 Concise vs Verbose
# =============================================================================


def load_math_pairs(n_samples: int = 50) -> list[tuple[str, str]]:
    """Load aligned ``(Concise-positive, Verbose-negative)`` MATH solutions.

    Reads rows from ``MATH_JSONL_PATH`` (one JSON object per line). Each row
    must expose ``problem``, ``concise_solution`` and ``verbose_solution``
    fields. Returned strings combine the full problem statement with the
    respective solution so both sides carry the question context.

    Raises:
        ValueError: If fewer than ``n_samples`` rows are available.
    """
    if n_samples <= 0:
        return []

    rows: list[dict] = []
    with open(MATH_JSONL_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) < n_samples:
        raise ValueError(
            f"Requested {n_samples} MATH pairs but only {len(rows)} rows "
            f"are available (need at least 50)"
        )

    pairs: list[tuple[str, str]] = []
    for row in rows[:n_samples]:
        problem = str(row["problem"])
        concise = f"{problem}\n{row['concise_solution']}"
        verbose = f"{problem}\n{row['verbose_solution']}"
        pairs.append((concise, verbose))
    return pairs


# =============================================================================
# FLORES+ French vs English
# =============================================================================


def _load_flores_split(config: str) -> dict[int, str]:
    stream = cast(
        Iterable[dict],
        load_dataset(
            FLORES_DATASET,
            config,
            split=_FLORES_SPLIT,
            streaming=True,
            revision=FLORES_REVISION,
            token=os.environ.get("HF_TOKEN"),
        ),
    )
    indexed: dict[int, str] = {}
    for row in stream:
        row_id = int(row["id"])
        if row_id in indexed:
            raise ValueError(f"Duplicate FLORES+ row ID: {row_id}")
        indexed[row_id] = str(row["text"])
    return indexed


_FLORES_PAIR_CACHE: dict[int, list[tuple[str, str]]] = {}


def load_flores_pairs(n_samples: int = 50) -> list[tuple[str, str]]:
    """Load aligned ``(French-positive, English-negative)`` FLORES+ devtest pairs.

    Pulls the ``fra_Latn`` and ``eng_Latn`` configs of ``FLORES_DATASET`` at
    ``FLORES_REVISION``, indexes both by numeric row id, and returns the
    first ``n_samples`` shared ids as ``(french, english)`` pairs.

    Raises:
        ValueError: If fewer than ``n_samples`` aligned rows are available.
    """
    if n_samples <= 0:
        return []

    cached = _FLORES_PAIR_CACHE.get(n_samples)
    if cached is not None:
        return list(cached)

    french_by_id = _load_flores_split(_FLORES_FRENCH_CONFIG)
    english_by_id = _load_flores_split(_FLORES_ENGLISH_CONFIG)
    shared_ids = sorted(set(french_by_id) & set(english_by_id))
    if len(shared_ids) < n_samples:
        raise ValueError(
            f"Requested {n_samples} aligned FLORES+ pairs, "
            f"but only {len(shared_ids)} were aligned"
        )
    pairs = [
        (french_by_id[row_id], english_by_id[row_id])
        for row_id in shared_ids[:n_samples]
    ]
    _FLORES_PAIR_CACHE[n_samples] = pairs
    return list(pairs)


# =============================================================================
# WinoGender Female vs Male
# =============================================================================


_WINOGENDER_PRONOUNS = {
    "$NOM_PRONOUN": ("she", "he"),
    "$POSS_PRONOUN": ("her", "his"),
    "$ACC_PRONOUN": ("her", "him"),
}


def _allocate_winogender_forms(
    capacities: dict[str, int], n_samples: int
) -> dict[str, int]:
    pairs_per_answer = n_samples // 2
    forms = [form for form in _WINOGENDER_PRONOUNS if capacities.get(form, 0) > 0]
    if sum(capacities.get(form, 0) for form in forms) < pairs_per_answer:
        raise ValueError(
            f"Requested {n_samples} balanced WinoGender pairs, but the pinned "
            "templates do not contain enough matched answer/form rows"
        )

    allocations = {form: 0 for form in _WINOGENDER_PRONOUNS}
    seeded_forms = forms[:pairs_per_answer]
    for form in seeded_forms:
        allocations[form] = 1

    remaining = pairs_per_answer - len(seeded_forms)
    if remaining == 0:
        return allocations

    remaining_capacities = {
        form: capacities[form] - allocations[form] for form in forms
    }
    total_remaining_capacity = sum(remaining_capacities.values())
    raw_shares = {
        form: remaining * remaining_capacities[form] / total_remaining_capacity
        for form in forms
    }
    floor_shares = {form: int(raw_shares[form]) for form in forms}
    for form, count in floor_shares.items():
        allocations[form] += count

    leftover = remaining - sum(floor_shares.values())
    by_remainder = sorted(
        forms,
        key=lambda form: (raw_shares[form] - floor_shares[form], capacities[form]),
        reverse=True,
    )
    for form in by_remainder:
        if leftover == 0:
            break
        if allocations[form] < capacities[form]:
            allocations[form] += 1
            leftover -= 1

    if leftover:
        raise ValueError("Could not allocate the requested WinoGender form quotas")
    return allocations


def load_winogender_pairs(n_samples: int = 50) -> list[tuple[str, str]]:
    """Load aligned ``(Female-positive, Male-negative)`` WinoGender pairs.

    Fetches ``templates.tsv`` from the pinned ``WINOGENDER_DATASET`` revision
    via ``urllib.request.urlopen``. The canonical columns are occupation,
    participant, answer, and sentence. Template placeholders are instantiated
    before nominative, possessive-determiner, and accusative variants are paired.

    Every pronoun-form quota is split equally between ``answer=0`` and
    ``answer=1`` rows. Quotas are proportional to the balanced capacity of the
    pinned templates, with at least one row per answer for every available form.
    For the default 50 pairs this yields 36 she/he, 10 her/his, and 4 her/him.

    Raises:
        ValueError: If ``n_samples`` is odd or the balanced form buckets cannot
            provide the requested number of pairs.
    """
    if n_samples <= 0:
        return []
    if n_samples % 2:
        raise ValueError("WinoGender n_samples must be even for balanced pairing")

    with urllib.request.urlopen(_WINOGENDER_TEMPLATES_URL, timeout=30) as response:
        content = response.read().decode("utf-8")

    reader = csv.reader(io.StringIO(content), delimiter="\t")
    rows = list(reader)
    if not rows:
        raise ValueError("WinoGender templates.tsv is empty")
    header = rows[0]
    try:
        occupation_idx = header.index("occupation(0)")
        participant_idx = header.index("other-participant(1)")
        answer_idx = header.index("answer")
        sentence_idx = header.index("sentence")
    except ValueError as exc:
        raise ValueError(
            f"WinoGender templates.tsv missing required column: {exc}"
        ) from exc

    buckets: dict[str, dict[int, list[tuple[str, str]]]] = {
        form: {0: [], 1: []} for form in _WINOGENDER_PRONOUNS
    }
    last_idx = max(occupation_idx, participant_idx, answer_idx, sentence_idx)
    for row in rows[1:]:
        if len(row) <= last_idx:
            continue
        try:
            answer = int(row[answer_idx])
        except (TypeError, ValueError):
            continue
        if answer not in (0, 1):
            continue
        sentence = row[sentence_idx]
        sentence = sentence.replace("$OCCUPATION", row[occupation_idx])
        sentence = sentence.replace("$PARTICIPANT", row[participant_idx])
        forms = [form for form in _WINOGENDER_PRONOUNS if form in sentence]
        if len(forms) != 1:
            continue
        form = forms[0]
        if sentence.count(form) != 1:
            continue
        female_pronoun, male_pronoun = _WINOGENDER_PRONOUNS[form]
        female = sentence.replace(form, female_pronoun)
        male = sentence.replace(form, male_pronoun)
        buckets[form][answer].append((female, male))

    capacities = {
        form: min(len(by_answer[0]), len(by_answer[1]))
        for form, by_answer in buckets.items()
    }
    allocations = _allocate_winogender_forms(capacities, n_samples)
    selected: list[tuple[str, str]] = []
    for form in _WINOGENDER_PRONOUNS:
        per_answer = allocations[form]
        for answer in (0, 1):
            selected.extend(buckets[form][answer][:per_answer])
    return selected


# =============================================================================
# Public API: Contrastive Text Loading
# =============================================================================


_PAIRED_LOADERS: dict[str, Callable[[int], list[tuple[str, str]]]] = {
    "python_vs_cpp": load_humaneval_x_pairs,
    "concise_math_reasoning_vs_verbose_math_reasoning": load_math_pairs,
    "french_vs_english_language": load_flores_pairs,
    "female_vs_male_gender": load_winogender_pairs,
}


def load_contrastive_texts(
    concept: str,
    n_samples: int = 50,
) -> tuple[list[str], list[str]]:
    """Load ``(positive, negative)`` text samples for a paired concept.

    Args:
        concept: One of the keys defined in ``PAIRED_CONCEPTS``.
        n_samples: Number of aligned pairs per side (default: 50).

    Returns:
        ``(positive_texts, negative_texts)`` — two lists of ``n_samples``
        strings. The first element of each underlying pair is treated as the
        positive direction (Python / Concise / French / Female).

    Raises:
        ValueError: If ``concept`` is not a supported paired concept.
    """
    if concept not in PAIRED_CONCEPTS:
        raise ValueError(
            f"Unknown concept {concept!r}. "
            f"Supported paired concepts: {sorted(PAIRED_CONCEPTS)}"
        )
    pairs = _PAIRED_LOADERS[concept](n_samples)
    positives = [positive for positive, _ in pairs]
    negatives = [negative for _, negative in pairs]
    return positives, negatives
