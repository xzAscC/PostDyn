#!/usr/bin/env python3
"""Materialize the 9 contrastive datasets to ``datasets/*.json``.

Each dataset is streamed **one at a time** and written to disk before the next
one is fetched, so peak memory stays bounded. ``--only NAME`` runs a single
dataset, ``--force`` overwrites existing files.

Datasets::

    humaneval_x   zai-org/humaneval-x          (5 langs, pinned rev)
    minif2f       Tonic/MiniF2F
    beyondx       Johnson0213/BeyondX
    math500       HuggingFaceH4/MATH-500
    belebele      facebook/belebele             (5 dialects)
    winogender    rudinger/winogender-schemas   (templates.tsv pinned rev)
    sst2          glue/sst2
    llm_lat_harmful   LLM-LAT/harmful-dataset
    llm_lat_benign    LLM-LAT/benign-dataset

Usage::

    uv run python experiments/download_datasets.py
    uv run python experiments/download_datasets.py --only belebele
    uv run python experiments/download_datasets.py --only humaneval_x --force
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import sys
import urllib.request
from collections.abc import Callable, Iterable  # noqa: F401

# Make ``src`` importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset  # noqa: E402

from src.dataset_store import (  # noqa: E402
    BELEBELE_FILE,
    BELEBELE_SHARED_KEY,
    BEYONDX_FILE,
    DATASETS_DIR,
    HUMANEVAL_X_FILE,
    HUMANEVAL_X_SHARED_KEY,
    LLM_LAT_BENIGN_FILE,
    LLM_LAT_HARMFUL_FILE,
    MATH500_FILE,
    MINIF2F_FILE,
    N_SHARED_ITEMS,
    SHARED_SAMPLE_SEED,
    SST2_FILE,
    WINOGENDER_FILE,
    dataset_exists,
    dataset_path,
    sample_shared_indices,
    save_json,
    save_shared_ids,
)

# =============================================================================
# Pinned dataset metadata
# =============================================================================

HUMANEVAL_X_DATASET: str = "zai-org/humaneval-x"
HUMANEVAL_X_REVISION: str = "62c78627f3072a1454fa0cb0184737cafe5e4198"
HUMANEVAL_X_LANGS: tuple[str, ...] = ("python", "cpp", "java", "js", "go")

MINIF2F_DATASET: str = "Tonic/MiniF2F"
BEYONDX_DATASET: str = "Johnson0213/BeyondX"
MATH500_DATASET: str = "HuggingFaceH4/MATH-500"
BELEBELE_DATASET: str = "facebook/belebele"
BELEBELE_DIALECTS: tuple[str, ...] = (
    "eng_Latn",
    "fra_Latn",
    "deu_Latn",
    "zho_Hans",
    "jpn_Jpan",
)

WINOGENDER_DATASET: str = "rudinger/winogender-schemas"
WINOGENDER_REVISION: str = "1c7f8b481ad8a234b41e9f76a424d6e856e13f7f"
WINOGENDER_TEMPLATES_URL: str = (
    f"https://raw.githubusercontent.com/{WINOGENDER_DATASET}/"
    f"{WINOGENDER_REVISION}/data/templates.tsv"
)

SST2_DATASET: str = "glue"
SST2_CONFIG: str = "sst2"

LLM_LAT_HARMFUL_DATASET: str = "LLM-LAT/harmful-dataset"
LLM_LAT_BENIGN_DATASET: str = "LLM-LAT/benign-dataset"

TEXT_FIELDS: tuple[str, ...] = ("prompt", "text", "content", "question")


# =============================================================================
# Utilities
# =============================================================================


def _log(msg: str) -> None:
    print(msg, flush=True)


def _strip_code_fences(text: str) -> str:
    if "```" not in text:
        return text
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("```")
    )


def _extract_text(example: dict) -> str:
    for field in TEXT_FIELDS:
        if field in example and example[field]:
            return str(example[field])
    if "messages" in example:
        parts = []
        for msg in example["messages"]:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        if parts:
            return "\n".join(parts)
    raise KeyError(
        f"No text field found. Tried {list(TEXT_FIELDS)} + 'messages'. "
        f"Keys: {list(example.keys())}"
    )


def _should_write(name: str, force: bool) -> bool:
    if force or not dataset_exists(name):
        return True
    _log(f"  [skip] {name} exists (use --force to overwrite)")
    return False


def _release() -> None:
    gc.collect()


# =============================================================================
# HumanEval-X
# =============================================================================


def _humaneval_x_lang_url(lang: str) -> str:
    return (
        f"https://huggingface.co/datasets/{HUMANEVAL_X_DATASET}/resolve/"
        f"{HUMANEVAL_X_REVISION}/data/{lang}/data/humaneval.jsonl"
    )


def _humaneval_x_numeric_id(task_id) -> int:
    suffix = str(task_id).rsplit("/", maxsplit=1)[-1]
    suffix = suffix.replace("HumanEval", "").replace("/", "")
    return int(suffix)


def download_humaneval_x(force: bool = False) -> None:
    name = HUMANEVAL_X_FILE
    _log(f"\n=== {name} ({HUMANEVAL_X_DATASET} @ {HUMANEVAL_X_REVISION[:10]}) ===")
    if not _should_write(name, force):
        return

    languages: dict[str, list[dict]] = {}
    all_id_sets: dict[str, set[int]] = {}
    for lang in HUMANEVAL_X_LANGS:
        url = _humaneval_x_lang_url(lang)
        _log(f"  streaming {lang}: {url}")
        stream = load_dataset("json", data_files=url, split="train", streaming=True)
        items: list[dict] = []
        ids: set[int] = set()
        for row in stream:
            try:
                numeric_id = _humaneval_x_numeric_id(row["task_id"])
            except (KeyError, TypeError, ValueError):
                continue
            prompt = str(row.get("prompt", ""))
            canonical = str(row.get("canonical_solution", ""))
            code = _strip_code_fences(prompt + canonical)
            if not code.strip():
                continue
            items.append(
                {
                    "task_id": str(row["task_id"]),
                    "numeric_id": numeric_id,
                    "prompt": prompt,
                    "canonical_solution": canonical,
                    "code": code,
                }
            )
            ids.add(numeric_id)
        _log(f"    {lang}: {len(items)} items")
        languages[lang] = items
        all_id_sets[lang] = ids

    shared_pool = set.intersection(*all_id_sets.values()) if all_id_sets else set()
    _log(f"  shared numeric ids across all langs: {len(shared_pool)}")
    shared_ids = sample_shared_indices(
        sorted(shared_pool), n=N_SHARED_ITEMS, seed=SHARED_SAMPLE_SEED
    )
    _log(f"  sampled {len(shared_ids)} pinned ids (seed={SHARED_SAMPLE_SEED})")

    # Pin into shared_item_ids.json only if missing or invalid.
    from src.dataset_store import get_shared_ids

    existing = get_shared_ids()
    pinned = existing.get(HUMANEVAL_X_SHARED_KEY)
    if isinstance(pinned, list) and len(pinned) == N_SHARED_ITEMS:
        _log(f"  [pin] reusing existing pinned list of {len(pinned)} ids")
        shared_ids = list(pinned)
    else:
        save_shared_ids({HUMANEVAL_X_SHARED_KEY: shared_ids})
        _log(f"  [pin] wrote new pinned list of {len(shared_ids)} ids")

    payload = {
        "dataset": "humaneval-x",
        "revision": HUMANEVAL_X_REVISION,
        "languages": languages,
    }
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({sum(len(v) for v in languages.values())} total items)")
    del languages, payload
    _release()


# =============================================================================
# MiniF2F
# =============================================================================


def download_minif2f(force: bool = False) -> None:
    name = MINIF2F_FILE
    _log(f"\n=== {name} ({MINIF2F_DATASET}) ===")
    if not _should_write(name, force):
        return

    items: list[dict] = []
    # MiniF2F ships 4 splits; we pool them to collect enough valid pairs.
    splits = ["test", "valid", "dev", "train"]
    seen_keys: set[str] = set()
    for split in splits:
        try:
            stream = load_dataset(MINIF2F_DATASET, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            _log(f"    split {split}: unavailable ({exc})")
            continue
        for row in stream:
            informal = row.get("informal_prefix") or row.get("informal")
            formal = row.get("formal_statement")
            if not informal or not formal:
                continue
            key = str(row.get("name") or row.get("id") or f"{informal}{formal}")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            items.append(
                {
                    "id": key,
                    "informal": str(informal),
                    "formal": str(formal),
                }
            )
        _log(f"    after split {split}: {len(items)} pairs")
        if len(items) >= 200:
            break

    if len(items) < 50:
        raise RuntimeError(f"MiniF2F: only {len(items)} usable pairs (need >=50)")
    payload = {"dataset": "minif2f", "items": items}
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({len(items)} pairs)")
    del items, payload
    _release()


# =============================================================================
# BeyondX
# =============================================================================


def download_beyondx(force: bool = False) -> None:
    name = BEYONDX_FILE
    _log(f"\n=== {name} ({BEYONDX_DATASET}) ===")
    if not _should_write(name, force):
        return

    items: list[dict] = []
    for split in ["train", "test"]:
        try:
            stream = load_dataset(BEYONDX_DATASET, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            _log(f"    split {split}: unavailable ({exc})")
            continue
        idx = 0
        for row in stream:
            problem = row.get("problem")
            equations = row.get("system_of_equations")
            if not problem or equations is None:
                continue
            eq_str = (
                equations
                if isinstance(equations, str)
                else json.dumps(equations, ensure_ascii=False)
            )
            items.append(
                {
                    "id": f"beyondx_{idx}",
                    "problem": str(problem),
                    "equations": eq_str,
                }
            )
            idx += 1
        _log(f"    after split {split}: {len(items)} items")
        if len(items) >= 200:
            break

    if len(items) < 50:
        raise RuntimeError(f"BeyondX: only {len(items)} usable items (need >=50)")
    payload = {"dataset": "beyondx", "items": items}
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({len(items)} items)")
    del items, payload
    _release()


# =============================================================================
# MATH-500
# =============================================================================


def download_math500(force: bool = False) -> None:
    name = MATH500_FILE
    _log(f"\n=== {name} ({MATH500_DATASET}) ===")
    if not _should_write(name, force):
        return

    # Per slides: take FIRST 50 problems in dataset order (no shuffle).
    stream = load_dataset(MATH500_DATASET, split="test", streaming=True)
    items: list[dict] = []
    for row in stream:
        problem = str(row.get("problem", ""))
        solution = str(row.get("solution", ""))
        answer = str(row.get("answer", ""))
        if not problem or not solution:
            continue
        uid = str(
            row.get("unique_id") or row.get("problem_id") or f"math500_{len(items)}"
        )
        direct_answer = f"The answer is {answer}"
        items.append(
            {
                "unique_id": uid,
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "direct_answer": direct_answer,
                "cot_text": problem + "\n" + solution,
                "direct_text": problem + "\n" + direct_answer,
            }
        )
        if len(items) >= 50:
            break

    if len(items) < 50:
        raise RuntimeError(f"MATH-500: only {len(items)} usable rows (need 50)")
    payload = {"dataset": "math500", "items": items}
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({len(items)} items)")
    del items, payload
    _release()


# =============================================================================
# Belebele
# =============================================================================


def download_belebele(force: bool = False) -> None:
    name = BELEBELE_FILE
    _log(f"\n=== {name} ({BELEBELE_DATASET}, 5 dialects) ===")
    if not _should_write(name, force):
        return

    dialects: dict[str, list[dict]] = {}
    key_sets: dict[str, set[tuple[str, int]]] = {}
    for dialect in BELEBELE_DIALECTS:
        _log(f"  streaming {dialect}")
        try:
            stream = load_dataset(
                BELEBELE_DATASET,
                dialect,
                split="test",
                streaming=True,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"    {dialect}: unavailable ({exc})")
            continue
        items: list[dict] = []
        keys: set[tuple[str, int]] = set()
        for row in stream:
            link = row.get("link")
            qnum_raw = row.get("question_number")
            if link is None or qnum_raw is None:
                continue
            try:
                qnum = int(qnum_raw)
            except (TypeError, ValueError):
                continue
            passage = str(row.get("flores_passage", "") or row.get(" passage", ""))
            question = str(row.get("question", ""))
            opts = []
            for i in range(1, 5):
                v = row.get(f"mc_answer{i}")
                if v:
                    opts.append(str(v))
            text_parts = [passage, question]
            text_parts.extend(opts)
            text = "\n".join(p for p in text_parts if p)
            items.append(
                {
                    "link": str(link),
                    "question_number": qnum,
                    "text": text,
                }
            )
            keys.add((str(link), qnum))
        _log(f"    {dialect}: {len(items)} items")
        dialects[dialect] = items
        key_sets[dialect] = keys

    if len(key_sets) < 2:
        raise RuntimeError(f"Belebele: only {len(key_sets)} dialects fetched")
    shared_keys_pool = set.intersection(*key_sets.values())
    _log(f"  shared keys across dialects: {len(shared_keys_pool)}")

    # Pin / reuse pinned list.
    from src.dataset_store import get_shared_ids

    existing = get_shared_ids()
    pinned = existing.get(BELEBELE_SHARED_KEY)
    if isinstance(pinned, list) and len(pinned) == N_SHARED_ITEMS:
        _log(f"  [pin] reusing existing pinned list of {len(pinned)} keys")
        shared_keys = [
            (
                str(entry["link"]),
                int(entry["question_number"]),
            )
            for entry in pinned
            if isinstance(entry, dict)
            and "link" in entry
            and "question_number" in entry
        ]
    else:
        shared_keys = sample_shared_indices(
            sorted(shared_keys_pool), n=N_SHARED_ITEMS, seed=SHARED_SAMPLE_SEED
        )
        save_shared_ids(
            {
                BELEBELE_SHARED_KEY: [
                    {"link": link, "question_number": qnum}
                    for link, qnum in shared_keys
                ]
            }
        )
        _log(f"  [pin] wrote new pinned list of {len(shared_keys)} keys")

    payload = {
        "dataset": "belebele",
        "dialects": dialects,
    }
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({len(dialects)} dialects)")
    del dialects, payload
    _release()


# =============================================================================
# WinoGender (templates.tsv)
# =============================================================================


def download_winogender(force: bool = False) -> None:
    name = WINOGENDER_FILE
    _log(f"\n=== {name} ({WINOGENDER_DATASET} @ {WINOGENDER_REVISION[:10]}) ===")
    if not _should_write(name, force):
        return

    _log(f"  fetching {WINOGENDER_TEMPLATES_URL}")
    with urllib.request.urlopen(WINOGENDER_TEMPLATES_URL, timeout=60) as response:
        content = response.read().decode("utf-8")
    reader = csv.reader(io.StringIO(content), delimiter="\t")
    rows = [row for row in reader]
    if not rows:
        raise RuntimeError("WinoGender templates.tsv is empty")
    header = rows[0]
    items: list[dict] = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        items.append({str(header[i]): row[i] for i in range(len(header))})
    payload = {"dataset": "winogender", "rows": items}
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({len(items)} rows)")
    del items, payload
    _release()


# =============================================================================
# SST-2
# =============================================================================


def download_sst2(force: bool = False) -> None:
    name = SST2_FILE
    _log(f"\n=== {name} ({SST2_DATASET}/{SST2_CONFIG}) ===")
    if not _should_write(name, force):
        return

    # Per MUST DO: stream enough per class so the loader can sample 50 each.
    target_per_class = 5000
    items: list[dict] = []
    counts: dict[int, int] = {0: 0, 1: 0}
    try:
        stream = load_dataset(SST2_DATASET, SST2_CONFIG, split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001
        _log(f"  fallback to stanfordnlp/sst2: {exc}")
        stream = load_dataset("stanfordnlp/sst2", split="train", streaming=True)
    for row in stream:
        sentence = row.get("sentence")
        label = row.get("label")
        if sentence is None or label is None:
            continue
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            continue
        if label_int not in (0, 1):
            continue
        if counts[label_int] >= target_per_class:
            if all(c >= target_per_class for c in counts.values()):
                break
            continue
        counts[label_int] += 1
        items.append({"sentence": str(sentence), "label": label_int})
        if all(c >= target_per_class for c in counts.values()):
            break
    payload = {"dataset": "sst2", "items": items, "counts": counts}
    save_json(dataset_path(name), payload)
    _log(f"  wrote {name} ({len(items)} items, counts={counts})")
    del items, payload
    _release()


# =============================================================================
# LLM-LAT harmful / benign
# =============================================================================


def _download_llm_lat(
    dataset_id: str, file_name: str, target: int, force: bool
) -> None:
    _log(f"\n=== {file_name} ({dataset_id}) ===")
    if not _should_write(file_name, force):
        return
    stream = load_dataset(dataset_id, split="train", streaming=True)
    items: list[str] = []
    for row in stream:
        try:
            text = _extract_text(row)
        except KeyError:
            continue
        if text and text.strip():
            items.append(text)
        if len(items) >= target:
            break
    payload = {"dataset": dataset_id, "texts": items}
    save_json(dataset_path(file_name), payload)
    _log(f"  wrote {file_name} ({len(items)} texts)")
    del items, payload
    _release()


def download_llm_lat_harmful(force: bool = False) -> None:
    _download_llm_lat(
        LLM_LAT_HARMFUL_DATASET, LLM_LAT_HARMFUL_FILE, target=200, force=force
    )


def download_llm_lat_benign(force: bool = False) -> None:
    _download_llm_lat(
        LLM_LAT_BENIGN_DATASET, LLM_LAT_BENIGN_FILE, target=200, force=force
    )


# =============================================================================
# CLI
# =============================================================================

DOWNLOADERS: dict[str, Callable[..., object]] = {
    "humaneval_x": download_humaneval_x,
    "minif2f": download_minif2f,
    "beyondx": download_beyondx,
    "math500": download_math500,
    "belebele": download_belebele,
    "winogender": download_winogender,
    "sst2": download_sst2,
    "llm_lat_harmful": download_llm_lat_harmful,
    "llm_lat_benign": download_llm_lat_benign,
}


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Materialize contrastive datasets to datasets/*.json",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=sorted(DOWNLOADERS.keys()),
        help="Download only this dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    order = [args.only] if args.only else list(DOWNLOADERS.keys())
    failures: list[tuple[str, str]] = []
    for name in order:
        fn = DOWNLOADERS[name]
        try:
            fn(force=args.force)
        except Exception as exc:  # noqa: BLE001
            import traceback

            _log(f"  [FAIL] {name}: {exc}")
            _log(traceback.format_exc())
            failures.append((name, str(exc)))
        _release()

    _log("\n" + "=" * 60)
    if failures:
        _log(f"Done with {len(failures)} failures:")
        for n, err in failures:
            _log(f"  - {n}: {err}")
        # Exit 0 per MUST DO ("Exit 0 on success") but surface failures.
        # The goal says "If a dataset is gated/fails, still write the loader
        # and document; continue other datasets." So do NOT abort.
        _log("(continuing — non-fatal)")
    else:
        _log("All requested downloads completed.")
    _log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
