"""Local dataset store for materialized HF datasets.

Datasets are streamed one-at-a-time by ``experiments/download_datasets.py``
and written as JSON files under ``datasets/``. Loaders in
``src/contrastive_datasets.py`` read these JSON files offline.

Layout::

    datasets/
        humaneval_x.json
        minif2f.json
        beyondx.json
        math500.json
        belebele.json
        winogender.json
        sst2.json
        llm_lat_harmful.json
        llm_lat_benign.json
        shared_item_ids.json     # pinned samples (code task ids, belebele keys)
        .gitkeep

The ``shared_item_ids.json`` file pins the *deterministic* 50-item subsamples
used to build all 20 Code and all 20 IF directed concept pairs so that every
pair shares the same underlying items.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
SHARED_IDS_PATH: Path = DATASETS_DIR / "shared_item_ids.json"

# =============================================================================
# Pinned dataset file names
# =============================================================================

HUMANEVAL_X_FILE: str = "humaneval_x.json"
MINIF2F_FILE: str = "minif2f.json"
BEYONDX_FILE: str = "beyondx.json"
MATH500_FILE: str = "math500.json"
BELEBELE_FILE: str = "belebele.json"
WINOGENDER_FILE: str = "winogender.json"
SST2_FILE: str = "sst2.json"
LLM_LAT_HARMFUL_FILE: str = "llm_lat_harmful.json"
LLM_LAT_BENIGN_FILE: str = "llm_lat_benign.json"

ALL_DATASET_FILES: tuple[str, ...] = (
    HUMANEVAL_X_FILE,
    MINIF2F_FILE,
    BEYONDX_FILE,
    MATH500_FILE,
    BELEBELE_FILE,
    WINOGENDER_FILE,
    SST2_FILE,
    LLM_LAT_HARMFUL_FILE,
    LLM_LAT_BENIGN_FILE,
)

# =============================================================================
# Pinned samplers
# =============================================================================

#: Seed for every ``random.Random`` subsample in this project.
SHARED_SAMPLE_SEED: int = 42

#: Default number of shared items sampled per domain.
N_SHARED_ITEMS: int = 50


def _path_for(name: str) -> Path:
    """Return the absolute path of a dataset JSON file by short name."""
    return DATASETS_DIR / name


# =============================================================================
# I/O helpers
# =============================================================================


def load_json(path: str | Path) -> dict:
    """Load a JSON object from ``path`` as a plain dict.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, obj: dict) -> None:
    """Write ``obj`` to ``path`` as pretty JSON (atomic-ish, UTF-8).

    The parent directory is created on demand. Existing files are overwritten.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, p)


def dataset_path(name: str) -> Path:
    """Public alias for :func:`_path_for`."""
    return _path_for(name)


def dataset_exists(name: str) -> bool:
    """Return True if ``datasets/<name>`` exists."""
    return _path_for(name).exists()


def load_dataset_json(name: str) -> dict:
    """Load ``datasets/<name>`` and return the parsed object."""
    return load_json(_path_for(name))


# =============================================================================
# Shared item IDs (pinned subsamples for Code and IF)
# =============================================================================

#: Key in shared_item_ids.json -> list[int] of HumanEval-X numeric task ids.
HUMANEVAL_X_SHARED_KEY: str = "humaneval_x_task_ids"
#: Key in shared_item_ids.json -> list[{"link", "question_number"}] for Belebele.
BELEBELE_SHARED_KEY: str = "belebele_keys"


def _empty_shared() -> dict:
    return {HUMANEVAL_X_SHARED_KEY: [], BELEBELE_SHARED_KEY: []}


def get_shared_ids() -> dict:
    """Load the pinned shared item-id registry from disk.

    Returns a dict with two keys::

        {
            "humaneval_x_task_ids": [int, ...],   # length N_SHARED_ITEMS
            "belebele_keys": [{"link": str, "question_number": int}, ...],
        }

    Missing files yield an empty registry with empty lists.
    """
    if not SHARED_IDS_PATH.exists():
        return _empty_shared()
    try:
        data = load_json(SHARED_IDS_PATH)
    except (OSError, json.JSONDecodeError):
        return _empty_shared()
    result = _empty_shared()
    for key in (HUMANEVAL_X_SHARED_KEY, BELEBELE_SHARED_KEY):
        value = data.get(key)
        if isinstance(value, list):
            result[key] = value
    return result


def save_shared_ids(obj: dict) -> None:
    """Persist the shared item-id registry, merging with any existing keys.

    Top-level keys present on disk but not in ``obj`` are preserved.
    """
    existing: dict = {}
    if SHARED_IDS_PATH.exists():
        try:
            existing = load_json(SHARED_IDS_PATH)
        except (OSError, json.JSONDecodeError):
            existing = {}
    existing.update(obj)
    save_json(SHARED_IDS_PATH, existing)


# =============================================================================
# Samplers
# =============================================================================


def sample_shared_indices(
    pool,
    n: int = N_SHARED_ITEMS,
    seed: int = SHARED_SAMPLE_SEED,
) -> list:
    """Deterministically sample ``n`` items from ``pool`` preserving order.

    ``pool`` may be either a list (order preserved) or a set/frozenset (sorted
    first to keep determinism). Returns the selected items in their original
    order within ``pool``.
    """
    if isinstance(pool, (set, frozenset)):
        ordered = sorted(pool, key=_sort_key_mixed)
    else:
        ordered = list(pool)
    if len(ordered) <= n:
        return ordered
    rng = random.Random(seed)
    # Sample indices, then return pool items in original order.
    selected_idx = sorted(rng.sample(range(len(ordered)), n))
    return [ordered[i] for i in selected_idx]


def _sort_key_mixed(item):
    """Sort helper for mixed int / tuple / str keys (HumanEval-X ids, Belebele keys)."""
    if isinstance(item, int):
        return (0, item, "")
    if isinstance(item, tuple):
        return (1, 0, "|".join(str(x) for x in item))
    if isinstance(item, dict):
        return (1, 0, "|".join(str(v) for v in item.values()))
    return (2, 0, str(item))


def humaneval_x_shared_ids() -> list[int]:
    """Return the pinned 50 HumanEval-X numeric task ids (empty if unpinned)."""
    return list(get_shared_ids().get(HUMANEVAL_X_SHARED_KEY, []))


def belebele_shared_keys() -> list[dict]:
    """Return the pinned 50 Belebele (link, question_number) keys (empty if unpinned)."""
    return list(get_shared_ids().get(BELEBELE_SHARED_KEY, []))
