"""Domain prompt loaders for differential-subspace experiments.

Three domains used as concept / reference groups (PostDyn.tex):

* ``math``  — competition-style math prompts
* ``code``  — programming prompts
* ``text``  — general natural-language prompts

Sources (in priority order per domain):
1. Local JSON under ``datasets/`` (offline, preferred)
2. HuggingFace Dolci RL-Zero datasets when local materialization is missing

Sampling is deterministic via ``SHARED_SAMPLE_SEED``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from src.dataset_store import (
    DATASETS_DIR,
    HUMANEVAL_X_FILE,
    MATH500_FILE,
    SHARED_SAMPLE_SEED,
    SST2_FILE,
    load_json,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_N_DOMAIN_SAMPLES: int = 100
MEMO_DOMAIN_NAMES: tuple[str, ...] = (
    "math",
    "code",
    "instruction_following",
    "general_reasoning",
    "wikitext",
)
# Historical callers use ``text`` for Dolci General / local SST-2.
DOMAIN_NAMES: tuple[str, ...] = ("math", "code", "text")

#: Concept pairs for the Math-trajectory differential-subspace experiment.
#: Each entry is (concept_name, concept_domain, ref_domain).
DEFAULT_CONCEPT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("math_vs_code", "math", "code"),
    ("math_vs_text", "math", "text"),
)

DOLCI_HF_IDS: dict[str, str] = {
    "math": "allenai/Dolci-RL-Zero-Math-7B",
    "code": "allenai/Dolci-RL-Zero-Code-7B",
    "text": "allenai/Dolci-RL-Zero-General-7B",
    "instruction_following": "allenai/Dolci-RL-Zero-IF-7B",
    "general_reasoning": "allenai/Dolci-RL-Zero-General-7B",
}
DOLCI_HF_REVISIONS: dict[str, str] = {
    "math": "93b5e498577804a93fb11bfe8821428f8535f2d8",
    "text": "7e415e2b556bb74b3ce3924708772d43145ed6a3",
    "code": "main",
    "instruction_following": "main",
    "general_reasoning": "main",
}

_LOCAL_CACHE_DIR: Path = DATASETS_DIR / "domain_prompts"
_USER_PREFIX_RE = re.compile(r"^\s*user:\s*", re.IGNORECASE)
WIKITEXT_HF_ID: str = "Salesforce/wikitext"
WIKITEXT_CONFIG: str = "wikitext-103-raw-v1"
WIKITEXT_SPLIT: str = "train"
WIKITEXT_HF_REVISION: str = "main"
_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class DomainPromptSelection:
    """A deterministic prompt draw and the provenance needed to reproduce it."""

    domain: str
    prompts: tuple[str, ...]
    source: dict[str, str]
    fingerprint: str

    def as_list(self) -> list[str]:
        return list(self.prompts)


# =============================================================================
# Public API
# =============================================================================


def load_domain_prompts(
    domain: str,
    n_samples: int = DEFAULT_N_DOMAIN_SAMPLES,
    *,
    seed: int = SHARED_SAMPLE_SEED,
    prefer_local: bool = True,
    allow_hf: bool = True,
    streaming: bool = True,
) -> list[str]:
    """Load ``n_samples`` raw user prompts for a domain.

    Args:
        domain: One of ``math`` / ``code`` / ``text``.
        n_samples: Number of prompts (default 100).
        seed: RNG seed for deterministic subsampling.
        prefer_local: Try local JSON caches first.
        allow_hf: Fall back to HuggingFace Dolci datasets.

    Returns:
        List of plain prompt strings (no chat template applied).
    """
    domain = _canonical_domain(domain)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if streaming and allow_hf and domain in MEMO_DOMAIN_NAMES:
        return load_domain_prompt_selection(
            domain, n_samples=n_samples, seed=seed, prefer_local=prefer_local
        ).as_list()

    loaders: list[Callable[[], list[str]]] = []
    if prefer_local:
        loaders.append(lambda: _load_cached_domain(domain))
        loaders.append(lambda: _load_builtin_local(domain))
    if allow_hf:
        loaders.append(lambda: _load_dolci_hf(domain))

    last_err: Optional[Exception] = None
    pool: list[str] = []
    for loader in loaders:
        try:
            pool = _normalize_prompts(loader())
        except Exception as exc:  # noqa: BLE001 — try next source
            last_err = exc
            pool = []
        if len(pool) >= n_samples:
            break

    if len(pool) < n_samples:
        detail = f" last error: {last_err}" if last_err is not None else ""
        raise ValueError(
            f"Domain {domain!r}: need {n_samples} prompts, only found {len(pool)}.{detail}"
        )

    domain_seed = int.from_bytes(
        hashlib.sha256(domain.encode("utf-8")).digest()[:8], "big"
    )
    rng = random.Random(seed + domain_seed)
    # Stable order then sample without replacement.
    unique = list(dict.fromkeys(pool))
    if len(unique) < n_samples:
        raise ValueError(
            f"Domain {domain!r}: only {len(unique)} unique prompts after dedup "
            f"(need {n_samples})"
        )
    return rng.sample(unique, n_samples)


def load_domain_prompt_selection(
    domain: str,
    n_samples: int = 1000,
    *,
    seed: int = SHARED_SAMPLE_SEED,
    prefer_local: bool = True,
) -> DomainPromptSelection:
    """Stream a memo source and return a deterministic provenance-bound draw."""
    domain = _canonical_domain(domain)
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    if prefer_local and domain in DOMAIN_NAMES:
        for loader in (_load_cached_domain, _load_builtin_local):
            try:
                prompts = _normalize_prompts(loader(domain))
            except Exception:  # noqa: BLE001 - fall through to HF
                continue
            if len(set(prompts)) >= n_samples:
                selected = _select_prompts(prompts, domain, n_samples, seed)
                return _make_selection(domain, selected, _local_source(domain))

    if domain == "wikitext":
        revision = resolve_hub_dataset_revision(WIKITEXT_HF_ID, WIKITEXT_HF_REVISION)
        stream = _load_wikitext_stream(revision=revision)
        source = {
            "kind": "huggingface",
            "hf_id": WIKITEXT_HF_ID,
            "config": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "revision": revision,
        }
    else:
        revision = resolve_hub_dataset_revision(
            DOLCI_HF_IDS[domain], DOLCI_HF_REVISIONS[domain]
        )
        stream = _load_dolci_hf_stream(domain, revision=revision)
        source = {
            "kind": "dolci",
            "hf_id": DOLCI_HF_IDS[domain],
            "revision": revision,
        }
    selected = _select_prompts(stream, domain, n_samples, seed)
    if len(selected) < n_samples:
        raise ValueError(
            f"Domain {domain!r}: need {n_samples} unique prompts, only found {len(selected)}"
        )
    return _make_selection(domain, selected, source)


def load_dolci_domain_prompts(
    domain: str,
    n_samples: int = DEFAULT_N_DOMAIN_SAMPLES,
    *,
    seed: int = SHARED_SAMPLE_SEED,
) -> list[str]:
    domain = domain.lower().strip()
    if domain not in (
        "math",
        "text",
        "code",
        "instruction_following",
        "general_reasoning",
    ):
        raise ValueError(f"Strict Dolci source does not support {domain!r}")
    if domain not in ("math", "text"):
        return load_domain_prompt_selection(
            domain, n_samples=n_samples, seed=seed, prefer_local=False
        ).as_list()
    pool = _normalize_prompts(
        _load_dolci_hf(domain, revision=DOLCI_HF_REVISIONS[domain])
    )
    unique = list(dict.fromkeys(pool))
    if len(unique) < n_samples:
        raise ValueError(
            f"Dolci domain {domain!r}: need {n_samples} prompts, only found {len(unique)}"
        )
    domain_seed = int.from_bytes(
        hashlib.sha256(domain.encode("utf-8")).digest()[:8], "big"
    )
    return random.Random(seed + domain_seed).sample(unique, n_samples)


def load_concept_pair_texts(
    concept_name: str,
    concept_domain: str,
    ref_domain: str,
    n_samples: int = DEFAULT_N_DOMAIN_SAMPLES,
    **kwargs,
) -> tuple[list[str], list[str]]:
    """Return ``(concept_prompts, ref_prompts)`` for one differential pair."""
    pos = load_domain_prompts(concept_domain, n_samples=n_samples, **kwargs)
    neg = load_domain_prompts(ref_domain, n_samples=n_samples, **kwargs)
    return pos, neg


def load_all_default_pairs(
    n_samples: int = DEFAULT_N_DOMAIN_SAMPLES,
    **kwargs,
) -> dict[str, tuple[list[str], list[str]]]:
    """Load both default pairs: math_vs_code and math_vs_text."""
    # Share domain draws so math prompts are identical across both pairs.
    domains = {
        d: load_domain_prompts(d, n_samples=n_samples, **kwargs) for d in DOMAIN_NAMES
    }
    out: dict[str, tuple[list[str], list[str]]] = {}
    for name, c_dom, r_dom in DEFAULT_CONCEPT_PAIRS:
        out[name] = (list(domains[c_dom]), list(domains[r_dom]))
    return out


def materialize_domain_cache(
    domain: str,
    prompts: list[str],
) -> Path:
    """Write prompts to ``datasets/domain_prompts/{domain}.json``."""
    _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _LOCAL_CACHE_DIR / f"{domain}.json"
    payload = {
        "domain": domain,
        "n": len(prompts),
        "prompts": list(prompts),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


# =============================================================================
# Local sources
# =============================================================================


def _canonical_domain(domain: str) -> str:
    value = domain.lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "if": "instruction_following",
        "instruction": "instruction_following",
        "general": "general_reasoning",
        "reasoning": "general_reasoning",
        "wiki_text": "wikitext",
    }
    value = aliases.get(value, value)
    valid = set(DOMAIN_NAMES) | set(MEMO_DOMAIN_NAMES)
    if value not in valid:
        raise ValueError(
            f"Unknown domain {domain!r}; expected one of {tuple(sorted(valid))}"
        )
    return value


def _prompt_fingerprint(prompts: Iterable[str]) -> str:
    payload = json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_selection(
    domain: str, prompts: Iterable[str], source: dict[str, str]
) -> DomainPromptSelection:
    ordered = tuple(prompts)
    return DomainPromptSelection(
        domain, ordered, dict(source), _prompt_fingerprint(ordered)
    )


def _local_source(domain: str) -> dict[str, str]:
    return {"kind": "local", "domain": domain}


def _select_prompts(
    prompts: Iterable[str], domain: str, n_samples: int, seed: int
) -> list[str]:
    domain_seed = hashlib.sha256(domain.encode("utf-8")).digest()
    heap: list[tuple[int, str, str]] = []
    selected: set[str] = set()
    for raw in prompts:
        text = _strip_user_prefix(str(raw))
        if not text or text in selected:
            continue
        digest = hashlib.sha256(
            seed.to_bytes(8, "big", signed=True) + domain_seed + text.encode("utf-8")
        ).digest()
        priority = int.from_bytes(digest, "big")
        candidate = (-priority, text, text)
        if len(heap) < n_samples:
            heapq.heappush(heap, candidate)
            selected.add(text)
        elif candidate > heap[0]:
            _, _, evicted = heapq.heapreplace(heap, candidate)
            selected.discard(evicted)
            selected.add(text)
    return [item[1] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def resolve_hub_dataset_revision(repo_id: str, revision: str) -> str:
    """Return the immutable Hub commit SHA for a dataset revision.

    A SHA is accepted directly after validation. Every symbolic revision is
    resolved through the Hub before it can reach ``load_dataset``; failures
    therefore prevent mutable or unknown refs from being streamed.
    """
    if _SHA40_RE.fullmatch(revision):
        return revision.lower()
    try:
        from huggingface_hub import HfApi

        resolved = HfApi().dataset_info(repo_id, revision=revision).sha
    except Exception as exc:  # noqa: BLE001 - fail closed with context
        raise RuntimeError(
            f"Could not resolve immutable revision for dataset {repo_id!r} "
            f"at {revision!r}"
        ) from exc
    if not isinstance(resolved, str) or _SHA40_RE.fullmatch(resolved) is None:
        raise RuntimeError(
            f"Hub returned an invalid commit SHA for dataset {repo_id!r}: {resolved!r}"
        )
    return resolved.lower()


def _load_dolci_hf_stream(domain: str, *, revision: str | None = None) -> Iterator[str]:
    from datasets import load_dataset

    hf_id = DOLCI_HF_IDS[domain]
    resolved_revision = resolve_hub_dataset_revision(
        hf_id, revision or DOLCI_HF_REVISIONS[domain]
    )
    stream = load_dataset(
        hf_id, split="train", streaming=True, revision=resolved_revision
    )
    for row in stream:
        if isinstance(row, dict):
            text = _extract_prompt_from_row(
                {str(key): value for key, value in row.items()}
            )
            if text:
                yield text


def _load_wikitext_stream(*, revision: str | None = None) -> Iterator[str]:
    from datasets import load_dataset

    resolved_revision = resolve_hub_dataset_revision(
        WIKITEXT_HF_ID, revision or WIKITEXT_HF_REVISION
    )
    stream = load_dataset(
        WIKITEXT_HF_ID,
        WIKITEXT_CONFIG,
        split=WIKITEXT_SPLIT,
        streaming=True,
        revision=resolved_revision,
    )
    for row in stream:
        if isinstance(row, dict):
            text = _extract_prompt_from_row(
                {str(key): value for key, value in row.items()}
            )
            if text:
                yield text


def load_memo_domain_prompts(
    domain: str, n_samples: int = 1000, *, seed: int = SHARED_SAMPLE_SEED
) -> DomainPromptSelection:
    return load_domain_prompt_selection(domain, n_samples=n_samples, seed=seed)


def _load_cached_domain(domain: str) -> list[str]:
    path = _LOCAL_CACHE_DIR / f"{domain}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = load_json(path)
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Empty domain cache: {path}")
    return [str(p) for p in prompts]


def _load_builtin_local(domain: str) -> list[str]:
    if domain == "math":
        return _from_math500()
    if domain == "code":
        return _from_humaneval_x()
    if domain == "text":
        return _from_sst2()
    raise ValueError(domain)


def _from_math500() -> list[str]:
    data = load_json(DATASETS_DIR / MATH500_FILE)
    items = data.get("items") or data.get("data") or []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("problem") or item.get("prompt") or item.get("question")
        if text:
            out.append(str(text).strip())
    return out


def _from_humaneval_x() -> list[str]:
    data = load_json(DATASETS_DIR / HUMANEVAL_X_FILE)
    out: list[str] = []

    languages = data.get("languages")
    if isinstance(languages, dict):
        for lang in ("python", "py"):
            items = languages.get(lang) or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("prompt") or item.get("declaration") or item.get("code")
                if text:
                    out.append(str(text).strip())
            if out:
                return out
        for items in languages.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("prompt") or item.get("declaration") or item.get("code")
                if text:
                    out.append(str(text).strip())
        if out:
            return out

    items = data.get("items") or data.get("data") or []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = (
            item.get("prompt")
            or item.get("declaration")
            or item.get("canonical_prompt")
            or item.get("text")
        )
        if text:
            out.append(str(text).strip())
    return out


def _from_sst2() -> list[str]:
    data = load_json(DATASETS_DIR / SST2_FILE)
    items = data.get("items") or data.get("data") or []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("sentence") or item.get("text") or item.get("prompt")
        if text:
            out.append(str(text).strip())
    return out


# =============================================================================
# HuggingFace Dolci
# =============================================================================


def _load_dolci_hf(domain: str, *, revision: str | None = None) -> list[str]:
    from datasets import load_dataset

    hf_id = DOLCI_HF_IDS[domain]
    resolved_revision = resolve_hub_dataset_revision(
        hf_id, revision or DOLCI_HF_REVISIONS[domain]
    )
    ds = load_dataset(hf_id, split="train", revision=resolved_revision)
    out: list[str] = []
    for raw in ds:
        row_dict: dict[str, object] = (
            {str(key): value for key, value in raw.items()}
            if isinstance(raw, dict)
            else {}
        )
        text = _extract_prompt_from_row(row_dict)
        if text:
            out.append(text)
    if out:
        # Cache for offline reuse.
        try:
            materialize_domain_cache(domain, out)
        except OSError:
            pass
    return out


def _extract_prompt_from_row(row: dict[str, object]) -> str:
    # Math Dolci: messages[0].content or prompt
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict) and first.get("content"):
            return _strip_user_prefix(str(first["content"]))

    prompt = row.get("prompt")
    if prompt is not None:
        return _strip_user_prefix(str(prompt))

    content = row.get("content") or row.get("text") or row.get("question")
    if content is not None:
        return _strip_user_prefix(str(content))
    return ""


def _strip_user_prefix(text: str) -> str:
    return _USER_PREFIX_RE.sub("", text).strip()


def _normalize_prompts(prompts: Iterable[str]) -> list[str]:
    out: list[str] = []
    for p in prompts:
        s = _strip_user_prefix(str(p))
        if s:
            out.append(s)
    return out
