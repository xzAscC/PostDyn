"""Deterministic source enumeration and domain prompt pools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

SAMPLING_SEED = 42


@dataclass(frozen=True)
class PromptRecord:
    """A raw user prompt and its source identity."""

    id: str
    prompt: str
    source: str


@dataclass(frozen=True)
class DomainPool:
    """A deterministic, provenance-bearing prompt sample."""

    domain: str
    records: tuple[PromptRecord, ...]
    requested_n: int
    actual_n: int
    dataset_revision: str
    seed: int
    fingerprint: str


def _default_snapshot() -> Path:
    root = (
        Path.home()
        / ".cache/huggingface/hub/datasets--allenai--Dolci-Think-SFT-7B/snapshots"
    )
    snapshots = sorted(root.glob("*/data"))
    if not snapshots:
        raise FileNotFoundError(f"No Dolci snapshot data found under {root}")
    return snapshots[-1]


def enumerate_sources(snapshot_dir: str | Path | None = None) -> dict[str, int]:
    """Count each distinct ``dataset_source`` across parquet shards."""
    import pyarrow.parquet as pq

    path = Path(snapshot_dir) if snapshot_dir is not None else _default_snapshot()
    counts: dict[str, int] = {}
    for shard in sorted(path.rglob("*.parquet")):
        column = pq.read_table(shard, columns=["dataset_source"])["dataset_source"]
        for source in column.to_pylist():
            if source is not None:
                key = str(source)
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def load_mapping(path: str | Path) -> dict[str, list[str]]:
    """Load domain-to-source mapping, ignoring metadata/exclusion keys."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _first_user(row: dict[str, Any]) -> str | None:
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return None if content is None else str(content)
    return None


def build_pool(
    domain: str,
    rows_by_source: dict[str, Any],
    n: int,
    seed: int = SAMPLING_SEED,
    dataset_revision: str = "unknown",
    allowed_sources: frozenset[str] | set[str] | None = None,
) -> DomainPool:
    """Build a seeded, deduplicated pool from rows grouped by source name.

    ``rows_by_source`` maps a source name to its raw dataset rows. When
    ``allowed_sources`` is given, only those sources contribute candidates
    (rows carrying a different ``dataset_source`` are dropped as well).
    """
    selected: dict[str, Any] = dict(rows_by_source)
    allowed: set[str] | None = None
    if allowed_sources is not None:
        allowed = set(allowed_sources)
        selected = {name: rows for name, rows in selected.items() if name in allowed}
    candidates: list[tuple[str, PromptRecord]] = []
    for source, rows in selected.items():
        for row in rows:
            prompt = _first_user(row)
            if prompt is None or not prompt.strip():
                continue
            row_source = str(row.get("dataset_source", source))
            if allowed is not None and row_source not in allowed:
                continue
            record = PromptRecord(str(row["id"]), prompt, row_source)
            key = f"{seed}|{dataset_revision}|{record.id}|{record.prompt}"
            candidates.append((hashlib.sha256(key.encode()).hexdigest(), record))
    candidates.sort(key=lambda item: item[0])
    seen: set[str] = set()
    records: list[PromptRecord] = []
    for _, record in candidates:
        normalized = re.sub(r"\s+", " ", record.prompt.strip())
        if normalized in seen:
            continue
        seen.add(normalized)
        records.append(record)
        if len(records) == n:
            break
    fingerprint = hashlib.sha256(
        json.dumps(
            [(record.id, record.prompt) for record in records], separators=(",", ":")
        ).encode()
    ).hexdigest()
    return DomainPool(
        domain, tuple(records), n, len(records), dataset_revision, seed, fingerprint
    )


def load_domain_rows(
    mapping: dict[str, list[str]],
    snapshot_dir: str | Path | None = None,
) -> tuple[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Scan Dolci parquet shards once and group mapped rows by domain and source.

    Returns ``(dataset_revision, rows)`` where only sources listed in
    ``mapping`` are retained; every other source is excluded.
    """
    import pyarrow.parquet as pq

    path = Path(snapshot_dir) if snapshot_dir is not None else _default_snapshot()
    source_to_domains: dict[str, list[str]] = {}
    for domain, sources in mapping.items():
        for source in sources:
            source_to_domains.setdefault(source, []).append(domain)
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        domain: {} for domain in mapping
    }
    revision = path.parent.name if path.parent.name != "snapshots" else "unknown"
    for shard in sorted(path.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["dataset_source", "id", "messages"])
        for batch in table.to_batches():
            for row in batch.to_pylist():
                source = row.get("dataset_source")
                if source is None:
                    continue
                for domain in source_to_domains.get(str(source), ()):
                    domain_rows = rows[domain].setdefault(str(source), [])
                    domain_rows.append(row)
    return revision, rows


def _iter_snapshot_rows(
    mapped_sources: set[str],
    snapshot_dir: str | Path | None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(dataset_source, row)`` for mapped sources in one shard pass."""
    import pyarrow.parquet as pq

    path = Path(snapshot_dir) if snapshot_dir is not None else _default_snapshot()
    for shard in sorted(path.rglob("*.parquet")):
        table = pq.read_table(shard, columns=["dataset_source", "id", "messages"])
        for row in table.to_pylist():
            source = row.get("dataset_source")
            if source is not None and str(source) in mapped_sources:
                yield str(source), row


def materialize_pools(
    mapping_path: str | Path,
    out_dir: str | Path,
    n: int = 3 * 5120,
    snapshot_dir: str | Path | None = None,
) -> None:
    """Write one JSON pool per configured domain.

    ``n`` defaults to the largest family sample count (3 x d_model of 32B) so
    smaller families consume a deterministic prefix of the same pool.

    Streams the parquet shards with a bounded per-domain selection heap, so
    memory stays ~O(n) instead of holding every mapped row. For distinct
    prompts the selected set matches :func:`build_pool` exactly (n smallest
    hashes); when duplicate prompts exist, the surviving record is
    deterministic under shard order and content-identical up to row id.
    """
    import heapq
    import itertools

    mapping = load_mapping(mapping_path)
    allowed = {domain: frozenset(sources) for domain, sources in mapping.items()}
    source_to_domain = {
        source: domain for domain, sources in mapping.items() for source in sources
    }
    path = Path(snapshot_dir) if snapshot_dir is not None else _default_snapshot()
    revision = path.parent.name if path.parent.name != "snapshots" else "unknown"
    domains = ("math", "code", "instruction_following", "general_reasoning")
    heaps: dict[str, list[tuple[int, str, int, PromptRecord]]] = {
        d: [] for d in domains
    }
    seen: dict[str, set[str]] = {d: set() for d in domains}
    counters = itertools.count()
    for source, row in _iter_snapshot_rows(set(source_to_domain), snapshot_dir):
        domain = source_to_domain[source]
        prompt = _first_user(row)
        if prompt is None or not prompt.strip():
            continue
        record = PromptRecord(str(row["id"]), prompt, source)
        key = f"{SAMPLING_SEED}|{revision}|{record.id}|{record.prompt}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        numkey = int(digest[:16], 16)
        heap = heaps[domain]
        full = len(heap) == n
        if full and numkey >= -heap[0][0]:
            continue
        normalized = re.sub(r"\s+", " ", record.prompt.strip())
        if normalized in seen[domain]:
            continue
        if full:
            _, _, _, evicted = heapq.heappop(heap)
            seen[domain].discard(re.sub(r"\s+", " ", evicted.prompt.strip()))
        heapq.heappush(heap, (-numkey, digest, next(counters), record))
        seen[domain].add(normalized)

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for domain in domains:
        ordered = sorted(heaps[domain], key=lambda item: item[1])
        records = [item[3] for item in ordered]
        fingerprint = hashlib.sha256(
            json.dumps(
                [(r.id, r.prompt) for r in records], separators=(",", ":")
            ).encode()
        ).hexdigest()
        pool = DomainPool(
            domain,
            tuple(records),
            n,
            len(records),
            revision,
            SAMPLING_SEED,
            fingerprint,
        )
        (target / f"{domain}.json").write_text(
            json.dumps(asdict(pool), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_pool(path: str | Path) -> DomainPool:
    """Load a materialized domain pool JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["records"] = tuple(PromptRecord(**record) for record in payload["records"])
    return DomainPool(**payload)
