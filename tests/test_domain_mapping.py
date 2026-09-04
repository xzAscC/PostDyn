from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from postdyn.data import (
    DomainPool,
    PromptRecord,
    build_pool,
    enumerate_sources,
    load_domain_rows,
    load_mapping,
    load_pool,
    materialize_pools,
)


def _write_shard(path: Path, source: str, start: int, count: int) -> None:
    rows = [
        {
            "dataset_source": source,
            "id": f"{source}-{index}",
            "messages": [{"role": "user", "content": f"prompt {index}"}],
        }
        for index in range(start, start + count)
    ]
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("dataset_source", pa.string()),
                pa.field("id", pa.string()),
                pa.field(
                    "messages",
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("role", pa.string()),
                                pa.field("content", pa.string()),
                            ]
                        )
                    ),
                ),
            ]
        ),
    )
    pq.write_table(table, path)


@pytest.fixture
def source_rows() -> dict[str, dict[str, list[dict[str, object]]]]:
    return {
        "math": {
            "mapped-source": [
                {
                    "id": f"mapped-{index}",
                    "dataset_source": "mapped-source",
                    "messages": [{"role": "user", "content": f"prompt {index}"}],
                }
                for index in range(12)
            ],
            "unmapped-source": [
                {
                    "id": "unmapped-1",
                    "dataset_source": "unmapped-source",
                    "messages": [{"role": "user", "content": "must exclude"}],
                }
            ],
        }
    }


def test_enumerate_sources_merges_parquet_shard_counts(tmp_path: Path) -> None:
    _write_shard(tmp_path / "one.parquet", "source-a", 0, 2)
    _write_shard(tmp_path / "two.parquet", "source-a", 2, 1)
    _write_shard(tmp_path / "three.parquet", "source-b", 0, 4)
    assert enumerate_sources(tmp_path) == {"source-a": 3, "source-b": 4}


def test_load_mapping_returns_domain_to_source_names(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"math": ["source-a"], "code": ["source-b"]}))
    assert load_mapping(path) == {"math": ["source-a"], "code": ["source-b"]}


def test_build_pool_is_deterministic_and_seeded(source_rows) -> None:
    first = build_pool("math", source_rows["math"], n=5, seed=42)
    second = build_pool("math", source_rows["math"], n=5, seed=42)
    other = build_pool("math", source_rows["math"], n=5, seed=43)
    assert first.records == second.records
    assert first.fingerprint == second.fingerprint
    assert first.records != other.records


def test_build_pool_clamps_deduplicates_and_uses_first_user_prompt() -> None:
    rows = {
        "math": {
            "source": [
                {
                    "id": "one",
                    "dataset_source": "source",
                    "messages": [
                        {"role": "system", "content": "ignore"},
                        {"role": "user", "content": "  same   prompt "},
                    ],
                },
                {
                    "id": "two",
                    "dataset_source": "source",
                    "messages": [{"role": "user", "content": "same prompt"}],
                },
                {
                    "id": "empty",
                    "dataset_source": "source",
                    "messages": [{"role": "user", "content": "  "}],
                },
            ]
        }
    }
    pool = build_pool("math", rows["math"], n=99, seed=0)
    assert pool.requested_n == 99
    assert pool.actual_n < pool.requested_n
    assert pool.actual_n == len(pool.records) == 1
    assert pool.records[0].prompt == "  same   prompt "


def test_build_pool_excludes_sources_not_listed_for_domain(source_rows) -> None:
    pool = build_pool(
        "math",
        source_rows["math"],
        n=99,
        seed=0,
        allowed_sources={"mapped-source"},
    )
    assert all(record.source == "mapped-source" for record in pool.records)
    assert "must exclude" not in {record.prompt for record in pool.records}


def test_domain_pool_is_frozen() -> None:
    pool = DomainPool("math", (), 0, 0, "revision", 0, "fingerprint")
    with pytest.raises((AttributeError, TypeError)):
        setattr(pool, "domain", "code")


def test_load_domain_rows_filters_to_mapped_sources(tmp_path: Path) -> None:
    _write_shard(tmp_path / "one.parquet", "source-a", 0, 3)
    _write_shard(tmp_path / "two.parquet", "source-b", 0, 2)
    revision, rows = load_domain_rows({"math": ["source-a"]}, tmp_path)
    assert set(rows) == {"math"}
    assert set(rows["math"]) == {"source-a"}
    assert len(rows["math"]["source-a"]) == 3
    assert all(row["dataset_source"] == "source-a" for row in rows["math"]["source-a"])
    assert isinstance(revision, str) and revision


def test_materialize_pools_round_trips_four_domain_jsons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                domain: ["mapped-source"]
                for domain in (
                    "math",
                    "code",
                    "instruction_following",
                    "general_reasoning",
                )
            }
        )
    )
    domains = ("math", "code", "instruction_following", "general_reasoning")

    def fake_load_domain_rows(mapping, snapshot_dir=None):
        rows: dict[str, dict[str, list[dict[str, object]]]] = {}
        for dom in domains:
            rows[dom] = {
                "mapped-source": [
                    {
                        "id": f"{dom}-1",
                        "dataset_source": "mapped-source",
                        "messages": [{"role": "user", "content": f"{dom} prompt"}],
                    }
                ]
            }
        return "revision-x", rows

    monkeypatch.setattr("postdyn.data.load_domain_rows", fake_load_domain_rows)
    out_dir = tmp_path / "pools"
    materialize_pools(mapping_path, out_dir, n=64)
    for domain in domains:
        loaded = load_pool(out_dir / f"{domain}.json")
        assert loaded == DomainPool(
            domain,
            (PromptRecord(f"{domain}-1", f"{domain} prompt", "mapped-source"),),
            64,
            1,
            "revision-x",
            42,
            loaded.fingerprint,
        )
