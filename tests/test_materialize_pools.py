"""Tests for the data-collection CLI (scripts/materialize_pools.py)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "rev-fixture" / "data"
    data_dir.mkdir(parents=True)
    _write_shard(data_dir / "one.parquet", "source-a", 0, 20)
    _write_shard(data_dir / "two.parquet", "source-b", 0, 5)
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(json.dumps({"math": ["source-a"], "code": ["source-b"]}))
    return mapping_path, data_dir


def test_materialize_pools_cli_writes_deterministic_pools(tmp_path: Path) -> None:
    materialize = importlib.import_module("scripts.materialize_pools")
    mapping_path, data_dir = _fixture(tmp_path)

    first_out = tmp_path / "pools-1"
    status = materialize.main(
        [
            "--mapping",
            str(mapping_path),
            "--out-dir",
            str(first_out),
            "--n",
            "6",
            "--snapshot-dir",
            str(data_dir),
        ]
    )
    assert status == 0

    second_out = tmp_path / "pools-2"
    assert (
        materialize.main(
            [
                "--mapping",
                str(mapping_path),
                "--out-dir",
                str(second_out),
                "--n",
                "6",
                "--snapshot-dir",
                str(data_dir),
            ]
        )
        == 0
    )

    from postdyn.data import load_pool

    for domain, expected_n in (("math", 6), ("code", 5)):
        first = load_pool(first_out / f"{domain}.json")
        second = load_pool(second_out / f"{domain}.json")
        assert first.actual_n == expected_n
        assert first.dataset_revision == "rev-fixture"
        assert [record.id for record in first.records] == [
            record.id for record in second.records
        ]
        assert first.fingerprint == second.fingerprint
    for domain in ("instruction_following", "general_reasoning"):
        assert load_pool(first_out / f"{domain}.json").actual_n == 0


def test_materialize_pools_cli_rejects_nonpositive_n(tmp_path: Path) -> None:
    materialize = importlib.import_module("scripts.materialize_pools")
    mapping_path, data_dir = _fixture(tmp_path)
    status = materialize.main(
        [
            "--mapping",
            str(mapping_path),
            "--out-dir",
            str(tmp_path / "pools"),
            "--n",
            "0",
            "--snapshot-dir",
            str(data_dir),
        ]
    )
    assert status == 2
