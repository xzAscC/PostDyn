from __future__ import annotations

import sys
from pathlib import Path

from postdyn import bench

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.q2_common as q2_common


def test_load_items_returns_cached_list_objects(monkeypatch) -> None:
    val = [bench.BenchItem("v", "validation", {})]
    test = [bench.BenchItem("t", "test", {})]
    calls = 0

    def load_benchmark(_spec):
        nonlocal calls
        calls += 1
        return val, test

    monkeypatch.setattr(bench, "load_benchmark", load_benchmark)
    domain = "math"
    q2_common._ITEMS_CACHE.pop((domain, 1, False), None)

    first_val, first_test = q2_common.load_items(domain, 1, tiny=False)
    second_val, second_test = q2_common.load_items(domain, 1, tiny=False)

    assert first_val is second_val
    assert first_test is second_test
    assert calls == 1
