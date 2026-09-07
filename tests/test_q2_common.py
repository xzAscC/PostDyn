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


def test_load_items_threads_livecodebench_jsonl_env(
    monkeypatch, tmp_path
) -> None:
    import importlib.util as _util
    import json as _json
    import sys as _sys

    spec = _util.spec_from_file_location(
        "q2_common_envtest", "scripts/q2_common.py"
    )
    mod = _util.module_from_spec(spec)
    _sys.modules["q2_common_envtest"] = mod
    spec.loader.exec_module(mod)
    rows = [
        {
            "question_id": str(i),
            "question_content": f"code {i}",
            "input_output": _json.dumps({"inputs": [], "outputs": []}),
        }
        for i in range(40)
    ]
    src = tmp_path / "release_v6_test.jsonl"
    src.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    monkeypatch.setenv("POSTDYN_LIVECODEBENCH_JSONL", str(src))
    val, test = mod.load_items("code", None, tiny=False)
    assert len(val) == 30 and len(test) == 10
    assert val[0].prompt.startswith("code ")
