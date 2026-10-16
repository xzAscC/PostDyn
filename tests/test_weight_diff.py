from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from postdyn.persistence import close_all_jsonl_handles
from postdyn.weight_diff import (
    BasisCache,
    compare,
    effective_rank,
    gini,
    parse_name,
    principal_basis,
    select_uniform,
    tensor_files,
    update_stats,
)


def _save_sharded(root: Path, shards: list[dict[str, torch.Tensor]]) -> Path:
    """Write a checkpoint as several safetensors shards plus an HF index."""
    root.mkdir(parents=True)
    weight_map = {}
    for i, shard in enumerate(shards):
        name = f"model-{i:05d}.safetensors"
        save_file(shard, str(root / name))
        weight_map.update(dict.fromkeys(shard, name))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return root


def test_parse_name_splits_layer_and_component() -> None:
    assert parse_name("model.layers.3.self_attn.q_proj.weight") == (3, "self_attn.q_proj.weight")
    assert parse_name("model.embed_tokens.weight") == (None, "model.embed_tokens.weight")


def test_select_uniform_keeps_ends_and_short_lists() -> None:
    assert select_uniform(list(range(5)), 10) == list(range(5))
    picked = select_uniform(list(range(43)), 10)
    assert len(picked) == 10 and picked[0] == 0 and picked[-1] == 42
    assert picked == sorted(set(picked))


def test_gini_extremes() -> None:
    assert gini(torch.ones(100)) == pytest.approx(0.0, abs=1e-12)
    one_hot = torch.zeros(100)
    one_hot[7] = 3.0
    assert gini(one_hot) == pytest.approx(1 - 1 / 100)
    assert gini(torch.zeros(4)) is None


def test_gini_matches_mean_difference_formula() -> None:
    x = torch.rand(57, dtype=torch.float64)
    mean_abs_diff = (x[:, None] - x[None, :]).abs().mean()
    assert gini(x) == pytest.approx(float(mean_abs_diff / (2 * x.mean())), rel=1e-6)


def test_effective_rank_of_equal_singular_values() -> None:
    assert effective_rank(torch.tensor([2.0, 2.0, 2.0, 0.0])) == pytest.approx(3.0)
    assert effective_rank(torch.tensor([5.0])) == pytest.approx(1.0)


def test_principal_basis_shapes_use_ten_percent() -> None:
    u, v = principal_basis(torch.randn(40, 20))
    assert u.shape == (40, 2) and v.shape == (20, 2)


def test_update_stats_matrix_metrics() -> None:
    start = torch.eye(4, dtype=torch.bfloat16)
    end = start.clone()
    end[0, 1] = 1.0  # row 0 rotates by 45 degrees; rows 1-3 unchanged
    end[3, 0] = 1e-6  # changed in bf16 but within the 1e-5 tolerance
    stats = update_stats(start, end, basis=principal_basis(start.float(), frac=0.5))
    assert stats["changed_tol"] == 1
    assert stats["sparsity"] == pytest.approx(15 / 16)
    assert stats["one_minus_cos"] == pytest.approx((1 - math.cos(math.pi / 4)) / 4, rel=1e-3)
    assert stats["rank"] == 2 and stats["max_rank"] == 4
    assert stats["rank_ratio"] == pytest.approx(0.5)
    assert 1 <= stats["erank"] <= 2
    assert 0 <= stats["overlap"] <= 1
    assert stats["gini"] > 0.9


def test_one_minus_cos_resolves_tiny_rotations() -> None:
    torch.manual_seed(0)
    w0 = torch.randn(64, 4096)
    w1 = w0 + 1e-4 * torch.randn(64, 4096)  # 1 - cos ~ 5e-9, below float32 resolution of cos
    a, b = w0.double(), w1.double()
    exact = float((1 - (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean())
    assert update_stats(w0, w1)["one_minus_cos"] == pytest.approx(exact, rel=1e-3)


def test_update_stats_vector_has_only_vector_metrics() -> None:
    stats = update_stats(torch.ones(5), torch.ones(5) * 2)
    assert stats["sparsity"] == 0.0
    assert stats["one_minus_cos"] == pytest.approx(0.0, abs=1e-6)
    assert stats["gini"] == pytest.approx(0.0, abs=1e-6)
    assert "rank" not in stats and "overlap" not in stats


def test_update_stats_zero_update() -> None:
    w = torch.randn(3, 3)
    stats = update_stats(w, w, basis=principal_basis(w))
    assert stats["zero_update"] is True
    assert stats["rank"] == 0
    assert stats["gini"] is None and stats["erank"] is None and stats["overlap"] is None


def test_basis_cache_reuses_saved_basis(tmp_path: Path) -> None:
    cache = BasisCache(tmp_path / "basis")
    w = torch.randn(30, 10)
    u1, _ = cache.get("model.layers.0.mlp.up_proj.weight", w)
    u2, _ = cache.get("model.layers.0.mlp.up_proj.weight", torch.zeros(30, 10))  # loaded, not recomputed
    assert torch.equal(u1, u2)


def test_tensor_files_reads_index_and_single_file(tmp_path: Path) -> None:
    sharded = _save_sharded(tmp_path / "a", [{"x": torch.zeros(2)}, {"y": torch.zeros(2)}])
    assert {k: v.name for k, v in tensor_files(sharded).items()} == {
        "x": "model-00000.safetensors",
        "y": "model-00001.safetensors",
    }
    single = tmp_path / "b"
    single.mkdir()
    save_file({"z": torch.zeros(2)}, str(single / "model.safetensors"))
    assert set(tensor_files(single)) == {"z"}


def test_compare_writes_one_row_per_tensor_and_resumes(tmp_path: Path) -> None:
    w = torch.randn(6, 6).to(torch.bfloat16)
    norm = torch.ones(6, dtype=torch.bfloat16)
    start = _save_sharded(tmp_path / "start", [{"model.layers.0.mlp.up_proj.weight": w}, {"model.norm.weight": norm}])
    end = _save_sharded(tmp_path / "end", [{"model.layers.0.mlp.up_proj.weight": w + 1, "model.norm.weight": norm}])
    out = tmp_path / "pair.jsonl"
    compare(start, end, out, basis_dir=tmp_path / "basis")
    close_all_jsonl_handles()
    rows = {r["name"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows["model.layers.0.mlp.up_proj.weight"]["layer"] == 0
    assert rows["model.layers.0.mlp.up_proj.weight"]["component"] == "mlp.up_proj.weight"
    assert "overlap" in rows["model.layers.0.mlp.up_proj.weight"]
    assert rows["model.norm.weight"]["zero_update"] is True

    compare(start, end, out, basis_dir=tmp_path / "basis")  # resume: nothing new is written
    close_all_jsonl_handles()
    assert len(out.read_text().splitlines()) == 2


def test_compare_rejects_mismatched_tensor_sets(tmp_path: Path) -> None:
    start = _save_sharded(tmp_path / "start", [{"a": torch.zeros(2)}])
    end = _save_sharded(tmp_path / "end", [{"b": torch.zeros(2)}])
    with pytest.raises(ValueError, match="tensor names differ"):
        compare(start, end, tmp_path / "pair.jsonl")
