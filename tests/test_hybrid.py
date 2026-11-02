from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from postdyn import hybrid

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
OTHERS = ("post_attention_layernorm", "post_feedforward_layernorm", "self_attn.q_norm")


def _names(layers: int) -> list[str]:
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for layer in range(layers):
        names += [f"model.layers.{layer}.{name}.weight" for name in PROJECTIONS + OTHERS]
    return names


def _write_model(path, layers: int, value: float) -> None:
    path.mkdir()
    names = _names(layers)
    # Two shards in a different order from the names, as real checkpoints have.
    shards = {"model-1.safetensors": names[::2], "model-2.safetensors": names[1::2]}
    weight_map = {}
    for shard, shard_names in shards.items():
        save_file(
            {name: torch.full((2, 2), value, dtype=torch.bfloat16) for name in shard_names},
            path / shard,
            metadata={"format": "pt"},
        )
        weight_map |= dict.fromkeys(shard_names, shard)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps({"num_hidden_layers": layers}))
    (path / "tokenizer.json").write_text("{}")


def _weights(path) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for shard in path.glob("*.safetensors")
        for name, tensor in load_file(shard).items()
    }


def test_layer_parts_are_symmetric_and_cover_every_layer():
    parts = hybrid.layer_parts(32)
    assert (parts["front"], parts["middle"], parts["back"]) == (
        range(0, 11),
        range(11, 21),
        range(21, 32),
    )
    assert [len(part) for part in hybrid.layer_parts(28).values()] == [9, 10, 9]


@pytest.mark.parametrize(
    ("condition", "count", "layers"),
    [
        ("attn-front", 4 * 3, {0, 1, 2}),
        ("attn-back", 4 * 3, {6, 7, 8}),
        ("mlp-middle", 3 * 3, {3, 4, 5}),
    ],
)
def test_reverted_names_pick_one_component_of_one_part(condition, count, layers):
    names = hybrid.reverted_names(_names(9), condition, 9)
    assert len(names) == count
    assert {int(name.split(".")[2]) for name in names} == layers
    component = "self_attn" if condition.startswith("attn") else "mlp"
    assert all(f".{component}." in name and "_proj" in name for name in names)
    assert not any("norm" in name for name in names)


def test_the_six_conditions_touch_each_projection_once():
    names = _names(9)
    picked = [n for c in hybrid.CONDITIONS for n in hybrid.reverted_names(names, c, 9)]
    assert len(hybrid.CONDITIONS) == 6
    assert len(picked) == len(set(picked)) == 9 * len(PROJECTIONS)


def test_unknown_condition_is_refused():
    with pytest.raises(ValueError):
        hybrid.reverted_names(_names(9), "embed-front", 9)


def test_build_reverts_only_the_chosen_weights(tmp_path):
    trained, base, out = tmp_path / "trained", tmp_path / "base", tmp_path / "attn-middle"
    _write_model(trained, 9, 1.0)
    _write_model(base, 9, 2.0)
    hybrid.build(trained, base, out, "attn-middle")
    weights = _weights(out)
    assert set(weights) == set(_names(9))
    reverted = hybrid.reverted_names(_names(9), "attn-middle", 9)
    for name, tensor in weights.items():
        assert tensor.dtype == torch.bfloat16
        assert tensor.eq(2.0 if name in reverted else 1.0).all(), name
    assert (out / "tokenizer.json").exists()
    assert (out / "model.safetensors.index.json").exists()
    manifest = json.loads((out / "hybrid.json").read_text())
    assert manifest["condition"] == "attn-middle"
    assert sorted(manifest["reverted"]) == sorted(reverted)


def test_build_refuses_a_shape_mismatch(tmp_path):
    trained, base = tmp_path / "trained", tmp_path / "base"
    _write_model(trained, 9, 1.0)
    _write_model(base, 9, 2.0)
    name = "model.layers.0.self_attn.q_proj.weight"
    shard = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"][name]
    tensors = load_file(base / shard)
    tensors[name] = torch.zeros((3, 2), dtype=torch.bfloat16)
    save_file(tensors, base / shard, metadata={"format": "pt"})
    out = tmp_path / "attn-front"
    with pytest.raises(ValueError, match="shape"):
        hybrid.build(trained, base, out, "attn-front")
    assert not out.exists()


def test_build_leaves_nothing_half_written(tmp_path):
    trained, base = tmp_path / "trained", tmp_path / "base"
    _write_model(trained, 9, 1.0)
    _write_model(base, 9, 2.0)
    (base / "model-1.safetensors").unlink()
    out = tmp_path / "mlp-back"
    with pytest.raises(FileNotFoundError):
        hybrid.build(trained, base, out, "mlp-back")
    assert not out.exists()
    assert list(tmp_path.glob("mlp-back*")) == []
