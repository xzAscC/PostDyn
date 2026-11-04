"""A post-trained model with one component of some layers back at its base weights."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

COMPONENTS = {
    "attn": ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
    "mlp": ("mlp", ("gate_proj", "up_proj", "down_proj")),
}
PARTS = ("front", "middle", "back")
CONDITIONS = tuple(f"{component}-{part}" for component in COMPONENTS for part in PARTS)
INDEX = "model.safetensors.index.json"


def layer_parts(n_layers: int) -> dict[str, range]:
    """Three parts of equal size where possible, symmetric: 32 layers give 11, 10, 11."""
    side = round(n_layers / 3)
    return {
        "front": range(0, side),
        "middle": range(side, n_layers - side),
        "back": range(n_layers - side, n_layers),
    }


def reverted_names(names: Iterable[str], condition: str, n_layers: int) -> set[str]:
    """Projection weights the condition takes from the base; norms stay post-trained."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    component, part = condition.split("-")
    module, projections = COMPONENTS[component]
    pattern = re.compile(
        rf"model\.layers\.(\d+)\.{module}\.(?:{'|'.join(projections)})\.weight"
    )
    layers = layer_parts(n_layers)[part]
    return {
        name for name in names if (match := pattern.fullmatch(name)) and int(match[1]) in layers
    }


def build(trained: Path, base: Path, out: Path, condition: str) -> None:
    """Write the model to ``out``; a folder that exists is complete, so a killed job redoes it."""
    trained, base, out = Path(trained), Path(base), Path(out)
    n_layers = json.loads((trained / "config.json").read_text())["num_hidden_layers"]
    weight_map = json.loads((trained / INDEX).read_text())["weight_map"]
    base_map = json.loads((base / INDEX).read_text())["weight_map"]
    reverted = reverted_names(weight_map, condition, n_layers)
    temporary = out.with_name(out.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        for path in trained.iterdir():
            if path.suffix != ".safetensors":
                shutil.copy(path, temporary / path.name)
        for shard in sorted(set(weight_map.values())):
            tensors = load_file(trained / shard)
            names = sorted(reverted & tensors.keys())
            # Open each base shard once, for all the weights this shard takes from it.
            for base_shard in sorted({base_map[name] for name in names}):
                with safe_open(base / base_shard, framework="pt") as source:
                    for name in names:
                        if base_map[name] != base_shard:
                            continue
                        tensor = source.get_tensor(name)
                        if tensor.shape != tensors[name].shape:
                            raise ValueError(f"{name}: shape {tensor.shape} in the base")
                        if tensor.dtype != tensors[name].dtype:
                            raise ValueError(f"{name}: dtype {tensor.dtype} in the base")
                        tensors[name] = tensor
            save_file(tensors, temporary / shard, metadata={"format": "pt"})
        manifest = {
            "condition": condition,
            "trained": str(trained),
            "base": str(base),
            "reverted": sorted(reverted),
        }
        (temporary / "hybrid.json").write_text(json.dumps(manifest, indent=2))
        temporary.rename(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
