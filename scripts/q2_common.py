from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]

from postdyn.config import BENCHMARKS, MODEL_FAMILIES
from postdyn.persistence import load_eigensystem, append_jsonl, atomic_write_json
from postdyn.uploader import uploader_from_args

CAPS = {
    "math500": (1024, 2048),
    "mmlu_pro": (256, 512),
    "ifeval": (512, 768),
    "livecodebench": (1024, 1024),
}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--family", choices=("7b", "32b"), required=True)
    parser.add_argument("--q1-root", type=Path, required=True)
    parser.add_argument("--scale", choices=("tiny", "full"), default="full")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--quantization", choices=("nf4",), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--domains", nargs="+", choices=tuple(BENCHMARKS), default=list(BENCHMARKS)
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument(
        "--upload-to",
        default=None,
        help="dataset repo id for artifact uploads (env: POSTDYN_UPLOAD_TO)",
    )
    parser.add_argument("--limit", type=positive_int)
    return parser


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def family_config(family: str, scale: str) -> Any:
    if scale == "tiny":
        return SimpleNamespace(key=family, d_model=8, n_layers=2, layers=(0, 1))
    return MODEL_FAMILIES[family]


def output_root(args: argparse.Namespace, experiment: str) -> Path:
    return args.output or Path("logs") / "q2" / args.family / experiment


def completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.is_file():
        return set()
    result: set[tuple[Any, ...]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if {"domain", "layer", "alpha", "condition"} <= row.keys():
                result.add(
                    (row["domain"], row["layer"], row["alpha"], row["condition"])
                )
    return result


def completed_item_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.is_file():
        return set()
    result: set[tuple[Any, ...]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if {"domain", "layer", "alpha", "condition", "item_id"} <= row.keys():
                result.add(
                    (
                        row["domain"],
                        row["layer"],
                        row["alpha"],
                        row["condition"],
                        row["item_id"],
                    )
                )
    return result


def validation_scores(
    path: Path, domain: str, conditions: set[str] | None = None
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[bool]] = {}
    if not path.is_file():
        return []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("domain") != domain or (
            conditions is not None and row.get("condition") not in conditions
        ):
            continue
        key = (row["layer"], row["alpha"], row["condition"])
        grouped.setdefault(key, []).append(bool(row["correct"]))
    return [
        {
            "layer": key[0],
            "alpha": key[1],
            "condition": key[2],
            "accuracy": sum(values) / len(values),
        }
        for key, values in grouped.items()
    ]


def require_bases(
    root: Path, domain: str, layers: tuple[int, ...], stage: str
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    found: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    missing: list[str] = []
    for layer in layers:
        base = root / "eigensystems" / stage / str(layer) / domain
        if (
            not base.with_suffix(".json").is_file()
            or not base.with_suffix(".safetensors").is_file()
        ):
            missing.append(str(base))
        else:
            found[layer] = load_eigensystem(base)
    if missing:
        raise SystemExit(f"Q1 {stage} bases missing for {domain}: {missing[0]}")
    return found


def tiny_bases(
    root: Path, domains: list[str], layers: tuple[int, ...], stages: tuple[str, ...]
) -> None:
    from postdyn.persistence import save_eigensystem

    for stage in stages:
        for domain in domains:
            for layer in layers:
                base = root / "eigensystems" / stage / str(layer) / domain
                if (
                    not base.with_suffix(".json").is_file()
                    or not base.with_suffix(".safetensors").is_file()
                ):
                    save_eigensystem(
                        base, torch.arange(8, 0, -1, dtype=torch.float64), torch.eye(8)
                    )


class TinyTokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "</s>"
    chat_template = "tiny"

    def apply_chat_template(self, conversation, add_generation_prompt=True, **kwargs):
        return conversation[0]["content"]

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        return_offsets_mapping=False,
        **kwargs,
    ):
        rows = [texts] if isinstance(texts, str) else list(texts)
        ids = [
            [max(1, len(word)) % 7 + 1 for word in row.split()] or [1] for row in rows
        ]
        width = max(map(len, ids))
        padded = [[0] * (width - len(row)) + row for row in ids]
        result: dict[str, Any] = {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(
                [[int(x != 0) for x in row] for row in padded]
            ),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                [(0, 0)] * (width - len(ids[i]))
                + [(0, len(word)) for word in rows[i].split()]
                for i in range(len(rows))
            ]
        return result

    def batch_decode(self, rows, **kwargs):
        return ["1" for _ in rows]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, n_embd=8, num_hidden_layers=2)
        self.embed = torch.nn.Embedding(16, 8)
        self.transformer = SimpleNamespace(
            h=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        )
        self.device = torch.device("cpu")

    def forward(
        self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=True
    ):
        hidden = self.embed(input_ids)
        states = [hidden]
        for block in self.transformer.h:
            hidden = block(hidden)
            states.append(hidden)
        return SimpleNamespace(
            hidden_states=tuple(states), logits=torch.zeros((*input_ids.shape, 16))
        )

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=1,
        do_sample=False,
        **kwargs,
    ):
        return torch.cat(
            (input_ids, torch.ones((input_ids.shape[0], 1), dtype=input_ids.dtype)),
            dim=1,
        )


def load_runtime(args: argparse.Namespace, stage: str):
    if args.scale == "tiny":
        return TinyModel(), TinyTokenizer()
    from transformers import AutoTokenizer
    from postdyn.models import load_model

    ref = MODEL_FAMILIES[args.family].checkpoints(getattr(args, "sft_lr", "1e-4"))
    checkpoint = next(x for x in ref if x.name == stage)
    return load_model(
        checkpoint, args.dtype, args.quantization, args.device
    ), AutoTokenizer.from_pretrained(checkpoint.repo, revision=checkpoint.revision)


def load_items(domain: str, limit: int | None, tiny: bool):
    from postdyn.bench import BenchItem, load_benchmark

    if tiny:
        items = [BenchItem(str(i), f"prompt {i}", {"answer": "1"}) for i in range(40)]
        val, test = items[:30], items[30:]
        return (val[:limit] if limit else val), (test[:limit] if limit else test)
    val, test = load_benchmark(BENCHMARKS[domain])
    return (val[:limit] if limit else val), (test[:limit] if limit else test)


def write_manifest(root: Path, args: argparse.Namespace, experiment: str) -> None:
    atomic_write_json(
        root / "manifest.json",
        {
            "experiment": experiment,
            "family": args.family,
            "scale": args.scale,
            "seed": 42,
            "greedy": True,
            "model_repo": None,
            "model_revision": None,
            "dataset_revision": None,
            "pool_fingerprints": {},
            "params_digest": None,
            "code_contract_hash": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def write_identity_manifest(run_dir: Path, identity: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        validate_identity(existing, identity)
    atomic_write_json(path, identity)


def validate_identity(existing: dict[str, Any], identity: dict[str, Any]) -> None:
    mismatched = [key for key, value in identity.items() if existing.get(key) != value]
    if mismatched:
        raise SystemExit(f"resume identity mismatch: {','.join(sorted(mismatched))}")


def final_checkpoint_pair(family: str, stage: str, sft_lr: str = "1e-4") -> list[str]:
    refs = MODEL_FAMILIES[family].checkpoints(sft_lr)
    checkpoint = next(ref for ref in reversed(refs) if ref.stage == stage)
    return [checkpoint.repo, checkpoint.revision]


def checkpoint_pairs(
    family: str, stages: tuple[str, ...], sft_lr: str = "1e-4"
) -> list[list[str]]:
    return [final_checkpoint_pair(family, stage, sft_lr) for stage in stages]


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(path: Path, row: dict[str, Any]) -> None:
    append_jsonl(path, row)


def start_uploader(args: argparse.Namespace, root: Path):
    """Optional background uploader enabled by --upload-to / POSTDYN_UPLOAD_TO."""
    handle = uploader_from_args(getattr(args, "upload_to", None), ROOT)
    if handle:
        handle.start()
    return handle


def finish_uploader(handle, output: Path) -> None:
    """Submit the run tree (append-only artifacts) and drain the worker."""
    if handle is None:
        return
    handle.submit_tree(output, relative_to=ROOT)
    summary = handle.finish()
    print(
        f"upload: {summary['uploaded']} uploaded, "
        f"{summary['skipped']} skipped, "
        f"{summary['failed']} failed"
    )
