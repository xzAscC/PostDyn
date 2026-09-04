"""Run the Q1 checkpoint covariance and spectral trajectory experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from postdyn.config import DOMAINS, MODEL_FAMILIES, CheckpointRef, FamilyConfig
from postdyn.data import DomainPool, PromptRecord, load_pool
from postdyn.extract import OnlineCovariance, extract_layer_hiddens
from postdyn.models import load_model, prune_revision_cache, release_model
from postdyn.persistence import (
    RunDir,
    append_jsonl,
    atomic_write_json,
    load_eigensystem,
    save_eigensystem,
    tee_log,
)
from postdyn.spectra import (
    band_slices,
    eigensystem,
    match_eigenvectors,
    rank_displacement,
    spectral_metrics,
    subsim,
)


class _TinyTokenizer:
    """Tokenizer implementing the small interface required by extraction."""

    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts: list[str], **_: object) -> dict[str, torch.Tensor]:
        encoded = [[(ord(char) % 62) + 2 for char in text] or [1] for text in texts]
        width = max(len(item) for item in encoded)
        padded, masks = [], []
        for item in encoded:
            gap = [self.pad_token_id] * (width - len(item))
            mask = [0] * len(gap) + [1] * len(item)
            padded.append(gap + item)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class _TinyConfig:
    hidden_size = 8
    d_model = 8
    num_hidden_layers = 2
    n_layers = 2


class _TinyModel(torch.nn.Module):
    """Small deterministic CPU model used by the offline tiny scale."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.config = _TinyConfig()
        self.embedding = torch.nn.Embedding(64, 8)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(2))
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(64, 8, generator=generator))
            for layer in self.layers:
                linear = cast(torch.nn.Linear, layer)
                linear.weight.copy_(torch.randn(8, 8, generator=generator) / 3)
                linear.bias.copy_(torch.randn(8, generator=generator) / 10)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self, input_ids: torch.Tensor, output_hidden_states: bool = False, **_: object
    ) -> Any:
        state = self.embedding(input_ids)
        states = [state]
        for layer in self.layers:
            state = torch.tanh(layer(state))
            states.append(state)
        return type("TinyOutput", (), {"hidden_states": tuple(states)})()


def _tiny_pool(domain: str, n: int) -> DomainPool:
    records = tuple(
        PromptRecord(
            f"tiny-{domain}-{i}", f"{domain} synthetic prompt {i}", "synthetic"
        )
        for i in range(max(n, 32))
    )
    return DomainPool(domain, records, n, min(n, len(records)), "tiny", 42, "tiny")


def _tiny_checkpoint_model(checkpoint: CheckpointRef) -> tuple[Any, Any]:
    seed = int(hashlib.sha256(checkpoint.name.encode()).hexdigest()[:8], 16)
    return _TinyModel(seed), _TinyTokenizer()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(MODEL_FAMILIES), required=True)
    parser.add_argument("--scale", choices=("tiny", "smoke", "full"), required=True)
    parser.add_argument("--sft-lr", choices=("1e-4", "5e-5"), default="1e-4")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--stages", default=None)
    parser.add_argument(
        "--checkpoints", "--checkpoint", dest="checkpoints", default=None
    )
    parser.add_argument("--domains", "--limit-domains", dest="domains", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--attention-budget", type=int, default=8_388_608)
    parser.add_argument("--allow-short-pool", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def _select_checkpoints(
    args: argparse.Namespace, family: FamilyConfig
) -> list[CheckpointRef]:
    checkpoints = list(family.checkpoints(args.sft_lr))
    if args.scale == "tiny":
        checkpoints = [
            next(c for c in checkpoints if c.name == name)
            for name in ("base", "sft", "dpo", "rlvr")
        ]
    elif args.scale == "smoke":
        checkpoints = [
            c for c in checkpoints if c.name in {"base", "sft", "dpo", "rlvr"}
        ]
    if args.checkpoints:
        wanted = [item.strip() for item in args.checkpoints.split(",") if item.strip()]
        unknown = set(wanted) - {c.name for c in checkpoints}
        if unknown:
            raise ValueError(f"unknown checkpoint(s): {sorted(unknown)}")
        checkpoints = [c for c in checkpoints if c.name in wanted]
    if args.stages:
        stages = {item.strip() for item in args.stages.split(",") if item.strip()}
        checkpoints = [c for c in checkpoints if c.stage in stages]
    if not checkpoints:
        raise ValueError("checkpoint selection must not be empty")
    return checkpoints


def _domains(args: argparse.Namespace) -> list[str]:
    domains = [
        item.strip()
        for item in (args.domains or ",".join(DOMAINS)).split(",")
        if item.strip()
    ]
    unknown = set(domains) - set(DOMAINS)
    if unknown:
        raise ValueError(f"unknown domain(s): {sorted(unknown)}")
    return domains


def _load_pools(
    args: argparse.Namespace, domains: list[str], n: int
) -> dict[str, DomainPool]:
    if args.scale == "tiny":
        return {domain: _tiny_pool(domain, n) for domain in domains}
    pools: dict[str, DomainPool] = {}
    for domain in domains:
        path = ROOT / "data" / "domain_prompts" / f"{domain}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing domain pool: {path}")
        pools[domain] = load_pool(path)
    return pools


def _checkpoint_model(
    args: argparse.Namespace, checkpoint: CheckpointRef
) -> tuple[Any, Any]:
    if args.scale == "tiny":
        return _tiny_checkpoint_model(checkpoint)
    from transformers import AutoTokenizer

    model = load_model(checkpoint, args.dtype, args.quantization, args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.repo, revision=checkpoint.revision
    )
    return model, tokenizer


def _base_path(run: RunDir, checkpoint: str, layer: int, domain: str) -> Path:
    return run.path("eigensystems", checkpoint, str(layer), domain)


def _eigensystem_complete(
    run: RunDir, checkpoint: str, layer: int, domain: str
) -> bool:
    base = _base_path(run, checkpoint, layer, domain)
    return all(
        base.with_suffix(suffix).is_file() for suffix in (".json", ".safetensors")
    )


def _write_analysis(
    run: RunDir, checkpoints: list[CheckpointRef], layers: list[int], domains: list[str]
) -> None:
    bands = ("high", "mid", "low")

    def band_comparison(
        first: CheckpointRef, second: CheckpointRef, layer: int, domain: str
    ) -> dict[str, Any]:
        first_values, first_vectors = load_eigensystem(
            _base_path(run, first.name, layer, domain)
        )
        second_values, second_vectors = load_eigensystem(
            _base_path(run, second.name, layer, domain)
        )
        slices = band_slices(first_vectors.shape[0])
        subsim_bands = {
            band: subsim(first_vectors[:, sl], second_vectors[:, sl])
            for band, sl in zip(bands, slices)
        }
        del first_values, first_vectors, second_values, second_vectors
        return {"subsim_bands": subsim_bands}

    adjacent: list[dict[str, Any]] = [
        {
            "from": first.name,
            "to": second.name,
            "subsim": {},
            "reordering": {},
        }
        for first, second in zip(checkpoints, checkpoints[1:])
    ]
    for layer in layers:
        for domain in domains:
            # Keep only the previous and current eigensystems resident while shifting.
            prev_values, prev_vectors = load_eigensystem(
                _base_path(run, checkpoints[0].name, layer, domain)
            )
            for pair_index, second in enumerate(checkpoints[1:]):
                cur_values, cur_vectors = load_eigensystem(
                    _base_path(run, second.name, layer, domain)
                )
                slices = band_slices(prev_vectors.shape[0])
                subsim_bands = {
                    band: subsim(prev_vectors[:, sl], cur_vectors[:, sl])
                    for band, sl in zip(bands, slices)
                }
                pi = match_eigenvectors(prev_vectors, cur_vectors)
                displacement = rank_displacement(pi)
                key = f"layer_{layer}/{domain}"
                adjacent[pair_index]["subsim"][key] = subsim_bands
                adjacent[pair_index]["reordering"][key] = {
                    "mean": mean(displacement.tolist()),
                    "median": median(displacement.tolist()),
                    "p90": sorted(displacement.tolist())[
                        int(0.9 * (len(displacement) - 1))
                    ],
                }
                adjacent[pair_index].setdefault("_by_unit", {})[key] = {
                    "pi": pi.tolist(),
                    "D": displacement.tolist(),
                    "subsim_bands": subsim_bands,
                }
                del prev_values, prev_vectors
                prev_values, prev_vectors = cur_values, cur_vectors
                del cur_values, cur_vectors
            del prev_values, prev_vectors
    for record in adjacent:
        all_displacements = list(record["reordering"].values())
        record["reordering"] = {
            stat: mean(item[stat] for item in all_displacements)
            for stat in ("mean", "median", "p90")
        }
        record["reordering"]["by_unit"] = record.pop("_by_unit")
        by_unit = record["subsim"]
        record["subsim"] = {
            **{band: mean(item[band] for item in by_unit.values()) for band in bands},
            "by_unit": by_unit,
        }
    summary: dict[str, Any] = {
        "checkpoints": {},
        "adjacent_pairs": adjacent,
        "layers": layers,
        "domains": domains,
    }
    base = next((c for c in checkpoints if c.stage == "base"), checkpoints[0])
    for checkpoint in checkpoints:
        item: dict[str, Any] = {}
        stage_final = next(
            (c for c in reversed(checkpoints) if c.stage == checkpoint.stage),
            checkpoint,
        )
        for layer in layers:
            for domain in domains:
                current_values, current_vectors = load_eigensystem(
                    _base_path(run, checkpoint.name, layer, domain)
                )
                metrics = spectral_metrics(current_values)
                del current_values, current_vectors
                base_comparison = band_comparison(checkpoint, base, layer, domain)
                final_comparison = band_comparison(
                    checkpoint, stage_final, layer, domain
                )
                item[f"layer_{layer}/{domain}"] = {
                    "vs_base": base_comparison,
                    "vs_stage_final": final_comparison,
                    "metrics": metrics,
                }
        unit_values = [value for key, value in item.items() if key.startswith("layer_")]
        item["vs_base"] = {
            band: mean(value["vs_base"]["subsim_bands"][band] for value in unit_values)
            for band in bands
        }
        item["vs_stage_final"] = {
            band: mean(
                value["vs_stage_final"]["subsim_bands"][band] for value in unit_values
            )
            for band in bands
        }
        summary["checkpoints"][checkpoint.name] = item
    atomic_write_json(run.path("analysis", "summary.json"), summary)


def run(args: argparse.Namespace) -> int:
    family = MODEL_FAMILIES[args.family]
    checkpoints = _select_checkpoints(args, family)
    layers = list(family.layers)
    if args.scale == "tiny":
        layers = [0, 1]
    elif args.scale == "smoke":
        layers = layers[:2]
    domains = _domains(args)
    requested_n = args.limit or (32 if args.scale == "tiny" else family.n_samples())
    pools = _load_pools(args, domains, requested_n)
    n_by_domain = {
        domain: min(requested_n, pools[domain].actual_n) for domain in domains
    }
    actual_n_by_domain = {domain: pools[domain].actual_n for domain in domains}
    required_pool_n = 3 * 8 if args.scale == "tiny" else family.n_samples()
    short_domains = {
        domain: pools[domain].actual_n
        for domain in domains
        if pools[domain].actual_n < required_pool_n
    }
    pool_fingerprints = {domain: pools[domain].fingerprint for domain in domains}
    if any(n <= 0 for n in n_by_domain.values()):
        raise ValueError("selected domain pools contain no records")
    output = Path(args.output) if args.output else ROOT / "logs" / "q1" / args.family
    run_dir = RunDir(output)
    completed = run_dir.completed_units()
    manifest = run_dir.manifest()
    manifest_path = run_dir.path("manifest.json")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "family": args.family,
            "scale": args.scale,
            "checkpoints": [c.name for c in checkpoints],
            "domains": domains,
            "layers": layers,
            "n": n_by_domain,
            "pool_fingerprints": pool_fingerprints,
        }
        mismatches = [
            key for key, value in expected.items() if previous.get(key) != value
        ]
        if mismatches:
            raise SystemExit(
                "resume manifest mismatch for "
                + ", ".join(mismatches)
                + "; pick a fresh --output directory"
            )
    if short_domains and not args.allow_short_pool:
        details = ", ".join(
            f"{domain} ({actual}/{required_pool_n})"
            for domain, actual in short_domains.items()
        )
        raise SystemExit(
            f"short domain pools: {details}; rematerialize larger pools or pass "
            "--allow-short-pool"
        )
    manifest.update(
        {
            "family": args.family,
            "scale": args.scale,
            "checkpoints": [c.name for c in checkpoints],
            "layers": layers,
            "domains": domains,
            "n": n_by_domain,
            "actual_n": actual_n_by_domain,
            "pool_fingerprints": pool_fingerprints,
            "short_pool": bool(short_domains),
        }
    )
    atomic_write_json(run_dir.path("manifest.json"), manifest)
    with tee_log(run_dir):
        for checkpoint in checkpoints:
            model, tokenizer = _checkpoint_model(args, checkpoint)
            try:
                for domain in domains:
                    missing = [
                        layer
                        for layer in layers
                        if (checkpoint.name, layer, domain) not in completed
                        or not _eigensystem_complete(
                            run_dir, checkpoint.name, layer, domain
                        )
                    ]
                    hidden = {}
                    if missing:
                        # One forward pass per (checkpoint, domain) covers all
                        # layers; token-budget batching bounds the transient
                        # hidden-state tensor inside the VRAM budget.
                        started = time.monotonic()
                        hidden = extract_layer_hiddens(
                            model,
                            tokenizer,
                            [
                                r.prompt
                                for r in pools[domain].records[: n_by_domain[domain]]
                            ],
                            layers,
                            args.batch_size,
                            args.max_length,
                            token_budget=args.token_budget,
                            attention_budget=args.attention_budget,
                        )
                        print(
                            f"checkpoint={checkpoint.name} domain={domain} "
                            f"extracted {len(layers)} layers "
                            f"in {time.monotonic() - started:.2f}s"
                        )
                    for layer in layers:
                        unit = (checkpoint.name, layer, domain)
                        if unit in completed and _eigensystem_complete(run_dir, *unit):
                            print(f"[skip] {unit}")
                            continue
                        covariance = OnlineCovariance()
                        covariance.update(hidden[layer])
                        values, vectors = eigensystem(covariance.covariance)
                        save_eigensystem(
                            _base_path(run_dir, checkpoint.name, layer, domain),
                            values,
                            vectors,
                        )
                        row = {
                            "unit": list(unit),
                            "checkpoint": checkpoint.name,
                            "layer": layer,
                            "domain": domain,
                            **spectral_metrics(values),
                            "n": n_by_domain[domain],
                            "short_pool": bool(short_domains),
                        }
                        append_jsonl(run_dir.path("metrics.jsonl"), row)
                        completed.add(unit)
                        print(
                            f"checkpoint={checkpoint.name} layer={layer} domain={domain}"
                            f" effective_rank={row['effective_rank']:.1f} saved"
                        )
            finally:
                if args.scale != "tiny":
                    model = None
                    release_model(model)
                    if checkpoint.name not in {"base", "sft", "dpo", "rlvr"}:
                        prune_revision_cache(checkpoint)
        _write_analysis(run_dir, checkpoints, layers, domains)
        print(f"summary: {run_dir.path('analysis', 'summary.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
