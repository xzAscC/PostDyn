"""RQ1 per-example DiM projection statistics.

For each domain (math, code, IF with WikiText as neg; safety with benign as neg),
compute the cosine similarity and raw projection of every positive-set hidden
state onto the DiM direction, then report mean and std.

If DiM sits in a high-variance band of Σ+, the projection std should be large.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from postdyn.extract import OnlineCovariance, extract_layer_hiddens
from postdyn.persistence import RunDir, append_jsonl, atomic_write_json, tee_log

LAYERS = (6, 17, 28)
DEFAULT_N = 5000
MODEL_REPO = "allenai/Olmo-3-1025-7B"


def _dim_projection_stats(
    hiddens: torch.Tensor, dim_vec: torch.Tensor
) -> dict[str, float]:
    h = hiddens.to(dtype=torch.float64)
    d = dim_vec.to(dtype=torch.float64)
    projections = h @ d
    h_norms = h.norm(dim=1)
    d_norm = d.norm()
    cos_sims = projections / (h_norms * d_norm + 1e-30)
    return {
        "proj_mean": float(projections.mean().item()),
        "proj_std": float(projections.std().item()),
        "cos_mean": float(cos_sims.mean().item()),
        "cos_std": float(cos_sims.std().item()),
    }


def _load_wikitext(n: int, seed: int = 42) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train")
    texts = [row["text"] for row in ds if row["text"].strip() and len(row["text"]) > 50]
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=rng)[:n].tolist()
    return [texts[i] for i in indices]


def _load_domain_prompts(domain: str, n: int) -> list[str]:
    from postdyn.data import load_pool

    pool_path = ROOT / "data" / "domain_prompts" / f"{domain}.json"
    pool = load_pool(pool_path)
    return [r.prompt for r in pool.records[:n]]


def _load_harmful(n: int | None = None, seed: int = 42) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
    texts = [row["rejected"] for row in ds if row["rejected"].strip()]
    if n is not None and n < len(texts):
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(texts), generator=rng)[:n].tolist()
        return [texts[i] for i in indices]
    return texts


def _load_benign(n: int, seed: int = 42) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("LLM-LAT/benign-dataset", split="train")
    texts = [row["response"] for row in ds if row["response"].strip()]
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=rng)[:n].tolist()
    return [texts[i] for i in indices]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--attention-budget", type=int, default=8_388_608)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--layers", default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    from postdyn.models import load_model, release_model
    from transformers import AutoTokenizer

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(LAYERS)
    )
    output = Path(args.output) if args.output else ROOT / "logs" / "rq1_dim_projection"
    run_dir = RunDir(output)
    completed = set()
    metrics_path = run_dir.path("metrics.jsonl")
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed.add((row["domain"], row["layer"]))

    domain_configs: list[dict[str, Any]] = [
        {"name": "math", "neg_type": "wikitext"},
        {"name": "code", "neg_type": "wikitext"},
        {"name": "instruction_following", "neg_type": "wikitext"},
        {"name": "safety", "neg_type": "benign"},
    ]

    manifest = {
        "model": MODEL_REPO,
        "n": args.n,
        "domains": [c["name"] for c in domain_configs],
        "layers": layers,
        "dtype": args.dtype,
    }
    atomic_write_json(run_dir.path("manifest.json"), manifest)

    with tee_log(run_dir):
        from postdyn.config import CheckpointRef

        ref = CheckpointRef("base", MODEL_REPO, "main", "base")
        print(f"Loading model {MODEL_REPO}...")
        model = load_model(ref, args.dtype, None, args.device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        d_model = model.config.hidden_size
        print(f"Model loaded (d_model={d_model})")

        try:
            # --- Load all text sets ---
            print(f"Loading WikiText (n={args.n})...")
            wikitext = _load_wikitext(args.n)
            print(f"Loaded {len(wikitext)} WikiText prompts")

            print("Loading harmful responses...")
            harmful = _load_harmful()
            n_safety = len(harmful)
            print(f"Loaded {n_safety} harmful responses")

            print(f"Loading benign responses (n={n_safety})...")
            benign = _load_benign(n_safety)
            print(f"Loaded {len(benign)} benign responses")

            pos_texts = {
                "math": _load_domain_prompts("math", args.n),
                "code": _load_domain_prompts("code", args.n),
                "instruction_following": _load_domain_prompts("instruction_following", args.n),
                "safety": harmful,
            }
            neg_texts = {
                "wikitext": wikitext,
                "benign": benign,
            }
            for name, texts in pos_texts.items():
                print(f"  {name}: {len(texts)} positive prompts")

            # --- Extract neg hidden states ---
            neg_hiddens: dict[str, dict[int, torch.Tensor]] = {}
            need_wiki = any(
                (cfg["name"], l) not in completed
                for cfg in domain_configs if cfg["neg_type"] == "wikitext"
                for l in layers
            )
            if need_wiki:
                print(f"Extracting WikiText hidden states ({len(wikitext)} prompts)...")
                t0 = time.monotonic()
                neg_hiddens["wikitext"] = extract_layer_hiddens(
                    model, tokenizer, wikitext, layers,
                    args.batch_size, args.max_length,
                    token_budget=args.token_budget,
                    attention_budget=args.attention_budget,
                    return_device=args.device,
                )
                print(f"WikiText extraction: {time.monotonic() - t0:.1f}s")

            need_benign = any(
                (cfg["name"], l) not in completed
                for cfg in domain_configs if cfg["neg_type"] == "benign"
                for l in layers
            )
            if need_benign:
                print(f"Extracting benign hidden states ({len(benign)} prompts)...")
                t0 = time.monotonic()
                neg_hiddens["benign"] = extract_layer_hiddens(
                    model, tokenizer, benign, layers,
                    args.batch_size, args.max_length,
                    token_budget=args.token_budget,
                    attention_budget=args.attention_budget,
                    return_device=args.device,
                )
                print(f"Benign extraction: {time.monotonic() - t0:.1f}s")

            # --- Per-domain extraction and metrics ---
            for cfg in domain_configs:
                domain = cfg["name"]
                need = any((domain, l) not in completed for l in layers)
                if not need:
                    print(f"[skip] {domain}: all layers complete")
                    continue

                print(f"Extracting {domain} hidden states ({len(pos_texts[domain])} prompts)...")
                t0 = time.monotonic()
                pos_h = extract_layer_hiddens(
                    model, tokenizer, pos_texts[domain], layers,
                    args.batch_size, args.max_length,
                    token_budget=args.token_budget,
                    attention_budget=args.attention_budget,
                    return_device=args.device,
                )
                print(f"{domain} extraction: {time.monotonic() - t0:.1f}s")

                neg_h = neg_hiddens[cfg["neg_type"]]

                for layer in layers:
                    if (domain, layer) in completed:
                        print(f"[skip] {domain}/layer_{layer}")
                        continue

                    ph = pos_h[layer].to(dtype=torch.float64)
                    nh = neg_h[layer].to(dtype=torch.float64)
                    mu_pos = ph.mean(dim=0)
                    mu_neg = nh.mean(dim=0)
                    dim_vec = mu_pos - mu_neg

                    stats = _dim_projection_stats(ph, dim_vec)

                    row = {
                        "domain": domain,
                        "layer": layer,
                        "n_pos": int(ph.shape[0]),
                        "n_neg": int(nh.shape[0]),
                        "dim_norm": float(dim_vec.norm().item()),
                        **stats,
                    }
                    append_jsonl(metrics_path, row)
                    completed.add((domain, layer))

                    print(
                        f"  {domain}/layer_{layer}: "
                        f"cos={stats['cos_mean']:.4f}±{stats['cos_std']:.4f}  "
                        f"proj={stats['proj_mean']:.2f}±{stats['proj_std']:.2f}"
                    )

                del pos_h
                torch.cuda.empty_cache()

        finally:
            release_model(model)

        results: list[dict[str, Any]] = []
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        atomic_write_json(run_dir.path("summary.json"), {
            "manifest": manifest,
            "results": results,
        })
        print(f"\nDone. Results: {run_dir.path('summary.json')}")

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
