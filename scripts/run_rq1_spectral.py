"""RQ1 spectral overlap: DiM vs covariance, DiM vs M+, and M+ vs covariance.

For each domain (math, code, instruction_following) with WikiText as the
negative reference, this script computes three families of overlap metrics
at selected layers of allenai/Olmo-3-1025-7B:

  1. E_high / E_low: energy of the DiM direction in the top-k and bottom-k
     eigenvectors of the within-domain covariance Σ+.
  2. Alignment of the DiM direction with the top eigenvector of the uncentered
     second moment M+ = E[hh^T].
  3. Alignment of the top-5 eigenvectors of M+ with the high/low-variance
     bands of Σ+.

Results are persisted incrementally as JSON after each (domain, layer) unit.
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
from postdyn.spectra import eigensystem, energy_high_low


POSITIVE_DOMAINS = ("math", "code", "instruction_following")
LAYERS = (6, 17, 28)
K_FRACTION = 1 / 3
M_TOP_K = 5
DEFAULT_N = 5000
MODEL_REPO = "allenai/Olmo-3-1025-7B"


def _load_wikitext(n: int, seed: int = 42) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [row["text"] for row in ds if row["text"].strip() and len(row["text"]) > 50]
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(texts), generator=rng)[:n].tolist()
    return [texts[i] for i in indices]


def _load_domain_prompts(domain: str, n: int) -> list[str]:
    from postdyn.data import load_pool

    pool_path = ROOT / "data" / "domain_prompts" / f"{domain}.json"
    if not pool_path.is_file():
        raise FileNotFoundError(f"Missing domain pool: {pool_path}")
    pool = load_pool(pool_path)
    return [r.prompt for r in pool.records[:n]]


def _compute_overlap_metrics(
    pos_hiddens: torch.Tensor,
    neg_hiddens: torch.Tensor,
    d_model: int,
) -> dict[str, Any]:
    k = d_model // 3

    pos_cov = OnlineCovariance()
    pos_cov.update(pos_hiddens)
    mu_pos = pos_cov.mean
    sigma_pos = pos_cov.covariance

    neg_cov = OnlineCovariance()
    neg_cov.update(neg_hiddens)
    mu_neg = neg_cov.mean

    dim_vec = mu_pos - mu_neg
    dim_norm = dim_vec.norm().item()

    sigma_vals, sigma_vecs = eigensystem(sigma_pos)

    # --- Metric 1: DiM vs Σ+ eigenvectors ---
    dim_vs_cov = energy_high_low(dim_vec, sigma_vecs, k)

    # --- Metric 2: DiM vs M+ top eigenvector ---
    second_moment = sigma_pos + mu_pos.unsqueeze(1) @ mu_pos.unsqueeze(0)
    m_vals, m_vecs = eigensystem(second_moment)

    w1 = m_vecs[:, 0]
    dim_hat = dim_vec / dim_vec.norm()
    dim_w1_overlap = float((w1 @ dim_hat).square().item())

    w1_vs_cov = energy_high_low(w1, sigma_vecs, k)

    # --- Metric 3: M+ top-5 vs Σ+ high/low bands ---
    m_top5_profiles = {}
    m_top5_e_high_avg = 0.0
    m_top5_e_low_avg = 0.0
    for j in range(M_TOP_K):
        wj_result = energy_high_low(m_vecs[:, j], sigma_vecs, k)
        m_top5_profiles[f"w{j+1}"] = {
            "E_high": wj_result["E_high"],
            "E_low": wj_result["E_low"],
            "profile": wj_result["profile"],
        }
        m_top5_e_high_avg += wj_result["E_high"]
        m_top5_e_low_avg += wj_result["E_low"]
    m_top5_e_high_avg /= M_TOP_K
    m_top5_e_low_avg /= M_TOP_K

    return {
        "dim_norm": dim_norm,
        "dim_vs_covariance": {
            "E_high": dim_vs_cov["E_high"],
            "E_low": dim_vs_cov["E_low"],
            "k": dim_vs_cov["k"],
            "random_baseline": dim_vs_cov["random_baseline"],
            "profile": dim_vs_cov["profile"],
        },
        "dim_vs_M_top1": {
            "overlap": dim_w1_overlap,
            "w1_E_high": w1_vs_cov["E_high"],
            "w1_E_low": w1_vs_cov["E_low"],
            "w1_profile": w1_vs_cov["profile"],
        },
        "M_top5_vs_covariance": {
            "avg_E_high": m_top5_e_high_avg,
            "avg_E_low": m_top5_e_low_avg,
            "per_eigenvector": m_top5_profiles,
        },
        "sigma_top5_eigenvalues": [float(v) for v in sigma_vals[:5]],
        "sigma_bottom5_eigenvalues": [float(v) for v in sigma_vals[-5:]],
        "M_top5_eigenvalues": [float(v) for v in m_vals[:5]],
    }


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
    parser.add_argument(
        "--domains", default=None,
        help="comma-separated subset of math,code,instruction_following",
    )
    parser.add_argument(
        "--layers", default=None,
        help="comma-separated layer indices (default: 6,17,28)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    from postdyn.models import load_model, release_model
    from transformers import AutoTokenizer

    domains = (
        [d.strip() for d in args.domains.split(",")]
        if args.domains
        else list(POSITIVE_DOMAINS)
    )
    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(LAYERS)
    )
    output = Path(args.output) if args.output else ROOT / "logs" / "rq1_spectral"
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

    manifest = {
        "model": MODEL_REPO,
        "n": args.n,
        "domains": domains,
        "layers": layers,
        "k_fraction": K_FRACTION,
        "m_top_k": M_TOP_K,
        "dtype": args.dtype,
        "device": args.device,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "token_budget": args.token_budget,
        "attention_budget": args.attention_budget,
    }
    atomic_write_json(run_dir.path("manifest.json"), manifest)

    with tee_log(run_dir):
        print(f"Loading WikiText negative set (n={args.n})...")
        neg_prompts = _load_wikitext(args.n)
        print(f"Loaded {len(neg_prompts)} WikiText prompts")

        pos_prompts: dict[str, list[str]] = {}
        for domain in domains:
            prompts = _load_domain_prompts(domain, args.n)
            pos_prompts[domain] = prompts
            print(f"Loaded {len(prompts)} {domain} prompts")

        from postdyn.config import MODEL_FAMILIES, CheckpointRef

        ref = CheckpointRef("base", MODEL_REPO, "main", "base")
        print(f"Loading model {MODEL_REPO}...")
        model = load_model(ref, args.dtype, None, args.device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        d_model = model.config.hidden_size
        print(f"Model loaded (d_model={d_model})")

        neg_hiddens: dict[int, torch.Tensor] = {}
        try:
            need_neg = any(
                (domain, layer) not in completed
                for domain in domains
                for layer in layers
            )
            if need_neg:
                print(f"Extracting WikiText hidden states ({len(neg_prompts)} prompts)...")
                started = time.monotonic()
                neg_hiddens = extract_layer_hiddens(
                    model, tokenizer, neg_prompts, layers,
                    args.batch_size, args.max_length,
                    token_budget=args.token_budget,
                    attention_budget=args.attention_budget,
                    return_device=args.device,
                )
                print(f"WikiText extraction: {time.monotonic() - started:.1f}s")

            for domain in domains:
                need_domain = any(
                    (domain, layer) not in completed for layer in layers
                )
                if not need_domain:
                    print(f"[skip] {domain}: all layers complete")
                    continue

                print(f"Extracting {domain} hidden states ({len(pos_prompts[domain])} prompts)...")
                started = time.monotonic()
                pos_hiddens = extract_layer_hiddens(
                    model, tokenizer, pos_prompts[domain], layers,
                    args.batch_size, args.max_length,
                    token_budget=args.token_budget,
                    attention_budget=args.attention_budget,
                    return_device=args.device,
                )
                print(f"{domain} extraction: {time.monotonic() - started:.1f}s")

                for layer in layers:
                    if (domain, layer) in completed:
                        print(f"[skip] {domain}/layer_{layer}")
                        continue

                    print(f"Computing metrics: {domain}/layer_{layer}...")
                    metrics = _compute_overlap_metrics(
                        pos_hiddens[layer], neg_hiddens[layer], d_model
                    )
                    row = {
                        "domain": domain,
                        "layer": layer,
                        "n_pos": len(pos_prompts[domain]),
                        "n_neg": len(neg_prompts),
                        **metrics,
                    }
                    append_jsonl(metrics_path, row)
                    completed.add((domain, layer))

                    print(
                        f"  {domain}/layer_{layer}: "
                        f"E_high={metrics['dim_vs_covariance']['E_high']:.4f} "
                        f"E_low={metrics['dim_vs_covariance']['E_low']:.4f} "
                        f"DiM-w1={metrics['dim_vs_M_top1']['overlap']:.4f}"
                    )

                del pos_hiddens
                torch.cuda.empty_cache()

        finally:
            release_model(model)

        # Write summary
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
