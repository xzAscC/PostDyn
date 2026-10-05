"""RQ1 spectral overlap for safety: harmful vs benign responses.

Positive set: rejected responses from LLM-LAT/harmful-dataset.
Negative set: responses from LLM-LAT/benign-dataset.

Computes the same three overlap metrics as run_rq1_spectral.py:
  1. E_high / E_low of DiM in Σ+ eigenvectors.
  2. DiM alignment with M+ top eigenvector.
  3. M+ top-5 vs Σ+ high/low bands.
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

LAYERS = (6, 17, 28)
K_FRACTION = 1 / 3
M_TOP_K = 5
MODEL_REPO = "allenai/Olmo-3-1025-7B"


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

    dim_vs_cov = energy_high_low(dim_vec, sigma_vecs, k)

    second_moment = sigma_pos + mu_pos.unsqueeze(1) @ mu_pos.unsqueeze(0)
    m_vals, m_vecs = eigensystem(second_moment)

    w1 = m_vecs[:, 0]
    dim_hat = dim_vec / dim_vec.norm()
    dim_w1_overlap = float((w1 @ dim_hat).square().item())

    w1_vs_cov = energy_high_low(w1, sigma_vecs, k)

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
    parser.add_argument("--n", type=int, default=None,
                        help="samples per set (default: all harmful, matched benign)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--token-budget", type=int, default=4096)
    parser.add_argument("--attention-budget", type=int, default=8_388_608)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--layers", default=None,
                        help="comma-separated layer indices (default: 6,17,28)")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    from postdyn.models import load_model, release_model
    from transformers import AutoTokenizer

    layers = (
        [int(x) for x in args.layers.split(",")]
        if args.layers
        else list(LAYERS)
    )
    output = Path(args.output) if args.output else ROOT / "logs" / "rq1_spectral_safety"
    run_dir = RunDir(output)
    completed = set()
    metrics_path = run_dir.path("metrics.jsonl")
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed.add(row["layer"])

    print("Loading harmful responses (LLM-LAT/harmful-dataset rejected)...")
    pos_texts = _load_harmful(args.n)
    n = len(pos_texts)
    print(f"Loaded {n} harmful responses")

    print(f"Loading benign responses (LLM-LAT/benign-dataset response, n={n})...")
    neg_texts = _load_benign(n)
    print(f"Loaded {len(neg_texts)} benign responses")

    manifest = {
        "model": MODEL_REPO,
        "positive": "LLM-LAT/harmful-dataset (rejected)",
        "negative": "LLM-LAT/benign-dataset (response)",
        "n_pos": n,
        "n_neg": len(neg_texts),
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
        from postdyn.config import CheckpointRef

        ref = CheckpointRef("base", MODEL_REPO, "main", "base")
        print(f"Loading model {MODEL_REPO}...")
        model = load_model(ref, args.dtype, None, args.device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        d_model = model.config.hidden_size
        print(f"Model loaded (d_model={d_model})")

        try:
            need_any = any(layer not in completed for layer in layers)
            if not need_any:
                print("All layers already complete, nothing to do.")
                return 0

            print(f"Extracting harmful hidden states ({n} prompts)...")
            started = time.monotonic()
            pos_hiddens = extract_layer_hiddens(
                model, tokenizer, pos_texts, layers,
                args.batch_size, args.max_length,
                token_budget=args.token_budget,
                attention_budget=args.attention_budget,
                return_device=args.device,
            )
            print(f"Harmful extraction: {time.monotonic() - started:.1f}s")

            print(f"Extracting benign hidden states ({len(neg_texts)} prompts)...")
            started = time.monotonic()
            neg_hiddens = extract_layer_hiddens(
                model, tokenizer, neg_texts, layers,
                args.batch_size, args.max_length,
                token_budget=args.token_budget,
                attention_budget=args.attention_budget,
                return_device=args.device,
            )
            print(f"Benign extraction: {time.monotonic() - started:.1f}s")

            for layer in layers:
                if layer in completed:
                    print(f"[skip] safety/layer_{layer}")
                    continue

                print(f"Computing metrics: safety/layer_{layer}...")
                metrics = _compute_overlap_metrics(
                    pos_hiddens[layer], neg_hiddens[layer], d_model
                )
                row = {
                    "domain": "safety",
                    "layer": layer,
                    "n_pos": n,
                    "n_neg": len(neg_texts),
                    **metrics,
                }
                append_jsonl(metrics_path, row)
                completed.add(layer)

                print(
                    f"  safety/layer_{layer}: "
                    f"E_high={metrics['dim_vs_covariance']['E_high']:.4f} "
                    f"E_low={metrics['dim_vs_covariance']['E_low']:.4f} "
                    f"DiM-w1={metrics['dim_vs_M_top1']['overlap']:.4f}"
                )

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
