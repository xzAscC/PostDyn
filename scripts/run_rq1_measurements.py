"""RQ1 Measurements 1-4 (paper Sec. RQ1 approach) on allenai/Olmo-3-1025-7B.

Domains: math, code, instruction_following (reference: WikiText) and safety
(harmful-dataset rejected vs benign-dataset response). Each set is split into
an estimation part (DiM, Sigma+, M+) and a held-out evaluation part (projected
means/variances and per-token DiM cosines). Observations for the moments are
final-token states; Measurement 4 uses every token of the evaluation prompts.
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
from safetensors.torch import save_file

from postdyn.extract import OnlineCovariance, extract_layer_hiddens, extract_token_cosines
from postdyn.persistence import RunDir, append_jsonl, atomic_write_json, tee_log
from postdyn.spectra import (
    cosine_summary,
    eigensystem,
    energy_curve,
    energy_profile,
    mean_contributions,
    split_indices,
    subspace_overlap_curve,
)
from scripts.run_rq1_dim_projection import (
    _load_benign,
    _load_domain_prompts,
    _load_harmful,
    _load_wikitext,
)

LAYERS = (6, 17, 28)
MODEL_REPO = "allenai/Olmo-3-1025-7B"
DOMAINS = {
    "math": "wikitext",
    "code": "wikitext",
    "instruction_following": "wikitext",
    "safety": "benign",
}


def k_grid(d: int) -> list[int]:
    K = d // 3
    ks = [2**p for p in range(K.bit_length()) if 2**p < K]
    return ks + [K]


def _moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    acc = OnlineCovariance()
    acc.update(x)
    return acc.mean, acc.covariance


def layer_measurements(
    pos_est: torch.Tensor,
    neg_est: torch.Tensor,
    pos_eval: torch.Tensor,
    neg_eval: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    d = pos_est.shape[1]
    K = d // 3
    ks = k_grid(d)
    mu_p, sigma = _moments(pos_est)
    mu_n, _ = _moments(neg_est)
    dim = mu_p - mu_n
    d_hat = dim / dim.norm()
    lam, u = eigensystem(sigma)
    m = sigma + torch.outer(mu_p, mu_p)
    eta, w = eigensystem(m)

    m1 = energy_curve(dim, u, ks)
    m1["profile"] = energy_profile(dim, u).tolist()
    m2 = energy_curve(dim, w, ks)
    m3 = mean_contributions(sigma, mu_p, eta, w, K)

    mu_pe, sigma_e = _moments(pos_eval)
    m3_eval = mean_contributions(sigma_e, mu_pe, eta, w, K)
    eval_proj = {}
    for name, h in (("pos", pos_eval), ("neg", neg_eval)):
        p = h.double() @ d_hat
        eval_proj[name] = {"mean": float(p.mean()), "var": float(p.var(unbiased=False))}

    row = {
        "d": d,
        "K": K,
        "dim_norm": float(dim.norm()),
        "m1": m1,
        "m2": {"k": m2["k"], "E_M": m2["E_high"], "E_M_K": m2["E_high"][-1]},
        "m3": m3,
        "m3_eval": {k: m3_eval[k] for k in ("rho_bar", "rho_var", "R_K", "v", "b")},
        "dim_projection_eval": eval_proj,
        "sigma_eigenvalues": lam.tolist(),
        "u_mu_pos": (u.T @ mu_p).tolist(),
        "u_mu_neg": (u.T @ mu_n.double()).tolist(),
        "A_k": subspace_overlap_curve(u, w, ks),
    }
    return row, dim


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-est", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=1000)
    p.add_argument("--min-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--token-budget", type=int, default=4096)
    p.add_argument("--attention-budget", type=int, default=8_388_608)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", default=",".join(map(str, LAYERS)))
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--output", default=str(ROOT / "logs" / "rq1_measurements"))
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from postdyn.config import CheckpointRef
    from postdyn.models import load_model, release_model

    layers = [int(x) for x in args.layers.split(",")]
    domains = args.domains.split(",")
    run_dir = RunDir(args.output)
    metrics_path = run_dir.path("metrics.jsonl")
    done: set[tuple[str, int]] = set()
    if metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["domain"], r["layer"]))
    todo = [dom for dom in domains if any((dom, l) not in done for l in layers)]
    atomic_write_json(run_dir.path("manifest.json"), {**vars(args), "model": MODEL_REPO})

    with tee_log(run_dir):
        if not todo:
            print("All units complete.")
            return 0
        texts: dict[str, list[str]] = {}
        n_total = args.n_est + args.n_eval
        for dom in todo:
            texts[dom] = _load_harmful(n_total) if dom == "safety" else _load_domain_prompts(dom, n_total)
            ref = DOMAINS[dom]
            if ref not in texts:
                texts[ref] = (_load_wikitext if ref == "wikitext" else _load_benign)(n_total)
        for name, t in texts.items():
            print(f"{name}: {len(t)} texts")

        model = load_model(CheckpointRef("base", MODEL_REPO, "main", "base"), args.dtype, None, args.device)
        tok = AutoTokenizer.from_pretrained(MODEL_REPO)
        try:
            last: dict[str, dict[int, torch.Tensor]] = {}
            splits: dict[str, tuple[list[int], list[int]]] = {}

            def states(name: str) -> dict[int, torch.Tensor]:
                if name not in last:
                    t0 = time.monotonic()
                    last[name] = extract_layer_hiddens(
                        model, tok, texts[name], layers, args.batch_size, args.max_length,
                        token_budget=args.token_budget, attention_budget=args.attention_budget,
                        return_device=args.device,
                    )
                    n = len(texts[name])
                    n_eval = max(min(args.n_eval, n - args.n_est), min(args.min_eval, n))
                    splits[name] = split_indices(n, n_eval / n, args.seed)
                    print(f"[extract] {name}: {time.monotonic() - t0:.1f}s")
                return last[name]

            for dom in todo:
                ref = DOMAINS[dom]
                hp, hn = states(dom), states(ref)
                (pe, pv), (ne, nv) = splits[dom], splits[ref]
                dims: dict[int, torch.Tensor] = {}
                rows: dict[int, dict[str, Any]] = {}
                for l in layers:
                    row, dims[l] = layer_measurements(hp[l][pe], hn[l][ne], hp[l][pv], hn[l][nv])
                    rows[l] = row
                t0 = time.monotonic()
                tensors: dict[str, torch.Tensor] = {}
                cos_summary: dict[int, dict[str, Any]] = {l: {} for l in layers}
                for side, name, idx in (("pos", dom, pv), ("neg", ref, nv)):
                    cos = extract_token_cosines(
                        model, tok, [texts[name][i] for i in idx], dims,
                        max_length=args.max_length, token_budget=args.token_budget,
                    )
                    for l in layers:
                        cos_summary[l][side] = cosine_summary(cos[l])
                        tensors[f"L{l}_{side}_cos"] = torch.cat(cos[l])
                        tensors[f"L{l}_{side}_len"] = torch.tensor([len(c) for c in cos[l]])
                save_file(tensors, str(run_dir.path(f"token_cosines_{dom}.safetensors")))
                print(f"[tokens] {dom}: {time.monotonic() - t0:.1f}s")
                for l in layers:
                    if (dom, l) in done:
                        continue
                    record = {"domain": dom, "reference": ref, "layer": l,
                              "n_est": [len(pe), len(ne)], "n_eval": [len(pv), len(nv)],
                              **rows[l], "m4": cos_summary[l]}
                    append_jsonl(metrics_path, record)
                    m3, m4 = rows[l]["m3"], cos_summary[l]
                    print(
                        f"  {dom}/L{l}: E_high(K)={rows[l]['m1']['E_high'][-1]:.3f} "
                        f"E_low(K)={rows[l]['m1']['E_low'][-1]:.3f} E_M(K)={rows[l]['m2']['E_M_K']:.3f} "
                        f"rho_bar={m3['rho_bar']:.2e} R_K={m3['R_K']:.3f} "
                        f"cos+={m4['pos']['mean']:.3f}±{m4['pos']['std']:.3f} "
                        f"cos-={m4['neg']['mean']:.3f}±{m4['neg']['std']:.3f}"
                    )
                del last[dom]
                torch.cuda.empty_cache()
        finally:
            release_model(model)

        results = [json.loads(x) for x in metrics_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        atomic_write_json(run_dir.path("summary.json"), {"manifest": vars(args), "results": results})
        print(f"Done. {run_dir.path('summary.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
