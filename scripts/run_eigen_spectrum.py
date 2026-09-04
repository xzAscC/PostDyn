#!/usr/bin/env python3
"""Eigenvalue-extreme summary for math vs wikitext at the final checkpoints.

One focused experiment (PostDyn): with 10k raw prompts per domain, measure at
every slide-formula layer of two models

  * ``lambda_min`` of each domain covariance (Sigma_math, Sigma_wiki), and
  * the signed extremes of the difference DeltaSigma = Sigma_math - Sigma_wiki
    (largest positive / most negative eigenvalue),

Models (Olmo-3-7B Think family, immutable ``main`` revisions):

  * ``sft_final``  — allenai/Olmo-3-7B-Think-SFT
  * ``rlvr_final`` — allenai/Olmo-3-7B-Think

The distribution of each quantity is taken across the 10 sampled layers.
Prompts, tokenizer preflight, extraction (final attention token, raw prompt,
no chat template) reuse the Think differential-subspace pipeline verbatim.

Writes incrementally under ``logs/eig_spectrum_10k/``::

    run.log                          (tee of every print)
    prompts/{math,wikitext}.json     (deterministic draw + provenance)
    layers/{model}/layer_{L}.json    (written after each layer)
    summary.json                     (rewritten after each model)

Figure (PDF): ``figs/eig_spectrum_10k.pdf``.

Usage::

    uv run python scripts/run_eigen_spectrum.py
    uv run python scripts/run_eigen_spectrum.py --quick
    uv run python scripts/run_eigen_spectrum.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from postdyn.config import EXPERIMENT_LAYERS_7B, FIGS_DIR, PROJECT_ROOT
from postdyn.concept_dynamics import _clean_hf_cache, _load_model_and_tokenizer
from postdyn.dataset_store import SHARED_SAMPLE_SEED
from postdyn.eigen_spectrum import (
    DEFAULT_TAIL_K,
    atomic_write_json,
    build_layer_metrics,
    plot_layer_lines,
)
from postdyn.think_sft_differential_experiment import model_config

# Allow both `python -m scripts.run_eigen_spectrum` and direct-script invocation.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.run_think_sft_differential_subspace as think_runner

RESULTS_ROOT = Path(PROJECT_ROOT) / "logs" / "eigen_spectrum_10k"
RESULTS_ROOT_QUICK = Path(PROJECT_ROOT) / "logs" / "eigen_spectrum_10k_quick"
FIGURE_PATH = Path(FIGS_DIR) / "eig_spectrum_10k.pdf"
N_SAMPLES: int = 10_000
DOMAINS: tuple[str, str] = ("math", "wikitext")

#: label -> postdyn model key; both resolved at their released ``main``.
MODEL_KEYS: dict[str, str] = {
    "sft_final": "olmo3-think-sft",
    "rlvr_final": "olmo3-think-rlvr",
}


class Tee:
    """Duplicate stdout to a log file (incremental-run rule)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        self._stdout = sys.stdout

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._file.write(data)
        if "\n" in data:
            self.flush()
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()


def _setup_signature(
    *,
    label: str,
    model_key: str,
    revision: str,
    layers: list[int],
    n_samples: int,
    seed: int,
    tail_k: int,
    fingerprints: dict[str, str],
) -> str:
    payload = {
        "label": label,
        "model_key": model_key,
        "revision": revision,
        "layers": layers,
        "n_samples": n_samples,
        "seed": seed,
        "tail_k": tail_k,
        "prompt_fingerprints": sorted(fingerprints.items()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _layer_path(root: Path, label: str, layer: int) -> Path:
    return root / "layers" / label / f"layer_{layer}.json"


def _layer_complete(path: Path, setup_sig: str) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(data, dict)
        and data.get("setup_signature") == setup_sig
        and isinstance(data.get("metrics"), dict)
    )


def _row_from_payload(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "model": label,
        "layer": int(payload["layer"]),
        "lambda_min_concept": metrics["concept"]["lambda_min"],
        "lambda_min_reference": metrics["reference"]["lambda_min"],
        "lambda_max_pos": metrics["difference"]["lambda_max_pos"],
        "lambda_min_neg": metrics["difference"]["lambda_min_neg"],
    }


def collect_rows(
    root: Path, labels: list[str], layers: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in labels:
        for layer in layers:
            path = _layer_path(root, label, layer)
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(_row_from_payload(label, payload))
    rows.sort(key=lambda row: (str(row["model"]), int(row["layer"])))
    return rows


def run_model(
    label: str,
    *,
    root: Path,
    layers: list[int],
    n_samples: int,
    prompts: dict[str, list[str]],
    seed: int,
    tail_k: int,
    max_seq_len: int,
    token_budget: int,
    keep_hf_cache: bool,
    revision: str,
    setup_sig: str,
) -> list[dict[str, Any]]:
    model_key = MODEL_KEYS[label]
    cfg = model_config(model_key)
    wanted = [
        _layer_complete(_layer_path(root, label, layer), setup_sig) for layer in layers
    ]
    if all(wanted):
        print(f"[skip] {label} ({cfg.hf_id}@{revision}) already complete")
        return collect_rows(root, [label], layers)

    print(f"\n{'=' * 60}")
    print(f"Eigen extremes: {label} = {cfg.hf_id}@{revision}")
    print(f"Layers={layers} n_samples={n_samples} tail_k={tail_k}")
    print(f"{'=' * 60}")

    t0 = time.time()
    model, tokenizer = _load_model_and_tokenizer(cfg, revision)
    acts: dict[str, dict[int, torch.Tensor]] = {}
    try:
        for domain in DOMAINS:
            texts = prompts[domain][:n_samples]
            print(f"  Extracting domain={domain} n={len(texts)} ...")
            lengths = think_runner.preflight_tokenizer_prompts(
                tokenizer, texts, max_seq_len
            )
            acts[domain] = think_runner.extract_raw_layer_activations(
                model,
                tokenizer,
                texts,
                layers,
                token_budget=token_budget,
                lengths=lengths,
            )
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for layer in layers:
        t_layer = time.time()
        metrics = build_layer_metrics(
            acts["math"][layer], acts["wikitext"][layer], k=tail_k
        )
        payload = {
            "model": label,
            "model_key": model_key,
            "hf_id": cfg.hf_id,
            "revision": revision,
            "layer": layer,
            "n_samples": n_samples,
            "n_concept": metrics["concept"]["n"],
            "n_reference": metrics["reference"]["n"],
            "domains": {"concept": "math", "reference": "wikitext"},
            "tail_k": tail_k,
            "setup_signature": setup_sig,
            "metrics": metrics,
            "elapsed_seconds": round(time.time() - t_layer, 2),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(_layer_path(root, label, layer), payload)
        diff = metrics["difference"]
        print(
            f"  layer={layer}: "
            f"lam_min(math)={metrics['concept']['lambda_min']:.3e} "
            f"lam_min(wiki)={metrics['reference']['lambda_min']:.3e} "
            f"lam_max(Delta)={diff['lambda_max_pos']:.3e} "
            f"lam_min(Delta)={diff['lambda_min_neg']:.3e} "
            f"({time.time() - t_layer:.1f}s)"
        )

    if not keep_hf_cache:
        try:
            _clean_hf_cache(cfg.hf_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  cache clean warning: {exc}")

    print(f"  Done {label} in {time.time() - t0:.1f}s")
    return collect_rows(root, [label], layers)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=None, help="Output root directory")
    p.add_argument(
        "--models",
        type=str,
        default="sft_final,rlvr_final",
        help=f"Comma-separated subset of {sorted(MODEL_KEYS)}",
    )
    p.add_argument("--layers", type=str, default=None, help="Comma-separated layers")
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--tail-k", type=int, default=DEFAULT_TAIL_K)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=SHARED_SAMPLE_SEED)
    p.add_argument("--extract-token-budget", type=int, default=8192)
    p.add_argument("--keep-hf-cache", action="store_true")
    p.add_argument("--no-hf", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--figure", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--quick", action="store_true", help="Smoke: sft_final, 2 layers, 16 samples"
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    layers = list(EXPERIMENT_LAYERS_7B)
    n_samples = args.samples
    if args.quick:
        root = args.output or RESULTS_ROOT_QUICK
        labels = ["sft_final"]
        layers = layers[:2]
        n_samples = think_runner.quick_sample_count(n_samples)
    else:
        root = args.output or RESULTS_ROOT
        requested = [x.strip() for x in args.models.split(",") if x.strip()]
        unknown = sorted(set(requested) - set(MODEL_KEYS))
        if unknown:
            raise ValueError(
                f"unknown model label(s) {unknown}; valid: {sorted(MODEL_KEYS)}"
            )
        labels = requested
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if not layers or n_samples <= 0:
        raise ValueError("layers and samples must be positive")
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate layer selection")

    root.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(root / "run.log")  # type: ignore[assignment]

    print(f"Output root: {root}")
    print(
        f"Models: {[(label, model_config(MODEL_KEYS[label]).hf_id) for label in labels]}"
    )
    print(f"Layers ({len(layers)}): {layers}")
    print(f"Samples/domain: {n_samples}  tail_k={args.tail_k}  seed={args.seed}")
    if args.dry_run:
        print("Dry run: plan validated; no datasets or models loaded.")
        return 0

    think_runner.apply_concept_filter("math_vs_wikitext")
    prompts = think_runner.prepare_domain_prompts(
        root,
        n_samples=n_samples,
        seed=args.seed,
        allow_hf=not args.no_hf,
        max_seq_len=args.max_seq_len,
        use_chat_template=False,
    )
    fingerprints = {
        domain: think_runner._prompt_fingerprint(prompts[domain][:n_samples])
        for domain in DOMAINS
    }
    for domain in DOMAINS:
        print(f"  domain {domain}: {len(prompts[domain][:n_samples])} prompts")

    rows: list[dict[str, Any]] = []
    model_records: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}
    for label in labels:
        cfg = model_config(MODEL_KEYS[label])
        revision = think_runner.resolve_model_revision(cfg.hf_id, cfg.revision)
        setup_sig = _setup_signature(
            label=label,
            model_key=MODEL_KEYS[label],
            revision=revision,
            layers=layers,
            n_samples=n_samples,
            seed=args.seed,
            tail_k=args.tail_k,
            fingerprints=fingerprints,
        )
        rows += run_model(
            label,
            root=root,
            layers=layers,
            n_samples=n_samples,
            prompts=prompts,
            seed=args.seed,
            tail_k=args.tail_k,
            max_seq_len=args.max_seq_len,
            token_budget=args.extract_token_budget,
            keep_hf_cache=args.keep_hf_cache,
            revision=revision,
            setup_sig=setup_sig,
        )
        model_records[label] = {
            "model_key": MODEL_KEYS[label],
            "hf_id": cfg.hf_id,
            "revision": revision,
            "n_samples": {d: len(prompts[d][:n_samples]) for d in DOMAINS},
        }
        summary = {
            "experiment": "eigen_spectrum_10k",
            "domains": {"concept": "math", "reference": "wikitext"},
            "n_samples": n_samples,
            "seed": args.seed,
            "tail_k": args.tail_k,
            "layers": layers,
            "models": model_records,
            "rows": rows,
            "figure": None,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(root / "summary.json", summary)

    figure_path = args.figure
    if figure_path is None:
        name = "eig_spectrum_10k_quick.pdf" if args.quick else "eig_spectrum_10k.pdf"
        figure_path = Path(FIGS_DIR) / name
    if not args.no_plot and rows:
        returned = plot_layer_lines(rows, Path(figure_path))
        if returned is not None:
            print(f"Figure: {returned}")
            summary["figure"] = str(returned)
            atomic_write_json(root / "summary.json", summary)

    print("\n===== EIGEN SPECTRUM READY =====")
    print(f"summary: {root / 'summary.json'}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
