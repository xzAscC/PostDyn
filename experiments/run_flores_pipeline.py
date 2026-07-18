#!/usr/bin/env python3
"""Pipelined French-English extraction with model prefetch overlap."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download

from src.concept_dynamics import (
    _clean_hf_cache,
    compute_dynamics_analysis,
    run_model_extraction,
)
from src.config import EXPERIMENT_LAYERS_7B, MODEL_CHECKPOINTS, OLMO3_VARIANTS
from src.contrastive_datasets import load_flores_pairs

CONCEPT = "french_vs_english_language"
ALL_CONCEPTS = [
    "python_vs_cpp",
    "concise_math_reasoning_vs_verbose_math_reasoning",
    "french_vs_english_language",
    "female_vs_male_gender",
]
OUTPUT_DIR = "results/concept_dynamics_paired"
LAYERS = EXPERIMENT_LAYERS_7B
N_SAMPLES = 50
MAX_SEQ_LEN = 2048
MODEL_ORDER = [
    "olmo3-think-sft",
    "olmo3-instruct-sft",
    "olmo3-rl-zero-math",
    "olmo3-rl-zero-code",
    "olmo3-rl-zero-if",
    "olmo3-rl-zero-general",
    "olmo3-rl-zero-mix",
]


def _results_path() -> Path:
    return Path(OUTPUT_DIR) / "extraction_results.json"


def _load_results() -> dict:
    path = _results_path()
    if path.exists():
        return json.loads(path.read_text())
    return {"checkpoints_done": [], "extraction": {}}


def _save_results(results: dict) -> None:
    path = _results_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def _planned_jobs() -> list[tuple[str, str, Any]]:
    jobs: list[tuple[str, str, Any]] = []
    for name in MODEL_ORDER:
        config = OLMO3_VARIANTS[name]
        for ckpt in MODEL_CHECKPOINTS.get(name, ["main"]):
            jobs.append((name, ckpt, config))
    return jobs


def _checkpoint_has_concept(
    vectors_dir: str,
    name: str,
    ckpt: str,
    concept: str,
    layers: list[int] | None = None,
) -> bool:
    required_layers = layers if layers is not None else list(LAYERS)
    ckpt_dir = Path(vectors_dir) / name / ckpt
    if not ckpt_dir.is_dir():
        return False
    for layer in required_layers:
        meta_path = ckpt_dir / f"layer_{layer}.json"
        tensor_path = ckpt_dir / f"layer_{layer}.safetensors"
        if not meta_path.is_file() or not tensor_path.is_file():
            return False
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        names = {
            entry.get("name")
            for entry in metadata.get("concepts", [])
            if isinstance(entry, dict)
        }
        if concept not in names:
            return False
    return True


def _prefetch(config: Any, revision: str) -> str:
    print(f"  [prefetch] {config.hf_id} rev={revision}", flush=True)
    path = snapshot_download(
        repo_id=config.hf_id,
        revision=revision,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        max_workers=8,
    )
    print(f"  [prefetch done] {config.hf_id} rev={revision}", flush=True)
    return path


def main() -> None:
    vectors_dir = str(Path(OUTPUT_DIR) / "vectors")
    Path(vectors_dir).mkdir(parents=True, exist_ok=True)
    results = _load_results()
    jobs = [
        (n, c, cfg)
        for n, c, cfg in _planned_jobs()
        if not _checkpoint_has_concept(vectors_dir, n, c, CONCEPT, LAYERS)
    ]
    print(f"Planned remaining jobs: {len(jobs)}", flush=True)

    load_flores_pairs(N_SAMPLES)
    print("FLORES pairs cached", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending: dict[int, Future] = {}

            def ensure_prefetch(job_idx: int) -> None:
                if job_idx >= len(jobs) or job_idx in pending:
                    return
                _, n_ckpt, n_config = jobs[job_idx]
                pending[job_idx] = pool.submit(_prefetch, n_config, n_ckpt)

            ensure_prefetch(0)

            for idx, (name, ckpt, config) in enumerate(jobs):
                ckpt_key = f"{name}/{ckpt}"
                ensure_prefetch(idx)
                pending.pop(idx).result()
                ensure_prefetch(idx + 1)

                try:
                    stats = run_model_extraction(
                        config,
                        [CONCEPT],
                        LAYERS,
                        N_SAMPLES,
                        vectors_dir,
                        MAX_SEQ_LEN,
                        checkpoint=ckpt,
                        revision=ckpt,
                    )
                    prior = results.get("extraction", {}).get(ckpt_key, {})
                    if isinstance(prior, dict) and "concepts" in prior:
                        concepts = list(dict.fromkeys(prior["concepts"] + [CONCEPT]))
                        stats = {**stats, "concepts": concepts}
                    results.setdefault("extraction", {})[ckpt_key] = stats
                    if ckpt_key not in results["checkpoints_done"]:
                        results["checkpoints_done"].append(ckpt_key)
                except Exception as exc:
                    import traceback

                    traceback.print_exc()
                    results.setdefault("extraction", {})[ckpt_key] = {"error": str(exc)}

                results["scope"] = "flores_french_english_in_progress"
                _save_results(results)

                remaining_for_model = [j for j in jobs[idx + 1 :] if j[0] == name]
                if not remaining_for_model:
                    for jidx in list(pending):
                        if jobs[jidx][0] == name:
                            pending.pop(jidx).result()
                    next_name = jobs[idx + 1][0] if idx + 1 < len(jobs) else None
                    if next_name != name:
                        _clean_hf_cache(config.hf_id)

    print("Recomputing four-concept dynamics...", flush=True)
    dynamics = compute_dynamics_analysis(
        OUTPUT_DIR,
        MODEL_ORDER,
        ALL_CONCEPTS,
        LAYERS,
    )
    results = _load_results()
    results["dynamics"] = dynamics
    results["scope"] = "full_four_concept_runtime_evidence"
    _save_results(results)
    n_errors = sum(
        1
        for v in results.get("extraction", {}).values()
        if isinstance(v, dict) and "error" in v
    )
    print(
        f"DONE checkpoints={len(results.get('checkpoints_done', []))} errors={n_errors}",
        flush=True,
    )
    if n_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
