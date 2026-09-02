#!/usr/bin/env python3
"""One-shot migration: legacy v0 concept sidecars → provenance-bound v1.

Rewrites **JSON sidecars only**. Tensor ``.safetensors`` bytes are never
modified. Source-text fingerprints are derived from the current canonical
contrastive loaders (the same texts used by the extraction driver).

Usage::

    uv run python experiments/migrate_concept_sidecars.py [--dry-run]
    uv run python experiments/migrate_concept_sidecars.py --root results/rl_zero_code_syntax

Exit codes:
    0  all sidecars migrated or already valid v1
    1  one or more sidecars flagged for re-extraction / validation failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.concept_dynamics import (  # noqa: E402
    EXPECTED_D_MODEL,
    SIDECAR_SCHEMA,
    SIDECAR_VERSION,
    build_concept_source_entries,
    compute_sidecar_source_fingerprint,
    validate_concept_sidecar,
)
from src.config import OLMO3_VARIANTS  # noqa: E402
from src.contrastive_datasets import load_contrastive_texts  # noqa: E402
from src.rl_zero_experiment import (  # noqa: E402
    EXPERIMENT_CONCEPTS,
    N_SAMPLES,
    PRIMARY_USE_CHAT_TEMPLATE,
    RL_ZERO_CODE_RESULTS_ROOT,
)

DEFAULT_MAX_SEQ_LEN = 2048
PROTOCOL = "raw"
CONCEPT_VECTORS_SUBDIR = "concept_vectors"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".migrate_", suffix=".json.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_canonical_concept_sources(
    n_samples: int = N_SAMPLES,
) -> dict[str, tuple[list[str], list[str]]]:
    sources: dict[str, tuple[list[str], list[str]]] = {}
    for concept in EXPERIMENT_CONCEPTS:
        pos, neg = load_contrastive_texts(concept, n_samples)
        sources[concept] = (pos, neg)
    return sources


def _model_meta(model_name: str) -> tuple[str, str]:
    cfg = OLMO3_VARIANTS.get(model_name)
    if cfg is None:
        return model_name, ""
    return cfg.hf_id, getattr(cfg, "revision", "") or ""


def build_v1_sidecar(
    legacy: dict[str, Any],
    *,
    model_name: str,
    checkpoint: str,
    layer_idx: int,
    concept_sources: dict[str, tuple[list[str], list[str]]],
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> dict[str, Any]:
    concepts_in = legacy.get("concepts")
    if not isinstance(concepts_in, list) or not concepts_in:
        raise ValueError("legacy sidecar missing concepts list")

    names = [c["name"] for c in concepts_in if isinstance(c, dict) and "name" in c]
    if set(names) != set(EXPERIMENT_CONCEPTS):
        raise ValueError(
            f"concept set mismatch: got {sorted(names)}, expected {list(EXPERIMENT_CONCEPTS)}"
        )

    filtered = {n: concept_sources[n] for n in sorted(names)}
    entries = build_concept_source_entries(filtered)
    by_name = {e[0]: e for e in entries}
    aggregate_fp = compute_sidecar_source_fingerprint([(e[0], e[1]) for e in entries])
    hf_id, _base_rev = _model_meta(model_name)
    revision = checkpoint

    d_models = [int(c.get("d_model", -1)) for c in concepts_in if isinstance(c, dict)]
    if not d_models or any(d != EXPECTED_D_MODEL for d in d_models):
        raise ValueError(f"unexpected d_model values: {d_models}")

    out_concepts: list[dict[str, Any]] = []
    for c in concepts_in:
        if not isinstance(c, dict):
            raise ValueError("concept entry is not an object")
        name = c["name"]
        _, fp, pos_sha, neg_sha = by_name[name]
        n_pos = int(c["n_positive"])
        n_neg = int(c["n_negative"])
        if n_pos != len(pos_sha) or n_neg != len(neg_sha):
            raise ValueError(
                f"{name}: n_positive/n_negative ({n_pos}/{n_neg}) do not match "
                f"canonical source lengths ({len(pos_sha)}/{len(neg_sha)})"
            )
        out_concepts.append(
            {
                "name": name,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "d_model": int(c["d_model"]),
                "positive_text_sha256": pos_sha,
                "negative_text_sha256": neg_sha,
                "source_fingerprint": fp,
            }
        )

    return {
        "schema": SIDECAR_SCHEMA,
        "version": SIDECAR_VERSION,
        "concepts": out_concepts,
        "layer_idx": layer_idx,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "protocol": PROTOCOL,
        "revision": revision,
        "hf_id": hf_id,
        "max_seq_len": max_seq_len,
        "use_chat_template": bool(PRIMARY_USE_CHAT_TEMPLATE),
        "source_fingerprint": aggregate_fp,
        "d_model": EXPECTED_D_MODEL,
        # Honest origin: tensors were not recomputed; source hashes are bound
        # to the current canonical texts and must be treated as assumed.
        "provenance_origin": "migrated_v0_assumed_canonical_sources",
        "tensor_sha256": None,  # filled by migrate_one after hashing
    }


def migrate_one(
    json_path: Path,
    concept_sources: dict[str, tuple[list[str], list[str]]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Migrate one sidecar. Returns a report row."""
    st_path = json_path.with_suffix(".safetensors")
    rel = str(json_path)
    if not st_path.is_file():
        return {
            "path": rel,
            "status": "re_extract_required",
            "reason": "missing_tensor",
        }

    parts = json_path.parts
    # .../concept_vectors/<model>/<checkpoint>/layer_N.json
    try:
        cv_idx = parts.index(CONCEPT_VECTORS_SUBDIR)
        model_name = parts[cv_idx + 1]
        checkpoint = parts[cv_idx + 2]
        layer_name = json_path.stem  # layer_28
        layer_idx = int(layer_name.split("_", 1)[1])
    except (ValueError, IndexError) as exc:
        return {
            "path": rel,
            "status": "re_extract_required",
            "reason": f"path_parse_error:{exc}",
        }

    try:
        legacy = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": rel,
            "status": "re_extract_required",
            "reason": f"json_error:{exc}",
        }

    tensor_sha_before = _sha256_file(st_path)

    already_v1 = (
        legacy.get("schema") == SIDECAR_SCHEMA
        and legacy.get("version") == SIDECAR_VERSION
    )
    if already_v1:
        ok = validate_concept_sidecar(
            legacy,
            expected_model_name=model_name,
            expected_checkpoint=checkpoint,
            expected_layer_idx=layer_idx,
            expected_d_model=EXPECTED_D_MODEL,
            expected_max_seq_len=DEFAULT_MAX_SEQ_LEN,
            expected_protocol=PROTOCOL,
            expected_use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
            expected_concept_sources=concept_sources,
        )
        needs_origin_stamp = legacy.get("provenance_origin") not in (
            "extraction",
            "migrated_v0_assumed_canonical_sources",
        )
        if ok and not needs_origin_stamp:
            return {
                "path": rel,
                "status": "verified",
                "tensor_sha256": tensor_sha_before,
            }
        if not ok and legacy.get("provenance_origin") == "extraction":
            return {
                "path": rel,
                "status": "re_extract_required",
                "reason": "v1_inconsistent",
            }
        # Fall through to rebuild metadata (still no tensor rewrite).

    try:
        v1 = build_v1_sidecar(
            legacy,
            model_name=model_name,
            checkpoint=checkpoint,
            layer_idx=layer_idx,
            concept_sources=concept_sources,
        )
    except ValueError as exc:
        return {
            "path": rel,
            "status": "re_extract_required",
            "reason": str(exc),
        }

    v1["tensor_sha256"] = tensor_sha_before
    if not validate_concept_sidecar(
        v1,
        expected_model_name=model_name,
        expected_checkpoint=checkpoint,
        expected_layer_idx=layer_idx,
        expected_d_model=EXPECTED_D_MODEL,
        expected_max_seq_len=DEFAULT_MAX_SEQ_LEN,
        expected_protocol=PROTOCOL,
        expected_use_chat_template=PRIMARY_USE_CHAT_TEMPLATE,
        expected_concept_sources=concept_sources,
    ):
        return {
            "path": rel,
            "status": "re_extract_required",
            "reason": "built_v1_failed_validation",
        }

    if dry_run:
        return {
            "path": rel,
            "status": "would_migrate",
            "tensor_sha256": tensor_sha_before,
        }

    _atomic_write_json(json_path, v1)
    tensor_sha_after = _sha256_file(st_path)
    if tensor_sha_after != tensor_sha_before:
        return {
            "path": rel,
            "status": "re_extract_required",
            "reason": "tensor_bytes_changed",
            "tensor_sha256_before": tensor_sha_before,
            "tensor_sha256_after": tensor_sha_after,
        }
    return {
        "path": rel,
        "status": "migrated",
        "tensor_sha256": tensor_sha_after,
        "provenance_origin": v1["provenance_origin"],
    }


def migrate_tree(
    root: Path,
    *,
    dry_run: bool = False,
    n_samples: int = N_SAMPLES,
) -> dict[str, Any]:
    cv_root = root / CONCEPT_VECTORS_SUBDIR
    if not cv_root.is_dir():
        raise FileNotFoundError(f"concept_vectors dir missing: {cv_root}")

    concept_sources = load_canonical_concept_sources(n_samples)
    rows: list[dict[str, Any]] = []
    for json_path in sorted(cv_root.glob("*/*/layer_*.json")):
        rows.append(migrate_one(json_path, concept_sources, dry_run=dry_run))

    counts = {
        "migrated": 0,
        "verified": 0,
        "would_migrate": 0,
        "re_extract_required": 0,
    }
    for row in rows:
        status = row["status"]
        if status in counts:
            counts[status] += 1
        else:
            counts["re_extract_required"] += 1

    return {
        "root": str(root),
        "dry_run": dry_run,
        "n_sidecars": len(rows),
        "counts": counts,
        "rows": rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default=RL_ZERO_CODE_RESULTS_ROOT,
        help="Experiment results root (default: %(default)s)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without writing",
    )
    p.add_argument(
        "--report",
        default="",
        help="Optional path for migration_report.json (default: <root>/migration_report.json)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help="Paired samples per concept (default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    report = migrate_tree(root, dry_run=bool(args.dry_run), n_samples=int(args.samples))
    report_path = Path(args.report) if args.report else root / "migration_report.json"
    _atomic_write_json(report_path, report)

    counts = report["counts"]
    print(f"root: {report['root']}")
    print(f"dry_run: {report['dry_run']}")
    print(f"n_sidecars: {report['n_sidecars']}")
    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"report: {report_path}")

    if counts.get("re_extract_required", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
