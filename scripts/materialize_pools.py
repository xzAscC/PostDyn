"""Materialize the four fixed domain prompt pools (data-collection step)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from postdyn.data import load_pool, materialize_pools

POOL_DOMAINS = ("math", "code", "instruction_following", "general_reasoning")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", default=str(ROOT / "configs/domain_sources.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data/domain_prompts"))
    parser.add_argument(
        "--n",
        type=int,
        default=2 * 3 * 5120,
        help="pool size per domain (default 30,720 = 2x max 3d; 7B consumes the "
        "deterministic 12,288 prefix; headroom enables robustness resampling)",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Dolci snapshot data directory (default: auto-resolve the local "
        "Hugging Face cache)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        materialize_pools(
            args.mapping, args.out_dir, n=args.n, snapshot_dir=args.snapshot_dir
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for domain in POOL_DOMAINS:
        pool = load_pool(Path(args.out_dir) / f"{domain}.json")
        print(
            f"{domain}: {pool.actual_n}/{pool.requested_n} records, "
            f"revision {pool.dataset_revision[:12]}, fingerprint {pool.fingerprint[:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
