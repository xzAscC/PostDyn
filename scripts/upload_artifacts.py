"""Upload (and optionally prune) local experiment artifacts to the Hub."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from postdyn import uploader


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=None,
        help="target dataset repo id (default: POSTDYN_UPLOAD_TO env)",
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["data", "logs"],
        help="local directories/files to upload (repo layout mirrors these)",
    )
    parser.add_argument(
        "--prune-uploaded",
        action="store_true",
        help="after uploading, delete local intermediate eigensystems that "
        "are confirmed uploaded and whose family analysis is complete "
        "(finals/manifests/analysis/pools are kept)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo or uploader.os.environ.get("POSTDYN_UPLOAD_TO")
    if not repo:
        print("error: --repo or POSTDYN_UPLOAD_TO is required", file=sys.stderr)
        return 2
    handle = uploader.ArtifactUploader(
        repo, state_path=ROOT / ".upload_state.json"
    )
    handle.start()
    for item in args.paths:
        handle.submit_tree(ROOT / item, relative_to=ROOT)
    summary = handle.finish()
    print(
        f"upload complete: {summary['uploaded']} uploaded, "
        f"{summary['skipped']} skipped, {summary['failed']} failed"
    )
    for failure in summary["failures"]:
        print(f"  failed: {failure}")
    if args.prune_uploaded:
        state = json.loads((ROOT / ".upload_state.json").read_text())
        for local in uploader.determine_prunable(state, ROOT):
            local.unlink()
            print(f"pruned: {local.relative_to(ROOT)}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
