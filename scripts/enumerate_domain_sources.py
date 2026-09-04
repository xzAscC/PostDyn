"""Enumerate dataset sources in a cached Dolci snapshot."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

enumerate_sources = import_module("postdyn.data").enumerate_sources


def main() -> None:
    """Parse CLI arguments and print or write source counts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(enumerate_sources(args.snapshot_dir), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
