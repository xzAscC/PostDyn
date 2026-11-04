"""Build one replacement model: python scripts/make_passk_hybrid.py TRAINED BASE OUT CONDITION."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from postdyn import hybrid  # noqa: E402

if __name__ == "__main__":
    trained, base, out, condition = sys.argv[1:]
    hybrid.build(Path(trained), Path(base), Path(out), condition)
    print(f"built {out}", flush=True)
