from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_checkpoints import checkpoint_list  # noqa: E402
from scripts.make_weight_runs import RUNS, pick_steps  # noqa: E402


def test_pick_steps_matches_whole_name_and_drops_final() -> None:
    names = ["Math-7B-Instruct-step31-deepscaler", "DAPO_Math-7B-Instruct-step16-deepscaler",
             "Math-7B-Instruct-step620-deepscaler", "Math-7B-Instruct-step62-deepscaler"]
    picked = pick_steps(names, r"Math-7B-Instruct-step(\d+)-deepscaler", final="Math-7B-Instruct-step620-deepscaler")
    assert picked == [(31, "Math-7B-Instruct-step31-deepscaler"), (62, "Math-7B-Instruct-step62-deepscaler")]


def test_pick_steps_keeps_ten_evenly_spaced() -> None:
    names = [f"step{s}" for s in range(1000, 44000, 1000)]
    picked = pick_steps(names, r"step(\d+)")
    assert len(picked) == 10 and picked[0][0] == 1000 and picked[-1][0] == 43000


def test_runs_cover_both_rows_of_every_family() -> None:
    rows = {(r["family"], r["row"]) for r in RUNS}
    assert {f for f, _ in rows} == {"olmo3-7b", "olmo3-32b", "qwen-math-7b-scaleml", "qwen-math-7b-cutelemonlili"}
    assert all((f, row) in rows for f, _ in rows for row in ("SFT", "RL"))


def test_checkpoint_list_puts_starts_before_ends_without_duplicates() -> None:
    runs = [{"start": {"repo": "a", "revision": "main"}, "final": {"repo": "b", "revision": "main"},
             "intermediates": [{"repo": "b", "revision": "s1", "step": 1}]},
            {"start": {"repo": "a", "revision": "main"}, "final": {"repo": "c", "revision": "main"},
             "intermediates": []}]
    assert checkpoint_list(runs, "intermediate") == [("a", "main"), ("b", "s1")]
    assert checkpoint_list(runs, "final") == [("a", "main"), ("b", "main"), ("c", "main")]
