from __future__ import annotations

import pytest
import torch

from postdyn import spectrum
from scripts import run_spectrum


def test_complete_texts_drop_truncated_answers():
    row = {
        "samples": [
            {"text": "a", "finish_reason": "stop"},
            {"text": "b", "finish_reason": "length"},
            {"text": "c", "finish_reason": "stop"},
            {"text": "d", "finish_reason": "stop"},
        ]
    }
    assert run_spectrum.complete_texts(row, 2) == ["a", "c"]


def test_common_ids_keep_the_fixed_order_and_need_every_run():
    rows = {"base": {"a": {}, "b": {}, "c": {}}, "post": {"c": {}, "a": {}}}
    assert run_spectrum.common_ids(["c", "b", "a"], rows, 5) == ["c", "a"]
    assert run_spectrum.common_ids(["c", "b", "a"], rows, 1) == ["c"]


def test_prefix_batches_weigh_the_rows_that_pick_each_prefix_token():
    # The second answer is shorter than the prefix, so all of it is used.
    answers = [list(range(10, 30)), [40, 41]]
    (ids, mask, weights), = run_spectrum.prefix_batches([1, 2], answers, 4, 8, 0)
    assert ids.shape == (2, 6)
    assert ids[0].tolist() == [1, 2, 10, 11, 12, 13]
    assert ids[1].tolist() == [1, 2, 40, 41, 0, 0]
    assert mask[1].tolist() == [1, 1, 1, 1, 0, 0]
    # Rows 1..4 pick tokens 10..13: the last prompt row is in, the last prefix row is out.
    assert (weights[0] > 0).tolist() == [False, True, True, True, True, False]
    assert (weights[1] > 0).tolist() == [False, True, True, False, False, False]
    assert weights.sum(1).tolist() == [1.0, 1.0]


def test_prefix_batches_split_and_skip_empty_answers():
    batches = list(run_spectrum.prefix_batches([1], [[5], [], [6], [7]], 100, 2, 0))
    assert [len(ids) for ids, _, _ in batches] == [1, 2]


def test_select_ids_use_the_rate_over_all_samples():
    base = {"a": ["wrong"] * 3 + ["correct"], "b": ["correct"] * 4, "c": ["wrong"] * 4}
    post = {"a": ["correct"] * 4, "b": ["correct"] * 4, "c": ["wrong"] * 4, "d": ["correct"] * 4}
    assert run_spectrum.select_ids(base, post, 0.25, 0.75, {"a", "b", "c", "d"}) == ["a"]
    # A problem without moments cannot be ablated, so it is never selected.
    assert run_spectrum.select_ids(base, post, 0.25, 0.75, {"b", "c"}) == []


def test_keep_knobs_refuses_to_resume_under_other_knobs(tmp_path):
    run_spectrum.keep_knobs(tmp_path, {"k": {0: 32}, "prefix_tokens": 100})
    run_spectrum.keep_knobs(tmp_path, {"k": {0: 32}, "prefix_tokens": 100})
    with pytest.raises(ValueError, match="other knobs"):
        run_spectrum.keep_knobs(tmp_path, {"k": {0: 128}, "prefix_tokens": 100})


def test_one_k_skips_layers_with_a_smaller_stable_k_and_keeps_the_range():
    rows = [{"layer": layer, "k": k} for layer, k in enumerate((64, 0, 32, 16, 32))]
    assert run_spectrum.steered_layers(rows, None, None) == {0: 64, 1: 0, 2: 32, 3: 16, 4: 32}
    assert run_spectrum.steered_layers(rows, 32, None) == {0: 32, 2: 32, 4: 32}
    assert run_spectrum.steered_layers(rows, 32, run_spectrum.layer_range("1-3")) == {2: 32}


def test_a_map_is_fit_without_the_fold_it_is_used_on():
    folds = [spectrum.FoldMoments({0: 1}, 2), spectrum.FoldMoments({0: 1}, 2)]
    for held in folds:
        for problem, fold, gap in (("a", 0, 1.0), ("b", 1, 3.0)):
            running = spectrum.ProblemMoments({0: 1}, torch.device("cpu"))
            running.update({0: torch.tensor([[-gap], [gap]])}, torch.ones(2))
            held.add(problem, fold, *running.finish())
    maps = run_spectrum.maps_for("mean", (folds[0], folds[1]), 1, {0: 1}, 0)
    assert maps[0].shift.item() == 0.0
    assert folds[0].stats(0, leave_out=1).sigma.item() == 1.0
    assert folds[0].stats(0, only=(1,)).sigma.item() == 9.0
