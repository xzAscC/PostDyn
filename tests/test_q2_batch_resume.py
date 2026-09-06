from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from postdyn.persistence import close_all_jsonl_handles

ROOT = Path(__file__).parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_runtime(monkeypatch, tmp_path: Path, item_count: int = 5):
    exp2 = load_script("run_q2_exp2")
    output = tmp_path / "exp2"
    items = [
        SimpleNamespace(id=f"item-{index}", reference="reference")
        for index in range(item_count)
    ]
    calls: list[list[str]] = []

    monkeypatch.setattr(exp2.common, "output_root", lambda args, _: output)
    monkeypatch.setattr(exp2.common, "start_uploader", lambda *args: None)
    monkeypatch.setattr(exp2.common, "finish_uploader", lambda *args: None)
    monkeypatch.setattr(exp2.common, "write_identity_manifest", lambda *args: None)
    monkeypatch.setattr(exp2.common, "load_runtime", lambda *args: (object(), object()))
    monkeypatch.setattr(
        exp2.common,
        "load_items",
        lambda domain, limit, tiny: ([], items),
    )
    bases = {0: (torch.ones(8), torch.eye(8))}
    monkeypatch.setattr(exp2.common, "tiny_bases", lambda *args: None)
    monkeypatch.setattr(exp2.common, "require_bases", lambda *args: bases)
    monkeypatch.setattr(
        exp2,
        "sentence_final_states",
        lambda *args: torch.zeros((1, 8)),
    )
    monkeypatch.setattr(
        exp2,
        "item_subsims",
        lambda *args: (torch.ones(1), (0.25, 0.5), 1),
    )
    monkeypatch.setattr(exp2, "verify", lambda *args: True)

    from postdyn import bench

    def generate(model, tokenizer, batch, **kwargs):
        calls.append([item.id for item in batch])
        return [
            SimpleNamespace(text="answer", captured={}, prompt_token_len=0)
            for item in batch
        ]

    monkeypatch.setattr(bench, "generate", generate)
    args = exp2.parse_args(
        [
            "--family",
            "7b",
            "--scale",
            "tiny",
            "--q1-root",
            str(tmp_path / "q1"),
            "--domains",
            "math",
            "--limit",
            str(item_count),
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    return exp2, args, output, calls, items


def _write_done(output: Path, ids: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "solutions_math.jsonl"
    path.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item_id,
                    "correct": True,
                    "V_i": 1.0,
                    "subsim_high": 0.25,
                    "subsim_low": 0.5,
                    "n_sentences": 1,
                }
            )
            + "\n"
            for item_id in ids
        )
    )


def test_exp2_skips_completed_batches(monkeypatch, tmp_path: Path) -> None:
    exp2, args, output, calls, _ = _fake_runtime(monkeypatch, tmp_path)
    _write_done(output, ["item-0", "item-1"])

    exp2.run(args)

    assert calls == [["item-2", "item-3"], ["item-4"]]


def test_exp2_regenerates_partial_batch_whole(monkeypatch, tmp_path: Path) -> None:
    exp2, args, output, calls, _ = _fake_runtime(monkeypatch, tmp_path)
    _write_done(output, ["item-0"])

    exp2.run(args)

    assert calls == [["item-0", "item-1"], ["item-2", "item-3"], ["item-4"]]


def test_exp2_resume_appends_no_duplicates(monkeypatch, tmp_path: Path) -> None:
    exp2, args, output, calls, _ = _fake_runtime(monkeypatch, tmp_path)
    exp2.run(args)
    path = output / "solutions_math.jsonl"
    first_count = len(path.read_text().splitlines())

    exp2.run(args)

    assert len(path.read_text().splitlines()) == first_count
    assert calls == [["item-0", "item-1"], ["item-2", "item-3"], ["item-4"]]


def _exp3_args(tmp_path: Path):
    exp3 = load_script("run_q2_exp3")
    args = exp3.parse_args(
        [
            "--family",
            "7b",
            "--scale",
            "tiny",
            "--q1-root",
            str(tmp_path / "q1"),
            "--domains",
            "math",
            "--limit",
            "3",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "exp3"),
        ]
    )
    cfg = exp3.common.family_config(args.family, args.scale)
    exp3.common.tiny_bases(args.q1_root, args.domains, cfg.layers, (args.model, "rlvr"))
    return exp3, args


def test_exp3_passes_validation_done_ids_to_evaluate(
    monkeypatch, tmp_path: Path
) -> None:
    exp3, args = _exp3_args(tmp_path)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "domain": "math",
                    "layer": 0,
                    "alpha": 1.0,
                    "condition": "replace",
                    "item_id": item_id,
                }
            )
            + "\n"
            for item_id in ("0",)
        )
    )
    calls = []
    original = exp3.exp1._evaluate

    def spy(*call_args, **kwargs):
        calls.append((call_args, kwargs))
        return original(*call_args, **kwargs)

    monkeypatch.setattr(exp3.exp1, "_evaluate", spy)
    exp3.run(args)

    validation_calls = [
        (call_args, kwargs)
        for call_args, kwargs in calls
        if call_args[9] == "replace"
        and kwargs.get("replacement") is not None
        and call_args[2][0].id in {"0", "1", "2"}
    ]
    assert {
        call_args[4]: kwargs["done_ids"] for call_args, kwargs in validation_calls
    } == {
        0: {"0"},
        1: set(),
    }


def test_exp3_passes_condition_specific_done_ids(monkeypatch, tmp_path: Path) -> None:
    exp3, args = _exp3_args(tmp_path)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "eval_math_baseline.jsonl").write_text(
        json.dumps({"item_id": "30"}) + "\n"
    )
    calls = []
    original = exp3.exp1._evaluate

    def spy(*call_args, **kwargs):
        calls.append((call_args, kwargs))
        return original(*call_args, **kwargs)

    monkeypatch.setattr(exp3.exp1, "_evaluate", spy)
    exp3.run(args)

    test_calls = {
        "baseline": next(call for call in calls if call[0][9] == "baseline"),
        "own_only": next(call for call in calls if call[0][9] == "own_only"),
        "replace": [call for call in calls if call[0][9] == "replace"][-1],
    }
    assert test_calls["baseline"][1]["done_ids"] == {"30"}
    assert test_calls["own_only"][1]["done_ids"] == set()
    assert test_calls["replace"][1]["done_ids"] == set()


def test_exp3_resume_appends_no_duplicates(tmp_path: Path) -> None:
    exp3, args = _exp3_args(tmp_path)
    exp3.run(args)
    paths = [args.output / "validation.jsonl"] + [
        args.output / f"eval_math_{condition}.jsonl" for condition in exp3.CONDITIONS
    ]
    first_counts = {path: len(path.read_text().splitlines()) for path in paths}

    exp3.run(args)

    assert {path: len(path.read_text().splitlines()) for path in paths} == first_counts


def test_partial_validation_accuracy_uses_completed_rows_exp3(
    monkeypatch, tmp_path: Path
) -> None:
    exp3, args = _exp3_args(tmp_path)
    args.limit = 4
    validation_items = [
        SimpleNamespace(id=f"v{index}", reference={}) for index in range(4)
    ]
    monkeypatch.setattr(
        exp3.common,
        "load_items",
        lambda *_args: (validation_items, [SimpleNamespace(id="t0", reference={})]),
    )
    monkeypatch.setattr(
        exp3,
        "procrustes_align",
        lambda left, right: torch.eye(left.shape[1]),
    )
    monkeypatch.setattr(
        exp3.exp1,
        "verify",
        lambda _benchmark, text, _reference: text in {"v0", "v2"},
    )
    from postdyn import bench

    monkeypatch.setattr(
        bench,
        "generate",
        lambda _model, _tokenizer, batch, **_kwargs: [
            SimpleNamespace(item_id=item.id, text=item.id) for item in batch
        ],
    )
    exp3.run(args)
    path = args.output / "validation.jsonl"
    original_rows = [json.loads(line) for line in path.read_text().splitlines()]
    target = [row for row in original_rows if row["layer"] == 0]
    original_accuracy = target[0]["accuracy"]
    close_all_jsonl_handles()
    path.write_text(
        "".join(
            json.dumps(row) + "\n" for row in original_rows if row not in target[:2]
        )
    )
    calls = []
    original_evaluate = exp3.exp1._evaluate

    def spy(*call_args, **kwargs):
        calls.append((call_args, kwargs))
        return original_evaluate(*call_args, **kwargs)

    monkeypatch.setattr(exp3.exp1, "_evaluate", spy)
    exp3.run(args)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    target_rows = [row for row in rows if row["layer"] == 0]
    validation_calls = [call for call in calls if call[0][2] is validation_items]
    assert len(validation_calls) == 1
    assert {row["accuracy"] for row in target_rows} == {original_accuracy}
