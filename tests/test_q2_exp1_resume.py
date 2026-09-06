from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generation(item_id: str, text: str = "answer") -> SimpleNamespace:
    return SimpleNamespace(item_id=item_id, text=text)


def items(*pairs: tuple[str, object]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(id=item_id, prompt=item_id, reference=reference)
        for item_id, reference in pairs
    ]


def test_run_with_uses_injected_runtime_loader(tmp_path: Path, monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    args = tiny_args(tmp_path)
    loaded = []

    def injected_loader():
        loaded.append(True)
        return object(), object()

    monkeypatch.setattr(
        exp1.common, "load_runtime", lambda *_args: pytest.fail("common loader called")
    )
    patch_tiny_run(exp1, monkeypatch)

    exp1.run_with(args, injected_loader)

    assert len(loaded) == 1


def test_evaluate_skips_completed_batches(monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    calls = []
    monkeypatch.setattr(exp1, "verify", lambda *_args: True)
    monkeypatch.setattr(
        "postdyn.bench.generate",
        lambda _model, _tokenizer, batch, **_kwargs: (
            calls.append(list(batch)) or [generation(item.id) for item in batch]
        ),
    )

    result = exp1._evaluate(
        object(),
        object(),
        items(("a", {}), ("b", {}), ("c", {}), ("d", {})),
        "math500",
        0,
        1.0,
        None,
        2,
        1,
        "high",
        done_ids={"a", "b"},
    )

    assert [[item.id for item in batch] for batch in calls] == [["c", "d"]]
    assert [row["item_id"] for row in result] == ["c", "d"]


def test_evaluate_regenerates_partial_batch_whole(monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    calls = []
    monkeypatch.setattr(exp1, "verify", lambda *_args: True)
    monkeypatch.setattr(
        "postdyn.bench.generate",
        lambda _model, _tokenizer, batch, **_kwargs: (
            calls.append(list(batch)) or [generation(item.id) for item in batch]
        ),
    )

    exp1._evaluate(
        object(),
        object(),
        items(("a", {}), ("b", {}), ("c", {}), ("d", {})),
        "math500",
        0,
        1.0,
        None,
        2,
        1,
        "high",
        done_ids={"a"},
    )

    assert [[item.id for item in batch] for batch in calls] == [["a", "b"], ["c", "d"]]


def test_evaluate_reference_first_wins(monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    references = []
    monkeypatch.setattr(
        "postdyn.bench.generate",
        lambda _model, _tokenizer, batch, **_kwargs: [generation("same")],
    )
    monkeypatch.setattr(
        exp1,
        "verify",
        lambda _benchmark, _text, reference: references.append(reference) or True,
    )

    exp1._evaluate(
        object(),
        object(),
        items(("same", {"answer": "first"}), ("same", {"answer": "second"})),
        "math500",
        0,
        1.0,
        None,
        4,
        1,
        "high",
    )

    assert references == [{"answer": "first"}]


def test_summary_uses_captured_lengths(tmp_path: Path, monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    args = tiny_args(tmp_path)
    calls = []
    monkeypatch.setattr(
        exp1.common,
        "load_items",
        lambda *call_args: (
            calls.append(call_args)
            or (items(("v", {"answer": "1"})), items(("t", {"answer": "1"})))
        ),
    )
    patch_tiny_run(exp1, monkeypatch)
    monkeypatch.setattr(
        exp1.common, "load_runtime", lambda *_args: (object(), object())
    )

    exp1.run_with(args, lambda: (object(), object()))

    assert len(calls) == 1
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["math"]["n"] == 1


def test_partial_resume_appends_no_duplicates(tmp_path: Path, monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    args = tiny_args(tmp_path)
    patch_tiny_run(exp1, monkeypatch)

    exp1.run_with(args, lambda: (object(), object()))
    before = {
        path.name: len(path.read_text().splitlines())
        for path in args.output.glob("*.jsonl")
    }
    exp1.run_with(args, lambda: (object(), object()))
    after = {
        path.name: len(path.read_text().splitlines())
        for path in args.output.glob("*.jsonl")
    }

    assert after == before


def tiny_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        family="7b",
        scale="tiny",
        q1_root=tmp_path / "q1",
        output=tmp_path / "out",
        dtype="float32",
        quantization=None,
        device="cpu",
        batch_size=2,
        limit=1,
        model="rlvr",
        domains=["math"],
        upload_to=None,
    )


def patch_tiny_run(exp1, monkeypatch) -> None:
    monkeypatch.setattr(exp1.common, "tiny_bases", lambda *_args: None)
    monkeypatch.setattr(
        exp1,
        "require_bases",
        lambda *_args: {
            0: (torch.ones(8), torch.eye(8)),
            1: (torch.ones(8), torch.eye(8)),
        },
    )
    monkeypatch.setattr(
        exp1,
        "register_ablation_hook",
        lambda *_args: SimpleNamespace(remove=lambda: None),
    )
    monkeypatch.setattr(exp1, "mean_hidden_norm", lambda *_args: 1.0)
    monkeypatch.setattr(exp1, "verify", lambda *_args: True)
    monkeypatch.setattr(
        "postdyn.bench.generate",
        lambda _model, _tokenizer, batch, **_kwargs: [
            generation(item.id) for item in batch
        ],
    )
    monkeypatch.setattr(exp1.common, "start_uploader", lambda *_args: None)
    monkeypatch.setattr(exp1.common, "finish_uploader", lambda *_args: None)


def test_partial_validation_accuracy_uses_completed_rows(
    tmp_path: Path, monkeypatch
) -> None:
    exp1 = load_script("run_q2_exp1")
    args = tiny_args(tmp_path)
    args.limit = 4
    patch_tiny_run(exp1, monkeypatch)
    validation_items = items(
        ("v0", {"correct": True}),
        ("v1", {"correct": False}),
        ("v2", {"correct": True}),
        ("v3", {"correct": False}),
    )
    monkeypatch.setattr(
        exp1.common,
        "load_items",
        lambda *_args: (validation_items, items(("t0", {}))),
    )
    monkeypatch.setattr(
        exp1,
        "verify",
        lambda _benchmark, text, _reference: text in {"v0", "v2"},
    )
    monkeypatch.setattr(
        "postdyn.bench.generate",
        lambda _model, _tokenizer, batch, **_kwargs: [
            generation(item.id, item.id) for item in batch
        ],
    )

    exp1.run_with(args, lambda: (object(), object()))
    original_rows = [
        json.loads(line)
        for line in (args.output / "validation.jsonl").read_text().splitlines()
    ]
    original_accuracy = next(
        row["accuracy"]
        for row in original_rows
        if row["layer"] == 0 and row["alpha"] == 0.1 and row["condition"] == "high"
    )
    selected = (args.output / "selected.json").read_bytes()
    target = [
        row
        for row in original_rows
        if row["layer"] == 0 and row["alpha"] == 0.1 and row["condition"] == "high"
    ]
    remaining = [row for row in original_rows if row not in target[:2]]
    (args.output / "validation.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in remaining)
    )
    calls = []
    original_generate = exp1._evaluate

    def spy(*call_args, **kwargs):
        calls.append((call_args, kwargs))
        return original_generate(*call_args, **kwargs)

    monkeypatch.setattr(exp1, "_evaluate", spy)
    exp1.run_with(args, lambda: (object(), object()))

    validation_rows = [
        json.loads(line)
        for line in (args.output / "validation.jsonl").read_text().splitlines()
    ]
    target_rows = [
        row
        for row in validation_rows
        if row["layer"] == 0 and row["alpha"] == 0.1 and row["condition"] == "high"
    ]
    validation_calls = [call for call in calls if call[0][2] is validation_items]
    assert len(validation_calls) == 1
    assert validation_calls[0][0][4:6] == (0, 0.1)
    assert validation_calls[0][0][9] == "high"
    assert {row["accuracy"] for row in target_rows} == {original_accuracy}
    assert (args.output / "selected.json").read_bytes() == selected


def test_exp1_resume_guard_includes_sft_lr(tmp_path: Path, monkeypatch) -> None:
    exp1 = load_script("run_q2_exp1")
    args = tiny_args(tmp_path)
    args.sft_lr = "5e-5"
    args.output.mkdir(parents=True)
    manifest = exp1.identity_for(args)
    del manifest["sft_lr"]
    (args.output / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(SystemExit, match=r"resume identity mismatch.*sft_lr"):
        exp1.run_with(args, lambda: pytest.fail("runtime should not load"))
