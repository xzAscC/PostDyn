from __future__ import annotations

import base64
import json
import pickle
import zlib
from types import SimpleNamespace

import pytest
import torch

import postdyn.bench as bench


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []
        self.chat_template = "present"

    def apply_chat_template(self, **kwargs):
        self.calls.append(kwargs)
        return "templated"

    def __call__(self, prompts, **kwargs):
        rows = prompts if isinstance(prompts, list) else [prompts]
        if kwargs.get("return_tensors") == "pt":
            import torch as _torch

            ids = _torch.tensor([[len(p)] for p in rows])
            return {"input_ids": ids, "attention_mask": _torch.ones_like(ids)}
        return {
            "input_ids": [[len(p)] for p in rows],
            "attention_mask": [[1] for _ in rows],
        }

    def batch_decode(self, rows, **kwargs):
        return [f"decoded-{row[-1]}" for row in rows]


def test_apply_chat_template_uses_user_message_and_generation_prompt() -> None:
    tokenizer = FakeTokenizer()
    assert bench.apply_chat_template(tokenizer, "hi") == "templated"
    assert tokenizer.calls == [
        {
            "conversation": [{"role": "user", "content": "hi"}],
            "add_generation_prompt": True,
        }
    ]


def test_apply_chat_template_can_call_tokenizer_positional_variant() -> None:
    tokenizer = SimpleNamespace(chat_template="yes")
    tokenizer.apply_chat_template = lambda messages, add_generation_prompt: "ok"
    assert bench.apply_chat_template(tokenizer, "hi") == "ok"


def test_load_benchmark_is_deterministic_disjoint_and_clamped(monkeypatch) -> None:
    items = [{"id": str(i), "problem": f"p{i}", "answer": str(i)} for i in range(40)]
    monkeypatch.setattr(bench, "load_dataset", lambda *args, **kwargs: {"train": items})
    val, test = bench.load_benchmark("math500", n_val=30, seed=42)
    val_again, test_again = bench.load_benchmark("math500", n_val=30, seed=42)
    assert len(val) == 30
    assert {item.id for item in val}.isdisjoint(item.id for item in test)
    assert [item.id for item in val] == [item.id for item in val_again]
    assert [item.id for item in test] == [item.id for item in test_again]
    assert [item.id for item in test] == sorted(item.id for item in test)


def test_ifeval_loader_reads_real_jsonl_schema(tmp_path) -> None:
    path = tmp_path / "ifeval_input_data.jsonl"
    rows = [
        {
            "key": f"key-{i}",
            "prompt": f"Follow instruction {i}",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        }
        for i in range(40)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    val, test = bench.load_benchmark("ifeval", n_val=30, source=path)
    item = val[0]
    assert item.id.startswith("key-")
    assert item.reference == {
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }
    assert len(val) == 30
    assert {item.id for item in val}.isdisjoint(item.id for item in test)


def test_livecodebench_loader_reads_release_v6_jsonl_schema(tmp_path) -> None:
    path = tmp_path / "test6.jsonl"
    rows = [
        {
            "question_id": f"question-{i}",
            "question_content": f"Solve question {i}",
            "starter_code": "def solve():\n    pass" if i == 0 else "",
            "public_test_cases": json.dumps(
                [{"input": "1", "output": "1", "testtype": "stdin"}]
            ),
        }
        for i in range(40)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    val, test = bench.load_benchmark("livecodebench", n_val=30, source=path)
    item = next(item for item in val if item.id == "question-0")
    assert "Solve question 0" in item.prompt
    assert "Starter code:" in item.prompt
    assert "Write a Python solution." in item.prompt
    assert item.reference == {
        "cases": [{"input": "1", "output": "1", "testtype": "stdin"}],
        "func_name": None,
    }
    assert {item.id for item in val}.isdisjoint(item.id for item in test)


def test_livecodebench_loader_uses_function_prompt_without_stdin_instruction(
    tmp_path,
) -> None:
    path = tmp_path / "test6.jsonl"
    rows = [
        {
            "question_id": f"question-{i}",
            "question_content": "Return the answer.",
            "starter_code": "class Solution:\n    def solve(self, x):\n        pass"
            if i == 0
            else "",
            "public_test_cases": json.dumps(
                [
                    {
                        "input": "1",
                        "output": "1",
                        "testtype": "functional" if i == 0 else "stdin",
                    }
                ]
            ),
            "metadata": json.dumps({"func_name": "solve"}) if i == 0 else "{}",
        }
        for i in range(40)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    val, _ = bench.load_benchmark("livecodebench", n_val=40, source=path)
    functional = next(item for item in val if item.id == "question-0")
    stdin = next(item for item in val if item.id == "question-1")
    assert "Complete the following function" in functional.prompt
    assert "Read from stdin" not in functional.prompt
    assert "Read from stdin" in stdin.prompt


def test_livecodebench_loader_merges_public_then_private_cases(tmp_path) -> None:
    path = tmp_path / "test6.jsonl"
    public = [{"input": "1", "output": "1", "testtype": "stdin"}]
    private = [{"input": "2", "output": "2", "testtype": "functional"}]
    encoded_private = base64.b64encode(
        zlib.compress(pickle.dumps(json.dumps(private)))
    ).decode("ascii")
    rows = [
        {
            "question_id": f"question-{i}",
            "question_content": f"Solve question {i}",
            "starter_code": "",
            "public_test_cases": json.dumps(public),
            "private_test_cases": encoded_private,
            "metadata": json.dumps({"func_name": "solve"}),
        }
        for i in range(40)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    val, _ = bench.load_benchmark("livecodebench", n_val=30, source=path)

    item = next(item for item in val if item.id == "question-0")
    assert item.reference == {"cases": public + private, "func_name": "solve"}


def test_livecodebench_loader_raises_on_corrupt_private_cases(tmp_path) -> None:
    path = tmp_path / "test6.jsonl"
    public = [{"input": "1", "output": "1", "testtype": "stdin"}]
    rows = [
        {
            "question_id": f"question-{i}",
            "question_content": f"Solve question {i}",
            "starter_code": "",
            "public_test_cases": json.dumps(public),
            "private_test_cases": "not-valid-zlib-data",
        }
        for i in range(40)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="question-0"):
        bench.load_benchmark("livecodebench", n_val=30, source=path)


def test_mmlu_pro_loader_reads_test_parquet_and_formats_all_options(tmp_path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "test.parquet"
    rows = [
        {
            "question_id": str(i),
            "question": f"Question {i}",
            "options": list("ABCDEFGHIJ"),
            "answer": "J",
        }
        for i in range(40)
    ]
    parquet.write_table(pyarrow.Table.from_pylist(rows), path)
    val, test = bench.load_benchmark("mmlu_pro", n_val=30, source=path)
    item = val[0]
    assert "A. A" in item.prompt and "J. J" in item.prompt
    assert item.prompt.endswith("Answer with a single letter (A-J).")
    assert item.reference == {"answer": "J"}
    assert {item.id for item in val}.isdisjoint(item.id for item in test)


def test_load_benchmark_adapts_math_shape_and_preserves_split(monkeypatch) -> None:
    rows = [{"id": str(i), "problem": f"p{i}", "answer": "1"} for i in range(40)]
    monkeypatch.setattr(bench, "load_dataset", lambda *args, **kwargs: {"train": rows})
    val, test = bench.load_benchmark("math500", n_val=30)
    assert len(val) == 30
    assert {item.id for item in val}.isdisjoint(item.id for item in test)


def test_generate_is_greedy_batches_and_returns_decoded_generations() -> None:
    tokenizer = FakeTokenizer()
    calls = []

    class Model:
        def generate(self, input_ids, attention_mask, **kwargs):
            calls.append((input_ids, kwargs))
            return [[*row, 99] for row in input_ids]

    items = [bench.BenchItem(str(i), f"p{i}", {}) for i in range(3)]
    generations = bench.generate(
        Model(), tokenizer, items, chat_template=True, batch_size=2
    )
    assert len(calls) == 2
    assert all(call[1]["do_sample"] is False for call in calls)
    assert len(generations) == 3
    assert [generation.item_id for generation in generations] == ["0", "1", "2"]
    assert all(generation.text.startswith("decoded-") for generation in generations)


def test_generate_moves_all_inputs_through_model_device_helper(monkeypatch) -> None:
    tokenizer = FakeTokenizer()
    seen = []

    def ensure_device(inputs, device):
        seen.append((inputs, device))
        return inputs

    monkeypatch.setattr(bench, "_ensure_device", ensure_device)

    class Model:
        def parameters(self):
            return iter(())

        def generate(self, input_ids, attention_mask, **kwargs):
            return [[*row, 99] for row in input_ids]

    bench.generate(Model(), tokenizer, [bench.BenchItem("0", "p", {})])
    assert len(seen) == 1
    assert set(seen[0][0]) == {"input_ids", "attention_mask"}


def test_generate_capture_uses_block_output_layer_index(monkeypatch) -> None:
    tokenizer = FakeTokenizer()

    class Model:
        def parameters(self):
            return iter(())

        def generate(self, input_ids, attention_mask, **kwargs):
            return torch.tensor([[*row, 99] for row in input_ids])

        def __call__(self, **kwargs):
            return SimpleNamespace(
                hidden_states=(
                    torch.tensor([[0.0]]),
                    torch.tensor([[1.0]]),
                    torch.tensor([[2.0]]),
                )
            )

    captures = bench.generate(
        Model(), tokenizer, [bench.BenchItem("0", "p", {})], capture_layers=[1]
    )
    assert captures[0].captured is not None
    assert captures[0].captured[1].item() == 2.0
