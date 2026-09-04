from __future__ import annotations

from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("name", ["math500", "mmlu_pro", "ifeval", "livecodebench"])
def test_load_benchmark_adapts_all_frozen_dataset_shapes(
    monkeypatch, name: str
) -> None:
    if name == "math500":
        rows = [{"id": str(i), "problem": f"p{i}", "answer": "1"} for i in range(40)]
    elif name == "mmlu_pro":
        rows = [
            {
                "id": str(i),
                "question": f"q{i}",
                "options": list("ABCDEFGHIJ"),
                "answer": "A",
            }
            for i in range(40)
        ]
    elif name == "ifeval":
        rows = [
            {"id": str(i), "prompt": f"p{i}", "instr": [], "kwargs": []}
            for i in range(40)
        ]
    else:
        rows = [
            {"id": str(i), "prompt": f"p{i}", "code": "", "test-cases": []}
            for i in range(40)
        ]
    monkeypatch.setattr(bench, "load_dataset", lambda *args, **kwargs: {"train": rows})
    val, test = bench.load_benchmark(name, n_val=30)
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
