from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import types
from argparse import Namespace

import numpy as np
import pytest

from postdyn import passk
from postdyn.bench import BenchItem
from scripts import run_passk


def test_pass_at_k_matches_the_definition():
    assert passk.pass_at_k(0, 64, 64) == 0.0
    assert passk.pass_at_k(1, 64, 64) == 1.0
    assert passk.pass_at_k(16, 64, 1) == pytest.approx(0.25)
    # One correct in 4 samples, draw 2: the correct one is drawn half of the time.
    assert passk.pass_at_k(1, 4, 2) == pytest.approx(0.5)


def test_pass_at_k_is_exact_for_every_subset():
    from itertools import combinations

    flags = [1, 0, 0, 1, 0, 0]
    subsets = list(combinations(flags, 3))
    assert passk.pass_at_k(2, 6, 3) == pytest.approx(sum(map(any, subsets)) / len(subsets))


def test_pass_at_k_rejects_k_above_the_sample_count():
    with pytest.raises(ValueError):
        passk.pass_at_k(1, 8, 16)


def test_final_response_drops_the_thinking():
    assert passk.final_response("plan \\boxed{1}</think>The answer is \\boxed{2}") == (
        "The answer is \\boxed{2}"
    )
    assert passk.final_response("no thinking \\boxed{2}") == "no thinking \\boxed{2}"


def test_chat_response_without_closed_thinking_has_no_answer():
    # The chat prompt ends in <think>, so an unclosed response never left the thinking.
    sample = {"text": "maybe \\boxed{5}", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "5"}, "chat") == "no_answer"
    assert passk.classify("math", sample, {"answer": "5"}, "simple") == "correct"


def test_extract_answer_takes_the_last_boxed_with_nested_braces():
    assert passk.extract_answer("\\boxed{1} then \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"


def test_extract_answer_reads_the_answer_line():
    assert passk.extract_answer("Steps\nAnswer: $42$\n") == "42"


def test_extract_answer_ignores_answers_in_running_text():
    # The old verifier fell back on any "answer = x" in the text, a false positive source
    # that pass@64 multiplies.
    assert passk.extract_answer("so the answer = 7 might work, or not") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Answer: 5**", "5"),
        ("Answer: **10**", "10"),
        ("Answer: $5$ dollars", "5"),
        ("\\boxed{3}\nmore work\nAnswer: 5", "5"),
        ("Answer: 3\nchecking\n\\boxed{5}", "5"),
        ("Answer: \\boxed{5}", "5"),
        ("put it in \\boxed{}\nAnswer: 5", "5"),
        ("Answer:\n5", "5"),
        ("Answer:\n$5$\n", "5"),
        ("**Answer**: 5", "5"),
        ("### Answer: 5", "5"),
        ("answer: 5", "5"),
        ("Answers: 2", "2"),
        ("**Final answer: 2.", "2"),
        ("Answer: 5 people", "5"),
        ("Answer: 18.90 dollars", "18.90"),
        ("Answer: 90°", "90^\\circ"),
        ("Answer: $\\$36$", "\\$36"),
        ("Answer: $-2$, $1$", "-2, 1"),
        ("\\boxed{5}\n\nAnswer: The final answer is 5.", "5"),
        ("\\boxed{5}\nAnswer: see the box above", "5"),
    ],
)
def test_extract_answer_handles_common_forms(text, expected):
    assert passk.extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Answer:\n\n[",
        "Answer:\n\nLet me think again",
        "Answer:\n3\nand then some more\n",
        "\\boxed{}",
        "\\boxed{  }",
        "\\boxed{3}\nredo\n\\boxed{5",
        "So,answer: 5",
    ],
)
def test_extract_answer_finds_nothing_in_empty_forms(text):
    assert passk.extract_answer(text) is None


@pytest.mark.parametrize("prediction", ["5} but really 6", "3}", "{5"])
def test_unbalanced_answers_never_verify(prediction):
    assert not passk._math_correct(prediction, prediction.strip("{}").split("}")[0])


@pytest.mark.parametrize(
    ("text", "gold", "outcome"),
    [
        ("Answer: $3$ or $5$", "3", "wrong"),
        ("Answer: $x=3$ or $x=5$", "3", "wrong"),
        ("Answer: 5 or 6", "5", "wrong"),
        ("Answer: $-2$, $1$", "-2,1", "correct"),
        ("Answer: $\\$36$", "\\$36", "correct"),
        ("Answer: 5 people", "5", "correct"),
        ("Answer: 18.90 dollars", "\\$18.90", "correct"),
        ("Answer: 90°", "90^\\circ", "correct"),
    ],
)
def test_answer_lines_score_like_boxes(text, gold, outcome):
    sample = {"text": text, "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": gold}, "simple") == outcome


def test_last_code_block_takes_the_final_block():
    text = "```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
    assert passk.last_code_block(text) == "print(2)\n"
    assert passk.last_code_block("no block") is None


def test_classify_math_counts_a_truncated_answer_as_truncated():
    sample = {"text": "\\boxed{5}", "finish_reason": "length"}
    assert passk.classify("math", sample, {"answer": "5"}, "simple") == "truncated"


def test_classify_math_reads_only_the_final_response():
    sample = {"text": "\\boxed{5}</think>So \\boxed{6}", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "5"}, "simple") == "wrong"
    assert passk.classify("math", sample, {"answer": "6"}, "simple") == "correct"


def test_classify_math_without_a_final_answer():
    sample = {"text": "I give up.", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "5"}, "simple") == "no_answer"


def test_classify_math_accepts_equal_forms():
    sample = {"text": "\\boxed{0.5}", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "\\frac{1}{2}"}, "simple") == "correct"


def test_classify_code_runs_the_last_block(tmp_path):
    reference = {"cases": [{"input": "", "output": "2\n"}], "func_name": None}
    good = {"text": "```python\nprint(1)\n```\n```python\nprint(2)\n```", "finish_reason": "stop"}
    bad = {"text": "```python\nprint(2)\n```\n```python\nprint(1)\n```", "finish_reason": "stop"}
    assert passk.classify("code", good, reference, "simple") == "correct"
    assert passk.classify("code", bad, reference, "simple") == "wrong"
    assert passk.classify("code", {"text": "none", "finish_reason": "stop"}, reference, "simple") == "no_answer"


def test_classify_code_gets_the_official_imports():
    reference = {
        "cases": [{"testtype": "functional", "input": "[1, 2]", "output": "3"}],
        "func_name": "add",
    }
    text = (
        "```python\nclass Solution:\n"
        "    def add(self, nums: List[int]) -> int:\n        return sum(nums)\n```"
    )
    sample = {"text": text, "finish_reason": "stop"}
    assert passk.classify("code", sample, reference, "simple") == "correct"


def test_sandbox_runs_threads_and_numpy():
    reference = {"cases": [{"input": "", "output": "6\n"}], "func_name": None}
    code = (
        "import threading\nimport numpy as np\n"
        "threading.stack_size(1 << 26)\n"
        "thread = threading.Thread(target=lambda: print(int(np.arange(4).sum())))\n"
        "thread.start()\nthread.join()\n"
    )
    sample = {"text": f"```python\n{code}```", "finish_reason": "stop"}
    assert passk.classify("code", sample, reference, "simple") == "correct"


def _rows(counts, n=4):
    rows = []
    for i, correct in enumerate(counts):
        rows.append(
            {
                "id": str(i),
                "outcomes": ["correct"] * correct + ["wrong"] * (n - correct),
                "tokens": [10] * n,
            }
        )
    return rows


def test_task_summary_reports_each_k_and_the_outcomes():
    summary = passk.task_summary(_rows([4, 2, 0, 0]), ks=(1, 4))
    assert summary["pass@1"]["value"] == pytest.approx(1.5 / 4)
    assert summary["pass@4"]["value"] == pytest.approx(0.5)
    assert summary["outcomes"] == {"correct": pytest.approx(0.375), "wrong": pytest.approx(0.625)}
    assert summary["n_problems"] == 4
    assert summary["n_samples"] == 4


def test_task_summary_skips_k_above_the_sample_count():
    assert "pass@64" not in passk.task_summary(_rows([1, 2]), ks=(1, 64))


def test_task_summary_rejects_unequal_sample_counts():
    rows = _rows([1, 2])
    rows[0]["outcomes"].pop()
    with pytest.raises(ValueError):
        passk.task_summary(rows, ks=(1,))


def test_compare_pairs_problems_by_id():
    a = _rows([0, 0, 4, 4])
    b = list(reversed(_rows([4, 4, 4, 4])))
    result = passk.compare(a, b, ks=(1,))
    assert result["pass@1"]["difference"] == pytest.approx(0.5)
    assert result["pass@1"]["low"] <= 0.5 <= result["pass@1"]["high"]


def test_compare_rejects_different_problem_sets():
    with pytest.raises(ValueError):
        passk.compare(_rows([1, 2]), _rows([1, 2, 3]), ks=(1,))


def test_bootstrap_interval_is_reproducible():
    values = np.arange(20, dtype=float)
    assert passk.bootstrap_interval(values) == passk.bootstrap_interval(values)


def test_prompts_keep_their_format_rules():
    assert passk.user_text("math", "1+1?", "chat").endswith("\\boxed{}.")
    simple = passk.user_text("math", "1+1?", "simple")
    assert simple.startswith("Solve the following math problem step by step.")
    assert "1+1?" in simple and simple.endswith('after "Answer:"')
    assert "```python" in passk.user_text("code", "Add.", "chat")
    with pytest.raises(ValueError):
        passk.user_text("math", "1+1?", "other")


class _FakeOutput:
    def __init__(self, text, finish_reason):
        self.text = text
        self.finish_reason = finish_reason
        self.token_ids = [1, 2, 3]


class _FakeRequest:
    def __init__(self, n):
        self.outputs = [_FakeOutput(f"sample {i}</think>\\boxed{{5}}", "stop") for i in range(n)]


class _FakeLLM:
    calls: list[int] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_tokenizer(self):
        return types.SimpleNamespace(
            chat_template="template",
            apply_chat_template=lambda conversation, tokenize, add_generation_prompt: (
                f"<chat>{conversation[0]['content']}"
            ),
            convert_tokens_to_ids=lambda token: 7,
        )

    def generate(self, prompts, params):
        _FakeLLM.calls.append(len(prompts))
        return [_FakeRequest(params.n) for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    module = types.SimpleNamespace(LLM=_FakeLLM, SamplingParams=lambda **kwargs: Namespace(**kwargs))
    monkeypatch.setitem(sys.modules, "vllm", module)
    _FakeLLM.calls = []


def _args(tmp_path, **overrides):
    values = dict(
        model="m",
        tokenizer="t",
        prompt_format="chat",
        tasks=["math500"],
        samples=2,
        temperature=1.0,
        top_p=1.0,
        max_tokens=16,
        chunk_problems=2,
        limit=0,
        seed=0,
        lcb_source=None,
        gpu_memory_utilization=0.9,
        max_num_seqs=8,
        out_dir=str(tmp_path),
    )
    return Namespace(**(values | overrides))


def _items(monkeypatch, n=5):
    items = [BenchItem(str(i), f"q{i}", {"answer": "5"}) for i in range(n)]
    monkeypatch.setattr(run_passk, "load_items", lambda task, limit, source: items)


def test_generate_writes_chunks_and_resumes(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path))
    chunks = sorted((tmp_path / "math500").glob("chunk_*.jsonl.gz"))
    assert len(chunks) == 3
    assert _FakeLLM.calls == [2, 2, 1]
    with gzip.open(chunks[0], "rt") as f:
        row = json.loads(f.readline())
    assert row["id"] == "0" and len(row["samples"]) == 2
    assert row["samples"][0] == {
        "text": "sample 0</think>\\boxed{5}",
        "finish_reason": "stop",
        "tokens": 3,
    }

    chunks[1].unlink()
    _FakeLLM.calls = []
    run_passk.generate(_args(tmp_path))
    assert _FakeLLM.calls == [2]


def test_generate_stops_when_settings_change(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path))
    with pytest.raises(ValueError):
        run_passk.generate(_args(tmp_path, temperature=0.6))


@pytest.mark.parametrize(
    "change", [{"chunk_problems": 3}, {"tokenizer": "other"}, {"lcb_source": "other"}]
)
def test_generate_stops_when_chunking_or_inputs_change(tmp_path, monkeypatch, fake_vllm, change):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path))
    with pytest.raises(ValueError):
        run_passk.generate(_args(tmp_path, **change))


def test_score_refuses_an_unfinished_run(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path))
    next(iter(sorted((tmp_path / "math500").glob("chunk_*.jsonl.gz")))).unlink()
    with pytest.raises(ValueError, match="unfinished"):
        run_passk.score(Namespace(out_dir=str(tmp_path), workers=1, ks=[1]))


def test_score_reads_the_generate_time_source(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path, lcb_source="the-source"))
    seen = []
    items = run_passk.load_items("math500", 0, None)
    monkeypatch.setattr(
        run_passk, "load_items", lambda task, limit, source: seen.append(source) or items
    )
    run_passk.score(Namespace(out_dir=str(tmp_path), workers=1, ks=[1]))
    assert seen == ["the-source"]


def test_generate_uses_the_prompt_format(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch, n=1)
    seen = []
    monkeypatch.setattr(
        _FakeLLM,
        "generate",
        lambda self, prompts, params: seen.extend(prompts) or [_FakeRequest(params.n) for _ in prompts],
    )
    run_passk.generate(_args(tmp_path))
    assert seen[0].startswith("<chat>q0")
    seen.clear()
    run_passk.generate(_args(tmp_path / "simple", prompt_format="simple"))
    assert seen[0].startswith("Solve the following math problem")


def test_score_and_compare_end_to_end(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    a, b = tmp_path / "a", tmp_path / "b"
    run_passk.generate(_args(a))
    run_passk.generate(_args(b))
    run_passk.score(Namespace(out_dir=str(a), workers=1, ks=[1, 2]))
    run_passk.score(Namespace(out_dir=str(b), workers=1, ks=[1, 2]))
    summary = json.loads((a / "summary.json").read_text())
    assert summary["math500"]["pass@2"]["value"] == 1.0

    out = tmp_path / "compare.json"
    run_passk.compare_runs(Namespace(a=str(a), b=str(b), ks=[1, 2], out=str(out)))
    assert json.loads(out.read_text())["math500"]["pass@1"]["difference"] == 0.0


def _scored_run(path, monkeypatch, **settings):
    _items(monkeypatch)
    run_passk.generate(_args(path, **settings))
    run_passk.score(Namespace(out_dir=str(path), workers=1, ks=[1]))


def test_compare_refuses_different_sampling(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _scored_run(tmp_path / "b", monkeypatch, temperature=0.6)
    with pytest.raises(ValueError, match="temperature"):
        run_passk.compare_runs(
            Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"))
        )


def test_compare_refuses_different_test_cases(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch, lcb_source="v5")
    _scored_run(tmp_path / "b", monkeypatch, lcb_source="v6")
    with pytest.raises(ValueError, match="lcb_source"):
        run_passk.compare_runs(
            Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"))
        )


def test_compare_allows_different_models_and_prompts(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _scored_run(tmp_path / "b", monkeypatch, model="other", prompt_format="simple")
    run_passk.compare_runs(
        Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"))
    )


def test_compare_refuses_different_tasks(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _scored_run(tmp_path / "b", monkeypatch)
    (tmp_path / "b" / "scores" / "math500.jsonl").rename(tmp_path / "b" / "scores" / "other.jsonl")
    with pytest.raises(ValueError, match="tasks"):
        run_passk.compare_runs(
            Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"))
        )


def test_job_script_has_valid_syntax():
    subprocess.run(["bash", "-n", "scripts/run_passk_osc.sh"], check=True)


def test_extract_answer_keeps_a_trailing_period_out_of_the_comparison():
    sample = {"text": "Steps\nAnswer: 12.\n", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "12"}, "simple") == "correct"


def test_binomial_helper_stays_exact_for_64_samples():
    assert passk.pass_at_k(2, 64, 16) == pytest.approx(
        1 - math.comb(62, 16) / math.comb(64, 16)
    )
