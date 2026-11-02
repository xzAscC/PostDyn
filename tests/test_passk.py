from __future__ import annotations

import gzip
import json
import math
import os
import subprocess
import sys
import time
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
    assert passk.classify("math", sample, {"answer": "5"}, "plain") == "correct"


@pytest.mark.parametrize(
    ("text", "gold", "outcome"),
    [
        ("Thus \\boxed{5}.", "5", "correct"),
        ("\\boxed{\\frac{1}{2}}", "0.5", "correct"),
        ("\\boxed{3}\nredo\n\\boxed{5}", "5", "correct"),
        ("\\boxed{3}\nredo\n\\boxed{5}", "3", "wrong"),
        ("\\boxed{5 \\text{ people}}", "5", "correct"),
        ("\\boxed{\\$18.90}", "\\$18.90", "correct"),
        ("\\boxed{90^\\circ}", "90^\\circ", "correct"),
        ("\\boxed{-2,1}", "-2,1", "correct"),
        ("so the answer = 7 might work", "7", "no_answer"),
        ("Answer: 7", "7", "no_answer"),
        ("\\boxed{}", "7", "no_answer"),
        ("\\boxed{7}\nredo\n\\boxed{5", "7", "no_answer"),
    ],
)
def test_only_the_last_closed_box_counts(text, gold, outcome):
    sample = {"text": text, "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": gold}, "plain") == outcome


def test_last_code_block_takes_the_final_block():
    text = "```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
    assert passk.last_code_block(text) == "print(2)\n"
    assert passk.last_code_block("no block") is None


def test_classify_math_counts_a_truncated_answer_as_truncated():
    sample = {"text": "\\boxed{5}", "finish_reason": "length"}
    assert passk.classify("math", sample, {"answer": "5"}, "plain") == "truncated"


def test_classify_math_reads_only_the_final_response():
    sample = {"text": "\\boxed{5}</think>So \\boxed{6}", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "5"}, "plain") == "wrong"
    assert passk.classify("math", sample, {"answer": "6"}, "plain") == "correct"


def test_classify_math_without_a_final_answer():
    sample = {"text": "I give up.", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "5"}, "plain") == "no_answer"


def test_classify_math_accepts_equal_forms():
    sample = {"text": "\\boxed{0.5}", "finish_reason": "stop"}
    assert passk.classify("math", sample, {"answer": "\\frac{1}{2}"}, "plain") == "correct"


def test_classify_code_runs_the_last_block(tmp_path):
    reference = {"cases": [{"input": "", "output": "2\n"}], "func_name": None}
    good = {"text": "```python\nprint(1)\n```\n```python\nprint(2)\n```", "finish_reason": "stop"}
    bad = {"text": "```python\nprint(2)\n```\n```python\nprint(1)\n```", "finish_reason": "stop"}
    assert passk.classify("code", good, reference, "plain") == "correct"
    assert passk.classify("code", bad, reference, "plain") == "wrong"
    assert passk.classify("code", {"text": "none", "finish_reason": "stop"}, reference, "plain") == "no_answer"


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
    assert passk.classify("code", sample, reference, "plain") == "correct"


def test_sandbox_runs_threads_and_numpy():
    reference = {"cases": [{"input": "", "output": "6\n"}], "func_name": None}
    code = (
        "import threading\nimport numpy as np\n"
        "threading.stack_size(1 << 26)\n"
        "thread = threading.Thread(target=lambda: print(int(np.arange(4).sum())))\n"
        "thread.start()\nthread.join()\n"
    )
    sample = {"text": f"```python\n{code}```", "finish_reason": "stop"}
    assert passk.classify("code", sample, reference, "plain") == "correct"


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


def test_every_model_gets_the_same_user_text():
    assert passk.user_text("math", "1+1?").endswith("\\boxed{}.")
    assert passk.user_text("math", "1+1?").startswith("1+1?")
    assert "```python" in passk.user_text("code", "Add.")


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


def test_transformers_5_is_refused():
    run_passk.check_transformers("4.57.1")
    with pytest.raises(RuntimeError, match="breaks Olmo 3"):
        run_passk.check_transformers("5.12.1")


@pytest.fixture
def fake_vllm(monkeypatch):
    monkeypatch.setattr(run_passk, "check_transformers", lambda version: None)
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
        run_passk.score(Namespace(out_dir=str(tmp_path), workers=1, ks=[1], partial=False))


def test_score_reads_the_generate_time_source(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path, lcb_source="the-source"))
    seen = []
    items = run_passk.load_items("math500", 0, None)
    monkeypatch.setattr(
        run_passk, "load_items", lambda task, limit, source: seen.append(source) or items
    )
    run_passk.score(Namespace(out_dir=str(tmp_path), workers=1, ks=[1], partial=False))
    assert seen == ["the-source"]


def test_two_jobs_do_not_take_the_same_chunk(tmp_path):
    chunk = tmp_path / "chunk_000000.jsonl.gz"
    assert run_passk.claim(chunk)
    assert not run_passk.claim(chunk)


def test_a_dead_job_gives_its_chunk_back(tmp_path):
    chunk = tmp_path / "chunk_000000.jsonl.gz"
    assert run_passk.claim(chunk)
    old = time.time() - (run_passk.CLAIM_HOURS + 1) * 3600
    os.utime(chunk.with_suffix(".claim"), (old, old))
    assert run_passk.claim(chunk)


def test_generate_skips_chunks_another_job_holds(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    (tmp_path / "math500").mkdir(parents=True)
    run_passk.claim(tmp_path / "math500" / "chunk_000002.jsonl.gz")
    run_passk.generate(_args(tmp_path))
    assert _FakeLLM.calls == [2, 1]
    assert not (tmp_path / "math500" / "chunk_000002.jsonl.gz").exists()


def test_generate_releases_its_claims(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path))
    assert not list((tmp_path / "math500").glob("*.claim"))


def test_a_failed_chunk_is_released(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)

    def fail(self, prompts, params):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(_FakeLLM, "generate", fail)
    with pytest.raises(RuntimeError):
        run_passk.generate(_args(tmp_path))
    assert not list((tmp_path / "math500").glob("*.claim"))


def test_partial_score_marks_the_run_unfinished(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path))
    next(iter(sorted((tmp_path / "math500").glob("chunk_*.jsonl.gz")))).unlink()
    run_passk.score(Namespace(out_dir=str(tmp_path), workers=1, ks=[1], partial=True))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["math500"]["n_problems"] == 3
    assert summary["complete"] is False


def test_compare_common_uses_the_problems_both_runs_finished(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _items(monkeypatch)
    run_passk.generate(_args(tmp_path / "b"))
    next(iter(sorted((tmp_path / "b" / "math500").glob("chunk_*.jsonl.gz")))).unlink()
    run_passk.score(Namespace(out_dir=str(tmp_path / "b"), workers=1, ks=[1], partial=True))
    out = tmp_path / "c.json"
    args = dict(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(out))
    with pytest.raises(ValueError):
        run_passk.compare_runs(Namespace(**args, common=False))
    run_passk.compare_runs(Namespace(**args, common=True))
    assert json.loads(out.read_text())["math500"]["n_common"] == 3


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
    run_passk.generate(_args(tmp_path / "plain", prompt_format="plain"))
    assert seen[0] == "q0\n\n" + passk.BOXED_INSTRUCTION


def test_score_and_compare_end_to_end(tmp_path, monkeypatch, fake_vllm):
    _items(monkeypatch)
    a, b = tmp_path / "a", tmp_path / "b"
    run_passk.generate(_args(a))
    run_passk.generate(_args(b))
    run_passk.score(Namespace(out_dir=str(a), workers=1, ks=[1, 2], partial=False))
    run_passk.score(Namespace(out_dir=str(b), workers=1, ks=[1, 2], partial=False))
    summary = json.loads((a / "summary.json").read_text())
    assert summary["math500"]["pass@2"]["value"] == 1.0

    out = tmp_path / "compare.json"
    run_passk.compare_runs(Namespace(a=str(a), b=str(b), ks=[1, 2], out=str(out), common=False))
    assert json.loads(out.read_text())["math500"]["pass@1"]["difference"] == 0.0


def _scored_run(path, monkeypatch, **settings):
    _items(monkeypatch)
    run_passk.generate(_args(path, **settings))
    run_passk.score(Namespace(out_dir=str(path), workers=1, ks=[1], partial=False))


def test_compare_refuses_different_sampling(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _scored_run(tmp_path / "b", monkeypatch, temperature=0.6)
    with pytest.raises(ValueError, match="temperature"):
        run_passk.compare_runs(
            Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"), common=False)
        )


def test_compare_refuses_different_test_cases(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch, lcb_source="v5")
    _scored_run(tmp_path / "b", monkeypatch, lcb_source="v6")
    with pytest.raises(ValueError, match="lcb_source"):
        run_passk.compare_runs(
            Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"), common=False)
        )


def test_compare_allows_different_models_and_prompts(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _scored_run(tmp_path / "b", monkeypatch, model="other", prompt_format="plain")
    run_passk.compare_runs(
        Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"), common=False)
    )


def test_compare_refuses_different_tasks(tmp_path, monkeypatch, fake_vllm):
    _scored_run(tmp_path / "a", monkeypatch)
    _scored_run(tmp_path / "b", monkeypatch)
    (tmp_path / "b" / "scores" / "math500.jsonl").rename(tmp_path / "b" / "scores" / "other.jsonl")
    with pytest.raises(ValueError, match="tasks"):
        run_passk.compare_runs(
            Namespace(a=str(tmp_path / "a"), b=str(tmp_path / "b"), ks=[1], out=str(tmp_path / "c"), common=False)
        )


def test_job_script_has_valid_syntax():
    subprocess.run(["bash", "-n", "scripts/run_passk_osc.sh"], check=True)


def test_binomial_helper_stays_exact_for_64_samples():
    assert passk.pass_at_k(2, 64, 16) == pytest.approx(
        1 - math.comb(62, 16) / math.comb(64, 16)
    )
