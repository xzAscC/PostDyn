from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

import pytest

import postdyn.math500_eval as math500_eval
from postdyn.math500_eval import (
    MATH500_COUNT,
    evaluate_first50,
    item_filename,
    load_first50,
    score_answer,
)


DATASET = Path(__file__).parents[1] / "data" / "math500.json"


class FakeTokenizer:
    eos_token_id: int | None = None
    pad_token_id: int | None = None

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return "candidate"


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool, int]] = []

    def generate(
        self,
        input_ids: Sequence[int],
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
    ) -> list[int]:
        self.calls.append((str(len(input_ids)), max_new_tokens, do_sample, num_beams))
        return list(input_ids) + [999]


class BatchTokenizer(FakeTokenizer):
    pad_token_id: int | None = 0


class BatchGenerator:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[list[list[int]], list[list[int]]]] = []

    def generate_batch(
        self,
        input_ids,
        attention_mask,
        *,
        max_new_tokens,
        do_sample=False,
        num_beams=1,
    ):
        self.batch_calls.append(
            ([list(row) for row in input_ids], [list(row) for row in attention_mask])
        )
        return [list(row) + [999] for row in input_ids]

    def generate(self, input_ids, *, max_new_tokens, do_sample=False, num_beams=1):
        raise AssertionError("singleton generation was not expected")


class RaisingBatchGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def generate_batch(
        self, input_ids, attention_mask, *, max_new_tokens, do_sample=False, num_beams=1
    ):
        self.batch_calls += 1
        raise RuntimeError("batch failed")


class MalformedBatchGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def generate_batch(
        self, input_ids, attention_mask, *, max_new_tokens, do_sample=False, num_beams=1
    ):
        self.batch_calls += 1
        return [list(row[:-1]) for row in input_ids]


class FakeVerifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def parse(self, expression: str, *, fallback_mode: str) -> list[object]:
        assert fallback_mode == "no_fallback"
        return [expression]

    def verify(self, gold: list[object], target: list[object]) -> bool:
        self.calls.append((str(gold[0]), str(target[0])))
        return True


class InvalidVerifier:
    def parse(self, expression: str, *, fallback_mode: str) -> list[object]:
        assert fallback_mode == "no_fallback"
        return []

    def verify(self, gold: list[object], target: list[object]) -> bool:
        raise AssertionError("verify must not run for invalid parses")


def test_loads_stored_first50_in_order() -> None:
    items, dataset_hash = load_first50(DATASET)
    assert len(items) == MATH500_COUNT
    assert [item.index for item in items] == list(range(MATH500_COUNT))
    assert items[0].problem == json.loads(DATASET.read_text())["items"][0]["problem"]
    assert len(dataset_hash) == 64


def test_fake_evaluation_is_exactly_50_and_resumes(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    generator = FakeGenerator()
    verifier = FakeVerifier()
    summary = evaluate_first50(
        model="fake/olmo",
        revision="fake-revision",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=tokenizer,
        generator=generator,
        max_new_tokens=17,
        dtype="float32",
        quantization="none",
        verifier=verifier,
    )
    assert summary.n_expected == MATH500_COUNT
    assert summary.n_processed == MATH500_COUNT
    assert summary.n_correct == MATH500_COUNT
    assert len(generator.calls) == MATH500_COUNT
    assert len(verifier.calls) == MATH500_COUNT
    assert len(list(tmp_path.glob("math500_*.json"))) == MATH500_COUNT
    assert (tmp_path / item_filename(0)).read_text().find("prompt_template") >= 0

    second = evaluate_first50(
        model="fake/olmo",
        revision="fake-revision",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=tokenizer,
        generator=generator,
        max_new_tokens=17,
        dtype="float32",
        quantization="none",
        verifier=verifier,
    )
    assert second.n_processed == MATH500_COUNT
    assert len(generator.calls) == MATH500_COUNT


def test_batches_only_misses_in_stable_order_and_maps_rows(tmp_path: Path) -> None:
    first = BatchGenerator()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=BatchTokenizer(),
        generator=first,
        batch_size=2,
        verifier=FakeVerifier(),
    )
    assert len(first.batch_calls) == 25
    first_ids, first_masks = first.batch_calls[0]
    assert first_masks[0][-1] == 1
    assert first_masks[1] == [1] * len(first_ids[1])
    (tmp_path / item_filename(0)).unlink()
    (tmp_path / item_filename(1)).unlink()
    second = BatchGenerator()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=BatchTokenizer(),
        generator=second,
        batch_size=2,
        verifier=FakeVerifier(),
    )
    assert len(second.batch_calls) == 1
    assert len(second.batch_calls[0][0]) == 2
    assert json.loads((tmp_path / item_filename(0)).read_text())["index"] == 0
    assert json.loads((tmp_path / item_filename(1)).read_text())["index"] == 1


def test_batch_size_falls_back_to_unchanged_singleton_path(tmp_path: Path) -> None:
    generator = FakeGenerator()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generator=generator,
        batch_size=2,
        verifier=FakeVerifier(),
    )
    assert len(generator.calls) == MATH500_COUNT


@pytest.mark.parametrize(
    "generator_type", [RaisingBatchGenerator, MalformedBatchGenerator]
)
def test_batch_failure_disables_future_batch_attempts(
    tmp_path: Path, generator_type
) -> None:
    generator = generator_type()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=BatchTokenizer(),
        generator=generator,
        batch_size=2,
        verifier=FakeVerifier(),
    )
    assert generator.batch_calls == 1
    assert len(generator.calls) == MATH500_COUNT


def test_evaluate_rejects_batch_size_three_before_loading_dataset(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="either 1 or 2"):
        evaluate_first50(
            model="fake/olmo",
            revision="r1",
            dataset_path=tmp_path / "does-not-exist.json",
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generator=FakeGenerator(),
            batch_size=3,
        )


def test_invalid_parse_is_incorrect_and_tampered_cache_regenerates(
    tmp_path: Path,
) -> None:
    parsed, correct, error = score_answer("1", "not math", InvalidVerifier())
    assert (parsed, correct, error) == (False, False, "invalid_parse")

    tokenizer = FakeTokenizer()
    generator = FakeGenerator()
    verifier = FakeVerifier()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=tokenizer,
        generator=generator,
        verifier=verifier,
    )
    first = json.loads((tmp_path / item_filename(0)).read_text())
    first["answer"] = "tampered"
    first["correct"] = True
    (tmp_path / item_filename(0)).write_text(json.dumps(first))
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=tokenizer,
        generator=generator,
        verifier=verifier,
    )
    assert len(generator.calls) == MATH500_COUNT + 1


def test_keyboard_interrupt_resumes_from_authoritative_item_cache(
    tmp_path: Path,
) -> None:
    class InterruptingGenerator(FakeGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.remaining = 2

        def generate(self, input_ids, *, max_new_tokens, do_sample=False, num_beams=1):
            if self.remaining == 0:
                raise KeyboardInterrupt
            self.remaining -= 1
            return super().generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
            )

    interrupted = InterruptingGenerator()
    try:
        evaluate_first50(
            model="fake/olmo",
            revision="r1",
            dataset_path=DATASET,
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generator=interrupted,
            verifier=FakeVerifier(),
            experiment_identity={"condition": "baseline"},
        )
    except KeyboardInterrupt:
        pass
    assert (tmp_path / item_filename(0)).is_file()
    assert (tmp_path / item_filename(1)).is_file()

    resumed = FakeGenerator()
    summary = evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generator=resumed,
        verifier=FakeVerifier(),
        experiment_identity={"condition": "baseline"},
    )
    assert summary.n_processed == MATH500_COUNT
    assert len(resumed.calls) == MATH500_COUNT - 2


def test_experiment_identity_mismatch_regenerates_cache(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    first = FakeGenerator()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=tokenizer,
        generator=first,
        verifier=FakeVerifier(),
        experiment_identity={"condition": "baseline"},
    )
    second = FakeGenerator()
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=tokenizer,
        generator=second,
        verifier=FakeVerifier(),
        experiment_identity={"condition": "layer_3_U_pos"},
    )
    assert len(second.calls) == MATH500_COUNT


def test_generation_failure_does_not_become_cached_answer(tmp_path: Path) -> None:
    class FailingGenerator(FakeGenerator):
        def generate(self, input_ids, *, max_new_tokens, do_sample=False, num_beams=1):
            raise RuntimeError("transient generation failure")

    with pytest.raises(RuntimeError, match="transient generation failure"):
        evaluate_first50(
            model="fake/olmo",
            revision="r1",
            dataset_path=DATASET,
            output_dir=tmp_path,
            tokenizer=FakeTokenizer(),
            generator=FailingGenerator(),
            verifier=FakeVerifier(),
        )

    assert not (tmp_path / item_filename(0)).exists()
    assert not (tmp_path / "summary.json").exists()

    retry = FakeGenerator()
    summary = evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generator=retry,
        verifier=FakeVerifier(),
    )
    assert summary.n_processed == MATH500_COUNT
    assert len(retry.calls) == MATH500_COUNT


def test_authoritative_summary_fails_closed_without_math_verify(
    tmp_path: Path, monkeypatch
) -> None:
    evaluate_first50(
        model="fake/olmo",
        revision="r1",
        dataset_path=DATASET,
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        generator=FakeGenerator(),
        verifier=FakeVerifier(),
    )

    monkeypatch.setitem(sys.modules, "math_verify", None)
    assert (
        math500_eval.load_authoritative_summary(
            output_dir=tmp_path,
            model="fake/olmo",
            model_key="",
            revision="r1",
            dataset_path=DATASET,
            max_new_tokens=2048,
            dtype="bfloat16",
            quantization="none",
            experiment_identity={},
        )
        is None
    )
