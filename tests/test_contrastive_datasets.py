"""Tests for contrastive_datasets module (TDD).

Covers the four paired steering concepts:

    HumanEval-X Python-positive   (python_vs_cpp)
    MATH-500    Concise-positive  (concise_math_reasoning_vs_verbose_math_reasoning)
    FLORES+     French-positive   (french_vs_english_language)
    WinoGender  Female-positive   (female_vs_male_gender)

All dataset/file/URL access is mocked.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.contrastive_datasets as cd
from src.contrastive_datasets import (
    HUMANEVAL_X_DATASET,
    HUMANEVAL_X_REVISION,
    _extract_text,
    _stream_n_samples,
    load_contrastive_texts,
    load_humaneval_x_pairs,
)


class TestExtractText:
    def test_extracts_text_field(self):
        assert _extract_text({"text": "hello world"}) == "hello world"

    def test_extracts_prompt_field(self):
        assert _extract_text({"prompt": "solve x+1=2"}) == "solve x+1=2"

    def test_extracts_content_field(self):
        assert _extract_text({"content": "some content"}) == "some content"

    def test_extracts_question_field(self):
        assert _extract_text({"question": "what is 2+2?"}) == "what is 2+2?"

    def test_extracts_from_messages_chat_format(self):
        example = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        text = _extract_text(example)
        assert "hello" in text
        assert "hi there" in text

    def test_raises_on_no_text_field(self):
        with pytest.raises(KeyError, match="text"):
            _extract_text({"id": 123, "label": "foo"})

    def test_returns_empty_string_for_empty_text_field(self):
        assert _extract_text({"text": ""}) == ""


class TestStreamNSamples:
    def test_returns_exactly_n_samples(self):
        def gen():
            for i in range(1000):
                yield {"text": f"sample {i}"}

        samples = _stream_n_samples(gen(), n=10)
        assert len(samples) == 10
        assert samples[0] == "sample 0"
        assert samples[9] == "sample 9"

    def test_returns_all_if_fewer_than_n(self):
        def gen():
            for i in range(3):
                yield {"text": f"s{i}"}

        assert len(_stream_n_samples(gen(), n=10)) == 3

    def test_handles_zero_n(self):
        def gen():
            yield {"text": "x"}

        assert len(_stream_n_samples(gen(), n=0)) == 0

    def test_filters_empty_text(self):
        def gen():
            yield {"text": "good"}
            yield {"text": ""}
            yield {"text": "also good"}

        assert _stream_n_samples(gen(), n=5) == ["good", "also good"]


def _humaneval_row(language: str, task_id: int, fenced: bool = False) -> dict:
    prompt = f"{language} prompt {task_id}\n"
    solution = f"{language} solution {task_id}\n"
    if fenced:
        prompt = f"```{language}\n{prompt}```\n"
        solution = f"```{language}\n{solution}```\n"
    prefix = "Python" if language == "python" else "CPP"
    return {
        "task_id": f"{prefix}/{task_id}",
        "prompt": prompt,
        "canonical_solution": solution,
    }


def _humaneval_rows(language: str, ids, fenced: bool = False) -> list:
    return [_humaneval_row(language, i, fenced=fenced) for i in ids]


def _wire_humaneval_loader(mock_load, python_rows, cpp_rows) -> None:
    def fake(name, *args, **kwargs):
        assert name == "json"
        rows = cpp_rows if "/cpp/" in kwargs["data_files"] else python_rows
        return iter(rows)

    mock_load.side_effect = fake


def _flores_row(language: str, row_id: int) -> dict:
    return {
        "id": row_id,
        "text": f"{language} sentence number {row_id}",
        "url": f"http://example.org/{row_id}",
    }


def _flores_rows(language: str, ids) -> list:
    return [_flores_row(language, i) for i in ids]


def _wire_flores_loader(mock_load, french_rows, english_rows) -> None:
    def fake(*args, **kwargs):
        config = args[1] if len(args) > 1 else kwargs.get("name")
        cfg = (config or "").lower()
        if cfg == "fra_latn":
            return iter(french_rows)
        if cfg == "eng_latn":
            return iter(english_rows)
        raise AssertionError(f"unexpected flores config: {config!r}")

    mock_load.side_effect = fake


def _winogender_tsv_content() -> str:
    lines = ["occupation(0)\tother-participant(1)\tanswer\tsentence"]
    forms = [
        ("nom", "$NOM_PRONOUN", 18, "said that {pronoun} would return soon"),
        ("poss", "$POSS_PRONOUN", 5, "reviewed {pronoun} report"),
        ("acc", "$ACC_PRONOUN", 2, "spoke to {pronoun}"),
    ]
    row_id = 0
    for answer, subject in ((0, "zerosubject"), (1, "onesubject")):
        for form, placeholder, count, clause in forms:
            for _ in range(count):
                sentence = (
                    "The $OCCUPATION told the $PARTICIPANT that "
                    + clause.format(pronoun=placeholder)
                    + "."
                )
                lines.append(
                    f"worker-{subject}-{form}-{row_id}\tclient\t{answer}\t{sentence}"
                )
                row_id += 1
    return "\n".join(lines) + "\n"


@contextmanager
def _patched_winogender_url(tsv_content: str):
    mock_resp = MagicMock()
    mock_resp.read.return_value = tsv_content.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        yield mock_urlopen


def _math_row(row_id: int) -> dict:
    return {
        "unique_id": f"math_{row_id}",
        "problem": f"What is {row_id}+1?",
        "gold_answer": str(row_id + 1),
        "verbose_solution": (
            f"To compute {row_id}+1, we add one to {row_id}.\n"
            f"The final answer is {row_id + 1}."
        ),
        "concise_solution": f"{row_id + 1}",
    }


def _math_rows(ids) -> list:
    return [_math_row(i) for i in ids]


def _write_math_jsonl(path: Path, rows) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


class TestPairedConceptsRegistry:
    def test_registry_is_defined(self):
        assert hasattr(cd, "PAIRED_CONCEPTS")
        assert getattr(cd, "PAIRED_CONCEPTS") is not None

    def test_registry_has_exactly_four_paired_concepts(self):
        registry = getattr(cd, "PAIRED_CONCEPTS")
        assert set(registry.keys()) == {
            "python_vs_cpp",
            "concise_math_reasoning_vs_verbose_math_reasoning",
            "french_vs_english_language",
            "female_vs_male_gender",
        }

    def test_legacy_python_to_cpp_key_not_in_registry(self):
        registry = getattr(cd, "PAIRED_CONCEPTS")
        assert "python_to_cpp" not in registry

    def test_python_vs_cpp_metadata(self):
        entry = getattr(cd, "PAIRED_CONCEPTS")["python_vs_cpp"]
        assert entry["name"] == "HumanEval-X Python-positive"
        assert entry["dataset"] == "HumanEval-X"
        assert entry["direction"] == "Python-C++"
        assert entry["positive"] == "Python"
        assert entry["negative"] == "C++"

    def test_concise_math_metadata(self):
        entry = getattr(cd, "PAIRED_CONCEPTS")[
            "concise_math_reasoning_vs_verbose_math_reasoning"
        ]
        assert entry["name"] == "MATH concise-positive"
        assert entry["dataset"] == "MATH-500"
        assert entry["direction"] == "Concise-Verbose"
        assert entry["positive"] == "Concise"
        assert entry["negative"] == "Verbose"

    def test_french_english_metadata(self):
        entry = getattr(cd, "PAIRED_CONCEPTS")["french_vs_english_language"]
        assert entry["name"] == "FLORES+ French-positive"
        assert entry["dataset"] == "FLORES+"
        assert entry["direction"] == "French-English"
        assert entry["positive"] == "French"
        assert entry["negative"] == "English"

    def test_female_male_metadata(self):
        entry = getattr(cd, "PAIRED_CONCEPTS")["female_vs_male_gender"]
        assert entry["name"] == "WinoGender Female-positive"
        assert entry["dataset"] == "WinoGender"
        assert entry["direction"] == "Female-Male"
        assert entry["positive"] == "Female"
        assert entry["negative"] == "Male"


class TestHumanEvalXPairsPythonPositive:
    @patch("src.contrastive_datasets.load_dataset")
    def test_returns_exactly_50_pairs_for_task_ids_0_to_49(self, mock_load):
        ids = list(range(50))
        python_rows = _humaneval_rows("python", ids)
        cpp_rows = _humaneval_rows("cpp", ids)
        _wire_humaneval_loader(mock_load, python_rows, cpp_rows)

        pairs = load_humaneval_x_pairs(n_samples=50)

        assert len(pairs) == 50
        for idx, (py, cpp) in enumerate(pairs):
            assert f"python prompt {idx}" in py
            assert f"cpp prompt {idx}" in cpp
        assert pairs[0][0].startswith("python prompt 0")
        assert pairs[-1][1].startswith("cpp prompt 49")

    @patch("src.contrastive_datasets.load_dataset")
    def test_strips_markdown_code_fences(self, mock_load):
        python_rows = _humaneval_rows("python", [0, 1], fenced=True)
        cpp_rows = _humaneval_rows("cpp", [0, 1], fenced=True)
        _wire_humaneval_loader(mock_load, python_rows, cpp_rows)

        pairs = load_humaneval_x_pairs(n_samples=2)

        assert len(pairs) == 2
        for py, cpp in pairs:
            assert "```" not in py
            assert "```" not in cpp

    @patch("src.contrastive_datasets.load_dataset")
    def test_uses_pinned_humaneval_x_raw_jsonl_files(self, mock_load):
        def fake(name, *args, **kwargs):
            language = "cpp" if "/cpp/" in kwargs["data_files"] else "python"
            return iter(_humaneval_rows(language, [0]))

        mock_load.side_effect = fake

        load_humaneval_x_pairs(n_samples=1)

        assert mock_load.call_count == 2
        for call in mock_load.call_args_list:
            assert call.args[0] == "json"
            assert HUMANEVAL_X_DATASET in call.kwargs["data_files"]
            assert HUMANEVAL_X_REVISION in call.kwargs["data_files"]
            assert call.kwargs["split"] == "train"
            assert call.kwargs["streaming"] is True

    @patch("src.contrastive_datasets.load_dataset")
    def test_strict_gate_raises_when_fewer_shared_than_requested(self, mock_load):
        mock_load.side_effect = [
            iter(_humaneval_rows("python", [0, 1])),
            iter(_humaneval_rows("cpp", [0])),
        ]
        with pytest.raises(ValueError, match="aligned"):
            load_humaneval_x_pairs(n_samples=2)

    @patch("src.contrastive_datasets.load_dataset")
    def test_python_is_positive_direction_in_load_contrastive_texts(self, mock_load):
        python_rows = _humaneval_rows("python", [0, 1])
        cpp_rows = _humaneval_rows("cpp", [0, 1])
        _wire_humaneval_loader(mock_load, python_rows, cpp_rows)

        pos, neg = load_contrastive_texts("python_vs_cpp", n_samples=2)

        assert len(pos) == 2
        assert len(neg) == 2
        assert all("python prompt" in p for p in pos)
        assert all("cpp prompt" in n for n in neg)

    @patch("src.contrastive_datasets.load_dataset")
    def test_python_to_cpp_legacy_key_is_unsupported(self, mock_load):
        mock_load.side_effect = lambda *args, **kwargs: iter(
            _humaneval_rows("python", [0])
        )
        with pytest.raises(ValueError, match="concept"):
            load_contrastive_texts("python_to_cpp", n_samples=1)


class TestFloresPairsFrenchPositive:
    def test_flores_dataset_constant_is_openlanguagedata_plus(self):
        assert getattr(cd, "FLORES_DATASET") == "openlanguagedata/flores_plus"

    @patch("src.contrastive_datasets.load_dataset")
    def test_uses_fra_latn_and_eng_latn_configs(self, mock_load):
        french_rows = _flores_rows("french", range(60))
        english_rows = _flores_rows("english", range(60))
        _wire_flores_loader(mock_load, french_rows, english_rows)

        load_flores_pairs = getattr(cd, "load_flores_pairs")
        load_flores_pairs(n_samples=10)

        configs_used: set = set()
        for call in mock_load.call_args_list:
            if len(call.args) > 1:
                configs_used.add(call.args[1])
            if "name" in call.kwargs:
                configs_used.add(call.kwargs["name"])
        assert "fra_Latn" in configs_used
        assert "eng_Latn" in configs_used

    @patch("src.contrastive_datasets.load_dataset")
    def test_aligns_french_and_english_by_row_id(self, mock_load):
        french_rows = _flores_rows("french", range(20))
        english_rows = _flores_rows("english", range(20))
        _wire_flores_loader(mock_load, french_rows, english_rows)

        pairs = getattr(cd, "load_flores_pairs")(n_samples=10)

        assert len(pairs) == 10
        for idx, (fr, en) in enumerate(pairs):
            assert isinstance(fr, str)
            assert isinstance(en, str)
            assert str(idx) in fr
            assert str(idx) in en
            assert fr.startswith("french")
            assert en.startswith("english")

    @patch("src.contrastive_datasets.load_dataset")
    def test_limited_to_first_50_pairs_in_id_order(self, mock_load):
        french_rows = _flores_rows("french", range(80))
        english_rows = _flores_rows("english", range(80))
        _wire_flores_loader(mock_load, french_rows, english_rows)

        pairs = getattr(cd, "load_flores_pairs")(n_samples=50)

        assert len(pairs) == 50
        for idx, (fr, _) in enumerate(pairs):
            assert f"number {idx}" in fr

    @patch("src.contrastive_datasets.load_dataset")
    def test_returns_single_token_string_per_pair_member(self, mock_load):
        french_rows = _flores_rows("french", range(10))
        english_rows = _flores_rows("english", range(10))
        _wire_flores_loader(mock_load, french_rows, english_rows)

        pairs = getattr(cd, "load_flores_pairs")(n_samples=10)

        for fr, en in pairs:
            assert isinstance(fr, str)
            assert isinstance(en, str)
            assert len(fr.strip()) > 0
            assert len(en.strip()) > 0

    @patch("src.contrastive_datasets.load_dataset")
    def test_strict_gate_raises_when_fewer_aligned_than_requested(self, mock_load):
        _wire_flores_loader(
            mock_load,
            _flores_rows("french", [0, 1]),
            _flores_rows("english", [0]),
        )

        with pytest.raises(ValueError, match="aligned"):
            getattr(cd, "load_flores_pairs")(n_samples=2)

    @patch("src.contrastive_datasets.load_dataset")
    def test_duplicate_row_id_raises_instead_of_overwriting(self, mock_load):
        duplicate = _flores_rows("french", [0, 0])
        _wire_flores_loader(mock_load, duplicate, _flores_rows("english", [0]))

        with pytest.raises(ValueError, match=r"Duplicate FLORES\+ row ID: 0"):
            getattr(cd, "load_flores_pairs")(n_samples=1)

    @patch("src.contrastive_datasets.load_dataset")
    def test_french_is_positive_direction_in_load_contrastive_texts(self, mock_load):
        french_rows = _flores_rows("french", range(10))
        english_rows = _flores_rows("english", range(10))
        _wire_flores_loader(mock_load, french_rows, english_rows)

        pos, neg = load_contrastive_texts("french_vs_english_language", n_samples=10)

        assert len(pos) == 10
        assert len(neg) == 10
        assert all("french" in p.lower() for p in pos)
        assert all("english" in n.lower() for n in neg)


class TestWinogenderPairsFemalePositive:
    def test_winogender_constants_are_pinned(self):
        assert getattr(cd, "WINOGENDER_DATASET") == "rudinger/winogender-schemas"
        revision = getattr(cd, "WINOGENDER_REVISION")
        assert isinstance(revision, str)
        assert revision and len(revision) >= 7

    def test_loads_templates_from_pinned_github_url(self):
        with _patched_winogender_url(_winogender_tsv_content()) as mock_urlopen:
            getattr(cd, "load_winogender_pairs")(n_samples=50)
            assert mock_urlopen.called
            called_url = str(mock_urlopen.call_args[0][0])
            assert "rudinger/winogender-schemas" in called_url
            assert "/data/templates.tsv" in called_url
            assert getattr(cd, "WINOGENDER_REVISION") in called_url
            assert mock_urlopen.call_args.kwargs["timeout"] == 30

    def test_returns_deterministic_50_pairs(self):
        with _patched_winogender_url(_winogender_tsv_content()):
            pairs_a = getattr(cd, "load_winogender_pairs")(n_samples=50)
            pairs_b = getattr(cd, "load_winogender_pairs")(n_samples=50)
        assert pairs_a == pairs_b
        assert len(pairs_a) == 50

    def test_odd_sample_count_raises_instead_of_under_delivering(self):
        with pytest.raises(ValueError, match="even"):
            getattr(cd, "load_winogender_pairs")(n_samples=5)

    def test_each_pair_differs_by_exactly_one_gendered_pronoun(self):
        with _patched_winogender_url(_winogender_tsv_content()):
            pairs = getattr(cd, "load_winogender_pairs")(n_samples=50)

        for female, male in pairs:
            replacements = (("she", "he"), ("her", "his"), ("her", "him"))
            assert (
                sum(
                    re.sub(rf"\b{f}\b", m, female.lower(), count=1) == male.lower()
                    for f, m in replacements
                )
                == 1
            )

    def test_pronoun_forms_use_deterministic_36_10_4_stratification(self):
        with _patched_winogender_url(_winogender_tsv_content()):
            pairs = getattr(cd, "load_winogender_pairs")(n_samples=50)

        male_forms = [male.lower() for _, male in pairs]
        assert sum(bool(re.search(r"\bhe\b", text)) for text in male_forms) == 36
        assert sum(bool(re.search(r"\bhis\b", text)) for text in male_forms) == 10
        assert sum(bool(re.search(r"\bhim\b", text)) for text in male_forms) == 4
        assert not any("$" in text for pair in pairs for text in pair)

    def test_answer_stratification_is_25_zero_and_25_one(self):
        with _patched_winogender_url(_winogender_tsv_content()):
            pairs = getattr(cd, "load_winogender_pairs")(n_samples=50)

        zero_count = sum(1 for f, _ in pairs if "zerosubject" in f.lower())
        one_count = sum(1 for f, _ in pairs if "onesubject" in f.lower())
        assert zero_count == 25
        assert one_count == 25
        assert zero_count + one_count == len(pairs)

    def test_female_is_positive_direction_in_load_contrastive_texts(self):
        with _patched_winogender_url(_winogender_tsv_content()):
            pos, neg = load_contrastive_texts("female_vs_male_gender", n_samples=50)

        assert len(pos) == 50
        assert len(neg) == 50
        assert all(
            any(re.search(rf"\b{token}\b", text.lower()) for token in ("she", "her"))
            for text in pos
        )
        assert all(
            any(
                re.search(rf"\b{token}\b", text.lower())
                for token in ("he", "his", "him")
            )
            for text in neg
        )


class TestMathPairsConcisePositive:
    def test_default_n_samples_is_50(self, tmp_path, monkeypatch):
        path = _write_math_jsonl(tmp_path / "math.jsonl", _math_rows(range(60)))
        monkeypatch.setattr(cd, "MATH_JSONL_PATH", path, raising=False)

        pairs = getattr(cd, "load_math_pairs")()

        assert len(pairs) == 50

    def test_loads_local_jsonl_and_returns_pairs(self, tmp_path, monkeypatch):
        path = _write_math_jsonl(tmp_path / "math.jsonl", _math_rows(range(60)))
        monkeypatch.setattr(cd, "MATH_JSONL_PATH", path, raising=False)

        pairs = getattr(cd, "load_math_pairs")(n_samples=10)

        assert len(pairs) == 10
        for concise, verbose in pairs:
            assert isinstance(concise, str)
            assert isinstance(verbose, str)

    def test_each_text_contains_full_problem_and_solution(self, tmp_path, monkeypatch):
        path = _write_math_jsonl(tmp_path / "math.jsonl", _math_rows(range(60)))
        monkeypatch.setattr(cd, "MATH_JSONL_PATH", path, raising=False)

        pairs = getattr(cd, "load_math_pairs")(n_samples=10)

        for concise, verbose in pairs:
            assert "What is" in concise
            assert "What is" in verbose
            assert len(verbose) > len(concise)

    def test_strict_50_row_gate_raises_when_fewer_than_50(self, tmp_path, monkeypatch):
        path = _write_math_jsonl(tmp_path / "math.jsonl", _math_rows(range(30)))
        monkeypatch.setattr(cd, "MATH_JSONL_PATH", path, raising=False)

        with pytest.raises(ValueError, match="50"):
            getattr(cd, "load_math_pairs")(n_samples=50)

    def test_concise_is_positive_direction_in_load_contrastive_texts(
        self, tmp_path, monkeypatch
    ):
        path = _write_math_jsonl(tmp_path / "math.jsonl", _math_rows(range(60)))
        monkeypatch.setattr(cd, "MATH_JSONL_PATH", path, raising=False)

        pos, neg = load_contrastive_texts(
            "concise_math_reasoning_vs_verbose_math_reasoning", n_samples=10
        )

        assert len(pos) == 10
        assert len(neg) == 10
        for concise, verbose in zip(pos, neg):
            assert len(concise) < len(verbose)
