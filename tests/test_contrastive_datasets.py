"""Tests for the 46-concept contrastive dataset loaders.

All dataset/file/URL access is mocked — tests are fully offline. Covers:

* Concept registry (46 canonical keys + 3 aliases).
* Code domain (20 directed HumanEval-X pairs from materialized JSON).
* Math domain (MiniF2F / BeyondX / MATH-500 CoT vs direct).
* IF domain (20 directed Belebele pairs from materialized JSON).
* General domain (WinoGender nominative / SST-2 / LLM-LAT refusal).
* Public API (``load_contrastive_texts``, ``list_concepts``, ``all_concept_keys``).
* Legacy aliases and clear error messages on missing datasets.
"""

from __future__ import annotations

import json
import re
from itertools import permutations
from unittest.mock import patch

import pytest

import src.contrastive_datasets as cd
from src.contrastive_datasets import (
    ALIASES,
    CODE_LANGS,
    CONCEPTS,
    IF_LANGS,
    _extract_text,
    _stream_n_samples,
    _strip_code_fences,
    all_concept_keys,
    list_concepts,
    load_contrastive_texts,
)


# =============================================================================
# Fixtures: materialized JSON payloads written to a tmp datasets dir
# =============================================================================


@pytest.fixture(autouse=True)
def _isolated_datasets(tmp_path, monkeypatch):
    """Point dataset_store paths at a tmp dir for every test."""
    import src.dataset_store as store

    monkeypatch.setattr(store, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(store, "SHARED_IDS_PATH", tmp_path / "shared_item_ids.json")
    # Reload the contrastive_datasets path references too.
    monkeypatch.setattr(cd, "humaneval_x_shared_ids", store.humaneval_x_shared_ids)
    monkeypatch.setattr(cd, "belebele_shared_keys", store.belebele_shared_keys)
    monkeypatch.setattr(cd, "dataset_path", store.dataset_path)
    monkeypatch.setattr(cd, "load_dataset_json", store.load_dataset_json)
    yield tmp_path


def _write_json(path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def _humaneval_payload(
    langs=("python", "cpp", "java", "js", "go"),
    n_ids: int = 60,
    shared_subset: int = 50,
):
    languages = {}
    for lang in langs:
        items = []
        for i in range(n_ids):
            prompt = f"{lang} prompt {i}\n"
            canonical = f"{lang} solution {i}\n"
            items.append(
                {
                    "task_id": f"{lang.upper()}/{i}",
                    "numeric_id": i,
                    "prompt": prompt,
                    "canonical_solution": canonical,
                    "code": prompt + canonical,
                }
            )
        languages[lang] = items
    return {
        "dataset": "humaneval-x",
        "revision": "pinned-test-rev",
        "languages": languages,
    }


def _belebele_payload(
    dialects=("eng_Latn", "fra_Latn", "deu_Latn", "zho_Hans", "jpn_Jpan"),
    n_items: int = 60,
):
    dialects_data = {}
    for dialect in dialects:
        items = []
        for i in range(n_items):
            items.append(
                {
                    "link": f"link-{i}",
                    "question_number": i + 1,
                    "text": f"{dialect} passage {i}\nQ{i}\nA\nB\nC\nD",
                }
            )
        dialects_data[dialect] = items
    return {"dataset": "belebele", "dialects": dialects_data}


def _minif2f_payload(n: int = 60):
    items = [
        {
            "id": f"m2f_{i}",
            "informal": f"Solve x + {i} = {i + 1}",
            "formal": f"theorem t{i} : x + {i} = {i + 1}",
        }
        for i in range(n)
    ]
    return {"dataset": "minif2f", "items": items}


def _beyondx_payload(n: int = 60):
    items = [
        {
            "id": f"bx_{i}",
            "problem": f"Problem number {i}",
            "equations": json.dumps([f"x = {i}"]),
        }
        for i in range(n)
    ]
    return {"dataset": "beyondx", "items": items}


def _math500_payload(n: int = 50):
    items = []
    for i in range(n):
        problem = f"What is {i}+{i}?"
        solution = f"Solution: {i}+{i}=2{i}."
        answer = str(2 * i)
        direct = f"The answer is {answer}"
        items.append(
            {
                "unique_id": f"m500_{i}",
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "direct_answer": direct,
                "cot_text": problem + "\n" + solution,
                "direct_text": problem + "\n" + direct,
            }
        )
    return {"dataset": "math500", "items": items}


def _winogender_payload():
    rows = []
    for i in range(60):
        rows.append(
            {
                "occupation(0)": f"worker-{i}",
                "other-participant(1)": "client",
                "answer": str(i % 2),
                "sentence": f"The $OCCUPATION said that $NOM_PRONOUN would help.",
            }
        )
    return {"dataset": "winogender", "rows": rows}


def _sst2_payload(per_class: int = 100):
    items = []
    for i in range(per_class):
        items.append({"sentence": f"bad movie number {i}", "label": 0})
        items.append({"sentence": f"great film number {i}", "label": 1})
    return {"dataset": "sst2", "items": items, "counts": {0: per_class, 1: per_class}}


def _llm_lat_payload(kind: str, n: int = 100):
    texts = [f"{kind} text number {i}" for i in range(n)]
    return {"dataset": f"LLM-LAT/{kind}", "texts": texts}


def _write_all_datasets(tmp_path):
    _write_json(tmp_path / "humaneval_x.json", _humaneval_payload())
    _write_json(tmp_path / "belebele.json", _belebele_payload())
    _write_json(tmp_path / "minif2f.json", _minif2f_payload())
    _write_json(tmp_path / "beyondx.json", _beyondx_payload())
    _write_json(tmp_path / "math500.json", _math500_payload())
    _write_json(tmp_path / "winogender.json", _winogender_payload())
    _write_json(tmp_path / "sst2.json", _sst2_payload())
    _write_json(tmp_path / "llm_lat_harmful.json", _llm_lat_payload("harmful-dataset"))
    _write_json(tmp_path / "llm_lat_benign.json", _llm_lat_payload("benign-dataset"))


# =============================================================================
# Text helper tests (pure functions)
# =============================================================================


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


class TestStreamNSamples:
    def test_returns_exactly_n_samples(self):
        def gen():
            for i in range(1000):
                yield {"text": f"sample {i}"}

        assert _stream_n_samples(gen(), n=10)[9] == "sample 9"

    def test_filters_empty_text(self):
        def gen():
            yield {"text": "good"}
            yield {"text": ""}
            yield {"text": "also good"}

        assert _stream_n_samples(gen(), n=5) == ["good", "also good"]


class TestStripCodeFences:
    def test_removes_fences(self):
        text = "```python\ncode\n```"
        assert _strip_code_fences(text) == "code"

    def test_preserves_unfenced(self):
        assert _strip_code_fences("plain code\n") == "plain code\n"


# =============================================================================
# Concept registry
# =============================================================================


class TestConceptRegistry:
    def test_registry_has_46_canonical_concepts(self):
        assert len(all_concept_keys()) == 46

    def test_paired_concepts_alias_matches_concepts(self):
        assert cd.PAIRED_CONCEPTS is CONCEPTS

    def test_code_concepts_are_20_directed_pairs(self):
        code_keys = [k for k, v in CONCEPTS.items() if v["domain"] == "code"]
        assert len(code_keys) == 20
        expected = {f"code_{a}_vs_{b}" for a, b in permutations(CODE_LANGS, 2)}
        assert set(code_keys) == expected

    def test_if_concepts_are_20_directed_pairs(self):
        if_keys = [k for k, v in CONCEPTS.items() if v["domain"] == "if"]
        assert len(if_keys) == 20
        expected = {f"if_{a}_vs_{b}" for a, b in permutations(IF_LANGS, 2)}
        assert set(if_keys) == expected

    def test_math_concepts_are_exactly_three(self):
        math_keys = [k for k, v in CONCEPTS.items() if v["domain"] == "math"]
        assert set(math_keys) == {
            "math_informal_vs_formal",
            "math_nl_vs_equations",
            "math_cot_vs_direct",
        }

    def test_general_concepts_are_exactly_three(self):
        gen_keys = [k for k, v in CONCEPTS.items() if v["domain"] == "general"]
        assert set(gen_keys) == {
            "gender_she_vs_he",
            "sentiment_label0_vs_label1",
            "refusal_harmful_vs_benign",
        }

    def test_legacy_aliases_are_available(self):
        assert set(ALIASES.keys()) == {
            "python_vs_cpp",
            "french_vs_english_language",
            "female_vs_male_gender",
        }
        for old, new in ALIASES.items():
            assert new in CONCEPTS

    def test_list_concepts_includes_canonical_and_aliases(self):
        all_keys = set(list_concepts())
        assert all_keys >= set(all_concept_keys())
        assert all_keys >= set(ALIASES.keys())

    def test_legacy_concise_verbose_key_is_not_registered(self):
        # The old concise/verbose key must be dropped (wrong semantics).
        assert "concise_math_reasoning_vs_verbose_math_reasoning" not in CONCEPTS
        assert "concise_math_reasoning_vs_verbose_math_reasoning" not in ALIASES


class TestConceptMetadata:
    def test_code_python_vs_cpp_follows_slide_polarity(self):
        entry = CONCEPTS["code_python_vs_cpp"]
        # slide arrow python -> cpp means +cpp -python
        assert entry["positive"] == "cpp"
        assert entry["negative"] == "python"
        assert entry["src"] == "python"
        assert entry["tgt"] == "cpp"

    def test_if_eng_vs_fra_follows_slide_polarity(self):
        entry = CONCEPTS["if_eng_vs_fra"]
        # slide arrow eng -> fra means +fra -eng
        assert entry["positive"] == "fra"
        assert entry["negative"] == "eng"

    def test_gender_she_vs_he_polarity(self):
        entry = CONCEPTS["gender_she_vs_he"]
        # slide arrow she -> he means +he -she
        assert entry["positive"] == "he"
        assert entry["negative"] == "she"

    def test_sentiment_polarity(self):
        entry = CONCEPTS["sentiment_label0_vs_label1"]
        assert entry["positive"] == "label1"
        assert entry["negative"] == "label0"

    def test_refusal_polarity(self):
        entry = CONCEPTS["refusal_harmful_vs_benign"]
        assert entry["positive"] == "benign"
        assert entry["negative"] == "harmful"

    def test_math_informal_vs_formal_polarity(self):
        entry = CONCEPTS["math_informal_vs_formal"]
        assert entry["positive"] == "formal"
        assert entry["negative"] == "informal"

    def test_math_nl_vs_equations_polarity(self):
        entry = CONCEPTS["math_nl_vs_equations"]
        assert entry["positive"] == "equations"
        assert entry["negative"] == "nl"


# =============================================================================
# HumanEval-X directed pairs (Code)
# =============================================================================


class TestHumanEvalXDirectedPairs:
    def test_loads_pinned_shared_ids_from_json(self, _isolated_datasets):
        _write_json(_isolated_datasets / "humaneval_x.json", _humaneval_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {"humaneval_x_task_ids": list(range(50))},
        )
        pairs = cd.load_humaneval_x_directed_pairs("python", "cpp", n_samples=50)
        assert len(pairs) == 50
        for pos, neg in pairs:
            assert "cpp" in pos
            assert "python" in neg

    def test_tgt_is_positive_src_is_negative(self, _isolated_datasets):
        _write_json(_isolated_datasets / "humaneval_x.json", _humaneval_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {"humaneval_x_task_ids": list(range(10))},
        )
        pairs = cd.load_humaneval_x_directed_pairs("python", "go", n_samples=10)
        for pos, neg in pairs:
            assert pos.startswith("go")
            assert neg.startswith("python")

    def test_all_20_code_concepts_load(self, _isolated_datasets):
        _write_json(_isolated_datasets / "humaneval_x.json", _humaneval_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {"humaneval_x_task_ids": list(range(50))},
        )
        code_keys = [k for k, v in CONCEPTS.items() if v["domain"] == "code"]
        assert len(code_keys) == 20
        for concept in code_keys:
            pos, neg = load_contrastive_texts(concept, n_samples=50)
            assert len(pos) == 50
            assert len(neg) == 50
            entry = CONCEPTS[concept]
            assert entry["tgt"] in pos[0]
            assert entry["src"] in neg[0]

    def test_src_eq_tgt_raises(self, _isolated_datasets):
        _write_json(_isolated_datasets / "humaneval_x.json", _humaneval_payload())
        with pytest.raises(ValueError, match="src == tgt"):
            cd.load_humaneval_x_directed_pairs("python", "python", n_samples=5)

    def test_unknown_language_raises(self, _isolated_datasets):
        _write_json(_isolated_datasets / "humaneval_x.json", _humaneval_payload())
        with pytest.raises(ValueError, match="Unknown HumanEval-X"):
            cd.load_humaneval_x_directed_pairs("ruby", "python", n_samples=5)

    def test_missing_dataset_raises_with_clear_hint(self, _isolated_datasets):
        with pytest.raises(FileNotFoundError, match="download_datasets.py"):
            cd.load_humaneval_x_directed_pairs("python", "cpp", n_samples=5)

    def test_strips_markdown_code_fences(self, _isolated_datasets):
        payload = _humaneval_payload()
        for item in payload["languages"]["python"][:5]:
            item["code"] = "```python\n" + item["code"] + "```"
        _write_json(_isolated_datasets / "humaneval_x.json", payload)
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {"humaneval_x_task_ids": list(range(5))},
        )
        pairs = cd.load_humaneval_x_directed_pairs("python", "cpp", n_samples=5)
        for pos, neg in pairs:
            assert "```" not in pos
            assert "```" not in neg

    def test_legacy_alias_python_vs_cpp_routes_to_code_python_vs_cpp(
        self, _isolated_datasets
    ):
        _write_json(_isolated_datasets / "humaneval_x.json", _humaneval_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {"humaneval_x_task_ids": list(range(5))},
        )
        pos, neg = load_contrastive_texts("python_vs_cpp", n_samples=5)
        for p in pos:
            assert "cpp" in p
        for n in neg:
            assert "python" in n


# =============================================================================
# Math domain
# =============================================================================


class TestMathPairs:
    def test_minif2f_formal_is_positive(self, _isolated_datasets):
        _write_json(_isolated_datasets / "minif2f.json", _minif2f_payload(n=60))
        pairs = cd.load_minif2f_pairs(n_samples=10)
        assert len(pairs) == 10
        for formal, informal in pairs:
            assert formal.startswith("theorem")
            assert informal.startswith("Solve")

    def test_beyondx_equations_is_positive(self, _isolated_datasets):
        _write_json(_isolated_datasets / "beyondx.json", _beyondx_payload(n=60))
        pairs = cd.load_beyondx_pairs(n_samples=10)
        assert len(pairs) == 10
        for equations, problem in pairs:
            assert "x =" in equations
            assert problem.startswith("Problem")

    def test_math_cot_vs_direct_direct_is_positive(self, _isolated_datasets):
        _write_json(_isolated_datasets / "math500.json", _math500_payload(n=50))
        pairs = cd.load_math_cot_vs_direct_pairs(n_samples=10)
        assert len(pairs) == 10
        for direct, cot in pairs:
            assert "The answer is" in direct
            assert "Solution:" in cot

    def test_math_informal_vs_formal_via_public_api(self, _isolated_datasets):
        _write_json(_isolated_datasets / "minif2f.json", _minif2f_payload(n=60))
        pos, neg = load_contrastive_texts("math_informal_vs_formal", n_samples=10)
        assert all(p.startswith("theorem") for p in pos)
        assert all(n.startswith("Solve") for n in neg)

    def test_math_nl_vs_equations_via_public_api(self, _isolated_datasets):
        _write_json(_isolated_datasets / "beyondx.json", _beyondx_payload(n=60))
        pos, neg = load_contrastive_texts("math_nl_vs_equations", n_samples=10)
        assert all("x =" in p for p in pos)
        assert all(n.startswith("Problem") for n in neg)

    def test_math_cot_vs_direct_via_public_api(self, _isolated_datasets):
        _write_json(_isolated_datasets / "math500.json", _math500_payload(n=50))
        pos, neg = load_contrastive_texts("math_cot_vs_direct", n_samples=10)
        assert all("The answer is" in p for p in pos)
        assert all("Solution:" in n for n in neg)

    def test_missing_minif2f_raises(self, _isolated_datasets):
        with pytest.raises(FileNotFoundError, match="download_datasets"):
            cd.load_minif2f_pairs(n_samples=10)

    def test_too_few_pairs_raises(self, _isolated_datasets):
        _write_json(_isolated_datasets / "minif2f.json", _minif2f_payload(n=5))
        with pytest.raises(ValueError, match="only 5"):
            cd.load_minif2f_pairs(n_samples=10)


# =============================================================================
# Belebele directed pairs (IF)
# =============================================================================


class TestBelebeleDirectedPairs:
    def test_loads_50_pairs_per_directed_concept(self, _isolated_datasets):
        _write_json(_isolated_datasets / "belebele.json", _belebele_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {
                "belebele_keys": [
                    {"link": f"link-{i}", "question_number": i + 1} for i in range(50)
                ]
            },
        )
        pairs = cd.load_belebele_directed_pairs("eng", "fra", n_samples=50)
        assert len(pairs) == 50

    def test_tgt_dialect_is_positive(self, _isolated_datasets):
        _write_json(_isolated_datasets / "belebele.json", _belebele_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {
                "belebele_keys": [
                    {"link": f"link-{i}", "question_number": i + 1} for i in range(10)
                ]
            },
        )
        pairs = cd.load_belebele_directed_pairs("eng", "fra", n_samples=10)
        for pos, neg in pairs:
            assert pos.startswith("fra_Latn")
            assert neg.startswith("eng_Latn")

    def test_all_20_if_concepts_load(self, _isolated_datasets):
        _write_json(_isolated_datasets / "belebele.json", _belebele_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {
                "belebele_keys": [
                    {"link": f"link-{i}", "question_number": i + 1} for i in range(50)
                ]
            },
        )
        if_keys = [k for k, v in CONCEPTS.items() if v["domain"] == "if"]
        assert len(if_keys) == 20
        for concept in if_keys:
            pos, neg = load_contrastive_texts(concept, n_samples=50)
            assert len(pos) == 50
            assert len(neg) == 50

    def test_missing_dataset_raises(self, _isolated_datasets):
        with pytest.raises(FileNotFoundError, match="download_datasets"):
            cd.load_belebele_directed_pairs("eng", "fra", n_samples=5)

    def test_unknown_language_raises(self, _isolated_datasets):
        _write_json(_isolated_datasets / "belebele.json", _belebele_payload())
        with pytest.raises(ValueError, match="Unknown IF language"):
            cd.load_belebele_directed_pairs("xyz", "eng", n_samples=5)

    def test_legacy_french_vs_english_alias(self, _isolated_datasets):
        _write_json(_isolated_datasets / "belebele.json", _belebele_payload())
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {
                "belebele_keys": [
                    {"link": f"link-{i}", "question_number": i + 1} for i in range(5)
                ]
            },
        )
        pos, neg = load_contrastive_texts("french_vs_english_language", n_samples=5)
        # New polarity per slides: eng -> fra means +fra -eng
        assert all(p.startswith("fra_Latn") for p in pos)
        assert all(n.startswith("eng_Latn") for n in neg)


# =============================================================================
# WinoGender (gender she -> he)
# =============================================================================


class TestWinogenderPairs:
    def test_he_is_positive_she_is_negative(self, _isolated_datasets):
        _write_json(_isolated_datasets / "winogender.json", _winogender_payload())
        pairs = cd.load_winogender_pairs(n_samples=50)
        assert len(pairs) == 50
        for he, she in pairs:
            assert re.search(r"\bhe\b", he.lower())
            assert re.search(r"\bshe\b", she.lower())

    def test_pairs_are_minimal_nominative_swaps(self, _isolated_datasets):
        _write_json(_isolated_datasets / "winogender.json", _winogender_payload())
        pairs = cd.load_winogender_pairs(n_samples=50)
        for he, she in pairs:
            swapped = re.sub(r"\bhe\b", "she", he, count=1)
            assert swapped == she

    def test_legacy_female_vs_male_alias_routes_with_new_polarity(
        self, _isolated_datasets
    ):
        _write_json(_isolated_datasets / "winogender.json", _winogender_payload())
        pos, neg = load_contrastive_texts("female_vs_male_gender", n_samples=50)
        # New polarity per slides: +he -she
        assert all(re.search(r"\bhe\b", p.lower()) for p in pos)
        assert all(re.search(r"\bshe\b", n.lower()) for n in neg)

    def test_missing_dataset_falls_back_to_http(self, _isolated_datasets):
        tsv = (
            "\n".join(
                [
                    "occupation(0)\tother-participant(1)\tanswer\tsentence",
                    "worker\tclient\t0\tThe $OCCUPATION said $NOM_PRONOUN left.",
                    "nurse\tpatient\t1\tThe $OCCUPATION saw $NOM_PRONOUN leave.",
                ]
            )
            + "\n"
        )
        with patch("urllib.request.urlopen") as mock_open:
            response = mock_open.return_value
            response.__enter__.return_value = response
            response.read.return_value = tsv.encode("utf-8")
            pairs = cd.load_winogender_pairs(n_samples=2)
        assert len(pairs) == 2


# =============================================================================
# SST-2
# =============================================================================


class TestSST2Pairs:
    def test_label1_is_positive_label0_is_negative(self, _isolated_datasets):
        _write_json(_isolated_datasets / "sst2.json", _sst2_payload())
        pairs = cd.load_sst2_pairs(n_samples=10)
        assert len(pairs) == 10
        for pos, neg in pairs:
            assert "great" in pos
            assert "bad" in neg

    def test_public_api(self, _isolated_datasets):
        _write_json(_isolated_datasets / "sst2.json", _sst2_payload())
        pos, neg = load_contrastive_texts("sentiment_label0_vs_label1", n_samples=20)
        assert len(pos) == 20
        assert len(neg) == 20

    def test_missing_dataset_raises(self, _isolated_datasets):
        with pytest.raises(FileNotFoundError, match="download_datasets"):
            cd.load_sst2_pairs(n_samples=5)


# =============================================================================
# LLM-LAT refusal
# =============================================================================


class TestRefusalPairs:
    def test_benign_is_positive_harmful_is_negative(self, _isolated_datasets):
        _write_json(
            _isolated_datasets / "llm_lat_benign.json",
            _llm_lat_payload("benign-dataset"),
        )
        _write_json(
            _isolated_datasets / "llm_lat_harmful.json",
            _llm_lat_payload("harmful-dataset"),
        )
        pairs = cd.load_refusal_pairs(n_samples=10)
        assert len(pairs) == 10
        for pos, neg in pairs:
            assert "benign" in pos
            assert "harmful" in neg

    def test_public_api(self, _isolated_datasets):
        _write_json(
            _isolated_datasets / "llm_lat_benign.json",
            _llm_lat_payload("benign-dataset"),
        )
        _write_json(
            _isolated_datasets / "llm_lat_harmful.json",
            _llm_lat_payload("harmful-dataset"),
        )
        pos, neg = load_contrastive_texts("refusal_harmful_vs_benign", n_samples=10)
        assert all("benign" in p for p in pos)
        assert all("harmful" in n for n in neg)


# =============================================================================
# Public API integration
# =============================================================================


class TestPublicAPI:
    def test_all_46_concepts_resolve(self, _isolated_datasets):
        _write_all_datasets(_isolated_datasets)
        _write_json(
            _isolated_datasets / "shared_item_ids.json",
            {
                "humaneval_x_task_ids": list(range(50)),
                "belebele_keys": [
                    {"link": f"link-{i}", "question_number": i + 1} for i in range(50)
                ],
            },
        )
        for concept in all_concept_keys():
            pos, neg = load_contrastive_texts(concept, n_samples=50)
            assert len(pos) == 50, concept
            assert len(neg) == 50, concept

    def test_unknown_concept_raises(self):
        with pytest.raises(ValueError, match="Unknown concept"):
            load_contrastive_texts("not_a_real_concept", n_samples=5)

    def test_zero_n_samples_returns_empty(self, _isolated_datasets):
        _write_json(_isolated_datasets / "minif2f.json", _minif2f_payload(n=60))
        pos, neg = load_contrastive_texts("math_informal_vs_formal", n_samples=0)
        assert pos == []
        assert neg == []

    def test_missing_dataset_message_mentions_runner(self, _isolated_datasets):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_contrastive_texts("math_cot_vs_direct", n_samples=10)
        assert "download_datasets.py" in str(excinfo.value)


# =============================================================================
# dataset_store integration
# =============================================================================


class TestDatasetStore:
    def test_get_shared_ids_empty_when_missing(self, _isolated_datasets):
        import src.dataset_store as store

        data = store.get_shared_ids()
        assert data["humaneval_x_task_ids"] == []
        assert data["belebele_keys"] == []

    def test_save_and_reload_shared_ids(self, _isolated_datasets):
        import src.dataset_store as store

        store.save_shared_ids(
            {
                "humaneval_x_task_ids": list(range(50)),
                "belebele_keys": [{"link": "l1", "question_number": 1}],
            }
        )
        data = store.get_shared_ids()
        assert data["humaneval_x_task_ids"] == list(range(50))
        assert data["belebele_keys"] == [{"link": "l1", "question_number": 1}]

    def test_save_shared_ids_preserves_existing_keys(self, _isolated_datasets):
        import src.dataset_store as store

        store.save_shared_ids({"humaneval_x_task_ids": [1, 2, 3]})
        store.save_shared_ids({"belebele_keys": [{"link": "l", "question_number": 1}]})
        data = store.get_shared_ids()
        assert data["humaneval_x_task_ids"] == [1, 2, 3]
        assert data["belebele_keys"] == [{"link": "l", "question_number": 1}]

    def test_sample_shared_indices_is_deterministic(self):
        import src.dataset_store as store

        pool = list(range(200))
        a = store.sample_shared_indices(pool, n=50, seed=42)
        b = store.sample_shared_indices(pool, n=50, seed=42)
        assert a == b
        assert len(a) == 50
