"""Tests for the eight-class probe-record collection and activation persistence.

Covers:
  * 8 x 50 record balance (400 total)
  * cross-language group alignment (same task IDs across all six code classes)
  * gender pairing (she/he from the same WinoGender template)
  * no group leakage (groups are disjoint, correct sizes)
  * safetensors + JSON roundtrip (save then load, verify tensor equality)
  * mock activation extraction (no real model, no chat template)

No GPU or network required -- the mock model returns deterministic hidden
states matching the HF transformers calling convention.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from postdyn.probe_activations import (
    CODE_PROBE_CLASSES,
    GENDER_PROBE_CLASSES,
    PROBE_CLASSES,
    PROTOCOL,
    ProbeRecord,
    build_code_records,
    build_gender_records,
    build_probe_records,
    compute_records_fingerprint,
    default_activations_root,
    extract_probe_activations,
    group_records,
    is_layer_complete,
    load_layer_activations,
    load_manifest,
    load_records_json,
    load_target_task_ids,
    record_text_sha256,
    run_extraction_with_resume,
    save_layer_activations,
    save_records_json,
    validate_probe_records,
    validate_sidecar_record_identity,
)
from postdyn.rl_zero_experiment import (
    N_SAMPLES,
    RL_ZERO_CODE_RESULTS_ROOT,
    PROBE_CLASSES as EXP_PROBE_CLASSES,
)


# =============================================================================
# Mock model / tokenizer (same convention as test_concept_dynamics.py)
# =============================================================================


class _MockModelOutput:
    """Mimics HF transformers model output with .hidden_states tuple."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class MockModel:
    """Mock transformer returning deterministic hidden states.

    hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, d),
    matching the HF convention where index 0 = embedding layer.
    """

    def __init__(self, n_layers: int = 4, d_model: int = 8, seq_len: int = 5):
        self.config = type(
            "Config", (), {"num_hidden_layers": n_layers, "hidden_size": d_model}
        )()
        self.device = torch.device("cpu")
        self._n_layers = n_layers
        self._d_model = d_model
        self._seq_len = seq_len

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        bs = input_ids.shape[0] if input_ids is not None else 1
        seq = input_ids.shape[1] if input_ids is not None else self._seq_len
        torch.manual_seed(hash((bs, seq, self._d_model)) & 0xFFFF)
        hidden_states = tuple(
            torch.randn(bs, seq, self._d_model) for _ in range(self._n_layers + 1)
        )
        return _MockModelOutput(hidden_states)


class MockTokenizer:
    """Mock tokenizer returning fixed-size input_ids."""

    def __init__(self, seq_len: int = 5):
        self._seq_len = seq_len
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=True,
        max_length=None,
        padding=False,
    ):
        batch = 1 if isinstance(text, str) else len(text)
        input_ids = torch.randint(0, 1000, (batch, self._seq_len))
        attention_mask = torch.ones(batch, self._seq_len)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture(scope="module")
def all_records() -> list[ProbeRecord]:
    return build_probe_records()


@pytest.fixture(scope="module")
def target_ids() -> list[int]:
    return load_target_task_ids()


def _skip_if_no_data():
    """Skip if python_syntax_pairs.json is not built."""
    try:
        load_target_task_ids()
    except (FileNotFoundError, KeyError):
        pytest.skip("python_syntax_pairs.json not built")


# =============================================================================
# 8 x 50 balance
# =============================================================================


class TestRecordBalance:
    """Exactly 400 records, 50 per class, protocol=raw."""

    def test_total_is_four_hundred(self, all_records):
        assert len(all_records) == 400

    def test_fifty_per_class(self, all_records):
        counts: dict[str, int] = {}
        for r in all_records:
            counts[r.label] = counts.get(r.label, 0) + 1
        assert set(counts) == set(PROBE_CLASSES)
        for label in PROBE_CLASSES:
            assert counts[label] == N_SAMPLES, label

    def test_protocol_is_raw(self, all_records):
        assert all(r.protocol == "raw" for r in all_records)

    def test_all_labels_are_valid_probe_classes(self, all_records):
        for r in all_records:
            assert r.label in PROBE_CLASSES, r

    def test_sample_ids_are_unique(self, all_records):
        sids = [r.sample_id for r in all_records]
        assert len(set(sids)) == len(sids)

    def test_no_empty_texts(self, all_records):
        for r in all_records:
            assert r.text.strip(), f"empty text for {r.sample_id}"

    def test_probe_classes_match_experiment_config(self):
        assert PROBE_CLASSES is EXP_PROBE_CLASSES


# =============================================================================
# Cross-language group alignment
# =============================================================================


class TestCodeGroupAlignment:
    """All six code classes share the same 50 target task IDs, grouped."""

    def test_code_records_use_target_ids(self, all_records, target_ids):
        code_records = [r for r in all_records if r.label in CODE_PROBE_CLASSES]
        source_ids = sorted(int(r.source_id) for r in code_records)
        # Each target ID appears 6 times (6 code classes).
        assert source_ids == sorted(target_ids * len(CODE_PROBE_CLASSES))

    def test_each_code_group_has_six_variants(self, all_records):
        groups = group_records(all_records)
        for gid, members in groups.items():
            if gid.startswith("code:"):
                labels = sorted(m.label for m in members)
                assert labels == sorted(CODE_PROBE_CLASSES), gid

    def test_code_group_members_share_source_id(self, all_records):
        groups = group_records(all_records)
        for gid, members in groups.items():
            if gid.startswith("code:"):
                ids = {m.source_id for m in members}
                assert len(ids) == 1, f"{gid} has mixed source IDs: {ids}"
                # group_id suffix matches source_id.
                assert gid == f"code:{members[0].source_id}"

    def test_target_ids_match_python_syntax_pairs(self, target_ids):
        """The code classes must use the SAME IDs as python_syntax_pairs.json."""
        from postdyn.dataset_store import load_dataset_json, PYTHON_SYNTAX_PAIRS_FILE

        data = load_dataset_json(PYTHON_SYNTAX_PAIRS_FILE)
        pairs_ids = sorted(int(i) for i in data["selection"]["target_ids"])
        assert target_ids == pairs_ids

    def test_non_python_code_loaded_from_humaneval_x(self, all_records, target_ids):
        """cpp/js/java/go texts must match humaneval_x.json for the task ID."""
        from postdyn.dataset_store import load_dataset_json, HUMANEVAL_X_FILE

        hx = load_dataset_json(HUMANEVAL_X_FILE)
        hx_by_lang: dict[str, dict[int, str]] = {}
        for lang, items in hx["languages"].items():
            by_id: dict[int, str] = {}
            for item in items:
                nid = int(item["numeric_id"])
                code = str(item.get("code") or "")
                if "```" in code:
                    code = "\n".join(
                        ln
                        for ln in code.splitlines()
                        if not ln.lstrip().startswith("```")
                    )
                by_id[nid] = code
            hx_by_lang[lang] = by_id

        lang_map = {"cpp": "cpp", "js": "js", "java": "java", "go": "go"}
        for r in all_records:
            if r.label not in lang_map:
                continue
            tid = int(r.source_id)
            lang = lang_map[r.label]
            assert tid in hx_by_lang[lang], f"{r.label} task {tid} not in humaneval_x"
            assert r.text == hx_by_lang[lang][tid], r.sample_id

    def test_python_valid_and_error_from_pairs_file(self, all_records, target_ids):
        """python_valid / python_syntax_error texts must come from the pairs file."""
        from postdyn.dataset_store import load_dataset_json, PYTHON_SYNTAX_PAIRS_FILE

        data = load_dataset_json(PYTHON_SYNTAX_PAIRS_FILE)
        pairs_by_id: dict[int, dict[str, str]] = {}
        for item in data["items"]:
            pairs_by_id[int(item["numeric_id"])] = {
                "positive": str(item["positive"]),
                "negative": str(item["negative"]),
            }

        for r in all_records:
            if r.label == "python_valid":
                tid = int(r.source_id)
                assert r.text == pairs_by_id[tid]["positive"], r.sample_id
            elif r.label == "python_syntax_error":
                tid = int(r.source_id)
                assert r.text == pairs_by_id[tid]["negative"], r.sample_id

    def test_code_groups_are_fifty(self, all_records):
        groups = group_records(all_records)
        code_groups = [g for g in groups if g.startswith("code:")]
        assert len(code_groups) == 50


# =============================================================================
# Gender pairing
# =============================================================================


class TestGenderPairing:
    """she and he come from the same WinoGender template, sharing group_id."""

    def test_gender_records_are_one_hundred(self, all_records):
        gender = [r for r in all_records if r.label in GENDER_PROBE_CLASSES]
        assert len(gender) == 100

    def test_fifty_gender_groups(self, all_records):
        groups = group_records(all_records)
        gender_groups = [g for g in groups if g.startswith("gender:")]
        assert len(gender_groups) == 50

    def test_each_gender_group_has_she_and_he(self, all_records):
        groups = group_records(all_records)
        for gid, members in groups.items():
            if gid.startswith("gender:"):
                labels = sorted(m.label for m in members)
                assert labels == ["he", "she"], gid

    def test_she_and_he_share_group_and_template(self, all_records):
        import re

        groups = group_records(all_records)
        for gid, members in groups.items():
            if not gid.startswith("gender:"):
                continue
            she = next(m for m in members if m.label == "she")
            he = next(m for m in members if m.label == "he")
            # Same template index.
            assert she.source_id == he.source_id, gid
            # Texts differ only in the pronoun (word-boundary swap).
            assert re.sub(r"\bshe\b", "he", she.text) == he.text, gid

    def test_she_text_contains_she_pronoun(self, all_records):
        for r in all_records:
            if r.label == "she":
                assert " she " in f" {r.text} " or r.text.startswith("she "), (
                    r.sample_id
                )

    def test_he_text_contains_he_pronoun(self, all_records):
        for r in all_records:
            if r.label == "he":
                assert " he " in f" {r.text} " or r.text.startswith("he "), r.sample_id


# =============================================================================
# No group leakage
# =============================================================================


class TestNoGroupLeakage:
    """Groups are disjoint: every record belongs to exactly one group."""

    def test_every_record_in_a_group(self, all_records):
        groups = group_records(all_records)
        grouped = sum(len(v) for v in groups.values())
        assert grouped == len(all_records)

    def test_code_groups_have_exactly_six(self, all_records):
        groups = group_records(all_records)
        for gid, members in groups.items():
            if gid.startswith("code:"):
                assert len(members) == 6, f"{gid} has {len(members)}"

    def test_gender_groups_have_exactly_two(self, all_records):
        groups = group_records(all_records)
        for gid, members in groups.items():
            if gid.startswith("gender:"):
                assert len(members) == 2, f"{gid} has {len(members)}"

    def test_no_record_in_two_groups(self, all_records):
        groups = group_records(all_records)
        seen: set[str] = set()
        for members in groups.values():
            for m in members:
                assert m.sample_id not in seen, f"{m.sample_id} in two groups"
                seen.add(m.sample_id)

    def test_code_and_gender_groups_disjoint_by_prefix(self, all_records):
        groups = group_records(all_records)
        for gid in groups:
            assert gid.startswith("code:") or gid.startswith("gender:"), gid

    def test_code_group_ids_are_unique_per_task(self, all_records):
        groups = group_records(all_records)
        code_gids = [g for g in groups if g.startswith("code:")]
        suffixes = [g.split(":", 1)[1] for g in code_gids]
        assert len(set(suffixes)) == len(suffixes)

    def test_gender_group_ids_are_unique_per_template(self, all_records):
        groups = group_records(all_records)
        gender_gids = [g for g in groups if g.startswith("gender:")]
        suffixes = [g.split(":", 1)[1] for g in gender_gids]
        assert len(set(suffixes)) == len(suffixes)


# =============================================================================
# Validation
# =============================================================================


class TestValidateProbeRecords:
    def test_valid_records_pass(self, all_records):
        validate_probe_records(all_records)

    def test_wrong_count_raises(self, all_records):
        with pytest.raises(ValueError, match="expected 400"):
            validate_probe_records(all_records[:399])

    def test_wrong_protocol_raises(self, all_records):
        bad = [
            ProbeRecord(
                sample_id=r.sample_id,
                label=r.label,
                text=r.text,
                group_id=r.group_id,
                source_id=r.source_id,
                protocol="chat",
            )
            for r in all_records
        ]
        with pytest.raises(ValueError, match="protocol"):
            validate_probe_records(bad)

    def test_duplicate_sample_id_raises(self, all_records):
        bad = list(all_records)
        bad[0] = ProbeRecord(
            sample_id=bad[1].sample_id,
            label=bad[0].label,
            text=bad[0].text,
            group_id=bad[0].group_id,
            source_id=bad[0].source_id,
        )
        with pytest.raises(ValueError, match="duplicate sample IDs"):
            validate_probe_records(bad)


# =============================================================================
# Records JSON roundtrip
# =============================================================================


class TestRecordsJsonRoundtrip:
    def test_save_and_load_records(self, tmp_path, all_records):
        path = save_records_json(str(tmp_path), all_records)
        assert os.path.exists(path)
        loaded = load_records_json(str(tmp_path))
        assert len(loaded) == len(all_records)
        for orig, got in zip(all_records, loaded):
            assert got.sample_id == orig.sample_id
            assert got.label == orig.label
            assert got.text == orig.text
            assert got.group_id == orig.group_id
            assert got.source_id == orig.source_id
            assert got.protocol == orig.protocol

    def test_records_json_has_protocol_raw(self, tmp_path, all_records):
        save_records_json(str(tmp_path), all_records)
        with open(os.path.join(str(tmp_path), "records.json")) as f:
            data = json.load(f)
        assert data["protocol"] == "raw"
        assert data["n_records"] == 400

    def test_load_missing_records_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_records_json(str(tmp_path))


# =============================================================================
# Safetensors + JSON roundtrip
# =============================================================================


class TestLayerActivationsRoundtrip:
    def test_save_and_load_layer(self, tmp_path, all_records):
        n, d = 400, 16
        activations = torch.randn(n, d, dtype=torch.float32)
        base = save_layer_activations(
            str(tmp_path),
            "test_model",
            "step_1",
            3,
            activations,
            all_records,
        )
        assert os.path.exists(base + ".safetensors")
        assert os.path.exists(base + ".json")

        loaded_acts, sidecar = load_layer_activations(
            str(tmp_path), "test_model", "step_1", 3
        )
        assert loaded_acts.shape == (n, d)
        assert torch.allclose(loaded_acts, activations)
        assert sidecar["n_records"] == n
        assert sidecar["d_model"] == d
        assert sidecar["layer_idx"] == 3
        assert sidecar["protocol"] == "raw"
        assert sidecar["model_name"] == "test_model"
        assert sidecar["checkpoint"] == "step_1"
        assert len(sidecar["sample_ids"]) == n
        assert sidecar["sample_ids"] == [r.sample_id for r in all_records]

    def test_save_converts_to_float32_cpu(self, tmp_path, all_records):
        activations = torch.randn(400, 8, dtype=torch.float64)
        save_layer_activations(str(tmp_path), "m", "c", 0, activations, all_records)
        loaded, _ = load_layer_activations(str(tmp_path), "m", "c", 0)
        assert loaded.dtype == torch.float32

    def test_save_wrong_row_count_raises(self, tmp_path, all_records):
        activations = torch.randn(399, 8)
        with pytest.raises(ValueError, match="activations rows"):
            save_layer_activations(str(tmp_path), "m", "c", 0, activations, all_records)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_layer_activations(str(tmp_path), "m", "c", 0)

    def test_is_layer_complete(self, tmp_path, all_records):
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 400) is False
        activations = torch.randn(400, 8)
        save_layer_activations(str(tmp_path), "m", "c", 3, activations, all_records)
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 400) is True
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 399) is False

    def test_sidecar_labels_and_group_ids_match(self, tmp_path, all_records):
        activations = torch.randn(400, 8)
        save_layer_activations(str(tmp_path), "m", "c", 3, activations, all_records)
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 3)
        assert sidecar["labels"] == [r.label for r in all_records]
        assert sidecar["group_ids"] == [r.group_id for r in all_records]


# =============================================================================
# Mock activation extraction
# =============================================================================


class TestMockExtraction:
    """Extract activations with a mock model (no real model, no chat template)."""

    def test_extraction_returns_correct_shape(self, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        layers = [0, 1, 3]
        acts = extract_probe_activations(all_records, model, tokenizer, layers)
        assert set(acts.keys()) == set(layers)
        for ly in layers:
            assert acts[ly].shape == (400, 8)
            assert acts[ly].dtype == torch.float32

    def test_extraction_is_deterministic(self, all_records):
        """Same inputs produce same activations (mock seed is stable per call)."""
        model = MockModel(n_layers=2, d_model=4)
        tokenizer = MockTokenizer(seq_len=3)
        acts1 = extract_probe_activations(all_records[:10], model, tokenizer, [1])
        acts2 = extract_probe_activations(all_records[:10], model, tokenizer, [1])
        assert torch.allclose(acts1[1], acts2[1])

    def test_extraction_passes_use_chat_template_false(self, all_records, monkeypatch):
        """Verify extract_probe_activations always uses use_chat_template=False."""
        captured: dict = {}

        from postdyn import probe_activations as pa_mod

        original = pa_mod.extract_layer_activations

        def spy(model, tokenizer, texts, layers, **kwargs):
            captured["use_chat_template"] = kwargs.get("use_chat_template")
            captured["texts_len"] = len(texts)
            return {ly: torch.empty(0, 4) for ly in layers}

        monkeypatch.setattr(pa_mod, "extract_layer_activations", spy)
        try:
            extract_probe_activations(
                all_records[:5], MockModel(), MockTokenizer(), [0]
            )
        finally:
            pa_mod.extract_layer_activations = original

        assert captured["use_chat_template"] is False
        assert captured["texts_len"] == 5


# =============================================================================
# Resume runner
# =============================================================================


class TestRunExtractionResume:
    """run_extraction_with_resume skips completed layers and writes manifests."""

    def test_full_run_extracts_all_layers(self, tmp_path, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        result = run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1, 3],
            str(tmp_path),
            "test_model",
            "step_1",
        )
        assert result["extracted"] == [0, 1, 3]
        assert result["skipped"] == []
        for ly in [0, 1, 3]:
            assert is_layer_complete(str(tmp_path), "test_model", "step_1", ly, 400)

    def test_resume_skips_completed_layers(self, tmp_path, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)

        # First run: extract layers 0 and 1.
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
        )
        assert is_layer_complete(str(tmp_path), "m", "c", 0, 400)
        assert is_layer_complete(str(tmp_path), "m", "c", 1, 400)

        # Second run: request 0, 1, 3 -- 0 and 1 should be skipped.
        result = run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1, 3],
            str(tmp_path),
            "m",
            "c",
        )
        assert result["extracted"] == [3]
        assert sorted(result["skipped"]) == [0, 1]

    def test_manifest_tracks_completed_layers(self, tmp_path, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 2],
            str(tmp_path),
            "m",
            "c",
        )
        manifest = load_manifest(str(tmp_path), "m", "c")
        assert manifest["completed_layers"] == [0, 2]
        assert manifest["n_records"] == 400
        assert manifest["protocol"] == "raw"
        assert manifest["d_model"] == 8

    def test_records_json_written_on_run(self, tmp_path, all_records):
        model = MockModel(n_layers=2, d_model=4)
        tokenizer = MockTokenizer(seq_len=3)
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
        )
        assert os.path.exists(os.path.join(str(tmp_path), "records.json"))

    def test_resume_does_not_re_extract(self, tmp_path, all_records):
        """When all layers are done, no extraction call is made."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)

        # Complete all layers.
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
        )

        # Spy: extraction should not be called.
        called: list[bool] = []

        from postdyn import probe_activations as pa_mod

        original = pa_mod.extract_probe_activations

        def spy(*args, **kwargs):
            called.append(True)
            return {0: torch.empty(400, 8)}

        pa_mod.extract_probe_activations = spy
        try:
            result = run_extraction_with_resume(
                all_records,
                model,
                tokenizer,
                [0, 1],
                str(tmp_path),
                "m",
                "c",
            )
        finally:
            pa_mod.extract_probe_activations = original

        assert called == []
        assert sorted(result["skipped"]) == [0, 1]
        assert result["extracted"] == []


# =============================================================================
# Results-root isolation
# =============================================================================


class TestResultsRootIsolation:
    def test_default_root_under_rl_zero_code_syntax(self):
        root = default_activations_root()
        assert root.startswith(RL_ZERO_CODE_RESULTS_ROOT)
        assert "concept_dynamics_multi" not in root

    def test_default_root_has_activations_suffix(self):
        root = default_activations_root()
        assert root.endswith("activations")


# =============================================================================
# Records fingerprint (deterministic SHA-256 over ordered identity)
# =============================================================================


def _fake_records(n: int = 5, prefix: str = "r") -> list[ProbeRecord]:
    """Build ``n`` lightweight records (no data files needed)."""
    return [
        ProbeRecord(
            sample_id=f"{prefix}:{i}",
            label="python_valid",
            text=f"text {i}",
            group_id=f"code:{i}",
            source_id=str(i),
        )
        for i in range(n)
    ]


class TestRecordsFingerprint:
    """compute_records_fingerprint is deterministic, order-sensitive, and
    captures sample_id, label, group_id, source_id, and text content."""

    def test_returns_hex_sha256(self):
        fp = compute_records_fingerprint(_fake_records(3))
        assert isinstance(fp, str)
        assert len(fp) == 64
        int(fp, 16)  # valid hex

    def test_deterministic_same_records(self):
        records = _fake_records(5)
        assert compute_records_fingerprint(records) == compute_records_fingerprint(
            records
        )

    def test_empty_records_still_hex(self):
        fp = compute_records_fingerprint([])
        assert len(fp) == 64

    def test_changes_with_record_order(self):
        """Reordering records must change the fingerprint."""
        r = _fake_records(3)
        reordered = [r[2], r[0], r[1]]
        assert compute_records_fingerprint(r) != compute_records_fingerprint(reordered)

    def test_changes_with_sample_id(self):
        r = _fake_records(3)
        modified = [
            ProbeRecord(
                sample_id="CHANGED",
                label=r[0].label,
                text=r[0].text,
                group_id=r[0].group_id,
                source_id=r[0].source_id,
            ),
            *r[1:],
        ]
        assert compute_records_fingerprint(r) != compute_records_fingerprint(modified)

    def test_changes_with_label(self):
        r = _fake_records(3)
        modified = [
            ProbeRecord(
                sample_id=r[0].sample_id,
                label="CHANGED",
                text=r[0].text,
                group_id=r[0].group_id,
                source_id=r[0].source_id,
            ),
            *r[1:],
        ]
        assert compute_records_fingerprint(r) != compute_records_fingerprint(modified)

    def test_changes_with_group_id(self):
        r = _fake_records(3)
        modified = [
            ProbeRecord(
                sample_id=r[0].sample_id,
                label=r[0].label,
                text=r[0].text,
                group_id="CHANGED",
                source_id=r[0].source_id,
            ),
            *r[1:],
        ]
        assert compute_records_fingerprint(r) != compute_records_fingerprint(modified)

    def test_changes_with_source_id(self):
        r = _fake_records(3)
        modified = [
            ProbeRecord(
                sample_id=r[0].sample_id,
                label=r[0].label,
                text=r[0].text,
                group_id=r[0].group_id,
                source_id="CHANGED",
            ),
            *r[1:],
        ]
        assert compute_records_fingerprint(r) != compute_records_fingerprint(modified)

    def test_changes_with_text_content(self):
        """Modifying just the text (not other identity fields) must change the fp."""
        r = _fake_records(3)
        modified = [
            ProbeRecord(
                sample_id=r[0].sample_id,
                label=r[0].label,
                text="COMPLETELY DIFFERENT TEXT",
                group_id=r[0].group_id,
                source_id=r[0].source_id,
            ),
            *r[1:],
        ]
        assert compute_records_fingerprint(r) != compute_records_fingerprint(modified)

    def test_stable_across_extra_record_fields(self):
        """Fingerprint must not change if a record's protocol field changes
        (it is derived, not identity)."""
        r = _fake_records(3)
        modified_protocol = [
            ProbeRecord(
                sample_id=rec.sample_id,
                label=rec.label,
                text=rec.text,
                group_id=rec.group_id,
                source_id=rec.source_id,
                protocol="chat",
            )
            for rec in r
        ]
        assert compute_records_fingerprint(r) == compute_records_fingerprint(
            modified_protocol
        )


class TestRecordsJsonFingerprint:
    """save_records_json writes the fingerprint into records.json."""

    def test_records_json_has_fingerprint(self, tmp_path):
        records = _fake_records(5)
        save_records_json(str(tmp_path), records)
        with open(os.path.join(str(tmp_path), "records.json")) as f:
            data = json.load(f)
        expected = compute_records_fingerprint(records)
        assert data["records_fingerprint"] == expected

    def test_records_json_fingerprint_matches_load(self, tmp_path, all_records):
        save_records_json(str(tmp_path), all_records)
        loaded = load_records_json(str(tmp_path))
        expected = compute_records_fingerprint(all_records)
        assert compute_records_fingerprint(loaded) == expected


# =============================================================================
# Hardened is_layer_complete (strict validation)
# =============================================================================


class TestIsLayerCompleteStrict:
    """is_layer_complete with expected_* kwargs validates all identity fields."""

    def test_strict_pass_when_all_match(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert is_layer_complete(
            str(tmp_path),
            "m",
            "c",
            3,
            10,
            expected_d_model=8,
            expected_max_seq_len=512,
            expected_protocol="raw",
            expected_records_fingerprint=compute_records_fingerprint(records),
        )

    def test_rejects_wrong_max_seq_len(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_max_seq_len=2048)
            is False
        )

    def test_rejects_when_sidecar_lacks_max_seq_len(self, tmp_path):
        """Old-style sidecar (no max_seq_len) rejected when expected_max_seq_len given."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 3, acts, records)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_max_seq_len=512)
            is False
        )

    def test_rejects_wrong_protocol(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_protocol="chat")
            is False
        )

    def test_rejects_wrong_d_model(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_d_model=16)
            is False
        )

    def test_rejects_wrong_model_name(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_model_name="WRONG",
            )
            is False
        )

    def test_rejects_wrong_checkpoint(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_checkpoint="WRONG",
            )
            is False
        )

    def test_rejects_wrong_layer_idx(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_layer_idx=99)
            is False
        )

    def test_rejects_wrong_records_fingerprint(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records_fingerprint="0" * 64,
            )
            is False
        )

    def test_rejects_when_sidecar_lacks_fingerprint(self, tmp_path):
        """Old-style sidecar (no fingerprint field) rejected when expected given."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        # Manually strip the fingerprint to simulate pre-hardening sidecar.
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        del sidecar["records_fingerprint"]
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records_fingerprint=compute_records_fingerprint(records),
            )
            is False
        )

    def test_rejects_stale_tensor_shape(self, tmp_path):
        """Sidecar says (10, 8) but safetensors is (10, 4) — stale tensor."""
        records = _fake_records(10)
        # Write a valid layer file first.
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        # Overwrite safetensors with a different shape (simulating staleness).
        from safetensors.torch import save_file as _save_file

        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        _save_file({"activations": torch.randn(10, 4)}, base + ".safetensors")
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_d_model=8,
            )
            is False
        )

    def test_rejects_malformed_sidecar_json(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        # Corrupt the JSON sidecar.
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        with open(base + ".json", "w") as f:
            f.write("{NOT VALID JSON")
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 10) is False

    def test_rejects_missing_safetensors(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        os.remove(base + ".safetensors")
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 10) is False

    def test_backward_compat_count_only(self, tmp_path):
        """Old-style call (only expected_n_records) still works."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 3, acts, records)
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 10) is True
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 20) is False


class TestSaveLayerActivationsMetadata:
    """save_layer_activations writes max_seq_len + fingerprint into the sidecar."""

    def test_sidecar_has_max_seq_len_when_provided(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=2048
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 3)
        assert sidecar["max_seq_len"] == 2048

    def test_sidecar_has_fingerprint_when_provided(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 3)
        assert sidecar["records_fingerprint"] == compute_records_fingerprint(records)

    def test_sidecar_has_source_ids(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 3)
        assert sidecar["source_ids"] == [r.source_id for r in records]

    def test_sidecar_max_seq_len_null_when_not_provided(self, tmp_path):
        """Backward compat: max_seq_len is null when not passed."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 3, acts, records)
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 3)
        assert sidecar.get("max_seq_len") is None

    def test_atomic_write_no_tmp_left(self, tmp_path):
        """No .tmp files remain after save."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        for dirpath, _, filenames in os.walk(str(tmp_path)):
            for fn in filenames:
                assert not fn.endswith(".tmp"), f"leftover temp file: {fn}"


# =============================================================================
# Hardened run_extraction_with_resume (threads max_seq_len + fingerprint)
# =============================================================================


class TestRunExtractionResumeHardened:
    """run_extraction_with_resume validates max_seq_len and records identity."""

    def test_run_writes_max_seq_len_to_sidecars(self, tmp_path, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=2048,
        )
        for ly in [0, 1]:
            _, sidecar = load_layer_activations(str(tmp_path), "m", "c", ly)
            assert sidecar["max_seq_len"] == 2048

    def test_run_writes_fingerprint_to_sidecars(self, tmp_path, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 0)
        expected = compute_records_fingerprint(all_records)
        assert sidecar["records_fingerprint"] == expected

    def test_run_manifest_has_max_seq_len_and_fingerprint(self, tmp_path, all_records):
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=2048,
        )
        manifest = load_manifest(str(tmp_path), "m", "c")
        assert manifest["max_seq_len"] == 2048
        assert manifest["records_fingerprint"] == compute_records_fingerprint(
            all_records
        )

    def test_resume_rejects_max_seq_len_change(self, tmp_path, all_records):
        """First run with max_seq_len=512, second with 2048 → re-extract."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)

        # First run.
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
        )

        # Second run with different max_seq_len → must re-extract.
        result = run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=2048,
        )
        assert 0 in result["extracted"]
        assert 0 not in result["skipped"]

    def test_resume_rejects_records_change(self, tmp_path, all_records):
        """Changing record texts → fingerprint mismatch → re-extract."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)

        # First run.
        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
        )

        # Modify one record's text.
        modified = list(all_records)
        modified[0] = ProbeRecord(
            sample_id=modified[0].sample_id,
            label=modified[0].label,
            text="COMPLETELY DIFFERENT",
            group_id=modified[0].group_id,
            source_id=modified[0].source_id,
        )
        # The modified list has 400 records but is no longer valid (class
        # balance may break).  We bypass validation by patching.
        result = run_extraction_with_resume(
            modified,
            model,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
        )
        assert 0 in result["extracted"]

    def test_resume_passes_when_unchanged(self, tmp_path, all_records):
        """Same max_seq_len + same records → skip."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)

        run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=2048,
        )
        result = run_extraction_with_resume(
            all_records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=2048,
        )
        assert result["extracted"] == []
        assert sorted(result["skipped"]) == [0, 1]

    def test_resume_rejects_d_model_change(self, tmp_path, all_records):
        """If d_model changes between runs, the old layers are re-extracted."""
        model8 = MockModel(n_layers=4, d_model=8)
        model16 = MockModel(n_layers=4, d_model=16)
        tokenizer = MockTokenizer(seq_len=5)

        run_extraction_with_resume(
            all_records,
            model8,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
            expected_d_model=8,
        )
        result = run_extraction_with_resume(
            all_records,
            model16,
            tokenizer,
            [0],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
            expected_d_model=16,
        )
        assert 0 in result["extracted"]


# =============================================================================
# Per-record text provenance helpers
# =============================================================================


class TestRecordTextSha256:
    """record_text_sha256 is the reusable single-record text hash."""

    def test_returns_hex_sha256(self):
        r = _fake_records(1)[0]
        h = record_text_sha256(r)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_matches_sha256_of_text_bytes(self):
        import hashlib

        r = _fake_records(1)[0]
        assert (
            record_text_sha256(r) == hashlib.sha256(r.text.encode("utf-8")).hexdigest()
        )

    def test_changes_with_text_content(self):
        a = _fake_records(1)[0]
        b = ProbeRecord(
            sample_id=a.sample_id,
            label=a.label,
            text="COMPLETELY DIFFERENT TEXT",
            group_id=a.group_id,
            source_id=a.source_id,
        )
        assert record_text_sha256(a) != record_text_sha256(b)

    def test_stable_for_identical_text(self):
        a = _fake_records(1)[0]
        b = ProbeRecord(
            sample_id="other:id",
            label="other_label",
            text=a.text,
            group_id="other:group",
            source_id="other_source",
        )
        # Only the text matters for the text hash.
        assert record_text_sha256(a) == record_text_sha256(b)


class TestValidateSidecarRecordIdentity:
    """validate_sidecar_record_identity recomputes identity from records and
    rejects absent or mismatched provenance."""

    def test_accepts_matching_sidecar(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 0)
        assert validate_sidecar_record_identity(sidecar, records) is True

    def test_rejects_when_text_sha256_absent(self, tmp_path):
        """Old migrated sidecar (no text_sha256) must be rejected."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        del sidecar["text_sha256"]
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_when_source_ids_absent(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        del sidecar["source_ids"]
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_when_fingerprint_absent(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        del sidecar["records_fingerprint"]
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_mismatched_sample_ids(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["sample_ids"] = list(reversed(sidecar["sample_ids"]))
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_mismatched_labels(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["labels"] = ["x"] * 10
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_mismatched_group_ids(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["group_ids"] = ["g"] * 10
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_mismatched_source_ids(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["source_ids"] = [str(100 + i) for i in range(10)]
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_mismatched_text_sha256(self, tmp_path):
        """Tampering with text_sha256 (but not fingerprint) is still caught."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["text_sha256"] = ["0" * 64] * 10
        assert validate_sidecar_record_identity(sidecar, records) is False

    def test_rejects_wrong_record_count(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 0)
        # Fewer records than the sidecar describes.
        assert validate_sidecar_record_identity(sidecar, records[:5]) is False

    def test_rejects_mismatched_fingerprint(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["records_fingerprint"] = "1" * 64
        assert validate_sidecar_record_identity(sidecar, records) is False


# =============================================================================
# Sidecar / records.json text provenance persistence
# =============================================================================


class TestSidecarTextProvenance:
    """save_layer_activations and save_records_json persist text_sha256 and
    ordered source_ids, binding extraction-time text provenance."""

    def test_sidecar_has_text_sha256(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 3)
        assert "text_sha256" in sidecar
        assert sidecar["text_sha256"] == [record_text_sha256(r) for r in records]

    def test_sidecar_text_sha256_order_matches_records(self, tmp_path):
        records = _fake_records(12)
        acts = torch.randn(12, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 0)
        for r, h in zip(records, sidecar["text_sha256"]):
            assert h == record_text_sha256(r)

    def test_sidecar_source_ids_order_matches_records(self, tmp_path):
        records = _fake_records(7)
        acts = torch.randn(7, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        _, sidecar = load_layer_activations(str(tmp_path), "m", "c", 0)
        assert sidecar["source_ids"] == [r.source_id for r in records]

    def test_records_json_has_ordered_arrays(self, tmp_path):
        records = _fake_records(9)
        save_records_json(str(tmp_path), records)
        with open(os.path.join(str(tmp_path), "records.json")) as f:
            data = json.load(f)
        assert data["sample_ids"] == [r.sample_id for r in records]
        assert data["labels"] == [r.label for r in records]
        assert data["group_ids"] == [r.group_id for r in records]
        assert data["source_ids"] == [r.source_id for r in records]
        assert data["text_sha256"] == [record_text_sha256(r) for r in records]
        assert data["records_fingerprint"] == compute_records_fingerprint(records)

    def test_records_json_load_preserves_records(self, tmp_path):
        """Adding top-level arrays must not break load_records_json."""
        records = _fake_records(6)
        save_records_json(str(tmp_path), records)
        loaded = load_records_json(str(tmp_path))
        assert [r.text for r in loaded] == [r.text for r in records]
        assert compute_records_fingerprint(loaded) == compute_records_fingerprint(
            records
        )


# =============================================================================
# Hardened is_layer_complete with expected_records (strict identity recompute)
# =============================================================================


def _strip_text_sha256(sidecar_path: str) -> None:
    """Helper: remove text_sha256 from a sidecar on disk (simulate old artifact)."""
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    del sidecar["text_sha256"]
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f)


class TestIsLayerCompleteExpectedRecords:
    """is_layer_complete with expected_records recomputes identity from
    ProbeRecords and rejects absent or mismatched provenance."""

    def test_accepts_when_records_match(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_records=records)
            is True
        )

    def test_rejects_old_sidecar_without_text_sha256(self, tmp_path):
        """A migrated sidecar lacking text_sha256 is incomplete -> re-extract."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        _strip_text_sha256(base + ".json")
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_records=records)
            is False
        )

    def test_rejects_when_source_ids_absent(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        del sidecar["source_ids"]
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_records=records)
            is False
        )

    def test_rejects_wrong_record_order(self, tmp_path):
        """Reordering records (same set) must fail because order is part of identity."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        reordered = list(reversed(records))
        assert (
            is_layer_complete(
                str(tmp_path), "m", "c", 3, 10, expected_records=reordered
            )
            is False
        )

    def test_rejects_wrong_protocol(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records=records,
                expected_protocol="chat",
            )
            is False
        )

    def test_rejects_wrong_model_name(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records=records,
                expected_model_name="WRONG",
            )
            is False
        )

    def test_rejects_wrong_checkpoint(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records=records,
                expected_checkpoint="WRONG",
            )
            is False
        )

    def test_rejects_wrong_layer_idx(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records=records,
                expected_layer_idx=99,
            )
            is False
        )

    def test_rejects_wrong_d_model(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                3,
                10,
                expected_records=records,
                expected_d_model=16,
            )
            is False
        )

    def test_rejects_rank1_tensor(self, tmp_path):
        """A rank-1 tensor must be rejected under expected_records (requires rank-2)."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        # Overwrite the safetensors with a rank-1 tensor of matching row count.
        from safetensors.torch import save_file as _save_file

        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        _save_file({"activations": torch.randn(10)}, base + ".safetensors")
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_records=records)
            is False
        )

    def test_rejects_stale_d_model_in_tensor(self, tmp_path):
        """Sidecar says d=8 but tensor is d=4 -> rejected (rank-2 but wrong shape)."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        from safetensors.torch import save_file as _save_file

        base = os.path.join(str(tmp_path), "m", "c", "layer_3")
        _save_file({"activations": torch.randn(10, 4)}, base + ".safetensors")
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 3, 10, expected_records=records)
            is False
        )

    def test_backward_compat_count_only_still_works(self, tmp_path):
        """Without expected_records, the count-only check still accepts a valid layer."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 3, acts, records)
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 10) is True
        assert is_layer_complete(str(tmp_path), "m", "c", 3, 20) is False


# =============================================================================
# Atomic publication order (tensor before sidecar) + failure cleanup
# =============================================================================


class TestAtomicPublicationOrder:
    """save_layer_activations never exposes a JSON sidecar before its tensor,
    uses secure unique temp paths, and cleans up on failure."""

    def test_no_temp_files_after_save(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )
        for dirpath, _, filenames in os.walk(str(tmp_path)):
            for fn in filenames:
                assert ".tmp" not in fn, f"leftover temp file: {fn}"

    def test_tensor_published_before_sidecar(self, tmp_path, monkeypatch):
        """Interpose between the two os.replace calls and assert the tensor is
        already on disk when the sidecar lands."""
        import postdyn.probe_activations as pa

        events: list[str] = []
        real_replace = os.replace

        def tracked_replace(src, dst):
            events.append(f"replace:{os.path.basename(dst)}")
            return real_replace(src, dst)

        monkeypatch.setattr(pa.os, "replace", tracked_replace)

        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
        )

        # The first publication must be the safetensors, then the sidecar JSON.
        pub = [e for e in events if e.startswith("replace:")]
        assert pub[0].endswith(".safetensors"), pub
        assert pub[1].endswith(".json"), pub

    def test_failure_cleans_up_temp_files(self, tmp_path, monkeypatch):
        """If safetensors write fails, no temp files are left behind."""
        import postdyn.probe_activations as pa

        from safetensors.torch import save_file as _real_save

        def boom(*args, **kwargs):
            raise RuntimeError("simulated write failure")

        monkeypatch.setattr(pa, "save_file", boom)

        records = _fake_records(10)
        acts = torch.randn(10, 8)
        with pytest.raises(RuntimeError, match="simulated write failure"):
            save_layer_activations(
                str(tmp_path), "m", "c", 3, acts, records, max_seq_len=512
            )
        for dirpath, _, filenames in os.walk(str(tmp_path)):
            for fn in filenames:
                assert ".tmp" not in fn, f"leftover temp after failure: {fn}"


# =============================================================================
# Unsafe migration removed: old sidecars are rejected / re-extracted, never blessed
# =============================================================================


class TestUnsafeMigrationRemoved:
    """The CPU-only metadata migration was unsafe: it stamped records that only
    proved sample_ids/labels/groups/shape (and copied source_ids) with a
    fingerprint that implies text provenance, without proving the original text.

    Coverage of the old migration matrix is preserved but rerouted: the same
    malformed/old-style sidecars are now rejected by strict validation and
    re-extracted by run_extraction_with_resume, never blessed in place.
    """

    @pytest.fixture(autouse=True)
    def _bypass_structure_validation(self, monkeypatch):
        """These tests target provenance, not the 8-class balance, so they use
        lightweight _fake_records. run_extraction_with_resume validates the
        full 400-record structure; bypass that here to keep the tests focused."""
        import postdyn.probe_activations as pa

        monkeypatch.setattr(pa, "validate_probe_records", lambda recs: None)

    def test_migrate_extraction_metadata_symbol_removed(self):
        """The unsafe migration function is no longer exported."""
        import postdyn.probe_activations as pa

        assert not hasattr(pa, "migrate_extraction_metadata")
        assert "migrate_extraction_metadata" not in pa.__all__

    # ------------------------------------------------------------------
    # Old-style sidecars (the migration's "compatible" case) are now rejected
    # ------------------------------------------------------------------

    def test_old_style_sidecar_lacking_text_sha256_rejected(self, tmp_path):
        """Previously 'compatible' old sidecars are incomplete under strict check."""
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 0, acts, records)
        save_layer_activations(str(tmp_path), "m", "c", 1, acts, records)
        for ly in (0, 1):
            base = os.path.join(str(tmp_path), "m", "c", f"layer_{ly}")
            _strip_text_sha256(base + ".json")
        for ly in (0, 1):
            assert (
                is_layer_complete(
                    str(tmp_path), "m", "c", ly, 10, expected_records=records
                )
                is False
            )

    def test_run_extraction_re_extracts_old_style_sidecars(self, tmp_path):
        """run_extraction_with_resume re-extracts sidecars lacking text_sha256."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        records = _fake_records(10)

        # Pretend an earlier (unsafe) run wrote old-style sidecars.
        acts = torch.randn(10, 8)
        for ly in (0, 1):
            save_layer_activations(str(tmp_path), "m", "c", ly, acts, records)
            base = os.path.join(str(tmp_path), "m", "c", f"layer_{ly}")
            _strip_text_sha256(base + ".json")

        result = run_extraction_with_resume(
            records,
            model,
            tokenizer,
            [0, 1],
            str(tmp_path),
            "m",
            "c",
            max_seq_len=512,
        )
        assert sorted(result["extracted"]) == [0, 1]
        assert result["skipped"] == []
        # After re-extraction the sidecars carry text_sha256 and validate.
        for ly in (0, 1):
            _, sidecar = load_layer_activations(str(tmp_path), "m", "c", ly)
            assert "text_sha256" in sidecar
            assert is_layer_complete(
                str(tmp_path), "m", "c", ly, 10, expected_records=records
            )

    def test_resume_skips_when_provenance_complete(self, tmp_path):
        """When sidecars already carry text_sha256 and match, nothing re-extracts."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        records = _fake_records(10)

        # First run writes provenance-carrying sidecars.
        run_extraction_with_resume(
            records, model, tokenizer, [0, 1], str(tmp_path), "m", "c", max_seq_len=512
        )
        result = run_extraction_with_resume(
            records, model, tokenizer, [0, 1], str(tmp_path), "m", "c", max_seq_len=512
        )
        assert result["extracted"] == []
        assert sorted(result["skipped"]) == [0, 1]

    # ------------------------------------------------------------------
    # Re-routed rejection matrix (formerly migration-rejects cases)
    # ------------------------------------------------------------------

    def test_rejects_shape_mismatch(self, tmp_path):
        """d_model=4 on disk but caller expects d=8 -> rejected (re-extract)."""
        records = _fake_records(10)
        acts = torch.randn(10, 4)
        save_layer_activations(str(tmp_path), "m", "c", 0, acts, records)
        assert (
            is_layer_complete(
                str(tmp_path),
                "m",
                "c",
                0,
                10,
                expected_records=records,
                expected_d_model=8,
            )
            is False
        )

    def test_rejects_sample_id_mismatch(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 0, acts, records)
        different_records = _fake_records(10, prefix="OTHER")
        assert (
            is_layer_complete(
                str(tmp_path), "m", "c", 0, 10, expected_records=different_records
            )
            is False
        )

    def test_rejects_wrong_record_count(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 0, acts, records)
        fewer = _fake_records(5)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 5, expected_records=fewer)
            is False
        )

    def test_rejects_when_no_artifacts(self, tmp_path):
        """Missing entirely -> incomplete (run must re-extract)."""
        records = _fake_records(10)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )

    def test_run_rejects_missing_layer_and_re_extracts(self, tmp_path):
        """A partial checkpoint (some layers missing) is re-extracted, not blessed."""
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        records = _fake_records(10)
        # Only layer 0 exists; layer 1 is missing.
        save_layer_activations(
            str(tmp_path), "m", "c", 0, torch.randn(10, 8), records, max_seq_len=512
        )
        result = run_extraction_with_resume(
            records, model, tokenizer, [0, 1], str(tmp_path), "m", "c", max_seq_len=512
        )
        assert 1 in result["extracted"]

    def test_rejects_wrong_protocol(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["protocol"] = "chat"
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )

    def test_rejects_absent_protocol(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        del sidecar["protocol"]
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )

    def test_rejects_wrong_model_name(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["model_name"] = "WRONG"
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )

    def test_rejects_wrong_checkpoint(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["checkpoint"] = "WRONG"
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )

    def test_rejects_wrong_layer_idx(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json") as f:
            sidecar = json.load(f)
        sidecar["layer_idx"] = 99
        with open(base + ".json", "w") as f:
            json.dump(sidecar, f)
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )

    def test_rejects_malformed_sidecar_json(self, tmp_path):
        records = _fake_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(
            str(tmp_path), "m", "c", 0, acts, records, max_seq_len=512
        )
        base = os.path.join(str(tmp_path), "m", "c", "layer_0")
        with open(base + ".json", "w") as f:
            f.write("{NOT VALID JSON")
        assert (
            is_layer_complete(str(tmp_path), "m", "c", 0, 10, expected_records=records)
            is False
        )
