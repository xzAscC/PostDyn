"""Tests for experiments/run_rl_zero_syntax_extraction.py.

Covers:
  * Checkpoint selection (``--only base|rl|all``, ``--checkpoints``, ``--limit``)
  * Model resolution (base -> BASE_MODEL, RL -> TARGET_MODEL)
  * Output isolation (no writes to concept_dynamics_multi)
  * File-based resume (concept-layer completeness, probe-layer completeness)
  * Narrow checkpoint runner: single model load serves concept + probe paths
  * Atomic per-checkpoint manifest (six concepts + completed probe layers)
  * CLI parsing and end-to-end main() with mocks

No real model, GPU, or network is used. The mock model/tokenizer follow the
same HF calling convention as test_concept_dynamics.py and
test_probe_activations.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import experiments.run_rl_zero_syntax_extraction as driver
from experiments.run_rl_zero_syntax_extraction import (
    DEFAULT_MAX_SEQ_LEN,
    PAIRED_CONCEPT_RESULTS_ROOT,
    PAIRED_CONCEPT_RESULTS_ROOT_QUICK,
    PROTOCOL,
    RL_CHECKPOINTS,
    RL_ZERO_CODE_RESULTS_ROOT,
)
from src.concept_dynamics import (
    ConceptVector,
    compute_concept_vector,
    load_concept_vectors,
    save_concept_vectors,
)
from src.probe_activations import (
    ProbeRecord,
    compute_records_fingerprint,
    is_layer_complete,
    save_layer_activations,
)
from src.rl_zero_experiment import (
    BASE_CHECKPOINT,
    BASE_MODEL,
    EXPERIMENT_CHECKPOINTS,
    EXPERIMENT_CONCEPTS,
    N_SAMPLES,
    PROBE_CLASSES,
    TARGET_MODEL,
)

# =============================================================================


# =============================================================================
# Mock model / tokenizer (same convention as test_concept_dynamics.py)
# =============================================================================


class _MockModelOutput:
    """Mimics HF transformers model output with .hidden_states tuple."""

    def __init__(self, hidden_states: tuple[torch.Tensor, ...]):
        self.hidden_states = hidden_states


class MockModel:
    """Mock transformer returning deterministic hidden states.

    hidden_states is a tuple of (n_layers + 1) tensors, each (batch, seq, d).
    """

    def __init__(self, n_layers: int = 4, d_model: int = 8, seq_len: int = 5):
        self.config = type(
            "Config", (), {"num_hidden_layers": n_layers, "hidden_size": d_model}
        )()
        self.device = torch.device("cpu")
        self._n_layers = n_layers
        self._d_model = d_model
        self._seq_len = seq_len
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids=None, attention_mask=None, **kwargs):
        bs = input_ids.shape[0] if input_ids is not None else 1
        seq = input_ids.shape[1] if input_ids is not None else self._seq_len
        torch.manual_seed(hash((bs, seq, self._d_model)) & 0xFFFF)
        hidden_states = tuple(
            torch.randn(bs, seq, self._d_model) for _ in range(self._n_layers + 1)
        )
        return _MockModelOutput(hidden_states)

    def parameters(self):
        return iter([torch.zeros(1, device=self.device)])


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
# Fake factories
# =============================================================================


def _fake_concept_texts(concept: str, n_samples: int) -> tuple[list[str], list[str]]:
    """Return ``n_samples`` fake positive/negative texts for a concept."""
    pos = [f"pos {concept} {i}" for i in range(n_samples)]
    neg = [f"neg {concept} {i}" for i in range(n_samples)]
    return pos, neg


def _fake_probe_records(n: int = 10) -> list[ProbeRecord]:
    """Build ``n`` lightweight ProbeRecords (no data files needed)."""
    return [
        ProbeRecord(
            sample_id=f"fake:{i}",
            label="python_valid",
            text=f"fake text {i}",
            group_id=f"code:{i}",
            source_id=str(i),
        )
        for i in range(n)
    ]


def _make_probe_extraction_spy(d_model: int = 8):
    """Return ``(spy_fn, received_models)`` for mocking run_extraction_with_resume.

    The spy writes valid layer files (so the manifest builder sees them as
    complete) and records every model object it receives.  It forwards
    ``max_seq_len`` to ``save_layer_activations`` so the hardened resume
    checks pass.
    """
    received: list[Any] = []

    def spy(
        records,
        model,
        tokenizer,
        layers,
        output_dir,
        model_name,
        checkpoint,
        **kwargs,
    ):
        received.append(model)
        max_seq_len = kwargs.get("max_seq_len")
        for ly in layers:
            acts = torch.randn(len(records), d_model)
            save_layer_activations(
                output_dir,
                model_name,
                checkpoint,
                ly,
                acts,
                records,
                max_seq_len=max_seq_len,
            )
        return {
            "model_name": model_name,
            "checkpoint": checkpoint,
            "layers": list(layers),
            "skipped": [],
            "extracted": list(layers),
            "n_records": len(records),
            "d_model": d_model,
            "max_seq_len": max_seq_len,
            "records_fingerprint": kwargs.get("records_fingerprint"),
        }

    return spy, received


# =============================================================================
# Checkpoint selection
# =============================================================================


class TestSelectCheckpoints:
    """select_checkpoints honours --only, --checkpoints, --limit."""

    def test_only_base(self):
        assert driver.select_checkpoints(only="base") == ["main"]

    def test_only_rl(self):
        result = driver.select_checkpoints(only="rl")
        assert result == RL_CHECKPOINTS
        assert len(result) == 10

    def test_only_all(self):
        result = driver.select_checkpoints(only="all")
        assert result == list(EXPERIMENT_CHECKPOINTS)
        assert len(result) == 11
        assert result[0] == "main"

    def test_checkpoints_subset(self):
        result = driver.select_checkpoints(
            only="all", checkpoint_subset=["main", "step_100"]
        )
        assert result == ["main", "step_100"]

    def test_checkpoints_subset_rl_only(self):
        result = driver.select_checkpoints(
            only="rl", checkpoint_subset=["step_100", "step_2900"]
        )
        assert result == ["step_100", "step_2900"]

    def test_checkpoints_subset_not_in_pool(self):
        """Names not in the pool are silently ignored."""
        result = driver.select_checkpoints(only="base", checkpoint_subset=["step_100"])
        assert result == []

    def test_limit(self):
        result = driver.select_checkpoints(only="all", limit=3)
        assert len(result) == 3
        assert result == EXPERIMENT_CHECKPOINTS[:3]

    def test_limit_zero_or_negative(self):
        """Non-positive limit returns the full pool (no truncation)."""
        result = driver.select_checkpoints(only="base", limit=0)
        assert result == ["main"]

    def test_invalid_only_raises(self):
        with pytest.raises(ValueError, match="must be 'base', 'rl', or 'all'"):
            driver.select_checkpoints(only="bogus")

    def test_all_eleven_checkpoints_by_default(self):
        assert len(driver.select_checkpoints()) == 11


# =============================================================================
# Model resolution
# =============================================================================


class TestModelForCheckpoint:
    """model_for_checkpoint maps base/RL checkpoints to the right ModelConfig."""

    def test_base_returns_base_model(self):
        cfg = driver.model_for_checkpoint("main")
        assert cfg.name == BASE_MODEL.name
        assert cfg.hf_id == BASE_MODEL.hf_id

    def test_rl_returns_target_model(self):
        cfg = driver.model_for_checkpoint("step_100")
        assert cfg.name == TARGET_MODEL.name
        assert cfg.hf_id == TARGET_MODEL.hf_id

    def test_last_rl_checkpoint(self):
        cfg = driver.model_for_checkpoint("step_2900")
        assert cfg.name == TARGET_MODEL.name

    def test_invalid_checkpoint_raises(self):
        with pytest.raises(ValueError, match="unknown checkpoint"):
            driver.model_for_checkpoint("bogus")


# =============================================================================
# Output isolation
# =============================================================================


class TestOutputIsolation:
    """assert_output_isolated rejects legacy directories."""

    def test_accepts_isolated_root(self, tmp_path):
        driver.assert_output_isolated(str(tmp_path))

    def test_rejects_concept_dynamics_multi(self):
        with pytest.raises(ValueError, match="concept_dynamics_multi"):
            driver.assert_output_isolated(PAIRED_CONCEPT_RESULTS_ROOT)

    def test_rejects_concept_dynamics_multi_quick(self):
        with pytest.raises(ValueError):
            driver.assert_output_isolated(PAIRED_CONCEPT_RESULTS_ROOT_QUICK)

    def test_rejects_nested_in_legacy(self):
        nested = os.path.join(PAIRED_CONCEPT_RESULTS_ROOT, "subdir")
        with pytest.raises(ValueError, match="nested inside"):
            driver.assert_output_isolated(nested)

    def test_rejects_path_containing_concept_dynamics_multi(self):
        bad = os.path.join(
            os.path.dirname(RL_ZERO_CODE_RESULTS_ROOT), "concept_dynamics_multi_custom"
        )
        with pytest.raises(ValueError, match="concept_dynamics_multi"):
            driver.assert_output_isolated(bad)

    def test_accepts_default_rl_zero_root(self):
        driver.assert_output_isolated(RL_ZERO_CODE_RESULTS_ROOT)


# =============================================================================
# File-based resume: concept layer completeness
# =============================================================================


class TestConceptLayerComplete:
    """is_concept_layer_complete validates actual files."""

    def test_returns_false_when_missing(self, tmp_path):
        assert (
            driver.is_concept_layer_complete(str(tmp_path), "m", "c", 0, "concept_a", 5)
            is False
        )

    def test_returns_true_when_present(self, tmp_path):
        cv = _make_dummy_cv("concept_a", "m", 0, 5)
        save_concept_vectors({"concept_a": cv}, str(tmp_path), "m", 0, "c")
        assert (
            driver.is_concept_layer_complete(str(tmp_path), "m", "c", 0, "concept_a", 5)
            is True
        )

    def test_returns_false_when_wrong_concept(self, tmp_path):
        cv = _make_dummy_cv("concept_b", "m", 0, 5)
        save_concept_vectors({"concept_b": cv}, str(tmp_path), "m", 0, "c")
        assert (
            driver.is_concept_layer_complete(str(tmp_path), "m", "c", 0, "concept_a", 5)
            is False
        )

    def test_returns_false_when_insufficient_samples(self, tmp_path):
        cv = _make_dummy_cv("concept_a", "m", 0, 3)
        save_concept_vectors({"concept_a": cv}, str(tmp_path), "m", 0, "c")
        assert (
            driver.is_concept_layer_complete(str(tmp_path), "m", "c", 0, "concept_a", 5)
            is False
        )

    def test_concept_completed_layers_partial(self, tmp_path):
        for ly in [0, 2]:
            cv = _make_dummy_cv("c", "m", ly, 5)
            save_concept_vectors({"c": cv}, str(tmp_path), "m", ly, "ck")
        result = driver.concept_completed_layers(
            str(tmp_path), "m", "ck", "c", [0, 1, 2], 5
        )
        assert result == [0, 2]


def _make_dummy_cv(
    concept: str, model_name: str, layer: int, n: int, d: int = 4
) -> ConceptVector:
    """Build a dummy ConceptVector with ``n`` positive/negative samples."""
    pos = torch.randn(n, d)
    neg = torch.randn(n, d)
    return compute_concept_vector(
        pos, neg, concept_name=concept, model_name=model_name, layer_idx=layer
    )


# =============================================================================
# File-based resume: probe layer completeness
# =============================================================================


class TestProbeCompletedLayers:
    """probe_completed_layers validates actual probe files."""

    def test_returns_empty_when_nothing_written(self, tmp_path):
        result = driver.probe_completed_layers(str(tmp_path), "m", "c", [0, 1, 2], 10)
        assert result == []

    def test_returns_completed_layers(self, tmp_path):
        records = _fake_probe_records(10)
        for ly in [0, 2]:
            acts = torch.randn(10, 8)
            save_layer_activations(str(tmp_path), "m", "c", ly, acts, records)
        result = driver.probe_completed_layers(str(tmp_path), "m", "c", [0, 1, 2], 10)
        assert result == [0, 2]

    def test_wrong_record_count_excluded(self, tmp_path):
        records = _fake_probe_records(10)
        acts = torch.randn(10, 8)
        save_layer_activations(str(tmp_path), "m", "c", 0, acts, records)
        # 20 expected but only 10 written -> not complete
        result = driver.probe_completed_layers(str(tmp_path), "m", "c", [0], 20)
        assert result == []


# =============================================================================
# Regression: checkpoint-level guards must thread probe_records so old sidecars
# (matching fingerprint + max_seq_len but lacking text_sha256) are rejected.
# =============================================================================


class TestCheckpointProbeRecordsRegression:
    """Runtime QA found the checkpoint-level resume guards checked fingerprint
    + max_seq_len but did NOT pass expected_records, so the 110 migrated
    sidecars (no text_sha256) were skipped as 'complete' before
    run_extraction_with_resume could reject them. These tests pin the fix:
    probe_records threads through probe_completed_layers /
    build_checkpoint_manifest / _is_checkpoint_complete into
    is_layer_complete(expected_records=...)."""

    def _write_old_probe_sidecar(
        self,
        acts_dir,
        model_name,
        checkpoint,
        layer,
        records,
        d_model=8,
        max_seq_len=512,
    ):
        """Write a sidecar whose fingerprint + max_seq_len match but which lacks
        text_sha256 (the exact shape of the unsafe migrated artifacts)."""
        acts = torch.randn(len(records), d_model)
        save_layer_activations(
            acts_dir,
            model_name,
            checkpoint,
            layer,
            acts,
            records,
            max_seq_len=max_seq_len,
        )
        base = os.path.join(acts_dir, model_name, checkpoint, f"layer_{layer}")
        with open(base + ".json") as f:
            sc = json.load(f)
        del sc["text_sha256"]
        with open(base + ".json", "w") as f:
            json.dump(sc, f)

    def test_probe_completed_layers_rejects_old_sidecar_with_records(self, tmp_path):
        records = _fake_probe_records(10)
        fp = compute_records_fingerprint(records)
        self._write_old_probe_sidecar(
            str(tmp_path), "m", "c", 0, records, max_seq_len=512
        )

        # OLD behaviour (the gap): fingerprint + max_seq_len match -> "complete".
        buggy = driver.probe_completed_layers(
            str(tmp_path),
            "m",
            "c",
            [0],
            10,
            expected_max_seq_len=512,
            expected_records_fingerprint=fp,
            expected_protocol="raw",
        )
        assert buggy == [0]

        # FIXED behaviour: threading probe_records forces identity re-derivation,
        # which rejects the sidecar lacking text_sha256.
        fixed = driver.probe_completed_layers(
            str(tmp_path), "m", "c", [0], 10, probe_records=records
        )
        assert fixed == []

    def test_build_checkpoint_manifest_rejects_old_sidecar_with_records(self, tmp_path):
        records = _fake_probe_records(10)
        acts_dir = driver.activations_dir(str(tmp_path))
        self._write_old_probe_sidecar(acts_dir, "m", "c", 3, records, max_seq_len=512)

        # Without probe_records the manifest falsely reports the probe complete.
        manifest_unchecked = driver.build_checkpoint_manifest(
            str(tmp_path),
            "m",
            "c",
            "c",
            "hf/id",
            [],
            [3],
            5,
            10,
            max_seq_len=512,
            records_fingerprint=compute_records_fingerprint(records),
        )
        assert manifest_unchecked["probe_activations"]["completed_layers"] == [3]

        # With probe_records the manifest must report the probe layer incomplete.
        manifest_checked = driver.build_checkpoint_manifest(
            str(tmp_path),
            "m",
            "c",
            "c",
            "hf/id",
            [],
            [3],
            5,
            10,
            max_seq_len=512,
            records_fingerprint=compute_records_fingerprint(records),
            probe_records=records,
        )
        assert manifest_checked["probe_activations"]["completed_layers"] == []
        assert manifest_checked["probe_activations"]["complete"] is False

    def test_is_checkpoint_complete_false_for_old_probe_sidecar(self, tmp_path):
        records = _fake_probe_records(10)
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        cvec_dir = driver.concept_vectors_dir(str(tmp_path))
        acts_dir = driver.activations_dir(str(tmp_path))
        for ly in layers:
            all_cvs = {c: _make_dummy_cv(c, "m", ly, 5) for c in concepts}
            save_concept_vectors(all_cvs, cvec_dir, "m", ly, "c")
        self._write_old_probe_sidecar(
            acts_dir, "m", "c", layers[0], records, max_seq_len=512
        )

        # OLD: without probe_records the checkpoint looks complete -> would skip.
        assert (
            driver._is_checkpoint_complete(
                str(tmp_path),
                "m",
                "c",
                concepts,
                layers,
                5,
                10,
                max_seq_len=512,
                records_fingerprint=compute_records_fingerprint(records),
            )
            is True
        )

        # FIXED: with probe_records the checkpoint is incomplete -> must run.
        assert (
            driver._is_checkpoint_complete(
                str(tmp_path),
                "m",
                "c",
                concepts,
                layers,
                5,
                10,
                max_seq_len=512,
                records_fingerprint=compute_records_fingerprint(records),
                probe_records=records,
            )
            is False
        )

    def test_run_checkpoint_re_extracts_old_probe_sidecar(self, tmp_path, monkeypatch):
        """End-to-end: an old probe sidecar must force a model load + re-extract,
        not be skipped. Mirrors the QA scenario that surfaced the bug."""
        import src.probe_activations as pa

        # Lightweight fake records don't satisfy the 8-class structural validator
        # inside run_extraction_with_resume; bypass it to keep this regression
        # focused on provenance, not class balance.
        monkeypatch.setattr(pa, "validate_probe_records", lambda recs: None)

        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        cvec_dir = driver.concept_vectors_dir(str(tmp_path))
        acts_dir = driver.activations_dir(str(tmp_path))

        for ly in layers:
            all_cvs = {c: _make_dummy_cv(c, TARGET_MODEL.name, ly, 5) for c in concepts}
            save_concept_vectors(all_cvs, cvec_dir, TARGET_MODEL.name, ly, "step_100")
        self._write_old_probe_sidecar(
            acts_dir,
            TARGET_MODEL.name,
            "step_100",
            layers[0],
            probe_records,
            max_seq_len=512,
        )

        load_count = [0]
        model = MockModel(n_layers=4, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)

        def fake_loader(config, revision):
            load_count[0] += 1
            return model, tokenizer

        result = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            max_seq_len=512,
            model_loader=fake_loader,
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        # The model was loaded (not skipped) because the old sidecar is incomplete.
        assert load_count[0] == 1
        assert result["model_loaded"] is True
        for ly in layers:
            assert is_layer_complete(
                acts_dir,
                TARGET_MODEL.name,
                "step_100",
                ly,
                10,
                expected_records=probe_records,
            )


# =============================================================================
# Atomic per-checkpoint manifest
# =============================================================================


class TestCheckpointManifest:
    """build_checkpoint_manifest and write/load roundtrip."""

    def test_manifest_records_six_concepts(self, tmp_path):
        """Manifest must list all six experiment concepts."""
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path),
            "olmo3-rl-zero-code",
            "step_100",
            "step_100",
            "allenai/Olmo-3-7B-RL-Zero-Code",
            list(EXPERIMENT_CONCEPTS),
            [3, 6],
            N_SAMPLES,
            400,
        )
        assert set(manifest["concepts"]) == set(EXPERIMENT_CONCEPTS)
        assert len(manifest["concepts"]) == 6

    def test_manifest_records_completed_probe_layers(self, tmp_path):
        """Probe layers present on disk appear in the manifest."""
        records = _fake_probe_records(10)
        for ly in [3, 6]:
            acts = torch.randn(10, 8)
            save_layer_activations(
                driver.activations_dir(str(tmp_path)),
                "m",
                "c",
                ly,
                acts,
                records,
            )
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path), "m", "c", "c", "hf/id", [], [3, 6, 9], 5, 10
        )
        assert manifest["probe_activations"]["completed_layers"] == [3, 6]
        assert manifest["probe_activations"]["complete"] is False

    def test_manifest_complete_when_all_present(self, tmp_path):
        """Manifest reports complete=True when every concept + probe layer exists."""
        concepts = list(EXPERIMENT_CONCEPTS)
        layers = [3]
        cvec_dir = driver.concept_vectors_dir(str(tmp_path))
        for ly in layers:
            all_cvs: dict[str, ConceptVector] = {}
            for concept in concepts:
                all_cvs[concept] = _make_dummy_cv(concept, "m", ly, 5)
            save_concept_vectors(all_cvs, cvec_dir, "m", ly, "c")
        records = _fake_probe_records(10)
        for ly in layers:
            acts = torch.randn(10, 8)
            save_layer_activations(
                driver.activations_dir(str(tmp_path)), "m", "c", ly, acts, records
            )
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path), "m", "c", "c", "hf/id", concepts, layers, 5, 10
        )
        assert manifest["complete"] is True
        assert manifest["probe_activations"]["complete"] is True

    def test_write_then_load_roundtrip(self, tmp_path):
        manifest = driver.write_checkpoint_manifest(
            str(tmp_path), "m", "c", "c", "hf/id", [], [], 5, 10
        )
        loaded = driver.load_checkpoint_manifest(str(tmp_path), "m", "c")
        assert loaded["model_name"] == "m"
        assert loaded["checkpoint"] == "c"
        assert loaded["protocol"] == PROTOCOL

    def test_load_missing_returns_empty(self, tmp_path):
        assert driver.load_checkpoint_manifest(str(tmp_path), "m", "c") == {}

    def test_manifest_records_metadata(self, tmp_path):
        """Manifest preserves model/revision/hf_id/protocol metadata."""
        manifest = driver.write_checkpoint_manifest(
            str(tmp_path),
            "olmo3-rl-zero-code",
            "step_100",
            "step_100",
            "allenai/Olmo-3-7B-RL-Zero-Code",
            list(EXPERIMENT_CONCEPTS),
            [3, 6, 9],
            N_SAMPLES,
            400,
        )
        assert manifest["hf_id"] == "allenai/Olmo-3-7B-RL-Zero-Code"
        assert manifest["revision"] == "step_100"
        assert manifest["use_chat_template"] is False
        assert manifest["protocol"] == "raw"
        assert manifest["n_samples"] == N_SAMPLES
        assert manifest["layers"] == [3, 6, 9]

    def test_manifest_use_chat_template_is_false(self, tmp_path):
        """The manifest must always report use_chat_template=False (raw protocol)."""
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path), "m", "c", "c", "hf", [], [], 5, 10
        )
        assert manifest["use_chat_template"] is False


# =============================================================================
# Narrow checkpoint runner: single model load
# =============================================================================


class TestRunCheckpointExtraction:
    """run_checkpoint_extraction loads the model once for concept + probe."""

    @pytest.fixture
    def mock_setup(self):
        """Return a dict of mock objects and factories."""
        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        load_count = [0]
        loaded_model: list[Any] = [None]

        def fake_loader(config, revision):
            load_count[0] += 1
            loaded_model[0] = model
            return model, tokenizer

        return {
            "model": model,
            "tokenizer": tokenizer,
            "loader": fake_loader,
            "load_count": load_count,
            "loaded_model": loaded_model,
        }

    def test_full_run_writes_concept_vectors(self, tmp_path, monkeypatch, mock_setup):
        """After a full run, concept vector safetensors exist at every layer."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0, 1, 2]
        n = 5
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        cvec_dir = driver.concept_vectors_dir(str(tmp_path))
        for ly in layers:
            vectors = load_concept_vectors(cvec_dir, TARGET_MODEL.name, ly, "step_100")
            assert concepts[0] in vectors
            assert vectors[concepts[0]].n_positive == n
            assert vectors[concepts[0]].n_negative == n

    def test_full_run_writes_probe_activations(self, tmp_path, monkeypatch, mock_setup):
        """After a full run, probe activation files exist at every layer."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0, 1]
        probe_records = _fake_probe_records(10)
        spy, received = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        acts_dir = driver.activations_dir(str(tmp_path))
        for ly in layers:
            assert is_layer_complete(acts_dir, TARGET_MODEL.name, "step_100", ly, 10)

    def test_model_loaded_exactly_once(self, tmp_path, monkeypatch, mock_setup):
        """A single model load serves both concept and probe extraction."""
        concepts = ["python_valid_vs_syntax_error", "code_python_vs_cpp"]
        layers = [0, 1]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        assert mock_setup["load_count"][0] == 1, "model should be loaded exactly once"

    def test_probe_receives_same_model_as_loader(
        self, tmp_path, monkeypatch, mock_setup
    ):
        """The probe extraction path must receive the same model object."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, received = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        assert len(received) == 1
        assert received[0] is mock_setup["loaded_model"][0]

    def test_resume_skips_model_load(self, tmp_path, monkeypatch, mock_setup):
        """When all files are on disk, no model is loaded at all."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        n = 5
        probe_records = _fake_probe_records(10)

        # First run: write everything.
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)
        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )
        assert mock_setup["load_count"][0] == 1

        # Second run: should skip entirely (load_count stays at 1).
        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )
        assert mock_setup["load_count"][0] == 1, "resume should not reload model"

    def test_resume_reports_model_loaded_false(self, tmp_path, monkeypatch, mock_setup):
        """The result dict reports model_loaded=False when resuming."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        # First run.
        first = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )
        assert first["model_loaded"] is True

        # Second run.
        second = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )
        assert second["model_loaded"] is False

    def test_manifest_written_after_run(self, tmp_path, monkeypatch, mock_setup):
        """The atomic per-checkpoint manifest is written after extraction."""
        concepts = list(EXPERIMENT_CONCEPTS)
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        manifest = driver.load_checkpoint_manifest(
            str(tmp_path), TARGET_MODEL.name, "step_100"
        )
        assert manifest != {}
        assert manifest["complete"] is True
        assert set(manifest["concepts"]) == set(EXPERIMENT_CONCEPTS)
        for concept in concepts:
            assert manifest["concepts"][concept]["complete"] is True
        assert manifest["probe_activations"]["complete"] is True

    def test_manifest_records_protocol_raw(self, tmp_path, monkeypatch, mock_setup):
        """The manifest always records protocol='raw' and use_chat_template=False."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "main",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        manifest = result["manifest"]
        assert manifest["protocol"] == "raw"
        assert manifest["use_chat_template"] is False

    def test_base_checkpoint_uses_base_model(self, tmp_path, monkeypatch, mock_setup):
        """The base 'main' checkpoint resolves to BASE_MODEL."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "main",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        assert result["model_name"] == BASE_MODEL.name
        assert result["hf_id"] == BASE_MODEL.hf_id
        assert result["revision"] == "main"

    def test_rl_checkpoint_uses_target_model(self, tmp_path, monkeypatch, mock_setup):
        """An RL checkpoint resolves to TARGET_MODEL."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "step_1700",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        assert result["model_name"] == TARGET_MODEL.name
        assert result["hf_id"] == TARGET_MODEL.hf_id
        assert result["revision"] == "step_1700"

    def test_partial_concept_resume(self, tmp_path, monkeypatch, mock_setup):
        """If one concept layer is already done (v1 provenance), it is skipped."""
        concepts = ["python_valid_vs_syntax_error", "code_python_vs_cpp"]
        layers = [0, 1]
        n = 5
        probe_records = _fake_probe_records(10)

        # Pre-write concept[0] at layer 0 as a v1 sidecar (matching the
        # source texts that _fake_concept_texts returns).
        cvec_dir = driver.concept_vectors_dir(str(tmp_path))
        cv = _make_dummy_cv(concepts[0], TARGET_MODEL.name, 0, n)
        pos0, neg0 = _fake_concept_texts(concepts[0], n)
        save_concept_vectors(
            {concepts[0]: cv},
            cvec_dir,
            TARGET_MODEL.name,
            0,
            "step_100",
            protocol="raw",
            revision="step_100",
            hf_id=TARGET_MODEL.hf_id,
            max_seq_len=DEFAULT_MAX_SEQ_LEN,
            use_chat_template=False,
            concept_sources={concepts[0]: (pos0, neg0)},
        )

        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        # concept[0] layer 0 was skipped (v1 sidecar matched), layer 1 was extracted.
        assert 0 in result["concepts"][concepts[0]]["skipped"]
        assert 1 in result["concepts"][concepts[0]]["extracted"]


# =============================================================================
# Output isolation: no writes to legacy dirs
# =============================================================================


class TestNoLegacyWrites:
    """The driver must never write to concept_dynamics_multi or related dirs."""

    def test_all_writes_under_output_root(self, tmp_path, monkeypatch):
        """Every file created by the runner lives under output_root."""
        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        output_root = str(tmp_path / "rl_zero_code_syntax")

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=output_root,
            model_loader=lambda c, r: (model, tokenizer),
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )

        # Walk all created files and verify they're under output_root.
        for dirpath, dirnames, filenames in os.walk(output_root):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                assert fpath.startswith(output_root), (
                    f"file {fpath} is outside output_root"
                )

    def test_nothing_written_to_paired_concept_root(self, tmp_path, monkeypatch):
        """The paired-concept results dir must not receive any new files."""
        legacy_roots = [
            PAIRED_CONCEPT_RESULTS_ROOT,
            PAIRED_CONCEPT_RESULTS_ROOT_QUICK,
        ]
        # Snapshot existing files in legacy dirs (if they exist).
        before: dict[str, set[str]] = {}
        for root in legacy_roots:
            if os.path.isdir(root):
                files = set()
                for dp, dn, fns in os.walk(root):
                    for fn in fns:
                        files.add(os.path.join(dp, fn))
                before[root] = files

        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=["python_valid_vs_syntax_error"],
            layers=[0],
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=lambda c, r: (model, tokenizer),
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: _fake_probe_records(10),
            unload_gpu=False,
        )

        # Verify no new files appeared in legacy dirs.
        for root in legacy_roots:
            if os.path.isdir(root):
                after = set()
                for dp, dn, fns in os.walk(root):
                    for fn in fns:
                        after.add(os.path.join(dp, fn))
                new_files = after - before.get(root, set())
                assert new_files == set(), (
                    f"unexpected new files in legacy dir {root}: {new_files}"
                )

    def test_summary_written_only_under_output_root(self, tmp_path):
        """The summary path resolves under the output root."""
        path = driver.summary_path(str(tmp_path))
        assert path.startswith(str(tmp_path))
        assert "concept_dynamics_multi" not in path

    def test_manifest_path_under_output_root(self, tmp_path):
        """The manifest path resolves under the output root."""
        path = driver.manifest_file_path(str(tmp_path), "model", "ckpt")
        assert path.startswith(str(tmp_path))
        assert "concept_dynamics_multi" not in path


# =============================================================================
# Global summary
# =============================================================================


class TestWriteSummary:
    """write_summary produces a valid, resumable extraction_summary.json."""

    def test_summary_written_atomically(self, tmp_path):
        per_ckpt = {
            "olmo3-base/main": {
                "manifest": {"complete": True},
            },
            "olmo3-rl-zero-code/step_100": {
                "manifest": {"complete": False},
            },
        }
        path = driver.write_summary(
            str(tmp_path),
            ["main", "step_100"],
            per_ckpt,
            concepts=["c1"],
            layers=[3],
            n_samples=50,
        )
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["checkpoints_complete"] == 1
        assert data["checkpoints_errors"] == 0
        assert data["protocol"] == PROTOCOL

    def test_summary_under_output_root(self, tmp_path):
        path = driver.write_summary(
            str(tmp_path), [], {}, concepts=[], layers=[], n_samples=0
        )
        assert path.startswith(str(tmp_path))


# =============================================================================
# CLI parsing
# =============================================================================


class TestParseArgs:
    """parse_args handles all CLI flags."""

    def test_defaults(self):
        args = driver.parse_args([])
        assert args.only == "all"
        assert args.samples == N_SAMPLES
        assert args.keep_hf_cache is False
        assert args.quick is False
        assert args.max_seq_len == DEFAULT_MAX_SEQ_LEN
        assert args.output is None
        assert args.limit is None
        assert args.checkpoints is None
        assert args.layers is None

    def test_only_base(self):
        args = driver.parse_args(["--only", "base"])
        assert args.only == "base"

    def test_only_rl(self):
        args = driver.parse_args(["--only", "rl"])
        assert args.only == "rl"

    def test_checkpoints_csv(self):
        args = driver.parse_args(["--checkpoints", "main,step_100,step_200"])
        assert args.checkpoints == "main,step_100,step_200"

    def test_limit(self):
        args = driver.parse_args(["--limit", "3"])
        assert args.limit == 3

    def test_layers_csv(self):
        args = driver.parse_args(["--layers", "3,6,9"])
        assert args.layers == "3,6,9"

    def test_samples(self):
        args = driver.parse_args(["--samples", "10"])
        assert args.samples == 10

    def test_output(self):
        args = driver.parse_args(["--output", "/tmp/my_output"])
        assert args.output == "/tmp/my_output"

    def test_keep_hf_cache(self):
        args = driver.parse_args(["--keep-hf-cache"])
        assert args.keep_hf_cache is True

    def test_quick(self):
        args = driver.parse_args(["--quick"])
        assert args.quick is True

    def test_max_seq_len(self):
        args = driver.parse_args(["--max-seq-len", "1024"])
        assert args.max_seq_len == 1024


# =============================================================================
# CLI main() end-to-end with mocks
# =============================================================================


class TestMainEndToEnd:
    """main() orchestrates the full driver with mocked model loading."""

    def test_quick_mode_runs_one_checkpoint(self, tmp_path, monkeypatch):
        """--quick mode extracts 1 base checkpoint with 1 layer and 5 samples."""
        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)
        monkeypatch.setattr(driver, "EXPECTED_D_MODEL", 8)
        monkeypatch.setattr(
            driver,
            "_load_model_and_tokenizer",
            lambda config, rev: (model, tokenizer),
        )
        monkeypatch.setattr(driver, "_clean_hf_cache", lambda hf_id: None)
        # Stub concept_texts_fn via monkeypatching the module-level import.
        monkeypatch.setattr(
            "src.contrastive_datasets.load_contrastive_texts", _fake_concept_texts
        )
        # Stub probe record building so no data files are needed.
        monkeypatch.setattr(
            "src.probe_activations.build_probe_records",
            lambda: _fake_probe_records(10),
        )

        output = str(tmp_path / "rl_zero_code_syntax_quick")
        ret = driver.main(["--quick", "--output", output])

        assert ret == 0
        # Verify summary was written.
        assert os.path.exists(driver.summary_path(output))
        # Verify concept vectors were written.
        cvec_dir = driver.concept_vectors_dir(output)
        assert os.path.exists(os.path.join(cvec_dir, BASE_MODEL.name, "main"))
        # Verify manifest was written.
        manifest = driver.load_checkpoint_manifest(output, BASE_MODEL.name, "main")
        assert manifest["complete"] is True

    def test_main_writes_summary_after_each_checkpoint(self, tmp_path, monkeypatch):
        """The summary is updated after each checkpoint (resumable)."""
        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)
        monkeypatch.setattr(
            driver,
            "_load_model_and_tokenizer",
            lambda config, rev: (model, tokenizer),
        )
        monkeypatch.setattr(driver, "_clean_hf_cache", lambda hf_id: None)
        monkeypatch.setattr(
            "src.contrastive_datasets.load_contrastive_texts", _fake_concept_texts
        )
        monkeypatch.setattr(
            "src.probe_activations.build_probe_records",
            lambda: _fake_probe_records(10),
        )

        output = str(tmp_path / "out")
        ret = driver.main(
            [
                "--only",
                "base",
                "--layers",
                "0",
                "--samples",
                "5",
                "--output",
                output,
                "--max-seq-len",
                "512",
            ]
        )
        assert ret == 0
        summary_path = driver.summary_path(output)
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert len(summary["per_checkpoint"]) == 1
        assert "olmo3-base/main" in summary["per_checkpoint"]

    def test_main_rejects_legacy_output(self, monkeypatch):
        """main() exits with code 2 when output collides with a legacy dir."""
        ret = driver.main(["--output", PAIRED_CONCEPT_RESULTS_ROOT])
        assert ret == 2

    def test_main_no_checkpoints_selected(self, tmp_path, monkeypatch):
        """main() exits with code 2 when no checkpoints match."""
        ret = driver.main(
            [
                "--only",
                "base",
                "--checkpoints",
                "step_100",
                "--output",
                str(tmp_path),
            ]
        )
        assert ret == 2

    def test_main_resume_skips_completed_checkpoint(self, tmp_path, monkeypatch):
        """Running main() twice: the second run skips the completed checkpoint."""
        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        load_count = [0]

        def counting_loader(config, rev):
            load_count[0] += 1
            return model, tokenizer

        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)
        monkeypatch.setattr(driver, "_load_model_and_tokenizer", counting_loader)
        monkeypatch.setattr(driver, "_clean_hf_cache", lambda hf_id: None)
        monkeypatch.setattr(driver, "EXPECTED_D_MODEL", 8)
        monkeypatch.setattr(
            "src.contrastive_datasets.load_contrastive_texts", _fake_concept_texts
        )
        monkeypatch.setattr(
            "src.probe_activations.build_probe_records",
            lambda: _fake_probe_records(10),
        )

        output = str(tmp_path / "out")
        # First run.
        driver.main(
            [
                "--only",
                "base",
                "--layers",
                "0",
                "--samples",
                "5",
                "--output",
                output,
                "--max-seq-len",
                "512",
            ]
        )
        assert load_count[0] == 1

        # Second run: should skip (model not loaded).
        driver.main(
            [
                "--only",
                "base",
                "--layers",
                "0",
                "--samples",
                "5",
                "--output",
                output,
                "--max-seq-len",
                "512",
            ]
        )
        assert load_count[0] == 1, "second run should skip model load"


# =============================================================================
# Hardened checkpoint manifest (max_seq_len + records_fingerprint)
# =============================================================================


class TestCheckpointManifestHardened:
    """The per-checkpoint manifest carries max_seq_len and records_fingerprint."""

    def test_manifest_records_max_seq_len(self, tmp_path):
        records = _fake_probe_records(10)
        acts = torch.randn(10, 8)
        for ly in [3, 6]:
            save_layer_activations(
                driver.activations_dir(str(tmp_path)),
                "m",
                "c",
                ly,
                acts,
                records,
                max_seq_len=2048,
            )
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path),
            "m",
            "c",
            "c",
            "hf/id",
            [],
            [3, 6],
            5,
            10,
            max_seq_len=2048,
            records_fingerprint=compute_records_fingerprint(records),
        )
        assert manifest["max_seq_len"] == 2048

    def test_manifest_records_fingerprint(self, tmp_path):
        records = _fake_probe_records(10)
        acts = torch.randn(10, 8)
        for ly in [3]:
            save_layer_activations(
                driver.activations_dir(str(tmp_path)),
                "m",
                "c",
                ly,
                acts,
                records,
                max_seq_len=512,
            )
        expected_fp = compute_records_fingerprint(records)
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path),
            "m",
            "c",
            "c",
            "hf",
            [],
            [3],
            5,
            10,
            max_seq_len=512,
            records_fingerprint=expected_fp,
        )
        assert manifest["records_fingerprint"] == expected_fp

    def test_manifest_complete_only_when_strict_checks_pass(self, tmp_path):
        """Probe layers written without max_seq_len are NOT complete under
        strict manifest validation."""
        records = _fake_probe_records(10)
        acts = torch.randn(10, 8)
        for ly in [3]:
            # Write WITHOUT max_seq_len (simulates pre-hardening artifact).
            save_layer_activations(
                driver.activations_dir(str(tmp_path)), "m", "c", ly, acts, records
            )
        manifest = driver.build_checkpoint_manifest(
            str(tmp_path),
            "m",
            "c",
            "c",
            "hf",
            [],
            [3],
            5,
            10,
            max_seq_len=2048,
            records_fingerprint=compute_records_fingerprint(records),
        )
        # Probe layers fail strict max_seq_len check → not complete.
        assert manifest["probe_activations"]["complete"] is False
        assert manifest["probe_activations"]["completed_layers"] == []


# =============================================================================
# Hardened run_checkpoint_extraction resume
# =============================================================================


class TestRunCheckpointResumeHardened:
    """The driver's resume path validates max_seq_len and records identity."""

    @pytest.fixture
    def mock_setup(self):
        model = MockModel(n_layers=32, d_model=8)
        tokenizer = MockTokenizer(seq_len=5)
        load_count = [0]

        def fake_loader(config, revision):
            load_count[0] += 1
            return model, tokenizer

        return {
            "model": model,
            "tokenizer": tokenizer,
            "loader": fake_loader,
            "load_count": load_count,
        }

    def test_run_threads_max_seq_len_to_manifest(
        self, tmp_path, monkeypatch, mock_setup
    ):
        """The manifest records the max_seq_len used during extraction."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
            max_seq_len=2048,
        )
        assert result["manifest"]["max_seq_len"] == 2048

    def test_run_threads_fingerprint_to_manifest(
        self, tmp_path, monkeypatch, mock_setup
    ):
        """The manifest records the ordered records fingerprint."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        result = driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=5,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
        )
        expected_fp = compute_records_fingerprint(probe_records)
        assert result["manifest"]["records_fingerprint"] == expected_fp

    def test_resume_rejects_max_seq_len_change(self, tmp_path, monkeypatch, mock_setup):
        """Changing max_seq_len between runs forces re-extraction (model reload)."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        n = 5
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        # First run with max_seq_len=512.
        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
            max_seq_len=512,
        )
        assert mock_setup["load_count"][0] == 1

        # Second run with max_seq_len=2048: model must be reloaded.
        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
            max_seq_len=2048,
        )
        assert mock_setup["load_count"][0] == 2, (
            "changed max_seq_len must trigger reload"
        )

    def test_resume_passes_when_unchanged(self, tmp_path, monkeypatch, mock_setup):
        """Same max_seq_len + same records → skip (no model reload)."""
        concepts = ["python_valid_vs_syntax_error"]
        layers = [0]
        n = 5
        probe_records = _fake_probe_records(10)
        spy, _ = _make_probe_extraction_spy(d_model=8)
        monkeypatch.setattr(driver, "run_extraction_with_resume", spy)

        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
            max_seq_len=512,
        )
        assert mock_setup["load_count"][0] == 1

        # Second run: same config → skip.
        driver.run_checkpoint_extraction(
            "step_100",
            concepts=concepts,
            layers=layers,
            n_samples=n,
            output_root=str(tmp_path),
            model_loader=mock_setup["loader"],
            concept_texts_fn=_fake_concept_texts,
            probe_records_fn=lambda: probe_records,
            unload_gpu=False,
            max_seq_len=512,
        )
        assert mock_setup["load_count"][0] == 1, "unchanged config should skip"
