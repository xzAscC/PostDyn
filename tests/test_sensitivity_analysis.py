"""Tests for the raw-vs-chat direction sensitivity analysis (SECONDARY).

Coverage areas (per the task brief):

1. **Metadata validation** -- ``validate_chat_metadata`` accepts matching
   metadata and rejects every field mismatch (model / layer / concept /
   sample count / d_model) with a clear ``MetadataMismatchError``.
2. **Comparison math** -- ``cosine_similarity`` / ``norm_of`` /
   ``compare_vectors`` produce the right numbers on known tensors.
3. **Atomic output** -- ``write_sensitivity_json`` leaves no partial file on a
   serialization failure and writes the full payload on success.
4. **Path isolation** -- the driver refuses to write inside the primary
   ``concept_dynamics_multi`` store (including its parent root) or any path
   containing a ``metrics`` segment; the old chat results tree is NEVER
   modified (read-only reuse).
5. **Missing-data handling** -- raw-missing / chat-missing / metadata-rejected
   entries are recorded honestly per-entry instead of aborting the run.
6. **Full driver** -- with mock raw + chat vectors, ``sensitivity.json`` is
   produced under the sensitivity output dir with the expected summary counts.
7. **Gated target extraction** -- the chat-TARGET extractor fails clearly
   (``ChatTemplateUnavailableError``) when the tokenizer has no usable
   ``apply_chat_template``, skips already-present checkpoints, and otherwise
   writes into the sensitivity-only target dir.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from postdyn.concept_dynamics import ConceptVector, save_concept_vectors
from postdyn.sensitivity_analysis import (
    ChatTemplateUnavailableError,
    DEFAULT_SENSITIVITY_OUTPUT_ROOT,
    LIMITATIONS,
    MetadataMismatchError,
    PROTOCOL,
    RELATED_CONCEPTS,
    SENSITIVITY_CHECKPOINTS,
    SENSITIVITY_CONCEPTS,
    SENSITIVITY_D_MODEL,
    SENSITIVITY_LAYERS,
    SENSITIVITY_MODEL,
    SENSITIVITY_N_SAMPLES,
    SensitivityEntry,
    TARGET_CONCEPT,
    VectorSpec,
    _assert_chat_target_isolation,
    _assert_path_isolation,
    _entry_from_spec,
    _validate_output_filename,
    compare_vectors,
    cosine_similarity,
    extract_missing_chat_target,
    load_chat_vector,
    load_raw_vector,
    norm_of,
    run_sensitivity_analysis,
    validate_chat_metadata,
    write_sensitivity_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_PATH = REPO_ROOT / "scripts" / "run_sensitivity_analysis.py"


# =============================================================================
# Helpers
# =============================================================================


def _spec(
    concept: str = TARGET_CONCEPT,
    checkpoint: str = "step_100",
    layer_idx: int = 3,
    n_samples: int = SENSITIVITY_N_SAMPLES,
    d_model: int = SENSITIVITY_D_MODEL,
    model: str = SENSITIVITY_MODEL,
) -> VectorSpec:
    return VectorSpec(
        model=model,
        checkpoint=checkpoint,
        layer_idx=layer_idx,
        concept=concept,
        n_samples=n_samples,
        d_model=d_model,
    )


def _mock_cv(
    concept: str,
    model: str = SENSITIVITY_MODEL,
    layer_idx: int = 3,
    n: int = SENSITIVITY_N_SAMPLES,
    d: int = SENSITIVITY_D_MODEL,
    seed: int = 0,
    raw_scale: float = 1.0,
) -> ConceptVector:
    """Build a deterministic ConceptVector with a controllable raw_direction."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(d, generator=g) * raw_scale
    return ConceptVector(
        concept_name=concept,
        model_name=model,
        layer_idx=layer_idx,
        steering_vector=raw / raw.norm().clamp(min=1e-10),
        raw_direction=raw,
        positive_mean=torch.randn(d, generator=g),
        negative_mean=torch.randn(d, generator=g),
        positive_std=torch.randn(d, generator=g).abs(),
        negative_std=torch.randn(d, generator=g).abs(),
        n_positive=n,
        n_negative=n,
        d_model=d,
    )


def _save_mock_store(
    root_dir: str,
    model: str,
    checkpoint: str,
    layer_idx: int,
    concepts: dict[str, ConceptVector],
) -> None:
    """Mirror concept_dynamics.save_concept_vectors layout for several concepts."""
    save_concept_vectors(concepts, root_dir, model, layer_idx, checkpoint)


def _hash_tree(path: str) -> dict[str, str]:
    """SHA-256 of every file under ``path`` (used for read-only verification)."""
    out: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, path)
            out[rel] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


# =============================================================================
# 1. Metadata validation
# =============================================================================


class TestMetadataValidation:
    def _meta(
        self,
        model: str = SENSITIVITY_MODEL,
        layer: int = 3,
        concept: str = TARGET_CONCEPT,
        n_pos: int = SENSITIVITY_N_SAMPLES,
        n_neg: int = SENSITIVITY_N_SAMPLES,
        d_model: int = SENSITIVITY_D_MODEL,
    ) -> dict[str, Any]:
        return {
            "model_name": model,
            "layer_idx": layer,
            "concepts": [
                {
                    "name": concept,
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "d_model": d_model,
                }
            ],
        }

    def test_accepts_matching_metadata(self) -> None:
        validate_chat_metadata(self._meta(), _spec())

    def test_accepts_superset_samples(self) -> None:
        meta = self._meta(n_pos=100, n_neg=100)
        validate_chat_metadata(meta, _spec(n_samples=50))

    def test_rejects_wrong_model(self) -> None:
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(self._meta(model="other-model"), _spec())
        assert any("model_name" in m for m in ei.value.mismatches)

    def test_rejects_wrong_layer(self) -> None:
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(self._meta(layer=99), _spec(layer_idx=3))
        assert any("layer_idx" in m for m in ei.value.mismatches)

    def test_rejects_missing_concept(self) -> None:
        meta = self._meta()
        meta["concepts"] = [
            {
                "name": "something_else",
                "n_positive": 50,
                "n_negative": 50,
                "d_model": 4096,
            }
        ]
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(meta, _spec())
        assert any("concept:" in m for m in ei.value.mismatches)

    def test_rejects_insufficient_positive_samples(self) -> None:
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(self._meta(n_pos=10), _spec(n_samples=50))
        assert any("n_positive" in m for m in ei.value.mismatches)

    def test_rejects_insufficient_negative_samples(self) -> None:
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(self._meta(n_neg=10), _spec(n_samples=50))
        assert any("n_negative" in m for m in ei.value.mismatches)

    def test_rejects_wrong_d_model(self) -> None:
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(self._meta(d_model=2048), _spec(d_model=4096))
        assert any("d_model" in m for m in ei.value.mismatches)

    def test_collects_multiple_mismatches_at_once(self) -> None:
        meta = self._meta(model="x", layer=1, d_model=8)
        with pytest.raises(MetadataMismatchError) as ei:
            validate_chat_metadata(meta, _spec())
        assert len(ei.value.mismatches) >= 3


# =============================================================================
# 2. Comparison math
# =============================================================================


class TestComparisonMath:
    def test_cosine_identical_vectors_is_one(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_opposite_vectors_is_minus_one(self) -> None:
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_similarity(a, -a) == pytest.approx(-1.0)

    def test_cosine_orthogonal_vectors_is_zero(self) -> None:
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_zero_vector_returns_zero(self) -> None:
        a = torch.zeros(4)
        b = torch.ones(4)
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            cosine_similarity(torch.zeros(3), torch.zeros(4))

    def test_norm_of_known_tensor(self) -> None:
        v = torch.tensor([3.0, 4.0])
        assert norm_of(v) == pytest.approx(5.0)

    def test_compare_vectors_signed_norm_diff(self) -> None:
        unit_raw = _mock_cv("c", seed=1, raw_scale=1.0)
        raw = _mock_cv("c", seed=1, raw_scale=3.0)
        chat = _mock_cv("c", seed=2, raw_scale=1.0)
        out = compare_vectors(raw, chat)
        assert out["norm_raw"] == pytest.approx(
            3.0 * unit_raw.raw_direction.norm().item()
        )
        assert out["norm_diff"] == pytest.approx(out["norm_raw"] - out["norm_chat"])
        assert -1.0 <= out["cosine"] <= 1.0

    def test_compare_vectors_d_model_mismatch_raises(self) -> None:
        raw = _mock_cv("c", d=4096)
        chat = _mock_cv("c", d=2048)
        with pytest.raises(ValueError):
            compare_vectors(raw, chat)


# =============================================================================
# 3. Atomic output
# =============================================================================


class TestAtomicWrite:
    def test_successful_write_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "sensitivity.json"
        payload = {"protocol": {"primary": False}, "entries": [1, 2, 3]}
        written = write_sensitivity_json(str(out), payload)
        assert written == str(out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == payload

    def test_creates_output_directory_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "sensitivity.json"
        write_sensitivity_json(str(nested), {"x": 1})
        assert nested.exists()

    def test_no_partial_file_on_serialization_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "sensitivity.json"
        out.write_text("PREEXISTING")

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("disk full")

        monkeypatch.setattr("json.dump", boom)
        with pytest.raises(RuntimeError):
            write_sensitivity_json(str(out), {"big": "payload"})

        # Pre-existing content untouched and no .tmp left behind.
        assert out.read_text() == "PREEXISTING"
        leftover = [p for p in tmp_path.iterdir() if p.name != "sensitivity.json"]
        assert leftover == []

    def test_atomic_replace_seen_by_reader(self, tmp_path: Path) -> None:
        out = tmp_path / "sensitivity.json"
        write_sensitivity_json(str(out), {"v": 1})
        write_sensitivity_json(str(out), {"v": 2})
        assert json.loads(out.read_text()) == {"v": 2}


# =============================================================================
# 4. Path isolation
# =============================================================================


class TestPathIsolation:
    def test_rejects_output_inside_chat_vectors_dir(self, tmp_path: Path) -> None:
        chat = tmp_path / "concept_dynamics_multi" / "vectors"
        chat.mkdir(parents=True)
        out = chat / "sensitivity"
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_path_isolation(str(out), str(chat))

    def test_rejects_output_inside_primary_store_parent(self, tmp_path: Path) -> None:
        chat = tmp_path / "concept_dynamics_multi" / "vectors"
        chat.mkdir(parents=True)
        out = tmp_path / "concept_dynamics_multi" / "sensitivity"
        with pytest.raises(ValueError, match="primary store"):
            _assert_path_isolation(str(out), str(chat))

    def test_rejects_metrics_segment_anywhere(self, tmp_path: Path) -> None:
        out = tmp_path / "logs" / "metrics" / "sensitivity"
        with pytest.raises(ValueError, match="'metrics' segment"):
            _assert_path_isolation(str(out))

    def test_accepts_isolated_output_dir(self, tmp_path: Path) -> None:
        chat = tmp_path / "concept_dynamics_multi" / "vectors"
        raw = tmp_path / "concept_dynamics_raw" / "vectors"
        out = tmp_path / "sensitivity"
        # No raise.
        _assert_path_isolation(str(out), str(chat), str(raw))

    def test_driver_refuses_to_write_into_primary_store(self, tmp_path: Path) -> None:
        chat_old = tmp_path / "primary" / "vectors"
        chat_old.mkdir(parents=True)
        raw = tmp_path / "raw" / "vectors"
        raw.mkdir(parents=True)
        out = tmp_path / "primary" / "sensitivity"
        with pytest.raises(ValueError):
            run_sensitivity_analysis(
                raw_vectors_dir=str(raw),
                chat_old_vectors_dir=str(chat_old),
                chat_target_vectors_dir=str(tmp_path / "tgt"),
                output_dir=str(out),
            )
        assert not out.exists()


# =============================================================================
# 4b. Chat-TARGET write-root isolation
# =============================================================================


class TestChatTargetIsolation:
    """The chat-TARGET extraction write-root must be isolated.

    ``chat_target_vectors_dir`` is the ONLY place the sensitivity pass ever
    WRITES (during the gated extraction). It must:

    * nest STRICTLY UNDER the designated sensitivity output root (the same
      root that holds ``sensitivity.json``), and
    * NEVER lie inside the raw primary store, the old-chat primary store, or
      either store's parent.

    The check runs at three layers: the shared helper, the
    ``extract_missing_chat_target`` library entry, and the CLI orchestrator.
    """

    # --- shared helper: primary-store rejection ---

    def test_helper_rejects_target_inside_raw_vectors_dir(self, tmp_path: Path) -> None:
        raw = tmp_path / "concept_dynamics_raw" / "vectors"
        raw.mkdir(parents=True)
        tgt = raw / "tgt"
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_chat_target_isolation(
                str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                raw_vectors_dir=str(raw),
            )

    def test_helper_rejects_target_inside_raw_parent(self, tmp_path: Path) -> None:
        raw = tmp_path / "concept_dynamics_raw" / "vectors"
        raw.mkdir(parents=True)
        tgt = tmp_path / "concept_dynamics_raw" / "tgt"
        with pytest.raises(ValueError, match="primary store"):
            _assert_chat_target_isolation(
                str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                raw_vectors_dir=str(raw),
            )

    def test_helper_rejects_target_inside_old_chat_vectors_dir(
        self, tmp_path: Path
    ) -> None:
        old = tmp_path / "concept_dynamics_multi" / "vectors"
        old.mkdir(parents=True)
        tgt = old / "tgt"
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_chat_target_isolation(
                str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                chat_old_vectors_dir=str(old),
            )

    def test_helper_rejects_target_inside_old_chat_parent(self, tmp_path: Path) -> None:
        old = tmp_path / "concept_dynamics_multi" / "vectors"
        old.mkdir(parents=True)
        tgt = tmp_path / "concept_dynamics_multi" / "tgt"
        with pytest.raises(ValueError, match="primary store"):
            _assert_chat_target_isolation(
                str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                chat_old_vectors_dir=str(old),
            )

    # --- shared helper: sensitivity-root containment ---

    def test_helper_rejects_target_outside_sensitivity_root(
        self, tmp_path: Path
    ) -> None:
        tgt = tmp_path / "elsewhere" / "tgt"
        with pytest.raises(ValueError, match="sensitivity output root"):
            _assert_chat_target_isolation(
                str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
            )

    def test_helper_rejects_target_equal_to_sensitivity_root(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sensitivity"
        with pytest.raises(ValueError, match="SUBDIR"):
            _assert_chat_target_isolation(
                str(root),
                sensitivity_output_root=str(root),
            )

    def test_helper_accepts_target_nested_under_sensitivity_root(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sensitivity"
        tgt = root / "chat_target_vectors"
        _assert_chat_target_isolation(
            str(tgt),
            sensitivity_output_root=str(root),
            raw_vectors_dir=str(tmp_path / "raw" / "vectors"),
            chat_old_vectors_dir=str(tmp_path / "old" / "vectors"),
        )

    def test_helper_accepts_shipped_default_layout(self) -> None:
        root = DEFAULT_SENSITIVITY_OUTPUT_ROOT
        tgt = os.path.join(root, "chat_target_vectors")
        _assert_chat_target_isolation(tgt, sensitivity_output_root=root)

    # --- library entry: extract_missing_chat_target ---

    def test_extract_rejects_target_in_old_chat_before_model_load(
        self, tmp_path: Path
    ) -> None:
        old = tmp_path / "concept_dynamics_multi" / "vectors"
        old.mkdir(parents=True)
        tgt = old / "tgt"

        loader_calls: list[str] = []

        def loader(model_config, revision=None):
            loader_calls.append(revision or "?")
            return object(), _FakeTokenizerWithTemplate()

        with pytest.raises(ValueError, match="Path-isolation violation"):
            extract_missing_chat_target(
                chat_target_vectors_dir=str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                chat_old_vectors_dir=str(old),
                load_model_and_tokenizer=loader,
                extract_fn=lambda *a, **k: {},
            )
        assert loader_calls == []

    def test_extract_rejects_target_outside_root_before_model_load(
        self, tmp_path: Path
    ) -> None:
        tgt = tmp_path / "elsewhere" / "tgt"
        loader_calls: list[str] = []

        def loader(model_config, revision=None):
            loader_calls.append(revision or "?")
            return object(), _FakeTokenizerWithTemplate()

        with pytest.raises(ValueError, match="sensitivity output root"):
            extract_missing_chat_target(
                chat_target_vectors_dir=str(tgt),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                load_model_and_tokenizer=loader,
                extract_fn=lambda *a, **k: {},
            )
        assert loader_calls == []

    def test_extract_accepts_nested_target_and_proceeds(self, tmp_path: Path) -> None:
        root = tmp_path / "sensitivity"
        tgt = root / "chat_target_vectors"

        def loader(model_config, revision=None):
            return object(), _FakeTokenizerWithTemplate()

        def extract_fn(model_config, concepts, layers, n_samples, out_dir, **kw):
            ckpt = kw["checkpoint"]
            for layer in layers:
                _save_mock_store(
                    out_dir,
                    model_config.name,
                    ckpt,
                    layer,
                    {TARGET_CONCEPT: _mock_cv(TARGET_CONCEPT, layer_idx=layer)},
                )
            return {"checkpoint": ckpt}

        summary = extract_missing_chat_target(
            chat_target_vectors_dir=str(tgt),
            sensitivity_output_root=str(root),
            load_model_and_tokenizer=loader,
            extract_fn=extract_fn,
        )
        assert len(summary["checkpoints"]["extracted"]) == 3

    # --- CLI orchestrator ---

    def test_cli_rejects_target_dir_in_old_chat_store(self, tmp_path: Path) -> None:
        cli = _load_cli()
        old = tmp_path / "concept_dynamics_multi" / "vectors"
        old.mkdir(parents=True)
        tgt = old / "tgt"
        with pytest.raises(SystemExit) as ei:
            cli.main(
                [
                    "--chat-old-vectors-dir",
                    str(old),
                    "--raw-vectors-dir",
                    str(tmp_path / "raw" / "vectors"),
                    "--chat-target-vectors-dir",
                    str(tgt),
                    "--output-dir",
                    str(tmp_path / "sensitivity"),
                ]
            )
        assert ei.value.code == 2

    def test_cli_rejects_target_dir_in_raw_store(self, tmp_path: Path) -> None:
        cli = _load_cli()
        raw = tmp_path / "concept_dynamics_raw" / "vectors"
        raw.mkdir(parents=True)
        tgt = raw / "tgt"
        with pytest.raises(SystemExit) as ei:
            cli.main(
                [
                    "--chat-old-vectors-dir",
                    str(tmp_path / "old" / "vectors"),
                    "--raw-vectors-dir",
                    str(raw),
                    "--chat-target-vectors-dir",
                    str(tgt),
                    "--output-dir",
                    str(tmp_path / "sensitivity"),
                ]
            )
        assert ei.value.code == 2

    def test_cli_rejects_target_dir_outside_sensitivity_root(
        self, tmp_path: Path
    ) -> None:
        cli = _load_cli()
        tgt = tmp_path / "elsewhere" / "tgt"
        with pytest.raises(SystemExit) as ei:
            cli.main(
                [
                    "--chat-old-vectors-dir",
                    str(tmp_path / "old" / "vectors"),
                    "--raw-vectors-dir",
                    str(tmp_path / "raw" / "vectors"),
                    "--chat-target-vectors-dir",
                    str(tgt),
                    "--output-dir",
                    str(tmp_path / "sensitivity"),
                ]
            )
        assert ei.value.code == 2

    def test_cli_accepts_default_layout(self) -> None:
        cli = _load_cli()
        args = cli.parse_args([])
        assert os.path.abspath(args.chat_target_vectors_dir).startswith(
            os.path.abspath(args.output_dir) + os.sep
        )
        from postdyn.sensitivity_analysis import _assert_chat_target_isolation

        _assert_chat_target_isolation(
            args.chat_target_vectors_dir,
            sensitivity_output_root=args.output_dir,
        )


# =============================================================================
# 4c. Output-filename basename validation
# =============================================================================


class TestOutputFilenameValidation:
    """``output_filename`` MUST be a simple, nonempty basename.

    ``os.path.join(output_dir, output_filename)`` would otherwise let a caller
    smuggle ``../`` traversal, absolute paths, or separator-containing names
    past the output dir and write the sensitivity payload elsewhere.
    """

    def test_accepts_simple_basename(self) -> None:
        _validate_output_filename("sensitivity.json")

    def test_accepts_filename_with_dots_in_middle(self) -> None:
        _validate_output_filename("my.report.v2.json")

    def test_accepts_no_extension(self) -> None:
        _validate_output_filename("sensitivity")

    @pytest.mark.parametrize("bad", ["", ".", ".."])
    def test_rejects_special_and_empty(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename(bad)

    def test_rejects_parent_traversal_suffix(self) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename("../evil.json")

    def test_rejects_subpath_with_forward_separator(self) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename("subdir/evil.json")

    def test_rejects_subpath_with_back_separator(self) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename("subdir\\evil.json")

    def test_rejects_absolute_unix_path(self) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename("/etc/passwd")

    def test_rejects_leading_separator_only(self) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename("/")

    def test_rejects_windows_drive_prefix(self) -> None:
        with pytest.raises(ValueError):
            _validate_output_filename("C:evil.json")


# =============================================================================
# 4d. Canonical containment (realpath + commonpath, defeats symlinks/traversal)
# =============================================================================


class TestCanonicalContainment:
    """Path-isolation guards must use canonical REAL paths.

    ``os.path.abspath`` + ``startswith`` is lexical: it does not resolve
    symlinks, so a symlinked ``output_dir`` whose target lives inside a
    primary store (or a symlinked forbidden root that covers the output)
    silently bypasses the guard. These tests pin the canonical behaviour.
    """

    def test_symlinked_output_into_primary_is_rejected(self, tmp_path: Path) -> None:
        primary = tmp_path / "concept_dynamics_multi" / "vectors"
        primary.mkdir(parents=True)
        link = tmp_path / "sensitivity_link"
        link.symlink_to(primary)
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_path_isolation(str(link), str(primary))

    def test_symlinked_forbidden_root_covering_output_is_rejected(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "safe" / "out"
        out.mkdir(parents=True)
        store_parent = tmp_path / "safe"
        link = tmp_path / "primary_link"
        link.symlink_to(store_parent)
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_path_isolation(str(out), str(link))

    def test_parent_traversal_in_output_dir_is_rejected(self, tmp_path: Path) -> None:
        primary = tmp_path / "concept_dynamics_multi" / "vectors"
        primary.mkdir(parents=True)
        # ``../`` inside output_dir lands inside the primary store after norm.
        out = primary / ".." / "sensitivity_escape"
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_path_isolation(str(out), str(primary))

    def test_nonexisting_output_under_existing_primary_is_rejected(
        self, tmp_path: Path
    ) -> None:
        primary = tmp_path / "concept_dynamics_multi" / "vectors"
        primary.mkdir(parents=True)
        out = primary / "does" / "not" / "exist"
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_path_isolation(str(out), str(primary))

    def test_symlinked_chat_target_into_primary_is_rejected(
        self, tmp_path: Path
    ) -> None:
        primary = tmp_path / "concept_dynamics_multi" / "vectors"
        primary.mkdir(parents=True)
        tgt_link = tmp_path / "tgt_link"
        tgt_link.symlink_to(primary)
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_chat_target_isolation(
                str(tgt_link),
                sensitivity_output_root=str(tmp_path / "sensitivity"),
                chat_old_vectors_dir=str(primary),
            )

    def test_symlinked_target_pointing_outside_root_is_rejected(
        self, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        root = tmp_path / "sensitivity"
        root.mkdir()
        tgt_link = root / "escape_link"
        tgt_link.symlink_to(elsewhere)
        with pytest.raises(ValueError, match="sensitivity output root"):
            _assert_chat_target_isolation(
                str(tgt_link), sensitivity_output_root=str(root)
            )

    def test_symlinked_target_into_raw_store_is_rejected(self, tmp_path: Path) -> None:
        raw = tmp_path / "concept_dynamics_raw" / "vectors"
        raw.mkdir(parents=True)
        root = tmp_path / "sensitivity"
        root.mkdir()
        tgt_link = root / "raw_link"
        tgt_link.symlink_to(raw)
        with pytest.raises(ValueError, match="Path-isolation violation"):
            _assert_chat_target_isolation(
                str(tgt_link),
                sensitivity_output_root=str(root),
                raw_vectors_dir=str(raw),
            )


# =============================================================================
# 4e. Resolved-output containment inside run_sensitivity_analysis
# =============================================================================


class TestResolvedOutputContainment:
    """The driver must validate ``output_filename`` AND the final resolved path.

    Even after the basename check, a symlinked ``output_dir`` whose canonical
    target escapes the approved area must be caught before the write.
    """

    def test_driver_rejects_traversal_filename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        with pytest.raises(ValueError):
            run_sensitivity_analysis(
                raw_vectors_dir=dirs["raw"],
                chat_old_vectors_dir=dirs["chat_old"],
                chat_target_vectors_dir=dirs["chat_tgt"],
                output_dir=str(tmp_path / "sensitivity"),
                output_filename="../evil.json",
            )
        assert not (tmp_path / "evil.json").exists()

    def test_driver_rejects_absolute_filename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        with pytest.raises(ValueError):
            run_sensitivity_analysis(
                raw_vectors_dir=dirs["raw"],
                chat_old_vectors_dir=dirs["chat_old"],
                chat_target_vectors_dir=dirs["chat_tgt"],
                output_dir=str(tmp_path / "sensitivity"),
                output_filename="/etc/passwd",
            )

    def test_driver_rejects_separator_filename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        with pytest.raises(ValueError):
            run_sensitivity_analysis(
                raw_vectors_dir=dirs["raw"],
                chat_old_vectors_dir=dirs["chat_old"],
                chat_target_vectors_dir=dirs["chat_tgt"],
                output_dir=str(tmp_path / "sensitivity"),
                output_filename="sub/evil.json",
            )

    def test_driver_rejects_dot_filename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        with pytest.raises(ValueError):
            run_sensitivity_analysis(
                raw_vectors_dir=dirs["raw"],
                chat_old_vectors_dir=dirs["chat_old"],
                chat_target_vectors_dir=dirs["chat_tgt"],
                output_dir=str(tmp_path / "sensitivity"),
                output_filename=".",
            )

    def test_driver_rejects_empty_filename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        with pytest.raises(ValueError):
            run_sensitivity_analysis(
                raw_vectors_dir=dirs["raw"],
                chat_old_vectors_dir=dirs["chat_old"],
                chat_target_vectors_dir=dirs["chat_tgt"],
                output_dir=str(tmp_path / "sensitivity"),
                output_filename="",
            )

    def test_driver_rejects_symlinked_output_into_primary(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        primary = Path(dirs["chat_old"])
        link = tmp_path / "sensitivity_link"
        link.symlink_to(primary)
        with pytest.raises(ValueError, match="Path-isolation violation"):
            run_sensitivity_analysis(
                raw_vectors_dir=dirs["raw"],
                chat_old_vectors_dir=dirs["chat_old"],
                chat_target_vectors_dir=dirs["chat_tgt"],
                output_dir=str(link),
            )

    def test_driver_accepts_default_filename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        out = tmp_path / "sensitivity"
        run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(out),
        )
        assert (out / "sensitivity.json").exists()

    def test_driver_accepts_custom_basename(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        out = tmp_path / "sensitivity"
        run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(out),
            output_filename="custom_report.json",
        )
        assert (out / "custom_report.json").exists()
        assert not (out / "sensitivity.json").exists()


# =============================================================================
# 5/6. Full driver behaviour with mock vectors
# =============================================================================


def _build_full_mock_layout(tmp_path: Path) -> dict[str, str]:
    """Build a tmp layout with raw + old-chat + empty chat-target stores.

    Raw store: target + related + control all present.
    Old chat store: related + control present, TARGET ABSENT (as in real life).
    Chat target store: empty (so target chat is 'chat_missing' by default).
    """
    raw = tmp_path / "raw" / "vectors"
    chat_old = tmp_path / "primary" / "vectors"
    chat_tgt = tmp_path / "sensitivity" / "chat_target_vectors"

    for ckpt in SENSITIVITY_CHECKPOINTS:
        for layer in SENSITIVITY_LAYERS:
            raw_vecs: dict[str, ConceptVector] = {}
            chat_vecs: dict[str, ConceptVector] = {}
            for concept in SENSITIVITY_CONCEPTS:
                raw_vecs[concept] = _mock_cv(
                    concept, layer_idx=layer, seed=hash((concept, ckpt, layer, "raw"))
                )
                if concept != TARGET_CONCEPT:
                    chat_vecs[concept] = _mock_cv(
                        concept,
                        layer_idx=layer,
                        seed=hash((concept, ckpt, layer, "chat")),
                    )
            _save_mock_store(str(raw), SENSITIVITY_MODEL, ckpt, layer, raw_vecs)
            _save_mock_store(str(chat_old), SENSITIVITY_MODEL, ckpt, layer, chat_vecs)
    return {
        "raw": str(raw),
        "chat_old": str(chat_old),
        "chat_tgt": str(chat_tgt),
    }


class TestDriverBehaviour:
    def test_target_is_chat_missing_related_control_compared(
        self, tmp_path: Path
    ) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        out = tmp_path / "sensitivity"
        payload = run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(out),
        )
        s = payload["summary"]
        # 3 ckpts * 10 layers * 6 concepts = 180 entries.
        assert s["n_entries"] == 3 * 10 * 6
        # 5 related/control per (ckpt,layer) * 30 = 150 compared.
        assert s["n_compared"] == 3 * 10 * 5
        # 1 target per (ckpt,layer) * 30 = 30 chat_missing.
        assert s["n_chat_missing"] == 3 * 10 * 1
        assert s["n_raw_missing"] == 0
        assert s["n_metadata_rejected"] == 0

        target_entries = [
            e for e in payload["entries"] if e["concept"] == TARGET_CONCEPT
        ]
        assert len(target_entries) == 30
        assert all(e["status"] == "chat_missing" for e in target_entries)

        related_entries = [
            e for e in payload["entries"] if e["concept"] in RELATED_CONCEPTS
        ]
        assert all(e["status"] == "compared" for e in related_entries)
        assert all(e["cosine"] is not None for e in related_entries)
        assert all(e["norm_diff"] is not None for e in related_entries)

    def test_old_chat_store_is_read_only(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        before = _hash_tree(dirs["chat_old"])
        run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(tmp_path / "sensitivity"),
        )
        after = _hash_tree(dirs["chat_old"])
        assert before == after

    def test_output_lives_only_under_sensitivity_dir(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        out = tmp_path / "sensitivity"
        run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(out),
        )
        assert (out / "sensitivity.json").exists()
        # Nothing written into the raw store, the old chat store, or the
        # (still-empty) target dir.
        assert not (Path(dirs["chat_tgt"]) / "sensitivity.json").exists()
        assert not (Path(dirs["chat_old"]) / "sensitivity.json").exists()

    def test_raw_missing_recorded_when_raw_store_absent(self, tmp_path: Path) -> None:
        chat_old = tmp_path / "primary" / "vectors"
        chat_old.mkdir(parents=True)
        payload = run_sensitivity_analysis(
            raw_vectors_dir=str(tmp_path / "nonexistent" / "raw"),
            chat_old_vectors_dir=str(chat_old),
            chat_target_vectors_dir=str(tmp_path / "tgt"),
            output_dir=str(tmp_path / "sensitivity"),
        )
        assert payload["summary"]["n_raw_missing"] == 3 * 10 * 6
        assert payload["summary"]["n_compared"] == 0

    def test_metadata_rejected_recorded_per_entry(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        # Corrupt ONE old-chat metadata file: wrong model_name.
        bad = Path(dirs["chat_old"]) / SENSITIVITY_MODEL / "step_100" / "layer_3.json"
        meta = json.loads(bad.read_text())
        meta["model_name"] = "tampered"
        bad.write_text(json.dumps(meta))

        payload = run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(tmp_path / "sensitivity"),
        )
        # step_100/layer_3 has 5 related/control concepts -> 5 rejections.
        rejected = [e for e in payload["entries"] if e["status"] == "metadata_rejected"]
        assert len(rejected) == 5
        assert {e["checkpoint"] for e in rejected} == {"step_100"}
        assert {e["layer_idx"] for e in rejected} == {3}
        assert all(
            e["concept"] in RELATED_CONCEPTS or e["concept"] == "gender_she_vs_he"
            for e in rejected
        )
        # All other related/control entries still compared.
        assert payload["summary"]["n_compared"] == 3 * 10 * 5 - 5

    def test_payload_carries_protocol_and_limitations(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        payload = run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(tmp_path / "sensitivity"),
        )
        assert payload["protocol"]["name"] == PROTOCOL["name"]
        assert payload["protocol"]["primary"] is False
        assert payload["protocol"]["metrics"] == ["cosine", "norm_diff"]
        assert isinstance(payload["limitations"], list) and payload["limitations"]
        assert payload["scope"]["model"] == SENSITIVITY_MODEL
        assert payload["scope"]["checkpoints"] == SENSITIVITY_CHECKPOINTS

    def test_cosine_and_norm_diff_values_are_consistent(self, tmp_path: Path) -> None:
        dirs = _build_full_mock_layout(tmp_path)
        payload = run_sensitivity_analysis(
            raw_vectors_dir=dirs["raw"],
            chat_old_vectors_dir=dirs["chat_old"],
            chat_target_vectors_dir=dirs["chat_tgt"],
            output_dir=str(tmp_path / "sensitivity"),
        )
        for e in payload["entries"]:
            if e["status"] != "compared":
                continue
            assert -1.0 <= e["cosine"] <= 1.0
            assert e["norm_diff"] == pytest.approx(e["norm_raw"] - e["norm_chat"])


# =============================================================================
# 7. Gated chat-TARGET extraction
# =============================================================================


class _FakeTokenizerNoTemplate:
    pad_token = "<pad>"
    eos_token = "<eos>"


class _FakeTokenizerWithTemplate:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        return "".join(m["content"] for m in messages)


class TestGatedTargetExtraction:
    def test_fails_clearly_without_chat_template(self, tmp_path: Path) -> None:
        def loader(model_config, revision=None):
            return object(), _FakeTokenizerNoTemplate()

        with pytest.raises(ChatTemplateUnavailableError, match="apply_chat_template"):
            extract_missing_chat_target(
                chat_target_vectors_dir=str(tmp_path / "tgt"),
                sensitivity_output_root=str(tmp_path),
                load_model_and_tokenizer=loader,
                extract_fn=lambda *a, **k: {},
            )

    def test_fails_clearly_when_template_raises_on_probe(self, tmp_path: Path) -> None:
        class Bad(_FakeTokenizerWithTemplate):
            def apply_chat_template(self, *a, **k):
                raise RuntimeError("broken template")

        def loader(model_config, revision=None):
            return object(), Bad()

        with pytest.raises(ChatTemplateUnavailableError, match="probe"):
            extract_missing_chat_target(
                chat_target_vectors_dir=str(tmp_path / "tgt"),
                sensitivity_output_root=str(tmp_path),
                load_model_and_tokenizer=loader,
                extract_fn=lambda *a, **k: {},
            )

    def test_skips_checkpoints_already_present(self, tmp_path: Path) -> None:
        tgt = tmp_path / "tgt"
        # Pre-populate the target store for ALL checkpoints/layers.
        for ckpt in SENSITIVITY_CHECKPOINTS:
            for layer in SENSITIVITY_LAYERS:
                _save_mock_store(
                    str(tgt),
                    SENSITIVITY_MODEL,
                    ckpt,
                    layer,
                    {TARGET_CONCEPT: _mock_cv(TARGET_CONCEPT, layer_idx=layer)},
                )

        def loader(model_config, revision=None):
            return object(), _FakeTokenizerWithTemplate()

        calls: list[str] = []

        def extract_fn(model_config, concepts, layers, n_samples, out_dir, **kw):
            calls.append(kw.get("checkpoint", "?"))
            return {"checkpoint": kw.get("checkpoint")}

        summary = extract_missing_chat_target(
            chat_target_vectors_dir=str(tgt),
            sensitivity_output_root=str(tmp_path),
            load_model_and_tokenizer=loader,
            extract_fn=extract_fn,
        )
        assert summary["checkpoints"]["skipped_present"] == SENSITIVITY_CHECKPOINTS
        assert summary["checkpoints"]["extracted"] == []
        assert calls == []  # nothing extracted

    def test_extracts_missing_and_writes_target_vectors(self, tmp_path: Path) -> None:
        tgt = tmp_path / "tgt"

        def loader(model_config, revision=None):
            return object(), _FakeTokenizerWithTemplate()

        def extract_fn(model_config, concepts, layers, n_samples, out_dir, **kw):
            ckpt = kw["checkpoint"]
            assert concepts == [TARGET_CONCEPT]
            assert kw["use_chat_template"] is True
            for layer in layers:
                _save_mock_store(
                    out_dir,
                    model_config.name,
                    ckpt,
                    layer,
                    {TARGET_CONCEPT: _mock_cv(TARGET_CONCEPT, layer_idx=layer)},
                )
            return {"checkpoint": ckpt}

        summary = extract_missing_chat_target(
            chat_target_vectors_dir=str(tgt),
            sensitivity_output_root=str(tmp_path),
            load_model_and_tokenizer=loader,
            extract_fn=extract_fn,
        )
        assert len(summary["checkpoints"]["extracted"]) == 3
        assert summary["checkpoints"]["skipped_present"] == []
        # Files actually landed in the sensitivity-only target dir.
        for ckpt in SENSITIVITY_CHECKPOINTS:
            for layer in SENSITIVITY_LAYERS:
                assert (
                    tgt / SENSITIVITY_MODEL / ckpt / f"layer_{layer}.safetensors"
                ).exists()


# =============================================================================
# CLI smoke test (mocked extraction hooks; no real model)
# =============================================================================


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_sensitivity_analysis", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sensitivity_analysis"] = module
    spec.loader.exec_module(module)
    return module


class TestCLI:
    def test_default_raw_vectors_dir_matches_primary_experiment(self) -> None:
        cli = _load_cli()
        args = cli.parse_args([])
        assert args.raw_vectors_dir == os.path.join(
            cli.RL_ZERO_CODE_RESULTS_ROOT, "concept_vectors"
        )

    def test_cli_writes_sensitivity_json(self, tmp_path: Path) -> None:
        cli = _load_cli()
        dirs = _build_full_mock_layout(tmp_path)
        rc = cli.main(
            [
                "--raw-vectors-dir",
                dirs["raw"],
                "--chat-old-vectors-dir",
                dirs["chat_old"],
                "--chat-target-vectors-dir",
                dirs["chat_tgt"],
                "--output-dir",
                str(tmp_path / "sensitivity"),
            ]
        )
        assert rc == 0
        out = tmp_path / "sensitivity" / "sensitivity.json"
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["protocol"]["primary"] is False
        assert payload["summary"]["n_entries"] == 3 * 10 * 6

    def test_cli_refuses_output_in_primary_store(self, tmp_path: Path) -> None:
        cli = _load_cli()
        chat_old = tmp_path / "concept_dynamics_multi" / "vectors"
        chat_old.mkdir(parents=True)
        with pytest.raises(SystemExit) as ei:
            cli.main(
                [
                    "--chat-old-vectors-dir",
                    str(chat_old),
                    "--raw-vectors-dir",
                    str(tmp_path / "raw" / "vectors"),
                    "--output-dir",
                    str(tmp_path / "concept_dynamics_multi" / "sensitivity"),
                ]
            )
        assert ei.value.code == 2

    def test_cli_extract_target_chat_passes_hooks_through(self, tmp_path: Path) -> None:
        cli = _load_cli()
        out = tmp_path / "sensitivity"
        tgt = out / "tgt"
        called: list[str] = []

        def loader(model_config, revision=None):
            return object(), _FakeTokenizerWithTemplate()

        def extract_fn(model_config, concepts, layers, n_samples, out_dir, **kw):
            called.append(kw["checkpoint"])
            return {"checkpoint": kw["checkpoint"]}

        rc = cli.main(
            [
                "--extract-target-chat",
                "--chat-target-vectors-dir",
                str(tgt),
                "--output-dir",
                str(out),
            ],
            extract_fn=extract_fn,
            load_model_and_tokenizer=loader,
        )
        assert rc == 0
        assert called == SENSITIVITY_CHECKPOINTS
