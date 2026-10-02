from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import cast

import pytest
import torch
from postdyn.data import DomainPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

q1 = importlib.import_module("scripts.run_q1")
robustness = importlib.import_module("scripts.run_q1_robustness")


def _run_args(output: Path) -> list[str]:
    return [
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--output",
        str(output),
        "--domains",
        "math,code",
        "--checkpoints",
        "base,sft,dpo,rlvr",
        "--limit",
        "6",
    ]


def test_q1_writes_metrics_eigensystems_analysis_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "q1"

    assert q1.main(_run_args(output)) == 0

    rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4 * 2 * 2
    assert {tuple(row["unit"]) for row in rows} == {
        (checkpoint, layer, domain)
        for checkpoint in ("base", "sft", "dpo", "rlvr")
        for layer in (0, 1)
        for domain in ("math", "code")
    }
    eig_json = output / "eigensystems" / "base" / "0" / "math.json"
    eig_values = json.loads(eig_json.read_text())
    assert len(eig_values) == 8
    from postdyn.persistence import load_eigensystem

    values, vecs = load_eigensystem(eig_json.with_suffix(""))
    assert values.shape == (8,)
    assert vecs.shape == (8, 8)
    assert vecs.dtype == torch.float32

    summary = json.loads((output / "analysis" / "summary.json").read_text())
    assert summary["adjacent_pairs"]
    assert {"high", "mid", "low"}.issubset(summary["adjacent_pairs"][0]["subsim"])
    assert "mean" in summary["adjacent_pairs"][0]["reordering"]
    for pair in summary["adjacent_pairs"]:
        unit = pair["reordering"]["by_unit"]["layer_0/math"]
        assert len(unit["pi"]) == len(unit["D"]) == 8
        assert "subsim_bands" in unit
    for checkpoint_item in summary["checkpoints"].values():
        for key, value in checkpoint_item.items():
            if key.startswith("layer_"):
                assert set(value["vs_base"]) == {"subsim_bands"}
                assert set(value["vs_stage_final"]) == {"subsim_bands"}
    assert "vs_base" in summary["checkpoints"]["rlvr"]
    assert "vs_stage_final" in summary["checkpoints"]["sft"]

    before = (output / "metrics.jsonl").read_text()
    assert q1.main(_run_args(output)) == 0
    assert (output / "metrics.jsonl").read_text() == before


def test_q1_uses_per_domain_pool_sizes_and_manifest_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "q1"
    original = q1._load_pools

    def pools(args: object, domains: list[str], n: int) -> dict[str, DomainPool]:
        loaded = original(args, domains, n)
        return {
            "math": DomainPool(
                "math", loaded["math"].records, 32, 32, "math", 42, "math"
            ),
            "code": DomainPool(
                "code", loaded["code"].records[:20], 32, 20, "code", 42, "code"
            ),
        }

    monkeypatch.setattr(q1, "_load_pools", pools)
    args = _run_args(output)
    args[-1] = "32"
    with pytest.raises(
        SystemExit, match="rematerialize larger pools.*allow-short-pool"
    ):
        q1.main(args)
    args.append("--allow-short-pool")
    assert q1.main(args) == 0
    rows = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    assert {row["n"] for row in rows if row["domain"] == "math"} == {32}
    assert {row["n"] for row in rows if row["domain"] == "code"} == {20}
    assert {row["short_pool"] for row in rows if row["domain"] == "math"} == {False}
    assert {row["short_pool"] for row in rows if row["domain"] == "code"} == {True}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["n"] == {"math": 32, "code": 20}
    assert manifest["actual_n"] == {"math": 32, "code": 20}
    assert manifest["short_pool_domains"] == ["code"]


def test_q1_rejects_checkpoint_manifest_mismatch(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    assert q1.main(_run_args(output)) == 0
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"] = ["base"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="checkpoints"):
        q1.main(_run_args(output))


def test_q1_resume_identity_rejects_same_names_from_different_sft_schedule(
    tmp_path: Path,
) -> None:
    output = tmp_path / "q1"
    args = [
        "--family",
        "32b",
        "--scale",
        "tiny",
        "--output",
        str(output),
        "--checkpoints",
        "base,sft,dpo,rlvr",
        "--domains",
        "math",
        "--limit",
        "6",
        "--device",
        "cpu",
    ]
    assert q1.main(args) == 0

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["checkpoints"][0] == ["allenai/Olmo-3-1125-32B", "main"]

    assert q1.main(args) == 0
    changed = args + ["--sft-lr", "5e-5"]
    # Tiny scale selects only the four finals (revision "main" for every
    # sft_lr), so the checkpoint pairs match and sft_lr alone carries the
    # branch identity.
    with pytest.raises(SystemExit, match="sft_lr"):
        q1.main(changed)


def test_q1_resume_identity_stores_runtime_parameters(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    args = _run_args(output) + [
        "--dtype",
        "float32",
        "--quantization",
        "nf4",
        "--max-length",
        "128",
        "--token-budget",
        "64",
        "--attention-budget",
        "4096",
        "--batch-size",
        "4",
        "--device",
        "cpu",
    ]
    assert q1.main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sft_lr"] == "1e-4"
    assert manifest["dtype"] == "float32"
    assert manifest["quantization"] == "nf4"
    assert manifest["max_length"] == 128
    assert manifest["token_budget"] == 64
    assert manifest["attention_budget"] == 4096
    assert manifest["batch_size"] == 4
    assert manifest["device"] == "cpu"


def test_q1_passes_model_device_to_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[object] = []
    original = q1.extract_layer_hiddens

    def extract(*args: object, **kwargs: object) -> dict[int, torch.Tensor]:
        observed.append(kwargs["return_device"])
        return original(*args, **kwargs)

    monkeypatch.setattr(q1, "extract_layer_hiddens", extract)
    assert q1.main(_run_args(tmp_path / "q1") + ["--device", "cpu"]) == 0
    assert observed
    assert all(device == torch.device("cpu") for device in observed)


def test_q1_recomputes_unit_when_safetensors_is_missing(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    assert q1.main(_run_args(output)) == 0

    missing = output / "eigensystems" / "base" / "0" / "math.safetensors"
    missing.unlink()

    assert q1.main(_run_args(output)) == 0
    assert missing.is_file()


def test_extracts_only_missing_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "q1"
    args = _run_args(output) + ["--device", "cpu"]
    assert q1.main(args) == 0

    missing = output / "eigensystems" / "base" / "1" / "math.safetensors"
    missing.unlink()
    calls: list[list[int]] = []
    original = q1.extract_layer_hiddens

    def extract(*args: object, **kwargs: object) -> dict[int, torch.Tensor]:
        calls.append(cast(list[int], args[3]))
        return original(*args, **kwargs)

    monkeypatch.setattr(q1, "extract_layer_hiddens", extract)
    assert q1.main(args) == 0
    assert calls == [[1]]


def test_skips_forward_when_domain_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "q1"
    args = _run_args(output) + ["--device", "cpu"]
    assert q1.main(args) == 0
    monkeypatch.setattr(
        q1,
        "extract_layer_hiddens",
        lambda *args, **kwargs: pytest.fail("complete domain was extracted"),
    )
    assert q1.main(args) == 0


def test_analysis_eigensystem_lru(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "q1"
    args = _run_args(output) + ["--device", "cpu"]
    assert q1.main(args) == 0
    calls: list[Path] = []
    original = q1.load_eigensystem

    def load(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(q1, "load_eigensystem", load)
    checkpoints = [q1.MODEL_FAMILIES["7b"].checkpoints()[0]]
    q1._write_analysis(q1.RunDir(output), checkpoints, [0, 1], ["math", "code"])
    assert len(calls) == len({path for path in calls}) == 4


def test_analysis_output_unchanged(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    args = _run_args(output) + ["--device", "cpu"]
    assert q1.main(args) == 0
    summary = output / "analysis" / "summary.json"
    expected = summary.read_bytes()
    assert q1.main(args) == 0
    assert summary.read_bytes() == expected


def test_q1_rejects_resume_with_different_domains(tmp_path: Path) -> None:
    output = tmp_path / "q1"
    assert q1.main(_run_args(output)) == 0

    changed = _run_args(output)
    changed[changed.index("math,code")] = "math"
    with pytest.raises(SystemExit, match="mismatch.*fresh --output"):
        q1.main(changed)


def test_robustness_extracts_all_layers_once_per_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[int]] = []
    original = robustness.extract_layer_hiddens

    def extract(*args: object, **kwargs: object) -> dict[int, torch.Tensor]:
        layers = cast(list[int], args[3])
        calls.append(layers)
        return original(*args, **kwargs)

    monkeypatch.setattr(robustness, "extract_layer_hiddens", extract)
    args = [
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--checkpoint",
        "rlvr",
        "--domain",
        "math",
        "--repeats",
        "3",
        "--output",
        str(tmp_path),
        "--limit",
        "6",
    ]

    assert robustness.main(args) == 0
    assert calls == [[0, 1], [0, 1], [0, 1]]


def test_robustness_writes_repeat_files_and_spread_summary(tmp_path: Path) -> None:
    output = tmp_path / "robustness"
    args = [
        "--family",
        "7b",
        "--scale",
        "tiny",
        "--checkpoint",
        "rlvr",
        "--domain",
        "math",
        "--repeats",
        "3",
        "--output",
        str(output),
        "--limit",
        "6",
    ]

    assert robustness.main(args) == 0

    root = output / "7b" / "rlvr" / "math"
    record_sets = []
    for repeat in range(3):
        payload = json.loads((root / f"repeat_{repeat}.json").read_text())
        assert payload["repeat"] == repeat
        assert len(payload["eigenvalues"]) == 2
        assert "rank_stability" in payload
        assert "subsim_low_vs_first" in payload
        record_sets.append(set(payload["record_ids"]))
    assert len({frozenset(record_set) for record_set in record_sets}) == 3
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary["spread"]["std_effective_rank"]) == 2
    assert "mean_subsim_high_across_pairs" in summary["spread"]
    assert "mean_subsim_low_across_pairs" in summary["spread"]


def test_cli_validation_rejects_bad_family_and_small_repeat_count() -> None:
    with pytest.raises(SystemExit):
        q1.parse_args(["--family", "bad", "--scale", "tiny"])
    with pytest.raises(SystemExit):
        robustness.parse_args(["--family", "7b", "--repeats", "1"])


def test_q1_prefetch_overlaps_next_download_with_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    checkpoints = q1.MODEL_FAMILIES["7b"].checkpoints()
    selected = checkpoints[:3]

    monkeypatch.setattr(q1, "_select_checkpoints", lambda args, family: selected)

    def fake_model(args, checkpoint):
        events.append(f"load:{checkpoint.name}")
        return object(), object()

    def fake_extract(model, tokenizer, prompts, layers, *args, **kwargs):
        events.append(f"extract:{prompts[0].split()[0]}")
        return {layer: torch.zeros(1, 8) for layer in layers}

    def fake_prefetch(checkpoint):
        events.append(f"start:{checkpoint.name}")

        def join():
            events.append(f"join:{checkpoint.name}")
            return True

        return join

    monkeypatch.setattr(q1, "_checkpoint_model", fake_model)
    monkeypatch.setattr(q1, "extract_layer_hiddens", fake_extract)
    monkeypatch.setattr(q1, "start_prefetch", fake_prefetch)
    monkeypatch.setattr(q1, "release_model", lambda model: None)
    monkeypatch.setattr(q1, "prune_revision_cache", lambda checkpoint: None)
    monkeypatch.setattr(
        q1,
        "_should_prefetch",
        lambda args, index, checkpoints: index + 1 < len(checkpoints),
    )

    args = _run_args(tmp_path / "q1")
    args += ["--prefetch", "next"]
    assert q1.main(args) == 0
    assert events == [
        "load:base",
        "start:sft_step1000",
        "extract:math",
        "extract:code",
        "join:sft_step1000",
        "load:sft_step1000",
        "start:sft_step6000",
        "extract:math",
        "extract:code",
        "join:sft_step6000",
        "load:sft_step6000",
        "extract:math",
        "extract:code",
    ]


def test_q1_prefetch_none_never_starts(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    checkpoints = q1.MODEL_FAMILIES["7b"].checkpoints()
    monkeypatch.setattr(q1, "_select_checkpoints", lambda args, family: checkpoints[:2])
    monkeypatch.setattr(
        q1,
        "_checkpoint_model",
        lambda args, checkpoint: (
            events.append(f"load:{checkpoint.name}") or (object(), object())
        ),
    )
    monkeypatch.setattr(
        q1,
        "start_prefetch",
        lambda checkpoint: events.append("start") or (lambda: True),
    )
    monkeypatch.setattr(
        q1,
        "extract_layer_hiddens",
        lambda model, tokenizer, prompts, layers, *a, **k: {
            layer: torch.zeros(1, 8) for layer in layers
        },
    )
    monkeypatch.setattr(q1, "release_model", lambda model: None)
    monkeypatch.setattr(q1, "prune_revision_cache", lambda checkpoint: None)

    assert q1.main(_run_args(tmp_path / "q1") + ["--prefetch", "none"]) == 0
    assert "start" not in events


def test_robustness_pool_smaller_than_requested_falls_back_to_90_percent(
    tmp_path: Path, monkeypatch
) -> None:
    from postdyn.data import PromptRecord

    records = tuple(
        PromptRecord(f"gr-{i}", f"general prompt {i}", "science") for i in range(100)
    )
    pool = DomainPool("general_reasoning", records, 30720, 100, "rev", 42, "fp")
    monkeypatch.setattr(robustness, "_pool", lambda args, n: pool)
    from postdyn.config import CheckpointRef

    rlvr_final = CheckpointRef("rlvr", "allenai/Olmo-3-7B-Think", "main", "rlvr")
    monkeypatch.setattr(
        robustness, "_checkpoint", lambda args: (rlvr_final, (object(), object()))
    )
    monkeypatch.setattr(
        robustness,
        "extract_layer_hiddens",
        lambda model, tokenizer, prompts, layers, *a, **k: {
            layer: torch.eye(len(prompts), 8) for layer in layers
        },
    )

    args = [
        "--family",
        "7b",
        "--repeats",
        "3",
        "--domain",
        "general_reasoning",
        "--scale",
        "full",
        "--output",
        str(tmp_path / "rob"),
    ]
    assert robustness.main(args) == 0
    repeat_n = set()
    ids = []
    for repeat in range(3):
        payload = json.loads(
            (
                tmp_path
                / "rob"
                / "7b"
                / "rlvr"
                / "general_reasoning"
                / f"repeat_{repeat}.json"
            ).read_text()
        )
        repeat_n.add(payload["n"])
        ids.append(set(payload["record_ids"]))
    assert repeat_n == {90}
    assert len(ids[0]) == 90
    assert ids[0] != ids[1]
