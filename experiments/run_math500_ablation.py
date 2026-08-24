#!/usr/bin/env python3
"""Resumable first-50 MATH-500 ablations for the signed 7B subspaces."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import EXPERIMENT_LAYERS_7B, OLMO3_VARIANTS
from src.downstream_eval import (
    GreedyGenerator,
    TokenizerLike,
    compare_singleton_and_batch_token_ids,
)
from src.math500_eval import (
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_QUANTIZATION,
    evaluate_first50,
    load_authoritative_summary,
    load_first50,
    write_atomically,
)
from src.math500_ablation_validator import (
    NF4_CONFIG,
    STATIC_NF4_PROVENANCE,
    _validate_extraction_manifest,
)
from src.residual_ablation import residual_stream_ablation
from src.think_sft_differential_experiment import (
    FAMILY_THINK,
    SCALE_32B,
    SCALE_7B,
    TrajectoryConfig,
    layers_for_scale,
    root_for_trajectory,
    trajectory_config,
    validate_root_ownership,
    claim_root_ownership,
    ensure_root_ownership,
)

collect_valid_conditions = cast(
    Callable[..., tuple[list[dict[str, object]], list[str]]],
    importlib.import_module("src.math500_ablation_validator").collect_valid_conditions,
)

DEFAULT_DATASET = "datasets/math500.json"
DEFAULT_RESULT_ROOT = "results/math500_ablation_first50"
CONCEPT = "math_vs_text"
CONTRACT = "residual-ablation-all-tokens-v1"
RUNTIME_FILENAME = "runtime_provenance.json"
CANONICAL_32B_MAX_NEW_TOKENS = 2048


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed not in (1, 2):
        raise argparse.ArgumentTypeError("must be either 1 or 2")
    return parsed


def _validate_generation_batch_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise ValueError("generation batch size must be either 1 or 2")
    return value


class _Model(Protocol):
    def eval(self) -> object: ...
    def get_input_embeddings(self) -> object: ...
    def generate(
        self,
        input_ids: "torch.Tensor",
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        num_beams: int = 1,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> "torch.Tensor": ...


@dataclass(frozen=True)
class BasisArtifact:
    layer: int
    setup_signature: str
    sidecar_sha256: str
    tensor_sha256: str
    tensors: Mapping[str, object]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class AblationRunConfig:
    scale: str
    trajectory: str
    trajectory_config: TrajectoryConfig
    checkpoints: tuple[str, ...]
    revisions: Mapping[str, str]
    artifact_root: Path
    result_root: Path
    project_root: Path | None = None

    @property
    def model_key(self) -> str:
        return self.trajectory_config.model_key

    def revision_for(self, checkpoint: str) -> str:
        return self.revisions[checkpoint]


def _default_result_root(trajectory: str, scale: str = SCALE_7B) -> Path:
    if scale == SCALE_7B and trajectory == "sft":
        return Path(DEFAULT_RESULT_ROOT)
    scale_suffix = "" if scale == SCALE_7B else f"_{scale}"
    return Path("results") / f"math500_ablation_first50{scale_suffix}_{trajectory}"


def resolve_run_config(args: argparse.Namespace) -> AblationRunConfig:
    """Resolve a scale-specific trajectory before any model loading."""
    scale = cast(str, args.scale)
    trajectory = cast(str, args.trajectory)
    if scale == SCALE_32B and args.max_new_tokens != CANONICAL_32B_MAX_NEW_TOKENS:
        raise ValueError("32b requires max_new_tokens=2048 (--max-new-tokens 2048)")
    config = trajectory_config(FAMILY_THINK, scale, trajectory)
    project_root = None if args.project_root is None else Path(args.project_root)
    if project_root is not None and args.dataset == DEFAULT_DATASET:
        args.dataset = str(project_root / DEFAULT_DATASET)
    configured_checkpoints = tuple(config.checkpoints)
    requested = args.checkpoints
    checkpoints = configured_checkpoints if requested is None else tuple(requested)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("duplicate checkpoint arguments are not allowed")
    unknown = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint not in configured_checkpoints
    ]
    if unknown:
        raise ValueError(f"unknown {trajectory} checkpoint(s): {', '.join(unknown)}")
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    layers = tuple(args.layers)
    canonical_layers = tuple(layers_for_scale(scale))
    if args.layers == EXPERIMENT_LAYERS_7B and scale == SCALE_32B:
        layers = canonical_layers
    unknown_layers = [layer for layer in layers if layer not in canonical_layers]
    if unknown_layers:
        raise ValueError(
            f"unknown {scale} layer(s): {', '.join(map(str, unknown_layers))}"
        )
    args.layers = list(layers)
    if len(set(layers)) != len(layers):
        raise ValueError("duplicate layer arguments are not allowed")

    canonical_artifact_root = root_for_trajectory(
        FAMILY_THINK, scale, trajectory, project_root=project_root
    )
    artifact_arg = args.artifact_root
    artifact_root = (
        canonical_artifact_root if artifact_arg is None else Path(artifact_arg)
    )
    for other_trajectory in ("sft", "rlvr", "sft_lr_1e-4", "sft_lr_5e-5"):
        if other_trajectory == trajectory:
            continue
        try:
            other_artifact_root = root_for_trajectory(
                FAMILY_THINK, scale, other_trajectory, project_root=project_root
            )
        except ValueError:
            continue
        if artifact_root.resolve() == other_artifact_root.resolve():
            raise ValueError(
                f"artifact root {artifact_root} belongs to {other_trajectory}, not {trajectory}"
            )

    canonical_result_root = (
        _default_result_root(trajectory, scale)
        if project_root is None
        else Path(project_root) / _default_result_root(trajectory, scale)
    )
    result_arg = args.result_root
    result_root = canonical_result_root if result_arg is None else Path(result_arg)
    if artifact_root.resolve() == result_root.resolve():
        raise ValueError("artifact root and result root must differ")
    for other_scale in (SCALE_7B, SCALE_32B):
        trajectories = (
            ("sft", "rlvr")
            if other_scale == SCALE_7B
            else ("sft_lr_1e-4", "sft_lr_5e-5", "rlvr")
        )
        for other_trajectory in trajectories:
            if other_scale == scale and other_trajectory == trajectory:
                continue
            if (
                result_root.resolve()
                == (
                    _default_result_root(other_trajectory, other_scale)
                    if project_root is None
                    else Path(project_root)
                    / _default_result_root(other_trajectory, other_scale)
                ).resolve()
            ):
                owner = (
                    other_trajectory
                    if other_scale == scale
                    else f"{other_scale}/{other_trajectory}"
                )
                raise ValueError(
                    f"result root {result_root} belongs to {owner}, not "
                    f"{scale}/{trajectory}"
                )

    revision_arg = args.revision
    revisions = dict(config.revisions)
    if revision_arg is not None:
        if len(checkpoints) != 1:
            raise ValueError("--revision is only valid with one selected checkpoint")
        checkpoint = checkpoints[0]
        expected_revision = config.revisions[checkpoint]
        if revision_arg != expected_revision:
            raise ValueError(
                f"revision for {trajectory}/{checkpoint} must be {expected_revision}, "
                f"not {revision_arg}"
            )
        revisions[checkpoint] = revision_arg

    if scale == SCALE_32B:
        if args.dtype != "bfloat16":
            raise ValueError("32b requires bfloat16 dtype (--dtype bfloat16)")
        if args.quantization == DEFAULT_QUANTIZATION:
            args.quantization = "nf4"
        if args.quantization != "nf4":
            raise ValueError(
                "32b requires the NF4 quantized loader (--quantization nf4)"
            )
    return AblationRunConfig(
        scale,
        trajectory,
        config,
        checkpoints,
        {checkpoint: revisions[checkpoint] for checkpoint in checkpoints},
        artifact_root,
        result_root,
        project_root,
    )


def load_model_and_tokenizer(
    model_key: str,
    revision: str,
    dtype: str = DEFAULT_DTYPE,
    quantization: str = DEFAULT_QUANTIZATION,
) -> tuple[_Model, TokenizerLike]:
    if model_key in {"olmo3-32b-think-sft", "olmo3-32b-think-rlvr"}:
        raise ValueError("generic MATH loader cannot load 32B; use the NF4 loader")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    if quantization not in {"none", "4bit", "8bit"}:
        raise ValueError(f"unsupported quantization: {quantization!r}")
    hf_id = OLMO3_VARIANTS[model_key].hf_id
    raw_tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if getattr(raw_tokenizer, "pad_token", None) is None:
        setattr(raw_tokenizer, "pad_token", getattr(raw_tokenizer, "eos_token", None))
    kwargs: dict[str, object] = {
        "revision": revision,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if quantization == "none":
        kwargs["dtype"] = dtype_map[dtype]
    else:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit"
        )
    model = cast(
        _Model, cast(object, AutoModelForCausalLM.from_pretrained(hf_id, **kwargs))
    )
    model.eval()
    return model, cast(TokenizerLike, cast(object, raw_tokenizer))


def load_model_for_run(
    model_key: str,
    revision: str,
    dtype: str,
    quantization: str,
    *,
    project_root: Path | None = None,
) -> tuple[_Model, TokenizerLike, Mapping[str, object] | None]:
    """Load one checkpoint, using the immutable NF4 loader for 32B."""
    if model_key == "olmo3-32b-think-sft" or model_key == "olmo3-32b-think-rlvr":
        if dtype != "bfloat16" or quantization != "nf4":
            raise ValueError("32b requires bfloat16 dtype and NF4 quantization")
        _require_canonical_7b(project_root=project_root)
        from src.quantized_model_loader import load_olmo3_32b_think

        loaded = load_olmo3_32b_think(
            model_id=OLMO3_VARIANTS[model_key].hf_id,
            revision=revision,
        )
        diagnostics = loaded.diagnostics.as_dict()
        return (
            cast(_Model, loaded.model),
            cast(TokenizerLike, loaded.tokenizer),
            {
                "loader": "load_olmo3_32b_think",
                "nf4_config": dict(NF4_CONFIG),
                "diagnostics": diagnostics,
            },
        )
    model, tokenizer = load_model_and_tokenizer(
        model_key, revision, dtype, quantization
    )
    return model, tokenizer, None


def _require_canonical_7b(*, project_root: Path | None = None) -> None:
    require = importlib.import_module(
        "src.cross_pipeline_integrity"
    ).require_canonical_7b
    if project_root is None:
        require()
    else:
        require(project_root=project_root)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _artifact_paths(
    root: Path, model: str, checkpoint: str, layer: int
) -> tuple[Path, Path]:
    base = root / "U" / model / checkpoint / f"layer_{layer}" / CONCEPT
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def load_and_validate_basis(
    root: Path,
    model: str,
    checkpoint: str,
    revision: str,
    layer: int,
    expected_d_model: int = 4096,
) -> BasisArtifact:
    """Load one sidecar/tensor pair and reject any provenance or geometry mismatch."""
    import torch
    from safetensors.torch import load_file

    tensor_path, sidecar_path = _artifact_paths(root, model, checkpoint, layer)
    sidecar_bytes = sidecar_path.read_bytes()
    metadata_obj = json.loads(sidecar_bytes.decode("utf-8"))
    if not isinstance(metadata_obj, dict):
        raise ValueError(f"basis sidecar is not an object: {sidecar_path}")
    metadata: dict[str, object] = metadata_obj
    required = {
        "model",
        "checkpoint",
        "revision",
        "layer",
        "setup_signature",
        "d_model",
    }
    if not required.issubset(metadata):
        raise ValueError(f"basis sidecar missing provenance fields: {sidecar_path}")
    expected = {
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "layer": layer,
        "d_model": expected_d_model,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"basis {sidecar_path} has {key}={metadata.get(key)!r}, expected {value!r}"
            )
    setup_signature = metadata.get("setup_signature")
    if not isinstance(setup_signature, str) or not setup_signature:
        raise ValueError(f"basis sidecar has no setup_signature: {sidecar_path}")
    if expected_d_model == 5120:
        if metadata.get("loader_provenance") != STATIC_NF4_PROVENANCE:
            raise ValueError(
                f"basis sidecar has noncanonical static NF4 provenance: {sidecar_path}"
            )
        _, manifest_error = _validate_extraction_manifest(
            root, model, checkpoint, revision, layer, cast(str, setup_signature)
        )
        if manifest_error is not None:
            raise ValueError(manifest_error)
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {"U_pos", "U_neg", "eigenvalues_pos", "eigenvalues_neg"}:
        raise ValueError(f"unexpected tensor keys in {tensor_path}")
    for name in ("U_pos", "U_neg"):
        tensor = tensors[name]
        if tensor.ndim != 2 or tuple(tensor.shape[:1]) != (expected_d_model,):
            raise ValueError(
                f"{name} must have shape ({expected_d_model}, K), got {tuple(tensor.shape)}"
            )
        if not torch.is_floating_point(tensor) or not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"{name} must be finite floating point")
        gram = tensor.float().T @ tensor.float()
        identity = torch.eye(tensor.shape[1], dtype=gram.dtype)
        if not torch.allclose(gram, identity, atol=1e-4, rtol=1e-4):
            raise ValueError(f"{name} columns are not orthonormal")
        shape_key = f"{name.lower()}_shape"
        if metadata.get(shape_key) != list(tensor.shape):
            raise ValueError(f"{name} shape disagrees with sidecar")
        retained_key = "k_pos" if name == "U_pos" else "k_neg"
        if metadata.get(retained_key) != tensor.shape[1]:
            raise ValueError(f"{name} retained dimension disagrees with sidecar")
    for name, retained_key in (
        ("eigenvalues_pos", "k_pos"),
        ("eigenvalues_neg", "k_neg"),
    ):
        eigenvalues = tensors[name]
        if eigenvalues.ndim != 1 or eigenvalues.shape[0] < int(
            cast(int, metadata[retained_key])
        ):
            raise ValueError(f"{name} does not contain the retained spectrum")
    return BasisArtifact(
        layer,
        setup_signature,
        _sha256_bytes(sidecar_bytes),
        _sha256_bytes(tensor_path.read_bytes()),
        tensors,
        metadata,
    )


def _condition_identity(
    *,
    model: str,
    checkpoint: str,
    revision: str,
    condition: str,
    basis: BasisArtifact | None,
    dataset: Path,
    max_new_tokens: int,
    dtype: str,
    quantization: str,
    runtime_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ablation_contract": CONTRACT,
        "model": model,
        "revision": revision,
        "checkpoint": checkpoint,
        "condition": condition,
        "dataset": str(dataset),
        "generation": {
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "quantization": quantization,
            "greedy": True,
        },
    }
    if basis is not None:
        payload["basis"] = {
            "layer": basis.layer,
            "setup_signature": basis.setup_signature,
            "sidecar_sha256": basis.sidecar_sha256,
            "tensor_sha256": basis.tensor_sha256,
            "model": basis.metadata["model"],
            "checkpoint": basis.metadata["checkpoint"],
            "revision": basis.metadata["revision"],
        }
    else:
        payload["basis"] = None
    if runtime_provenance is not None:
        payload["runtime_provenance"] = dict(runtime_provenance)
    return payload


def _condition_dir(root: Path, checkpoint: str, condition: str) -> Path:
    safe = condition.replace("/", "_")
    return root / "checkpoints" / checkpoint / safe


def _runtime_path(root: Path, checkpoint: str) -> Path:
    return root / "checkpoints" / checkpoint / RUNTIME_FILENAME


def _read_runtime_provenance(
    root: Path, checkpoint: str
) -> Mapping[str, object] | None:
    try:
        value = json.loads(_runtime_path(root, checkpoint).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def run_checkpoint(
    *,
    args: argparse.Namespace,
    run_config: AblationRunConfig,
    checkpoint: str,
    load_model: Callable[[str, str, str, str], tuple[object, ...]] = load_model_for_run,
) -> list[dict[str, object]]:
    cfg = OLMO3_VARIANTS[run_config.model_key]
    if (
        run_config.scale == SCALE_32B
        and args.max_new_tokens != CANONICAL_32B_MAX_NEW_TOKENS
    ):
        raise ValueError("32b requires max_new_tokens=2048 (--max-new-tokens 2048)")
    if run_config.scale == SCALE_32B:
        _require_canonical_7b(project_root=run_config.project_root)
        _validate_32b_publication(run_config)
    revision = run_config.revision_for(checkpoint)
    layers = tuple(args.layers)
    (run_config.result_root / "checkpoints" / checkpoint / "summary.json").unlink(
        missing_ok=True
    )
    bases = {
        layer: load_and_validate_basis(
            run_config.artifact_root,
            cfg.name,
            checkpoint,
            revision,
            layer,
            expected_d_model=cfg.d_model,
        )
        for layer in layers
    }
    setup_signatures = {basis.setup_signature for basis in bases.values()}
    if len(setup_signatures) != 1:
        raise ValueError(f"basis setup signatures disagree for {checkpoint}")
    root = run_config.result_root
    dataset_path = Path(cast(str, args.dataset))
    max_new_tokens = int(args.max_new_tokens)
    dtype = cast(str, args.dtype)
    quantization = cast(str, args.quantization)
    runtime_provenance = (
        _read_runtime_provenance(root, checkpoint)
        if run_config.scale == SCALE_32B
        else None
    )
    force = bool(args.force)
    requested_batch_size = _validate_generation_batch_size(
        getattr(args, "generation_batch_size", 1)
    )
    conditions: list[tuple[str, int | None, str | None, BasisArtifact | None]] = [
        ("baseline", None, None, None)
    ]
    conditions.extend(
        (f"layer_{layer}_{sign}", layer, sign, bases[layer])
        for layer in layers
        for sign in ("U_pos", "U_neg")
    )
    identities = {
        condition: _condition_identity(
            model=cfg.name,
            checkpoint=checkpoint,
            revision=revision,
            condition=condition,
            basis=basis,
            dataset=dataset_path,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            quantization=quantization,
            runtime_provenance=runtime_provenance,
        )
        for condition, _, _, basis in conditions
    }
    if not force and (run_config.scale != SCALE_32B or runtime_provenance is not None):
        cached = [
            load_authoritative_summary(
                output_dir=_condition_dir(root, checkpoint, condition),
                model=cfg.hf_id,
                model_key=run_config.model_key,
                revision=revision,
                dataset_path=dataset_path,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                quantization=quantization,
                experiment_identity=identity,
            )
            for condition, identity in identities.items()
        ]
        if all(summary is not None for summary in cached):
            all_records, errors = collect_valid_conditions(
                root,
                trajectory=run_config.trajectory,
                dataset_path=dataset_path,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                quantization=quantization,
                artifact_root=run_config.artifact_root,
                scale=run_config.scale,
                selected_checkpoints=(checkpoint,),
                selected_layers=layers,
            )
            if errors:
                raise ValueError("validation failed:\n" + "\n".join(errors))
            records = [
                record
                for record in all_records
                if record.get("checkpoint") == checkpoint
            ]
            write_atomically(
                root / "checkpoints" / checkpoint / "summary.json",
                {"checkpoint": checkpoint, "conditions": records},
            )
            print(
                f"{checkpoint} batching disabled (fully cached; "
                f"requested={requested_batch_size}, effective=1)",
                flush=True,
            )
            return records
    summaries: list[dict[str, object]] = []
    loaded: tuple[object, ...] = ()
    model: _Model | None = None
    tokenizer: TokenizerLike | None = None
    generator: GreedyGenerator | None = None
    try:
        if load_model is load_model_for_run:
            default_loader = cast(Callable[..., tuple[object, ...]], load_model)
            loaded = default_loader(
                run_config.model_key,
                revision,
                args.dtype,
                args.quantization,
                project_root=run_config.project_root,
            )
        else:
            loaded = load_model(
                run_config.model_key, revision, args.dtype, args.quantization
            )
        if len(loaded) == 2:
            model, tokenizer = cast(tuple[_Model, TokenizerLike], loaded)
            loaded_runtime: Mapping[str, object] | None = runtime_provenance
        else:
            model, tokenizer, loaded_runtime = cast(
                tuple[_Model, TokenizerLike, Mapping[str, object] | None], loaded
            )
        if run_config.scale == SCALE_32B and loaded_runtime is None:
            raise ValueError("32b loader did not return runtime provenance")
        runtime_provenance = loaded_runtime
        if runtime_provenance is not None:
            write_atomically(_runtime_path(root, checkpoint), runtime_provenance)
        assert model is not None and tokenizer is not None
        generator = GreedyGenerator(
            model=model, tokenizer=tokenizer, device=_model_device(model)
        )
        effective_batch_size = 1
        if requested_batch_size > 1:
            canary_prompts = [
                item.problem for item in load_first50(dataset_path)[0][:2]
            ]
            if not layers:
                raise ValueError("batch canary requires at least one selected layer")
            representative_basis = bases[layers[0]]
            try:
                baseline_ok = compare_singleton_and_batch_token_ids(
                    tokenizer,
                    generator,
                    generator,
                    canary_prompts,
                    max_new_tokens=max_new_tokens,
                )
                with residual_stream_ablation(
                    model,
                    layer=layers[0],
                    basis=representative_basis.tensors["U_pos"],
                ):
                    ablated_ok = compare_singleton_and_batch_token_ids(
                        tokenizer,
                        generator,
                        generator,
                        canary_prompts,
                        max_new_tokens=max_new_tokens,
                    )
            except Exception as exc:
                print(
                    f"{checkpoint} batching fallback: canary error "
                    f"{type(exc).__name__}: {exc}; requested={requested_batch_size}, "
                    "effective=1",
                    flush=True,
                )
            else:
                if baseline_ok and ablated_ok:
                    effective_batch_size = requested_batch_size
                    print(
                        f"{checkpoint} batching enabled (requested={requested_batch_size}, "
                        f"effective={effective_batch_size})",
                        flush=True,
                    )
                else:
                    failed = (
                        "baseline" if not baseline_ok else "representative ablation"
                    )
                    print(
                        f"{checkpoint} batching fallback: {failed} canary mismatch; "
                        f"requested={requested_batch_size}, effective=1",
                        flush=True,
                    )
        else:
            print(
                f"{checkpoint} batching disabled (requested=1, effective=1)",
                flush=True,
            )
        for condition, layer, sign, basis in conditions:
            identity = _condition_identity(
                model=cfg.name,
                checkpoint=checkpoint,
                revision=revision,
                condition=condition,
                basis=basis,
                dataset=dataset_path,
                max_new_tokens=max_new_tokens,
                dtype=dtype,
                quantization=quantization,
                runtime_provenance=runtime_provenance,
            )
            progress = lambda message: print(
                f"{checkpoint}/{condition} {message}", flush=True
            )
            if basis is None:
                summary = evaluate_first50(
                    model=cfg.hf_id,
                    model_key=cfg.name,
                    revision=revision,
                    dataset_path=dataset_path,
                    output_dir=_condition_dir(root, checkpoint, condition),
                    tokenizer=tokenizer,
                    generator=generator,
                    max_new_tokens=max_new_tokens,
                    batch_size=effective_batch_size,
                    dtype=dtype,
                    quantization=quantization,
                    force=force,
                    progress=progress,
                    experiment_identity=identity,
                )
            else:
                assert layer is not None and sign is not None
                tensor = cast(object, basis.tensors[sign])
                with residual_stream_ablation(model, layer=layer, basis=tensor):
                    summary = evaluate_first50(
                        model=cfg.hf_id,
                        model_key=cfg.name,
                        revision=revision,
                        dataset_path=dataset_path,
                        output_dir=_condition_dir(root, checkpoint, condition),
                        tokenizer=tokenizer,
                        generator=generator,
                        max_new_tokens=max_new_tokens,
                        batch_size=effective_batch_size,
                        dtype=dtype,
                        quantization=quantization,
                        force=force,
                        progress=progress,
                        experiment_identity=identity,
                    )
            record = summary.to_dict()
            record["condition"] = condition
            record["checkpoint"] = checkpoint
            record["basis_provenance"] = identity["basis"]
            summaries.append(record)
        all_records, errors = collect_valid_conditions(
            root,
            trajectory=run_config.trajectory,
            dataset_path=dataset_path,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            quantization=quantization,
            artifact_root=run_config.artifact_root,
            scale=run_config.scale,
            selected_checkpoints=(checkpoint,),
            selected_layers=layers,
        )
        if errors:
            raise ValueError("validation failed:\n" + "\n".join(errors))
        checkpoint_records = [
            record for record in all_records if record.get("checkpoint") == checkpoint
        ]
        write_atomically(
            root / "checkpoints" / checkpoint / "summary.json",
            {"checkpoint": checkpoint, "conditions": checkpoint_records},
        )
        summaries = checkpoint_records
    finally:
        del generator, model, tokenizer, loaded, bases, conditions
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    return summaries


def _model_device(model: _Model) -> str:
    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    device = getattr(weight, "device", None)
    if device is not None and str(device) != "meta":
        return str(device)
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, Mapping):
        for name in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            mapped = device_map.get(name)
            if mapped is not None and str(mapped) != "meta":
                return str(mapped)
    try:
        parameters = getattr(model, "parameters")
        parameter = next(parameters())
    except (AttributeError, StopIteration):
        parameter = None
    parameter_device = getattr(parameter, "device", None)
    if parameter_device is not None and str(parameter_device) != "meta":
        return str(parameter_device)
    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) != "meta":
        return str(model_device)
    raise ValueError("no concrete execution device is available for model inputs")


def _validate_32b_publication(run_config: AblationRunConfig) -> None:
    from src.think_32b_differential_validator import (
        validate_full_canonical_publication,
    )

    publication = validate_full_canonical_publication(
        run_config.artifact_root, run_config.trajectory
    )
    if not publication.ok:
        raise ValueError(
            "32B extraction publication validation failed: " + publication.errors[0]
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable first-50 MATH-500 signed residual ablations"
    )
    parser.add_argument("--scale", choices=(SCALE_7B, SCALE_32B), default=SCALE_7B)
    parser.add_argument(
        "--trajectory",
        choices=("sft", "rlvr", "sft_lr_1e-4", "sft_lr_5e-5"),
        default="sft",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=EXPERIMENT_LAYERS_7B)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--generation-batch-size",
        type=_positive_int,
        default=1,
        help="execution batch size, supported values 1 or 2 (default: 1)",
    )
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    try:
        _validate_generation_batch_size(getattr(args, "generation_batch_size", 1))
    except ValueError as exc:
        print(f"ERROR: --generation-batch-size {exc}", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0:
        print("ERROR: --max-new-tokens must be positive", file=sys.stderr)
        return 2
    try:
        run_config = resolve_run_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        validate_root_ownership(
            run_config.artifact_root,
            family=FAMILY_THINK,
            scale=run_config.scale,
            trajectory=run_config.trajectory,
            model_key=run_config.model_key,
            checkpoints=list(run_config.checkpoints),
            revisions=dict(run_config.revisions),
            purpose="extraction",
            canonical=(
                run_config.artifact_root.resolve()
                == root_for_trajectory(
                    FAMILY_THINK,
                    run_config.scale,
                    run_config.trajectory,
                    project_root=run_config.project_root,
                ).resolve()
            ),
        )
        validate_root_ownership(
            run_config.result_root,
            family=FAMILY_THINK,
            scale=run_config.scale,
            trajectory=run_config.trajectory,
            model_key=run_config.model_key,
            checkpoints=list(run_config.checkpoints),
            revisions=dict(run_config.revisions),
            purpose="math500",
            canonical=(
                run_config.result_root.resolve()
                == (
                    _default_result_root(run_config.trajectory, run_config.scale)
                    if run_config.project_root is None
                    else run_config.project_root
                    / _default_result_root(run_config.trajectory, run_config.scale)
                ).resolve()
            ),
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if run_config.scale == SCALE_32B:
        try:
            _validate_32b_publication(run_config)
            require = importlib.import_module(
                "src.cross_pipeline_integrity"
            ).require_canonical_7b
            if run_config.project_root is None:
                require()
            else:
                require(project_root=run_config.project_root)
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    canonical_artifact = root_for_trajectory(
        FAMILY_THINK,
        run_config.scale,
        run_config.trajectory,
        project_root=run_config.project_root,
    ).resolve()
    canonical_result = (
        (
            run_config.project_root
            / _default_result_root(run_config.trajectory, run_config.scale)
        )
        if run_config.project_root is not None
        else _default_result_root(run_config.trajectory, run_config.scale)
    ).resolve()
    for root, purpose, canonical_root in (
        (run_config.artifact_root, "extraction", canonical_artifact),
        (run_config.result_root, "math500", canonical_result),
    ):
        if root.resolve() != canonical_root:
            claim_root_ownership(
                root,
                family=FAMILY_THINK,
                scale=run_config.scale,
                trajectory=run_config.trajectory,
                model_key=run_config.model_key,
                checkpoints=list(run_config.checkpoints),
                revisions=dict(run_config.revisions),
                purpose=purpose,
            )
    aggregate: list[dict[str, object]] = []
    run_config.result_root.joinpath("aggregate.json").unlink(missing_ok=True)
    for index, checkpoint in enumerate(run_config.checkpoints):
        try:
            run_checkpoint(
                args=args,
                run_config=run_config,
                checkpoint=checkpoint,
            )
            aggregate, errors = collect_valid_conditions(
                run_config.result_root,
                trajectory=run_config.trajectory,
                dataset_path=Path(cast(str, args.dataset)),
                max_new_tokens=int(args.max_new_tokens),
                dtype=cast(str, args.dtype),
                quantization=cast(str, args.quantization),
                artifact_root=run_config.artifact_root,
                scale=run_config.scale,
                selected_checkpoints=run_config.checkpoints[: index + 1],
                selected_layers=tuple(args.layers),
            )
            if errors:
                raise ValueError("validation failed:\n" + "\n".join(errors))
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    try:
        aggregate, errors = collect_valid_conditions(
            run_config.result_root,
            trajectory=run_config.trajectory,
            dataset_path=Path(cast(str, args.dataset)),
            max_new_tokens=int(args.max_new_tokens),
            dtype=cast(str, args.dtype),
            quantization=cast(str, args.quantization),
            artifact_root=run_config.artifact_root,
            scale=run_config.scale,
            selected_checkpoints=run_config.checkpoints,
            selected_layers=tuple(args.layers),
        )
        if errors:
            raise ValueError("validation failed:\n" + "\n".join(errors))
        write_atomically(
            run_config.result_root / "aggregate.json",
            {
                "trajectory": run_config.trajectory,
                "model_key": run_config.model_key,
                "conditions": aggregate,
            },
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"DONE: {len(aggregate)} condition summaries", flush=True)
    return 0


def rebuild_aggregate(
    root: Path,
    *,
    trajectory: str,
    model_key: str,
    scale: str = SCALE_7B,
    selected_checkpoints: tuple[str, ...] | None = None,
    selected_layers: tuple[int, ...] | None = None,
    dataset_path: Path = Path(DEFAULT_DATASET),
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    dtype: str = DEFAULT_DTYPE,
    quantization: str = DEFAULT_QUANTIZATION,
    artifact_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, object]:
    expected_model_key = trajectory_config(FAMILY_THINK, scale, trajectory).model_key
    if model_key != expected_model_key:
        raise ValueError(
            f"model_key {model_key!r} does not match trajectory {trajectory!r}; "
            f"expected {expected_model_key!r}"
        )
    if scale == SCALE_32B and max_new_tokens != CANONICAL_32B_MAX_NEW_TOKENS:
        raise ValueError("32b requires max_new_tokens=2048 (--max-new-tokens 2048)")
    if scale == SCALE_32B:
        _require_canonical_7b(project_root=project_root)
    resolved_artifact_root = (
        root_for_trajectory(FAMILY_THINK, scale, trajectory, project_root=project_root)
        if artifact_root is None
        else artifact_root
    )
    if scale == SCALE_32B:
        from src.think_32b_differential_validator import (
            validate_full_canonical_publication,
        )

        publication = validate_full_canonical_publication(
            resolved_artifact_root, trajectory
        )
        if not publication.ok:
            raise ValueError(
                "32B extraction publication validation failed: " + publication.errors[0]
            )
    ensure_root_ownership(
        root,
        family=FAMILY_THINK,
        scale=scale,
        trajectory=trajectory,
        model_key=expected_model_key,
        checkpoints=list(
            trajectory_config(FAMILY_THINK, scale, trajectory).checkpoints
            if selected_checkpoints is None
            else selected_checkpoints
        ),
        revisions=dict(trajectory_config(FAMILY_THINK, scale, trajectory).revisions),
        purpose="math500",
        canonical=(
            root.resolve()
            == (
                _default_result_root(trajectory, scale)
                if project_root is None
                else project_root / _default_result_root(trajectory, scale)
            ).resolve()
        ),
    )
    records, errors = collect_valid_conditions(
        root,
        trajectory=trajectory,
        dataset_path=dataset_path,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        quantization=quantization,
        artifact_root=resolved_artifact_root,
        scale=scale,
        selected_checkpoints=selected_checkpoints,
        selected_layers=selected_layers,
        project_root=project_root,
    )
    if errors:
        raise ValueError("validation failed:\n" + "\n".join(errors))
    aggregate: dict[str, object] = {
        "trajectory": trajectory,
        "model_key": model_key,
        "conditions": records,
    }
    write_atomically(root / "aggregate.json", aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
