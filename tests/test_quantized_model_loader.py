from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from postdyn import quantized_model_loader as loader
from postdyn.config import OLMO3_VARIANTS


def _configured_sft_revision() -> str:
    return next(
        iter(
            loader.CANONICAL_32B_MODEL_REVISIONS[
                OLMO3_VARIANTS["olmo3-32b-think-sft"].hf_id
            ]
        )
    )


def _valid_diagnostics() -> dict[str, object]:
    return {
        "placement": "gpu-only",
        "quant_backend": "bitsandbytes",
        "safetensors": True,
        "bitsandbytes_version": "0.49.2",
        "transformers_version": "4.57.1",
        "accelerate_version": "1.10.1",
        "device_map": {"": "cuda:0"},
        "peak_vram_bytes": 1,
        "fallback_reason": None,
    }


def test_nf4_diagnostics_accepts_exact_gpu_only_schema() -> None:
    assert loader.validate_nf4_load_diagnostics(_valid_diagnostics()) is None


@pytest.mark.parametrize(
    "alteration",
    [
        {"placement": "cpu"},
        {"placement": "auto-fallback"},
        {"device_map": {"": "meta"}},
        {"device_map": {"": "disk"}},
        {"quant_backend": "torchao"},
        {"safetensors": False},
        {"fallback_reason": "OOM"},
        {"missing": None},
    ],
)
def test_nf4_diagnostics_rejects_noncanonical_runtime_states(alteration) -> None:
    diagnostics = _valid_diagnostics()
    if "missing" in alteration:
        diagnostics.pop("placement")
    else:
        diagnostics.update(alteration)
    assert loader.validate_nf4_load_diagnostics(diagnostics) is not None


class FakeTorch:
    bfloat16 = "bf16"


def test_builds_exact_nf4_double_quant_config() -> None:
    config_type = Mock()
    fake_transformers = SimpleNamespace(BitsAndBytesConfig=config_type)
    with patch.object(
        loader.importlib, "import_module", return_value=fake_transformers
    ):
        loader.build_nf4_config(FakeTorch())
    config_type.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bf16",
        bnb_4bit_use_double_quant=True,
    )


def test_architecture_validation_requires_exact_32b_shape() -> None:
    loader._validate_config(SimpleNamespace(hidden_size=5120, num_hidden_layers=64))
    with pytest.raises(loader.ModelArchitectureError, match="hidden_size=5120"):
        loader._validate_config(SimpleNamespace(hidden_size=4096, num_hidden_layers=64))


def test_32b_loader_rejects_noncanonical_model_id_before_access() -> None:
    with pytest.raises(loader.QuantizedLoaderError, match="canonical"):
        loader.load_olmo3_32b_think(model_id="arbitrary/model")


@pytest.mark.parametrize(
    "model_key",
    ("olmo3-32b-think-sft", "olmo3-32b-think-rlvr"),
)
def test_32b_loader_accepts_configured_sft_and_rlvr_ids(
    model_key: str,
) -> None:
    with patch.object(
        loader,
        "check_quantization_dependencies",
        side_effect=loader.MissingQuantizationDependencyError("sentinel"),
    ) as check_dependencies:
        with pytest.raises(loader.MissingQuantizationDependencyError, match="sentinel"):
            loader.load_olmo3_32b_think(
                OLMO3_VARIANTS[model_key].hf_id,
                revision=(
                    _configured_sft_revision()
                    if model_key == "olmo3-32b-think-sft"
                    else next(
                        iter(
                            loader.CANONICAL_32B_MODEL_REVISIONS[
                                OLMO3_VARIANTS[model_key].hf_id
                            ]
                        )
                    )
                ),
            )
    check_dependencies.assert_called_once_with()


@pytest.mark.parametrize("revision", ["main", "v1", "", "abc", "a" * 39, "g" * 40])
def test_32b_loader_rejects_non_immutable_revision_before_dependency_access(
    revision: str,
) -> None:
    with patch.object(loader, "check_quantization_dependencies") as check_dependencies:
        with pytest.raises(loader.QuantizedLoaderError, match="40-hex"):
            loader.load_olmo3_32b_think(revision=revision)
    check_dependencies.assert_not_called()


def test_32b_loader_rejects_unknown_id_before_factory_access() -> None:
    dependency_check = Mock()
    module_factory = Mock()
    with (
        patch.object(loader, "check_quantization_dependencies", dependency_check),
        patch.object(loader.importlib, "import_module", module_factory),
    ):
        with pytest.raises(loader.QuantizedLoaderError, match="canonical"):
            loader.load_olmo3_32b_think(
                "allenai/Olmo-3-32B-Think-unknown", revision="a" * 40
            )
    dependency_check.assert_not_called()
    module_factory.assert_not_called()


def test_32b_loader_rejects_unconfigured_sha_before_dependency_access() -> None:
    with patch.object(loader, "check_quantization_dependencies") as check_dependencies:
        with pytest.raises(loader.QuantizedLoaderError, match="configured"):
            loader.load_olmo3_32b_think(
                OLMO3_VARIANTS["olmo3-32b-think-sft"].hf_id,
                revision="a" * 40,
            )
    check_dependencies.assert_not_called()


def test_canonical_nf4_provenance_is_shared_and_disables_fallback() -> None:
    assert loader.CANONICAL_NF4_PROVENANCE["allow_auto_fallback"] is False


def test_measured_budget_reserves_vram_and_rounds_down() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        mem_get_info=lambda _device: (23 * 1024**3 + 512, 24 * 1024**3),
    )
    budget = loader.measured_gpu_memory_budget(
        SimpleNamespace(cuda=cuda), reserve_bytes=1024, minimum_bytes=1
    )
    assert budget == {"cuda:0": "22GiB"}


def test_missing_dependency_error_is_actionable() -> None:
    with patch.object(
        loader,
        "_version",
        side_effect=lambda name: None if name == "bitsandbytes" else "4.57.1",
    ):
        with pytest.raises(
            loader.MissingQuantizationDependencyError, match="bitsandbytes"
        ):
            loader.check_quantization_dependencies()


def test_forward_requests_hidden_states() -> None:
    model = Mock()
    loader.forward_with_hidden_states(model, input_ids=[1])
    model.assert_called_once_with(input_ids=[1], output_hidden_states=True)


def test_generate_hooks_are_removed_after_generation() -> None:
    hook = Mock()
    handle = Mock()
    model = Mock()
    model.named_modules.return_value = [
        (
            "model.layers.3",
            SimpleNamespace(register_forward_hook=Mock(return_value=handle)),
        )
    ]
    model.generate.return_value = "tokens"
    assert (
        loader.generate_with_hooks(model, {"input_ids": [1]}, hook, layer_indices=(3,))
        == "tokens"
    )
    handle.remove.assert_called_once_with()
    model.generate.assert_called_once_with(input_ids=[1])


def test_layer_hooks_only_match_canonical_model_layers() -> None:
    selected = SimpleNamespace(register_forward_hook=Mock())
    other = SimpleNamespace(register_forward_hook=Mock())
    model = Mock()
    model.named_modules.return_value = [
        ("model.layers.3", selected),
        ("other.path.3", other),
    ]
    loader.register_forward_hooks(model, Mock(), layer_indices=(3,))
    selected.register_forward_hook.assert_called_once()
    other.register_forward_hook.assert_not_called()


def test_layer_hooks_reject_unknown_or_duplicate_indices() -> None:
    model = Mock()
    model.named_modules.return_value = []
    with pytest.raises(ValueError, match="layer"):
        loader.register_forward_hooks(model, Mock(), layer_indices=(-1, 64))
    with pytest.raises(ValueError, match="duplicate"):
        loader.register_forward_hooks(model, Mock(), layer_indices=(3, 3))


def test_generate_does_not_move_dispatched_model() -> None:
    hook = Mock()
    handle = Mock()
    model = Mock()
    model.named_modules.return_value = [
        (
            "model.layers.3",
            SimpleNamespace(register_forward_hook=Mock(return_value=handle)),
        )
    ]
    model.generate.return_value = "tokens"
    model.to.side_effect = AssertionError("dispatched models must not move")
    assert loader.generate_with_hooks(model, {"input_ids": [1]}, hook) == "tokens"
    model.to.assert_not_called()
    handle.remove.assert_called_once_with()


def test_load_does_not_retry_or_expose_auto_fallback() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=lambda: 123,
        mem_get_info=lambda _device: (23 * 1024**3, 24 * 1024**3),
        empty_cache=Mock(),
    )
    torch = SimpleNamespace(cuda=cuda, bfloat16="bfloat16")
    auto_config = SimpleNamespace(hidden_size=5120, num_hidden_layers=64)
    factory = Mock(side_effect=RuntimeError("OOM"))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=auto_config)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value="tokenizer")),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory),
    )
    with (
        patch.object(
            loader,
            "check_quantization_dependencies",
            return_value={
                "bitsandbytes": "0.49.2",
                "transformers": "4.57.1",
                "accelerate": "1.10.1",
            },
        ),
        patch.object(
            loader.importlib,
            "import_module",
            side_effect=lambda name: torch if name == "torch" else transformers,
        ),
        patch.object(loader, "build_nf4_config", return_value="nf4"),
    ):
        with pytest.raises(loader.QuantizedLoaderError, match="placement"):
            loader.load_olmo3_32b_think(revision=_configured_sft_revision())
    factory.assert_called_once()
    assert factory.call_args.kwargs["device_map"] == {"": "cuda:0"}
    assert factory.call_args.kwargs["quantization_config"] == "nf4"
    assert factory.call_args.kwargs["torch_dtype"] == "bfloat16"
    assert factory.call_args.kwargs["use_safetensors"] is True


@pytest.mark.parametrize(
    "device_map",
    [
        {"": "cpu"},
        {"": "disk"},
        {"": "meta"},
        {},
        {"": "auto"},
        {"": "cuda-malformed"},
    ],
)
def test_load_rejects_invalid_factory_placement_before_returning_wrapper(
    device_map,
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=lambda: 123,
        mem_get_info=lambda _device: (23 * 1024**3, 24 * 1024**3),
    )
    torch = SimpleNamespace(cuda=cuda, bfloat16="bf16")
    model = SimpleNamespace(
        hf_device_map=device_map,
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        ),
    )
    config = SimpleNamespace(hidden_size=5120, num_hidden_layers=64)
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value="tokenizer")),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with (
        patch.object(
            loader,
            "check_quantization_dependencies",
            return_value={
                "bitsandbytes": "0.49.2",
                "transformers": "4.57.1",
                "accelerate": "1.10.1",
            },
        ),
        patch.object(
            loader.importlib,
            "import_module",
            side_effect=lambda name: torch if name == "torch" else transformers,
        ),
        patch.object(loader, "build_nf4_config", return_value="nf4"),
    ):
        with pytest.raises(loader.QuantizedLoaderError, match="placement|device_map"):
            loader.load_olmo3_32b_think(revision=_configured_sft_revision())


def test_load_rejects_malformed_runtime_diagnostics_before_returning_wrapper() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=lambda: 123,
        mem_get_info=lambda _device: (23 * 1024**3, 24 * 1024**3),
    )
    torch = SimpleNamespace(cuda=cuda, bfloat16="bf16")
    model = SimpleNamespace(
        hf_device_map={"": "cuda:0"},
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        ),
    )
    config = SimpleNamespace(hidden_size=5120, num_hidden_layers=64)
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value="tokenizer")),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with (
        patch.object(
            loader,
            "check_quantization_dependencies",
            return_value={
                "bitsandbytes": "0.49.2",
                "transformers": "4.57.1",
                "accelerate": "1.10.1",
            },
        ),
        patch.object(
            loader.importlib,
            "import_module",
            side_effect=lambda name: torch if name == "torch" else transformers,
        ),
        patch.object(loader, "build_nf4_config", return_value="nf4"),
        patch.object(
            loader,
            "validate_nf4_load_diagnostics",
            return_value="malformed diagnostics",
        ) as validate,
    ):
        with pytest.raises(loader.QuantizedLoaderError, match="malformed diagnostics"):
            loader.load_olmo3_32b_think(revision=_configured_sft_revision())
    validate.assert_called_once()


def test_load_rejects_model_with_cpu_parameter_or_meta_buffer() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=Mock(),
        max_memory_allocated=lambda: 123,
        mem_get_info=lambda _device: (23 * 1024**3, 24 * 1024**3),
    )
    torch = SimpleNamespace(cuda=cuda, bfloat16="bf16")
    quant_config = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    class Params4bit:
        is_quantized = True
        device = "cuda:0"
        dtype = "float32"

    model = SimpleNamespace(
        hf_device_map={"": "cuda:0"},
        is_loaded_in_4bit=True,
        config=SimpleNamespace(quantization_config=quant_config),
        quantization_method=lambda: "bitsandbytes",
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        ),
        named_parameters=lambda: iter(
            [
                ("good", Params4bit()),
                ("bad", SimpleNamespace(device="cpu")),
            ]
        ),
        named_buffers=lambda: iter([]),
    )
    config = SimpleNamespace(hidden_size=5120, num_hidden_layers=64)
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value="tokenizer")),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with (
        patch.object(
            loader,
            "check_quantization_dependencies",
            return_value={
                "bitsandbytes": "0.49.2",
                "transformers": "4.57.1",
                "accelerate": "1.10.1",
            },
        ),
        patch.object(
            loader.importlib,
            "import_module",
            side_effect=lambda name: torch if name == "torch" else transformers,
        ),
        patch.object(loader, "build_nf4_config", return_value=quant_config),
    ):
        with pytest.raises(loader.QuantizedLoaderError, match="parameter"):
            loader.load_olmo3_32b_think(
                OLMO3_VARIANTS["olmo3-32b-think-sft"].hf_id,
                revision=next(
                    iter(
                        loader.CANONICAL_32B_MODEL_REVISIONS[
                            OLMO3_VARIANTS["olmo3-32b-think-sft"].hf_id
                        ]
                    )
                ),
            )


def test_loaded_runtime_rejects_metadata_only_model_without_4bit_parameter_signal() -> (
    None
):
    quant_config = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        config=SimpleNamespace(quantization_config=quant_config),
        quantization_method=lambda: "bitsandbytes",
        hf_device_map={"": "cuda:0"},
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        ),
        named_parameters=lambda: iter(
            [("weight", SimpleNamespace(device="cuda:0", dtype="bfloat16"))]
        ),
        named_buffers=lambda: iter([]),
    )
    diagnostics = loader.LoadDiagnostics(
        "gpu-only",
        bitsandbytes_version="0.49.2",
        transformers_version="4.57.1",
        accelerate_version="1.10.1",
        peak_vram_bytes=0,
    )
    with pytest.raises(loader.QuantizedLoaderError, match="genuine NF4"):
        loader._validate_loaded_runtime(model, diagnostics)


def test_nf4_attestation_rejects_spoofed_params4bit_class(monkeypatch) -> None:
    class Params4bit:
        dtype = "bfloat16"
        quant_state = SimpleNamespace(quant_type="nf4")

    fake_nn = SimpleNamespace(Params4bit=type("Params4bit", (), {}))
    monkeypatch.setattr(loader.importlib, "import_module", lambda name: fake_nn)
    model = SimpleNamespace(named_parameters=lambda: iter([("spoofed", Params4bit())]))

    with pytest.raises(loader.QuantizedLoaderError, match="genuine NF4"):
        loader._validate_nf4_parameter_attestation(model)


def test_nf4_attestation_rejects_genuine_params4bit_with_wrong_quant_type(
    monkeypatch,
) -> None:
    genuine_type = type(
        "Params4bit",
        (),
        {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
    )
    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda name: SimpleNamespace(Params4bit=genuine_type),
    )
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [
                (
                    "quantized",
                    genuine_type(
                        quant_state=SimpleNamespace(quant_type="fp4"),
                        dtype="bfloat16",
                    ),
                )
            ]
        )
    )

    with pytest.raises(loader.QuantizedLoaderError, match="genuine NF4"):
        loader._validate_nf4_parameter_attestation(model)


@pytest.mark.parametrize(
    "quantized_kwargs",
    [
        {"quant_state": SimpleNamespace(quant_type="fp4")},
        {},
    ],
)
def test_nf4_attestation_rejects_invalid_genuine_params4bit_alongside_valid_nf4(
    monkeypatch, quantized_kwargs
) -> None:
    genuine_type = type(
        "Params4bit",
        (),
        {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
    )
    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda name: SimpleNamespace(Params4bit=genuine_type),
    )
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [
                (
                    "valid_quantized",
                    genuine_type(
                        quant_state=SimpleNamespace(quant_type="nf4"),
                        dtype="float32",
                    ),
                ),
                (
                    "invalid_quantized",
                    genuine_type(dtype="float32", **quantized_kwargs),
                ),
            ]
        )
    )

    with pytest.raises(loader.QuantizedLoaderError, match="not NF4"):
        loader._validate_nf4_parameter_attestation(model)


def test_nf4_attestation_accepts_only_genuine_nf4_params4bit(monkeypatch) -> None:
    genuine_type = type(
        "Params4bit",
        (),
        {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
    )
    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda name: SimpleNamespace(Params4bit=genuine_type),
    )
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [
                (
                    "quantized",
                    genuine_type(
                        quant_state=SimpleNamespace(quant_type="NF4"),
                        dtype="float32",
                    ),
                ),
                ("ordinary", SimpleNamespace(dtype="bfloat16")),
            ]
        )
    )

    loader._validate_nf4_parameter_attestation(model)


def test_loaded_runtime_allows_cuda_float32_buffer_with_bfloat16_parameters(
    monkeypatch,
) -> None:
    genuine_type = type(
        "Params4bit",
        (),
        {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)},
    )
    monkeypatch.setattr(
        loader.importlib,
        "import_module",
        lambda name: SimpleNamespace(Params4bit=genuine_type),
    )
    quant_config = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        config=SimpleNamespace(quantization_config=quant_config),
        quantization_method=lambda: "bitsandbytes",
        hf_device_map={"": "cuda:0"},
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        ),
        named_parameters=lambda: iter(
            [
                (
                    "quantized",
                    genuine_type(
                        quant_state=SimpleNamespace(quant_type="nf4"),
                        dtype="float32",
                        device="cuda:0",
                    ),
                ),
                ("ordinary", SimpleNamespace(device="cuda:0", dtype="bfloat16")),
            ]
        ),
        named_buffers=lambda: iter(
            [("rotary.inv_freq", SimpleNamespace(device="cuda:0", dtype="float32"))]
        ),
    )
    diagnostics = loader.LoadDiagnostics(
        "gpu-only",
        bitsandbytes_version="0.49.2",
        transformers_version="4.57.1",
        accelerate_version="1.10.1",
        peak_vram_bytes=0,
    )

    loader._validate_loaded_runtime(model, diagnostics)


def test_loaded_runtime_rejects_ordinary_float_parameter_not_bfloat16() -> None:
    quant_config = SimpleNamespace(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )

    class Params4bit:
        is_quantized = True
        device = "cuda:0"
        dtype = "float32"

    model = SimpleNamespace(
        is_loaded_in_4bit=True,
        config=SimpleNamespace(quantization_config=quant_config),
        quantization_method=lambda: "bitsandbytes",
        hf_device_map={"": "cuda:0"},
        get_input_embeddings=lambda: SimpleNamespace(
            weight=SimpleNamespace(device="cuda:0")
        ),
        named_parameters=lambda: iter(
            [
                ("quantized", Params4bit()),
                ("ordinary", SimpleNamespace(device="cuda:0", dtype="float16")),
            ]
        ),
        named_buffers=lambda: iter([]),
    )
    diagnostics = loader.LoadDiagnostics(
        "gpu-only",
        bitsandbytes_version="0.49.2",
        transformers_version="4.57.1",
        accelerate_version="1.10.1",
        peak_vram_bytes=0,
    )
    with pytest.raises(loader.QuantizedLoaderError, match="not bfloat16"):
        loader._validate_loaded_runtime(model, diagnostics)


def test_cuda_is_required_before_any_checkpoint_factory_call() -> None:
    cuda = SimpleNamespace(is_available=lambda: False)
    torch = SimpleNamespace(cuda=cuda, bfloat16="bf16")
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=Mock()),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock()),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock()),
    )
    with (
        patch.object(
            loader,
            "check_quantization_dependencies",
            return_value={
                "bitsandbytes": "0.49.2",
                "transformers": "4.57.1",
                "accelerate": "1.10.1",
            },
        ),
        patch.object(
            loader.importlib,
            "import_module",
            side_effect=lambda name: torch if name == "torch" else transformers,
        ),
    ):
        with pytest.raises(loader.QuantizedLoaderError, match="requires CUDA"):
            loader.load_olmo3_32b_think(revision=_configured_sft_revision())
    transformers.AutoConfig.from_pretrained.assert_not_called()
    transformers.AutoTokenizer.from_pretrained.assert_not_called()
    transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
