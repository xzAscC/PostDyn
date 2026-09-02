import pytest
import torch
from torch import nn

from postdyn.residual_ablation import (  # pyright: ignore[reportMissingImports]
    ResidualStreamAblation,
    register_residual_ablation,
    residual_stream_ablation,
)


class _Block(nn.Module):
    def __init__(self, output_kind: str = "tensor") -> None:
        super().__init__()
        self.output_kind = output_kind
        self.register_buffer("marker", torch.tensor(17))

    def forward(self, hidden_states: torch.Tensor):
        if self.output_kind == "tuple":
            return hidden_states, self.marker
        if self.output_kind == "list":
            return [hidden_states, self.marker]
        return hidden_states


class _Backbone(nn.Module):
    def __init__(self, layer_count: int, output_kind: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(output_kind) for _ in range(layer_count)])


class _FakeOlmo3(nn.Module):
    def __init__(self, layer_count: int = 1, output_kind: str = "tensor") -> None:
        super().__init__()
        self.model = _Backbone(layer_count, output_kind)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            output = block(hidden_states)
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        return hidden_states


def _basis(d_model: int, *columns: int) -> torch.Tensor:
    return torch.eye(d_model)[:, list(columns)]


def _project_out(hidden_states: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return hidden_states - (hidden_states @ basis) @ basis.transpose(0, 1)


@pytest.mark.parametrize("shape", [(2, 5, 6), (3, 1, 6)])
def test_prefill_and_decode_ablate_every_token(shape: tuple[int, int, int]) -> None:
    model = _FakeOlmo3()
    hidden_states = torch.randn(shape)
    basis = _basis(shape[-1], 0, 3)
    baseline = model(hidden_states)

    with residual_stream_ablation(model, layer=0, U_target=basis):
        ablated = model(hidden_states)

    torch.testing.assert_close(ablated, _project_out(baseline, basis))
    torch.testing.assert_close(
        ablated @ basis,
        torch.zeros(*shape[:-1], basis.shape[1]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert len(model.model.layers[0]._forward_hooks) == 0


@pytest.mark.parametrize("parameter", ["U_target", "U_pos", "U_reference", "U_neg"])
def test_target_reference_basis_aliases(parameter: str) -> None:
    model = _FakeOlmo3()
    hidden_states = torch.randn(2, 3, 4)
    basis = _basis(4, 1)
    baseline = model(hidden_states)

    if parameter == "U_target":
        context = residual_stream_ablation(model, layer_idx=0, U_target=basis)
    elif parameter == "U_pos":
        context = residual_stream_ablation(model, layer_idx=0, U_pos=basis)
    elif parameter == "U_reference":
        context = residual_stream_ablation(model, layer_idx=0, U_reference=basis)
    else:
        context = residual_stream_ablation(model, layer_idx=0, U_neg=basis)

    with context:
        result = model(hidden_states)

    torch.testing.assert_close(result, _project_out(baseline, basis))


def test_positive_and_negative_bases_are_not_combined() -> None:
    model = _FakeOlmo3()
    basis = _basis(4, 0)

    with pytest.raises(ValueError, match="not combined"):
        ResidualStreamAblation(model, 0, U_pos=basis, U_neg=basis)


@pytest.mark.parametrize("output_kind", ["tuple", "list"])
def test_tuple_and_list_outputs_preserve_container_and_metadata(
    output_kind: str,
) -> None:
    model = _FakeOlmo3(output_kind=output_kind)
    block = model.model.layers[0]
    hidden_states = torch.randn(2, 3, 4)
    basis = _basis(4, 2)

    with residual_stream_ablation(model, 0, U_pos=basis):
        output = block(hidden_states)

    assert type(output) is (tuple if output_kind == "tuple" else list)
    torch.testing.assert_close(output[0], _project_out(hidden_states, basis))
    assert output[1] is block.marker
    assert len(block._forward_hooks) == 0


def test_multiple_layers_are_independent_and_removable() -> None:
    model = _FakeOlmo3(layer_count=2)
    hidden_states = torch.randn(2, 4, 5)
    first_basis = _basis(5, 0)
    second_basis = _basis(5, 1)
    first = register_residual_ablation(model, 0, U_pos=first_basis)
    second = register_residual_ablation(model, 1, U_neg=second_basis)

    both = model(hidden_states)
    after_first = _project_out(hidden_states, first_basis)
    expected = _project_out(after_first, second_basis)
    torch.testing.assert_close(both, expected)
    assert first.is_registered
    assert second.is_registered

    first.remove()
    assert not first.is_registered
    assert len(model.model.layers[0]._forward_hooks) == 0
    assert len(model.model.layers[1]._forward_hooks) == 1
    torch.testing.assert_close(
        model(hidden_states), _project_out(hidden_states, second_basis)
    )

    second.remove()
    assert len(model.model.layers[1]._forward_hooks) == 0
    torch.testing.assert_close(model(hidden_states), hidden_states)


def test_projection_promotes_dtype_and_returns_original_dtype_and_device() -> None:
    model = _FakeOlmo3()
    hidden_states = torch.randn(2, 3, 4, dtype=torch.float16)
    basis = _basis(4, 0, 2).to(dtype=torch.float64)

    with residual_stream_ablation(model, 0, U_pos=basis):
        result = model(hidden_states)

    expected = _project_out(hidden_states.to(torch.float64), basis).to(
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    assert result.dtype == hidden_states.dtype
    assert result.device == hidden_states.device
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)


def test_empty_basis_is_identity() -> None:
    model = _FakeOlmo3()
    hidden_states = torch.randn(2, 3, 4)
    empty_basis = torch.empty(4, 0)

    with residual_stream_ablation(model, 0, U_pos=empty_basis):
        result = model(hidden_states)

    assert result is hidden_states
    torch.testing.assert_close(result, hidden_states)
    assert len(model.model.layers[0]._forward_hooks) == 0


def test_basis_shape_and_orthonormality_are_validated() -> None:
    model = _FakeOlmo3()
    with pytest.raises(ValueError, match="shape"):
        ResidualStreamAblation(model, 0, U_pos=torch.ones(4))

    nonorthonormal = _basis(4, 0, 1).clone()
    nonorthonormal[0, 1] = 0.5
    with pytest.raises(ValueError, match="orthonormal"):
        ResidualStreamAblation(model, 0, U_pos=nonorthonormal)


def test_negative_layer_indices_are_rejected() -> None:
    model = _FakeOlmo3()
    basis = _basis(4, 0)

    with pytest.raises(ValueError, match="non-negative"):
        ResidualStreamAblation(model, -1, U_pos=basis)


def test_context_removes_hook_after_body_exception_and_restores_baseline() -> None:
    model = _FakeOlmo3()
    hidden_states = torch.randn(2, 3, 4)
    baseline = model(hidden_states)
    basis = _basis(4, 0)

    with pytest.raises(RuntimeError, match="body failure"):
        with residual_stream_ablation(model, 0, U_pos=basis):
            assert len(model.model.layers[0]._forward_hooks) == 1
            raise RuntimeError("body failure")

    assert len(model.model.layers[0]._forward_hooks) == 0
    torch.testing.assert_close(model(hidden_states), baseline)


def test_context_removes_hook_after_forward_exception() -> None:
    model = _FakeOlmo3()
    basis = _basis(4, 0)

    with pytest.raises(ValueError, match="width"):
        with residual_stream_ablation(model, 0, U_pos=basis):
            model(torch.randn(2, 3, 5))

    assert len(model.model.layers[0]._forward_hooks) == 0
