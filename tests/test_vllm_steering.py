"""Pure-logic tests for the vLLM steering row arithmetic (no vllm needed)."""

from types import SimpleNamespace

import torch

from postdyn import vllm_steering
from postdyn.spectrum import StateMap
from postdyn.vllm_steering import _rows_from_metadata


def _md(qsl, computed=None):
    return SimpleNamespace(
        query_start_loc=torch.tensor(qsl), num_computed_tokens=computed
    )


def test_prefill_rows_are_last_token_of_each_sequence() -> None:
    rows = _rows_from_metadata(_md([0, 6, 13, 20], [0, 0, 0]))
    assert rows == [5, 12, 19]


def test_single_token_decode_returns_none() -> None:
    assert _rows_from_metadata(_md([0, 1, 2, 3], [5, 9, 12])) is None


def test_resumed_prefill_chunks_are_skipped() -> None:
    # chunked continuation: computed>0 for every multi-token query -> skip
    assert _rows_from_metadata(_md([0, 8], [7])) is None


def test_mixed_batch_takes_only_fresh_whole_prefills() -> None:
    rows = _rows_from_metadata(_md([0, 5, 11], [4, 0]))
    assert rows == [10]


def test_missing_metadata_returns_none() -> None:
    assert _rows_from_metadata(None) is None
    assert _rows_from_metadata(SimpleNamespace()) is None


def fake_llm(blocks):
    runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=blocks)))
    worker = SimpleNamespace(worker=SimpleNamespace(model_runner=runner))
    executor = SimpleNamespace(driver_worker=worker)
    core = SimpleNamespace(engine_core=SimpleNamespace(model_executor=executor))
    return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core))


class PassBlock:
    def forward(self, positions, hidden_states):
        return hidden_states.clone()


def test_steered_rows_start_at_the_last_prompt_token():
    positions = torch.tensor([0, 1, 2, 3, 4, 5])
    # The last prompt row picks the first new token, so it is steered too.
    assert vllm_steering.steered_rows(positions, 3, 2).tolist() == [
        False, False, True, True, False, False,
    ]
    assert not vllm_steering.steered_rows(positions, 3, 0).any()


def test_prefix_maps_move_only_rows_in_the_prefix_and_detach():
    blocks = [PassBlock(), PassBlock()]
    original = blocks[0].forward
    prefix = vllm_steering.PrefixMaps(fake_llm(blocks), [0, 1])
    empty = torch.zeros(2, 0)
    gap = StateMap(empty, empty, torch.ones(2))
    prefix.set({0: gap, 1: gap}, {0: torch.zeros(2), 1: torch.zeros(2)}, 2, 2)
    # A prefill row and decode rows of two samples at different steps.
    out = blocks[0].forward(torch.tensor([1, 2, 3]), torch.zeros(3, 2, dtype=torch.bfloat16))
    assert out[:, 0].tolist() == [1.0, 1.0, 0.0]
    # Later layers of the same step reuse the rows found at the first layer.
    out = blocks[1].forward(torch.tensor([1, 2, 3]), torch.zeros(3, 2, dtype=torch.bfloat16))
    assert out[:, 0].tolist() == [1.0, 1.0, 0.0]
    prefix.detach()
    assert blocks[0].forward == original


class MLPBlock:
    """As Olmo 3: the residual plus the normed MLP output (here the input itself)."""

    def __init__(self):
        self.post_feedforward_layernorm = SimpleNamespace(forward=lambda x: x.clone())

    def forward(self, positions, hidden_states):
        return hidden_states + self.post_feedforward_layernorm.forward(hidden_states)


def test_the_mlp_site_rewrites_only_the_mlp_share_and_detaches():
    blocks = [MLPBlock()]
    norm = blocks[0].post_feedforward_layernorm
    original = norm.forward
    prefix = vllm_steering.PrefixMaps(fake_llm(blocks), [0], "mlp")
    empty = torch.zeros(2, 0)
    prefix.set({0: StateMap(empty, empty, torch.ones(2))}, {0: torch.zeros(2)}, 2, 2)
    out = blocks[0].forward(torch.tensor([1, 2, 3]), torch.ones(3, 2))
    # Residual 1 plus the MLP's 1, and the shift only on the MLP share of the steered rows.
    assert out[:, 0].tolist() == [3.0, 3.0, 2.0]
    prefix.detach()
    assert norm.forward == original
