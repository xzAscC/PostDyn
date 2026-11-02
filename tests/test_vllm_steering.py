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
    prefix.detach()
    assert blocks[0].forward == original
