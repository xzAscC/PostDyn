"""Pure-logic tests for the vLLM steering row arithmetic (no vllm needed)."""

from types import SimpleNamespace

import torch

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
