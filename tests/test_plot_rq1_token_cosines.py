from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_rq1_token_cosines import binned_stats


def test_binned_stats_groups_positions_and_drops_sparse_bins():
    cos = torch.tensor([0.1, 0.3, float("nan"), 0.5, 0.7, 0.2])
    lengths = torch.tensor([3, 2, 1])
    centers, mean, std = binned_stats(cos, lengths, edges=[0, 1, 3], min_count=2)
    assert len(centers) == 2
    assert mean.tolist() == pytest.approx([(0.1 + 0.5 + 0.2) / 3, (0.3 + 0.7) / 2])
    assert std[1] == pytest.approx(0.2)
    _, mean2, _ = binned_stats(cos, lengths, edges=[0, 1, 3], min_count=3)
    assert mean2.tolist() == pytest.approx([0.8 / 3])
