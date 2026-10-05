from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_rq1_token_cosines import position_stats


def test_position_stats_per_position_with_min_count():
    cos = torch.tensor([0.1, 0.3, float("nan"), 0.5, 0.7, 0.2])
    lengths = torch.tensor([3, 2, 1])
    pos, mean, std = position_stats(cos, lengths, min_count=2)
    assert pos.tolist() == [0, 1]
    assert mean.tolist() == pytest.approx([(0.1 + 0.5 + 0.2) / 3, (0.3 + 0.7) / 2])
    assert std[1] == pytest.approx(0.2)
