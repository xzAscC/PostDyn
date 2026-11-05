from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_rq1_mean_variance import w1_stats


def test_w1_stats():
    row = {
        "m3": {"v": [2.0, 1.0], "b": [9.0, 0.0]},
        "u_mu_pos": [3.0, 4.0],
        "sigma_eigenvalues": [8.0, 1.0],
    }
    cos, ratio = w1_stats(row)
    assert cos == pytest.approx(math.sqrt(9.0) / 5.0)
    assert ratio == pytest.approx(2.0 / 8.0)
