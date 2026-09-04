"""Tests for eigenvalue-extreme summaries of domain covariances and ΔΣ."""

from __future__ import annotations

import json

import pytest
import torch

from postdyn.eigen_spectrum import (
    DEFAULT_TAIL_K,
    atomic_write_json,
    build_layer_metrics,
    covariance_extremes,
    difference_extremes,
    plot_layer_lines,
)


def _exact_covariance_samples(variances: list[float]) -> torch.Tensor:
    """Rows whose empirical covariance is exactly diag(variances)."""
    n = len(variances) + 1
    centered = torch.eye(n) - torch.ones(n, n) / n
    basis, _ = torch.linalg.qr(centered.double())
    return (
        basis[:, : len(variances)] * torch.tensor(variances).sqrt() * n**0.5
    ).float()


def test_covariance_extremes_reports_min_max_and_tails():
    variances = [0.25, 4.0, 1.0, 9.0, 0.5]
    h = _exact_covariance_samples(variances)

    metrics = covariance_extremes(h, k=3)

    assert metrics["n"] == h.shape[0]
    assert metrics["d_model"] == h.shape[1]
    assert metrics["lambda_min"] == pytest.approx(0.25, rel=1e-4)
    assert metrics["lambda_max"] == pytest.approx(9.0, rel=1e-4)
    assert metrics["smallest"] == pytest.approx([0.25, 0.5, 1.0], rel=1e-4)
    assert metrics["largest"] == pytest.approx([9.0, 4.0, 1.0], rel=1e-4)
    assert metrics["trace"] == pytest.approx(sum(variances), rel=1e-4)


def test_covariance_extremes_tail_clamped_to_dimension():
    h = _exact_covariance_samples([2.0, 1.0])
    metrics = covariance_extremes(h, k=DEFAULT_TAIL_K)
    assert len(metrics["smallest"]) == 2
    assert len(metrics["largest"]) == 2


def test_covariance_extremes_rank_deficient_min_is_zero():
    torch.manual_seed(0)
    h = torch.randn(5, 16)
    metrics = covariance_extremes(h, k=4)
    assert abs(metrics["lambda_min"]) < 1e-8
    assert metrics["smallest"][1] < 1e-8
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics["largest"])


def test_difference_extremes_signed_max_and_min():
    h_concept = _exact_covariance_samples([5.0, 2.0, 1.3, 1.0, 1.0])
    h_ref = _exact_covariance_samples([1.0, 1.0, 1.0, 1.0, 4.0])

    metrics = difference_extremes(h_concept, h_ref, k=5)

    assert metrics["lambda_max_pos"] == pytest.approx(4.0, rel=1e-4)
    assert metrics["lambda_min_neg"] == pytest.approx(-3.0, rel=1e-4)
    assert metrics["n_pos"] == 3
    assert metrics["n_neg"] == 1
    assert metrics["top_pos"] == pytest.approx([4.0, 1.0, 0.3], rel=1e-4)
    assert metrics["bottom_neg"] == pytest.approx([-3.0], rel=1e-4)
    assert metrics["tr_concept"] == pytest.approx(10.3, rel=1e-4)
    assert metrics["tr_ref"] == pytest.approx(8.0, rel=1e-4)


def test_build_layer_metrics_structure():
    h_concept = _exact_covariance_samples([5.0, 2.0, 1.3, 1.0, 1.0])
    h_ref = _exact_covariance_samples([1.0, 1.0, 1.0, 1.0, 4.0])

    metrics = build_layer_metrics(h_concept, h_ref, k=3)

    assert set(metrics) == {"concept", "reference", "difference"}
    assert metrics["concept"]["lambda_min"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["reference"]["lambda_min"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["difference"]["lambda_max_pos"] == pytest.approx(4.0, rel=1e-4)
    assert metrics["difference"]["lambda_min_neg"] == pytest.approx(-3.0, rel=1e-4)


def test_all_metrics_are_json_serializable():
    h_concept = _exact_covariance_samples([3.0, 1.0, 0.5])
    h_ref = _exact_covariance_samples([1.0, 2.0, 0.5])
    metrics = build_layer_metrics(h_concept, h_ref, k=2)
    encoded = json.dumps(metrics)
    assert json.loads(encoded) == metrics


def test_atomic_write_json_round_trip(tmp_path):
    target = tmp_path / "nested" / "layer_3.json"
    payload = {"lambda_min": 1.5, "values": [1.0, 2.0]}
    atomic_write_json(target, payload)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob("nested/*.tmp"))


def test_plot_layer_lines_writes_pdf(tmp_path):
    rows = []
    for model in ("sft_final", "rlvr_final"):
        for layer in (3, 9, 17):
            rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "lambda_min_concept": 0.01 * layer,
                    "lambda_min_reference": 0.02 * layer,
                    "lambda_max_pos": 0.5 + 0.01 * layer,
                    "lambda_min_neg": -(0.3 + 0.01 * layer),
                }
            )
    out_pdf = tmp_path / "eig_spectrum.pdf"
    returned = plot_layer_lines(rows, out_pdf)
    assert out_pdf.is_file()
    assert out_pdf.stat().st_size > 0
    assert returned == out_pdf


def test_plot_layer_lines_empty_rows_no_crash(tmp_path):
    out_pdf = tmp_path / "empty.pdf"
    plot_layer_lines([], out_pdf)
    assert not out_pdf.exists()
