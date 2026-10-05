"""Tests for the RQ1 spectral overlap safety experiment (harmful vs benign)."""

from __future__ import annotations

import json
import torch
import pytest

from postdyn.extract import OnlineCovariance
from postdyn.spectra import eigensystem, energy_high_low


class TestSafetyDataLoading:
    """Verify harmful/benign dataset loading helpers."""

    def test_load_harmful_returns_strings(self):
        from datasets import load_dataset

        ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
        assert "rejected" in ds.column_names
        sample = ds[0]["rejected"]
        assert isinstance(sample, str) and len(sample) > 0

    def test_load_benign_returns_strings(self):
        from datasets import load_dataset

        ds = load_dataset("LLM-LAT/benign-dataset", split="train")
        assert "response" in ds.column_names
        sample = ds[0]["response"]
        assert isinstance(sample, str) and len(sample) > 0

    def test_harmful_size(self):
        from datasets import load_dataset

        ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
        assert len(ds) >= 4900


class TestSafetyOverlapPipeline:
    """End-to-end overlap metrics on synthetic harmful/benign hidden states."""

    @pytest.fixture
    def synthetic_safety_data(self):
        d = 32
        n = 200
        torch.manual_seed(99)

        harmful = torch.zeros(n, d, dtype=torch.float64)
        harmful[:, 0] = 4.0
        harmful[:, 2] = torch.randn(n, dtype=torch.float64) * 2.5
        harmful += torch.randn(n, d, dtype=torch.float64) * 0.1

        benign = torch.zeros(n, d, dtype=torch.float64)
        benign[:, 1] = 3.0
        benign += torch.randn(n, d, dtype=torch.float64) * 0.1

        return harmful, benign, d

    def test_dim_captures_separation(self, synthetic_safety_data):
        harmful, benign, d = synthetic_safety_data
        mu_h = harmful.mean(dim=0)
        mu_b = benign.mean(dim=0)
        dim_vec = mu_h - mu_b
        dim_hat = dim_vec / dim_vec.norm()
        assert dim_hat[0].abs().item() > 0.5 or dim_hat[1].abs().item() > 0.5

    def test_metrics_well_formed(self, synthetic_safety_data):
        harmful, benign, d = synthetic_safety_data
        k = d // 3

        cov = OnlineCovariance()
        cov.update(harmful)
        mu_h = cov.mean
        sigma = cov.covariance

        mu_b = benign.mean(dim=0).to(dtype=torch.float64)
        dim_vec = mu_h - mu_b

        vals, vecs = eigensystem(sigma)
        result = energy_high_low(dim_vec, vecs, k)

        assert 0.0 <= result["E_high"] <= 1.0
        assert 0.0 <= result["E_low"] <= 1.0
        assert len(result["profile"]) == d

    def test_second_moment_metric(self, synthetic_safety_data):
        harmful, benign, d = synthetic_safety_data

        cov = OnlineCovariance()
        cov.update(harmful)
        mu_h = cov.mean
        sigma = cov.covariance

        M = sigma + mu_h.unsqueeze(1) @ mu_h.unsqueeze(0)
        m_vals, m_vecs = eigensystem(M)

        w1 = m_vecs[:, 0]
        mu_b = benign.mean(dim=0).to(dtype=torch.float64)
        dim_vec = mu_h - mu_b
        dim_hat = dim_vec / dim_vec.norm()

        overlap = float((w1 @ dim_hat).square().item())
        assert 0.0 <= overlap <= 1.0

    def test_incremental_resume(self, tmp_path):
        from postdyn.persistence import append_jsonl

        metrics_path = tmp_path / "metrics.jsonl"
        append_jsonl(metrics_path, {"domain": "safety", "layer": 6, "done": True})

        completed = set()
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                completed.add((row["domain"], row["layer"]))

        assert ("safety", 6) in completed
        assert ("safety", 17) not in completed
