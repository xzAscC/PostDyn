import torch

from src.activation_analysis import compute_activation_alpha_req, compute_activation_rankme


def test_activation_rankme_uses_covariance_eigenvalues():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )

    rankme, ratio = compute_activation_rankme(features)

    centered = features - features.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / centered.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    expected_rankme = torch.exp(-torch.sum(p * torch.log(p))).item()

    assert abs(rankme - expected_rankme) < 1e-6
    assert abs(ratio - expected_rankme / features.shape[1]) < 1e-6


def test_activation_rankme_differs_from_raw_singular_value_entropy():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 2.0],
            [0.0, -2.0],
        ]
    )

    rankme, _ = compute_activation_rankme(features)

    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered.float())
    p_raw = singular_values / singular_values.sum()
    raw_svd_rankme = torch.exp(-torch.sum(p_raw * torch.log(p_raw))).item()

    assert abs(rankme - raw_svd_rankme) > 0.05


def test_activation_alpha_req_uses_covariance_eigenvalue_decay():
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 / 2.0, 0.0, 0.0],
            [0.0, -1.0 / 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 / 3.0, 0.0],
            [0.0, 0.0, -1.0 / 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0 / 4.0],
            [0.0, 0.0, 0.0, -1.0 / 4.0],
        ]
    )

    alpha = compute_activation_alpha_req(features, fit_range=(1, 4))

    assert abs(alpha - 2.0) < 1e-5
