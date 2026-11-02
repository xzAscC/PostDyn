from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from postdyn import spectrum
from postdyn.spectra import eigensystem


def covariance(seed: int, d: int = 6, n: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    factor = torch.randn(d, d, generator=generator, dtype=torch.float64)
    samples = torch.randn(n, d, generator=generator, dtype=torch.float64) @ factor
    centered = samples - samples.mean(0)
    return samples, centered.T @ centered / n


def stats_of(sigma: torch.Tensor, mean: torch.Tensor | None = None) -> spectrum.LayerStats:
    return spectrum.LayerStats(sigma, torch.zeros(len(sigma), dtype=torch.float64) if mean is None else mean)


def test_problem_moments_remove_each_problem_mean_only():
    moments = spectrum.ProblemMoments({0: 2}, torch.device("cpu"))
    # Two trajectories of equal weight around the mean (1, 1).
    moments.update({0: torch.tensor([[0.0, 1.0], [2.0, 1.0]])}, torch.ones(2))
    mean, scatter, weight = moments.finish()
    assert weight == pytest.approx(2.0)
    assert mean[0].tolist() == pytest.approx([1.0, 1.0])
    assert scatter[0].numpy() == pytest.approx(np.array([[2.0, 0.0], [0.0, 0.0]]))


def test_fold_moments_are_within_problem_and_leave_the_tested_fold_out():
    folds = spectrum.FoldMoments({0: 1}, 2)
    # A huge gap between problems must not enter the within-problem covariance.
    for problem, fold, value in (("a", 0, 0.0), ("b", 1, 100.0)):
        moments = spectrum.ProblemMoments({0: 1}, torch.device("cpu"))
        moments.update({0: torch.tensor([[value - 1.0], [value + 1.0]])}, torch.ones(2))
        folds.add(problem, fold, *moments.finish())
    assert folds.stats(0, leave_out=None).sigma.item() == pytest.approx(1.0)
    assert folds.stats(0, leave_out=1).mean.item() == pytest.approx(0.0)
    assert folds.stats(0, leave_out=0).mean.item() == pytest.approx(100.0)
    assert folds.problem_means["b"][0].item() == pytest.approx(100.0)


def test_fold_of_is_stable_and_in_range():
    assert spectrum.fold_of("problem-1", 5) == spectrum.fold_of("problem-1", 5)
    assert {spectrum.fold_of(f"p{i}", 5) for i in range(200)} == set(range(5))


def test_prefix_length_takes_the_share_and_at_least_one_token():
    assert spectrum.prefix_length(3000, 0.1) == 300
    assert spectrum.prefix_length(4, 0.1) == 1


def test_full_map_gives_the_post_covariance_at_full_rank():
    samples, c_base = covariance(0)
    _, c_post = covariance(1)
    shift = spectrum.build_map("full", stats_of(c_base), stats_of(c_post), k=6)
    moved = spectrum.apply_map(samples, shift, torch.zeros(6, dtype=torch.float64))
    centered = moved - moved.mean(0)
    assert (centered.T @ centered / len(moved)).numpy() == pytest.approx(c_post.numpy(), abs=1e-8)


def test_spectrum_map_keeps_base_vectors_and_takes_post_values():
    samples, c_base = covariance(0)
    _, c_post = covariance(1)
    shift = spectrum.build_map("spectrum", stats_of(c_base), stats_of(c_post), k=6)
    moved = spectrum.apply_map(samples, shift, torch.zeros(6, dtype=torch.float64))
    centered = moved - moved.mean(0)
    values, vectors = eigensystem(centered.T @ centered / len(moved))
    assert values.numpy() == pytest.approx(eigensystem(c_post)[0].numpy(), abs=1e-8)
    _, base_vectors = eigensystem(c_base)
    assert (vectors.T @ base_vectors).abs().diagonal().numpy() == pytest.approx([1.0] * 6, abs=1e-6)


def test_reorder_map_moves_variance_to_the_matched_rank_only():
    c_base = torch.diag(torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64))
    c_post = torch.diag(torch.tensor([1.0, 9.0, 4.0], dtype=torch.float64))
    shift = spectrum.build_map("reorder", stats_of(c_base), stats_of(c_post), k=3)
    moved = spectrum.apply_map(torch.eye(3, dtype=torch.float64), shift, torch.zeros(3, dtype=torch.float64))
    # The direction that has variance 1 in base has variance 9 afterwards: base's own values, moved.
    assert moved.diagonal().square().numpy() == pytest.approx([1 / 9, 9 / 4, 4.0], abs=1e-9)


def test_random_map_uses_base_values_in_a_shuffled_order():
    c_base = torch.diag(torch.tensor([9.0, 4.0, 1.0, 0.5], dtype=torch.float64))
    shift = spectrum.build_map("random", stats_of(c_base), stats_of(c_base), k=4, seed=3)
    moved = spectrum.apply_map(torch.eye(4, dtype=torch.float64), shift, torch.zeros(4, dtype=torch.float64))
    scaled = (moved.diagonal().square() * c_base.diagonal()).sort().values
    assert scaled.numpy() == pytest.approx(c_base.diagonal().sort().values.numpy(), abs=1e-9)


def test_mean_map_moves_every_token_by_the_mean_gap():
    sigma = torch.eye(2, dtype=torch.float64)
    base = stats_of(sigma, torch.tensor([1.0, 0.0], dtype=torch.float64))
    post = stats_of(sigma, torch.tensor([3.0, 1.0], dtype=torch.float64))
    shift = spectrum.build_map("mean", base, post, k=2)
    moved = spectrum.apply_map(torch.zeros(1, 2, dtype=torch.float64), shift, torch.zeros(2, dtype=torch.float64))
    assert moved[0].tolist() == pytest.approx([2.0, 1.0])


def test_identity_map_changes_nothing():
    samples, c_base = covariance(0)
    shift = spectrum.build_map("identity", stats_of(c_base), stats_of(c_base), k=3)
    assert spectrum.apply_map(samples, shift, samples[0]).equal(samples)


def test_top_k_leaves_the_rest_of_the_state_alone():
    c_base = torch.diag(torch.tensor([9.0, 4.0, 1.0], dtype=torch.float64))
    c_post = torch.diag(torch.tensor([16.0, 4.0, 25.0], dtype=torch.float64))
    shift = spectrum.build_map("spectrum", stats_of(c_base), stats_of(c_post), k=1)
    h = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    moved = spectrum.apply_map(h, shift, torch.zeros(3, dtype=torch.float64))
    assert moved[0].tolist() == pytest.approx([5 / 3, 2.0, 3.0])


def test_build_map_refuses_a_rank_deficient_top_k():
    sigma = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="positive"):
        spectrum.build_map("full", stats_of(sigma), stats_of(sigma), k=2)


def test_build_map_refuses_an_unknown_kind():
    sigma = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown"):
        spectrum.build_map("swap", stats_of(sigma), stats_of(sigma), k=1)


def test_off_diagonal_share_separates_scaling_from_rotation():
    c_base = torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))
    assert spectrum.off_diagonal_share(c_base, c_base * 2, 2) == pytest.approx(0.0)
    turn = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / 2**0.5
    rotated = turn @ c_base @ turn.T
    # [[2, 1], [1, 2]]: 2 of the squared size 10 is off the diagonal.
    assert spectrum.off_diagonal_share(c_base, rotated, 2) == pytest.approx(0.2)


def test_stable_rank_stops_where_two_halves_stop_agreeing():
    basis = torch.eye(4, dtype=torch.float64)
    other = basis.clone()
    other[:, [2, 3]] = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=torch.float64
    )
    swapped = torch.eye(4, dtype=torch.float64)[:, [0, 1, 3, 2]]
    assert spectrum.stable_k(basis, swapped, (1, 2, 3, 4), 0.9) == 2


class Block(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Toy(nn.Module):
    def __init__(self, layers: int) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Block() for _ in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.model.layers:
            x = block(x)
        return x


def test_residual_capture_returns_each_block_output():
    model = Toy(2)
    with spectrum.residual_capture(model) as captured:
        model(torch.ones(1, 3, 2))
    assert sorted(captured) == [0, 1]
    assert captured[1].shape == (1, 3, 2)


def step_map(d: int, gap: float) -> spectrum.StateMap:
    zero = torch.zeros(d, 0, dtype=torch.float64)
    return spectrum.StateMap(zero, zero, torch.full((d,), gap, dtype=torch.float64))


def test_prefix_steering_changes_the_last_position_for_the_first_steps_only():
    model = Toy(2)
    maps = {layer: step_map(2, 1.0) for layer in range(2)}
    means = {layer: torch.zeros(2, dtype=torch.float64) for layer in range(2)}
    with spectrum.prefix_steering(model, maps, means, steps=2):
        prompt = model(torch.zeros(1, 3, 2, dtype=torch.bfloat16))
        second = model(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
        third = model(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    # Both layers shift; earlier prompt positions stay as they were.
    assert prompt[0, :, 0].tolist() == [0.0, 0.0, 2.0]
    assert second[0, 0, 0].item() == 2.0
    assert third[0, 0, 0].item() == 0.0
    assert not model.model.layers[0]._forward_hooks


def test_prefix_steering_with_no_steps_is_the_plain_model():
    model = Toy(1)
    with spectrum.prefix_steering(model, {0: step_map(2, 1.0)}, {0: torch.zeros(2)}, steps=0):
        out = model(torch.zeros(1, 2, 2))
    assert out.abs().sum().item() == 0.0
