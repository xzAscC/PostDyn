"""Pure numerical tests for the grouped eight-class one-vs-rest logistic probe.

These tests are fully model-independent: they construct synthetic feature
matrices and verify the probe's mathematics, fold grouping, determinism, and
edge-case handling directly. No model is loaded or run.

Coverage areas (per the experiment brief's metric-4 requirements):

1. **EIGHT_LABELS** canonical order and count.
2. **Grouped K-fold** disjointness, completeness, determinism.
3. **Perfect separation** — AUROC = 1.0, balanced accuracy = 1.0.
4. **Chance-like data** — AUROC and balanced accuracy near 0.5.
5. **Repeatability** — same seed gives byte-identical results.
6. **Class imbalance** — rare classes are still evaluated when they span
   enough fold-distinct groups.
7. **Degenerate folds** — per-class skipping; error when a class is too
   concentrated for grouped CV.
8. **Finite checks** — all outputs are finite Python floats.
9. **Invalid inputs** — shape mismatch, out-of-range labels, NaN/Inf features,
   empty data, bad n_folds / alpha / label_names.
10. **Result structures** — frozen dataclasses with the documented fields.
11. **Logistic method** — ``method == "logistic"``, the L2 logistic objective
    is minimized to first-order optimality, the intercept is unregularized,
    optimizer failure / non-finite solutions raise clear errors, and logits
    (not ridge scores) drive the threshold/AUROC.
12. **Runtime smoke** — a realistic 400x4096, 8-class, 5-fold grid cell runs
    and returns finite metrics.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import pytest
import scipy.optimize
from scipy.special import expit

from postdyn.linear_probe import (
    EIGHT_LABELS,
    ClassProbeResult,
    LinearProbeResult,
    _logistic_fit,
    grouped_kfold_indices,
    linear_probe_score,
)

# =============================================================================
# Synthetic data builders
# =============================================================================


def _make_separable_data(
    n_per_class: int = 20,
    n_classes: int = 8,
    n_features: int = 12,
    seed: int = 0,
    scale: float = 10.0,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[Hashable]]:
    """Build perfectly linearly-separable multi-class data.

    Each class c occupies a distinct cluster centered at ``scale * e_c``
    (the c-th unit vector), so the one-vs-rest probe for class c recovers a
    direction along feature c that perfectly separates c from the rest.
    Each sample gets its own group so grouped CV cannot leak.
    """
    rng = np.random.default_rng(seed)
    n = n_per_class * n_classes
    X = np.zeros((n, n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    groups: list[Hashable] = []
    for c in range(n_classes):
        start = c * n_per_class
        end = start + n_per_class
        X[start:end, c] = scale
        X[start:end, :] += rng.normal(0.0, noise, (n_per_class, n_features))
        y[start:end] = c
        groups.extend(f"task_{c}_{i}" for i in range(n_per_class))
    return X, y, groups


def _make_chance_data(
    n: int = 800,
    n_classes: int = 8,
    n_features: int = 12,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[Hashable]]:
    """Build label-random data: features carry no class signal."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    groups: list[Hashable] = [f"g{i}" for i in range(n)]
    return X, y, groups


def _make_grouped_data(
    n_groups: int = 40,
    group_size: int = 5,
    n_classes: int = 8,
    n_features: int = 10,
    seed: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[Hashable]]:
    """Build data where samples share groups (one group → multiple samples).

    Each group has ``group_size`` samples whose labels are drawn independently
    but whose features carry a per-group bias.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    X = rng.standard_normal((n, n_features)).astype(np.float64)
    y = np.zeros(n, dtype=np.int64)
    groups: list[Hashable] = []
    for g in range(n_groups):
        base = g * group_size
        for j in range(group_size):
            y[base + j] = int(rng.integers(0, n_classes))
            groups.append(f"tmpl_{g}")
    return X, y, groups


def _make_ill_conditioned_fixture(
    seed: int = 999,
    n: int = 600,
    d: int = 200,
    log10_span: float = 6.0,
    margin: float = -0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ill-conditioned one-vs-rest fixture for budget regression.

    Features carry a geometric scale disparity across dimensions (Hessian
    condition number ~10**(2*log10_span)) plus a near-threshold signal, so the
    summed-loss L-BFGS-B fit needs >10**3 iterations at the old strict budget
    (maxiter=1000, ftol=1e-12) and exhausts it, but converges under the approved
    relaxed budget (maxiter=3000, ftol=1e-9, gtol=1e-6, maxls=50). It stands in
    for the real standardized 4096-dim grid cells that exhibited this failure.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d))
    scales = 10.0 ** np.linspace(-log10_span / 2.0, log10_span / 2.0, d)
    X = base * scales[None, :]
    w_true = np.zeros(d)
    w_true[0] = 1.0
    y = (X @ w_true > margin).astype(np.float64)
    return X, y


# =============================================================================
# 1. EIGHT_LABELS
# =============================================================================


class TestEightLabels:
    def test_exactly_eight_labels(self) -> None:
        assert len(EIGHT_LABELS) == 8

    def test_canonical_order(self) -> None:
        assert EIGHT_LABELS == (
            "python_valid",
            "python_syntax_error",
            "cpp",
            "js",
            "java",
            "go",
            "she",
            "he",
        )

    def test_all_unique(self) -> None:
        assert len(set(EIGHT_LABELS)) == 8

    def test_is_tuple(self) -> None:
        assert isinstance(EIGHT_LABELS, tuple)


# =============================================================================
# 2. Grouped K-fold
# =============================================================================


class TestGroupedKFold:
    """Disjointness, completeness, determinism of the fold builder."""

    def test_every_sample_in_exactly_one_test_fold(self) -> None:
        groups = [f"g{i % 10}" for i in range(100)]
        folds = grouped_kfold_indices(groups, n_folds=5, seed=42)
        all_test: list[int] = []
        for _, test in folds:
            all_test.extend(test.tolist())
        assert sorted(all_test) == list(range(100))

    def test_train_test_disjoint_per_fold(self) -> None:
        groups = [f"g{i % 10}" for i in range(100)]
        folds = grouped_kfold_indices(groups, n_folds=5, seed=42)
        for train_idx, test_idx in folds:
            assert set(train_idx.tolist()).isdisjoint(set(test_idx.tolist()))

    def test_train_union_is_all_minus_test(self) -> None:
        groups = [f"g{i % 10}" for i in range(100)]
        folds = grouped_kfold_indices(groups, n_folds=5, seed=42)
        for train_idx, test_idx in folds:
            assert sorted(train_idx.tolist()) == sorted(
                set(range(100)) - set(test_idx.tolist())
            )

    def test_whole_groups_stay_together(self) -> None:
        """The core invariant: no group's indices span multiple test folds,
        and a group's indices never appear in both train and test."""
        groups: list[Hashable] = []
        for g in range(20):
            groups.extend([f"task_{g}"] * 4)
        folds = grouped_kfold_indices(groups, n_folds=5, seed=7)
        index_to_group = {i: groups[i] for i in range(len(groups))}
        for train_idx, test_idx in folds:
            test_groups = {index_to_group[i] for i in test_idx}
            train_groups = {index_to_group[i] for i in train_idx}
            assert test_groups.isdisjoint(train_groups)
            # A group in test must have ALL its samples in test.
            for g in test_groups:
                g_indices = {i for i in range(len(groups)) if index_to_group[i] == g}
                assert g_indices.issubset(set(test_idx.tolist()))

    def test_deterministic_same_seed(self) -> None:
        groups = [f"g{i}" for i in range(50)]
        a = grouped_kfold_indices(groups, n_folds=5, seed=42)
        b = grouped_kfold_indices(groups, n_folds=5, seed=42)
        for (ta, sa), (tb, sb) in zip(a, b):
            assert np.array_equal(ta, tb)
            assert np.array_equal(sa, sb)

    def test_different_seeds_give_different_splits(self) -> None:
        groups = [f"g{i}" for i in range(50)]
        a = grouped_kfold_indices(groups, n_folds=5, seed=1)
        b = grouped_kfold_indices(groups, n_folds=5, seed=2)
        all_a = [tuple(te.tolist()) for _, te in a]
        all_b = [tuple(te.tolist()) for _, te in b]
        assert all_a != all_b

    def test_n_folds_too_few(self) -> None:
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            grouped_kfold_indices(["a", "b"], n_folds=1, seed=0)

    def test_n_folds_exceeds_unique_groups(self) -> None:
        with pytest.raises(ValueError, match="exceeds number of unique groups"):
            grouped_kfold_indices(["a", "b", "c"], n_folds=5, seed=0)

    def test_empty_groups(self) -> None:
        with pytest.raises(ValueError, match="groups must be non-empty"):
            grouped_kfold_indices([], n_folds=2, seed=0)

    def test_bool_n_folds_rejected(self) -> None:
        with pytest.raises(ValueError, match="not bool"):
            grouped_kfold_indices(["a", "b", "c", "d"], n_folds=True, seed=0)

    def test_float_n_folds_rejected(self) -> None:
        """A float n_folds must raise at this public boundary too."""
        with pytest.raises(ValueError, match="must be an int"):
            grouped_kfold_indices(["a", "b", "c", "d"], n_folds=cast(int, 2.0), seed=0)

    def test_integer_groups(self) -> None:
        groups = [0, 0, 1, 1, 2, 2, 3, 3]
        folds = grouped_kfold_indices(groups, n_folds=4, seed=0)
        assert len(folds) == 4


# =============================================================================
# 3. Perfect separation
# =============================================================================


class TestPerfectSeparation:
    """With orthogonal class clusters, every class probe should hit AUROC 1.0."""

    def test_all_classes_auroc_one(self) -> None:
        X, y, groups = _make_separable_data(
            n_per_class=25, n_classes=8, n_features=12, seed=0
        )
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert r.auroc == pytest.approx(1.0, abs=1e-6), (
                f"{name}: AUROC={r.auroc} (expected 1.0)"
            )

    def test_all_classes_balanced_accuracy_near_one(self) -> None:
        X, y, groups = _make_separable_data(
            n_per_class=25, n_classes=8, n_features=12, seed=0
        )
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert r.balanced_accuracy > 0.95, (
                f"{name}: balanced_accuracy={r.balanced_accuracy}"
            )

    def test_no_folds_skipped_when_balanced(self) -> None:
        X, y, groups = _make_separable_data(
            n_per_class=25, n_classes=8, n_features=12, seed=0
        )
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            assert result.by_label[name].n_folds_skipped == 0


# =============================================================================
# 4. Chance-like data
# =============================================================================


class TestChanceData:
    """Random features and labels → AUROC and balanced accuracy near 0.5."""

    def test_auroc_near_half(self) -> None:
        X, y, groups = _make_chance_data(n=800, n_classes=8, seed=1)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert 0.30 < r.auroc < 0.70, f"{name}: AUROC={r.auroc} (expected near 0.5)"

    def test_balanced_accuracy_near_half(self) -> None:
        X, y, groups = _make_chance_data(n=800, n_classes=8, seed=1)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert 0.30 < r.balanced_accuracy < 0.70, (
                f"{name}: balanced_accuracy={r.balanced_accuracy}"
            )


# =============================================================================
# 5. Repeatability
# =============================================================================


class TestRepeatability:
    def test_same_seed_identical_results(self) -> None:
        X, y, groups = _make_separable_data(seed=3)
        a = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        b = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        assert a.label_names == b.label_names
        for name in a.label_names:
            ra = a.by_label[name]
            rb = b.by_label[name]
            assert ra.balanced_accuracy == rb.balanced_accuracy
            assert ra.auroc == rb.auroc
            assert ra.n_positive == rb.n_positive
            assert ra.n_negative == rb.n_negative
            assert ra.n_folds_used == rb.n_folds_used
            assert ra.n_folds_skipped == rb.n_folds_skipped

    def test_different_seed_different_fold_counts_or_metrics(self) -> None:
        # With separable data the AUROC is always 1.0, but the fold
        # composition and hence exact balanced_accuracy can vary.  At minimum
        # the run must not crash and must return 8 classes.
        X, y, groups = _make_separable_data(seed=3)
        a = linear_probe_score(X, y, groups, n_folds=5, seed=11)
        b = linear_probe_score(X, y, groups, n_folds=5, seed=22)
        assert len(a.by_label) == len(b.by_label) == 8


# =============================================================================
# 6. Class imbalance
# =============================================================================


class TestClassImbalance:
    """Imbalanced classes should still be evaluated when they span enough
    fold-distinct groups."""

    def test_rare_class_evaluated_when_spans_multiple_groups(self) -> None:
        rng = np.random.default_rng(0)
        n_common = 100
        n_rare = 10
        n_total = n_common * 7 + n_rare
        X = rng.standard_normal((n_total, 8))
        y = np.zeros(n_total, dtype=np.int64)
        groups: list[Hashable] = []
        idx = 0
        for c in range(7):
            for _ in range(n_common):
                y[idx] = c
                X[idx, c] += 5.0
                groups.append(f"common_{c}_{idx}")
                idx += 1
        for _ in range(n_rare):
            y[idx] = 7
            X[idx, 7] += 5.0
            groups.append(f"rare_{idx}")
            idx += 1
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        rare = result.by_label[result.label_names[7]]
        assert rare.n_positive > 0
        assert rare.n_folds_used > 0
        assert rare.auroc > 0.8

    def test_imbalanced_does_not_crash(self) -> None:
        rng = np.random.default_rng(1)
        sizes = [80, 80, 80, 60, 40, 30, 20, 10]
        n = sum(sizes)
        X = rng.standard_normal((n, 8))
        y = np.zeros(n, dtype=np.int64)
        groups: list[Hashable] = []
        idx = 0
        for c, sz in enumerate(sizes):
            for _ in range(sz):
                y[idx] = c
                X[idx, c] += 3.0
                groups.append(f"task_{idx}")
                idx += 1
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        assert len(result.by_label) == 8
        for name in result.label_names:
            r = result.by_label[name]
            assert r.n_positive + r.n_negative == n


# =============================================================================
# 7. Degenerate folds
# =============================================================================


class TestDegenerateFolds:
    """Per-class skipping; error when a class is too concentrated."""

    def test_class_in_single_group_raises(self) -> None:
        """If a class's samples are all in one group, grouped CV cannot
        evaluate it and must raise."""
        rng = np.random.default_rng(0)
        n = 80
        X = rng.standard_normal((n, 8))
        y = np.zeros(n, dtype=np.int64)
        groups: list[Hashable] = []
        idx = 0
        for c in range(7):
            for _ in range(10):
                y[idx] = c
                X[idx, c] += 3.0
                groups.append(f"task_{idx}")
                idx += 1
        # Class 7: all 10 samples share the SAME group.
        for _ in range(10):
            y[idx] = 7
            X[idx, 7] += 3.0
            groups.append("single_group_class7")
            idx += 1
        with pytest.raises(ValueError, match="concentrated in too few folds"):
            linear_probe_score(X, y, groups, n_folds=5, seed=42)

    def test_class_too_rare_all_folds_degenerate(self) -> None:
        """A class with exactly one sample: every training fold is
        single-class for it → all folds skipped → ValueError."""
        rng = np.random.default_rng(0)
        n = 71
        X = rng.standard_normal((n, 8))
        y = np.zeros(n, dtype=np.int64)
        groups: list[Hashable] = []
        idx = 0
        for c in range(7):
            for _ in range(10):
                y[idx] = c
                X[idx, c] += 3.0
                groups.append(f"task_{idx}")
                idx += 1
        # Class 7: a single sample.
        y[idx] = 7
        X[idx, 7] += 3.0
        groups.append("only_class7_sample")
        with pytest.raises(ValueError, match="concentrated in too few folds"):
            linear_probe_score(X, y, groups, n_folds=5, seed=42)

    def test_some_folds_skipped_still_succeeds(self) -> None:
        """A class spanning two fold-distinct groups can lose some folds to
        degeneracy but still be evaluated."""
        rng = np.random.default_rng(0)
        n = 74
        X = rng.standard_normal((n, 8))
        y = np.zeros(n, dtype=np.int64)
        groups: list[Hashable] = []
        idx = 0
        for c in range(7):
            for _ in range(10):
                y[idx] = c
                X[idx, c] += 4.0
                groups.append(f"task_{idx}")
                idx += 1
        # Class 7: 4 samples, spread across 2 distinct groups.
        for j in range(4):
            y[idx] = 7
            X[idx, 7] += 4.0
            groups.append(f"rare_group_{j // 2}")
            idx += 1
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        r = result.by_label[result.label_names[7]]
        assert r.n_positive > 0
        assert r.n_folds_used >= 1


# =============================================================================
# 8. Finite checks
# =============================================================================


class TestFiniteChecks:
    def test_all_results_finite(self) -> None:
        X, y, groups = _make_separable_data(seed=5)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert math.isfinite(r.balanced_accuracy)
            assert math.isfinite(r.auroc)

    def test_nan_features_rejected(self) -> None:
        X, y, groups = _make_separable_data(seed=5)
        X[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            linear_probe_score(X, y, groups, n_folds=5, seed=42)

    def test_inf_features_rejected(self) -> None:
        X, y, groups = _make_separable_data(seed=5)
        X[0, 0] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            linear_probe_score(X, y, groups, n_folds=5, seed=42)

    def test_auroc_in_unit_interval(self) -> None:
        X, y, groups = _make_chance_data(n=400, seed=9)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert 0.0 <= r.auroc <= 1.0

    def test_balanced_accuracy_in_unit_interval(self) -> None:
        X, y, groups = _make_chance_data(n=400, seed=9)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert 0.0 <= r.balanced_accuracy <= 1.0


# =============================================================================
# 9. Invalid inputs
# =============================================================================


class TestInvalidInputs:
    def test_feature_label_length_mismatch(self) -> None:
        X = np.zeros((10, 4))
        y = np.zeros(9, dtype=int)
        groups = ["g"] * 10
        with pytest.raises(ValueError, match="length mismatch"):
            linear_probe_score(X, y, groups)

    def test_feature_group_length_mismatch(self) -> None:
        X = np.zeros((10, 4))
        y = np.zeros(10, dtype=int)
        groups = ["g"] * 9
        with pytest.raises(ValueError, match="length mismatch"):
            linear_probe_score(X, y, groups)

    def test_labels_out_of_range_high(self) -> None:
        X = np.zeros((10, 4))
        y = np.full(10, 8, dtype=int)
        groups = [f"g{i}" for i in range(10)]
        with pytest.raises(ValueError, match=r"labels must be in \[0"):
            linear_probe_score(X, y, groups)

    def test_labels_negative(self) -> None:
        X = np.zeros((10, 4))
        y = np.full(10, -1, dtype=int)
        groups = [f"g{i}" for i in range(10)]
        with pytest.raises(ValueError, match=r"labels must be in \[0"):
            linear_probe_score(X, y, groups)

    def test_non_integer_labels(self) -> None:
        X = np.zeros((10, 4))
        y = np.linspace(0.0, 1.0, 10)
        groups = [f"g{i}" for i in range(10)]
        with pytest.raises(ValueError, match="integer class indices"):
            linear_probe_score(X, y, groups)

    def test_empty_features(self) -> None:
        X = np.zeros((0, 4))
        y = np.zeros(0, dtype=int)
        groups: list[Hashable] = []
        with pytest.raises(ValueError, match="non-empty"):
            linear_probe_score(X, y, groups)

    def test_n_folds_too_few(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            linear_probe_score(X, y, groups, n_folds=1)

    def test_bool_n_folds_rejected(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="not bool"):
            linear_probe_score(X, y, groups, n_folds=True)

    def test_float_n_folds_rejected(self) -> None:
        """A float n_folds (e.g. 5.0) must raise, not silently truncate."""
        X, y, groups = _make_separable_data(seed=0)
        # ``cast`` keeps the static checker happy while the runtime value is a
        # float, so we genuinely exercise the non-int rejection path.
        with pytest.raises(ValueError, match="must be an int"):
            linear_probe_score(X, y, groups, n_folds=cast(int, 5.0))

    def test_float_n_folds_rejected_even_when_integral(self) -> None:
        """Even an integral float like 3.0 must be rejected (strict int)."""
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="must be an int"):
            linear_probe_score(X, y, groups, n_folds=cast(int, 3.0))

    def test_n_folds_exceeds_unique_groups(self) -> None:
        X = np.zeros((10, 4))
        y = np.zeros(10, dtype=int)
        groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
        with pytest.raises(ValueError, match="exceeds number of unique groups"):
            linear_probe_score(X, y, groups, n_folds=6)

    def test_alpha_zero(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="alpha must be a positive"):
            linear_probe_score(X, y, groups, alpha=0.0)

    def test_alpha_negative(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="alpha must be a positive"):
            linear_probe_score(X, y, groups, alpha=-1.0)

    def test_alpha_nan(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="alpha must be a positive"):
            linear_probe_score(X, y, groups, alpha=float("nan"))

    def test_alpha_inf(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        with pytest.raises(ValueError, match="alpha must be a positive"):
            linear_probe_score(X, y, groups, alpha=float("inf"))

    def test_duplicate_label_names(self) -> None:
        X = np.zeros((20, 4))
        y = np.tile(np.arange(2), 10)
        groups = [f"g{i}" for i in range(20)]
        with pytest.raises(ValueError, match="label_names must be unique"):
            linear_probe_score(X, y, groups, label_names=("a", "a", "b", "c"))

    def test_wrong_label_count(self) -> None:
        X = np.zeros((20, 4))
        y = np.tile(np.arange(4), 5)
        groups = [f"g{i}" for i in range(20)]
        with pytest.raises(ValueError, match="labels must be in"):
            linear_probe_score(X, y, groups, label_names=("a", "b"))

    def test_1d_features_rejected(self) -> None:
        X = np.zeros(20)
        y = np.zeros(20, dtype=int)
        groups = [f"g{i}" for i in range(20)]
        with pytest.raises(ValueError, match="must be 2-D"):
            linear_probe_score(X, y, groups)

    def test_3d_features_rejected(self) -> None:
        X = np.zeros((4, 5, 6))
        y = np.zeros(4, dtype=int)
        groups = [f"g{i}" for i in range(4)]
        with pytest.raises(ValueError, match="must be 2-D"):
            linear_probe_score(X, y, groups)


# =============================================================================
# 10. Result structures
# =============================================================================


class TestResultStructures:
    def test_linear_probe_result_fields(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42, alpha=2.0)
        assert isinstance(result, LinearProbeResult)
        assert result.label_names == EIGHT_LABELS
        assert result.n_folds == 5
        assert result.alpha == 2.0
        assert result.seed == 42
        assert result.standardize is True
        assert result.n_samples == len(y)
        assert result.n_groups == len(groups)

    def test_class_probe_result_fields(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            r = result.by_label[name]
            assert isinstance(r, ClassProbeResult)
            assert r.label == name
            assert isinstance(r.balanced_accuracy, float)
            assert isinstance(r.auroc, float)
            assert isinstance(r.n_positive, int)
            assert isinstance(r.n_negative, int)
            assert isinstance(r.n_folds_used, int)
            assert isinstance(r.n_folds_skipped, int)

    def test_by_label_keys_match_label_names(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        assert set(result.by_label.keys()) == set(result.label_names)
        assert len(result.by_label) == 8

    def test_pooled_counts_match_n_samples(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        n = len(y)
        for name in result.label_names:
            r = result.by_label[name]
            assert r.n_positive + r.n_negative == n

    def test_frozen_dataclass(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        with pytest.raises(Exception):
            setattr(result, "n_folds", 99)

    def test_standardize_false_works(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42, standardize=False)
        assert result.standardize is False
        for name in result.label_names:
            r = result.by_label[name]
            assert math.isfinite(r.auroc)

    def test_custom_label_names(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 6))
        y = rng.integers(0, 3, size=100).astype(np.int64)
        groups = [f"g{i}" for i in range(100)]
        names = ("alpha", "beta", "gamma")
        result = linear_probe_score(X, y, groups, label_names=names, seed=1)
        assert result.label_names == names


# =============================================================================
# 11. Grouped-data leakage safety (integration with multi-sample groups)
# =============================================================================


class TestGroupedDataLeakage:
    """With multi-sample groups, verify the probe cannot trivially memorize
    item identity via shared groups."""

    def test_runs_with_shared_groups(self) -> None:
        X, y, groups = _make_grouped_data(
            n_groups=40, group_size=5, n_classes=8, seed=2
        )
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        assert result.n_groups == 40
        assert result.n_samples == 200
        for name in result.label_names:
            r = result.by_label[name]
            assert r.n_positive + r.n_negative == 200

    def test_group_disjointness_holds_under_shared_groups(self) -> None:
        _X, _y, groups = _make_grouped_data(
            n_groups=40, group_size=5, n_classes=8, seed=2
        )
        folds = grouped_kfold_indices(groups, n_folds=5, seed=42)
        index_to_group = {i: groups[i] for i in range(len(groups))}
        for train_idx, test_idx in folds:
            train_groups = {index_to_group[i] for i in train_idx}
            test_groups = {index_to_group[i] for i in test_idx}
            assert train_groups.isdisjoint(test_groups)


# =============================================================================
# 12. Logistic method (objective, unregularized intercept, error paths)
# =============================================================================


class TestLogisticMethod:
    """Lock down the L2 logistic-regression estimator (metric-4 contract).

    These tests discriminate the logistic estimator from the previous ridge
    regression: the fitted parameters must satisfy first-order optimality of
    the summed logistic objective ``sum(logaddexp(0, eta) - y*eta) +
    0.5*alpha*||w||^2`` with an *unregularized* intercept, and optimizer
    failure must surface as a clear error.
    """

    def test_result_method_is_logistic(self) -> None:
        X, y, groups = _make_separable_data(seed=0)
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        assert isinstance(result, LinearProbeResult)
        assert result.method == "logistic"

    def test_logistic_fit_first_order_optimality(self) -> None:
        """At the fitted theta the logistic gradient is ~0 (ridge would not
        satisfy this)."""
        rng = np.random.default_rng(0)
        n, d = 300, 6
        X = rng.standard_normal((n, d))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(np.float64)
        alpha = 1.0
        w, b = _logistic_fit(X, y, alpha)
        eta = X @ w + b
        diff = expit(eta) - y
        grad_w = X.T @ diff + alpha * w
        grad_b = float(np.sum(diff))
        g_max = max(float(np.max(np.abs(grad_w))), abs(grad_b))
        assert g_max < 1e-4, f"max|grad|={g_max}"

    def test_logistic_fit_minimizes_stated_objective(self) -> None:
        """A random perturbation of the fitted weights must not lower the
        stated objective sum(logaddexp(0,eta)-y*eta)+0.5*alpha*||w||^2."""
        rng = np.random.default_rng(3)
        n, d = 400, 8
        X = rng.standard_normal((n, d))
        y = (X[:, 0] + 0.3 * X[:, 2] + rng.standard_normal(n) * 0.1 > 0.0).astype(
            np.float64
        )
        alpha = 2.0
        w, b = _logistic_fit(X, y, alpha)

        def obj(ww: np.ndarray, bb: float) -> float:
            eta = X @ ww + bb
            return float(
                np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * alpha * (ww @ ww)
            )

        base = obj(w, b)
        for _ in range(20):
            dw = rng.standard_normal(d) * 1e-3
            db = float(rng.standard_normal() * 1e-3)
            assert obj(w + dw, b + db) >= base - 1e-6

    def test_logistic_fit_intercept_unregularized(self) -> None:
        """With a crushing L2 penalty the weights shrink toward 0 but the
        intercept stays free to match the base rate of ``y`` (logit(p))."""
        rng = np.random.default_rng(1)
        n, d = 800, 5
        X = rng.standard_normal((n, d)) * 0.01  # negligible signal
        p = 0.25
        y = (rng.random(n) < p).astype(np.float64)
        alpha = 1e6
        w, b = _logistic_fit(X, y, alpha)
        assert float(np.max(np.abs(w))) < 1e-3, f"max|w|={float(np.max(np.abs(w)))}"
        # logit(p) = log(p / (1 - p))
        assert abs(b - math.log(p / (1.0 - p))) < 0.1, f"b={b}"

    def test_logistic_fit_returns_finite(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 4))
        y = (rng.random(200) < 0.4).astype(np.float64)
        w, b = _logistic_fit(X, y, 1.0)
        assert w.shape == (4,)
        assert math.isfinite(b)
        assert np.isfinite(w).all()

    def test_logistic_fit_deterministic(self) -> None:
        rng = np.random.default_rng(5)
        X = rng.standard_normal((150, 5))
        y = (rng.random(150) < 0.5).astype(np.float64)
        w1, b1 = _logistic_fit(X, y, 0.7)
        w2, b2 = _logistic_fit(X, y, 0.7)
        np.testing.assert_array_equal(w1, w2)
        assert b1 == b2

    def test_optimizer_failure_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When L-BFGS-B reports non-success, a clear RuntimeError is raised."""
        from postdyn import linear_probe as lp

        class _Failed:
            success = False
            message = "forced convergence failure"
            nit = 3
            x = np.zeros(3)

        monkeypatch.setattr(lp, "minimize", lambda *a, **k: _Failed())
        with pytest.raises(RuntimeError, match="forced convergence failure"):
            _logistic_fit(np.zeros((10, 2)), np.zeros(10), 1.0)

    def test_nonfinite_solution_raises_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finite-but-non-finite solution still raises a clear error."""
        from postdyn import linear_probe as lp

        class _NonFinite:
            success = True
            message = "ok"
            nit = 1
            x = np.array([np.nan, np.inf, 1.0])

        monkeypatch.setattr(lp, "minimize", lambda *a, **k: _NonFinite())
        with pytest.raises(RuntimeError, match="non-finite"):
            _logistic_fit(np.zeros((10, 2)), np.zeros(10), 1.0)

    def test_logit_threshold_zero_perfect_separation(self) -> None:
        """End-to-end: with separable clusters the logit-threshold(0) balanced
        accuracy is high for every class (scores are logits, not [0,1])."""
        X, y, groups = _make_separable_data(
            n_per_class=25, n_classes=8, n_features=12, seed=0
        )
        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)
        for name in result.label_names:
            assert result.by_label[name].balanced_accuracy > 0.95, name


# =============================================================================
# 13. Optimizer budget regression (real-grid maxiter failure)
# =============================================================================


class TestOptimizerBudgetRegression:
    """Pin the approved L-BFGS-B controls after the real-grid maxiter failure.

    A real standardized 4096-dim cell crawled past the old ``maxiter=1000`` /
    ``ftol=1e-12`` budget on ``python_valid``. These tests lock the approved
    relaxed controls (``maxiter=3000``, ``ftol=1e-9``, ``gtol=1e-6``,
    ``maxls=50``), prove they rescue a deterministic ill-conditioned fixture
    that genuinely needs >1000 iterations at the old budget, and confirm genuine
    optimizer non-success still surfaces as an error (no silent maxiter pass).
    """

    def test_logistic_fit_uses_approved_optimizer_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from postdyn import linear_probe as lp

        captured: dict[str, Any] = {}
        real_minimize = scipy.optimize.minimize

        def spy(*args: Any, **kwargs: Any) -> Any:
            captured["method"] = kwargs.get("method")
            captured["jac"] = kwargs.get("jac")
            captured["options"] = kwargs.get("options")
            return real_minimize(*args, **kwargs)

        monkeypatch.setattr(lp, "minimize", spy)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 5))
        y = (rng.random(60) < 0.4).astype(np.float64)

        _logistic_fit(X, y, 1.0)

        assert captured["method"] == "L-BFGS-B"
        assert captured["jac"] is True
        assert captured["options"] == {
            "maxiter": 3000,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "maxls": 50,
        }

    def test_ill_conditioned_fixture_exhausts_old_strict_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under the old strict/low budget the fixture genuinely runs out of
        iterations (>1000 needed) and ``_logistic_fit`` raises."""
        from postdyn import linear_probe as lp

        X, y = _make_ill_conditioned_fixture()
        real_minimize = scipy.optimize.minimize
        captured: dict[str, Any] = {}

        def force_old_budget(*args: Any, **kwargs: Any) -> Any:
            kwargs["options"] = {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6}
            res = real_minimize(*args, **kwargs)
            captured["nit"] = int(res.nit)
            captured["success"] = bool(res.success)
            return res

        monkeypatch.setattr(lp, "minimize", force_old_budget)

        with pytest.raises(RuntimeError, match="did not converge"):
            _logistic_fit(X, y, 1e-3)
        assert bool(captured["success"]) is False
        assert int(captured["nit"]) >= 1000

    def test_ill_conditioned_fixture_converges_under_production_budget(self) -> None:
        """The approved production budget converges the same fixture to a
        finite, first-order-optimal solution (no RuntimeError)."""
        X, y = _make_ill_conditioned_fixture()
        w, b = _logistic_fit(X, y, 1e-3)

        assert w.shape == (X.shape[1],)
        assert np.isfinite(w).all()
        assert math.isfinite(b)
        # First-order optimality on the stated summed-loss objective.
        eta = X @ w + b
        diff = expit(eta) - y
        grad_w = X.T @ diff + 1e-3 * w
        grad_b = float(np.sum(diff))
        g_max = max(float(np.max(np.abs(grad_w))), abs(grad_b))
        assert g_max < 1e-4, f"max|grad|={g_max}"


# =============================================================================
# 14. Runtime smoke: realistic grid cell (400x4096, 8 classes, 5 folds)
# =============================================================================


class TestRuntimeSmoke400x4096:
    """The full experiment grid is 110 cells, each 400x4096 with 8 classes and
    5 folds. This smoke test confirms one such cell runs and returns finite
    metrics in a single fit-and-score pass (40 one-vs-rest logistic fits)."""

    def test_probe_400x4096_8class_5fold_finite(self) -> None:
        rng = np.random.default_rng(42)
        n, d, n_classes = 400, 4096, 8
        X = rng.standard_normal((n, d))
        y = rng.integers(0, n_classes, size=n).astype(np.int64)
        # Inject a weak per-class directional signal so every class is
        # non-degenerate and the probe has something to learn.
        for c in range(n_classes):
            X[y == c, c] += 3.0
        groups = [f"g{i}" for i in range(n)]

        result = linear_probe_score(X, y, groups, n_folds=5, seed=42)

        assert result.method == "logistic"
        assert len(result.by_label) == n_classes
        assert result.n_samples == n
        for name in result.label_names:
            r = result.by_label[name]
            assert math.isfinite(r.balanced_accuracy), name
            assert math.isfinite(r.auroc), name
            assert 0.0 <= r.balanced_accuracy <= 1.0, name
            assert 0.0 <= r.auroc <= 1.0, name
            assert r.n_positive + r.n_negative == n, name
