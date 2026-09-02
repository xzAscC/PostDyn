"""Grouped eight-class one-vs-rest L2 logistic linear-probe scoring.

This is **metric 4** of the four-metric representation/readout suite. Given a
matrix of pre-extracted
hidden states and one of eight category labels per sample, it trains a
**one-vs-rest L2 logistic regression** probe per class under **leakage-safe
task/template-grouped K-fold folds** and reports **balanced accuracy** and
**AUROC** per class.

The eight classes (in canonical order)::

    python_valid, python_syntax_error, cpp, js, java, go, she, he

* ``python_valid`` / ``python_syntax_error`` come from the target concept.
* ``cpp``, ``js``, ``java``, ``go`` come from the HumanEval-X code directions.
* ``she`` / ``he`` come from the WinoGender ``gender_she_vs_he`` control.

Design contract
---------------

* **Model-independent.** Operates purely on a feature matrix; no model is
  loaded, run, or imported here.
* **Leakage-safe grouped folds.** All samples that share a grouping key
  (HumanEval-X task id or WinoGender template id) stay in the *same* fold, so
  the probe cannot memorize item identity. The fold shuffle uses Python's
  ``random.Random(seed)`` which is stable across versions and platforms.
* **Deterministic.** Same inputs + ``seed`` always produce byte-identical
  results. Each logistic fit starts from a zero parameter vector and is solved
  with scipy's deterministic L-BFGS-B.
* **Per-fold train-only standardization.** Each fold's logistic probe is fit on
  features standardized with *that fold's training* mean/std only (applied to
  train and test), so the test fold never leaks into the scaling statistics and
  ``alpha`` is scale-invariant. Disable with ``standardize=False``.
* **Degenerate-fold tolerant.** A fold is *degenerate* for class ``c`` when the
  training side lacks either a positive or a negative example of ``c``; that
  fold is skipped for ``c`` and counted in ``n_folds_skipped``. If *every* fold
  is degenerate for a class (class too rare for the fold structure), a
  ``ValueError`` is raised rather than returning a meaningless metric.
* **Logits drive both metrics.** Each fit returns logits ``eta = X w + b``;
  predictions threshold logits at ``0`` (equivalently sigmoid(eta) >= 0.5) for
  balanced accuracy, and AUROC consumes the raw logits.
* **Finite checks.** All inputs are checked for NaN/Inf; all outputs are
  finite Python floats. Optimizer non-convergence or a non-finite solution
  raises a clear ``RuntimeError`` naming the failing class/fold.
* **Declared dependencies only.** Implemented with ``numpy`` and ``scipy``
  (both declared in ``pyproject.toml``; ``scipy>=1.14``). ``sklearn`` is
  intentionally not used.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# =============================================================================
# Constants
# =============================================================================

#: The eight canonical one-vs-rest probe classes, in fixed order.
EIGHT_LABELS: tuple[str, ...] = (
    "python_valid",
    "python_syntax_error",
    "cpp",
    "js",
    "java",
    "go",
    "she",
    "he",
)


# =============================================================================
# Result data structures
# =============================================================================


@dataclass(frozen=True)
class ClassProbeResult:
    """One-vs-rest logistic probe readout for a single class.

    Attributes:
        label: Human-readable class name (one of ``EIGHT_LABELS``).
        balanced_accuracy: ``0.5 * (TPR + TNR)`` on predictions pooled across
            all non-degenerate test folds (logits thresholded at ``0``).
        auroc: Area under the ROC curve via the Mann-Whitney U statistic with
            average ranks for ties, on the raw per-class **logits** pooled
            across all non-degenerate test folds.
        n_positive: Total positives of this class scored across all
            non-degenerate folds.
        n_negative: Total negatives (rest class) scored across all
            non-degenerate folds.
        n_folds_used: Folds in which the probe was successfully trained
            (training side had at least one positive and one negative).
        n_folds_skipped: Folds skipped for this class because the training
            side was single-class.
    """

    label: str
    balanced_accuracy: float
    auroc: float
    n_positive: int
    n_negative: int
    n_folds_used: int
    n_folds_skipped: int


@dataclass(frozen=True)
class LinearProbeResult:
    """Full one-vs-rest probe result over all classes.

    Attributes:
        label_names: Canonical class names in order (matches ``by_label`` keys).
        by_label: ``{class_name: ClassProbeResult}``. Iterating ``label_names``
            yields results in canonical order.
        n_folds: Number of folds requested.
        alpha: L2 penalty on the logistic weights (intercept is unregularized).
        seed: Seed used for the grouped-fold shuffle.
        standardize: Whether per-fold train standardization was applied.
        n_samples: Total number of samples scored (``features.shape[0]``).
        n_groups: Number of distinct grouping keys.
        method: Estimator identifier; always ``"logistic"`` (one binary L2
            logistic regression per class, fit by scipy L-BFGS-B).
    """

    label_names: tuple[str, ...]
    by_label: dict[str, ClassProbeResult]
    n_folds: int
    alpha: float
    seed: int
    standardize: bool
    n_samples: int
    n_groups: int
    method: str = "logistic"


# =============================================================================
# Input validation
# =============================================================================


def _to_float64_2d(features: object, name: str) -> np.ndarray:
    """Coerce ``features`` to a contiguous ``float64`` 2-D array."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D (n_samples, n_features), got shape {arr.shape}"
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return np.ascontiguousarray(arr)


def _to_int64_1d(labels: object, name: str) -> np.ndarray:
    """Coerce ``labels`` to a contiguous ``int64`` 1-D array."""
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D (n_samples,), got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    int_arr = arr.astype(np.int64)
    if not np.array_equal(int_arr, arr):
        raise ValueError(f"{name} must contain integer class indices")
    return np.ascontiguousarray(int_arr)


def _validate_inputs(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[Hashable],
    label_names: Sequence[str],
    n_folds: int,
    alpha: float,
) -> int:
    """Validate all public arguments. Returns the number of unique groups."""
    n_samples = features.shape[0]
    if labels.shape[0] != n_samples:
        raise ValueError(
            f"features and labels length mismatch: {n_samples} vs {labels.shape[0]}"
        )
    if len(groups) != n_samples:
        raise ValueError(
            f"features and groups length mismatch: {n_samples} vs {len(groups)}"
        )

    n_classes = len(label_names)
    if n_classes == 0:
        raise ValueError("label_names must be non-empty")
    if len(set(label_names)) != n_classes:
        raise ValueError(f"label_names must be unique, got {list(label_names)}")

    label_min = int(labels.min())
    label_max = int(labels.max())
    if label_min < 0:
        raise ValueError(f"labels must be in [0, {n_classes}), got minimum {label_min}")
    if label_max >= n_classes:
        raise ValueError(
            f"labels must be in [0, {n_classes}), got maximum {label_max}"
            + f" with {n_classes} label names"
        )

    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be a positive finite float, got {alpha!r}")

    # bool is an int subclass; reject it so True/False can't become 1/0 folds,
    # and reject any non-int (e.g. 5.0) so it can't slip through to range() and
    # raise a confusing TypeError deep inside the fold builder.
    if isinstance(n_folds, bool):
        raise ValueError(f"n_folds must be an int, not bool, got {n_folds!r}")
    if not isinstance(n_folds, int):
        raise ValueError(
            f"n_folds must be an int, got {type(n_folds).__name__}: {n_folds!r}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    # Collect unique groups (deterministic ordering for stable error messages).
    group_to_indices: dict[Hashable, list[int]] = {}
    for idx, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(idx)
    n_groups = len(group_to_indices)
    if n_groups < n_folds:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds number of unique groups ("
            + f"{n_groups}); grouped CV is impossible"
        )
    return n_groups


# =============================================================================
# Grouped K-fold (public: enables direct disjointness testing)
# =============================================================================


def grouped_kfold_indices(
    groups: Sequence[Hashable],
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic grouped K-fold: every whole group stays in one fold.

    Groups are collected, deterministically ordered, shuffled with a seeded
    Python ``random.Random`` (stable across versions/platforms), then assigned
    round-robin to folds. Each sample appears in exactly one test fold.

    Args:
        groups: Per-sample grouping key (HumanEval-X task id or WinoGender
            template id). Must be hashable.
        n_folds: Number of folds (>= 2, <= number of unique groups).
        seed: Seed for the group shuffle.

    Returns:
        List of ``(train_indices, test_indices)`` pairs, one per fold, as
        sorted ``int64`` arrays. Every test set is disjoint from its train set,
        and the union of all test sets is ``range(len(groups))``.

    Raises:
        ValueError: If ``n_folds < 2`` or exceeds the number of unique groups,
            or if ``groups`` is empty.
    """
    n = len(groups)
    if n == 0:
        raise ValueError("groups must be non-empty")
    # bool is an int subclass; reject it so True/False can't become 1/0 folds,
    # and reject any non-int (e.g. 5.0) before range() raises a TypeError.
    if isinstance(n_folds, bool):
        raise ValueError(f"n_folds must be an int, not bool, got {n_folds!r}")
    if not isinstance(n_folds, int):
        raise ValueError(
            f"n_folds must be an int, got {type(n_folds).__name__}: {n_folds!r}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    group_to_indices: dict[Hashable, list[int]] = {}
    for idx, g in enumerate(groups):
        group_to_indices.setdefault(g, []).append(idx)
    unique_groups = list(group_to_indices.keys())
    n_groups = len(unique_groups)
    if n_groups < n_folds:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds number of unique groups ("
            + f"{n_groups}); grouped CV is impossible"
        )

    # Deterministic pre-order (stable regardless of dict/set iteration order),
    # then seeded shuffle. The pre-order key avoids cross-type comparison
    # errors while remaining deterministic.
    unique_groups.sort(key=lambda g: (type(g).__name__, str(g)))
    shuffled = list(unique_groups)
    random.Random(seed).shuffle(shuffled)

    fold_member_indices: list[list[int]] = [[] for _ in range(n_folds)]
    for pos, g in enumerate(shuffled):
        fold_member_indices[pos % n_folds].extend(group_to_indices[g])

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(n, dtype=np.int64)
    for f in range(n_folds):
        test_idx = np.array(sorted(fold_member_indices[f]), dtype=np.int64)
        test_set = set(test_idx.tolist())
        train_idx = all_indices[[i not in test_set for i in range(n)]]
        folds.append((train_idx, test_idx))
    return folds


# =============================================================================
# L2 logistic regression (scipy L-BFGS-B)
# =============================================================================


def _standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using *train* mean/std; apply to both train and test."""
    mean = X_train.mean(axis=0, keepdims=True)
    # Population std (correction=0); floor guards zero-variance dims.
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std


#: L-BFGS-B iteration budget. Real grid cells (standardized 4096-dim,
#: near-separable one-vs-rest) reach ~10^3 iterations at strict tolerances, so
#: the cap leaves headroom while bounding worst-case cost across the 110 cells.
#: Paired with a relaxed ``ftol`` (1e-9, not 1e-12) and ``maxls=50`` so the line
#: search does not stall on ill-conditioned Hessians; ``gtol=1e-6`` still pins a
#: genuine first-order optimum, so genuine non-convergence surfaces as an error.
_LOGISTIC_MAXITER = 3000


def _logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    context: str = "logistic fit",
    maxiter: int = _LOGISTIC_MAXITER,
) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression with an unregularized intercept.

    Minimizes the summed logistic loss with an L2 penalty on the weights only::

        L(w, b) = sum_i [ logaddexp(0, eta_i) - y_i * eta_i ] + 0.5 * alpha * ||w||^2
        eta_i = X_i . w + b

    where ``logaddexp(0, eta) = log(1 + exp(eta))`` is the numerically stable
    softplus and the intercept ``b`` is intentionally **not** regularized. The
    gradient uses ``scipy.special.expit`` (numerically stable sigmoid). Solved
    with scipy's L-BFGS-B from a zero start, which is fully deterministic for a
    fixed feature matrix.

    Args:
        X: ``(n, d)`` float64 feature matrix (typically already standardized).
        y: ``(n,)`` float64 targets in ``[0, 1]``.
        alpha: Positive L2 penalty on the weights only (intercept exempt).
        context: Human-readable label included in raised errors (e.g. the
            class/fold being fit).
        maxiter: L-BFGS-B iteration cap.

    Returns:
        ``(weights, intercept)`` where ``weights`` has shape ``(d,)`` and
        ``intercept`` is a Python float. These define the logits
        ``eta = X @ weights + intercept``.

    Raises:
        RuntimeError: If L-BFGS-B reports non-convergence or returns a
            non-finite parameter vector. The message includes ``context`` and
            the optimizer's own diagnostic so the failing class/fold is clear.
    """
    d = X.shape[1]
    # theta = [w (d,), b (1,)]; zero start => deterministic across runs.
    theta0 = np.zeros(d + 1, dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w = theta[:d]
        b = theta[d]
        eta = X @ w + b
        # Stable softplus data loss + L2 on weights (intercept unregularized).
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta)) + 0.5 * alpha * float(
            np.dot(w, w)
        )
        # Stable sigmoid residual.
        diff = expit(eta) - y
        grad = np.empty(d + 1, dtype=np.float64)
        grad[:d] = X.T @ diff + alpha * w
        grad[d] = float(np.sum(diff))
        return loss, grad

    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 50},
    )

    if not res.success:
        raise RuntimeError(
            f"{context}: L-BFGS-B did not converge ({str(res.message).strip()!s}); "
            f"iterations={res.nit}"
        )
    theta = np.asarray(res.x, dtype=np.float64)
    if not np.isfinite(theta).all():
        raise RuntimeError(
            f"{context}: L-BFGS-B returned non-finite parameters "
            f"(min={float(np.min(theta))}, max={float(np.max(theta))})"
        )
    return np.ascontiguousarray(theta[:d]), float(theta[d])


# =============================================================================
# Metrics: balanced accuracy and AUROC
# =============================================================================


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """``0.5 * (TPR + TNR)``; absent classes contribute a neutral 0.5 term.

    Args:
        y_true: Boolean/0-1 ground truth, shape ``(n,)``.
        y_pred: Boolean/0-1 predictions (threshold already applied).

    Returns:
        Balanced accuracy in ``[0, 1]``.
    """
    truth = y_true.astype(bool)
    pred = y_pred.astype(bool)
    n_pos = int(truth.sum())
    n_neg = int((~truth).sum())
    tp = int((truth & pred).sum())
    tn = int((~truth & ~pred).sum())
    tpr = tp / n_pos if n_pos > 0 else 0.5
    tnr = tn / n_neg if n_neg > 0 else 0.5
    return 0.5 * (tpr + tnr)


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    """1-based average ranks, with tied values sharing their mean rank.

    This is the tie-corrected rank function that makes the Mann-Whitney U
    AUROC well-defined for arbitrary real-valued scores.
    """
    n = scores.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    # Stable sort so ties keep their original order.
    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and scores_sorted[j] == scores_sorted[i]:
            j += 1
        # Sorted positions i..j-1 (0-based) map to 1-based ranks i+1..j;
        # ties share their mean.
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via Mann-Whitney U with average ranks.

    ``AUROC = P(score_pos > score_neg)`` with ties counted as half-wins. For a
    test set with zero positives or zero negatives there is no discrimination
    information, so ``0.5`` (chance) is returned.

    Args:
        y_true: Boolean/0-1 ground truth, shape ``(n,)``.
        y_score: Real-valued scores (higher = more likely positive).

    Returns:
        AUROC in ``[0, 1]``.
    """
    truth = y_true.astype(bool)
    n_pos = int(truth.sum())
    n_neg = int((~truth).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(y_score)
    sum_pos_ranks = float(ranks[truth].sum())
    u_statistic = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return u_statistic / (n_pos * n_neg)


# =============================================================================
# Main entry point
# =============================================================================


def linear_probe_score(
    features: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[Hashable],
    *,
    label_names: Sequence[str] = EIGHT_LABELS,
    n_folds: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
    standardize: bool = True,
) -> LinearProbeResult:
    """Grouped one-vs-rest L2 logistic linear-probe scoring.

    For each class ``c``, trains a one-vs-rest L2 logistic regression probe
    (positive = ``c``, negative = union of all other classes) under
    leakage-safe grouped K-fold cross-validation, then reports balanced
    accuracy and AUROC on logits pooled across the non-degenerate folds.

    Args:
        features: ``(n_samples, n_features)`` float matrix of pre-extracted
            hidden states. NaN/Inf are rejected.
        labels: ``(n_samples,)`` integer class indices in
            ``[0, len(label_names))``.
        groups: ``(n_samples,)`` hashable grouping keys (e.g. HumanEval-X task
            ids or WinoGender template ids). Every sample sharing a key stays
            in the same fold.
        label_names: Human-readable class names (default: the canonical eight).
            Its length defines the number of classes.
        n_folds: Number of grouped folds (>= 2 and <= number of unique groups).
        alpha: L2 penalty on the logistic weights; the intercept is
            unregularized (positive, finite).
        seed: Seed for the grouped-fold shuffle (deterministic).
        standardize: If ``True`` (default), standardize features per fold using
            *train* mean/std only; if ``False``, use raw features.

    Returns:
        A :class:`LinearProbeResult` with one :class:`ClassProbeResult` per
        class in ``label_names`` order; ``result.method == "logistic"``.

    Raises:
        ValueError: On shape/length mismatch, out-of-range or non-integer
            labels, NaN/Inf features, invalid ``alpha``/``n_folds``, or when a
            class is too rare to be cross-validated (all folds degenerate for
            it).
        RuntimeError: If an individual logistic fit fails to converge or
            returns non-finite parameters. The message names the failing
            class and fold.
    """
    feats = _to_float64_2d(features, "features")
    lbls = _to_int64_1d(labels, "labels")
    names = tuple(label_names)
    n_groups = _validate_inputs(feats, lbls, groups, names, n_folds, alpha)

    n_samples = feats.shape[0]
    n_classes = len(names)
    folds = grouped_kfold_indices(groups, n_folds, seed)

    by_label: dict[str, ClassProbeResult] = {}
    for c in range(n_classes):
        label_name = names[c]
        is_positive = (lbls == c).astype(np.float64)

        pooled_truth: list[int] = []
        pooled_scores: list[float] = []
        folds_used = 0
        folds_skipped = 0

        for train_idx, test_idx in folds:
            y_train_c = is_positive[train_idx]
            y_test_c = is_positive[test_idx]
            n_train_pos = int(y_train_c.sum())
            n_train_neg = int((1.0 - y_train_c).sum())
            # Degenerate fold for this class: train side is single-class.
            if n_train_pos == 0 or n_train_neg == 0:
                folds_skipped += 1
                continue

            X_train = feats[train_idx]
            X_test = feats[test_idx]
            if standardize:
                X_train, X_test = _standardize_train_test(X_train, X_test)

            weights, intercept = _logistic_fit(
                X_train,
                y_train_c,
                alpha,
                context=(
                    f"class '{label_name}' fold with "
                    f"{n_train_pos} pos / {n_train_neg} neg train samples"
                ),
            )
            # Logits drive both metrics (threshold 0 <=> sigmoid >= 0.5).
            logits = X_test @ weights + intercept

            pooled_scores.extend(float(s) for s in logits)
            pooled_truth.extend(int(v) for v in y_test_c)
            folds_used += 1

        if folds_used == 0:
            n_pos_total = int(is_positive.sum())
            raise ValueError(
                f"Class '{label_name}' has {n_pos_total} sample(s) but could "
                + f"not be cross-validated: all {n_folds} fold(s) were "
                + f"degenerate on the training side. Use fewer folds or more "
                + f"data for this class."
            )

        truth_arr = np.asarray(pooled_truth, dtype=np.float64)
        score_arr = np.asarray(pooled_scores, dtype=np.float64)
        n_pos_pooled = int(truth_arr.sum())
        n_neg_pooled = int((1.0 - truth_arr).sum())
        if n_pos_pooled == 0 or n_neg_pooled == 0:
            n_pos_total = int(is_positive.sum())
            raise ValueError(
                f"Class '{label_name}' has {n_pos_total} sample(s) but its "
                + f"groups are concentrated in too few folds: the pooled test "
                + f"set has {n_pos_pooled} positive(s) and {n_neg_pooled} "
                + f"negative(s). Use fewer folds or ensure the class spans at "
                + f"least two fold-distinct groups."
            )

        pred_arr = (score_arr >= 0.0).astype(np.float64)

        balanced_acc = _balanced_accuracy(truth_arr, pred_arr)
        auc = _auroc(truth_arr, score_arr)

        if not (math.isfinite(balanced_acc) and math.isfinite(auc)):
            raise ValueError(
                f"Class '{label_name}' produced non-finite metric "
                + f"(balanced_accuracy={balanced_acc!r}, auroc={auc!r})"
            )

        by_label[label_name] = ClassProbeResult(
            label=label_name,
            balanced_accuracy=balanced_acc,
            auroc=auc,
            n_positive=int(truth_arr.sum()),
            n_negative=int((1.0 - truth_arr).sum()),
            n_folds_used=folds_used,
            n_folds_skipped=folds_skipped,
        )

    return LinearProbeResult(
        label_names=names,
        by_label=by_label,
        n_folds=n_folds,
        alpha=alpha,
        seed=seed,
        standardize=standardize,
        n_samples=n_samples,
        n_groups=n_groups,
        method="logistic",
    )


__all__ = [
    "EIGHT_LABELS",
    "ClassProbeResult",
    "LinearProbeResult",
    "grouped_kfold_indices",
    "linear_probe_score",
]
