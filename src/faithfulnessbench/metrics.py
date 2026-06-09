"""Evaluation metrics, implemented from scratch in numpy.

We deliberately avoid scikit-learn / scipy so the harness installs and runs from a
clean clone with only numpy. Every function here is unit-tested against hand-computed
values in ``tests/test_metrics.py``.

Conventions:
* ``scores``  -- higher means "more likely the positive class".
* ``labels``  -- array of {0, 1}; 1 is the positive class (here: "unfaithful").
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def _rankdata(a: ArrayLike) -> np.ndarray:
    """Average ranks (1-based), ties get the mean of the ranks they span.

    Equivalent to ``scipy.stats.rankdata(a, method="average")``.
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    sorted_ranks = np.empty(n, dtype=float)
    pos = 0
    while pos < n:
        end = pos
        while end + 1 < n and sorted_a[end + 1] == sorted_a[pos]:
            end += 1
        # ranks pos+1 .. end+1 (1-based); their mean:
        sorted_ranks[pos : end + 1] = (pos + end) / 2.0 + 1.0
        pos = end + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.

    Returns the probability that a random positive scores above a random negative
    (ties counted as 0.5). NaN if one class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return (fpr, tpr) points suitable for plotting an ROC curve.

    Points include the (0,0) origin and (1,1) corner; duplicate score thresholds
    are collapsed so each distinct threshold contributes one point.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = (labels[order] == 1).astype(float)
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    # Keep the last index of each run of equal scores (threshold boundaries).
    keep = np.r_[np.where(np.diff(s) != 0)[0], s.size - 1]
    tpr = np.r_[0.0, tps[keep] / n_pos]
    fpr = np.r_[0.0, fps[keep] / n_neg]
    return fpr, tpr


def expected_calibration_error(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 10
) -> float:
    """Expected Calibration Error with equal-width bins.

    ``probs`` are predicted probabilities of the positive class; ``labels`` are {0,1}.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = probs.size
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        confidence = float(probs[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


def reliability_curve(
    probs: ArrayLike, labels: ArrayLike, n_bins: int = 10
) -> dict[str, list]:
    """Per-bin (confidence, accuracy, count) for a reliability diagram (equal-width bins).

    Pairs with :func:`expected_calibration_error`. Empty bins get accuracy ``nan`` and a
    confidence at the bin centre so a renderer can skip them.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confidence: list[float] = []
    accuracy: list[float] = []
    count: list[int] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        c = int(mask.sum())
        count.append(c)
        confidence.append(float(probs[mask].mean()) if c else float((lo + hi) / 2))
        accuracy.append(float(labels[mask].mean()) if c else float("nan"))
    return {"bin_edges": edges.tolist(), "confidence": confidence, "accuracy": accuracy, "count": count}


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    """Pearson product-moment correlation. NaN if undefined (n<2 or zero variance)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt(float((xc * xc).sum()) * float((yc * yc).sum()))
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum() / denom)


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman rank correlation = Pearson on average ranks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return float("nan")
    return pearson(_rankdata(x), _rankdata(y))


def spearman_matrix(
    columns: Mapping[str, ArrayLike],
) -> tuple[list[str], np.ndarray]:
    """Spearman correlation matrix over named equal-length columns.

    Returns (ordered names, matrix) where ``matrix[i, j]`` is the Spearman
    correlation between columns i and j. The diagonal is 1.0.
    """
    names = list(columns.keys())
    k = len(names)
    mat = np.eye(k, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            r = spearman(columns[names[i]], columns[names[j]])
            mat[i, j] = mat[j, i] = r
    return names, mat


def bootstrap_mean_ci(
    values: ArrayLike,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, n, n)].mean() for _ in range(n_resamples)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def bootstrap_auc_ci(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for AUROC (resampling instances with replacement)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    n = scores.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        a = roc_auc(scores[idx], labels[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return (lo, hi)


def auc_summary(
    scores: ArrayLike, labels: ArrayLike, *, seed: int = 0
) -> dict[str, float]:
    """Convenience: AUROC plus a bootstrap 95% CI as a dict."""
    auc = roc_auc(scores, labels)
    lo, hi = bootstrap_auc_ci(scores, labels, seed=seed)
    return {"auc": auc, "ci_lo": lo, "ci_hi": hi}


def catch_rate_at_fpr(
    neg_scores: ArrayLike, pos_scores: ArrayLike, *, target_fpr: float = 0.01
) -> dict[str, float]:
    """Monitor-grade operating point: calibrate a threshold on the known-faithful
    negatives so the false-positive rate is ~``target_fpr``, then report the catch rate
    (fraction of unfaithful positives flagged) at that threshold. This is the
    "safety @ 1% FPR" framing of CoT-monitoring evaluations.
    """
    neg = np.sort(np.asarray(neg_scores, dtype=float))  # ascending
    pos = np.asarray(pos_scores, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return {"threshold": float("nan"), "fpr": float("nan"), "catch_rate": float("nan")}
    n = neg.size
    k = int(np.floor(target_fpr * n))  # at most this many false positives allowed
    tau = float(neg[n - 1 - k])  # the (k+1)-th largest negative -> flagging > tau keeps FPR <= target
    return {
        "threshold": tau,
        "fpr": float(np.mean(neg > tau)),  # conservative: <= target_fpr by construction
        "catch_rate": float(np.mean(pos > tau)),
    }


def permutation_auroc(
    scores: ArrayLike, labels: ArrayLike, *, seed: int = 0, n_perm: int = 200
) -> float:
    """Mean AUROC over ``n_perm`` random label permutations — a falsifiability baseline.

    Shuffling the labels destroys any score↔label structure, so even a perfectly
    separating score collapses to ~0.5. A probe whose *real* targeted AUROC is high but
    whose permuted AUROC is ~0.5 is measuring genuine structure, not a metric artifact;
    it proves the harness is *able* to report a non-success.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    aucs = [roc_auc(scores, rng.permutation(labels)) for _ in range(n_perm)]
    return float(np.mean(aucs))
