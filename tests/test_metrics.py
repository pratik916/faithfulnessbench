"""Hand-computed checks for the from-scratch metrics."""
from __future__ import annotations

import math

import numpy as np

from faithfulnessbench import metrics as M


def test_rankdata_handles_ties():
    np.testing.assert_allclose(M._rankdata([10, 10, 20]), [1.5, 1.5, 3.0])
    np.testing.assert_allclose(M._rankdata([3, 1, 2]), [3.0, 1.0, 2.0])


def test_auc_known_value():
    # 2 pos (0.35, 0.8), 2 neg (0.1, 0.4): 3 of 4 cross-pairs ordered correctly.
    auc = M.roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
    assert math.isclose(auc, 0.75, rel_tol=1e-9)


def test_auc_perfect_and_reversed():
    assert M.roc_auc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0
    assert M.roc_auc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0


def test_auc_ties_give_half():
    assert M.roc_auc([1.0, 1.0], [0, 1]) == 0.5


def test_auc_single_class_is_nan():
    assert math.isnan(M.roc_auc([0.2, 0.9], [1, 1]))
    assert math.isnan(M.roc_auc([0.2, 0.9], [0, 0]))


def test_roc_curve_endpoints_and_auc_consistency():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=400)
    labels = (rng.normal(loc=scores * 0.8) > 0).astype(int)
    fpr, tpr = M.roc_curve(scores, labels)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert math.isclose(fpr[-1], 1.0, abs_tol=1e-9)
    assert math.isclose(tpr[-1], 1.0, abs_tol=1e-9)
    # Trapezoidal area under the ROC curve must match the rank-based AUROC.
    area = float(np.trapezoid(tpr, fpr))
    assert math.isclose(area, M.roc_auc(scores, labels), abs_tol=1e-9)


def test_pearson_and_spearman():
    assert math.isclose(M.pearson([1, 2, 3], [2, 4, 6]), 1.0, abs_tol=1e-12)
    assert math.isclose(M.pearson([1, 2, 3], [6, 4, 2]), -1.0, abs_tol=1e-12)
    # Monotonic but non-linear: Spearman is exactly 1, Pearson is strictly less.
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    assert math.isclose(M.spearman(x, y), 1.0, abs_tol=1e-12)
    assert M.pearson(x, y) < 1.0


def test_correlation_undefined_cases():
    assert math.isnan(M.pearson([1.0], [1.0]))          # n < 2
    assert math.isnan(M.pearson([1, 1, 1], [1, 2, 3]))  # zero variance


def test_ece_extremes():
    # Perfectly confident and correct -> 0 error.
    assert math.isclose(M.expected_calibration_error([1.0, 1.0], [1, 1]), 0.0)
    # Perfectly confident and wrong -> maximal error.
    assert math.isclose(M.expected_calibration_error([1.0, 1.0], [0, 0]), 1.0)
    # Calibrated at 0.5.
    assert math.isclose(M.expected_calibration_error([0.5, 0.5], [0, 1]), 0.0)


def test_spearman_matrix_shape_and_symmetry():
    cols = {
        "a": [1, 2, 3, 4],
        "b": [4, 3, 2, 1],
        "c": [1, 3, 2, 4],
    }
    names, mat = M.spearman_matrix(cols)
    assert names == ["a", "b", "c"]
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat), [1, 1, 1])
    np.testing.assert_allclose(mat, mat.T)
    assert math.isclose(mat[0, 1], -1.0, abs_tol=1e-12)  # a vs b perfectly anti-ranked


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=300)
    labels = (rng.normal(loc=scores) > 0).astype(int)
    auc = M.roc_auc(scores, labels)
    lo, hi = M.bootstrap_auc_ci(scores, labels, n_resamples=500, seed=7)
    assert lo <= auc <= hi
    # Deterministic given the seed.
    lo2, hi2 = M.bootstrap_auc_ci(scores, labels, n_resamples=500, seed=7)
    assert (lo, hi) == (lo2, hi2)


def test_bootstrap_mean_ci_brackets_mean():
    v = np.linspace(0, 1, 200)
    lo, hi = M.bootstrap_mean_ci(v, n_resamples=500, seed=3)
    assert lo <= v.mean() <= hi
