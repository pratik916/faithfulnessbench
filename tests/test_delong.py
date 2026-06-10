"""DeLong + paired-permutation significance for AUROC differences (fb-7gr.1)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.validation import run_validation


def test_report_significance_is_on_the_noisy_substrate():
    v = run_validation(n_per_domain=10, seed=0)["validation"]
    ns = v["noisy_significance"]
    assert ns["substrate"].startswith("label_noise=")
    assert ns["best_single_probe"] in v["probes"]
    for k in ("gap", "p", "perm_p", "diff_ci_lo", "diff_ci_hi"):
        assert k in ns


def test_delong_aucs_match_roc_auc():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50), np.ones(50)]
    a = np.r_[rng.normal(0, 1, 50), rng.normal(1.0, 1, 50)]
    b = np.r_[rng.normal(0, 1, 50), rng.normal(0.3, 1, 50)]
    d = M.delong_test(a, b, y, n_perm=200, n_boot=200)
    assert d["auc_a"] == M.roc_auc(a, y)
    assert d["auc_b"] == M.roc_auc(b, y)
    assert abs(d["gap"] - (M.roc_auc(a, y) - M.roc_auc(b, y))) < 1e-12


def test_delong_identical_classifiers_have_no_difference():
    y = np.r_[np.zeros(30), np.ones(30)]
    x = np.r_[np.zeros(30), np.ones(30)]
    d = M.delong_test(x, x, y, n_perm=200, n_boot=200)
    assert d["gap"] == 0.0
    assert d["p"] == 1.0
    assert d["perm_p"] == 1.0
    assert d["diff_ci_lo"] == 0.0 and d["diff_ci_hi"] == 0.0


def test_delong_flags_a_clearly_better_classifier():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(100), np.ones(100)]
    strong = np.r_[rng.normal(0, 1, 100), rng.normal(3.0, 1, 100)]
    weak = np.r_[rng.normal(0, 1, 100), rng.normal(0.2, 1, 100)]
    d = M.delong_test(strong, weak, y, seed=0, n_perm=500, n_boot=500)
    assert d["gap"] > 0.2
    assert d["p"] < 0.05
    assert d["perm_p"] < 0.05
    assert d["diff_ci_lo"] > 0.0  # the difference CI excludes zero
