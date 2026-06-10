"""Chance-corrected agreement (Cohen's kappa) + Holm-Bonferroni correction (fb-7gr.3)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M


def test_cohen_kappa_hand_computed():
    assert M.cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0  # perfect agreement
    assert M.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0  # perfect disagreement
    assert abs(M.cohen_kappa([1, 0, 1, 0], [1, 0, 0, 1]) - 0.0) < 1e-12  # po == pe -> 0


def test_holm_correction_hand_computed():
    r = M.holm_correction([0.01, 0.04, 0.03])
    assert r["adjusted"] == [0.03, 0.06, 0.06]
    assert r["reject"] == [True, False, False]


def test_permutation_test_auroc_extremes():
    y = np.r_[np.zeros(50), np.ones(50)]
    perfect = np.r_[np.zeros(50), np.ones(50)]
    assert M.permutation_test_auroc(perfect, y, n_perm=500) < 0.01  # clear separation
    tied = np.full(100, 0.5)
    assert M.permutation_test_auroc(tied, y, n_perm=200) == 1.0  # AUROC 0.5, every perm ties


def test_report_has_agreement_kappa_and_holm_significance():
    from faithfulnessbench.validation import run_validation

    rep = run_validation(n_per_domain=10, seed=0)
    ak = rep["agreement_kappa"]
    assert ak["labels"] == ["SHI", "CSC", "SIM", "EAR"]
    for i in range(4):
        assert ak["matrix"][i][i] == 1.0  # a probe agrees perfectly with itself
    nps = rep["validation"]["noisy_per_probe_significance"]
    for p in ["SHI", "CSC", "SIM", "EAR"]:
        assert "holm_adjusted" in nps[p] and "reject" in nps[p]
