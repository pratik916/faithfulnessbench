"""Cluster (block) bootstrap CI + per-domain detection AUROC (fb-7gr.2)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.validation import run_validation


def test_cluster_ci_reduces_to_flat_for_singleton_clusters():
    vals = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
    ids = ["a", "b", "c", "d", "e"]  # each its own cluster
    flat = M.bootstrap_mean_ci(vals, seed=0)
    clus = M.bootstrap_cluster_ci(vals, ids, seed=0)
    assert np.allclose(flat, clus)


def test_cluster_ci_is_at_least_as_wide_on_nested_data():
    # 4 clusters, each a block of 10 identical values -> heavy within-cluster correlation.
    vals = np.repeat([0.0, 0.0, 1.0, 1.0], 10).astype(float)
    ids = [c for c in ("a", "b", "c", "d") for _ in range(10)]
    flat = M.bootstrap_mean_ci(vals, seed=0)
    clus = M.bootstrap_cluster_ci(vals, ids, seed=0)
    assert (clus[1] - clus[0]) >= (flat[1] - flat[0])


def test_report_has_per_domain_detection_auroc():
    v = run_validation(n_per_domain=10, seed=0)["validation"]
    pda = v["per_domain_auroc"]
    assert set(pda) == {"SHI", "CSC", "SIM", "EAR"}
    for probe, by_dom in pda.items():
        assert set(by_dom) == {"arithmetic", "mcq"}, (probe, by_dom)
        for auc in by_dom.values():
            assert auc == 1.0  # the clean signal separates within each domain too
