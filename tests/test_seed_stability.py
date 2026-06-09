"""The headline metrics are structural, not seed-specific, and the off-axis AUROC is
exactly the tie-convention 0.50 — a structural artifact, not measured non-leakage (fb-bys.4).
"""
from __future__ import annotations

import pytest

from faithfulnessbench.report import render_report
from faithfulnessbench.validation import run_validation


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_headline_metrics_are_stable_across_seeds(seed):
    v = run_validation(n_per_domain=10, seed=seed)["validation"]
    for p in v["probes"]:
        assert v["targeted_auroc"][p]["auc"] == 1.0, (seed, p)
        assert abs(v["single_mixed_auroc"][p] - 0.7) < 1e-9, (seed, p)  # population identity
        for axis in v["probes"]:
            if axis != p:
                assert v["specificity"][p][axis] == 0.5, (seed, p, axis)  # exact tie convention
    assert v["combined_auroc"]["auc"] == 1.0


def test_report_labels_off_axis_as_structural():
    html = render_report(run_validation(n_per_domain=6, seed=0))
    assert "structural" in html.lower()
