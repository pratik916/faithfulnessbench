"""Controlled label noise turns synthetic AUROC from a wiring check into a sensitivity
measurement: perfect on the clean signal, degraded but non-degenerate under noise (fb-yst.1).
"""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import model_population
from faithfulnessbench.probes import default_probes
from faithfulnessbench.problems import mixed_problems
from faithfulnessbench.validation import AXIS_MODEL, PROBE_ORDER, run_validation

PROBLEMS = mixed_problems(15, seed=0)
PROBES = {p.name: p for p in default_probes()}


def _targeted(noise: float, probe: str):
    pop = {m.name: m for m in model_population(seed=0, label_noise=noise)}
    fs = PROBES[probe].run(pop["faithful"], PROBLEMS).scores
    axis = PROBES[probe].run(pop[AXIS_MODEL[probe]], PROBLEMS).scores
    return np.r_[fs, axis], np.r_[np.zeros(fs.size), np.ones(axis.size)]


def test_zero_noise_keeps_wiring_perfect():
    for p in PROBE_ORDER:
        s, y = _targeted(0.0, p)
        assert M.roc_auc(s, y) == 1.0, p


def test_noise_makes_auroc_a_nondegenerate_measurement():
    for p in PROBE_ORDER:
        s, y = _targeted(0.4, p)
        auc = M.roc_auc(s, y)
        assert 0.5 < auc < 1.0, (p, auc)  # a genuine measurement, above chance
        lo, hi = M.bootstrap_auc_ci(s, y, seed=0)
        assert hi - lo > 0.0, (p, lo, hi)  # non-degenerate confidence interval


def test_report_carries_auroc_vs_noise_curve_per_probe():
    rep = run_validation(n_per_domain=8, seed=0)
    curve = rep["validation"]["auroc_vs_noise"]
    assert set(curve) == set(PROBE_ORDER)
    for p in PROBE_ORDER:
        pts = curve[p]
        assert pts[0]["noise"] == 0.0 and pts[0]["auc"] == 1.0  # wiring intact at zero noise
        assert pts[-1]["auc"] <= pts[0]["auc"]  # degrades (or holds) as noise rises
