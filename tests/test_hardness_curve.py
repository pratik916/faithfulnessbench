"""AUROC-vs-hardness: a sensitivity curve over *intrinsic task difficulty*, not relabeling.

Where the label-noise curve degrades every probe mechanically (flipping y% of labels caps
any classifier), the hard-instance substrate degrades only the probes with a genuine
behavioral blind spot. SHI (cue points at the correct answer) and EAR (answer obvious from
the premise) fall toward chance; CSC and SIM stay at ceiling because their interventions
remain discriminating. That asymmetry is the measurement.
"""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import model_population
from faithfulnessbench.probes import default_probes
from faithfulnessbench.problems import hard_mixed_problems
from faithfulnessbench.validation import AXIS_MODEL, PROBE_ORDER, run_validation

PROBES = {p.name: p for p in default_probes()}


def _targeted(hardness: float, probe: str):
    pop = {m.name: m for m in model_population(seed=0)}
    probs = hard_mixed_problems(15, seed=0, hardness=hardness)
    fs = PROBES[probe].run(pop["faithful"], probs).scores
    axis = PROBES[probe].run(pop[AXIS_MODEL[probe]], probs).scores
    return np.r_[fs, axis], np.r_[np.zeros(fs.size), np.ones(axis.size)]


def test_zero_hardness_keeps_wiring_perfect():
    for p in PROBE_ORDER:
        s, y = _targeted(0.0, p)
        assert M.roc_auc(s, y) == 1.0, p


def test_blind_spot_probes_degrade_but_robust_probes_hold():
    auc = {p: M.roc_auc(*_targeted(1.0, p)) for p in PROBE_ORDER}
    # SHI and EAR have genuine blind spots on these instances.
    assert auc["SHI"] < 0.6 and auc["EAR"] < 0.6, auc
    # CSC and SIM stay at ceiling — their interventions remain discriminating.
    assert auc["CSC"] == 1.0 and auc["SIM"] == 1.0, auc
    # The asymmetry is the point: robust probes strictly dominate the blind-spot ones.
    assert min(auc["CSC"], auc["SIM"]) > max(auc["SHI"], auc["EAR"]), auc


def test_intermediate_hardness_is_a_nondegenerate_measurement():
    for p in ("SHI", "EAR"):
        s, y = _targeted(0.5, p)
        auc = M.roc_auc(s, y)
        assert 0.5 < auc < 1.0, (p, auc)  # a genuine measurement, above chance, below ceiling
        lo, hi = M.bootstrap_auc_ci(s, y, seed=0)
        assert hi - lo > 0.0, (p, lo, hi)  # non-degenerate confidence interval


def test_report_carries_auroc_vs_hardness_curve_per_probe():
    rep = run_validation(n_per_domain=8, seed=0)
    val = rep["validation"]
    assert val["hardness_levels"][0] == 0.0
    curve = val["auroc_vs_hardness"]
    assert set(curve) == set(PROBE_ORDER)
    for p in PROBE_ORDER:
        pts = curve[p]
        assert pts[0]["hardness"] == 0.0 and pts[0]["auc"] == 1.0  # wiring intact at zero hardness
        aucs = [d["auc"] for d in pts]
        assert aucs == sorted(aucs, reverse=True)  # monotone non-increasing
    # Robust vs blind-spot split holds end to end.
    assert curve["CSC"][-1]["auc"] == 1.0 and curve["SIM"][-1]["auc"] == 1.0
    assert curve["SHI"][-1]["auc"] < 1.0 and curve["EAR"][-1]["auc"] < 1.0
