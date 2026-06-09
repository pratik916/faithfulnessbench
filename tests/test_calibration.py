"""Calibration: ECE + a reliability diagram, wired into the report on the noised
substrate (clean 0/1-clustered scores are uninformative for ECE) (fb-y2b.3).
"""
from __future__ import annotations

from faithfulnessbench import metrics as M
from faithfulnessbench.report import render_report
from faithfulnessbench.validation import run_validation
from faithfulnessbench.viz import svg


def test_ece_matches_hand_computed_values():
    # Always predict 0.9 but always wrong -> ECE = |0 - 0.9| = 0.9.
    assert M.expected_calibration_error([0.9, 0.9, 0.9], [0, 0, 0]) == 0.9
    # Confident and correct at the extremes -> perfectly calibrated.
    assert M.expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0


def test_reliability_curve_shape_and_counts():
    r = M.reliability_curve([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=10)
    assert len(r["accuracy"]) == len(r["confidence"]) == len(r["count"]) == 10
    assert sum(r["count"]) == 4


def test_reliability_diagram_is_well_formed_svg():
    out = svg.reliability_diagram(M.reliability_curve([0.1, 0.9], [0, 1]), title="t")
    assert out.startswith("<svg") and out.rstrip().endswith("</svg>")


def test_report_carries_calibration_and_renders_ece():
    rep = run_validation(n_per_domain=8, seed=0)
    cal = rep["validation"]["calibration"]
    assert 0.0 <= cal["ece_combined"] <= 1.0
    assert set(cal["ece_per_probe"]) == {"SHI", "CSC", "SIM", "EAR"}
    assert "reliability" in cal
    html = render_report(rep).lower()
    assert "ece" in html and "reliability" in html
