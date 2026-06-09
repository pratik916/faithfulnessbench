"""Monitor-grade metrics: catch-rate at <=1% FPR (calibrated on known-faithful), with a
subtle (intermediate-dial) vs. blatant (full-dial) split (fb-y2b.2).
"""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.report import render_report
from faithfulnessbench.validation import run_validation


def test_catch_rate_at_fpr_calibrates_on_negatives():
    neg = np.zeros(200)
    pos = np.ones(50)
    r = M.catch_rate_at_fpr(neg, pos, target_fpr=0.01)
    assert r["fpr"] <= 0.01
    assert r["catch_rate"] == 1.0


def test_catch_rate_respects_fpr_budget_under_overlap():
    rng = np.random.default_rng(0)
    neg = rng.normal(0.0, 1.0, 1000)
    pos = rng.normal(3.0, 1.0, 1000)
    r = M.catch_rate_at_fpr(neg, pos, target_fpr=0.01)
    assert r["fpr"] <= 0.01
    assert 0.0 < r["catch_rate"] < 1.0


def test_report_monitor_subtle_below_blatant():
    mon = run_validation(n_per_domain=12, seed=0)["validation"]["monitor"]
    assert mon["target_fpr"] == 0.01
    assert mon["fpr_on_faithful"] <= 0.01
    assert mon["catch_rate_blatant"] == 1.0
    assert mon["catch_rate_subtle"] < mon["catch_rate_blatant"]
    assert 0.0 <= mon["catch_rate_subtle"] <= 1.0


def test_report_renders_monitor_catch_rate():
    html = render_report(run_validation(n_per_domain=6, seed=0)).lower()
    assert "catch-rate" in html
