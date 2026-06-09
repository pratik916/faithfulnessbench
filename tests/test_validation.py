"""End-to-end validation engine, report rendering, and figure export."""
from __future__ import annotations

import json

from faithfulnessbench.report import build_figures, render_report, write_figures
from faithfulnessbench.validation import PROBE_ORDER, json_safe, run_validation, summarize

REPORT = run_validation(n_per_domain=6, seed=0, n_trials=2)


def test_targeted_detection_is_perfect():
    val = REPORT["validation"]
    for p in PROBE_ORDER:
        assert val["targeted_auroc"][p]["auc"] == 1.0, p


def test_specificity_is_diagonal():
    spec = REPORT["validation"]["specificity"]
    for p in PROBE_ORDER:
        assert spec[p][p] == 1.0  # catches its own axis
        for other in PROBE_ORDER:
            if other != p:
                assert spec[p][other] <= 0.6  # ~chance on other axes


def test_combined_detector_beats_best_single():
    val = REPORT["validation"]
    combined = val["combined_auroc"]["auc"]
    best_single = max(val["single_mixed_auroc"].values())
    assert combined == 1.0
    assert best_single < combined  # the card adds real signal over any one probe


def test_disagreement_and_correlation_present():
    corr = REPORT["correlation"]
    assert corr["labels"] == PROBE_ORDER
    assert all(corr["matrix"][i][i] == 1.0 for i in range(len(PROBE_ORDER)))
    assert "card, not a scalar" in REPORT["disagreement"]


def test_trace_examples_include_a_silent_flip():
    examples = REPORT["trace_examples"]
    assert len(examples) == 5
    flips = [
        e
        for e in examples
        if e["cued"]["answer"] != e["baseline"]["answer"] and not e["cued"]["acknowledged"]
    ]
    assert flips, "expected at least one silent (unfaithful) flip example"


def test_report_is_json_serializable_and_renders():
    json.dumps(json_safe(REPORT))  # results.json must be valid JSON
    html = render_report(REPORT)
    assert "<svg" in html and "const TRACES" in html and "</html>" in html
    assert isinstance(summarize(REPORT), str)


def test_figures_export(tmp_path):
    figs = build_figures(REPORT)
    assert set(figs) == {"detection_auroc", "roc", "specificity", "correlation", "faithfulness_matrix", "auroc_vs_noise"}
    paths = write_figures(REPORT, str(tmp_path / "assets"))
    assert len(paths) == 6
    for p in paths:
        assert p.endswith(".svg")
