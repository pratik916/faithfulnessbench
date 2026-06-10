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
    assert len(examples) == 7
    flips = [
        e
        for e in examples
        if e["cued"]["answer"] != e["baseline"]["answer"] and not e["cued"]["acknowledged"]
    ]
    assert flips, "expected at least one silent (unfaithful) flip example"


def test_trace_examples_carry_interactive_ear_and_csc_data():
    for t in REPORT["trace_examples"]:
        assert "ear" in t and len(t["ear"]) >= 2
        assert all({"fraction", "answer", "prefix", "matches_final"} <= set(s) for s in t["ear"])
        assert "csc" in t  # key always present (value may be None for a degenerate problem)
    # the pre_commit example must show early lock; the post_hoc example must fail to track
    pre = next(t for t in REPORT["trace_examples"] if "pre_commit" in t["id"] or t["ear"][0]["matches_final"])
    assert pre["ear"][0]["matches_final"]  # answer locked in from the empty prefix


def test_report_viewer_has_interactive_controls_and_is_json_safe():
    html = render_report(REPORT)
    assert "ear-frac" in html and "csc-toggle" in html  # the slider + toggle handlers
    assert "paintEar" in html and "paintCsc" in html
    json.dumps(json_safe(REPORT))  # the embedded trace data round-trips as valid JSON


def test_report_is_json_serializable_and_renders():
    json.dumps(json_safe(REPORT))  # results.json must be valid JSON
    html = render_report(REPORT)
    assert "<svg" in html and "const TRACES" in html and "</html>" in html
    assert isinstance(summarize(REPORT), str)


def test_figures_export(tmp_path):
    figs = build_figures(REPORT)
    assert set(figs) == {"detection_auroc", "roc", "specificity", "correlation", "faithfulness_matrix", "auroc_vs_noise", "reliability", "robustness", "whitebox"}
    paths = write_figures(REPORT, str(tmp_path / "assets"))
    assert len(paths) == 9
    for p in paths:
        assert p.endswith(".svg")


def test_whitebox_analogue_in_report():
    wb = REPORT["validation"]["whitebox"]
    # the contrast: white-box beats the evaded monitor on obfuscation, is blind on filler
    assert wb["obfuscation"]["white_box_auroc"] > wb["obfuscation"]["black_box_auroc"]
    assert wb["filler_blind_spot"]["black_box_fil_auroc"] > wb["filler_blind_spot"]["white_box_auroc"]


def test_robustness_frontier_in_report():
    rob = REPORT["validation"]["robustness"]
    assert set(rob["per_probe"]) == {"SHI", "CSC", "SIM", "EAR"}
    assert len(rob["combined"]) == len(rob["budgets"])
    # the headline finding: the text-reading SIM probe degrades under CoT obfuscation
    assert rob["per_probe"]["SIM"][-1] < rob["per_probe"]["SIM"][0]
