"""The committed HTML report and SVG figures must stay consistent with results.json.

CLAUDE.md says "regenerate the committed artifacts after any change that affects
numbers"; this enforces it. Rendering is deterministic, so a fresh render of the
committed results.json reproduces the committed report/figures byte-for-byte — any
difference means a stale artifact (fb-tch.4).
"""
from __future__ import annotations

import json
from pathlib import Path

from faithfulnessbench import artifacts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results" / "results.json"
REPORT = ROOT / "report" / "faithfulness_report.html"
FIGURES = ROOT / "docs" / "assets"


def test_committed_html_and_svgs_are_fresh():
    assert artifacts.stale_artifacts(RESULTS, REPORT, FIGURES) == []


def test_stale_artifacts_flags_missing_or_modified(tmp_path):
    msgs = artifacts.stale_artifacts(RESULTS, tmp_path / "none.html", tmp_path)
    assert msgs  # the report + every SVG are absent => stale
    assert any("none.html" in m for m in msgs)


def test_regenerating_artifacts_makes_the_check_pass(tmp_path):
    from faithfulnessbench.report import write_figures, write_report

    report = json.loads(RESULTS.read_text())
    write_report(report, str(tmp_path / "r.html"))
    write_figures(report, str(tmp_path))
    assert artifacts.stale_artifacts(RESULTS, tmp_path / "r.html", tmp_path) == []
