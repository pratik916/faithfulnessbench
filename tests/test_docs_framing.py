"""Docs carry the monitorability framing + honest foils (fb-y2b.1, fb-yst.5)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_monitorability_framing():
    t = (ROOT / "README.md").read_text()
    low = t.lower()
    assert "monitorab" in low  # the CoT-monitorability agenda
    assert "2507.11473" in t  # Korbak/Balesni et al. position paper
    assert "coverage" in low  # the faithfulness x coverage decomposition
    assert "white-box" in low or "white box" in low  # behavioral-vs-white-box scope
    assert "2510.04040" in t  # FaithCoT-Bench ~0.70 behavioral foil


def test_design_references_monitorability():
    low = (ROOT / "docs" / "DESIGN.md").read_text().lower()
    assert "monitorab" in low


def test_readme_positions_against_related_benchmarks():
    t = (ROOT / "README.md").read_text()
    assert "Related benchmarks" in t
    assert "RFEval" in t and "FaithCoT-Bench" in t
    assert "knowledge" in t.lower()  # the math-vs-knowledge scope caveat


def test_headline_reports_f1_and_kappa_beside_auroc():
    from faithfulnessbench.validation import run_validation

    hc = run_validation(n_per_domain=8, seed=0)["validation"]["headline_classification"]
    assert set(hc) == {"auroc", "f1", "kappa"}
    assert hc["f1"] == 1.0 and hc["kappa"] == 1.0  # by construction on the clean population


def test_design_has_third_domain_spike():
    low = (ROOT / "docs" / "DESIGN.md").read_text().lower()
    assert "third" in low and "domain" in low
    assert "corruptor" in low and "cotsimulator" in low  # the protocols a 3rd domain needs
    assert "recommendation: defer" in low  # the go/no-go
