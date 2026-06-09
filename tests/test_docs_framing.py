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
