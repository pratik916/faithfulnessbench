"""Model-agnostic card -> HTML report, used by the real-model score path (fb-0nc.6)."""
from __future__ import annotations

import math

from faithfulnessbench.report import render_card_report

_CARD = {
    "model_name": "claude-sonnet-4-6",
    "n_problems": 4,
    "composite_faithfulness": 0.667,
    "probe_scores": {
        "SHI": {"faithfulness": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "unfaithfulness": 0.0},
        "CSC": {"faithfulness": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "unfaithfulness": math.nan},
        "SIM": {"faithfulness": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "unfaithfulness": 1.0},
        "EAR": {"faithfulness": 1.0, "ci_lo": 1.0, "ci_hi": 1.0, "unfaithfulness": 0.0},
    },
    "per_domain": {},
    "extras": {"SIM": {"parse_failure_rate": 1.0}},
}


def test_render_card_report_from_cards_only():
    html = render_card_report(_CARD, title="t")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "claude-sonnet-4-6" in html
    assert "n/a" in html  # the nan CSC cell renders gracefully


def test_card_report_escapes_script_in_embedded_cot():
    traces = [{"cot": "</script><script>alert(1)</script>"}]
    html = render_card_report(_CARD, traces=traces)
    assert "</script><script>alert" not in html  # the raw injection is neutralised
    assert "<\\/script>" in html
