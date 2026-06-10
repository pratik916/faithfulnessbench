"""Judge calibration vs synthetic exact gold (fb-7gr.4)."""
from __future__ import annotations

from faithfulnessbench.audit import judge_audit, landis_koch_band
from faithfulnessbench.models.base import SubstringCueDetector
from faithfulnessbench.problems import Cue

CUE = Cue(target="9", text="a hint", marker="[[CUE:9]]")
# Half the CoTs acknowledge the cue (contain its marker), half don't.
ITEMS = [(f"work {CUE.marker} done", CUE) for _ in range(5)] + [("clean work", CUE) for _ in range(5)]


class _PerfectJudge:
    def mentions(self, cot, cue):
        return SubstringCueDetector().mentions(cot, cue)


class _NoisyJudge:
    def __init__(self):
        self.n = 0

    def mentions(self, cot, cue):
        self.n += 1
        base = SubstringCueDetector().mentions(cot, cue)
        return (not base) if self.n % 3 == 0 else base  # flip every third call


def test_landis_koch_bands():
    assert landis_koch_band(0.9) == "almost perfect"
    assert landis_koch_band(0.5) == "moderate"
    assert landis_koch_band(0.05) == "slight"
    assert landis_koch_band(float("nan")) == "undefined"


def test_perfect_judge_scores_kappa_one():
    a = judge_audit(_PerfectJudge(), SubstringCueDetector(), ITEMS)
    assert a["kappa_vs_gold"] == 1.0
    assert a["landis_koch"] == "almost perfect"
    assert a["self_consistency"] == 1.0
    assert a["human_reference"] == 0.80


def test_noisy_judge_scores_below_perfect():
    a = judge_audit(_NoisyJudge(), SubstringCueDetector(), ITEMS)
    assert a["kappa_vs_gold"] < 1.0


def test_card_report_shows_judge_reliability_line():
    from faithfulnessbench.report import render_card_report

    card = {
        "model_name": "m",
        "n_problems": 1,
        "composite_faithfulness": 1.0,
        "probe_scores": {"SHI": {"faithfulness": 1.0, "ci_lo": 1.0, "ci_hi": 1.0}},
        "per_domain": {},
        "extras": {},
        "judge_reliability": {
            "kappa_vs_gold": 0.7,
            "landis_koch": "substantial",
            "self_consistency": 0.95,
            "human_reference": 0.80,
        },
    }
    html = render_card_report(card)
    assert "Judge reliability" in html and "kappa=0.700" in html
