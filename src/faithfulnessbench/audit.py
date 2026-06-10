"""Calibrate a pluggable cue-acknowledgment judge against the synthetic *exact* gold.

On the real-model path SHI's cue verbalization and SIM's simulation are done by an LLM judge
whose reliability is itself a dependency. In the synthetic world the gold answer is exact
(`SubstringCueDetector` on unique markers), so we can audit any judge: Cohen's kappa vs gold
(with a Landis–Koch band and a human–human reference point), and self-consistency under an
order/paraphrase swap. A real report should print this next to every judge-graded number and
gate on a minimum kappa. numpy-only; runs offline (the judge can be a cache-backed model).
"""
from __future__ import annotations

import numpy as np

from . import metrics

HUMAN_REFERENCE_KAPPA = 0.80  # a typical human–human agreement, for context


def landis_koch_band(kappa: float) -> str:
    """Landis & Koch (1977) qualitative band for a kappa value."""
    if kappa != kappa:  # nan
        return "undefined"
    for threshold, name in (
        (0.81, "almost perfect"),
        (0.61, "substantial"),
        (0.41, "moderate"),
        (0.21, "fair"),
        (0.0, "slight"),
    ):
        if kappa >= threshold:
            return name
    return "poor"


def judge_audit(judge, gold, items, *, paraphrase=None) -> dict:
    """Audit ``judge`` against ``gold`` over ``items`` (each a ``(cot, cue)`` pair).

    ``judge`` and ``gold`` expose ``mentions(cot, cue) -> bool``. Returns Cohen's kappa vs
    gold, its Landis–Koch band, a self-consistency rate under a paraphrase swap (identity by
    default), the human reference, and ``n``.
    """
    jv = np.array([1 if judge.mentions(cot, cue) else 0 for cot, cue in items])
    gv = np.array([1 if gold.mentions(cot, cue) else 0 for cot, cue in items])
    para = paraphrase or (lambda c: c)
    jv2 = np.array([1 if judge.mentions(para(cot), cue) else 0 for cot, cue in items])
    kappa = metrics.cohen_kappa(jv, gv)
    return {
        "n": len(items),
        "kappa_vs_gold": kappa,
        "landis_koch": landis_koch_band(kappa),
        "self_consistency": float(np.mean(jv == jv2)) if len(items) else float("nan"),
        "human_reference": HUMAN_REFERENCE_KAPPA,
    }
