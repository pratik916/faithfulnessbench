"""SIM disambiguates malformed (unparseable) CoT from genuine simulatability misses (fb-0nc.4)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.base import Model, Trace
from faithfulnessbench.models.synthetic import model_population
from faithfulnessbench.probes.sim import SIMProbe
from faithfulnessbench.problems import mixed_problems

PROBLEMS = mixed_problems(8, seed=0)
POP = {m.name: m for m in model_population()}


def test_synthetic_sim_has_zero_parse_failures():
    res = SIMProbe().run(POP["faithful"], PROBLEMS)
    assert res.extra["parse_failure_rate"] == 0.0  # synthetic chains always parse


def test_synthetic_sim_auroc_unchanged_by_the_flag():
    f = SIMProbe().run(POP["faithful"], PROBLEMS).scores
    d = SIMProbe().run(POP["decoy_cot"], PROBLEMS).scores
    assert M.roc_auc(np.r_[f, d], np.r_[np.zeros(f.size), np.ones(d.size)]) == 1.0


class _FreeText(Model):
    """Emits free-text CoT with no parseable 'L op R = V' conclusion."""

    name = "freetext"

    def reason(self, problem, *, cue=None, trial=0):
        return Trace(answer=problem.answer, cot="I think about it", steps=["I think about it"])

    def continue_from_cot(self, problem, cot_steps, *, trial=0):
        return problem.answer

    def answer_from_prefix(self, problem, prefix_steps, *, trial=0):
        return problem.answer


def test_free_text_cot_is_flagged_as_parse_failure_not_genuine_miss():
    res = SIMProbe().run(_FreeText(), PROBLEMS)
    assert res.extra["parse_failure_rate"] == 1.0
    assert res.extra["genuine_miss_rate"] == 0.0  # all misses are parse failures, not non-predictive
