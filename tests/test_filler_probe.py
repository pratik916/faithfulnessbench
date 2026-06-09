"""Filler-Token probe (FIL): detects content-not-load-bearing CoT, extended-only (fb-zzt.1)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import (
    ConfigurableSyntheticModel,
    FaithfulnessProfile,
    model_population,
)
from faithfulnessbench.probes.fil import FillerProbe
from faithfulnessbench.problems import mixed_problems

PROBLEMS = mixed_problems(12, seed=0)


def _model(noise=0.0, **rates):
    return ConfigurableSyntheticModel("m", FaithfulnessProfile(seed=0, label_noise=noise, **rates))


def _auroc(model_a, model_b):
    a = FillerProbe().run(model_a, PROBLEMS).scores
    b = FillerProbe().run(model_b, PROBLEMS).scores
    return M.roc_auc(np.r_[a, b], np.r_[np.zeros(a.size), np.ones(b.size)])


def test_fil_fires_on_filler_model_not_faithful():
    assert FillerProbe().run(_model(), PROBLEMS).scores.mean() < 0.05
    assert FillerProbe().run(_model(p_filler=1.0), PROBLEMS).scores.mean() > 0.95


def test_fil_targeted_auroc_perfect_at_zero_noise():
    assert _auroc(_model(), _model(p_filler=1.0)) == 1.0


def test_fil_is_off_axis_chance_for_a_different_dial():
    # A sycophant (different axis) is faithful w.r.t. FIL -> exactly the tie-convention 0.5.
    assert _auroc(_model(), _model(p_hint_sycophancy=1.0)) == 0.5


def test_fil_auroc_is_nondegenerate_under_noise():
    auc = _auroc(_model(noise=0.4), _model(noise=0.4, p_filler=1.0))
    assert 0.5 < auc < 1.0


def test_p_filler_is_not_in_the_frozen_population():
    for m in model_population():
        assert m.profile.p_filler == 0.0
