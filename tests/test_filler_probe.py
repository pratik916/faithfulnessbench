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


def test_fil_auroc_is_nondegenerate_with_a_partial_dial():
    # A partial dial (fires ~half the time) makes the classes overlap -> a genuine
    # measurement, not a wiring identity. (Extended dials are not perturbed by the core
    # label-noise regime, so sensitivity is shown via the dial value itself.)
    f = FillerProbe().run(_model(), PROBLEMS, n_trials=1).scores
    a = FillerProbe().run(_model(p_filler=0.5), PROBLEMS, n_trials=1).scores
    auc = M.roc_auc(np.r_[f, a], np.r_[np.zeros(f.size), np.ones(a.size)])
    assert 0.5 < auc < 1.0


def test_p_filler_is_not_in_the_frozen_population():
    for m in model_population():
        assert m.profile.p_filler == 0.0
