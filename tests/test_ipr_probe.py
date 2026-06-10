"""Implicit Post-Hoc Rationalization probe (IPR), no-hint, extended-only (fb-zzt.3)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import (
    ConfigurableSyntheticModel,
    FaithfulnessProfile,
    model_population,
)
from faithfulnessbench.probes.ipr import IPRProbe
from faithfulnessbench.problems import contradictory_pairs

PAIRS = contradictory_pairs(12, seed=0)


def _model(noise=0.0, **rates):
    return ConfigurableSyntheticModel("m", FaithfulnessProfile(seed=0, label_noise=noise, **rates))


def _auroc(a, b):
    sa = IPRProbe().run(a, PAIRS).scores
    sb = IPRProbe().run(b, PAIRS).scores
    return M.roc_auc(np.r_[sa, sb], np.r_[np.zeros(sa.size), np.ones(sb.size)])


def test_contradictory_pairs_have_exactly_one_yes():
    by_pair: dict = {}
    for p in PAIRS:
        by_pair.setdefault(p.meta["pair_id"], []).append(p.answer)
    assert by_pair  # non-empty
    for answers in by_pair.values():
        assert sorted(answers) == ["No", "Yes"]  # exactly one Yes per contradictory pair


def test_ipr_fires_on_rationalizer_not_faithful():
    assert IPRProbe().run(_model(), PAIRS).scores.mean() < 0.05
    assert IPRProbe().run(_model(p_contradict=1.0), PAIRS).scores.mean() > 0.95


def test_ipr_targeted_auroc_perfect_and_off_axis_chance():
    assert _auroc(_model(), _model(p_contradict=1.0)) == 1.0
    assert _auroc(_model(), _model(p_hint_sycophancy=1.0)) == 0.5  # different axis -> chance


def test_ipr_auroc_nondegenerate_with_a_partial_dial():
    assert 0.5 < _auroc(_model(), _model(p_contradict=0.5)) < 1.0


def test_p_contradict_not_in_frozen_population():
    for m in model_population():
        assert m.profile.p_contradict == 0.0
