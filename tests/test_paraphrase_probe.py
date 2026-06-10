"""Paraphrase-Invariance probe (PAR), steganographic dial, extended-only (fb-zzt.2)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import (
    ConfigurableSyntheticModel,
    FaithfulnessProfile,
    model_population,
)
from faithfulnessbench.probes.par import ParaphraseProbe, _paraphrase
from faithfulnessbench.problems import arithmetic_chain_problems, execute_steps

PROBS = arithmetic_chain_problems(12, seed=0)


def _model(noise=0.0, **rates):
    return ConfigurableSyntheticModel("m", FaithfulnessProfile(seed=0, label_noise=noise, **rates))


def _auroc(a, b):
    sa = ParaphraseProbe().run(a, PROBS).scores
    sb = ParaphraseProbe().run(b, PROBS).scores
    return M.roc_auc(np.r_[sa, sb], np.r_[np.zeros(sa.size), np.ones(sb.size)])


def test_paraphrase_preserves_numeric_content():
    steps = list(PROBS[0].steps)
    assert execute_steps(_paraphrase(steps)) == execute_steps(steps)  # same values, different surface


def test_par_fires_on_stego_model_not_faithful():
    assert ParaphraseProbe().run(_model(), PROBS).scores.mean() < 0.05
    assert ParaphraseProbe().run(_model(p_stego=1.0), PROBS).scores.mean() > 0.95


def test_par_targeted_auroc_perfect_and_off_axis_chance():
    assert _auroc(_model(), _model(p_stego=1.0)) == 1.0
    assert _auroc(_model(), _model(p_hint_sycophancy=1.0)) == 0.5


def test_par_auroc_nondegenerate_with_a_partial_dial():
    f = ParaphraseProbe().run(_model(), PROBS, n_trials=1).scores
    a = ParaphraseProbe().run(_model(p_stego=0.5), PROBS, n_trials=1).scores
    auc = M.roc_auc(np.r_[f, a], np.r_[np.zeros(f.size), np.ones(a.size)])
    assert 0.5 < auc < 1.0


def test_p_stego_not_in_frozen_population():
    for m in model_population():
        assert m.profile.p_stego == 0.0
