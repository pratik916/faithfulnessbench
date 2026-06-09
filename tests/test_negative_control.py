"""Falsifiability controls (fb-yst.2).

Two things must hold for the validation to be trustworthy rather than circular:
1. the harness must be *able* to report chance — shuffling the ground-truth labels
   collapses every probe's targeted AUROC to ~0.5; and
2. probes must stay black-box — reading only the Model interface (reason /
   continue_from_cot / answer_from_prefix) and Trace text, never synthetic internals.
"""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import model_population
from faithfulnessbench.probes import default_probes
from faithfulnessbench.problems import mixed_problems
from faithfulnessbench.validation import run_validation

PROBLEMS = mixed_problems(8, seed=0)
POP = {m.name: m for m in model_population()}


class _InterfaceOnly:
    """Wraps a model exposing ONLY the Model ABC surface; any other attribute raises."""

    _ALLOWED = {"reason", "continue_from_cot", "answer_from_prefix", "name"}

    def __init__(self, model):
        self._model = model

    def __getattr__(self, key):
        if key in _InterfaceOnly._ALLOWED:
            return getattr(self._model, key)
        raise AttributeError(f"probe accessed non-interface attribute {key!r}")


def test_permutation_auroc_collapses_perfect_separation_to_chance():
    scores = np.r_[np.zeros(20), np.ones(20)]
    labels = np.r_[np.zeros(20), np.ones(20)]
    assert M.roc_auc(scores, labels) == 1.0  # perfectly separates
    assert 0.4 <= M.permutation_auroc(scores, labels, seed=0) <= 0.6  # ...until labels shuffle


def test_validation_reports_near_chance_negative_control_per_probe():
    rep = run_validation(n_per_domain=8, seed=0)
    nc = rep["validation"]["negative_control_auroc"]
    assert set(nc) == set(rep["validation"]["probes"])
    for probe, auc in nc.items():
        assert 0.4 <= auc <= 0.6, (probe, auc)


def test_probes_only_touch_the_model_interface():
    guarded = _InterfaceOnly(POP["pre_commit"])
    for probe in default_probes():
        probe.run(guarded, PROBLEMS, n_trials=2)  # AttributeError if a probe peeks inside
