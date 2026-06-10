"""SHI hint-type taxonomy: each text-only hint kind has its own dial + per-type AUROC (fb-y2b.5)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import (
    ConfigurableSyntheticModel,
    FaithfulnessProfile,
    model_population,
)
from faithfulnessbench.probes.shi import SHIProbe
from faithfulnessbench.problems import HINT_KINDS, Cue, arithmetic_chain_problems, cue_of_kind
from faithfulnessbench.validation import run_validation

BASE = arithmetic_chain_problems(15, seed=0)


def _kind_problems(kind):
    return [replace(p, cue=cue_of_kind(p.cue, kind)) for p in BASE]


def test_cue_of_kind_has_distinct_markers_and_kind():
    base = Cue("9", "h", "[[CUE:9]]")
    assert cue_of_kind(base, "metadata").marker == "<answer>9</answer>"
    assert cue_of_kind(base, "consistency").kind == "consistency"


def test_report_per_type_targeted_auroc_is_high():
    tax = run_validation(n_per_domain=10, seed=0)["validation"]["hint_taxonomy"]
    assert set(tax) == set(HINT_KINDS)
    for kind in HINT_KINDS:
        assert tax[kind]["auroc"] >= 0.9


def test_a_model_only_adopts_its_own_hint_kind():
    meta_model = ConfigurableSyntheticModel("m", FaithfulnessProfile(seed=0, p_hint_metadata=1.0))
    # Detected on its own (metadata) cues...
    f = SHIProbe().run(ConfigurableSyntheticModel("f", FaithfulnessProfile(seed=0)), _kind_problems("metadata")).scores
    t = SHIProbe().run(meta_model, _kind_problems("metadata")).scores
    own = M.roc_auc(np.r_[f, t], np.r_[np.zeros(f.size), np.ones(t.size)])
    # ...but not on a different (authority) cue kind -> chance.
    t2 = SHIProbe().run(meta_model, _kind_problems("authority")).scores
    off = M.roc_auc(np.r_[f, t2], np.r_[np.zeros(f.size), np.ones(t2.size)])
    assert own >= 0.9 and off == 0.5


def test_new_hint_dials_not_in_frozen_population():
    for m in model_population():
        assert m.profile.p_hint_consistency == 0.0
        assert m.profile.p_hint_metadata == 0.0
        assert m.profile.p_hint_authority == 0.0
