"""The committed pooled baseline is frozen; held-out extended models get a pairwise
targeted AUROC without moving the pooled numbers or the correlation matrix (fb-yst.3).
"""
from __future__ import annotations

import json

from faithfulnessbench.models.synthetic import ConfigurableSyntheticModel, FaithfulnessProfile
from faithfulnessbench.validation import json_safe, run_validation


def _norm(x):  # NaN-safe structural compare (the correlation matrix may carry NaN)
    return json.dumps(json_safe(x), sort_keys=True)


def test_no_extended_models_leaves_the_report_unchanged():
    base = run_validation(n_per_domain=8, seed=0)
    # The committed artifact carries no extended section, so it is unaffected.
    assert "extended_auroc" not in base["validation"]


def test_extended_model_does_not_move_the_frozen_pooled_numbers():
    base = run_validation(n_per_domain=8, seed=0)
    extra = ConfigurableSyntheticModel("extra_syco", FaithfulnessProfile(p_hint_sycophancy=1.0, seed=0))
    ext = run_validation(n_per_domain=8, seed=0, extended_models=[(extra, "SHI")])

    # Everything pooled over the frozen population is identical.
    for key in ("combined_auroc", "single_mixed_auroc", "targeted_auroc"):
        assert _norm(ext["validation"][key]) == _norm(base["validation"][key]), key
    assert _norm(ext["correlation"]) == _norm(base["correlation"])
    assert _norm(ext["cards"]) == _norm(base["cards"])

    # ...yet the held-out model gets its own pairwise targeted-AUROC entry.
    ea = ext["validation"]["extended_auroc"]
    assert "extra_syco" in ea
    assert ea["extra_syco"]["probe"] == "SHI"
    assert ea["extra_syco"]["auc"] == 1.0
