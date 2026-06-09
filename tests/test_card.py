"""Faithfulness Card aggregation and the cross-probe disagreement finding."""
from __future__ import annotations

import numpy as np

from faithfulnessbench.card import build_card, correlation_matrix, run_probes
from faithfulnessbench.models.synthetic import model_population
from faithfulnessbench.problems import mixed_problems

PROBLEMS = mixed_problems(10, seed=0)
POP = {m.name: m for m in model_population()}


def test_faithful_model_scores_near_one():
    card = build_card(POP["faithful"], PROBLEMS)
    assert card.composite_faithfulness > 0.95
    for name, d in card.probe_scores.items():
        assert d["faithfulness"] > 0.9, name


def test_fully_unfaithful_scores_near_zero():
    card = build_card(POP["fully_unfaithful"], PROBLEMS)
    assert card.composite_faithfulness < 0.05


def test_per_domain_breakdown_present():
    card = build_card(POP["faithful"], PROBLEMS)
    assert set(card.per_domain) == {"arithmetic", "mcq"}
    assert set(card.per_domain["arithmetic"]) == {"SHI", "CSC", "SIM", "EAR"}


def test_card_is_json_serializable():
    import json

    card = build_card(POP["sycophant"], PROBLEMS)
    json.dumps(card.to_dict())  # must not raise


def test_cross_probe_disagreement():
    """Pooled over the population, the four probes are near-independent: the diagonal
    is 1.0 and every off-diagonal correlation is far lower — the evidence that no
    single probe substitutes for the others."""
    names = ["SHI", "CSC", "SIM", "EAR"]
    per_model = [run_probes(m, PROBLEMS) for m in POP.values()]
    labels, mat = correlation_matrix(per_model, names)
    assert labels == names
    np.testing.assert_allclose(np.diag(mat), 1.0)
    off_diagonal = mat[~np.eye(len(names), dtype=bool)]
    assert np.nanmax(off_diagonal) < 0.6  # probes do not stand in for one another
