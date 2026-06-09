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


def test_stack_probe_scores_aligns_by_problem_id():
    """Alignment must be by problem id, not list position — so a reordered (or
    partially dropped) probe result can't silently corrupt the correlation matrix."""
    from faithfulnessbench.card import stack_probe_scores
    from faithfulnessbench.probes.base import ProbeResult

    a = ProbeResult("A", ["p2", "p1", "p3"], np.array([0.2, 0.1, 0.3]))
    b = ProbeResult("B", ["p1", "p3", "p2"], np.array([0.1, 0.3, 0.2]))  # same data, shuffled
    cols = stack_probe_scores([{"A": a, "B": b}], ["A", "B"])
    # Ordered by A's ids (p2, p1, p3); B is realigned to match.
    np.testing.assert_allclose(cols["A"], [0.2, 0.1, 0.3])
    np.testing.assert_allclose(cols["B"], [0.2, 0.1, 0.3])


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
