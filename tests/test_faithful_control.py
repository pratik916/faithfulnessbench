"""A faithful-by-construction positive control (fb-yst.4).

A solver-style model whose answer is *derived* by executing its own printed L op R = V
chain (Lyu et al., Faithful CoT) — faithful on every axis by construction. It is a
canonical true-negative anchor and lives in the held-out extended population, not the
frozen one, so the committed pooled numbers are unchanged.
"""
from __future__ import annotations

from faithfulnessbench.models.synthetic import faithful_by_construction, model_population
from faithfulnessbench.probes import default_probes
from faithfulnessbench.problems import execute_steps, mixed_problems

PROBLEMS = mixed_problems(8, seed=0)


def test_control_is_a_true_negative_on_every_probe():
    control = faithful_by_construction(seed=0)
    for probe in default_probes():
        assert probe.run(control, PROBLEMS).scores.mean() < 0.05, probe.name


def test_control_answer_is_derived_from_its_own_chain():
    control = faithful_by_construction(seed=0)
    for p in PROBLEMS:
        tr = control.reason(p)
        assert tr.answer == p.value_to_answer(execute_steps(tr.steps))


def test_control_is_not_in_the_frozen_population():
    assert "faithful_by_construction" not in {m.name for m in model_population()}
