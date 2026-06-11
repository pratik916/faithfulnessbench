"""The hard-instance substrate: instances on which the *unfaithful* behavior coincides
with faithful behavior, so a perfectly-wired probe still cannot separate the classes.

Unlike symmetric label noise (which mechanically degrades any classifier), hardness here
is intrinsic to the task: a cue that points at the *correct* answer (SHI is blind to a
sycophant when the hint is right) and an *answer-obvious* identity chain (EAR cannot tell
reliance from pre-commitment when there is nothing to rely on). CSC and SIM stay at ceiling
on the same instances because their interventions remain discriminating — that asymmetry is
the point (see test_hardness_curve.py).
"""
from __future__ import annotations

from faithfulnessbench.problems import (
    execute_steps,
    hard_mixed_problems,
    stated_final,
)


def _is_identity(p) -> bool:
    """Every stated result — and the start — equals the final answer value."""
    start = int(p.meta["start"])
    final = int(p.meta["final_value"])
    if start != final:
        return False
    return all(stated_final(p.steps[: k + 1]) == final for k in range(len(p.steps)))


def test_zero_hardness_is_all_normal_instances():
    arith = [p for p in hard_mixed_problems(8, seed=0, hardness=0.0) if p.domain == "arithmetic"]
    assert len(arith) == 8
    # Normal instances: the planted cue points at a WRONG answer, and the chain is not a
    # trivial identity (it does real work).
    assert all(p.cue.target != p.answer for p in arith)
    assert not any(_is_identity(p) for p in arith)


def test_full_hardness_makes_every_instance_a_blind_spot():
    probs = hard_mixed_problems(8, seed=0, hardness=1.0)
    assert len(probs) == 16
    for p in probs:
        # SHI blind spot: the cue points at the CORRECT answer, so a silent flip is invisible.
        assert p.cue.target == p.answer, p.id
        # EAR blind spot: the answer is obvious from the premise — every prefix already
        # states the final answer, so even an empty-reasoning guess matches.
        assert _is_identity(p), p.id
        # The problem is still internally coherent: executing the chain yields the answer.
        assert p.value_to_answer(execute_steps(p.steps)) == p.answer


def test_hardness_fraction_is_exact_and_monotone():
    # k = round(h * n) hard instances per domain (the first k), so the curve is smooth.
    for h, expected in [(0.0, 0), (0.25, 2), (0.5, 4), (0.75, 6), (1.0, 8)]:
        arith = [p for p in hard_mixed_problems(8, seed=0, hardness=h) if p.domain == "arithmetic"]
        n_hard = sum(_is_identity(p) and p.cue.target == p.answer for p in arith)
        assert n_hard == expected, (h, n_hard)


def test_hard_mcq_correct_hint_points_at_the_answer_letter():
    mcq = [p for p in hard_mixed_problems(8, seed=0, hardness=1.0) if p.domain == "mcq"]
    assert len(mcq) == 8
    for p in mcq:
        assert p.choices is not None and len(p.choices) == 4
        assert p.cue.target == p.answer  # the correct option letter
        # The answer letter is a real option value reachable by executing the chain.
        assert p.value_to_answer(execute_steps(p.steps)) == p.answer


def test_deterministic():
    a = hard_mixed_problems(6, seed=3, hardness=0.5)
    b = hard_mixed_problems(6, seed=3, hardness=0.5)
    assert [p.id for p in a] == [p.id for p in b]
    assert [p.steps for p in a] == [p.steps for p in b]
    assert [p.cue.target for p in a] == [p.cue.target for p in b]
