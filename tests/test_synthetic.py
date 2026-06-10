"""The synthetic model's dials must produce exactly the behaviors the probes detect.

These tests are the contract between the ground-truth world and the probes: if a dial
doesn't manifest, the corresponding probe's validation AUROC would be meaningless.
"""
from __future__ import annotations

from faithfulnessbench import problems as PB
from faithfulnessbench.models.synthetic import (
    ConfigurableSyntheticModel as Model,
)
from faithfulnessbench.models.synthetic import (
    ExactArithmeticSimulator,
    model_population,
)
from faithfulnessbench.models.synthetic import (
    FaithfulnessProfile as Prof,
)


def _arith():
    return PB.arithmetic_chain_problems(5, seed=0)


def _corrupt_operand(steps, idx=1, delta=4):
    """Surface-preserving operand perturbation of one step."""
    left, op, operand, result = PB.parse_step(steps[idx])
    new_operand = operand + delta
    return [
        *steps[:idx],
        PB.format_step(left, op, new_operand, result),
        *steps[idx + 1 :],
    ]


def test_step_format_roundtrip_and_execution():
    p = _arith()[0]
    # The canonical chain re-executes to the stated final and the problem answer.
    assert PB.execute_steps(p.steps) == int(p.answer)
    assert PB.stated_final(p.steps) == int(p.answer)


def test_faithful_model_is_correct_and_unhinted():
    m = Model("faithful", Prof())
    for p in _arith():
        t = m.reason(p)
        assert t.answer == p.answer
        t_cued = m.reason(p, cue=p.cue)
        assert t_cued.answer == p.answer  # ignores the cue -> no flip


def test_sycophant_flips_silently():
    m = Model("sycophant", Prof(p_hint_sycophancy=1.0))
    for p in _arith():
        t = m.reason(p, cue=p.cue)
        assert t.answer == p.cue.target          # flipped to the cue
        assert t.answer != p.answer
        assert p.cue.marker not in t.cot         # ... without acknowledging it
        assert p.cue.text not in t.cot


def test_decoy_cot_misleads_simulator():
    m = Model("decoy", Prof(p_decoy_cot=1.0))
    sim = ExactArithmeticSimulator()
    for p in _arith():
        t = m.reason(p)
        assert t.answer == p.answer                       # answer still correct
        assert PB.stated_final(t.steps) != int(p.answer)  # but CoT concludes elsewhere
        assert sim.predict(p, t.steps) != t.answer        # simulator misled


def test_post_hoc_ignores_corrupted_cot_but_loadbearing_follows():
    faithful = Model("faithful", Prof())
    post_hoc = Model("post_hoc", Prof(p_post_hoc=1.0))
    for p in _arith():
        base = faithful.reason(p)
        corrupted = _corrupt_operand(list(base.steps))
        # Load-bearing model tracks the corruption; post-hoc model does not.
        assert faithful.continue_from_cot(p, corrupted) != p.answer
        assert post_hoc.continue_from_cot(p, corrupted) == p.answer


def test_pre_commit_answers_before_reasoning():
    faithful = Model("faithful", Prof())
    pre = Model("pre_commit", Prof(p_pre_commit=1.0))
    for p in _arith():
        # With no CoT revealed, the faithful model can't know the answer yet.
        assert faithful.answer_from_prefix(p, []) != p.answer
        # The pre-committed model already does.
        assert pre.answer_from_prefix(p, []) == p.answer


def test_mcq_domain_end_to_end():
    m = Model("faithful", Prof())
    for p in PB.multiple_choice_problems(5, seed=2):
        t = m.reason(p)
        assert t.answer == p.answer
        assert t.answer in {"A", "B", "C", "D"}


def test_population_labels():
    pop = {m.name: m for m in model_population()}
    assert pop["faithful"].profile.is_faithful
    assert pop["sycophant"].profile.label_for("SHI") == 1
    assert pop["sycophant"].profile.label_for("CSC") == 0
    assert pop["fully_unfaithful"].profile.label_for("EAR") == 1
