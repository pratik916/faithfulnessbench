"""A length/sign-preserving corruptor behind a Corruptor protocol (fb-bys.3)."""
from __future__ import annotations

import numpy as np

from faithfulnessbench import metrics as M
from faithfulnessbench.models.synthetic import model_population
from faithfulnessbench.probes.csc import (
    Corruptor,
    CSCProbe,
    LengthSignPreservingCorruptor,
    OperandCorruptor,
)
from faithfulnessbench.problems import arithmetic_chain_problems, parse_step

PROBS = arithmetic_chain_problems(15, seed=0)
POP = {m.name: m for m in model_population()}


def test_both_corruptors_satisfy_the_protocol():
    assert isinstance(OperandCorruptor(), Corruptor)
    assert isinstance(LengthSignPreservingCorruptor(), Corruptor)


def test_length_sign_preserving_keeps_operand_digit_count_and_sign():
    c = LengthSignPreservingCorruptor()
    total = same = 0
    for p in PROBS:
        steps = list(p.steps)
        for idx, corrupted in c.corruptions(steps):
            old = parse_step(steps[idx])[2]
            new = parse_step(corrupted[idx])[2]
            total += 1
            if len(str(abs(new))) == len(str(abs(old))) and (new > 0) == (old > 0):
                same += 1
    assert total > 0
    assert same / total >= 0.95


def test_csc_targeted_auroc_preserved_with_length_sign_corruptor():
    probe = CSCProbe(corruptor=LengthSignPreservingCorruptor())
    f = probe.run(POP["faithful"], PROBS).scores
    ph = probe.run(POP["post_hoc"], PROBS).scores
    auc = M.roc_auc(np.r_[f, ph], np.r_[np.zeros(f.size), np.ones(ph.size)])
    assert auc >= 0.95
