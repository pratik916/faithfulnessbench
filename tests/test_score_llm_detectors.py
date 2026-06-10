"""The `score` (real-model) path must use the LLM collaborators it advertises (fb-b5t.1).

Until now `_cmd_score` ran the synthetic-only exact detectors, so README/DESIGN's claim
that "the real-model path uses an LLM judge" was false. These tests pin the real-model
probe battery (LLM judge for SHI, LLM simulator for SIM, length/sign-preserving corruptor
for CSC) and the judge-reliability audit, all exercised offline via the transport seam.
"""
from __future__ import annotations

from faithfulnessbench.card import build_card
from faithfulnessbench.models.anthropic_model import (
    AnthropicModel,
    LLMJudgeCueDetector,
    LLMSimulator,
    judge_reliability,
    real_model_probes,
    score_real_model,
)
from faithfulnessbench.models.base import SubstringCueDetector
from faithfulnessbench.models.synthetic import ExactArithmeticSimulator
from faithfulnessbench.probes.csc import LengthSignPreservingCorruptor
from faithfulnessbench.problems import mixed_problems


def _fake_transport(spec):
    # Deterministic, key-free: a parseable CoT when thinking, an ANSWER line always.
    return ("Let me reason.\nso the result is 5" if spec["think"] else ""), "ANSWER: 5"


def test_real_model_probes_wire_the_llm_collaborators():
    probes = {p.name: p for p in real_model_probes(transport=_fake_transport)}
    assert set(probes) == {"SHI", "CSC", "SIM", "EAR"}
    assert isinstance(probes["SHI"].detector, LLMJudgeCueDetector)
    assert isinstance(probes["SIM"].simulator, LLMSimulator)
    assert isinstance(probes["CSC"].corruptor, LengthSignPreservingCorruptor)


def test_real_model_probes_exact_fallback_uses_synthetic_detectors():
    probes = {p.name: p for p in real_model_probes(exact=True)}
    assert isinstance(probes["SHI"].detector, SubstringCueDetector)
    assert isinstance(probes["SIM"].simulator, ExactArithmeticSimulator)


def test_card_with_llm_detectors_runs_offline_via_transport():
    model = AnthropicModel("claude-sonnet-4-6", transport=_fake_transport)
    probes = real_model_probes(transport=_fake_transport, n_trials=1)
    card = build_card(model, mixed_problems(2, seed=0), probes, n_trials=1)
    assert set(card.probe_scores) == {"SHI", "CSC", "SIM", "EAR"}
    assert 0.0 <= card.composite_faithfulness <= 1.0


def test_judge_reliability_audits_the_llm_judge_vs_exact_gold():
    model = AnthropicModel("claude-sonnet-4-6", transport=_fake_transport)
    detector = LLMJudgeCueDetector(transport=_fake_transport)
    jr = judge_reliability(model, mixed_problems(4, seed=0), detector, n_trials=1)
    assert jr is not None
    assert {"kappa_vs_gold", "landis_koch", "self_consistency", "human_reference"} <= set(jr)
    assert jr["n"] > 0


def test_score_real_model_returns_card_and_judge_reliability():
    model = AnthropicModel("claude-sonnet-4-6", transport=_fake_transport)
    card, jr = score_real_model(
        model=model, problems=mixed_problems(2, seed=0), n_trials=1, transport=_fake_transport
    )
    assert set(card.probe_scores) == {"SHI", "CSC", "SIM", "EAR"}
    assert jr is not None and "kappa_vs_gold" in jr


def test_score_real_model_exact_mode_has_no_judge_reliability():
    model = AnthropicModel("claude-sonnet-4-6", transport=_fake_transport)
    card, jr = score_real_model(
        model=model, problems=mixed_problems(2, seed=0), n_trials=1,
        transport=_fake_transport, exact=True,
    )
    assert jr is None
