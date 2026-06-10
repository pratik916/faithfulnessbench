"""Thinking-channel vs answer-channel cue-acknowledgment divergence (fb-0nc.7)."""
from __future__ import annotations

import sys
from pathlib import Path

from faithfulnessbench.models.anthropic_model import (
    AnthropicModel,
    thinking_vs_answer_acknowledgment,
)
from faithfulnessbench.models.base import Model, Trace
from faithfulnessbench.problems import mixed_problems

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from record_replay import CACHE_PATH, MODEL, problems  # noqa: E402


def test_divergence_computed_from_both_channels_via_replay():
    m = AnthropicModel(model=MODEL, cache_path=CACHE_PATH)  # offline, no key
    d = thinking_vs_answer_acknowledgment(m, problems())
    assert d is not None
    assert 0.0 <= d["thinking_ack_rate"] <= 1.0
    assert 0.0 <= d["answer_ack_rate"] <= 1.0
    assert d["thinking_channel"] == "raw"  # sonnet emits raw thinking


def test_opus_channel_is_labeled_summarized_not_raw():
    def fake(spec):
        return ("summary of thinking", "ANSWER: 5")

    m = AnthropicModel("claude-opus-4-8", transport=fake)
    d = thinking_vs_answer_acknowledgment(m, mixed_problems(1, seed=0))
    assert d["thinking_channel"] == "summarized"  # guards against claiming the raw literature number


class _NoThinking(Model):
    name = "no_thinking"

    def reason(self, problem, *, cue=None, trial=0):
        return Trace(answer="1", cot="", steps=[], meta={"text": "ANSWER: 1"})

    def continue_from_cot(self, problem, cot_steps, *, trial=0):
        return "1"

    def answer_from_prefix(self, problem, prefix_steps, *, trial=0):
        return "1"


def test_absent_gracefully_without_a_thinking_channel():
    assert thinking_vs_answer_acknowledgment(_NoThinking(), mixed_problems(1, seed=0)) is None
