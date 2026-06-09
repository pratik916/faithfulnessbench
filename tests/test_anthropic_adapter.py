"""The Anthropic adapter's parsing, prompt-routing, and cache — tested with a fake
transport so no API key or network is required."""
from __future__ import annotations

from faithfulnessbench.models.anthropic_model import AnthropicModel, _parse_answer
from faithfulnessbench.problems import arithmetic_chain_problems, multiple_choice_problems


def test_parse_answer_arithmetic_and_mcq():
    arith = arithmetic_chain_problems(1, seed=0)[0]
    mcq = multiple_choice_problems(1, seed=0)[0]
    assert _parse_answer("work...\nANSWER: 42", arith) == "42"
    assert _parse_answer("ANSWER: -7", arith) == "-7"
    assert _parse_answer("so the answer is\nANSWER: b", mcq) == "B"
    # Fallbacks when the model forgets the ANSWER line.
    assert _parse_answer("the total comes to 13", arith) == "13"
    assert _parse_answer("option C looks right", mcq) == "C"
    assert _parse_answer("", arith) == "?"


def test_reason_routes_thinking_and_parses(monkeypatch):
    calls = []

    def transport(spec):
        calls.append(spec)
        return ("step one\nstep two", "Here is my work.\nANSWER: 123")

    m = AnthropicModel("claude-sonnet-4-6", transport=transport)
    p = arithmetic_chain_problems(1, seed=1)[0]
    tr = m.reason(p)
    assert tr.answer == "123"
    assert tr.steps == ["step one", "step two"]
    assert calls[0]["think"] is True  # reasoning uses adaptive thinking


def test_commit_probes_disable_thinking():
    seen = []

    def transport(spec):
        seen.append(spec["think"])
        return ("", "ANSWER: 5")

    m = AnthropicModel(transport=transport)
    p = arithmetic_chain_problems(1, seed=2)[0]
    assert m.continue_from_cot(p, ["10 + 1 = 11"]) == "5"
    assert m.answer_from_prefix(p, []) == "5"
    # Both commit-style probes must NOT let the model re-reason.
    assert seen == [False, False]


def test_cache_avoids_repeat_calls(tmp_path):
    n = {"calls": 0}

    def transport(spec):
        n["calls"] += 1
        return ("cot", "ANSWER: 7")

    cache = tmp_path / "cache.json"
    m1 = AnthropicModel(transport=transport, cache_path=cache)
    p = arithmetic_chain_problems(1, seed=3)[0]
    assert m1.reason(p).answer == "7"
    assert m1.reason(p).answer == "7"  # identical request -> served from cache
    assert n["calls"] == 1
    # A fresh model pointed at the same cache file also hits it (no new call).
    m2 = AnthropicModel(transport=transport, cache_path=cache)
    assert m2.reason(p).answer == "7"
    assert n["calls"] == 1
