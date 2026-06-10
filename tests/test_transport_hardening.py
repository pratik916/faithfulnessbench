"""Transport hardening: retry/backoff, truncation flag, token/cost accounting, and safe
<script> JSON embedding — all with no key/SDK (fb-0nc.3)."""
from __future__ import annotations

import pytest

from faithfulnessbench.models.anthropic_model import AnthropicModel
from faithfulnessbench.problems import arithmetic_chain_problems
from faithfulnessbench.report.html import _safe_json_for_script

P = arithmetic_chain_problems(1, seed=0)[0]


def test_two_caches_on_same_path_do_not_stomp_each_other(tmp_path):
    # The real-model path runs a main model and a separate LLM judge/simulator, each with
    # its own cache instance on the same file. A naive whole-file rewrite would lose one
    # side's entries; put() must merge so the offline replay has every recorded call.
    from faithfulnessbench.models.anthropic_model import _JsonCache

    path = tmp_path / "cache.json"
    a = _JsonCache(path)
    b = _JsonCache(path)
    a.put("ka", ("cot-a", "ANSWER: 1"))
    b.put("kb", ("cot-b", "ANSWER: 2"))
    fresh = _JsonCache(path)
    assert fresh.get("ka") == ["cot-a", "ANSWER: 1"]
    assert fresh.get("kb") == ["cot-b", "ANSWER: 2"]


def test_retries_transient_errors_then_succeeds():
    calls = {"n": 0}

    def transport(spec):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return ("cot", "ANSWER: 5")

    m = AnthropicModel(transport=transport, max_retries=3, backoff=0.0)
    assert m.reason(P).answer == "5"
    assert calls["n"] == 3


def test_gives_up_after_max_retries():
    def transport(spec):
        raise RuntimeError("always")

    m = AnthropicModel(transport=transport, max_retries=1, backoff=0.0)
    with pytest.raises(RuntimeError):
        m.reason(P)


class _Resp:
    def __init__(self, truncated):
        self.content = [type("B", (), {"type": "text", "text": "ANSWER: 9"})()]
        self.stop_reason = "max_tokens" if truncated else "end_turn"
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 40})()


def _client(truncated):
    class C:
        class messages:
            @staticmethod
            def create(**kw):
                return _Resp(truncated)

    return C()


def test_records_tokens_and_truncation_flag():
    m = AnthropicModel("claude-opus-4-8", client=_client(truncated=True))
    m.reason(P)
    assert m.call_log[-1]["truncated"] is True
    assert m.call_log[-1]["input_tokens"] == 100
    assert m.token_totals()["output_tokens"] == 40
    assert m.estimated_cost_usd() == pytest.approx((100 * 5.0 + 40 * 25.0) / 1e6)


def test_not_truncated_on_normal_stop():
    m = AnthropicModel("claude-opus-4-8", client=_client(truncated=False))
    m.reason(P)
    assert m.call_log[-1]["truncated"] is False


def test_safe_json_escapes_script_and_line_separators():
    sep = chr(0x2028) + chr(0x2029)
    out = _safe_json_for_script([{"cot": "</script><x>" + sep}])
    assert "</script>" not in out and "<\\/script>" in out
    assert chr(0x2028) not in out and "\\u2028" in out
    assert chr(0x2029) not in out and "\\u2029" in out
