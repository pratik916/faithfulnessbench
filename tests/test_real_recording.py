"""The committed real-Claude recording: it exists, reproduces offline, and exhibits the
gold-vs-judge degeneracy that motivates the LLM-graded real-model path.

Every test here runs **offline, no key, no network** — it replays the committed
``replay_cache/real_*.json`` (real Sonnet 4.6 + Opus 4.8). It is the regression gate for the
real-model receipts: the descriptive numbers in ``real_manifest.json`` must reproduce from the
caches, and the headline "the literal cue detector is useless on real CoT, the LLM judge is not"
finding must hold.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from record_replay import (  # noqa: E402
    EFFORT,
    GSM8K_SAMPLE,
    REAL_MANIFEST,
    REAL_MODELS,
    REAL_N_TRIALS,
    gsm8k_problems,
    real_cache,
    real_problems,
)

from faithfulnessbench.models.anthropic_model import (  # noqa: E402
    AnthropicModel,
    LLMJudgeCueDetector,
    score_real_model,
)
from faithfulnessbench.models.base import SubstringCueDetector  # noqa: E402

_SUBSTRATES = ("mixed", "gsm8k")


def _manifest() -> dict:
    return json.loads(REAL_MANIFEST.read_text())


def _probs(substrate: str):
    return real_problems() if substrate == "mixed" else gsm8k_problems()


def _close(a, b, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return math.isclose(a, b, abs_tol=tol)


def test_real_caches_and_manifest_committed():
    assert REAL_MANIFEST.exists(), "real_manifest.json must be committed"
    for model_id in REAL_MODELS:
        for sub in _SUBSTRATES:
            cache = real_cache(model_id, sub)
            assert cache.exists(), f"missing committed real cache: {cache}"


def test_real_manifest_reproduces_offline_no_key():
    # Replaying each committed cache through the same scoring path must reproduce the committed
    # manifest numbers exactly — offline, no key, no network. This is the real-model receipt.
    man = _manifest()["models"]
    for model_id in REAL_MODELS:
        for sub in _SUBSTRATES:
            cache = real_cache(model_id, sub)
            model = AnthropicModel(model=model_id, effort=EFFORT, cache_path=cache)
            card, _ = score_real_model(
                model=model, problems=_probs(sub), n_trials=REAL_N_TRIALS, cache_path=cache,
            )
            exp = man[model_id][sub]
            assert _close(card.composite_faithfulness, exp["composite_faithfulness"]), (model_id, sub)
            for probe, f in exp["probe_faithfulness"].items():
                assert _close(card.probe_scores[probe]["faithfulness"], f), (model_id, sub, probe)
            assert _close(card.extras["SHI"]["flip_rate"], exp["shi_flip_rate"]), (model_id, sub)
            # Replaying must not mutate the committed cache (every call is a hit).
            json.loads(cache.read_text())


def test_substring_gold_degenerate_but_llm_judge_works_on_real_cot():
    # The headline real-CoT finding: the exact substring cue-detector cannot grade free-text CoT
    # (real models paraphrase a hint, never echo the literal sentinel), while the LLM judge does
    # — which is why judge_kappa_vs_gold is 0.0 *by construction*, not a bug.
    cache = real_cache("claude-sonnet-4-6", "mixed")
    model = AnthropicModel(model="claude-sonnet-4-6", effort=EFFORT, cache_path=cache)
    gold = SubstringCueDetector()
    judge = LLMJudgeCueDetector("claude-haiku-4-5", cache_path=cache)
    gold_hits = judge_hits = n = 0
    for p in real_problems():
        if p.cue is None:
            continue
        tr = model.reason(p, cue=p.cue, trial=0)
        n += 1
        gold_hits += int(gold.mentions(tr.cot, p.cue))
        judge_hits += int(judge.mentions(tr.cot, p.cue))
    assert n >= 8, n
    assert gold_hits == 0, f"expected the literal marker to never appear in real CoT, got {gold_hits}"
    assert judge_hits >= n // 2, f"the LLM judge should detect most acknowledgments, got {judge_hits}/{n}"


def test_real_shi_flip_rate_is_zero_everywhere():
    # On this substrate the models acknowledge the planted hint but do not flip the answer.
    man = _manifest()["models"]
    for model_id in REAL_MODELS:
        for sub in _SUBSTRATES:
            assert man[model_id][sub]["shi_flip_rate"] == 0.0, (model_id, sub)


def test_real_gsm8k_transfer_degrades_csc_but_runs_shi():
    # The committed report/transfer.html is sourced from this: real GSM8K free-text CoT keeps
    # SHI running while CSC degrades (no parseable "L op R = V" chain to corrupt).
    from faithfulnessbench.transfer import build_gsm8k_transfer

    t = build_gsm8k_transfer(
        cache_path=str(real_cache("claude-sonnet-4-6", "gsm8k")),
        gsm8k_sample=str(GSM8K_SAMPLE),
        model="claude-sonnet-4-6", effort=EFFORT, n_trials=REAL_N_TRIALS,
    )
    assert t["gsm8k_real"]["SHI"]["ran"] is True
    assert t["gsm8k_real"]["CSC"]["ran"] is False
