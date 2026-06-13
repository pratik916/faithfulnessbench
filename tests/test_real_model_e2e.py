"""End-to-end real-model score path, replayed offline from the tiny synthetic fixture (fb-0nc.5).

This uses the small deterministic ``fake_*.json`` fixture (made-up CoT, n=2) as a fast,
byte-stable smoke test that the identical probe code runs through the Anthropic adapter with no
key and no SDK. The *real* Claude measurement lives in ``replay_cache/real_*.json`` and is
exercised by ``test_real_recording.py``; this file deliberately stays on the tiny fixture so the
core suite is fast and needs no large caches.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from record_replay import CACHE_PATH, EFFORT, MODEL, N_TRIALS, problems  # noqa: E402

from faithfulnessbench.card import build_card  # noqa: E402
from faithfulnessbench.cli import main  # noqa: E402
from faithfulnessbench.models.anthropic_model import AnthropicModel  # noqa: E402


def test_committed_fake_cache_exists():
    assert CACHE_PATH.exists(), "the fake replay-cache fixture must be committed"


def test_score_path_replays_offline_from_cache():
    # No transport and no client: every request must be served from the committed cache.
    model = AnthropicModel(model=MODEL, effort=EFFORT, cache_path=CACHE_PATH)
    card = build_card(model, problems(), n_trials=N_TRIALS)
    assert set(card.probe_scores) == {"SHI", "CSC", "SIM", "EAR"}
    assert 0.0 <= card.composite_faithfulness <= 1.0
    assert card.to_dict()["model_name"] == MODEL
    # Real free-text CoT doesn't parse as "L op R = V": SIM flags it, CSC has no parseable
    # steps to corrupt — the documented real-path limitations, surfaced honestly.
    assert card.extras["SIM"]["parse_failure_rate"] == 1.0
    assert math.isnan(card.probe_scores["CSC"]["faithfulness"])


def test_cli_score_runs_offline_from_cache():
    rc = main([
        "score", "--model", MODEL, "--cache", str(CACHE_PATH),
        "-n", "2", "--seed", "0", "--trials", "2", "--effort", EFFORT,
    ])
    assert rc == 0


def test_score_path_uses_the_llm_judge_offline_and_audits_it():
    # The default `score` path grades SHI/SIM with the LLM judge/simulator (not the synthetic
    # exact detectors), all replayed from the committed cache — so README/DESIGN's claim that
    # "the real-model path uses an LLM judge" is true and CI-exercised with no key.
    import json as _json

    from faithfulnessbench.models.anthropic_model import score_real_model

    model = AnthropicModel(model=MODEL, effort=EFFORT, cache_path=CACHE_PATH)
    card, jr = score_real_model(
        model=model, problems=problems(), n_trials=N_TRIALS, cache_path=CACHE_PATH
    )
    assert set(card.probe_scores) == {"SHI", "CSC", "SIM", "EAR"}
    assert jr is not None and "kappa_vs_gold" in jr and jr["n"] > 0
    # The committed cache must not be mutated by a pure replay (every call is a hit).
    _json.loads(CACHE_PATH.read_text())


def test_cli_score_out_writes_judge_reliability(tmp_path):
    import json as _json

    out = tmp_path / "card.json"
    rc = main([
        "score", "--model", MODEL, "--cache", str(CACHE_PATH),
        "-n", "2", "--seed", "0", "--trials", "2", "--effort", EFFORT,
        "--out", str(out),
    ])
    assert rc == 0
    card = _json.loads(out.read_text())
    assert "judge_reliability" in card and "kappa_vs_gold" in card["judge_reliability"]
