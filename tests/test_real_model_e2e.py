"""End-to-end real-model score path, replayed offline from a committed FAKE cache (fb-0nc.5).

The cache is a deterministic fixture, NOT real Claude data (see experiments/record_replay.py);
swapping in a real recording is one command with a key. This proves the identical probe code
runs through the Anthropic adapter with no key and no SDK.
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
