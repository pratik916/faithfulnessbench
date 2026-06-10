"""GSM8K real-math substrate on the real-model path, replayed offline (fb-jlg.1)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

from faithfulnessbench.card import build_card
from faithfulnessbench.gsm8k import load_gsm8k, parse_gsm8k_answer
from faithfulnessbench.models.anthropic_model import AnthropicModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from record_replay import EFFORT, GSM8K_CACHE, GSM8K_SAMPLE, MODEL, N_TRIALS, gsm8k_problems  # noqa: E402


def test_parse_gsm8k_answer_extracts_integer_after_marker():
    assert parse_gsm8k_answer("some work\n#### 72") == 72
    assert parse_gsm8k_answer("final #### 1,234") == 1234


def test_loader_parses_bundled_sample():
    probs = load_gsm8k(GSM8K_SAMPLE)
    assert len(probs) >= 5
    assert all(p.domain == "gsm8k" for p in probs)
    assert probs[0].answer == "72"


def test_probes_run_over_gsm8k_offline_from_cache():
    model = AnthropicModel(model=MODEL, effort=EFFORT, cache_path=GSM8K_CACHE)  # no transport/client/key
    card = build_card(model, gsm8k_problems(), n_trials=N_TRIALS)
    assert set(card.probe_scores) == {"SHI", "CSC", "SIM", "EAR"}
    # Free-text GSM8K CoT has no parseable L op R = V chain: CSC has nothing to corrupt
    # (honest degradation), while EAR runs on the model's own truncated reasoning.
    assert math.isnan(card.probe_scores["CSC"]["faithfulness"])
    assert not math.isnan(card.probe_scores["EAR"]["faithfulness"])
