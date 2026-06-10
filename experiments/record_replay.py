#!/usr/bin/env python3
"""Record a **fake** replay cache so the real-model `score` path runs end-to-end offline.

⚠️  THIS IS A FIXTURE, NOT REAL CLAUDE DATA.  ``fake_claude_transport`` is a deterministic
stub that returns plausible-looking (but made-up) chain-of-thought + answers. It exists so
the whole Anthropic adapter / probe / Faithfulness-Card pipeline can be exercised in CI with
no API key and no spend, and so the cache *format* is exactly what a real recording produces.

To record against the **real** model instead (costs money, needs a key), run:

    ANTHROPIC_API_KEY=sk-... python experiments/record_replay.py --real

which records the identical requests through the live SDK into the same cache file — a
one-command swap. The committed `replay_cache/fake_sonnet.json` is the fake fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faithfulnessbench.card import build_card  # noqa: E402
from faithfulnessbench.problems import mixed_problems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Shared config — matches `faithfulnessbench score -n 2 --seed 0 --trials 2` so the CLI
# replays from this cache offline. (Both recorder and the e2e test import these.)
MODEL = "claude-sonnet-4-6"
EFFORT = "medium"
N_PER_DOMAIN = 2
SEED = 0
N_TRIALS = 2
CACHE_PATH = ROOT / "experiments" / "replay_cache" / "fake_sonnet.json"


def problems():
    return mixed_problems(N_PER_DOMAIN, seed=SEED)


def fake_claude_transport(spec: dict) -> tuple[str, str]:
    """Deterministic FAKE Claude response keyed off the request (NOT real model output)."""
    h = int(hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), 16)
    value = h % 97
    if spec["think"]:
        cot = f"Let me reason about this.\nintermediate value {value - 4}\nso the result is {value}"
    else:
        cot = ""
    text = f"Reasoning done.\nANSWER: {value}"
    return cot, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="record against the live SDK (needs a key, costs money)")
    args = ap.parse_args()

    from faithfulnessbench.models.anthropic_model import AnthropicModel

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()  # fresh record

    kwargs = dict(model=MODEL, effort=EFFORT, cache_path=CACHE_PATH)
    model = AnthropicModel(**kwargs) if args.real else AnthropicModel(transport=fake_claude_transport, **kwargs)
    card = build_card(model, problems(), n_trials=N_TRIALS)

    n_entries = len(json.loads(CACHE_PATH.read_text()))
    print(f"{'REAL' if args.real else 'FAKE'} cache -> {CACHE_PATH} ({n_entries} entries)")
    print(f"Card composite faithfulness: {card.composite_faithfulness:.3f} (descriptive — fixture data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
