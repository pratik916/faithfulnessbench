#!/usr/bin/env python3
"""Record a replay cache so the real-model `score` path runs end-to-end offline.

Two modes:

* **Fake (default, no key, no spend).** ``fake_claude_transport`` is a deterministic stub
  returning plausible-but-made-up CoT + answers. The committed ``replay_cache/fake_*.json``
  fixtures let the whole adapter / probe / Faithfulness-Card pipeline run in CI with no key,
  and pin the cache *format* a real recording produces. Regenerate them with::

      python experiments/record_replay.py

* **Real (needs a key, costs money).** Records the **flagship models** (``claude-sonnet-4-6``
  and ``claude-opus-4-8``) over the mixed + GSM8K substrates at a standard sample size through
  the live SDK, into ``replay_cache/real_<model>[ _gsm8k].json`` plus a ``real_manifest.json``
  of the descriptive numbers. Real models have **no faithfulness ground truth**, so these are
  descriptive statistics (flip-rate, simulatability, early-lock, judge-reliability) — never an
  AUROC-vs-truth. Run::

      ANTHROPIC_API_KEY=sk-... python experiments/record_replay.py --real

  The cache makes every downstream replay (tests, transfer, the real-model report) run offline
  with no key. Re-running resumes from the cache (merge-on-put), so an interrupted run is cheap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faithfulnessbench.problems import mixed_problems  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPLAY_DIR = ROOT / "experiments" / "replay_cache"
GSM8K_SAMPLE = ROOT / "experiments" / "datasets" / "gsm8k_sample.jsonl"

# --- FAKE fixture config (committed, offline, no key) — keep stable; tests import these. ---
MODEL = "claude-sonnet-4-6"
EFFORT = "medium"
N_PER_DOMAIN = 2
SEED = 0
N_TRIALS = 2
CACHE_PATH = REPLAY_DIR / "fake_sonnet.json"
GSM8K_CACHE = REPLAY_DIR / "fake_gsm8k.json"

# --- REAL recording config (needs a key, costs money) — both flagship models, standard size. ---
REAL_MODELS = ["claude-sonnet-4-6", "claude-opus-4-8"]
REAL_N_PER_DOMAIN = 8
REAL_N_TRIALS = 3
REAL_MANIFEST = REPLAY_DIR / "real_manifest.json"


def problems():
    return mixed_problems(N_PER_DOMAIN, seed=SEED)


def real_problems():
    return mixed_problems(REAL_N_PER_DOMAIN, seed=SEED)


def gsm8k_problems():
    from faithfulnessbench.gsm8k import load_gsm8k

    return load_gsm8k(GSM8K_SAMPLE)


def _short(model: str) -> str:
    parts = model.split("-")
    return parts[1] if len(parts) > 1 else model


def real_cache(model: str, substrate: str) -> Path:
    suffix = "" if substrate == "mixed" else "_gsm8k"
    return REPLAY_DIR / f"real_{_short(model)}{suffix}.json"


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


def _record_fake() -> int:
    from faithfulnessbench.models.anthropic_model import AnthropicModel, score_real_model

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for cache, probs, label in (
        (CACHE_PATH, problems(), "mixed"),
        (GSM8K_CACHE, gsm8k_problems(), "gsm8k"),
    ):
        if cache.exists():
            cache.unlink()  # fresh record
        model = AnthropicModel(model=MODEL, effort=EFFORT, cache_path=cache, transport=fake_claude_transport)
        card, jr = score_real_model(
            model=model, problems=probs, n_trials=N_TRIALS,
            transport=fake_claude_transport, cache_path=cache,
        )
        n_entries = len(json.loads(cache.read_text()))
        krel = f"; judge kappa {jr['kappa_vs_gold']:.3f}" if jr else ""
        print(f"FAKE {label} cache -> {cache} ({n_entries} entries); composite {card.composite_faithfulness:.3f} (descriptive){krel}")
    return 0


def _descriptive_summary(card, jr) -> dict:
    d = card.to_dict()
    extras = d.get("extras", {})
    return {
        "n_problems": d["n_problems"],
        "composite_faithfulness": d["composite_faithfulness"],
        "probe_faithfulness": {p: d["probe_scores"][p]["faithfulness"] for p in d["probe_scores"]},
        "shi_flip_rate": extras.get("SHI", {}).get("flip_rate"),
        "shi_ack_rate_given_flip": extras.get("SHI", {}).get("ack_rate_given_flip"),
        "sim_parse_failure_rate": extras.get("SIM", {}).get("parse_failure_rate"),
        "judge_kappa_vs_gold": (jr or {}).get("kappa_vs_gold"),
        "judge_landis_koch": (jr or {}).get("landis_koch"),
    }


def _record_real() -> int:
    from faithfulnessbench.models.anthropic_model import AnthropicModel, score_real_model

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "WARNING": "Descriptive real-model statistics — real models have NO faithfulness "
                   "ground truth, so there is no AUROC-vs-truth here.",
        "config": {"models": REAL_MODELS, "n_per_domain": REAL_N_PER_DOMAIN,
                   "n_trials": REAL_N_TRIALS, "effort": EFFORT, "seed": SEED},
        "models": {},
        "estimated_cost_usd_main_only": 0.0,
    }
    for model_id in REAL_MODELS:
        manifest["models"][model_id] = {}
        main_cost = 0.0
        for substrate, probs in (("mixed", real_problems()), ("gsm8k", gsm8k_problems())):
            cache = real_cache(model_id, substrate)
            model = AnthropicModel(model=model_id, effort=EFFORT, cache_path=cache)  # transport=None -> live
            card, jr = score_real_model(
                model=model, problems=probs, n_trials=REAL_N_TRIALS, cache_path=cache,
            )
            n_entries = len(json.loads(cache.read_text()))
            main_cost += model.estimated_cost_usd()
            manifest["models"][model_id][substrate] = _descriptive_summary(card, jr)
            krel = f"; judge kappa {jr['kappa_vs_gold']:.3f}" if jr else ""
            print(f"REAL {model_id} {substrate} -> {cache} ({n_entries} entries); "
                  f"composite {card.composite_faithfulness:.3f} (descriptive){krel}; "
                  f"main-model ~${model.estimated_cost_usd():.2f}, {model.truncated_calls} truncated")
        manifest["estimated_cost_usd_main_only"] += main_cost
    REAL_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest -> {REAL_MANIFEST}")
    print(f"Estimated spend (main models only; excludes the cheap haiku judge): "
          f"~${manifest['estimated_cost_usd_main_only']:.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true", help="record against the live SDK (needs a key, costs money)")
    args = ap.parse_args()
    return _record_real() if args.real else _record_fake()


if __name__ == "__main__":
    raise SystemExit(main())
