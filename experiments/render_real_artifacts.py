#!/usr/bin/env python3
"""Render the committed real-model HTML pages offline from the recorded caches.

This regenerates the shareable real-model artifacts — ``report/real_model_report.html``
(Sonnet + Opus faithfulness cards side by side) and ``report/transfer.html`` (the GSM8K
cross-domain table) — by replaying the committed ``replay_cache/real_*.json`` through the
*same* render path the synthetic validation uses. It is **offline / no key / $0**: the
caches already hold every model + judge response, so nothing hits the network.

Run after a real recording (``experiments/record_replay.py --real``) or any time to
reproduce the pages from a clean clone::

    python experiments/render_real_artifacts.py

The numbers it renders match ``replay_cache/real_manifest.json`` exactly (same caches, same
seed/trials). The chain-of-thought captured by the CLI recorder is the model's *visible*
step-by-step text (``claude -p`` exposes no hidden extended-thinking blocks) — the honest
target for CoT-faithfulness, stated as such on the page.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import record_replay as rr  # noqa: E402  (sibling script; reuse its recording config)

from faithfulnessbench.models.anthropic_model import (  # noqa: E402
    AnthropicModel,
    score_real_model,
)
from faithfulnessbench.report import write_card_report  # noqa: E402
from faithfulnessbench.transfer import build_gsm8k_transfer, write_transfer_html  # noqa: E402

REPORT = ROOT / "report"

_SUBTITLE = (
    "Descriptive — real models have no faithfulness ground truth, so there is no AUROC-vs-truth "
    "here. Recorded through the local Claude Code CLI (claude -p), so the chain-of-thought is the "
    "model's visible step-by-step text, not hidden extended thinking — the honest target for "
    "CoT-faithfulness. Replays offline from experiments/replay_cache/real_*.json (no key, $0); "
    "numbers match real_manifest.json."
)


def _real_card_dict(model_id: str, substrate: str, problems) -> dict:
    cache = rr.real_cache(model_id, substrate)
    if not cache.exists():
        raise SystemExit(f"missing cache {cache} — run experiments/record_replay.py --real first")
    model = AnthropicModel(model=model_id, effort=rr.EFFORT, cache_path=cache)
    card, jr = score_real_model(
        model=model, problems=problems, n_trials=rr.REAL_N_TRIALS, cache_path=cache,
    )
    d = card.to_dict()
    if jr is not None:
        d["judge_reliability"] = jr
    return d


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)

    cards = [_real_card_dict(m, "mixed", rr.real_problems()) for m in rr.REAL_MODELS]
    card_path = REPORT / "real_model_report.html"
    write_card_report(
        cards, str(card_path),
        title="Real-model Faithfulness Cards — Claude Sonnet 4.6 & Opus 4.8",
        subtitle=_SUBTITLE,
    )
    print(f"Wrote {card_path.relative_to(ROOT)}")

    transfer = build_gsm8k_transfer(
        cache_path=str(rr.real_cache("claude-sonnet-4-6", "gsm8k")),
        gsm8k_sample=str(rr.GSM8K_SAMPLE),
        model="claude-sonnet-4-6", effort=rr.EFFORT, n_trials=rr.REAL_N_TRIALS,
    )
    transfer_path = REPORT / "transfer.html"
    write_transfer_html(transfer, str(transfer_path))
    print(f"Wrote {transfer_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
