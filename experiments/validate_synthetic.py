#!/usr/bin/env python3
"""Headline experiment: validate the four faithfulness probes against the synthetic
ground-truth population, then write results.json + the self-contained HTML report.

Run from the repo root (no API key required, fully deterministic):

    python experiments/validate_synthetic.py
    # or, after `pip install -e .`:
    faithfulnessbench validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faithfulnessbench.report import write_figures, write_report  # noqa: E402
from faithfulnessbench.validation import json_safe, run_validation, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=20, help="problems per domain")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--json", default=str(ROOT / "experiments" / "results" / "results.json"))
    ap.add_argument("--report", default=str(ROOT / "report" / "faithfulness_report.html"))
    ap.add_argument("--figures", default=str(ROOT / "docs" / "assets"))
    args = ap.parse_args()

    report = run_validation(
        n_per_domain=args.n, seed=args.seed, n_trials=args.trials,
        reproduce_cmd="python experiments/validate_synthetic.py",
    )
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(json_safe(report), indent=2))
    write_report(report, args.report)
    write_figures(report, args.figures)

    print(summarize(report))
    print(f"\nWrote results -> {args.json}")
    print(f"Wrote report  -> {args.report}")
    print(f"Wrote figures -> {args.figures}/*.svg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
