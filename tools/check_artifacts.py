#!/usr/bin/env python3
"""Fail if the committed HTML report or SVG figures are stale vs results.json.

Rendering is deterministic, so the committed report/figures must match a fresh render
of the committed results.json. Run ``python experiments/validate_synthetic.py`` to
regenerate them.

Usage (CI):  ``python tools/check_artifacts.py``  -> exit 0 if fresh, 1 if stale.
"""
from __future__ import annotations

import sys
from pathlib import Path

from faithfulnessbench.artifacts import stale_artifacts

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    msgs = stale_artifacts(
        ROOT / "experiments" / "results" / "results.json",
        ROOT / "report" / "faithfulness_report.html",
        ROOT / "docs" / "assets",
    )
    if msgs:
        print("STALE ARTIFACTS (regenerate with `python experiments/validate_synthetic.py`):", file=sys.stderr)
        for m in msgs:
            print(f"  {m}", file=sys.stderr)
        return 1
    print("OK: the committed HTML report and SVG figures match results.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
