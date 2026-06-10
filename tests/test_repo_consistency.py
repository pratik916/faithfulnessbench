"""Repo-consistency guards.

The stated test count (README shields badge + CLAUDE.md) must never drift from the
number pytest actually collects. These tests fail the moment someone adds or removes a
test without updating the count, so the advertised "N passing" stays honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_test_count import badge_count, claude_md_count, collected_test_count  # noqa: E402


def test_dead_code_first_operand_left_is_removed():
    import faithfulnessbench.problems as problems

    assert not hasattr(problems, "first_operand_left")


def test_parse_badge_count_extracts_number():
    badge = "![tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)"
    assert badge_count(badge) == 42


def test_readme_tests_badge_matches_collected_count():
    collected = collected_test_count()
    stated = badge_count((ROOT / "README.md").read_text())
    assert stated == collected, (
        f"README tests badge says {stated} but pytest collects {collected}; "
        "update the badge (and CLAUDE.md) to the live collected count."
    )


def test_claude_md_states_collected_count():
    collected = collected_test_count()
    stated = claude_md_count((ROOT / "CLAUDE.md").read_text())
    assert stated == collected, (
        f"CLAUDE.md states {stated} tests but pytest collects {collected}."
    )
