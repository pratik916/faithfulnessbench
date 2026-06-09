#!/usr/bin/env python3
"""Guard against the advertised test count drifting from reality.

The README shields badge and CLAUDE.md both state how many tests the suite has.
This script (and the matching tests in ``tests/test_repo_consistency.py``) fail if
that number disagrees with what ``pytest`` actually collects, so the "N passing"
claim stays honest as tests are added or removed.

Usage (CI):  ``python tools/check_test_count.py``  -> exit 0 if consistent, 1 on drift.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_BADGE_RE = re.compile(r"tests-(\d+)%20passing")
_CLAUDE_RE = re.compile(r"full suite \((\d+) tests")


def badge_count(readme_text: str) -> int:
    """The test count claimed by the README shields badge."""
    m = _BADGE_RE.search(readme_text)
    if not m:
        raise ValueError("could not find the tests shields badge in README text")
    return int(m.group(1))


def claude_md_count(claude_text: str) -> int:
    """The test count claimed by CLAUDE.md's `# full suite (N tests ...)` note."""
    m = _CLAUDE_RE.search(claude_text)
    if not m:
        raise ValueError("could not find the `full suite (N tests ...)` note in CLAUDE.md")
    return int(m.group(1))


def collected_test_count(root: Path = ROOT) -> int:
    """How many tests ``pytest`` collects, parsed robustly across output formats."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=str(root), capture_output=True, text=True,
    )
    text = proc.stdout
    # 1) flat node ids (e.g. "tests/test_x.py::test_y")
    nodes = re.findall(r"^\S+::\S+", text, re.M)
    if nodes:
        return len(nodes)
    # 2) per-file quiet summary (e.g. "tests/test_x.py: 12")
    per_file = re.findall(r"^\S+: (\d+)$", text, re.M)
    if per_file:
        return sum(int(n) for n in per_file)
    # 3) terminal summary (e.g. "57 tests collected")
    m = re.search(r"(\d+) tests? collected", text)
    if m:
        return int(m.group(1))
    raise RuntimeError(
        "could not parse pytest --collect-only output:\n"
        f"--- stdout ---\n{text}\n--- stderr ---\n{proc.stderr}"
    )


def main() -> int:
    collected = collected_test_count()
    readme = (ROOT / "README.md").read_text()
    claude = (ROOT / "CLAUDE.md").read_text()
    ok = True
    for name, stated in (("README badge", badge_count(readme)), ("CLAUDE.md", claude_md_count(claude))):
        if stated != collected:
            print(f"DRIFT: {name} states {stated} tests but pytest collects {collected}", file=sys.stderr)
            ok = False
    if not ok:
        print("Fix: update the README badge and CLAUDE.md to the live collected count.", file=sys.stderr)
        return 1
    print(f"OK: README badge and CLAUDE.md both state {collected} tests (matches pytest).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
