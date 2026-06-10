"""The GitHub Pages deploy publishes the self-contained report; the README links it (fb-0tm.2).

Dependency-free (no PyYAML in the dev extra): structural text checks on the workflow plus the
indentation sanity that every top-level key is column-0, so the YAML can't silently break.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / ".github" / "workflows" / "pages.yml"


def test_pages_workflow_publishes_the_report_with_correct_permissions():
    text = PAGES.read_text()
    assert "actions/upload-pages-artifact" in text
    assert "actions/deploy-pages" in text
    assert "report/faithfulness_report.html" in text and "_site/index.html" in text
    # GitHub Pages via Actions requires these token permissions or the deploy 403s.
    assert "pages: write" in text and "id-token: write" in text


def test_pages_workflow_has_no_obvious_indentation_breakage():
    # Every line is either blank, a comment, indented, or a column-0 mapping key "<word>:".
    import re

    for ln in PAGES.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#") or ln[0] in " \t":
            continue
        assert re.match(r"^[A-Za-z_][\w-]*:", ln), f"bad top-level line: {ln!r}"


def test_readme_links_the_live_pages_demo():
    rd = (ROOT / "README.md").read_text()
    assert "pratik916.github.io/faithfulnessbench" in rd
    assert "live interactive report" in rd.lower()


def test_readme_embeds_the_committed_trace_viewer_gif():
    gif = ROOT / "docs" / "assets" / "trace_viewer.gif"
    assert gif.exists() and gif.stat().st_size > 1000  # real animated file, committed
    assert "docs/assets/trace_viewer.gif" in (ROOT / "README.md").read_text()
