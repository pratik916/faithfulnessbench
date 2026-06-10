"""Chart accessibility: title/desc/aria, color-scale legend, label overflow (fb-tch.6)."""
from __future__ import annotations

from faithfulnessbench import viz
from faithfulnessbench.viz import svg


def test_correlation_color_importable_from_viz():
    assert callable(viz.correlation_color)


def test_charts_carry_a11y_metadata():
    bar = svg.bar_chart([("a", 0.5, 0.4, 0.6)], title="My Bars")
    assert 'role="img"' in bar and 'aria-label="My Bars"' in bar
    assert "<title>My Bars</title>" in bar and "<desc>" in bar


def test_heatmap_has_legend_and_truncates_long_labels():
    long_label = "an_extremely_long_model_name_that_overflows"
    out = svg.heatmap([long_label], ["c"], [[0.5]], title="H")
    assert "…" in out  # the long row label was truncated
    assert ">low<" in out and ">high<" in out  # the color-scale legend ends
    assert 'role="img"' in out
