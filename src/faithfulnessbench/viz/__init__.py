"""Dependency-free SVG charts and HTML report rendering."""
from __future__ import annotations

from .svg import bar_chart, correlation_color, faithfulness_color, heatmap, line_chart, roc_plot

__all__ = ["bar_chart", "heatmap", "roc_plot", "line_chart", "faithfulness_color", "correlation_color"]
