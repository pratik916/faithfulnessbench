"""Dependency-free SVG charts and HTML report rendering."""
from __future__ import annotations

from .svg import bar_chart, faithfulness_color, heatmap, roc_plot

__all__ = ["bar_chart", "heatmap", "roc_plot", "faithfulness_color"]
