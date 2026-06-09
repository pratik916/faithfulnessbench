"""Self-contained HTML report generation."""
from __future__ import annotations

from .html import build_figures, render_report, write_figures, write_report

__all__ = ["render_report", "write_report", "build_figures", "write_figures"]
