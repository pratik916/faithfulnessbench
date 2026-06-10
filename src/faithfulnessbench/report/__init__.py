"""Self-contained HTML report generation."""
from __future__ import annotations

from .html import (
    build_figures,
    render_card_report,
    render_report,
    write_card_report,
    write_figures,
    write_report,
)

__all__ = [
    "render_report",
    "write_report",
    "build_figures",
    "write_figures",
    "render_card_report",
    "write_card_report",
]
