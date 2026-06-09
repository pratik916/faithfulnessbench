"""Hand-rolled SVG charts — no matplotlib, no external assets.

Each function returns a self-contained ``<svg>...</svg>`` string that can be embedded
directly into HTML. We implement only the few chart types the report needs (bars with
optional CIs, overlaid ROC curves, and labelled heatmaps).
"""
from __future__ import annotations

import html
import math
from typing import Callable, Sequence


def _esc(text: object) -> str:
    return html.escape(str(text))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _rgb(r: float, g: float, b: float) -> str:
    return f"rgb({int(round(r))},{int(round(g))},{int(round(b))})"


def faithfulness_color(v: float) -> str:
    """0 -> red, 0.5 -> amber, 1 -> green (NaN -> grey)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "#cbd5e1"
    v = max(0.0, min(1.0, v))
    red, amber, green = (220, 38, 38), (245, 158, 11), (22, 163, 74)
    if v < 0.5:
        t = v / 0.5
        return _rgb(*(_lerp(red[i], amber[i], t) for i in range(3)))
    t = (v - 0.5) / 0.5
    return _rgb(*(_lerp(amber[i], green[i], t) for i in range(3)))


def correlation_color(v: float) -> str:
    """-1 -> red, 0 -> white, 1 -> blue (NaN -> grey)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "#e5e7eb"
    v = max(-1.0, min(1.0, v))
    white = (255, 255, 255)
    if v >= 0:
        blue = (37, 99, 235)
        return _rgb(*(_lerp(white[i], blue[i], v) for i in range(3)))
    red = (220, 38, 38)
    return _rgb(*(_lerp(white[i], red[i], -v) for i in range(3)))


def bar_chart(
    items: Sequence[tuple],
    *,
    title: str = "",
    vmax: float = 1.0,
    baseline: float | None = None,
    width: int = 640,
    height: int = 320,
    color: str = "#4f46e5",
    value_fmt: str = "{:.2f}",
) -> str:
    """Vertical bars. Each item is (label, value) or (label, value, ci_lo, ci_hi)."""
    pad_l, pad_r, pad_t, pad_b = 48, 16, 36, 56
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(items)
    slot = plot_w / max(n, 1)
    bw = slot * 0.6

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - max(0.0, min(vmax, v)) / vmax)

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    if title:
        parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="600">{_esc(title)}</text>')
    # y gridlines at 0, .25, .5, .75, 1 (scaled by vmax)
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gv = vmax * frac
        gy = y(gv)
        parts.append(f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width-pad_r}" y2="{gy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l-6}" y="{gy+3:.1f}" text-anchor="end" font-size="10" fill="#888">{gv:.2f}</text>')
    if baseline is not None:
        by = y(baseline)
        parts.append(f'<line x1="{pad_l}" y1="{by:.1f}" x2="{width-pad_r}" y2="{by:.1f}" stroke="#ef4444" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{width-pad_r}" y="{by-4:.1f}" text-anchor="end" font-size="9" fill="#ef4444">chance {baseline:.2f}</text>')
    for i, item in enumerate(items):
        label, value = item[0], item[1]
        cx = pad_l + slot * i + slot / 2
        bx = cx - bw / 2
        top = y(value)
        parts.append(f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{pad_t+plot_h-top:.1f}" fill="{color}" rx="2"/>')
        if len(item) >= 4:
            lo, hi = item[2], item[3]
            ylo, yhi = y(lo), y(hi)
            parts.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" stroke="#111" stroke-width="1.2"/>')
            for yy in (ylo, yhi):
                parts.append(f'<line x1="{cx-4:.1f}" y1="{yy:.1f}" x2="{cx+4:.1f}" y2="{yy:.1f}" stroke="#111" stroke-width="1.2"/>')
        parts.append(f'<text x="{cx:.1f}" y="{top-6:.1f}" text-anchor="middle" font-size="11" font-weight="600">{value_fmt.format(value)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{height-pad_b+16:.1f}" text-anchor="middle" font-size="11">{_esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def roc_plot(
    curves: Sequence[tuple],
    *,
    title: str = "",
    width: int = 440,
    height: int = 420,
) -> str:
    """Overlay ROC curves. Each curve is (label, fpr_list, tpr_list, auc, color)."""
    pad = 48
    plot = min(width, height) - 2 * pad

    def px(fpr: float) -> float:
        return pad + fpr * plot

    def py(tpr: float) -> float:
        return pad + (1 - tpr) * plot

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    if title:
        parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="600">{_esc(title)}</text>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="none" stroke="#ddd"/>')
    parts.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#cbd5e1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{pad+plot/2}" y="{height-12}" text-anchor="middle" font-size="11" fill="#555">false positive rate</text>')
    parts.append(f'<text x="14" y="{pad+plot/2}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90 14 {pad+plot/2})">true positive rate</text>')
    legend_y = pad + 8
    for label, fpr, tpr, auc, color in curves:
        pts = " ".join(f"{px(f):.1f},{py(t):.1f}" for f, t in zip(fpr, tpr))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{px(1)-6:.1f}" y="{legend_y:.1f}" text-anchor="end" font-size="11" fill="{color}">{_esc(label)} (AUC {auc:.2f})</text>')
        legend_y += 16
    parts.append("</svg>")
    return "".join(parts)


def line_chart(
    series: Sequence[tuple],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    xmax: float = 1.0,
    ymin: float = 0.0,
    ymax: float = 1.0,
    baseline: float | None = None,
    width: int = 440,
    height: int = 420,
) -> str:
    """Overlay line series. Each series is ``(label, x_list, y_list, color)``."""
    pad = 52
    plot = min(width, height) - 2 * pad

    def px(x: float) -> float:
        return pad + (x / xmax) * plot if xmax else pad

    def py(y: float) -> float:
        span = (ymax - ymin) or 1.0
        return pad + (1 - (y - ymin) / span) * plot

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    if title:
        parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="600">{_esc(title)}</text>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="none" stroke="#ddd"/>')
    for yv in (ymin, (ymin + ymax) / 2, ymax):
        parts.append(f'<line x1="{pad}" y1="{py(yv):.1f}" x2="{pad+plot}" y2="{py(yv):.1f}" stroke="#f1f5f9"/>')
        parts.append(f'<text x="{pad-6}" y="{py(yv)+3:.1f}" text-anchor="end" font-size="10" fill="#888">{yv:.2f}</text>')
    if baseline is not None:
        parts.append(f'<line x1="{pad}" y1="{py(baseline):.1f}" x2="{pad+plot}" y2="{py(baseline):.1f}" stroke="#cbd5e1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{pad}" y="{pad+plot+16:.1f}" text-anchor="middle" font-size="10" fill="#888">0</text>')
    parts.append(f'<text x="{pad+plot}" y="{pad+plot+16:.1f}" text-anchor="middle" font-size="10" fill="#888">{xmax:g}</text>')
    if xlabel:
        parts.append(f'<text x="{pad+plot/2}" y="{height-12}" text-anchor="middle" font-size="11" fill="#555">{_esc(xlabel)}</text>')
    if ylabel:
        parts.append(f'<text x="14" y="{pad+plot/2}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90 14 {pad+plot/2})">{_esc(ylabel)}</text>')
    legend_y = pad + 8
    for label, xs, ys, color in series:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{color}"/>')
        parts.append(f'<text x="{pad+plot-6:.1f}" y="{legend_y:.1f}" text-anchor="end" font-size="11" fill="{color}">{_esc(label)}</text>')
        legend_y += 16
    parts.append("</svg>")
    return "".join(parts)


def reliability_diagram(reliability: dict, *, title: str = "", width: int = 440, height: int = 420) -> str:
    """Reliability diagram: per-bin observed accuracy vs. the perfect-calibration diagonal."""
    acc = reliability["accuracy"]
    count = reliability["count"]
    n_bins = len(acc)
    pad = 52
    plot = min(width, height) - 2 * pad

    def px(x: float) -> float:
        return pad + x * plot

    def py(y: float) -> float:
        return pad + (1 - y) * plot

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    if title:
        parts.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="600">{_esc(title)}</text>')
    parts.append(f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="none" stroke="#ddd"/>')
    parts.append(f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" stroke="#cbd5e1" stroke-dasharray="4 3"/>')
    bw = plot / n_bins
    for i in range(n_bins):
        a = acc[i]
        if count[i] == 0 or a != a:  # skip empty / nan bins
            continue
        x0 = pad + i * bw
        parts.append(f'<rect x="{x0:.1f}" y="{py(a):.1f}" width="{bw-1:.1f}" height="{a*plot:.1f}" fill="#4f46e5" opacity="0.7"/>')
    parts.append(f'<text x="{pad+plot/2}" y="{height-12}" text-anchor="middle" font-size="11" fill="#555">predicted unfaithfulness (confidence)</text>')
    parts.append(f'<text x="14" y="{pad+plot/2}" text-anchor="middle" font-size="11" fill="#555" transform="rotate(-90 14 {pad+plot/2})">observed fraction unfaithful</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    matrix: Sequence[Sequence[float]],
    *,
    title: str = "",
    color_fn: Callable[[float], str] = faithfulness_color,
    cell: int = 64,
) -> str:
    """Labelled heatmap with the numeric value drawn in each cell."""
    pad_l, pad_t = 120, 56
    rows, cols = len(row_labels), len(col_labels)
    width = pad_l + cols * cell + 16
    height = pad_t + rows * cell + 16
    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    if title:
        parts.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" font-weight="600">{_esc(title)}</text>')
    for j, cl in enumerate(col_labels):
        cx = pad_l + j * cell + cell / 2
        parts.append(f'<text x="{cx:.1f}" y="{pad_t-8}" text-anchor="middle" font-size="11" font-weight="600">{_esc(cl)}</text>')
    for i, rl in enumerate(row_labels):
        cy = pad_t + i * cell + cell / 2
        parts.append(f'<text x="{pad_l-8}" y="{cy+4:.1f}" text-anchor="end" font-size="11">{_esc(rl)}</text>')
        for j in range(cols):
            v = float(matrix[i][j])
            x = pad_l + j * cell
            y = pad_t + i * cell
            disp = "—" if math.isnan(v) else f"{v:.2f}"
            txt_fill = "#111" if (math.isnan(v) or abs(v) < 0.6) else "#fff"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{color_fn(v)}" rx="3"/>')
            parts.append(f'<text x="{x+cell/2-1:.1f}" y="{y+cell/2+4:.1f}" text-anchor="middle" font-size="12" fill="{txt_fill}">{disp}</text>')
    parts.append("</svg>")
    return "".join(parts)
