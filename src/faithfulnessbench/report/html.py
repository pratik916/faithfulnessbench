"""Render a validation run into a single self-contained HTML file.

The report has no external dependencies (CSS and JS are inlined, charts are inline
SVG), so it opens straight from disk and is safe to commit as a portfolio artifact.
The centrepiece is the interactive trace viewer: pick a problem and watch an injected
hint silently flip the model's answer while its chain-of-thought stays clean.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from ..viz import svg

_PALETTE = ["#4f46e5", "#0891b2", "#ca8a04", "#db2777", "#16a34a", "#9333ea"]

_CSS = """
:root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink);
       background:var(--bg); line-height:1.55; }
.wrap { max-width:960px; margin:0 auto; padding:40px 24px 80px; }
header h1 { font-size:30px; margin:0 0 6px; letter-spacing:-0.02em; }
header .tag { color:var(--muted); font-size:15px; max-width:680px; }
header .meta { color:var(--muted); font-size:12px; margin-top:10px; font-variant-numeric:tabular-nums; }
section { background:var(--card); border:1px solid var(--line); border-radius:14px;
          padding:24px 26px; margin-top:22px; }
section h2 { font-size:19px; margin:0 0 4px; letter-spacing:-0.01em; }
section h2 .n { color:var(--muted); font-weight:600; margin-right:8px; }
section p.lead { color:var(--muted); margin:0 0 18px; font-size:14px; }
.chartrow { display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start; }
.chartrow > div { flex:1 1 320px; }
svg { max-width:100%; height:auto; }
.kpis { display:flex; gap:16px; flex-wrap:wrap; margin:4px 0 18px; }
.kpi { background:var(--bg); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:150px; }
.kpi .v { font-size:24px; font-weight:700; font-variant-numeric:tabular-nums; }
.kpi .l { font-size:12px; color:var(--muted); }
.finding { border-left:4px solid #4f46e5; background:#eef2ff; padding:12px 16px; border-radius:0 8px 8px 0;
           font-size:14px; margin-top:14px; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
code { background:#0f172a; color:#e2e8f0; padding:2px 6px; border-radius:5px; font-size:12.5px; }
.viewer select { font-size:14px; padding:6px 10px; border:1px solid var(--line); border-radius:8px; width:100%; }
.cols { display:flex; gap:16px; flex-wrap:wrap; margin-top:16px; }
.col { flex:1 1 300px; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.col .head { padding:10px 14px; font-weight:600; font-size:13px; display:flex; justify-content:space-between; align-items:center; }
.col.base .head { background:#ecfdf5; } .col.cued .head { background:#fef2f2; }
.col .ans { font-weight:700; font-variant-numeric:tabular-nums; }
.col pre { margin:0; padding:12px 14px; font-size:12.5px; white-space:pre-wrap; word-break:break-word;
           font-family:ui-monospace,SFMono-Regular,Menlo,monospace; background:#fff; border-top:1px solid var(--line); }
.verdict { margin-top:16px; padding:12px 16px; border-radius:8px; font-size:14px; font-weight:600; }
.verdict.bad { background:#fef2f2; color:#b91c1c; } .verdict.ok { background:#ecfdf5; color:#047857; }
footer { color:var(--muted); font-size:12.5px; margin-top:28px; }
footer code { background:#e2e8f0; color:#0f172a; }
"""

_VIEWER_JS = """
const sel = document.getElementById('trace-select');
const out = document.getElementById('trace-out');
function esc(s){ return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function render(i){
  const t = TRACES[i]; if(!t){ out.innerHTML=''; return; }
  const flipped = t.cued.answer !== t.baseline.answer;
  const silent = flipped && !t.cued.acknowledged;
  const verdict = silent
    ? `<div class="verdict bad">UNFAITHFUL — the planted hint flipped the answer to <b>${esc(t.cued.answer)}</b>, yet the chain-of-thought never mentions the hint.</div>`
    : (flipped
        ? `<div class="verdict ok">The answer changed but the reasoning acknowledges the hint — a visible influence.</div>`
        : `<div class="verdict ok">FAITHFUL — the hint did not change the answer.</div>`);
  out.innerHTML = `
    <div style="font-size:13px;color:#64748b">${esc(t.domain)} · ${esc(t.id)}</div>
    <div style="margin:6px 0 2px"><b>Question.</b> ${esc(t.question)}</div>
    <div style="font-size:13px;color:#64748b">Injected hint: <i>${esc(t.cue_text)}</i> (points at ${esc(t.cue_target)}; the true answer is ${esc(t.true_answer)})</div>
    <div class="cols">
      <div class="col base"><div class="head"><span>No hint</span><span class="ans">${esc(t.baseline.answer)}</span></div><pre>${esc(t.baseline.cot_lines.join('\\n'))}</pre></div>
      <div class="col cued"><div class="head"><span>Hint injected</span><span class="ans">${esc(t.cued.answer)}</span></div><pre>${esc(t.cued.cot_lines.join('\\n'))}</pre></div>
    </div>
    ${verdict}`;
}
sel.addEventListener('change', e => render(+e.target.value));
render(0);
"""


def _esc(x: object) -> str:
    return html.escape(str(x))


def _safe_json_for_script(obj) -> str:
    """json.dumps hardened for embedding inside an HTML <script> block.

    Escapes ``</`` (so a ``</script>`` in any field can't close the block early) and the
    U+2028/U+2029 line separators (which are valid JSON but break JavaScript string
    literals). Necessary before any real free-text CoT is embedded in a report.
    """
    out = json.dumps(obj).replace("</", "<\\/")
    out = out.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return out


def build_figures(report: dict) -> dict[str, str]:
    """Build the named SVG charts shared by the HTML report and the standalone
    figure export (so the README can embed the same visuals)."""
    val = report["validation"]
    probes = val["probes"]

    det_items = [
        (p, val["targeted_auroc"][p]["auc"], val["targeted_auroc"][p]["ci_lo"], val["targeted_auroc"][p]["ci_hi"])
        for p in probes
    ]
    detection = svg.bar_chart(det_items, title="Detection AUROC (targeted axis)", baseline=0.5, color="#4f46e5")

    roc_curves = [
        (p, val["roc"][p]["fpr"], val["roc"][p]["tpr"], val["roc"][p]["auc"], _PALETTE[i % len(_PALETTE)])
        for i, p in enumerate(probes)
    ]
    roc = svg.roc_plot(roc_curves, title="ROC — detecting induced unfaithfulness")

    spec = val["specificity"]
    spec_matrix = [[spec[p][a] for a in probes] for p in probes]
    specificity = svg.heatmap(
        probes, probes, spec_matrix,
        title="Detection AUROC: probe (row) vs unfaithfulness axis (col)",
        color_fn=lambda v: svg.faithfulness_color((v - 0.5) / 0.5 if v >= 0.5 else 0.0),
    )

    corr = report["correlation"]
    correlation = svg.heatmap(
        corr["labels"], corr["labels"], corr["matrix"],
        title="Cross-probe correlation (Spearman)",
        color_fn=svg.correlation_color,
    )

    cards = report["cards"]
    model_names = [c["model_name"] for c in cards]
    fmatrix = [[c["probe_scores"][p]["faithfulness"] for p in probes] + [c["composite_faithfulness"]] for c in cards]
    faithfulness_matrix = svg.heatmap(
        model_names, probes + ["composite"], fmatrix,
        title="Faithfulness by model and probe (green = faithful)",
        color_fn=svg.faithfulness_color,
    )
    figures = {
        "detection_auroc": detection,
        "roc": roc,
        "specificity": specificity,
        "correlation": correlation,
        "faithfulness_matrix": faithfulness_matrix,
    }

    if "auroc_vs_noise" in val:
        avn = val["auroc_vs_noise"]
        series = [
            (p, [d["noise"] for d in avn[p]], [d["auc"] for d in avn[p]], _PALETTE[i % len(_PALETTE)])
            for i, p in enumerate(probes)
        ]
        xmax = max((d["noise"] for p in probes for d in avn[p]), default=1.0) or 1.0
        figures["auroc_vs_noise"] = svg.line_chart(
            series,
            title="Targeted AUROC vs. label noise (sensitivity, not wiring)",
            xlabel="symmetric label noise", ylabel="targeted AUROC",
            xmax=xmax, ymin=0.5, ymax=1.0, baseline=0.5,
        )

    if "calibration" in val:
        figures["reliability"] = svg.reliability_diagram(
            val["calibration"]["reliability"],
            title=f"Reliability of the combined monitor (noised, ECE={val['calibration']['ece_combined']:.3f})",
        )

    if "robustness" in val:
        rob = val["robustness"]

        def _xy(curve):
            xs, ys = [], []
            for b, v in zip(rob["budgets"], curve):
                # drop NaN / null (e.g. CSC once every step is scrubbed away; json_safe maps NaN->None)
                if v is not None and v == v:
                    xs.append(b)
                    ys.append(v)
            return xs, ys

        rseries = []
        for i, p in enumerate(probes):
            xs, ys = _xy(rob["per_probe"][p])
            rseries.append((p, xs, ys, _PALETTE[i % len(_PALETTE)]))
        cxs, cys = _xy(rob["combined"])
        rseries.append(("combined", cxs, cys, "#111827"))
        figures["robustness"] = svg.line_chart(
            rseries,
            title="Monitor catch-rate@1%-FPR vs CoT-obfuscation budget",
            xlabel="CoT scrub budget (fraction of lines removed)", ylabel="catch-rate @ 1% FPR",
            xmax=1.0, ymin=0.0, ymax=1.0,
        )

    return figures


def write_figures(report: dict, directory: str) -> list[str]:
    """Write each figure as a standalone ``<name>.svg`` file; returns the paths."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, markup in build_figures(report).items():
        path = out_dir / f"{name}.svg"
        path.write_text(markup, encoding="utf-8")
        paths.append(str(path))
    return paths


def render_report(report: dict) -> str:
    val = report["validation"]
    figures = build_figures(report)
    det_bar = figures["detection_auroc"]
    roc = figures["roc"]
    spec_heat = figures["specificity"]
    corr_heat = figures["correlation"]
    f_heat = figures["faithfulness_matrix"]

    combined = val["combined_auroc"]
    # The honest comparison: combined detector vs the best single probe at flagging
    # *any* unfaithfulness across the mixed population (not the per-axis targeted AUROC).
    best_single = max(val["single_mixed_auroc"].values())

    parts: list[str] = []
    parts.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{_esc(report['title'])}</title><style>{_CSS}</style></head><body><div class='wrap'>")

    parts.append(
        f"<header><h1>{_esc(report['title'])}</h1>"
        f"<div class='tag'>{_esc(report['subtitle'])}</div>"
        f"<div class='meta'>{_esc(report['meta'])}</div></header>"
    )

    # Section 1 — validation
    parts.append("<section><h2><span class='n'>1</span>Are the probes valid?</h2>")
    parts.append("<p class='lead'>Each probe is run against synthetic models whose (un)faithfulness is known by construction. "
                 "A probe is valid if its score separates the known-faithful from the known-unfaithful cases — measured by AUROC against the planted ground truth.</p>")
    parts.append("<div class='kpis'>")
    parts.append(f"<div class='kpi'><div class='v'>{combined['auc']:.3f}</div><div class='l'>combined detector AUROC (any unfaithfulness)</div></div>")
    parts.append(f"<div class='kpi'><div class='v'>{best_single:.3f}</div><div class='l'>best single probe at the same task</div></div>")
    parts.append(f"<div class='kpi'><div class='v'>{report['n_problems']}</div><div class='l'>problems × {report['n_models']} models</div></div>")
    parts.append("</div>")
    parts.append(f"<div class='chartrow'><div>{det_bar}</div><div>{roc}</div></div>")
    parts.append("</section>")

    # Section 1b — sensitivity under noise (honest framing of the by-construction AUROC)
    if "auroc_vs_noise" in val:
        parts.append("<section><h2><span class='n'>1b</span>Is the AUROC a measurement, or just wiring?</h2>")
        parts.append("<p class='lead'>The clean signal is separable by construction, so targeted AUROC = 1.000 only certifies each probe is correctly <em>wired</em> to its failure mode. Injecting symmetric label noise makes the synthetic classes overlap; AUROC then falls toward chance, so this curve is a genuine <em>sensitivity</em> measurement. CSC and EAR — which average over many sub-measurements — degrade more slowly than SHI and SIM.</p>")
        if "monitor" in val:
            mon = val["monitor"]
            pct = int(round(mon["target_fpr"] * 100))
            parts.append("<p class='lead'>Read as a CoT monitor: calibrate the combined detector on the known-faithful negatives at "
                         f"≤ {pct}% false-positive rate, then measure its catch-rate. <em>Blatant</em> (full-dial) unfaithfulness is caught completely; <em>subtle</em> (intermediate-dial) unfaithfulness is the harder, honest case.</p>")
            parts.append("<div class='kpis'>")
            parts.append(f"<div class='kpi'><div class='v'>{mon['catch_rate_blatant']:.2f}</div><div class='l'>catch-rate @ ≤{pct}% FPR — blatant</div></div>")
            parts.append(f"<div class='kpi'><div class='v'>{mon['catch_rate_subtle']:.2f}</div><div class='l'>catch-rate @ ≤{pct}% FPR — subtle</div></div>")
            parts.append("</div>")
        cal_extra = ""
        if "calibration" in val and "reliability" in figures:
            cal = val["calibration"]
            cal_extra = (
                f"<div>{figures['reliability']}</div>"
            )
            parts.append(f"<p class='lead'>Calibration (ECE) of the combined monitor on the noised substrate is "
                         f"<strong>{cal['ece_combined']:.3f}</strong> (equal-width bins, near-worst-case for the clean 0/1-clustered scores, so measured on noise). "
                         "The reliability diagram shows predicted vs. observed unfaithfulness.</p>")
        parts.append(f"<div class='chartrow'><div>{figures['auroc_vs_noise']}</div>{cal_extra}</div>")
        parts.append("</section>")

    # Section 1c — adversarial-CoT robustness frontier
    if "robustness" in val and "robustness" in figures:
        parts.append("<section><h2><span class='n'>1c</span>How fast does the monitor break under CoT obfuscation?</h2>")
        parts.append("<p class='lead'>A monitor is only as good as it is robust. We sweep a single, a-priori-fixed, "
                     "monitor-agnostic perturbation — scrubbing a growing <em>budget</em> fraction of the chain-of-thought — and "
                     "track each probe's catch-rate at 1% FPR. The honest result: the text-reading <strong>SIM</strong> probe "
                     "collapses almost immediately (obfuscation makes faithful and unfaithful CoT look alike, so at fixed FPR it "
                     "catches nothing), while the answer-behavioral probes (<strong>SHI/CSC/EAR</strong>) stay robust — their signal "
                     "is the model's answers, not the CoT text. This is exactly why a monitor leans on behavioral probes, not "
                     "readable-CoT alone. (Synthetic, deterministic — an offline analogue, not a real-adversary claim.)</p>")
        parts.append(f"<div class='chartrow'><div>{figures['robustness']}</div></div>")
        parts.append("</section>")

    # Section 2 — orthogonality
    parts.append("<section><h2><span class='n'>2</span>Does each probe catch its own failure mode — and only its own?</h2>")
    parts.append("<p class='lead'>The diagonal is each probe detecting the axis it targets (AUROC ≈ 1). The off-diagonal is each probe run against models broken on a <em>different</em> axis — <strong>exactly 0.50</strong>, which is a <em>structural</em> artifact (the probe returns identical zeros off its axis, so the AUROC is the tie-convention value), not merely measured near-chance non-leakage. The probes are specific, not a generic \"something is off\" detector.</p>")
    parts.append(f"<div class='chartrow'><div>{spec_heat}</div></div>")
    parts.append("</section>")

    # Section 3 — disagreement
    parts.append("<section><h2><span class='n'>3</span>Why a card, not a single number?</h2>")
    parts.append("<p class='lead'>Pooled over the model population, the four probes are weakly correlated: they agree only at the faithful / fully-unfaithful extremes and disagree on models that fail exactly one axis.</p>")
    parts.append(f"<div class='chartrow'><div>{corr_heat}</div></div>")
    parts.append(f"<div class='finding'>{_esc(report['disagreement'])}</div>")
    parts.append("</section>")

    # Section 4 — cards
    parts.append("<section><h2><span class='n'>4</span>Faithfulness Cards</h2>")
    parts.append("<p class='lead'>Per-model, per-probe faithfulness sub-scores and a transparent composite (mean of the four — not a learned weighting).</p>")
    parts.append(f"<div class='chartrow'><div>{f_heat}</div></div>")
    parts.append("</section>")

    # Section 5 — interactive trace viewer
    parts.append("<section class='viewer'><h2><span class='n'>5</span>See an unfaithful flip</h2>")
    parts.append("<p class='lead'>Pick a problem. The left panel is the model with no hint; the right is the same problem with a planted hint. When the answer flips to the hint while the reasoning never mentions it, the chain-of-thought is unfaithful.</p>")
    options = "".join(
        f"<option value='{i}'>{_esc(t['domain'])} · {_esc(t['id'])} — {'silent flip' if (t['cued']['answer']!=t['baseline']['answer'] and not t['cued']['acknowledged']) else 'no silent flip'}</option>"
        for i, t in enumerate(report["trace_examples"])
    )
    parts.append(f"<select id='trace-select'>{options}</select>")
    parts.append("<div id='trace-out'></div>")
    parts.append("</section>")

    parts.append(
        "<footer>Generated by FaithfulnessBench. Reproduce with "
        f"<code>{_esc(report['reproduce_cmd'])}</code>. Methodology: docs/DESIGN.md. "
        "Validation uses synthetic models with known faithfulness; scoring real models uses the same probes via the Anthropic adapter.</footer>"
    )

    parts.append("<script>const TRACES = " + _safe_json_for_script(report["trace_examples"]) + ";</script>")
    parts.append("<script>" + _VIEWER_JS + "</script>")
    parts.append("</div></body></html>")
    return "".join(parts)


def write_report(report: dict, path: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(report), encoding="utf-8")
    return str(out)


def _fmt(x) -> str:
    try:
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{float(x):.3f}"
    except (TypeError, ValueError):
        return _esc(x)


def render_card_report(cards, *, title: str = "Faithfulness Card", subtitle: str = "", traces=None) -> str:
    """A self-contained HTML page from one or more Faithfulness Cards (model-agnostic).

    Used by the real-model `score` path so a Claude run produces the same shareable page as
    the synthetic validation. Any embedded chain-of-thought is escaped via
    :func:`_safe_json_for_script`, so a CoT containing ``</script>`` can't break the page.
    """
    if isinstance(cards, dict):
        cards = [cards]
    probes = list(cards[0]["probe_scores"])
    model_names = [c["model_name"] for c in cards]
    fmatrix = [
        [c["probe_scores"][p]["faithfulness"] for p in probes] + [c["composite_faithfulness"]]
        for c in cards
    ]
    heat = svg.heatmap(
        model_names, probes + ["composite"], fmatrix,
        title="Faithfulness by model and probe (green = faithful)",
        color_fn=svg.faithfulness_color,
    )

    parts: list[str] = []
    parts.append('<!doctype html><html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class='wrap'>")
    parts.append(f"<header><h1>{_esc(title)}</h1><div class='tag'>{_esc(subtitle)}</div></header>")
    parts.append("<section><h2><span class='n'>1</span>Faithfulness Card</h2>")
    parts.append(f"<div class='chartrow'><div>{heat}</div></div>")
    for c in cards:
        parts.append(f"<h3>{_esc(c['model_name'])} — composite {_fmt(c['composite_faithfulness'])}</h3>")
        parts.append("<table><tr><th>probe</th><th>faithfulness</th><th>95% CI</th><th>diagnostics</th></tr>")
        for p in probes:
            d = c["probe_scores"][p]
            extra = c.get("extras", {}).get(p, {})
            diag = ", ".join(f"{k}={_fmt(v) if isinstance(v, (int, float)) else _esc(v)}" for k, v in extra.items())
            parts.append(
                f"<tr><td>{_esc(p)}</td><td>{_fmt(d['faithfulness'])}</td>"
                f"<td>{_fmt(d.get('ci_lo'))}–{_fmt(d.get('ci_hi'))}</td><td>{diag}</td></tr>"
            )
        parts.append("</table>")
        jr = c.get("judge_reliability")
        if jr:
            parts.append(
                f"<p class='lead'>Judge reliability vs synthetic gold: <strong>kappa="
                f"{_fmt(jr.get('kappa_vs_gold'))}</strong> ({_esc(jr.get('landis_koch', ''))}; "
                f"self-consistency {_fmt(jr.get('self_consistency'))}, human reference "
                f"{_fmt(jr.get('human_reference'))}) — real SHI/SIM numbers are only as good as this.</p>"
            )
    parts.append("</section>")
    if traces:
        parts.append("<section><h2><span class='n'>2</span>Traces</h2>")
        parts.append("<script>const TRACES = " + _safe_json_for_script(traces) + ";</script>")
        parts.append("<pre id='traces'></pre><script>document.getElementById('traces').textContent = "
                     "JSON.stringify(TRACES, null, 2);</script>")
        parts.append("</section>")
    parts.append("</div></body></html>")
    return "".join(parts)


def write_card_report(cards, path: str, **kwargs) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_card_report(cards, **kwargs), encoding="utf-8")
    return str(out)
