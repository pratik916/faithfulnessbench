"""Cross-domain transfer: does the validated probe battery run on real grade-school math?

The synthetic substrate gives exact faithfulness labels (probe AUROC is validated there).
GSM8K does **not** — so this module reports a *descriptive* comparison only, never an
AUROC-vs-truth on the GSM8K side. The point it makes honestly is twofold: (1) the identical
probe code runs unchanged on real grade-school math, and (2) the structured-CoT probes
(CSC/SIM) degrade on free-text CoT (no parseable ``L op R = V`` chain) — the documented
real-path limitation — while SHI/EAR keep running.

This is the cheap, honest answer to "does it generalize beyond linear arithmetic?". It is
framed against FaithCoT-Bench, which finds counterfactual methods succeed on math but fail to
transfer to knowledge domains: our GSM8K evidence is *in-domain* (still arithmetic) and
descriptive, so it does not claim cross-domain *validity* transfer. numpy-only; the GSM8K side
replays offline from the committed cache (a labeled fake until a real recording is swapped in).
"""
from __future__ import annotations

import math

FAITHCOT_SCOPE_NOTE = (
    "FaithCoT-Bench (arXiv:2510.04040) finds counterfactual faithfulness methods succeed on "
    "math but fail to transfer to knowledge domains. This GSM8K evidence is in-domain "
    "(arithmetic) and descriptive — it shows the probes RUN unchanged on real grade-school "
    "math, not that faithfulness *validity* transfers across domains."
)

_DESCRIPTIVE_NOTE = (
    "Descriptive cross-domain comparison. The synthetic-arithmetic side has exact ground-truth "
    "labels (probe AUROC is validated there); the GSM8K side has NO faithfulness ground truth, "
    "so only descriptive statistics are reported — never an AUROC-vs-truth. The identical probe "
    "code runs unchanged on real grade-school math; CSC/SIM degrade on free-text CoT (no "
    "parseable 'L op R = V' chain), the documented real-path limitation, while SHI/EAR run."
)

_CORE = ("SHI", "CSC", "SIM", "EAR")


def _summarize(card) -> dict:
    """Per-probe descriptive summary: faithfulness (None if degraded), a ``ran`` flag, and the
    probe's scalar diagnostics (list-valued ones like EAR's match curve are dropped)."""
    out: dict[str, dict] = {}
    for name, d in card.probe_scores.items():
        f = d["faithfulness"]
        ran = not (isinstance(f, float) and math.isnan(f))
        diagnostics = {
            k: float(v)
            for k, v in card.extras.get(name, {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        out[name] = {"faithfulness": (float(f) if ran else None), "ran": ran, "diagnostics": diagnostics}
    return out


def cross_domain_transfer(synthetic_card, gsm8k_card) -> dict:
    """Build the descriptive transfer comparison from an in-domain synthetic card and a
    GSM8K (real-math, cached) card."""
    return {
        "synthetic_arithmetic": _summarize(synthetic_card),
        "gsm8k_real": _summarize(gsm8k_card),
        "note": _DESCRIPTIVE_NOTE,
        "faithcot_scope": FAITHCOT_SCOPE_NOTE,
    }


def build_gsm8k_transfer(
    *,
    cache_path,
    gsm8k_sample,
    model: str = "claude-sonnet-4-6",
    effort: str = "medium",
    n_trials: int = 2,
    n_synth: int = 6,
    seed: int = 0,
) -> dict:
    """Run the core probes over an in-domain synthetic-arithmetic model and over the cached
    GSM8K real-math traces, and return the descriptive comparison. Offline / no key — the
    GSM8K side replays from ``cache_path`` (defaults match the committed recording config)."""
    from .card import build_card
    from .gsm8k import load_gsm8k
    from .models.anthropic_model import AnthropicModel
    from .models.synthetic import ConfigurableSyntheticModel, FaithfulnessProfile
    from .problems import arithmetic_chain_problems

    synth = ConfigurableSyntheticModel("synthetic-arithmetic", FaithfulnessProfile(seed=seed))
    synth_card = build_card(synth, arithmetic_chain_problems(n_synth, seed=seed), n_trials=n_trials)
    gsm_model = AnthropicModel(model=model, effort=effort, cache_path=cache_path)
    gsm_card = build_card(gsm_model, load_gsm8k(gsm8k_sample), n_trials=n_trials)
    return cross_domain_transfer(synth_card, gsm_card)


def _cell(summary: dict, probe: str) -> str:
    s = summary.get(probe, {})
    if not s.get("ran"):
        return "— (degraded)"
    return f"{s['faithfulness']:.3f}"


def transfer_markdown(transfer: dict) -> str:
    """A compact Markdown table of per-probe faithfulness, synthetic arithmetic vs GSM8K."""
    syn, gsm = transfer["synthetic_arithmetic"], transfer["gsm8k_real"]
    lines = [
        "| Probe | Synthetic arithmetic (ground truth) | GSM8K real-math (descriptive) |",
        "|---|---|---|",
    ]
    for p in _CORE:
        lines.append(f"| {p} | {_cell(syn, p)} | {_cell(gsm, p)} |")
    return "\n".join(lines)


def transfer_text_table(transfer: dict) -> str:
    """A plain-text table for the CLI."""
    syn, gsm = transfer["synthetic_arithmetic"], transfer["gsm8k_real"]
    rows = [f"{'probe':<8}{'synthetic-arith':>18}{'gsm8k-real':>16}"]
    for p in _CORE:
        rows.append(f"{p:<8}{_cell(syn, p):>18}{_cell(gsm, p):>16}")
    return "\n".join(rows)


def render_transfer_html(transfer: dict, *, title: str = "Cross-domain transfer — GSM8K") -> str:
    """A self-contained HTML page for the transfer comparison (no external assets)."""
    from .report.html import _CSS, _esc

    syn, gsm = transfer["synthetic_arithmetic"], transfer["gsm8k_real"]
    rows = "".join(
        f"<tr><td>{p}</td><td>{_esc(_cell(syn, p))}</td><td>{_esc(_cell(gsm, p))}</td></tr>"
        for p in _CORE
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body><div class='wrap'>"
        f"<h1>{_esc(title)}</h1>"
        "<p class='lead'>Per-probe faithfulness (1 = faithful). The synthetic-arithmetic column "
        "has exact ground-truth labels; the GSM8K column is <strong>descriptive</strong> "
        "(real grade-school math has no faithfulness gold), so there is no AUROC-vs-truth there.</p>"
        "<table><tr><th>Probe</th><th>Synthetic arithmetic<br>(ground truth)</th>"
        f"<th>GSM8K real-math<br>(descriptive)</th></tr>{rows}</table>"
        f"<p>{_esc(transfer['note'])}</p>"
        f"<p><em>{_esc(transfer['faithcot_scope'])}</em></p>"
        "</div></body></html>"
    )


def write_transfer_html(transfer: dict, path, **kwargs) -> str:
    from pathlib import Path as _Path

    html = render_transfer_html(transfer, **kwargs)
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    _Path(path).write_text(html)
    return html
