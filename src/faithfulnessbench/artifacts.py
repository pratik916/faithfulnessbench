"""Shared artifact generation and the reproducibility check.

One code path builds the committed artifacts (`results.json`, the HTML report, and —
for the experiment driver — the SVG figures); the CLI and `experiments/validate_synthetic.py`
both call :func:`generate_artifacts` so they can never drift apart.

`validate --check` recomputes the validation and diffs it against the committed
`results.json` *without* overwriting it (:func:`check_against_committed`).

Field policy for the check (documented and stable):

* **Exact** (compared with ``abs_tol=1e-12``): the headline numbers — problem/model
  counts, every AUROC (targeted, combined, single-on-mixed, specificity), the Spearman
  correlation matrix, and each model's composite faithfulness and per-probe faithfulness.
  These are rank- and mean-based, so they reproduce bit-stably across platforms.
* **Ignored**: provenance and prose (``reproduce_cmd``, ``title``, ``subtitle``, ``meta``,
  ``disagreement``, ``trace_examples``) and the **bootstrap confidence intervals**
  (``ci_lo``/``ci_hi``) and ROC curves — these are seeded-resampling outputs that may
  differ in the last ULP across numpy versions, so they are advisory, not gated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .validation import json_safe, run_validation


def generate_artifacts(
    *,
    n_per_domain: int,
    seed: int,
    n_trials: int,
    json_path: str,
    report_path: str,
    reproduce_cmd: str,
    figures_dir: str | None = None,
) -> dict:
    """Run the validation and write results.json + the HTML report (+ figures). Returns the report."""
    from .report import write_figures, write_report

    report = run_validation(
        n_per_domain=n_per_domain, seed=seed, n_trials=n_trials, reproduce_cmd=reproduce_cmd
    )
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(json_safe(report), indent=2))
    write_report(report, report_path)
    if figures_dir is not None:
        write_figures(report, figures_dir)
    return report


def key_numbers(report: dict) -> dict:
    """Extract the documented set of headline numbers that must reproduce exactly."""
    v = report["validation"]
    probes = v["probes"]
    return {
        "n_problems": report["n_problems"],
        "n_models": report["n_models"],
        "combined_auroc": v["combined_auroc"]["auc"],
        "targeted_auroc": {p: v["targeted_auroc"][p]["auc"] for p in probes},
        "single_mixed_auroc": dict(v["single_mixed_auroc"]),
        "specificity": {p: dict(v["specificity"][p]) for p in probes},
        "correlation": report["correlation"]["matrix"],
        "composite": {c["model_name"]: c["composite_faithfulness"] for c in report["cards"]},
        "probe_faithfulness": {
            c["model_name"]: {p: c["probe_scores"][p]["faithfulness"] for p in c["probe_scores"]}
            for c in report["cards"]
        },
    }


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, val in obj.items():
            _flatten(f"{prefix}/{k}", val, out)
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            _flatten(f"{prefix}[{i}]", val, out)
    else:
        out[prefix] = obj


def diff_key_numbers(committed: dict, fresh: dict, *, abs_tol: float = 1e-12) -> list[str]:
    """Return human-readable drift messages for the exact-match key numbers ([] = identical)."""
    a: dict = {}
    b: dict = {}
    _flatten("", key_numbers(committed), a)
    _flatten("", key_numbers(fresh), b)
    msgs: list[str] = []
    for k in sorted(set(a) | set(b)):
        if k not in a:
            msgs.append(f"{k}: present in fresh but missing in committed")
            continue
        if k not in b:
            msgs.append(f"{k}: present in committed but missing in fresh")
            continue
        x, y = a[k], b[k]
        if isinstance(x, bool) or isinstance(y, bool):
            if x != y:
                msgs.append(f"{k}: committed={x!r} fresh={y!r}")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not math.isclose(x, y, rel_tol=0.0, abs_tol=abs_tol):
                msgs.append(f"{k}: committed={x} fresh={y} (|Δ|={abs(x - y):.3e} > {abs_tol:g})")
        elif x != y:
            msgs.append(f"{k}: committed={x!r} fresh={y!r}")
    return msgs


def check_against_committed(
    committed_path, *, n_per_domain: int, seed: int, n_trials: int, reproduce_cmd: str
) -> list[str]:
    """Recompute the validation and diff it against the committed results.json (no write)."""
    committed = json.loads(Path(committed_path).read_text())
    fresh = json_safe(
        run_validation(n_per_domain=n_per_domain, seed=seed, n_trials=n_trials, reproduce_cmd=reproduce_cmd)
    )
    return diff_key_numbers(committed, fresh)


def stale_artifacts(json_path, report_path, figures_dir) -> list[str]:
    """Return messages for any committed HTML report / SVG figure that is stale.

    Rendering is deterministic, so a fresh render of ``json_path`` must reproduce the
    committed report and figures byte-for-byte; any mismatch means someone changed the
    numbers without regenerating the artifacts ([] = everything fresh).
    """
    from .report import build_figures, render_report

    report = json.loads(Path(json_path).read_text())
    msgs: list[str] = []

    rp = Path(report_path)
    if not rp.exists():
        msgs.append(f"{rp} is missing (regenerate from {json_path})")
    elif rp.read_text() != render_report(report):
        msgs.append(f"{rp} is stale vs {json_path} (re-run the experiment driver)")

    fig_dir = Path(figures_dir)
    for key, svg in build_figures(report).items():
        sp = fig_dir / f"{key}.svg"
        if not sp.exists():
            msgs.append(f"{sp} is missing (regenerate from {json_path})")
        elif sp.read_text() != svg:
            msgs.append(f"{sp} is stale vs {json_path}")
    return msgs
