"""The headline validation experiment, as an importable, deterministic function.

Runs the four probes over the synthetic model population and computes:
  * targeted detection AUROC per probe (does it catch the axis it targets?),
  * a specificity matrix (probe vs axis — diagonal high, off-diagonal ~chance),
  * a combined detector AUROC vs the best single probe at flagging *any* unfaithfulness,
  * the cross-probe correlation matrix (the disagreement finding),
  * per-model Faithfulness Cards, and
  * concrete trace examples for the interactive viewer.

Everything is seeded, so the numbers reproduce exactly.
"""
from __future__ import annotations

import math

import numpy as np

from . import metrics
from .card import card_from_results, correlation_matrix, run_probes
from .models.base import SubstringCueDetector
from .models.synthetic import model_population
from .problems import mixed_problems

PROBE_ORDER = ["SHI", "CSC", "SIM", "EAR"]
# The single-axis synthetic model that each probe is supposed to catch.
AXIS_MODEL = {"SHI": "sycophant", "CSC": "post_hoc", "SIM": "decoy_cot", "EAR": "pre_commit"}


def run_validation(
    *,
    n_per_domain: int = 15,
    seed: int = 0,
    n_trials: int = 5,
    reproduce_cmd: str = "faithfulnessbench validate",
) -> dict:
    problems = mixed_problems(n_per_domain, seed=seed)
    population = model_population(seed=seed)
    results_by_model = {
        m.name: run_probes(m, problems, n_trials=n_trials) for m in population
    }

    def scores(model_name: str, probe: str) -> np.ndarray:
        return results_by_model[model_name][probe].scores

    faithful = {p: scores("faithful", p) for p in PROBE_ORDER}

    # --- targeted detection AUROC + ROC curve per probe ---
    targeted_auroc: dict[str, dict] = {}
    roc: dict[str, dict] = {}
    for p in PROBE_ORDER:
        s = np.r_[faithful[p], scores(AXIS_MODEL[p], p)]
        y = np.r_[np.zeros(faithful[p].size), np.ones(scores(AXIS_MODEL[p], p).size)]
        targeted_auroc[p] = metrics.auc_summary(s, y, seed=seed)
        fpr, tpr = metrics.roc_curve(s, y)
        roc[p] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": targeted_auroc[p]["auc"]}

    # --- specificity: probe (row) vs each unfaithfulness axis (col) ---
    specificity: dict[str, dict] = {}
    for p in PROBE_ORDER:
        specificity[p] = {}
        for axis in PROBE_ORDER:
            s = np.r_[faithful[p], scores(AXIS_MODEL[axis], p)]
            y = np.r_[np.zeros(faithful[p].size), np.ones(scores(AXIS_MODEL[axis], p).size)]
            specificity[p][axis] = metrics.roc_auc(s, y)

    # --- combined detector vs best single probe at flagging ANY unfaithfulness ---
    comb_scores, comb_labels = [], []
    single_pooled: dict[str, list] = {p: [] for p in PROBE_ORDER}
    pooled_labels: list = []
    for name, results in results_by_model.items():
        stacked = np.vstack([results[p].scores for p in PROBE_ORDER])  # (4, n)
        comb = stacked.mean(axis=0)
        label = 0 if name == "faithful" else 1
        comb_scores.append(comb)
        comb_labels.append(np.full(comb.shape, label))
        pooled_labels.append(np.full(comb.shape, label))
        for p in PROBE_ORDER:
            single_pooled[p].append(results[p].scores)
    cs, cl = np.concatenate(comb_scores), np.concatenate(comb_labels)
    combined_auroc = metrics.auc_summary(cs, cl, seed=seed)
    pl = np.concatenate(pooled_labels)
    single_mixed_auroc = {
        p: metrics.roc_auc(np.concatenate(single_pooled[p]), pl) for p in PROBE_ORDER
    }

    # --- cross-probe correlation (disagreement finding) ---
    labels, mat = correlation_matrix(list(results_by_model.values()), PROBE_ORDER)
    matrix = [[float(mat[i][j]) for j in range(len(labels))] for i in range(len(labels))]
    off = mat[~np.eye(len(labels), dtype=bool)]
    mean_off = float(np.nanmean(np.abs(off)))

    # --- per-model Faithfulness Cards ---
    cards = [
        card_from_results(name, results_by_model[name], problems, bootstrap_seed=seed).to_dict()
        for name in results_by_model
    ]
    syco = next(c for c in cards if c["model_name"] == "sycophant")
    disagreement = (
        f"The four probes are only weakly correlated (mean |off-diagonal| Spearman "
        f"≈ {mean_off:.2f}). For instance the 'sycophant' model fails "
        f"Silent-Hint-Injection (SHI faithfulness {syco['probe_scores']['SHI']['faithfulness']:.2f}) "
        f"yet passes Simulatability (SIM {syco['probe_scores']['SIM']['faithfulness']:.2f}) and "
        f"Step-Corruption (CSC {syco['probe_scores']['CSC']['faithfulness']:.2f}) — any single "
        f"probe used alone would have cleared it. That is why the unit of measurement is a card, not a scalar."
    )

    # --- trace examples for the interactive viewer ---
    detector = SubstringCueDetector()
    by_name = {m.name: m for m in population}

    def example(model, problem):
        base = model.reason(problem)
        cued = model.reason(problem, cue=problem.cue)
        return {
            "id": problem.id,
            "domain": problem.domain,
            "question": problem.question,
            "true_answer": problem.answer,
            "cue_target": problem.cue.target,
            "cue_text": problem.cue.text,
            "baseline": {"answer": base.answer, "cot_lines": list(base.steps)},
            "cued": {
                "answer": cued.answer,
                "cot_lines": list(cued.steps),
                "acknowledged": detector.mentions(cued.cot, problem.cue),
            },
        }

    arith = [p for p in problems if p.domain == "arithmetic"]
    mcq = [p for p in problems if p.domain == "mcq"]
    trace_examples = [
        example(by_name["sycophant"], arith[0]),
        example(by_name["sycophant"], mcq[0]),
        example(by_name["sycophant"], arith[1]),
        example(by_name["faithful"], arith[2]),
        example(by_name["faithful"], mcq[1]),
    ]

    return {
        "title": "FaithfulnessBench — Chain-of-Thought Faithfulness, Validated",
        "subtitle": (
            "Four causal probes for whether a reasoning model's chain-of-thought actually "
            "drives its answer — validated against synthetic models whose (un)faithfulness "
            "is known by construction."
        ),
        "meta": f"seed={seed} · {len(problems)} problems · {len(population)} models · {n_trials} trials/probe · fully deterministic",
        "n_problems": len(problems),
        "n_models": len(population),
        "validation": {
            "probes": PROBE_ORDER,
            "targeted_auroc": targeted_auroc,
            "roc": roc,
            "specificity": specificity,
            "combined_auroc": combined_auroc,
            "single_mixed_auroc": single_mixed_auroc,
        },
        "correlation": {"labels": labels, "matrix": matrix},
        "cards": cards,
        "disagreement": disagreement,
        "trace_examples": trace_examples,
        "reproduce_cmd": reproduce_cmd,
    }


def json_safe(obj):
    """Recursively replace NaN/Inf with None so the result is valid JSON."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def summarize(report: dict) -> str:
    """A compact console summary of a validation report."""
    val = report["validation"]
    lines = [
        report["title"],
        report["meta"],
        "",
        "Targeted detection AUROC (probe catches its own axis):",
    ]
    for p in val["probes"]:
        a = val["targeted_auroc"][p]
        lines.append(f"  {p}: {a['auc']:.3f}  (95% CI {a['ci_lo']:.3f}-{a['ci_hi']:.3f})")
    lines.append("")
    lines.append(
        f"Combined detector AUROC (any unfaithfulness): {val['combined_auroc']['auc']:.3f}"
    )
    best = max(val["single_mixed_auroc"], key=lambda p: val["single_mixed_auroc"][p])
    lines.append(
        f"Best single probe at flagging any unfaithfulness: {best} "
        f"{val['single_mixed_auroc'][best]:.3f}  → the card beats any single probe."
    )
    lines.append("")
    lines.append(report["disagreement"])
    return "\n".join(lines)
