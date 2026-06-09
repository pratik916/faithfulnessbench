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
from .probes import default_probes
from .problems import mixed_problems

PROBE_ORDER = ["SHI", "CSC", "SIM", "EAR"]
# The single-axis synthetic model that each probe is supposed to catch.
AXIS_MODEL = {"SHI": "sycophant", "CSC": "post_hoc", "SIM": "decoy_cot", "EAR": "pre_commit"}
# Label-noise levels swept for the AUROC-vs-noise sensitivity curve (fb-yst.1).
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]


def run_validation(
    *,
    n_per_domain: int = 15,
    seed: int = 0,
    n_trials: int = 5,
    reproduce_cmd: str = "faithfulnessbench validate",
    extended_models: list[tuple] | None = None,
) -> dict:
    """Run the headline validation over the FROZEN population.

    ``extended_models`` is an optional list of ``(model, target_probe)`` held-out models
    that each receive a *pairwise* targeted AUROC (faithful vs that model on its target
    probe) reported under ``validation['extended_auroc']`` — without entering any pooled
    metric (combined/single-mixed AUROC, correlation, cards). This keeps the committed
    headline numbers a frozen baseline that new models cannot silently move; deliberate
    re-baselining means adding a model to ``model_population`` in one explicit commit.
    """
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

    # --- negative control: shuffle the targeted labels -> AUROC must collapse to ~0.5,
    #     proving the high targeted AUROC reflects real structure and the harness *can* fail.
    negative_control_auroc: dict[str, float] = {}
    for p in PROBE_ORDER:
        s = np.r_[faithful[p], scores(AXIS_MODEL[p], p)]
        y = np.r_[np.zeros(faithful[p].size), np.ones(scores(AXIS_MODEL[p], p).size)]
        negative_control_auroc[p] = metrics.permutation_auroc(s, y, seed=seed)

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
        f"Sanity check — the probes do not spuriously co-fire: on this single-axis population they "
        f"agree only on the fully-unfaithful corner, giving a low mean off-diagonal Spearman ≈ {mean_off:.2f} "
        f"(the exact magnitude is a function of the population composition, not a measured property of the probes). "
        f"The substantive point is qualitative and robust: the 'sycophant' model fails "
        f"Silent-Hint-Injection (SHI faithfulness {syco['probe_scores']['SHI']['faithfulness']:.2f}) "
        f"yet passes Simulatability (SIM {syco['probe_scores']['SIM']['faithfulness']:.2f}) and "
        f"Step-Corruption (CSC {syco['probe_scores']['CSC']['faithfulness']:.2f}) — any single probe used alone "
        f"would have cleared it. That is why the unit of measurement is a card, not a scalar."
    )

    # --- AUROC-vs-noise: targeted AUROC as a sensitivity MEASUREMENT, not just wiring.
    #     At zero noise it reproduces the wiring check (1.0); as symmetric label noise
    #     rises the synthetic classes overlap and AUROC falls toward chance — so the
    #     number means something. The clean (frozen) population above is untouched.
    probes_by_name = {pr.name: pr for pr in default_probes()}
    auroc_vs_noise: dict[str, list] = {p: [] for p in PROBE_ORDER}
    for noise in NOISE_LEVELS:
        npop = {m.name: m for m in model_population(seed=seed, label_noise=noise)}
        n_faithful = {
            p: probes_by_name[p].run(npop["faithful"], problems, n_trials=n_trials).scores
            for p in PROBE_ORDER
        }
        for p in PROBE_ORDER:
            axis = probes_by_name[p].run(npop[AXIS_MODEL[p]], problems, n_trials=n_trials).scores
            s = np.r_[n_faithful[p], axis]
            y = np.r_[np.zeros(n_faithful[p].size), np.ones(axis.size)]
            auroc_vs_noise[p].append({"noise": noise, "auc": metrics.roc_auc(s, y)})

    # --- held-out extended population: pairwise targeted AUROC only, never pooled ---
    extended_auroc: dict[str, dict] = {}
    for model, target_probe in extended_models or []:
        ext_results = run_probes(model, problems, n_trials=n_trials)
        es = np.r_[faithful[target_probe], ext_results[target_probe].scores]
        ey = np.r_[
            np.zeros(faithful[target_probe].size),
            np.ones(ext_results[target_probe].scores.size),
        ]
        extended_auroc[model.name] = {"probe": target_probe, **metrics.auc_summary(es, ey, seed=seed)}

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

    validation_block = {
        "probes": PROBE_ORDER,
        "targeted_auroc": targeted_auroc,
        "negative_control_auroc": negative_control_auroc,
        "roc": roc,
        "specificity": specificity,
        "combined_auroc": combined_auroc,
        "single_mixed_auroc": single_mixed_auroc,
        "auroc_vs_noise": auroc_vs_noise,
        "noise_levels": NOISE_LEVELS,
    }
    if extended_auroc:  # only present when held-out models are supplied (keeps committed artifact stable)
        validation_block["extended_auroc"] = extended_auroc

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
        "validation": validation_block,
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
    if "negative_control_auroc" in val:
        nc = val["negative_control_auroc"]
        lines.append(
            "Negative control (shuffled labels) AUROC: "
            + ", ".join(f"{p} {nc[p]:.3f}" for p in val["probes"])
            + "  → ~0.5: the harness can report chance."
        )
    lines.append("")
    lines.append(report["disagreement"])
    return "\n".join(lines)
