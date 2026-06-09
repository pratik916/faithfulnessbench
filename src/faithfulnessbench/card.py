"""Aggregation: turn raw probe outputs into a Faithfulness Card, plus the cross-probe
correlation that underpins the disagreement finding.

A card reports, per probe, a faithfulness sub-score (1 - mean unfaithfulness) with a
bootstrap CI, a per-domain breakdown, a transparent composite (the mean of the four
sub-scores — intentionally not a learned weighting), and each probe's diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .models.base import Model
from .probes import ProbeResult, default_probes
from .probes.base import Probe
from .problems import Problem


@dataclass
class FaithfulnessCard:
    model_name: str
    probe_scores: dict[str, dict]  # probe -> {unfaithfulness, ci_lo, ci_hi, faithfulness}
    composite_faithfulness: float
    per_domain: dict[str, dict[str, float]]  # domain -> probe -> faithfulness
    extras: dict[str, dict]  # probe -> probe-specific diagnostics
    n_problems: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_problems": self.n_problems,
            "composite_faithfulness": _f(self.composite_faithfulness),
            "probe_scores": {
                p: {k: _f(v) for k, v in d.items()} for p, d in self.probe_scores.items()
            },
            "per_domain": {
                dom: {p: _f(v) for p, v in probes.items()}
                for dom, probes in self.per_domain.items()
            },
            "extras": self.extras,
        }


def _f(x) -> float:
    return float(x) if x is not None else float("nan")


def run_probes(
    model: Model,
    problems: list[Problem],
    probes: list[Probe] | None = None,
    *,
    n_trials: int = 5,
) -> dict[str, ProbeResult]:
    """Run every probe over the problems and return results keyed by probe name."""
    probes = probes or default_probes()
    return {p.name: p.run(model, problems, n_trials=n_trials) for p in probes}


def build_card(
    model: Model,
    problems: list[Problem],
    probes: list[Probe] | None = None,
    *,
    n_trials: int = 5,
    bootstrap_seed: int = 0,
) -> FaithfulnessCard:
    """Run the probes and aggregate. Use :func:`card_from_results` if you already
    have the probe results (the validation experiment runs each probe once)."""
    results = run_probes(model, problems, probes, n_trials=n_trials)
    return card_from_results(
        getattr(model, "name", "model"), results, problems, bootstrap_seed=bootstrap_seed
    )


def card_from_results(
    model_name: str,
    results: dict[str, ProbeResult],
    problems: list[Problem],
    *,
    bootstrap_seed: int = 0,
) -> FaithfulnessCard:
    domain_of = {p.id: p.domain for p in problems}

    probe_scores: dict[str, dict] = {}
    per_domain: dict[str, dict[str, float]] = {}
    extras: dict[str, dict] = {}
    sub_scores: list[float] = []

    for name, res in results.items():
        # Cluster bootstrap by problem id (honest under nesting; reduces to the flat
        # bootstrap here since each problem contributes one aggregated score).
        lo, hi = metrics.bootstrap_cluster_ci(res.scores, res.problem_ids, seed=bootstrap_seed)
        faithfulness = res.faithfulness()
        probe_scores[name] = {
            "unfaithfulness": res.mean,
            "faithfulness": faithfulness,
            "ci_lo": 1.0 - hi,  # CI on unfaithfulness flips to a CI on faithfulness
            "ci_hi": 1.0 - lo,
        }
        extras[name] = res.extra
        if not np.isnan(faithfulness):
            sub_scores.append(faithfulness)

        # Per-domain faithfulness for this probe.
        by_domain: dict[str, list[float]] = {}
        for pid, score in zip(res.problem_ids, res.scores):
            by_domain.setdefault(domain_of.get(pid, "?"), []).append(score)
        for dom, vals in by_domain.items():
            per_domain.setdefault(dom, {})[name] = 1.0 - float(np.mean(vals))

    composite = float(np.mean(sub_scores)) if sub_scores else float("nan")
    return FaithfulnessCard(
        model_name=model_name,
        probe_scores=probe_scores,
        composite_faithfulness=composite,
        per_domain=per_domain,
        extras=extras,
        n_problems=len(problems),
    )


def stack_probe_scores(
    per_model_results: list[dict[str, ProbeResult]], probe_names: list[str]
) -> dict[str, np.ndarray]:
    """Concatenate each probe's per-instance scores across models into aligned vectors.

    Alignment is explicit on ``problem_ids`` (the intersection of the problems every
    probe scored for that model, in the first probe's order), so column j of every
    probe's vector provably refers to the same (model, problem) instance even if a
    probe drops a problem the others keep (e.g. SHI skips cue-less problems).
    """
    columns: dict[str, list[float]] = {name: [] for name in probe_names}
    first = probe_names[0]
    for results in per_model_results:
        by_pid = {
            name: dict(zip(results[name].problem_ids, results[name].scores.tolist()))
            for name in probe_names
        }
        common = [
            pid for pid in results[first].problem_ids if all(pid in by_pid[n] for n in probe_names)
        ]
        for pid in common:
            for name in probe_names:
                columns[name].append(by_pid[name][pid])
    return {name: np.asarray(vals, dtype=float) for name, vals in columns.items()}


def correlation_matrix(
    per_model_results: list[dict[str, ProbeResult]], probe_names: list[str]
) -> tuple[list[str], np.ndarray]:
    """Spearman correlation of per-instance unfaithfulness scores across probes.

    Computed over the pooled instances of the whole model population — that pooling is
    what makes the off-diagonal structure (probe disagreement) visible.
    """
    columns = stack_probe_scores(per_model_results, probe_names)
    return metrics.spearman_matrix(columns)
