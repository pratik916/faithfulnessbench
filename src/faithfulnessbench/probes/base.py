"""Common probe interface.

Every probe consumes a :class:`~faithfulnessbench.models.base.Model` and a list of
problems and returns a continuous **unfaithfulness score in [0, 1]** per instance
(higher = more unfaithful), plus probe-specific diagnostics in ``extra``.
"""
from __future__ import annotations

import abc
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..models.base import Model
from ..problems import Cue, Problem


def majority_answer(
    model: Model, problem: Problem, n_trials: int, *, cue: Cue | None = None
) -> str:
    """The model's most common answer over ``n_trials`` realizations.

    A single draw is a noisy baseline for a stochastic real model; majority-voting the
    baseline answer makes SHI's flip test and CSC's tracking test robust. Deterministic
    given the (deterministic-per-trial) model, with ties broken by first occurrence.
    """
    answers = [model.reason(problem, cue=cue, trial=t).answer for t in range(max(1, n_trials))]
    return Counter(answers).most_common(1)[0][0]


@dataclass
class ProbeResult:
    probe: str
    problem_ids: list[str]
    scores: np.ndarray  # per-instance unfaithfulness, shape (n_instances,)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Coerce + validate at construction so a buggy probe fails loudly here rather
        # than silently corrupting downstream AUROC / card aggregation.
        self.scores = np.asarray(self.scores, dtype=float)
        self.validate()

    def validate(self) -> None:
        if self.scores.ndim != 1:
            raise ValueError(f"{self.probe}: scores must be 1-D, got shape {self.scores.shape}")
        if len(self.problem_ids) != self.scores.shape[0]:
            raise ValueError(
                f"{self.probe}: len(problem_ids)={len(self.problem_ids)} != "
                f"len(scores)={self.scores.shape[0]}"
            )
        if self.scores.size:
            if not np.all(np.isfinite(self.scores)):
                raise ValueError(f"{self.probe}: scores contain non-finite values")
            lo, hi = float(self.scores.min()), float(self.scores.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(f"{self.probe}: scores must lie in [0, 1], got [{lo}, {hi}]")

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores.size else float("nan")

    def faithfulness(self) -> float:
        """Aggregate faithfulness sub-score = 1 - mean unfaithfulness."""
        return 1.0 - self.mean if self.scores.size else float("nan")


class Probe(abc.ABC):
    name: str = "probe"

    @abc.abstractmethod
    def run(
        self, model: Model, problems: list[Problem], *, n_trials: int | None = None
    ) -> ProbeResult: ...
