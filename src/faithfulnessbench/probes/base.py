"""Common probe interface.

Every probe consumes a :class:`~faithfulnessbench.models.base.Model` and a list of
problems and returns a continuous **unfaithfulness score in [0, 1]** per instance
(higher = more unfaithful), plus probe-specific diagnostics in ``extra``.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from ..models.base import Model
from ..problems import Problem


@dataclass
class ProbeResult:
    probe: str
    problem_ids: list[str]
    scores: np.ndarray  # per-instance unfaithfulness, shape (n_instances,)
    extra: dict = field(default_factory=dict)

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
