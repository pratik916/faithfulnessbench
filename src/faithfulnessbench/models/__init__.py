"""Model adapters: the abstract interface, the synthetic ground-truth model, and
(optionally) the real Anthropic adapter."""
from __future__ import annotations

from .base import (
    CoTSimulator,
    CueDetector,
    Model,
    SubstringCueDetector,
    Trace,
)
from .synthetic import (
    ConfigurableSyntheticModel,
    ExactArithmeticSimulator,
    FaithfulnessProfile,
    model_population,
)

__all__ = [
    "Model",
    "Trace",
    "CueDetector",
    "CoTSimulator",
    "SubstringCueDetector",
    "ConfigurableSyntheticModel",
    "ExactArithmeticSimulator",
    "FaithfulnessProfile",
    "model_population",
]
