"""Convenience re-exports, imported by ``faithfulnessbench/__init__.py``.

Kept separate so the top-level package can import it defensively (a partial install
or mid-build state degrades to just ``__version__`` + ``metrics`` rather than erroring).
"""
from __future__ import annotations

from .card import FaithfulnessCard, build_card, card_from_results, run_probes
from .models.base import CoTSimulator, CueDetector, Model, SubstringCueDetector, Trace
from .models.synthetic import (
    ConfigurableSyntheticModel,
    ExactArithmeticSimulator,
    FaithfulnessProfile,
    model_population,
)
from .probes import (
    PROBES,
    CSCProbe,
    EARProbe,
    ProbeResult,
    SHIProbe,
    SIMProbe,
    default_probes,
)
from .problems import (
    Cue,
    Problem,
    arithmetic_chain_problems,
    mixed_problems,
    multiple_choice_problems,
)
from .validation import run_validation

__all__ = [
    "Problem",
    "Cue",
    "arithmetic_chain_problems",
    "multiple_choice_problems",
    "mixed_problems",
    "Model",
    "Trace",
    "CueDetector",
    "CoTSimulator",
    "SubstringCueDetector",
    "ConfigurableSyntheticModel",
    "ExactArithmeticSimulator",
    "FaithfulnessProfile",
    "model_population",
    "PROBES",
    "SHIProbe",
    "CSCProbe",
    "SIMProbe",
    "EARProbe",
    "ProbeResult",
    "default_probes",
    "FaithfulnessCard",
    "build_card",
    "card_from_results",
    "run_probes",
    "run_validation",
]
