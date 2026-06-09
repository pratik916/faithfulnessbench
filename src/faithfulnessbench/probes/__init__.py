"""The four orthogonal faithfulness probes."""
from __future__ import annotations

from .base import Probe, ProbeResult
from .csc import CSCProbe, OperandCorruptor
from .ear import EARProbe
from .shi import SHIProbe
from .sim import SIMProbe

# Probe name -> class, in canonical reporting order.
PROBE_CLASSES: dict[str, type[Probe]] = {
    "SHI": SHIProbe,
    "CSC": CSCProbe,
    "SIM": SIMProbe,
    "EAR": EARProbe,
}


def default_probes() -> list[Probe]:
    """Fresh instances of all four probes with default collaborators."""
    return [SHIProbe(), CSCProbe(), SIMProbe(), EARProbe()]


# Convenience: instantiated probes in canonical order.
PROBES = default_probes()

__all__ = [
    "Probe",
    "ProbeResult",
    "SHIProbe",
    "CSCProbe",
    "SIMProbe",
    "EARProbe",
    "OperandCorruptor",
    "PROBE_CLASSES",
    "PROBES",
    "default_probes",
]
