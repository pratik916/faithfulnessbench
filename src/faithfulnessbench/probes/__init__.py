"""The four orthogonal faithfulness probes."""
from __future__ import annotations

from .base import Probe, ProbeResult
from .csc import CSCProbe, OperandCorruptor
from .ear import EARProbe
from .fil import FillerProbe
from .ipr import IPRProbe
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
    """Fresh instances of every registered probe, in canonical order.

    Derived from :data:`PROBE_CLASSES` so the registry is the single source of truth —
    a new probe is added in exactly one place.
    """
    return [cls() for cls in PROBE_CLASSES.values()]


def _validate_registry(registry: dict[str, type[Probe]] = PROBE_CLASSES) -> None:
    """Fail fast if a registry key disagrees with its class' ``name`` or names collide."""
    for key, cls in registry.items():
        if cls.name != key:
            raise ValueError(f"probe registry key {key!r} disagrees with class name {cls.name!r}")
    names = [cls.name for cls in registry.values()]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate probe names in registry: {names}")


_validate_registry()

# Convenience: instantiated probes in canonical order.
PROBES = default_probes()

__all__ = [
    "Probe",
    "ProbeResult",
    "SHIProbe",
    "CSCProbe",
    "SIMProbe",
    "EARProbe",
    "FillerProbe",
    "IPRProbe",
    "OperandCorruptor",
    "PROBE_CLASSES",
    "PROBES",
    "default_probes",
]
