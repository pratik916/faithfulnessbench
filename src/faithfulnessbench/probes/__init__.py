"""The faithfulness probes: four orthogonal *core* probes plus four *held-out* extensions.

The default battery (:data:`PROBE_CLASSES`) is the four core probes — SHI, CSC, SIM, EAR —
that enter every Faithfulness Card and the committed validation report. The extended
probes (:data:`HELD_OUT_PROBE_CLASSES` — FIL, IPR, PAR, SHORTCUT) are deliberately *held
out* of the default battery: each is a single-axis probe whose synthetic targeted AUROC is
1.000 by construction, so promoting it into the headline battery would inflate the probe
count without contributing a new *kind* of evidence. They remain first-class and
individually validated, available for pairwise/extended-population use.
``tests/test_probe_registry.py`` pins this split so neither set can drift silently.
"""
from __future__ import annotations

from .base import Probe, ProbeResult
from .csc import CSCProbe, OperandCorruptor
from .ear import EARProbe
from .fil import FillerProbe
from .ipr import IPRProbe
from .par import ParaphraseProbe
from .shi import SHIProbe
from .shortcut import ShortcutProbe
from .sim import SIMProbe

# The default battery — probe name -> class, in canonical reporting order. These four
# enter every Faithfulness Card and the committed validation report.
PROBE_CLASSES: dict[str, type[Probe]] = {
    "SHI": SHIProbe,
    "CSC": CSCProbe,
    "SIM": SIMProbe,
    "EAR": EARProbe,
}

# Deliberately HELD OUT of the default battery (see module docstring). First-class probes,
# individually validated in their own test modules and usable in the extended population,
# but not promoted into the headline numbers. Kept disjoint from PROBE_CLASSES.
HELD_OUT_PROBE_CLASSES: dict[str, type[Probe]] = {
    "FIL": FillerProbe,
    "IPR": IPRProbe,
    "PAR": ParaphraseProbe,
    "SHORTCUT": ShortcutProbe,
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


_validate_registry(PROBE_CLASSES)
_validate_registry(HELD_OUT_PROBE_CLASSES)
if set(PROBE_CLASSES) & set(HELD_OUT_PROBE_CLASSES):
    raise ValueError("a probe is registered as both core and held-out")

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
    "ParaphraseProbe",
    "ShortcutProbe",
    "OperandCorruptor",
    "PROBE_CLASSES",
    "HELD_OUT_PROBE_CLASSES",
    "PROBES",
    "default_probes",
]
