"""Pins the probe registry so a (de)registration can never happen silently.

The project ships **four core** probes (SHI/CSC/SIM/EAR) as the default battery that
enters every Faithfulness Card and the committed validation report. The extended probes
(FIL/IPR/PAR/SHORTCUT) are **deliberately held out** of the default battery — each is a
single-axis probe whose synthetic targeted AUROC is 1.000 *by construction*, so adding it
to the headline battery would inflate the probe count without adding a new *kind* of
evidence and would dilute the disciplined "what 1.000 does and does not prove" story.
They remain first-class, individually validated, and available for pairwise/extended use.

This test locks that decision (fb-dvs.1): if someone moves a probe between the two
registries, drops one, or lets the public ``__all__`` drift from the registries, it fails.
"""
from __future__ import annotations

from faithfulnessbench.probes import (
    HELD_OUT_PROBE_CLASSES,
    PROBE_CLASSES,
    CSCProbe,
    EARProbe,
    FillerProbe,
    IPRProbe,
    ParaphraseProbe,
    SHIProbe,
    ShortcutProbe,
    SIMProbe,
    default_probes,
)

CORE = {"SHI", "CSC", "SIM", "EAR"}
HELD_OUT = {"FIL", "IPR", "PAR", "SHORTCUT"}


def test_default_battery_is_exactly_the_four_core_probes():
    assert set(PROBE_CLASSES) == CORE
    assert [p.name for p in default_probes()] == list(PROBE_CLASSES)


def test_extended_probes_are_registered_as_held_out_and_disjoint():
    assert set(HELD_OUT_PROBE_CLASSES) == HELD_OUT
    assert set(PROBE_CLASSES).isdisjoint(HELD_OUT_PROBE_CLASSES)


def test_every_registry_key_matches_its_class_name():
    for key, cls in {**PROBE_CLASSES, **HELD_OUT_PROBE_CLASSES}.items():
        assert cls.name == key


def test_registries_partition_all_exported_probe_classes():
    exported = {
        SHIProbe,
        CSCProbe,
        SIMProbe,
        EARProbe,
        FillerProbe,
        IPRProbe,
        ParaphraseProbe,
        ShortcutProbe,
    }
    registered = set(PROBE_CLASSES.values()) | set(HELD_OUT_PROBE_CLASSES.values())
    assert registered == exported
