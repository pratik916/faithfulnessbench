"""ProbeResult invariants + a single-sourced probe registry (fb-bys.1).

A buggy probe should fail loudly at construction rather than silently corrupting the
downstream AUROC/card aggregation, and the registry must not be able to drift from the
probe classes' own names.
"""
from __future__ import annotations

import numpy as np
import pytest

from faithfulnessbench.probes import PROBE_CLASSES, _validate_registry, default_probes
from faithfulnessbench.probes.base import ProbeResult
from faithfulnessbench.probes.shi import SHIProbe


def test_probe_result_rejects_out_of_range_scores():
    with pytest.raises(ValueError):
        ProbeResult("X", ["a"], np.array([1.5]))


def test_probe_result_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ProbeResult("X", ["a", "b"], np.array([0.5]))


def test_probe_result_accepts_valid_scores():
    r = ProbeResult("X", ["a", "b"], np.array([0.0, 1.0]))
    assert r.scores.shape == (2,)
    assert r.faithfulness() == 0.5


def test_registry_and_default_probes_are_consistent():
    assert [p.name for p in default_probes()] == list(PROBE_CLASSES)
    for i, (name, cls) in enumerate(PROBE_CLASSES.items()):
        assert cls.name == name
        assert isinstance(default_probes()[i], cls)


def test_validate_registry_rejects_a_name_mismatch():
    with pytest.raises(ValueError):
        _validate_registry({"NOT-SHI": SHIProbe})
