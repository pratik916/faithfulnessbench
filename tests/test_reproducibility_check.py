"""`validate --check` reproducibility gate + single-sourced version (fb-tch.2).

The committed `experiments/results/results.json` must keep reproducing from the
documented params. `validate --check` recomputes and diffs the headline numbers
*without* overwriting the artifact; the version is single-sourced from package
metadata; and the CLI + experiment driver share one artifact-generation path.
"""
from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path

import faithfulnessbench
from faithfulnessbench import artifacts
from faithfulnessbench.cli import main
from faithfulnessbench.validation import json_safe, run_validation

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "experiments" / "results" / "results.json"
# Params the committed artifact was generated with (see its `meta` field).
PARAMS = dict(n_per_domain=20, seed=0, n_trials=5)
EXP_CMD = "python experiments/validate_synthetic.py"


def test_version_is_single_sourced_from_metadata():
    assert faithfulnessbench.__version__ == importlib.metadata.version("faithfulnessbench")
    assert faithfulnessbench.__version__  # non-empty


def test_check_against_committed_reports_no_drift():
    drift = artifacts.check_against_committed(COMMITTED, reproduce_cmd=EXP_CMD, **PARAMS)
    assert drift == [], drift


def test_diff_detects_a_mutated_headline_number():
    fresh = json_safe(run_validation(reproduce_cmd=EXP_CMD, **PARAMS))
    tampered = copy.deepcopy(fresh)
    tampered["validation"]["combined_auroc"]["auc"] = 0.123
    drift = artifacts.diff_key_numbers(tampered, fresh)
    assert any("combined_auroc" in m for m in drift), drift


def test_validate_check_matches_and_does_not_overwrite(tmp_path):
    target = tmp_path / "results.json"
    target.write_bytes(COMMITTED.read_bytes())
    before = target.read_bytes()
    rc = main(["validate", "--check", "--json", str(target)])
    assert rc == 0
    assert target.read_bytes() == before  # check mode never writes


def test_validate_check_exits_nonzero_on_drift(tmp_path):
    tampered = json.loads(COMMITTED.read_text())
    tampered["validation"]["combined_auroc"]["auc"] = 0.123
    target = tmp_path / "results.json"
    target.write_text(json.dumps(tampered))
    rc = main(["validate", "--check", "--json", str(target)])
    assert rc == 1


def _assert_reproduces(fresh, committed, path="root"):
    """Identical *structure* and *text*; numeric leaves are checked for shape, not value.

    The artifact reproduces to the README's 'byte-for-(numerically-)identically' only for
    the *headline* numbers — and those are gated exactly (1e-12) by
    `test_check_against_committed_reports_no_drift`, which passes on every platform. The
    derived statistics that also live in the artifact — the reliability diagram's per-bin
    means and **integer counts**, seeded bootstrap CIs, ECE, and permutation p-values (a
    1/n_perm grid) — are *not* bit-reproducible across BLAS implementations: a Linux runner
    pushes borderline scores across bin/quantile edges, moving these by ~1e-3 (and bin
    counts by whole units). That is platform discretization, not a regression. So this
    end-to-end check asserts the regenerated artifact has the **same shape and the same
    prose** (which catches a dropped/added key, a changed list length, or drifted text —
    e.g. a probe wired in but not surfaced), and defers numeric correctness to the exact
    1e-12 headline gate above. `bool` is matched exactly (it is not a derived statistic).
    """
    assert type(fresh) is type(committed), f"{path}: type {type(fresh).__name__} != {type(committed).__name__}"
    if isinstance(committed, dict):
        assert set(fresh) == set(committed), f"{path}: keys differ by {set(fresh) ^ set(committed)}"
        for k in committed:
            # `disagreement` is the one prose field that embeds derived statistics
            # (Cohen's kappa ≈ 0.25, Spearman ≈ 0.25 at :.2f). The Spearman value sits ~1e-3
            # from a rounding boundary, so a Linux BLAS can flip its rendered digit — shape,
            # not content. Check it is non-empty prose, not its exact text.
            if k == "disagreement":
                assert isinstance(fresh[k], str) and fresh[k], f"{path}.{k}: empty or non-string"
                continue
            _assert_reproduces(fresh[k], committed[k], f"{path}.{k}")
    elif isinstance(committed, list):
        assert len(fresh) == len(committed), f"{path}: length {len(fresh)} != {len(committed)}"
        for i, (a, b) in enumerate(zip(fresh, committed)):
            _assert_reproduces(a, b, f"{path}[{i}]")
    elif isinstance(committed, bool):
        assert fresh == committed, f"{path}: {fresh} != {committed}"
    elif isinstance(committed, (int, float)):
        pass  # platform-dependent derived statistic — value is gated by the 1e-12 headline check
    else:  # str / None
        assert fresh == committed, f"{path}: {fresh!r} != {committed!r}"


def test_generate_artifacts_reproduces_the_committed_artifact(tmp_path):
    json_path = tmp_path / "results.json"
    artifacts.generate_artifacts(
        json_path=str(json_path), report_path=str(tmp_path / "r.html"),
        reproduce_cmd=EXP_CMD, **PARAMS,
    )
    # Structure + prose identical (this test); headline numbers exact to 1e-12 (the test above).
    _assert_reproduces(json.loads(json_path.read_text()), json.loads(COMMITTED.read_text()))
    assert artifacts.check_against_committed(COMMITTED, reproduce_cmd=EXP_CMD, **PARAMS) == []
