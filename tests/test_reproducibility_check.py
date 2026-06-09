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


def test_generate_artifacts_reproduces_the_committed_artifact(tmp_path):
    json_path = tmp_path / "results.json"
    artifacts.generate_artifacts(
        json_path=str(json_path), report_path=str(tmp_path / "r.html"),
        reproduce_cmd=EXP_CMD, **PARAMS,
    )
    assert json.loads(json_path.read_text()) == json.loads(COMMITTED.read_text())
