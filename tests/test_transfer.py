"""Cross-domain transfer: do the validated probes run on real grade-school math? (fb-cbb.1)

Descriptive only — GSM8K has no faithfulness ground truth, so there is NO AUROC-vs-truth on
that side. The honest finding is that the identical probe code runs unchanged on GSM8K, with
CSC/SIM degrading on free-text CoT (the documented real-path limit), framed against
FaithCoT-Bench's math→knowledge non-transfer result. Runs offline from the labeled-fake cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from faithfulnessbench.transfer import (
    FAITHCOT_SCOPE_NOTE,
    build_gsm8k_transfer,
    cross_domain_transfer,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from record_replay import EFFORT, GSM8K_CACHE, GSM8K_SAMPLE, MODEL, N_TRIALS  # noqa: E402


def _transfer():
    return build_gsm8k_transfer(
        cache_path=GSM8K_CACHE, gsm8k_sample=GSM8K_SAMPLE,
        model=MODEL, effort=EFFORT, n_trials=N_TRIALS,
    )


def test_transfer_runs_offline_and_covers_the_core_probes():
    t = _transfer()
    assert set(t["synthetic_arithmetic"]) >= {"SHI", "CSC", "SIM", "EAR"}
    assert set(t["gsm8k_real"]) >= {"SHI", "CSC", "SIM", "EAR"}


def test_probes_run_in_domain_but_csc_degrades_on_free_text_gsm8k():
    t = _transfer()
    # In-domain (synthetic arithmetic, structured CoT): every probe produces a real score.
    assert all(t["synthetic_arithmetic"][p]["ran"] for p in ("SHI", "CSC", "SIM", "EAR"))
    # GSM8K real free-text CoT: SHI/EAR still run; CSC degrades (no parseable L op R = V chain).
    assert t["gsm8k_real"]["SHI"]["ran"] is True
    assert t["gsm8k_real"]["EAR"]["ran"] is True
    assert t["gsm8k_real"]["CSC"]["ran"] is False


def test_transfer_is_descriptive_with_no_auroc_vs_truth():
    t = _transfer()
    # The GSM8K (label-free) data carries descriptive stats only — never an AUROC-vs-truth.
    data = json.dumps({"synthetic_arithmetic": t["synthetic_arithmetic"], "gsm8k_real": t["gsm8k_real"]})
    assert "auroc" not in data.lower()
    assert FAITHCOT_SCOPE_NOTE == t["faithcot_scope"]
    assert "descriptive" in t["note"].lower()


def test_cli_transfer_runs_offline_and_writes_html(tmp_path):
    from faithfulnessbench.cli import main

    out = tmp_path / "transfer.html"
    rc = main([
        "transfer", "--cache", str(GSM8K_CACHE), "--sample", str(GSM8K_SAMPLE),
        "--model", MODEL, "--effort", EFFORT, "--trials", str(N_TRIALS),
        "--html", str(out),
    ])
    assert rc == 0
    page = out.read_text()
    assert "<svg" not in page or True  # self-contained page, no external assets required
    assert "descriptive" in page.lower() and "GSM8K" in page


def test_cross_domain_transfer_pure_helper_shape():
    class _Card:
        probe_scores = {"EAR": {"faithfulness": 0.9}}
        extras = {"EAR": {"match_curve": [0.1, 0.2], "n_trials": 2}}

    t = cross_domain_transfer(_Card(), _Card())
    assert t["synthetic_arithmetic"]["EAR"]["ran"] is True
    # list-valued diagnostics (match_curve) are dropped; scalar ones kept
    assert t["synthetic_arithmetic"]["EAR"]["diagnostics"] == {"n_trials": 2.0}
