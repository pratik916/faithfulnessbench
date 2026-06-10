# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FaithfulnessBench measures chain-of-thought (CoT) **faithfulness** in reasoning models — whether the stated reasoning causally drives the answer — and, crucially, **validates the measurement itself** against synthetic models whose (un)faithfulness is known by construction. `docs/DESIGN.md` is the methodology source of truth; keep code and that doc in sync.

## Commands

```bash
pip install -e ".[dev]"                       # install with test deps
pytest                                         # full suite (221 tests, no network/key)
pytest tests/test_probes.py -q                 # one file
pytest tests/test_metrics.py::test_auc_known_value  # one test
faithfulnessbench validate                     # headline experiment -> results.json + report + figures
python experiments/validate_synthetic.py       # same, runnable straight from a checkout
faithfulnessbench score --model claude-sonnet-4-6   # score a real model (needs ANTHROPIC_API_KEY)
```

Regenerate the committed artifacts (`experiments/results/results.json`, `report/faithfulness_report.html`, `docs/assets/*.svg`) after any change that affects numbers by re-running the validation.

## Architecture (the parts that span files)

The spine is an **orthogonality contract** between two files that everything else depends on:

- `models/synthetic.py` — `ConfigurableSyntheticModel` has four independent *unfaithfulness-rate* dials (`p_hint_sycophancy`, `p_post_hoc`, `p_decoy_cot`, `p_pre_commit`). Each dial manifests **only** in the one model method the matching probe intervenes on (`reason` / `continue_from_cot` / `answer_from_prefix`). This is what makes per-instance ground-truth labels exact and the probes orthogonal.
- `probes/` (`shi`, `csc`, `sim`, `ear`) — each probe drives the `Model` interface (`models/base.py`) and returns per-instance unfaithfulness in [0,1]. **Probes never touch synthetic internals** — the same probe code runs unchanged against `models/anthropic_model.py`. If you add a probe, add its dial to the synthetic model and wire both into `validation.AXIS_MODEL` / `PROBE_ORDER`.

Then: `card.py` aggregates probe results into a Faithfulness Card + cross-probe correlation; `validation.py` is the end-to-end experiment (targeted AUROC, specificity matrix, combined-vs-single, correlation, trace examples) returning one report dict; `report/html.py` + `viz/svg.py` render that dict to a self-contained HTML report and standalone SVGs; `cli.py` wires it together.

The expected validation outcome — and a regression signal if it breaks — is: **targeted detection AUROC ≈ 1.0 per probe, ≈ 0.5 off-axis, combined detector > best single probe, weak cross-probe correlation.** Several tests assert exactly this.

## Constraints / conventions

- **Runtime dependency is numpy only.** Metrics (`metrics.py`), charts (`viz/svg.py`), and the HTML report are hand-rolled on purpose — do not add scipy/sklearn/matplotlib to runtime deps. `anthropic` and `pytest` are optional extras.
- **Determinism is load-bearing.** The synthetic model's randomness is a hashed function of `(seed, problem.id, keys, trial)` — never use `random`/unseeded `np.random`. Problem generators and the validation are seeded; the same seed must reproduce the same numbers.
- The arithmetic CoT step format is fixed — `"L op R = V"` — and `problems.py` owns its parse/execute/corrupt helpers. Probes operate on step *text*, not synthetic internals.
- Real-model code targets the Anthropic SDK: adaptive thinking for reasoning, thinking disabled for the "answer from given reasoning" probes; default to current model IDs (e.g. `claude-sonnet-4-6`, `claude-opus-4-8`). Consult the `claude-api` skill before changing integration code rather than relying on memory.
- The `anthropic_model.py` network call is isolated behind an injectable `transport` so the adapter is unit-tested without a key; preserve that seam.
