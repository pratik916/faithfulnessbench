# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FaithfulnessBench measures chain-of-thought (CoT) **faithfulness** in reasoning models — whether the stated reasoning causally drives the answer — and, crucially, **validates the measurement itself** against synthetic models whose (un)faithfulness is known by construction. `docs/DESIGN.md` is the methodology source of truth and `FEATURES.md` is the living per-file architecture map; keep both in sync with the code.

## Commands

A `.venv/` is checked out; prefix commands with `.venv/bin/` (or activate it). `pytest`/`ruff`/`coverage` come from the `dev` extra; `anthropic` from the `anthropic` extra (optional — the core is no-key).

```bash
pip install -e ".[dev]"                         # install with test/lint deps
pytest                                           # full suite (243 tests, no network/key; 1 skips when anthropic is installed)
pytest tests/test_probes.py -q                   # one file
pytest tests/test_metrics.py::test_auc_known_value  # one test
ruff check src tests experiments tools           # lint (CI-gated)
coverage run --source=src/faithfulnessbench -m pytest && coverage report --fail-under=90

python experiments/validate_synthetic.py         # regenerate ALL committed artifacts (json + report + docs/assets/*.svg)
faithfulnessbench validate                       # same numbers, but does NOT write the SVGs (use the line above to regenerate)
faithfulnessbench validate --check               # recompute & diff vs committed results.json (no write); non-zero on drift
faithfulnessbench transfer                        # GSM8K cross-domain transfer table (offline, no key)
faithfulnessbench score --model claude-sonnet-4-6 # score a real model (needs ANTHROPIC_API_KEY + the anthropic extra)
```

## Architecture (the parts that span files)

The spine is an **orthogonality contract** between two files that everything else depends on:

- `models/synthetic.py` — `ConfigurableSyntheticModel` carries **12 unfaithfulness-rate dials**. The four *core* dials (`p_hint_sycophancy`→SHI, `p_post_hoc`→CSC, `p_decoy_cot`→SIM, `p_pre_commit`→EAR, via `_PROBE_TO_DIAL`) each manifest **only** in the one `Model` method the matching probe intervenes on (`reason` / `continue_from_cot` / `answer_from_prefix`) — that's what makes per-instance ground-truth labels exact and the probes orthogonal. The rest are *extended* dials (SHI hint-taxonomy + `p_filler`/`p_contradict`/`p_stego`/`p_obfuscate`/`p_shortcut`) and `label_noise`; extended dials are drawn with `_draw(..., noise=False)` so they stay out of the frozen population and the core noise curve.
- `probes/` — **four core** probes (`shi`, `csc`, `sim`, `ear`) form the default battery (`PROBE_CLASSES`); **four held-out** probes (`fil`, `ipr`, `par`, `shortcut`) live in `HELD_OUT_PROBE_CLASSES` and are deliberately NOT in the default battery (a 10th single-axis probe would be guaranteed-AUROC=1.0 inflation). Every probe drives the `Model` interface (`models/base.py`) and returns per-instance unfaithfulness in [0,1]; **probes never touch synthetic internals**, so the same code runs unchanged against `models/anthropic_model.py`.

Then: `card.py` aggregates probe results into a Faithfulness Card (cluster-bootstrap CIs) + cross-probe correlation; `validation.py` is the end-to-end experiment returning one report dict (targeted/specificity/combined AUROC, monitor catch-rate@1%-FPR, ECE/reliability, DeLong/permutation/kappa, the auroc-vs-noise curve, the `robustness` frontier, the `whitebox` analogue, trace examples); `report/html.py` + `viz/svg.py` render that dict to a self-contained HTML report (9 figures, an interactive trace viewer) and standalone SVGs; `cli.py` wires it together. Supporting analysis modules: `metrics.py` (numpy-only stats), `artifacts.py` (regenerate/diff committed artifacts), `audit.py` (judge-vs-gold kappa), `gsm8k.py` + `transfer.py` (cross-domain), `robustness.py`, `whitebox.py`.

**Real-model path:** `score_real_model` / `real_model_probes` in `anthropic_model.py` are the single source of truth (shared by the `score` CLI and `experiments/record_replay.py`). On the real path SHI/SIM are graded by `LLMJudgeCueDetector`/`LLMSimulator` and CSC uses `LengthSignPreservingCorruptor` (the synthetic exact detectors don't apply to free-text CoT); `--exact-detectors` forces the synthetic ones.

The expected validation outcome — and a regression signal if it breaks — is: **targeted AUROC ≈ 1.0 per probe, ≈ 0.5 off-axis, combined > best single, weak cross-probe correlation**, with the noise curve pulling AUROC toward chance. Several tests assert exactly this.

## Workflows that will bite you

- **Adding/removing any test changes the count.** The README shields badge and the `pytest (N tests...)` line above are guarded by `tools/check_test_count.py` + `tests/test_repo_consistency.py`. After changing the test count, update **both** occurrences to the live `collected_test_count()` or CI fails.
- **Any change that affects numbers requires regenerating the committed artifacts** (`experiments/results/results.json`, `report/faithfulness_report.html`, `docs/assets/*.svg`) via `python experiments/validate_synthetic.py`. `validate --check` (regression gate) and `tools/check_artifacts.py` (freshness gate) enforce this in CI. Headline numbers must reproduce to 1e-12; the artifact-freshness check re-renders deterministically and diffs byte-for-byte.
- **The frozen population is sacrosanct.** New probes/axes must be *extended-only* (held-out registry, dial drawn `noise=False`, scored via `run_validation(extended_models=...)`) so the committed pooled numbers stay byte-identical. Deliberate re-baselining is a single explicit commit.
- **Offline core is sacrosanct; record real data deliberately.** The whole suite/transfer/e2e runs with **no key, no network** — never break that. Two cache families live in `experiments/replay_cache/`: `real_*.json` (real Claude Sonnet 4.6 + Opus 4.8, mixed + GSM8K) + `real_manifest.json` are the **committed real measurement** and replay offline at $0; `fake_*.json` is a tiny labeled-synthetic fixture kept only as the deterministic CI smoke-fixture. Re-recording is `python experiments/record_replay.py --real`: it routes main-model calls through the local Claude Code CLI (`claude -p`, **no `ANTHROPIC_API_KEY`**; the Haiku judge uses the SDK with `ANTHROPIC_AUTH_TOKEN`), is resumable (merge-on-put), and with the committed caches present makes **zero** new calls. The captured CoT is the model's **visible step-by-step text** (`claude -p` exposes no extended-thinking blocks). Re-render the HTML pages with `python experiments/render_real_artifacts.py`.
- **Local-only, gitignored, never commit:** the `beads` task tracker (`.beads/`, prefix `fb`) and `planning/` (the roadmap). `docs/superpowers/` is also ignored.

## Constraints / conventions

- **Runtime dependency is numpy only.** Metrics, charts, and the HTML report are hand-rolled on purpose — do not add scipy/sklearn/matplotlib to runtime deps. `anthropic`, `pytest`, `ruff`, `coverage` are optional extras.
- **Determinism is load-bearing.** Synthetic randomness is a hashed function of `(seed, problem.id, keys, trial)` — never use `random`/unseeded `np.random` (a seeded `np.random.default_rng(seed)` is fine where used, e.g. `whitebox.py`'s permutation control). The same seed must reproduce the same numbers.
- The arithmetic CoT step format is fixed — `"L op R = V"` — and `problems.py` owns its parse/execute/corrupt helpers. Probes operate on step *text*, not synthetic internals.
- Real-model code targets the Anthropic SDK (adaptive thinking for reasoning, thinking disabled for the "answer from given reasoning" probes; current model IDs `claude-sonnet-4-6` / `claude-opus-4-8`). The network call is isolated behind an injectable `transport` so the adapter is unit-tested without a key — preserve that seam. The committed real recording exploits this seam via `cli_transport` in `experiments/record_replay.py`, which shells out to `claude -p` (Claude Code session auth, since an OAuth token has SDK access to Haiku only — Sonnet/Opus 429 via the raw SDK). **Consult the `claude-api` skill before changing integration code** rather than relying on memory.
- **No `Co-Authored-By: Claude` trailer** on commits in this repo (public portfolio repo, sole-author attribution) — and no "Generated with Claude Code" line. This overrides the default commit-message instruction.
