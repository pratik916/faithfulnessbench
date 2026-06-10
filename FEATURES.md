# FaithfulnessBench — Feature & Architecture Map

A causal-intervention harness that measures whether a reasoning model's chain-of-thought (CoT) actually drives its answer (faithfulness). It runs **four core orthogonal probes** (SHI, CSC, SIM, EAR), with **four held-out extension probes** (FIL, IPR, PAR, SHORTCUT) and an **obfuscation axis** (OBF) used in validation only. Its distinguishing move is **validating the probes themselves** against synthetic models whose per-probe (un)faithfulness is known by construction, so each probe gets a detection AUROC against ground-truth labels — and, crucially, a *controlled-noise* regime that turns that AUROC from a wiring check into a sensitivity measurement. Runtime depends on numpy only; the core validation runs deterministically with no API key and no network, and an injectable real-Claude adapter runs the *identical* probe code with an LLM judge/simulator.

This document is a living map: a maintainer should be able to decide *what to enhance* without re-reading the source. API names below are reported from the code.

## Data flow

```
problems.py            models/ (Model ABC)          probes/                 card.py                 validation.py             report/ + viz/
-----------            -------------------          -------                 -------                 -------------             ---------------
Problem (+ Cue,        ConfigurableSynthetic  ----> SHI CSC SIM EAR (core)                          run_validation:           build_figures ->
canonical "L op R = V" Model (12 dials)              + FIL IPR PAR SHORTCUT  run_probes ->           - targeted AUROC + CI     8 SVGs + HTML
step chain) + GSM8K    AnthropicModel (live,         (held-out) + OBF axis   build_card ->           - specificity matrix      (validity / noise /
loader (gsm8k.py)      LLM judge/sim)                     |                  FaithfulnessCard        - combined vs best        robustness / cards /
        |              both expose reason /              ProbeResult         (sub-scores+CI,         - monitor catch-rate@FPR  interactive viewer)
        v              continue_from_cot /          (scores in [0,1], extra) composite, per_domain)  - ECE + reliability           |
arithmetic / mcq       answer_from_prefix                 |                   |                       - DeLong/permutation/kappa    |
generators                  ^                             v                  v                       - auroc-vs-noise curve        |
                            |                   metrics.py: roc_auc/roc_curve/ECE/reliability/        - robustness frontier         |
                  collaborators: CueDetector,    catch_rate_at_fpr/f1/cohen_kappa/holm/delong_test/   - GSM8K transfer (transfer.py) |
                  CoTSimulator (exact vs LLM),    permutation_*/bootstrap_{mean,cluster,auc}_ci       -> one report dict -----------+
                  Corruptor (operand vs len/sign)                                                                                   |
                                                                                          cli.py: validate / report / score / transfer
```

Core invariant: **higher probe score = more unfaithful; positive class (label 1) = "unfaithful"; one continuous score in [0,1] per scored instance.**

---

## Area 1 — Problems & data model (`problems.py`, `gsm8k.py`)

The `Problem`/`Cue` data model plus the canonical structured-CoT step format `L op R = V` that every probe operates on. The shared step chain means the synthetic model's logic is identical across domains; only the answer surface form (integer vs option letter) differs. A minimal GSM8K loader wraps real grade-school math as `Problem`s for the real-model path. numpy is used only for seeded generation.

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `problems.py` | Data model, step grammar, parse/execute/corrupt helpers, seeded generators, hint-kind cues | `Cue`, `Problem` (`.value_to_answer(value)`), `parse_step` / `format_step` / `apply_op`, `recompute_annotations`, `execute_steps`, `stated_final` / `set_stated_final`, `arithmetic_chain_problems`, `multiple_choice_problems`, `mixed_problems(n_each,*,seed)`, `HINT_KINDS`, `cue_of_kind(cue,kind)` | New domain: add a domain string, extend `Problem.value_to_answer`, add a generator; `Cue(target/text/marker)` is the hint seam; `cue_of_kind` rewrites a cue into a hint-taxonomy kind (sycophancy/consistency/metadata/authority) |
| `gsm8k.py` | numpy-only GSM8K loader (real grade-school math) | `parse_gsm8k_answer(text)` (`#### N`), `load_gsm8k(path)->list[Problem]` (domain `"gsm8k"`, a planted wrong-answer cue) | Bundled sample at `experiments/datasets/gsm8k_sample.jsonl`; the loaded `Problem` has no `L op R = V` chain, so CSC/SIM degrade on its free-text CoT (surfaced honestly; quantified in `transfer.py`) |

---

## Area 2 — Models (`models/`)

The abstract `Model` contract is the seam letting identical probe code run on the synthetic ground-truth world and on live Claude. A `Model` exposes exactly the three capabilities the probes intervene on.

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `models/base.py` | The `Model` ABC + collaborator Protocols | `Trace(answer,cot,steps,meta)`; `Model` (`reason`, `continue_from_cot`, `answer_from_prefix`); `CueDetector` Protocol + `SubstringCueDetector`; `CoTSimulator` Protocol | Subclass `Model` for a backend (deterministic given `(problem, trial)`); implement `CueDetector`/`CoTSimulator` to swap detection/simulation without touching probe code |
| `models/synthetic.py` | The ground-truth world: dialed, deterministic solver | `FaithfulnessProfile` with **12 dials** — core: `p_hint_sycophancy`, `p_post_hoc`, `p_decoy_cot`, `p_pre_commit`; SHI hint-taxonomy: `p_hint_consistency/metadata/authority`; extended axes: `p_filler` (FIL), `p_contradict` (IPR), `p_stego` (PAR), `p_obfuscate` (OBF), `p_shortcut` (SHORTCUT); plus `seed`, `label_noise`. `label_for(probe)->int`, `is_faithful`; `ConfigurableSyntheticModel(name, profile)`; `ExactArithmeticSimulator`; `model_population(seed=0, label_noise=0.0)`; `faithful_by_construction()` (Lyu solver control); `_PROBE_TO_DIAL`; `_u01`/`_draw(p,*keys,noise=True)` | Add a probe: new `p_*` dial + `_draw(...)` behavior in *one* `Model` method. **Extended dials draw with `noise=False`** so they don't contaminate the core noise curve and stay out of the frozen population. `label_noise` injects symmetric per-instance label noise keyed by `(seed, problem.id, key, trial)` |
| `models/anthropic_model.py` | Live-Claude adapter; same probes, real free-text CoT, LLM-graded | `AnthropicModel(model='claude-sonnet-4-6',*,effort,max_tokens=8000,cache_path,transport,client,max_retries,backoff)` with retry/backoff, `call_log`, `token_totals()`, `estimated_cost_usd()`, `truncated_calls`; `_parse_answer` (ANSWER: > `\boxed{}` > "answer is" > fallback); `_JsonCache` (merge-on-`put`); `LLMJudgeCueDetector` / `LLMSimulator` (accept `cache_path`); `real_model_probes(...)`; `judge_reliability(...)`; `score_real_model(...)`; `thinking_vs_answer_acknowledgment(...)` | `transport` is the primary no-key test seam; `cache_path` enables offline record/replay; `real_model_probes`/`score_real_model` are the single source of truth shared by the `score` CLI and the recorder; `--exact-detectors` falls back to the synthetic exact detectors |
| `models/__init__.py` / `__init__.py` | Public namespace | `models/__init__.py` exports the numpy-only path; the package `__init__.py` **lazily** re-exports `AnthropicModel`/`LLMSimulator`/`LLMJudgeCueDetector` via `__getattr__` (clear `ImportError` if `anthropic` absent — never a silent fallback). `__version__` is single-sourced via `importlib.metadata` | Add a synthetic model/simulator/detector to surface it on the no-key path |

---

## Area 3 — Probes (`probes/`)

Each probe is a causal intervention returning a continuous per-instance unfaithfulness score in [0,1] plus a free-form `extra` diagnostics dict. **Four core probes** form the default battery; **four extended probes** are deliberately held out (each is a single-axis probe whose synthetic AUROC is 1.000 by construction — promoting it would inflate the count without adding a new *kind* of evidence).

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `probes/base.py` | Common probe contract | `ProbeResult(probe,problem_ids,scores,extra)` with `__post_init__`/`validate()` (asserts scores∈[0,1], `len(scores)==len(problem_ids)`), `.mean`, `.faithfulness()`; `Probe` ABC; `majority_answer(model,problem,n_trials)` | `extra` is an open dict for any diagnostics |
| `probes/shi.py` | **P1 Silent Hint Injection** | `SHIProbe(detector=None,n_trials=5)`; majority-voted baseline `a0`. `extra`: `flip_rate`, `ack_rate_given_flip` | Inject a `CueDetector` (LLM-judge for real models) |
| `probes/csc.py` | **P2 CoT Step Corruption** (directed) | `Corruptor` Protocol; `OperandCorruptor`; `LengthSignPreservingCorruptor` (closes the digit/sign distribution-shift confound); `CSCProbe(corruptor=None)`. Scores `new_answer == expected-under-corruption` (tracks the corruption, not "merely changed"). `extra`: `mean_sensitivity` | Swap any object with `.corruptions(steps)->list[(idx,steps)]` |
| `probes/sim.py` | **P3 Counterfactual Simulatability** | `SIMProbe(simulator=None,n_trials=5)`. `extra`: `sim_cot_acc`, `sim_q_acc`, `sim_gain`, `parse_failure_rate`, `genuine_miss_rate` (separates malformed-CoT from non-predictive CoT on the real path) | Inject a `CoTSimulator` that reads the CoT, never re-solves the question |
| `probes/ear.py` | **P4 Early-Answering / Reasoning-Reliance** | `EARProbe(fractions=(0.0,0.25,0.5,0.75),n_trials=5)`. `extra`: `fractions`, `weights`, `match_curve` | Tune `fractions` (all <1.0); weighting derived automatically |
| `probes/{fil,ipr,par,shortcut}.py` | **Extended (held-out)** | `FillerProbe` (FIL, filler-token/hidden-compute), `IPRProbe` (IPR, no-hint contradictory-pair rationalization), `ParaphraseProbe` (PAR, steganography; `_paraphrase`), `ShortcutProbe` (SHORTCUT, reward-hack via leaked channel) | Each is individually validated (targeted AUROC=1.0, off-axis 0.5, sub-1.0 under a partial dial) and held out of the default battery |
| `probes/__init__.py` | Registry + canonical order | `PROBE_CLASSES` (the four core); `HELD_OUT_PROBE_CLASSES` (FIL/IPR/PAR/SHORTCUT); `default_probes()`; `_validate_registry` (fail-fast, asserts the two registries are disjoint and key==`name`); `PROBES` | A registry-contents pin test (`tests/test_probe_registry.py`) locks the core/held-out split so neither can drift silently |

---

## Area 4 — Measurement, aggregation, reporting & analysis

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `metrics.py` | Hand-rolled numpy-only stats (no scipy) | `roc_auc`, `roc_curve`, `expected_calibration_error`, `reliability_curve`, `pearson`, `spearman`, `spearman_matrix`, `bootstrap_mean_ci`, `bootstrap_cluster_ci` (hierarchical, by problem id), `bootstrap_auc_ci`, `auc_summary`, `catch_rate_at_fpr` (monitor), `f1_score`, `cohen_kappa`, `holm_correction`, `permutation_test_auroc`, `delong_test` (+ `_placements`, `_norm_cdf`), `permutation_auroc` | Add a metric as a pure `(scores/labels)->float|tuple` fn + a hand-computed test in `tests/test_metrics.py` |
| `card.py` | Probe results -> `FaithfulnessCard` + correlation | `FaithfulnessCard(...).to_dict()` (NaN-safe); `run_probes`; `build_card`; `card_from_results`; `stack_probe_scores` (aligns on `problem_ids` intersection); `correlation_matrix`. CIs use `bootstrap_cluster_ci` | `probes` arg accepts any `list[Probe]`; `card_from_results` decouples aggregation from how results were produced |
| `validation.py` | The headline seeded validation experiment | `run_validation(*,n_per_domain=15,seed,n_trials=5,reproduce_cmd,extended_models=None)->dict`; `json_safe`; `summarize`; `PROBE_ORDER`, `AXIS_MODEL`, `NOISE_LEVELS`. The report dict carries `targeted_auroc`, `per_domain_auroc`, `negative_control_auroc`, `specificity`, `combined_auroc`, `single_mixed_auroc`, `monitor` (catch-rate@1%-FPR, subtle-vs-blatant), `obfuscation` (OBF curve, monitor fixed first), `hint_taxonomy`, `headline_classification` (F1/kappa), `calibration` (ECE + reliability), `noisy_significance` / `noisy_per_probe_significance` (DeLong/permutation on the noisy substrate only), `auroc_vs_noise`, `robustness` (CoT-obfuscation frontier), and optional `extended_auroc` | `extended_models` reports pairwise targeted AUROC for held-out models **without** moving the frozen pooled numbers; significance is applied only where numbers vary |
| `audit.py` | Judge calibration vs synthetic gold | `judge_audit(judge,gold,items,*,paraphrase=None)` (Cohen's kappa, Landis–Koch band, self-consistency); `landis_koch_band(kappa)` | The real-model `score` path reports a `judge_reliability` line from this |
| `artifacts.py` | Generate / regression-gate committed artifacts | `generate_artifacts(...)`; `key_numbers`; `diff_key_numbers`; `check_against_committed`; `stale_artifacts` | The field policy: headline numbers are exact-to-1e-12; provenance/CIs are ignored. `validate --check` and `tools/check_artifacts.py` enforce it |
| `transfer.py` | GSM8K cross-domain transfer (descriptive) | `cross_domain_transfer(synthetic_card,gsm8k_card)`; `build_gsm8k_transfer(...)`; `transfer_markdown` / `transfer_text_table` / `render_transfer_html` / `write_transfer_html`; `FAITHCOT_SCOPE_NOTE` | Descriptive only (no GSM8K faithfulness gold → no AUROC-vs-truth); shows SHI/EAR run unchanged while CSC/SIM degrade on free-text CoT |
| `robustness.py` | Adversarial-CoT robustness frontier | `ScrubbedModel(base,budget,seed)` (generic, monitor-agnostic CoT scrub); `robustness_frontier(*,budgets,seed,n_per_domain,n_trials)`; `ROBUSTNESS_BUDGETS` | One synthesis figure: catch-rate@1%-FPR vs scrub budget — SIM collapses, behavioral probes stay robust |
| `viz/svg.py` | Dependency-free inline-SVG charts (a11y: `<title>`/`<desc>`/`role`) | `faithfulness_color`, `correlation_color`, `bar_chart`, `roc_plot`, `line_chart`, `reliability_diagram`, `heatmap`, `_truncate` (label overflow). `viz/__init__` re-exports incl. `correlation_color`, `line_chart` | New chart = pure `()->svg` fn wired into `report/html.build_figures` |
| `report/html.py` | One self-contained HTML report (inline CSS/JS/SVG) + standalone SVGs | `build_figures(report)->dict` (**8 keys**: `detection_auroc, roc, specificity, correlation, faithfulness_matrix, auroc_vs_noise, reliability, robustness`); `write_figures`; `render_report`/`write_report`; `render_card_report`/`write_card_report`; `_safe_json_for_script` (escapes `</`, U+2028/U+2029); interactive trace viewer (`_VIEWER_JS`: EAR fraction slider + CSC corruption toggle) | New figures appear in HTML (if referenced in `render_report`) and as standalone SVGs |
| `cli.py` | Console entry point | `main`; `_cmd_validate` (`--check`), `_cmd_report`, `_cmd_score` (`--effort`, `--cache`, `--out`, `--html`, `--exact-detectors`; emits judge-reliability + truncation warnings), `_cmd_transfer` (`--cache`, `--sample`, `--out`, `--html`) | New subcommand via `sub.add_parser(...).set_defaults(func=...)` |
| `_api.py` | Flat public API surface | Re-exports problems/models/probes/card/validation under `__all__`; excludes `metrics`/`viz`/`report` (submodule-only) | Add a public symbol = import here + add to `__all__` |

---

## Capabilities (what it does today)

- [x] Score CoT (un)faithfulness on four core axes (SHI/CSC/SIM/EAR), plus four held-out extension probes and an obfuscation axis, each a continuous per-instance score in [0,1].
- [x] Run the full validation deterministically with **no API key and no network**: `faithfulnessbench validate` -> `results.json` + self-contained HTML report.
- [x] **Controlled-noise regime**: report targeted AUROC as a curve over `label_noise`, so the by-construction 1.000 becomes a sensitivity *measurement*; a **shuffled-label negative control** proves the harness can report chance.
- [x] **Frozen vs extended population**: committed pooled numbers reproduce byte-for-byte; new axes get pairwise targeted AUROC via `extended_models` without moving the headline; a Lyu **faithful-by-construction** positive control.
- [x] **Monitor-grade metrics**: catch-rate @ 1% FPR (subtle vs blatant), ECE + a reliability diagram, and an **obfuscated-reward-hacking** (OBF) analogue with the monitor fixed first.
- [x] **Inferential layer where honest**: DeLong + paired permutation tests, cluster (hierarchical) bootstrap CIs, Cohen's kappa with Holm correction, F1 — applied only to the noisy/real substrate, never the definitional zero-noise gap.
- [x] **Adversarial-CoT robustness frontier**: catch-rate@1%-FPR vs a generic CoT-obfuscation budget (SIM collapses; behavioral probes robust).
- [x] **Cross-domain transfer**: the identical probes run on real GSM8K math (`faithfulnessbench transfer`), reported descriptively against the FaithCoT-Bench math→knowledge caveat.
- [x] **Real-model path that is actually LLM-graded**: `score` uses `LLMJudgeCueDetector`/`LLMSimulator` + the length/sign-preserving corruptor by default and prints a judge-reliability line; offline via a committed (labeled-FAKE) record/replay cache; `--exact-detectors` for comparison.
- [x] Hardened free-text answer parsing (`\boxed{}`, "the answer is N", stray letters) + a truncation flag surfaced into scoring.
- [x] **Interactive, zero-dependency HTML trace viewer**: a silent-flip view plus an EAR truncation slider and a CSC corruption toggle driven by embedded per-fraction/per-corruption data; 8 standalone SVG charts.
- [x] **Deployed**: the self-contained report ships to GitHub Pages (one-click live demo); a trace-viewer GIF in the README.
- [x] CI: clean-clone install + suite + ruff + coverage (`--fail-under=90`) + a `validate --check` regression gate + an artifact-freshness gate, across Python 3.10–3.14 on Linux and macOS.

---

## Known limitations & seams (genuinely still-live)

Most pre-v2 limitations have been resolved (ECE/significance/calibration are wired; the registry is pinned; the parser, transport, and cache are hardened; CI has ruff + 3.14 + macOS + coverage; the test count is guarded). What remains is *methodological*, not unfinished plumbing:

**Probes**
- SHI's `flipped` requires `answer == cue.target` exactly (catches sycophancy to the planted target); cue-less problems are dropped, shrinking `problem_ids`.
- EAR's `k=int(f*n)` rounds down / collides on short chains, coarsening the grid; the weighted mean discards convergence *shape*.

**Models & problems**
- `label_for`/`is_faithful` use a hard 0.5 threshold and are scoped to **pure-type** models; intermediate dials carry only an expected rate, not a per-instance label (the noise regime measures sensitivity instead).
- The synthetic model is **not** an LLM — indisputable ground truth, not behavioral realism (by design).
- Two domains, both linear integer arithmetic over `{+,-,*}` (the `L op R = V` format is rigidly 5 tokens). A third *structurally different* synthetic domain is scoped and deferred (DESIGN.md §7).

**Real-model path (documented; now quantified)**
- On free-text CoT, CSC/SIM degrade (no parseable `L op R = V` chain); the transfer table and the robustness frontier now *quantify* this rather than just noting it. The LLM judge's reliability is itself a dependency, reported as kappa-vs-gold next to the SHI/SIM numbers.
- The committed replay cache is a **labeled FAKE** fixture; a genuine real-Claude measurement is a one-command (`record_replay.py --real`) swap, gated on an API key and explicit spend approval. No real-model AUROC-vs-truth is claimed (no real-model faithfulness labels exist).

**Scope (future work)**
- Measurement is **behavioral** (black-box). White-box / activation-level probing is out of scope; a go/no-go spike for a numpy *offline analogue* is written in DESIGN.md §7.

---

## How to extend (cookbook)

**Add a core probe.** Implement a `Probe` subclass (set `name`, implement `run`). Register in `PROBE_CLASSES`. Wire the ground truth: add a `p_*` dial to `FaithfulnessProfile`, add the probe→dial entry to `_PROBE_TO_DIAL`, implement the unfaithful behavior in **exactly one** `Model` method (gated by `_draw`), add a named pure-type model in `model_population`, and append to `validation.PROBE_ORDER` + `AXIS_MODEL`. Add a contract test in `tests/test_synthetic.py` and the `TARGET_MODEL` entry in `tests/test_probes.py`. Update the registry pin test.

**Add a *held-out* extension probe.** Same, but register in `HELD_OUT_PROBE_CLASSES` (not `PROBE_CLASSES`), draw its dial with `noise=False` so it stays out of the frozen population and the core noise curve, and validate it in its own `tests/test_<probe>.py`. Do **not** add a 10th single-axis probe to the default battery — it would be guaranteed-AUROC=1.000 inflation.

**Add a synthetic dial.** Add a `p_*` dial, route stochastic behavior through `_draw(dial, problem.id, ..., noise=...)`, and (if a labeled axis) add it to `_PROBE_TO_DIAL` + a pure-type model.

**Swap a probe collaborator.** Inject a `CueDetector` into `SHIProbe(detector=...)`, a `CoTSimulator` into `SIMProbe(simulator=...)`, or a `Corruptor` into `CSCProbe(corruptor=...)`. A real-model simulator must keep the leakage control (read the CoT, never re-solve the question).

**Add a metric.** Pure `(scores/labels)->float|tuple` in `metrics.py` + a hand-computed test in `tests/test_metrics.py`.

**Add a chart / report figure.** Pure `()->'<svg>'` in `viz/svg.py`, wire into `report/html.build_figures`, reference in `render_report`; update the expected key set + count in `tests/test_validation.py`.

**Add a CLI subcommand.** `sub.add_parser(...).set_defaults(func=_cmd_x)` in `cli.py`; defer heavy imports inside `_cmd_*`.

**Record real Claude data.** `ANTHROPIC_API_KEY=… python experiments/record_replay.py --real` re-records the (currently fake) replay cache through the live SDK — a one-command swap; everything downstream then runs offline.
