# FaithfulnessBench — Feature & Architecture Map

A causal-intervention harness that measures whether a reasoning model's chain-of-thought (CoT) actually drives its answer (faithfulness), across four orthogonal probes (SHI, CSC, SIM, EAR). Its distinguishing move is **validating the probes themselves** against synthetic models whose per-probe (un)faithfulness is known by construction, so each probe gets a detection AUROC against ground-truth labels. Runtime depends on numpy only; the core validation runs deterministically with no API key and no network, and an injectable real-Claude adapter runs the *identical* probe code.

This document is a living map: a maintainer should be able to decide *what to enhance* without re-reading the source. API names below are reported verbatim from the code.

## Data flow

```
problems.py            models/ (Model ABC)          probes/                 card.py                 validation.py             report/ + viz/
-----------            -------------------          -------                 -------                 -------------             ---------------
Problem (+ Cue,        ConfigurableSynthetic  ----> SHI  --\                                                                  build_figures ->
canonical "L op R = V" Model (dialed)               CSC   \                 run_probes ->           run_validation:           5 SVGs + HTML
step chain)            AnthropicModel (live)        SIM    >-- ProbeResult --> build_card ->         - targeted AUROC + CI     (validity /
        |              both expose reason /          EAR  --/   (scores in    FaithfulnessCard       - specificity matrix      orthogonality /
        v              continue_from_cot /                      [0,1], extra) (sub-scores+CI,        - combined vs best        disagreement /
arithmetic / mcq       answer_from_prefix                                     composite, per_domain) - correlation matrix      cards / trace
generators                  ^                                       |          |                     - per-model cards          viewer)
                            |                                       v          v                     - trace_examples              |
                  collaborators: CueDetector,             metrics.py: roc_auc / roc_curve /          -> one report dict ----------+
                  CoTSimulator (exact vs LLM)             ECE / pearson / spearman /                                              |
                                                          bootstrap_*_ci / spearman_matrix                          cli.py: validate / report / score
```

Core invariant: **higher probe score = more unfaithful; positive class (label 1) = "unfaithful"; one continuous score in [0,1] per scored instance.**

---

## Area 1 — Problems & data model (`problems.py`)

The `Problem`/`Cue` data model plus the canonical structured-CoT step format `L op R = V` that every probe operates on. The shared step chain means the synthetic model's logic is identical across domains; only the answer surface form (integer vs option letter) differs. numpy is used only for seeded generation.

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `src/faithfulnessbench/problems.py` | Data model, step grammar, parse/execute/corrupt helpers, seeded generators | `Cue`, `Problem` (`.value_to_answer(value)`), `parse_step` / `format_step` / `apply_op`, `recompute_annotations`, `execute_steps`, `stated_final` / `set_stated_final`, `first_operand_left`, `arithmetic_chain_problems(n,*,seed,length_range)`, `multiple_choice_problems(n,*,seed,length_range)`, `mixed_problems(n_each,*,seed)` | New domain: add a domain string, extend `Problem.value_to_answer`, add a generator; `Cue(target/text/marker)` is the hint seam (`marker` is the unique sentinel the exact detector keys on); `_make_chain` controls difficulty/operators; `length_range`/`seeds` size the dataset |

Step helpers `parse/format/apply/execute/stated_final/recompute_annotations` are the contract CSC corruption and the simulator build on.

---

## Area 2 — Models (`models/`)

The abstract `Model` contract is the seam letting identical probe code run on the synthetic ground-truth world and on live Claude. A `Model` exposes exactly the three capabilities the four probes intervene on.

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `models/base.py` | The `Model` ABC + collaborator Protocols | `Trace(answer,cot,steps,meta)`; `Model` (abstract `reason(problem,*,cue=None,trial=0)->Trace`, `continue_from_cot(problem,cot_steps,*,trial=0)->str`, `answer_from_prefix(problem,prefix_steps,*,trial=0)->str`); `CueDetector` Protocol (`mentions(cot,cue)->bool`); `SubstringCueDetector`; `CoTSimulator` Protocol (`predict(problem,cot_steps)->str`) | Subclass `Model` for a backend (must be deterministic given `(problem, trial)`); implement `CueDetector`/`CoTSimulator` to swap detection/simulation without touching probe code; `trial` kwarg is the stochastic-re-realization seam |
| `models/synthetic.py` | The ground-truth world: dialed, deterministic solver | `FaithfulnessProfile` (frozen: `p_hint_sycophancy`, `p_post_hoc`, `p_decoy_cot`, `p_pre_commit`, `seed`; `label_for(probe)->int`, `is_faithful` property); `ConfigurableSyntheticModel(name, profile)`; `ExactArithmeticSimulator`; `model_population(seed=0)`; private `_PROBE_TO_DIAL = {SHI:p_hint_sycophancy, CSC:p_post_hoc, SIM:p_decoy_cot, EAR:p_pre_commit}`; helpers `_u01(*keys)` / `_draw(p,*keys)` | Add a probe: new `p_*` dial + entry in `_PROBE_TO_DIAL` + behavior in *exactly one* `Model` method (gated by `_draw`) + named pure-type model in `model_population` + wiring in `validation.PROBE_ORDER`/`AXIS_MODEL`. `_decoy_value`/`_target_value` are per-domain rendering seams; `_u01`/`_draw` is the deterministic-randomness seam keyed by `(seed, problem.id, key, trial)` |
| `models/anthropic_model.py` | Live-Claude adapter; same probes, real CoT | `AnthropicModel(model='claude-sonnet-4-6',*,effort,max_tokens=8000,cache_path,transport,client)`; `Transport = Callable[[dict],tuple[str,str]]`; `_parse_answer(text,problem)`; `_cot_to_steps(cot)`; `_JsonCache(path)` (`.get`/`.put`); `LLMJudgeCueDetector(...)`; `LLMSimulator(...)`; constants `SYSTEM_REASON`/`SYSTEM_COMMIT`; `_answer_kind` | `transport` is the primary seam (unit-test prompts/parsing with no key, or plug another SDK); `client` injects a prebuilt `anthropic.Anthropic()`/stub; `cache_path` enables offline record/replay; `SYSTEM_*`/`_answer_kind` are prompt-engineering seams; the thinking-config block in `_api_call` adapts to new model families (special-cases opus-4-7/4-8 summarized thinking) |
| `models/__init__.py` | Public namespace for the numpy-only path | `__all__`: `Model, Trace, CueDetector, CoTSimulator, SubstringCueDetector, ConfigurableSyntheticModel, ExactArithmeticSimulator, FaithfulnessProfile, model_population` | Add a synthetic model/simulator/detector here to surface it on the no-key path. **Deliberately omits `AnthropicModel`** so importing the package never requires the anthropic SDK |

---

## Area 3 — Probes (`probes/`)

Each probe is a causal intervention returning a continuous per-instance unfaithfulness score in [0,1] plus a free-form `extra` diagnostics dict.

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `probes/base.py` | Common probe contract | `ProbeResult(probe,problem_ids,scores:ndarray,extra={})`; `ProbeResult.mean` (property); `ProbeResult.faithfulness()` (=1-mean, nan if empty); `Probe` (ABC, class attr `name`); `Probe.run(model, problems, *, n_trials=None)->ProbeResult` (abstract) | Subclass `Probe`, implement `run`, set class attr `name`; `extra` is an open dict for any diagnostics with no schema change |
| `probes/shi.py` | **P1 Silent Hint Injection** — sycophancy without acknowledgment | `SHIProbe` (`name='SHI'`); `__init__(detector=None, n_trials=5)` (default `SubstringCueDetector`); `run(...)`. `extra`: `flip_rate`, `ack_rate_given_flip`, `n_trials` | Inject a custom `CueDetector` (e.g. LLM-judge for real models); `n_trials` per-call; cue styles defined in `problems.py` via `Problem.cue` |
| `probes/csc.py` | **P2 CoT Step Corruption** — is the chain causally load-bearing | `OperandCorruptor(deltas=(3,-3,4,-4,5,-5))`, `.corruptions(steps)->list[(idx,steps)]`; `CSCProbe` (`name='CSC'`), `__init__(corruptor=None)`, `run(...)` (`n_trials` accepted but **unused**). `extra`: `mean_sensitivity`. Score = `1 - sensitivity` | Swap any object with `.corruptions(steps)->list[(idx,steps)]`; `deltas` is the magnitude/sign knob; coherence delegated to `problems.py` helpers; intervention point is `continue_from_cot` |
| `probes/sim.py` | **P3 Counterfactual Simulatability** — does the CoT predict the answer | `SIMProbe` (`name='SIM'`); `__init__(simulator=None, n_trials=5)` (default `ExactArithmeticSimulator`); `run(...)`. `extra`: `sim_cot_acc`, `sim_q_acc`, `sim_gain` (=cot-q). Score = `1 - cot_hit` | Inject a custom `CoTSimulator` that **reads the CoT, never re-solves the question** (structural leakage control); a new domain needs a simulator that extracts a conclusion from that domain's CoT |
| `probes/ear.py` | **P4 Early-Answering / Reasoning-Reliance** — how early the answer locks | `EARProbe` (`name='EAR'`); `__init__(fractions=(0.0,0.25,0.5,0.75), n_trials=5)` (computes `_weights=(1-f)/sum`); `run(...)`. `extra`: `fractions`, `weights`, `n_trials`. Score = small-f-weighted match-vs-fraction | Tune `fractions` (all <1.0) and `n_trials`; weighting derived automatically; intervention is `answer_from_prefix`; per-fraction trial seed `t*1000+int(f*100)` decorrelates stochastic models |
| `probes/__init__.py` | Registry + canonical reporting order | `PROBE_CLASSES = {'SHI':SHIProbe,'CSC':CSCProbe,'SIM':SIMProbe,'EAR':EARProbe}`; `default_probes()->list[Probe]`; `PROBES` (module-level instances); re-exports base types + `OperandCorruptor` | Register a probe by appending to **both** `PROBE_CLASSES` and `default_probes()` (the single choke point the rest of the system reads). Note: `PROBES` is built at import time, so default constructors with cost run on import |

---

## Area 4 — Measurement, aggregation & reporting (`metrics.py`, `card.py`, `validation.py`, `viz/`, `report/`, `cli.py`, `_api.py`, `__init__.py`)

| Path | Purpose | Key public API | Extension points |
|---|---|---|---|
| `metrics.py` | Hand-rolled numpy-only stats | `_rankdata(a)`; `roc_auc(scores,labels)`; `roc_curve(...)->(fpr,tpr)`; `expected_calibration_error(probs,labels,n_bins=10)`; `pearson(x,y)`; `spearman(x,y)`; `spearman_matrix(columns)->(names,matrix)`; `bootstrap_mean_ci(values,*,n_resamples=2000,alpha=0.05,seed=0)`; `bootstrap_auc_ci(...)`; `auc_summary(scores,labels,*,seed=0)->{auc,ci_lo,ci_hi}` | Add a metric as a pure `(scores/labels)->float|tuple` fn + a hand-computed test in `tests/test_metrics.py`; `spearman_matrix` takes an arbitrary named-column mapping so new vector sets flow through unchanged; `bootstrap_*` kwargs tune CI behavior |
| `card.py` | Raw probe results -> `FaithfulnessCard` + cross-probe correlation | `FaithfulnessCard(model_name, probe_scores, composite_faithfulness, per_domain, extras, n_problems)` (`.to_dict()`, NaN-safe); `run_probes(model,problems,probes=None,*,n_trials=5)`; `build_card(...,bootstrap_seed=0)`; `card_from_results(model_name,results,problems,*,bootstrap_seed=0)`; `stack_probe_scores(per_model_results,probe_names)`; `correlation_matrix(per_model_results,probe_names)->(labels,matrix)` | `probes` arg accepts any `list[Probe]`; `card_from_results` decouples aggregation from how results were produced (real vs synthetic); `stack_probe_scores` aligns on `problem_ids` (intersection) so probes that drop problems still align; `per_domain` is computed generically from `Problem.domain` |
| `validation.py` | The headline seeded validation experiment | `run_validation(*,n_per_domain=15,seed=0,n_trials=5,reproduce_cmd=...)->dict`; `json_safe(obj)`; `summarize(report)->str`; constants `PROBE_ORDER=['SHI','CSC','SIM','EAR']`, `AXIS_MODEL` (probe->single-axis model) | Adding a probe means appending to `PROBE_ORDER` + `AXIS_MODEL`; targeted-AUROC/specificity/combined/correlation loops are driven off `PROBE_ORDER` and extend automatically; the report dict is the stable contract for renderers; `reproduce_cmd` stamps provenance |
| `viz/svg.py` | Dependency-free inline-SVG charts | `faithfulness_color(v)`; `correlation_color(v)`; `bar_chart(items,*,title,vmax=1.0,baseline=None,width,height,color,value_fmt)`; `roc_plot(curves,*,title,width,height)`; `heatmap(row_labels,col_labels,matrix,*,title,color_fn=faithfulness_color,cell=64)` (`viz/__init__` re-exports `bar_chart,heatmap,roc_plot,faithfulness_color`; `correlation_color` reachable only via `svg.`) | `heatmap` takes an arbitrary `color_fn`; width/height/cell/title/color kwargs tune layout; a new chart type is another pure `()->svg` fn wired into `report/html.build_figures`; `value_fmt` controls number formatting |
| `report/html.py` | One self-contained HTML report (inline CSS/JS/SVG) + standalone SVGs | `build_figures(report)->dict[str,str]` (keys: `detection_auroc, roc, specificity, correlation, faithfulness_matrix`); `write_figures(report,directory)`; `render_report(report)->str`; `write_report(report,path)`; module: `_PALETTE`, `_CSS`, `_VIEWER_JS`, `_esc` | `build_figures` is the single place charts are assembled; new figures appear in HTML (if referenced in `render_report`) and as standalone SVGs; `render_report` builds an ordered list of HTML-string sections; trace viewer is data-driven from `report['trace_examples']`; `_PALETTE` sets ROC colors |
| `cli.py` | Console entry point (`faithfulnessbench` / `faithbench`) | `main(argv=None)->int`; `_cmd_validate` (`-n=20,--seed,--trials,--report,--json`); `_cmd_report` (`--json,--report`); `_cmd_score` (`--model=claude-sonnet-4-6,-n,--seed,--trials,--effort {low,medium,high,max},--cache,--out`) | New subcommand via `sub.add_parser(...).set_defaults(func=...)`; heavy imports deferred per-command so `validate` works without anthropic; `score --cache` wires record/replay; `report` decouples rendering from running |
| `_api.py` | Flat public API surface | Re-exports under `__all__`: problems, models, probes, `FaithfulnessCard/build_card/card_from_results/run_probes`, `run_validation` | Add a public symbol = import it here + add to `__all__`; **deliberately excludes** `metrics`, `viz`, `report`, and `correlation_matrix`/`stack_probe_scores`/`spearman_matrix` (reachable only via submodule imports) |
| `__init__.py` | Top-level init; degrades gracefully | `__version__='0.1.0'`; always-available `metrics`; `* from ._api` inside a `try/except` | The defensive `try/except` tolerates partial/missing submodules; always-on `metrics` is the guaranteed numpy-only core; version bumped here |

---

## Capabilities (what it does today)

- [x] Score CoT (un)faithfulness on four orthogonal axes from one uniform interface (SHI / CSC / SIM / EAR), each a continuous per-instance score in [0,1].
- [x] Run the full validation experiment deterministically with **no API key and no network**: `faithfulnessbench validate` -> `results.json` + self-contained HTML report + console summary.
- [x] Dial in four orthogonal unfaithfulness modes independently via `FaithfulnessProfile`, each manifesting in exactly one `Model` method, giving per-instance ground-truth labels.
- [x] Instantiate a canonical pure-type model population (`faithful`, four single-axis, `fully_unfaithful`) via `model_population(seed)`.
- [x] Quantify each probe's **targeted detection AUROC** with a percentile bootstrap 95% CI; show a probe-vs-axis **specificity** heatmap (diagonal ~1.0, off-axis ~chance).
- [x] Compare a combined (mean-of-four) detector vs the best single probe on the mixed population ("a card, not a scalar").
- [x] Compute a pooled cross-probe **Spearman correlation matrix** (the disagreement finding).
- [x] Aggregate into a `FaithfulnessCard` per model: per-probe sub-score + CI, per-domain breakdown, transparent unweighted composite, NaN-safe JSON.
- [x] Generate seeded problem sets in two domains (arithmetic chains, 4-option MCQ) sharing one step chain, each carrying a planted wrong-answer cue.
- [x] Render an interactive, zero-dependency HTML trace viewer (baseline-vs-cued, silent-flip verdicts) + five standalone SVG charts.
- [x] Re-render HTML from an existing `results.json` without recomputing (`faithfulnessbench report`).
- [x] Score a live Claude reasoning model with the *same* probes (`faithfulnessbench score --model ... --effort ...`), with a record/replay JSON cache and LLM-graded collaborators (`LLMJudgeCueDetector`, `LLMSimulator`).
- [x] Unit-test the real adapter's prompt-building/parsing with **zero API cost** via an injectable `transport`.
- [x] CI proving clean-clone installability + suite + a `validate` smoke run across Python 3.10–3.13 on Linux.

---

## Known limitations & seams (where future work attaches)

**Probes**
- SHI/CSC take the baseline answer `a0` from a single `reason(trial=0)` draw; for a stochastic real model this is a noisy reference (EAR already recomputes `a0` per trial). *Seam: majority-vote/per-trial baseline.*
- SHI's `flipped` requires `answer == p.cue.target` exactly (only catches sycophancy to the planted target); cue-less problems are silently dropped, shrinking `problem_ids`.
- CSC is hard-wired to the arithmetic `L op R = V` grammar via `parse_step`; non-arithmetic domains yield zero corruptions and drop out. Sensitivity is binary "answer changed" — it does not verify the answer *tracks* the corrupted chain (a flip to an unrelated wrong answer counts as faithful). `n_trials` is ignored. Documented real-model confound: re-chaining propagates the operand delta so digit-counts/signs can change (distribution shift), inert only for the synthetic model.
- SIM scores raw CoT-only accuracy; `sim_gain` is uninformative for a near-perfect model (do not read as faithfulness). `ExactArithmeticSimulator` returns `'?'` (guaranteed miss) on an unparseable final step, conflating malformed-CoT with unfaithful-CoT. It tracks the *stated conclusion*, not the reasoning path.
- EAR: `k=int(f*n)` collides/rounds down on short chains, coarsening the grid; the weighted mean discards convergence *shape*. Real-model validity hinges on `answer_from_prefix` truly disabling re-reasoning.
- `base.py` enforces neither `scores ∈ [0,1]` nor `len(scores)==len(problem_ids)` (convention only); `ProbeResult` is per-instance only (per-trial/per-fraction raw data is collapsed and lost). Registration is split across `PROBE_CLASSES` + `default_probes()` with no sync check.

**Models & problems**
- `label_for`/`is_faithful` use a hard 0.5 threshold and are scoped to **pure-type** models; intermediate dials (e.g. 0.3) have no per-instance label, only an expected rate. `model_population` offers only single-axis + all-axis types (no graded/mixed population).
- Orthogonality depends on each dial touching exactly one method; the decoy-vs-silent-flip ordering in `reason()` is a subtle invariant to preserve. `_decoy_value` is a fixed heuristic (`final+7` / first non-final MCQ value), not adversarial.
- The synthetic model is **not** an LLM — indisputable ground truth, not behavioral realism (documented).
- Only two domains, both linear integer arithmetic over `{+,-,*}` (no division/branching/word problems); the `L op R = V` format is rigidly 5 tokens (no precedence). Cues always point at a plausible-wrong value (no correct/authority/social cues). MCQ is fixed at 4 options. `first_operand_left` is dead code; `value_to_answer` silently returns `'?'` for out-of-set MCQ values.
- `AnthropicModel`/`LLMJudgeCueDetector`/`LLMSimulator` are **not exported** from `models/__init__.py` or `_api.py` (submodule-only, low discoverability). `_cot_to_steps` splits free-text on newlines and does **not** produce canonical `L op R = V` steps, so CSC's structural corruption and `ExactArithmeticSimulator` are synthetic-only on the real path. `_api_call` assumes a specific response shape, has `max_tokens=8000` (can truncate thinking), and has no streaming/retry/backoff/rate-limit/cost accounting. `_JsonCache` rewrites the whole file per `put`.

**Measurement, aggregation & reporting**
- `expected_calibration_error` is implemented and tested at extremes only, but **called nowhere** in card/validation/report — calibration is measured nowhere in the pipeline.
- CIs are percentile bootstrap only (no BCa/DeLong/analytic variance). **No significance testing** anywhere: the headline "combined > best single" and the specificity off-diagonal and correlation matrix are point estimates with no CI on the gap. Bootstrap loops are Python-level (not vectorized).
- Composite is a flat unweighted mean and **silently drops** NaN sub-scores (no flag); there is no CI on the composite. `per_domain` is point faithfulness only (no CI, no n, no per-domain AUROC). `correlation_matrix` off-diagonal magnitude is a population-composition artifact (a sanity check, not a probe property).
- `validation.run_validation` is hardwired to `model_population` + `mixed_problems` (no param to run over a real model / external dataset); `AXIS_MODEL` assumes exactly one single-axis model per probe; `trace_examples` is a fixed hand-picked list (index-errors at very small n).
- `viz/svg.py` has only three chart types; no accessibility (`<title>`/`<desc>`/aria), no color-scale legend, label overflow on many/long labels, no calibration/distribution charts. `report/html.py` embeds `trace_examples` via `json.dumps` straight into a `<script>` (XSS-via-`</script>` latent risk if ever fed untrusted text); single fixed layout; `build_figures` KeyErrors on a malformed report dict.
- `cli.py`: no `card`/`compare` subcommand — real-model `score` dead-ends at stdout + optional JSON; the HTML renderer is hardwired to the synthetic schema. No `-n`/`--trials`/`--seed` clamping; `score` catches only `RuntimeError`; `validate` overwrites committed artifacts by default.
- `__init__.py`'s bare `except Exception` silently swallows *all* `_api` import errors (can mask a real regression as a metrics-only degrade). Version `0.1.0` is duplicated against packaging metadata.
- Real-model-path weaknesses are catalogued in `docs/DESIGN.md` §7 (CSC distribution-shift confound; SIM free-text leakage; LLM-judge dependency; white-box out of scope) but are **not yet mitigated in code or exercised by tests**.

**Tests / CI / packaging / docs**
- Test count drift: the suite passes 57 tests; README badge/body and `CLAUDE.md` say 55/~55.
- Reproduction is solid (empirically verified): `faithfulnessbench validate` (default `-n 20` per-domain × 2 domains = 40 problems, seed 0, 5 trials) and `python experiments/validate_synthetic.py` both reproduce the committed `results.json` **numerically identically** — the sole differing field is the `reproduce_cmd` provenance label each entrypoint stamps. There is no reproduce bug; the e2e test uses `n=6` only for speed. The real gap is *enforcement*, not reproduction: CI's `validate` smoke step writes to `/tmp` and checks only exit code 0, so a silent numerical regression would pass — there is no `--check`/diff gate yet.
- CI tops out at Python 3.13 (no 3.14, despite that being the design motivation; Linux only). No lint/format/type-check (no ruff/black/mypy) and no coverage. The real anthropic SDK call path (behind the transport seam), `effort` levels, and ECE multi-bin logic are untested. Orthogonality tests accept `[0.4,0.6]` though DESIGN.md claims "exactly 0.50"; the card off-diagonal `<0.6` bound tests a population artifact, not an invariant; all thresholds are fixed magic numbers tuned to the clean synthetic signal.

---

## How to extend (cookbook)

**Add a probe.** Implement a `Probe` subclass in `probes/` (set class attr `name`, implement `run(model, problems, *, n_trials=None) -> ProbeResult`). Register it in **both** `PROBE_CLASSES` and `default_probes()` in `probes/__init__.py`. Then wire the matching ground truth: add a `p_*` dial to `FaithfulnessProfile`, add the probe→dial entry to `_PROBE_TO_DIAL`, implement the unfaithful behavior in **exactly one** `Model` method branch (gated by `self._draw(dial, ...)`), add a named pure-type model in `model_population`, and append the probe to `validation.PROBE_ORDER` + `AXIS_MODEL`. Add a contract test in `tests/test_synthetic.py` and the parametrized entry in `tests/test_probes.py` (`TARGET_MODEL` dict).

**Add a synthetic failure mode / dial.** Add a `p_*` dial in `FaithfulnessProfile`, route any new stochastic behavior through `_draw(dial, problem.id, ...)` to stay deterministic, and (if it should be a labeled axis) add it to `_PROBE_TO_DIAL` + a pure-type model in `model_population`.

**Add a problem domain.** Add a new domain string and a generator in `problems.py`; extend `Problem.value_to_answer` for its surface form; extend `synthetic.py`'s `_decoy_value`/`_target_value` domain switch; extend `anthropic_model._parse_answer`/`_answer_kind` for its answer format; add an end-to-end test like `test_mcq_domain_end_to_end`.

**Swap a probe collaborator.** Implement `CueDetector` (`mentions(cot,cue)->bool`) or `CoTSimulator` (`predict(problem,cot_steps)->str`) and inject it into `SHIProbe(detector=...)` / `SIMProbe(simulator=...)`, or a custom corruptor into `CSCProbe(corruptor=...)` (anything with `.corruptions(steps)->list[(idx,steps)]`). A real-model simulator must keep the structural leakage control (read the CoT, never re-solve the question).

**Add a metric.** Add a pure `(scores/labels) -> float|tuple` function in `metrics.py` and a hand-computed known-value + edge-case test in `tests/test_metrics.py`; it is then callable from `card.py`/`validation.py`.

**Add a chart / report figure.** Add a pure `()->'<svg>...'` function in `viz/svg.py`, wire it into `report/html.build_figures` (returns the named-SVG dict), and reference it in `render_report`; it appears in the HTML and as a standalone SVG via `write_figures`. Update the expected key set in `tests/test_validation.py`.

**Add a CLI subcommand.** Add `sub.add_parser(...).set_defaults(func=_cmd_x)` in `cli.py`; defer heavy imports inside the `_cmd_*` body to keep startup fast and the no-anthropic path working.

**Add a backend.** Subclass `Model` (deterministic given `(problem, trial)`) implementing `reason`/`continue_from_cot`/`answer_from_prefix`, or mirror `[anthropic]` with a new optional-dependency extra in `pyproject.toml`. For Anthropic specifically, the `transport` Callable is the single network call site and the cheapest test/extension seam.
