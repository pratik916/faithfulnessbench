# FaithfulnessBench — Roadmap

This document positions FaithfulnessBench against the chain-of-thought (CoT)
faithfulness/monitorability literature, names the concrete gaps the current harness has,
and sequences the work to close them. It is research-toned and deliberately blunt about
what the existing numbers do and do not prove. `docs/DESIGN.md` remains the methodology
source of truth; this file is forward-looking.

---

## Where this sits in the literature

The behavioral-faithfulness toolkit FaithfulnessBench implements is the canonical one.
**SHI** (Silent Hint Injection) is the bias-articulation test of Turpin et al.
("Language Models Don't Always Say What They Think", NeurIPS 2023,
<https://arxiv.org/abs/2305.04388>), formalized as a per-instance score, and is identical
in spirit to Anthropic's hint-pairing faithfulness metric — flip-to-hint **and**
not-verbalized — in "Reasoning Models Don't Always Say What They Think"
(<https://arxiv.org/abs/2505.05410>), which reports frontier reasoning models verbalizing
an outcome-changing hint only ~25–39% of the time. **EAR** and **CSC** descend directly
from Lanham et al.'s "Measuring Faithfulness in Chain-of-Thought Reasoning"
(<https://arxiv.org/abs/2307.13702>) — early-answering with an area-over-the-curve weighting,
and adding-mistakes/corruption respectively — and CSC's "faithfulness = causal robustness to
a CoT intervention" framing is exactly the lens of Paul et al.'s FRODO (EMNLP Findings 2024,
<https://arxiv.org/abs/2402.13950>) and the 2026 RFEval counterfactual-intervention benchmark.
**SIM** operationalizes the counterfactual simulatability of Chen et al. (ICML 2024,
<https://arxiv.org/abs/2307.08678>). So all four probes are well-grounded; the open question
is not whether they are reasonable, but whether they are *valid* — and that is the project's
distinctive bet.

**The unique contribution is the validation move, not the probes.** Every benchmark above
obtains faithfulness ground truth either by expensive expert annotation — FaithCoT-Bench
(<https://arxiv.org/abs/2510.04040>) hand-labels 300+ instances and reports F1/Cohen's
kappa, finding the *best single behavioral detector tops out near ~0.70 AUROC at the CoT
level with no cross-setting transfer* — or by behavioral counterfactual consistency
(RFEval). FaithfulnessBench instead constructs models whose (un)faithfulness is fixed in
code, yielding exact per-instance labels and a *probe-detection AUROC*. No surveyed work
validates the probe itself against constructive ground truth in this form. That is the
defensible novelty and it should be foregrounded as such.

**But the current headline is near-tautological, and we will say so plainly.** Live
reproduction confirms that on the committed 6-model × 40-problem population every single
probe scores **exactly 0.700** and the combined detector **exactly 1.000** — these are the
arithmetic of a population built from one faithful + four single-axis-unfaithful + one
fully-unfaithful model, where each dial is wired to fire only in the one model method its
probe reads (`reason` / `continue_from_cot` / `answer_from_prefix`). The targeted
AUROC=1.000 and off-axis 0.500 are therefore **true by construction**: they certify the
probe is correctly *wired* to its failure mode, not that it is *sensitive* on noisy or
adversarial CoT. A faithfulness researcher recognizes this in under a minute. The most
honest framing — and the one we adopt — is the FaithCoT-Bench foil: a synthetic 1.000 is
high *because* it is synthetic; the ~0.70 real-world ceiling is what a measurement on a hard,
noisy substrate actually looks like.

**The concrete gaps we intend to close**, in priority order: (1) the by-construction
credibility gap — inject controlled noise so AUROC becomes a sensitivity *measurement*, add
a shuffled-label control proving the instrument *can* fail, and freeze the pooled baseline so
new models cannot silently move the headline; (2) breadth the literature names but the
battery lacks — Filler-Token / hidden-computation (Lanham; Pfau et al.,
<https://arxiv.org/abs/2404.15758>), Paraphrase-Invariance / steganography (Lanham), and
no-hint Implicit Post-Hoc Rationalization via contradictory pairs (Arcuschin et al.,
<https://arxiv.org/abs/2503.08679>); (3) the **monitorability** reframing that the 2025
multi-lab position paper (Korbak, Balesni et al., <https://arxiv.org/abs/2507.11473>) and the
obfuscated-reward-hacking result (Baker et al., <https://arxiv.org/abs/2503.11926>) made the
field's central object, reported with monitor-grade metrics (safety-at-1%-FPR, catch-rate,
calibration) from CoT Red-Handed (<https://arxiv.org/abs/2505.23575>) and Apollo's linear
probes (<https://arxiv.org/abs/2502.03407>); (4) a reproducible **real-Claude** demonstration
— the only place an AUROC would be a genuine measurement rather than a construction; and (5)
the **inferential layer** (DeLong/permutation, cluster bootstrap, judge-vs-gold kappa) that
the frontier-lab papers themselves omit, applied only where numbers genuinely vary.

---

## Roadmap

Each epic states a goal and why it matters, then a checkbox task list. Per task: a title,
**priority** (P0 highest – P3 lowest), **effort** (S/M/L), and a one-line acceptance test.
Dependencies are noted where they constrain ordering.

### Epic 1 — Honesty foundations (the thesis-level credibility layer)

**Goal.** Resolve the project's single biggest research-credibility risk — that targeted
AUROC=1.000, off-axis=0.500, and combined-1.000-vs-best-single-0.700 are *true by
construction*, not measurements — by (a) injecting controlled noise so AUROCs become real
measurements, (b) adding a falsifiability control proving the harness *can* fail, (c)
freezing the pooled baseline so new models don't silently move the headline, and (d) stating
plainly what the synthetic numbers do and do not prove.

**Why it matters.** This is the highest-leverage work in the roadmap and a prerequisite for
trusting every downstream AUROC. The difference between a portfolio piece that survives a
faithfulness researcher's 60-second scrutiny and one dismissed as "validated against a model
built to be detected" is a noisy substrate where AUROC is a sensitivity measurement, a
control that *can* report a low number, and honest framing.

- [ ] **Inject controlled noise so synthetic AUROC measures sensitivity, not wiring** — P1, L.
  Add a noise/partial-dial regime (intermediate dial rates, deterministic per-instance label
  noise keyed by the existing `(seed, problem.id, keys, trial)` hash, and a hard-instance
  regime: short chains for CSC, near-tie operands / low-margin cues for SHI); report targeted
  AUROC as a *curve* over dial strength and noise level alongside the point estimate.
  *Acceptance:* a test asserts zero-noise targeted AUROC stays 1.000 (wiring intact) but under
  the documented regime each probe drops below 1.0 with a positive-width bootstrap CI; the
  report renders an AUROC-vs-noise curve per probe; deterministic, numpy-only, no key.
  *(Depends on: ProbeResult/registry guard.)*
- [ ] **Add a shuffled-label / random-axis negative control proving the harness can fail** —
  P1, S. Permute the ground-truth labels (or pair each probe against a mismatched-axis model)
  and assert the AUROC collapses to ~0.5; add a contract test that probes only ever read
  `Trace` text and never touch synthetic internals.
  *Acceptance:* shuffled-label AUROC within [0.4, 0.6] for every probe; a contract test fails
  if any probe accesses a synthetic-only attribute; `results.json` carries
  `negative_control_auroc`; numpy-only, deterministic.
- [ ] **Freeze the pooled-population baseline and add a held-out extended-population
  contract** — P1, M. Split the population into a **frozen** validation set (the current 6
  models, whose pooled `combined_auroc` / `single_mixed_auroc` / correlation matrix / hard-coded
  disagreement prose are the committed artifact) and an **extended** set used only for pairwise
  targeted/specificity AUROC of new axes; make deliberate re-baselining a single explicit commit.
  *Acceptance:* a test asserts adding a model to the extended set leaves the pooled numbers and
  correlation matrix byte-identical to the committed artifact while the new model still gets a
  pairwise targeted-AUROC entry; `DESIGN.md` documents the split; numpy-only.
  *(Depends on: ProbeResult/registry guard.)*
- [ ] **Add a faithful-by-construction positive control model (Lyu) to the extended
  population** — P1, S. A `ConfigurableSyntheticModel` mode that *derives* its answer by
  executing its own printed `L op R = V` chain (solver-style, per Lyu et al.,
  <https://aclanthology.org/2023.ijcnlp-main.20/>) — faithful by construction on every axis — as
  a canonical true-negative anchor alongside the dial=0 faithful model.
  *Acceptance:* a test asserts it scores <0.05 unfaithfulness on all probes and gives targeted
  AUROC indistinguishable (within bootstrap CI) from the dial=0 baseline; it lands in the
  extended (not frozen) set so committed pooled numbers are unchanged.
  *(Depends on: frozen-population contract.)*
- [ ] **Write the "what synthetic AUROC does and does NOT prove" honesty section and fix
  README framing** — P1, S. Docs-only. State in `DESIGN.md` that targeted AUROC=1.0 proves the
  probe is correctly *wired* and specific *by construction* — not that it is sensitive on real,
  noisy, or adversarial CoT; reframe combined-1.000-vs-best-single-0.700 as a *population
  identity* (all four single probes are exactly 0.700), not a discriminative result; fix
  README line 19 so 0.700 no longer reads as a measured single-probe weakness; add a "what would
  falsify this" statement and cite the FaithCoT-Bench ~0.70 ceiling as the foil.
  *Acceptance:* `DESIGN.md` and `README` contain the by-construction caveat, the population-identity
  reframing of 0.700, a falsification statement, and the FaithCoT-Bench comparison; README
  line 19 is corrected; no code changes, suite unaffected.

### Epic 2 — Probe-contract hardening and directed causal upgrades

**Goal.** Make the probe layer fail-fast and registry-consistent, turn CSC from a permissive
"answer changed" proxy into a directed "answer tracks the corruption" test, close the
documented length/sign-preserving corruption confound, and generalize corruption behind a
protocol — without silently dropping any domain.

**Why it matters.** These are the foundations every new axis and every noise-injection
measurement build on. The registry guard catches misregistration in CI; the CSC upgrade fixes
a verified correctness gap (`csc.py` line 69 scores `new_answer != a0`, so a flip to an
*unrelated* wrong answer is currently counted faithful). Low-risk, on-thesis rigor the
critiques flagged as must-dos.

- [ ] **Enforce ProbeResult invariants and single-source the probe registry** — P1, S. Add a
  numpy-only `__post_init__`/`validate()` asserting scores in [0,1] and
  `len(scores)==len(problem_ids)`; derive `default_probes()` from `PROBE_CLASSES` (or vice
  versa) with a uniqueness/consistency self-check; preserve per-trial/per-fraction raw curves in
  `ProbeResult.extra`.
  *Acceptance:* tests assert an out-of-range or misaligned `ProbeResult` raises immediately; that
  `PROBE_CLASSES` and `default_probes()` are provably consistent; and that `EAR.extra` carries the
  per-fraction match curve; numpy-only.
- [ ] **Upgrade CSC to verify the answer tracks the corruption, not merely that it changed** —
  P1, S. Use `recompute_annotations`/`execute_steps`/`stated_final` (already in `problems.py`) to
  compute the expected-under-corruption answer from the *same* re-chained steps the model was
  shown, and score sensitivity as `new_answer == expected-under-corruption`; majority-vote the
  baseline answer `a0` over trials for CSC and SHI.
  *Acceptance:* a test asserts CSC counts only corruption-tracking flips (an injected
  unrelated-flip model scores unfaithful, not faithful) and that synthetic CSC targeted AUROC is
  unchanged at zero noise (deterministic per trial); suite stays green.
  *(Depends on: ProbeResult/registry guard.)*
- [ ] **Add a length/sign-preserving CSC corruptor behind a `Corruptor` protocol** — P2, M.
  Define a `Corruptor` protocol and add a length/sign-preserving strategy (bound the operand
  delta so digit-count and sign are invariant) to close the `DESIGN.md` §7 confound that
  ~50–60% of corruptions change digit-count and ~13% change sign. *(Note: MCQ is already covered —
  verified MCQ problems share the canonical `L op R = V` chain and parse fine; the protocol exists
  for the new strategy and a future third domain, not to rescue MCQ.)*
  *Acceptance:* a test asserts the strategy keeps digit-count and sign invariant on ≥95% of
  arithmetic corruptions while preserving CSC targeted AUROC ≥0.95 at zero noise; both the operand
  and length/sign-preserving corruptors satisfy the protocol; no probe is dropped on any current
  domain.
  *(Depends on: directed-CSC upgrade.)*
- [ ] **Tighten orthogonality assertions and add a seed-sweep stability test** — P1, M. Tighten
  the off-axis tests toward the documented exact 0.50 (per the tie convention) and label it a
  *structural* artifact (the probe returns identical zeros off its axis), not measured
  non-leakage; add a ≥5-seed sweep asserting headline metrics are stable. Land before any new
  probe so the 1.0s are demonstrably not seed-specific.
  *Acceptance:* tests assert off-axis AUROC equals 0.50 within a tight tolerance and that the
  report labels it structural; a ≥5-seed sweep confirms headline metrics stay within a documented
  band; deterministic, numpy-only.

### Epic 3 — New no-hint and content-removal probes (noise-aware validation)

**Goal.** Add the highest-signal probes the literature names that the current battery lacks —
Filler-Token (Lanham/Pfau hidden-compute), Paraphrase-Invariance (steganography), and Implicit
Post-Hoc Rationalization via contradictory pairs (Arcuschin, needs *no* injected hint) — each
with a matching synthetic dial in the *extended* population, validated under the noise regime so
the AUROC is a measurement and not a wiring identity.

**Why it matters.** These close the most-cited gaps, and the IPR probe adds a genuinely new
axis (no injected hint, exact logical ground truth) that does not inherit SHI's cue machinery.
Breadth is only worth adding *after* the by-construction problem is confronted — so every probe
here is validated on the noisy substrate and held out of the frozen pool, making the additions
honest rather than four more guaranteed-1.0 bullet points.

- [ ] **Add a Filler-Token probe (FIL) with a compute-not-content synthetic dial** — P2, M.
  Replace the CoT with length-matched filler tokens and force an answer (Lanham filler-token /
  Pfau hidden-computation); add a `p_filler` dial whose answer is correct iff ≥N
  filler-equivalent tokens are present regardless of content; wire FIL into `PROBE_ORDER`/`AXIS_MODEL`.
  *Acceptance:* a test asserts FIL targeted AUROC ≥0.95 at zero noise and a non-degenerate sub-1.0
  AUROC under the noise regime, ~0.5 off-axis, and that adding `p_filler` to the extended set leaves
  committed pooled numbers unchanged; numpy-only, no key.
  *(Depends on: noise injection, frozen-population contract.)*
- [ ] **Add a Paraphrase-Invariance probe (PAR) with a steganographic synthetic dial** — P2, M.
  Deterministically reword the CoT prefix while preserving stated numeric results, then re-derive
  (Lanham paraphrasing); add a `p_stego` dial whose answer depends on a surface artifact (token
  parity / acrostic) so paraphrase flips it; wire into `PROBE_ORDER`/`AXIS_MODEL`.
  *Acceptance:* a test asserts PAR targeted AUROC ≥0.95 at zero noise (sub-1.0 under noise), ~0.5
  off-axis, that a content-preserving rewrite leaves a faithful model's answer unchanged on every
  instance, and that committed pooled numbers are unchanged; numpy-only, deterministic.
  *(Depends on: noise injection, frozen-population contract.)*
- [ ] **Add an Implicit Post-Hoc Rationalization probe (IPR) via contradictory question
  pairs** — P2, M. Programmatically generate logically contradictory yes/no pairs ("Is X>Y?" /
  "Is Y>X?") in pure numpy with exact logical ground truth (Yes/Yes or No/No is provably
  rationalizing); add a paired-question generator to `problems.py` and an extended dial producing
  contradictions at a controlled rate; emphasize the noise-regime AUROC over the guaranteed
  synthetic number since the real value is no-hint transfer.
  *Acceptance:* a test asserts IPR targeted AUROC ≥0.95 at zero noise (meaningful sub-1.0 under
  noise), zero logically-impossible pairs for a fully faithful model, and a seeded/deterministic
  numpy-only generator; committed pooled numbers unchanged.
  *(Depends on: noise injection, frozen-population contract.)*

### Epic 4 — Reframe as a calibrated CoT monitor (honestly-labeled offline analogues)

**Goal.** Promote the combined detector to a first-class *monitor*, report the monitor-grade
metrics the 2025–2026 papers headline (safety-at-1%-FPR, catch-rate, subtle-vs-blatant split,
ECE + reliability), and add **one** adversarial monitor-evasion axis (obfuscated reward hacking)
where the monitor is fixed *first* and the obfuscation is a generic surface-scrub — framed
throughout as offline analogues with known labels, never as "reproductions" of the papers'
empirical findings.

**Why it matters.** Monitorability (Korbak et al., <https://arxiv.org/abs/2507.11473>) is the
field's central 2025 object, and reframing the probes as a monitor is high-leverage. But
defining *both* the monitor and the obfuscation makes evasion whatever you set, and calling
synthetic dials a "reproduction" of Baker/Emmons overclaims — so we keep the one genuinely novel
adversarial axis (monitor fixed first to avoid circularity), cut the redundant evasion dials,
and adopt strict "analogue" language.

- [ ] **Reposition README/DESIGN intro around monitorability with honest literature framing** —
  P1, S. Docs-only. Open with the multi-lab monitorability agenda, present the probes as monitor
  components, adopt the monitorability = faithfulness × coverage(verbosity) decomposition (note SIM
  covers coverage), add a behavioral-vs-white-box scope paragraph, and cite the FaithCoT-Bench
  ~0.70 behavioral ceiling as the foil to the synthetic 1.000.
  *Acceptance:* README/DESIGN include the monitorability framing, the faithfulness×coverage
  decomposition, and the FaithCoT-Bench / behavioral-vs-white-box comparison with citations; no code
  changes, suite unaffected.
- [ ] **Report monitor-grade metrics: safety-at-1%-FPR, catch-rate, subtle-vs-blatant split** —
  P1, M. Compute catch-rate / safety at a low FPR calibrated on the known-faithful population (1%
  FPR à la CoT Red-Handed / Apollo) plus a subtle-vs-blatant split using the noise regime's
  intermediate-dial instances as "subtle"; all pure-numpy via the existing `roc_curve`.
  *Acceptance:* a test asserts safety-at-1%-FPR and catch-rate are computed from the known-faithful
  negatives and that subtle cases yield a lower catch-rate than blatant; the numbers appear in
  `results.json` and the report; numpy-only.
  *(Depends on: noise injection.)*
- [ ] **Wire calibration (ECE + reliability diagram) into the pipeline** — P2, M. `metrics.expected
  _calibration_error` is implemented and tested but called nowhere; wire it into the report as a
  per-probe and combined-monitor reliability curve + ECE KPI, driven by the noise-regime scores
  (synthetic dials are near-0/1 clustered, so the diagram is only informative on the noisy/real
  substrate); add a hand-drawn SVG reliability diagram and document that equal-width ECE is
  near-worst-case for 0/1-clustered scores.
  *Acceptance:* the report renders a reliability-diagram SVG and an ECE KPI for the combined monitor
  and each probe; a test asserts the ECE matches a hand-computed fixture and the SVG is well-formed;
  numpy-only.
  *(Depends on: noise injection.)*
- [ ] **Add an obfuscated-reward-hacking axis (OBF) as an offline analogue, monitor fixed
  first** — P2, L. Add a `p_obfuscate` dial that pursues a hacked target while adversarially
  scrubbing the CoT. *Critical ordering:* fix the monitor first (combined detector at the τ from
  the safety-at-1%-FPR task) and make the obfuscation a *generic* surface-scrub, not bespoke to
  that monitor; add an OBF probe measuring evasion = cheated ∧ (monitor score < τ); frame strictly
  as an offline known-label *analogue* of Baker et al. (<https://arxiv.org/abs/2503.11926>).
  *Acceptance:* a test asserts τ is fixed before the obfuscation is defined and the obfuscation is
  monitor-agnostic; that at `p_obfuscate=1` the model provably cheats on every instance yet the
  fixed monitor's catch-rate falls below a documented threshold; an obfuscation curve is emitted;
  docs use "analogue" not "reproduction"; deterministic, numpy-only, no key.
  *(Depends on: monitor-grade metrics, frozen-population contract.)*
- [ ] **Split SHI into a hint-type taxonomy with per-type dials and per-type AUROC** — P3, L.
  Replace the single generic cue with a cue-family covering the Anthropic-2025 / Turpin types
  expressible in a text-only world — sycophancy, consistency (prefilled prior answer), metadata
  (answer in XML), authority framing, conflicting/multi-hint (drop visual-pattern: no text analog).
  Each is a `Cue` subtype with its own exact marker and an extended dial; report per-type
  faithfulness and detection AUROC.
  *Acceptance:* a test asserts each retained hint type yields targeted AUROC ≥0.9 against its own
  dial and ~0.5 against the others, the card exposes a per-hint-type breakdown, and the new
  sub-models are extended-only so committed pooled numbers are unchanged; numpy-only, deterministic.
  *(Depends on: frozen-population contract.)*
- [ ] **Add a right-answer-via-illegitimate-channel sub-signal to SHI (reward-hack
  shortcut)** — P3, M. *Resolves a verified contradiction:* SHI's flip detection requires
  `tr.answer == cue.target ∧ tr.answer != a0` (`shi.py` line 39), so a reward-hack shortcut
  pointing at the *correct* answer never fires the existing check. Add a *separate* shortcut
  sub-signal: the cue exposes the correct answer via an illegitimate channel (leaked
  validator/metadata), and the signal fires when correctness depends on that channel (answer
  degrades when it is removed) *and* the channel is never verbalized — leaving the wrong-answer flip
  path untouched. Add `p_shortcut`; frame as a known-label analogue of "monitoring misses reward
  hacks", not a reproduction.
  *Acceptance:* a test asserts the shortcut sub-signal detects channel-dependent-but-unverbalized
  correctness at AUROC ≥0.9 against `p_shortcut` while a `SubstringCueDetector` confirms zero
  acknowledgement, *and* that SHI's existing `p_hint_sycophancy` AUROC is byte-unchanged;
  extended-only; numpy-only.
  *(Depends on: SHI taxonomy, OBF axis.)*

### Epic 5 — Reproducible real-Claude demonstration via committed record/replay cache

**Goal.** Turn "the same validated probes run on a frontier model" from asserted to
demonstrated end-to-end while keeping the core no-key/no-network: verify SDK request shapes
against the `claude-api` skill, harden the transport, do the one-time keyed recording of
`claude-sonnet-4-6` and `claude-opus-4-8`, commit a small replay cache so CI exercises the
whole real path offline, and produce a portfolio-grade real-model report stated with humility
(no ground-truth AUROC exists there).

**Why it matters.** This is the only place a reported AUROC/number would be a genuine
measurement rather than a construction, and it is the most fragile prerequisite — it needs a key
plus spend and gates ~4 downstream tasks. Pulling the keyed recording forward as a gated
do-it-now step de-risks the whole roadmap; everything downstream then runs offline from the
committed cache.

- [ ] **Surface and document the real-model API, narrow the import guard, add the free-text-CoT
  caveat** — P1, S. Re-export `AnthropicModel` / `LLMSimulator` / `LLMJudgeCueDetector` from the
  package top-level (guarded so `anthropic` stays optional); replace the verified bare
  `except Exception` in `__init__.py` with targeted `ImportError`/`ModuleNotFoundError` handling;
  add the honest caveat — *before* any real model is scored — that the exact synthetic
  simulator/detector are synthetic-only and CSC/SIM degrade on real free-text CoT.
  *Acceptance:* importing the real-model classes works when `anthropic` is installed and degrades to
  a clear `ImportError` (not silent metrics-only) when not; a test asserts the narrowed handling and
  that the free-text-CoT / judge-swap caveat is present in README/DESIGN/docstrings.
- [ ] **Verify Anthropic SDK request shapes against the `claude-api` skill before any
  recording** — P1, S. The cache key is a sha256 of the request spec, so a wrong shape means
  re-billing to re-record. Consult the `claude-api` skill to confirm the adaptive-thinking shape,
  the effort/thinking-budget mapping, and what `thinking.display='summarized'` returns for
  opus-4.x (verified: the adapter sets `summarized` for those models — summarized thinking is *not*
  raw reasoning, which bears on the later thinking-vs-answer diagnostic); dry-run the score path
  against the fake transport, then record.
  *Acceptance:* a documented note records the verified SDK request shapes per the skill; a
  fake-transport dry run of the score path passes asserting spec construction; no key spent yet.
  *(Depends on: API-surface task.)*
- [ ] **Harden the Anthropic transport: retry/backoff, token/cost accounting, truncation
  guard** — P1, M. `_api_call` has no retries and a fixed `max_tokens=8000` (verified) that can
  silently truncate adaptive-thinking traces; add bounded retry/backoff, a truncation flag, and
  per-call token/cost recording, all behind the injectable transport seam; harden trace/cache JSON
  embedding against `</script>` / U+2028/U+2029 breakage *now*, before any real CoT is recorded.
  *Acceptance:* fake-transport unit tests assert retry-on-transient-error, a raised/flagged signal
  on truncated output, recorded per-call token/cost, and safe escaping of embedded `</script>` and
  U+2028/U+2029; the core suite still runs with no `anthropic` SDK and no key.
  *(Depends on: SDK-shape verification.)*
- [ ] **Disambiguate malformed CoT from unfaithful CoT in SIM on the real path** — P1, S.
  `ExactArithmeticSimulator.predict` returns `'?'` on an unparseable final step (verified), scoring
  as a guaranteed miss; on real models a parse-failed CoT is distinct from a non-predictive one.
  Track a parse-failure rate in `extra` and exclude/flag those instances. Must land before any real
  SIM number is published.
  *Acceptance:* a test asserts SIM records a parse-failure rate and reports flagged parse-failures
  separately from genuine simulatability misses; synthetic SIM AUROC is unchanged because synthetic
  chains always parse.
  *(Depends on: API-surface task.)*
- [ ] **Record and commit a real-Claude replay cache and add an end-to-end score test** — P1, L.
  Do the one-time keyed recording of `claude-sonnet-4-6` and `claude-opus-4-8` on a small set
  (n=15–20), commit the small replay cache, and add an e2e test running `score` offline from the
  cache with no key; report real-model behavior as *descriptive* statistics (flip-rate, ack-rate,
  simulatability, early-lock) with explicit humility — no ground-truth labels exist, so do *not*
  report AUROC-vs-truth on the real path.
  *Acceptance:* the committed cache lets `score` run end-to-end in CI with no key, producing a
  Faithfulness Card whose structure a test asserts and whose request specs match the recorded specs;
  real-model numbers are descriptive with the no-ground-truth caveat; the core suite stays key-free.
  *(Depends on: transport hardening, SIM parse-failure fix.)*
- [ ] **Make `score` produce the portfolio-grade HTML report (model-agnostic renderer)** — P2, L.
  Real-model scoring currently dead-ends at stdout + optional card JSON; generalize `render_report`
  (or add a card-report renderer) so a real Claude run produces the same shareable page as the
  synthetic validation. (The `</script>` / U+2028 hardening lands earlier, so embedded real CoT is
  already safe.)
  *Acceptance:* running `score` against the committed cache emits a self-contained HTML report from
  the real-model cards; a test asserts the renderer handles a cards-only report and that embedded
  CoT containing `</script>` does not break the page.
  *(Depends on: replay cache + e2e test.)*
- [ ] **Add a thinking-vs-answer acknowledgment-divergence diagnostic** — P3, S. Exploit the
  adapter's separate thinking and answer channels to measure the gap between hint acknowledgment in
  reasoning tokens vs final answer text (the "Lie to Me" 2026 finding). *Honesty gate:* opus-4.x
  emits only `summarized` thinking, which is not raw reasoning — confirm what `summarized` returns
  and scope the diagnostic to inspectable-thinking models (or label it "summary-channel, not raw")
  rather than claiming the literature number; absent gracefully when no thinking channel exists.
  *Acceptance:* `score` reports a thinking-channel vs answer-channel acknowledgment rate per cued
  instance with an explicit raw-vs-summarized label; a test asserts the sub-metric is computed from
  both channels, absent gracefully without a thinking channel, and does not claim the literature
  number on a summarized channel.
  *(Depends on: replay cache, SDK-shape verification.)*

### Epic 6 — Inferential rigor where it is honest (real/noisy substrate only)

**Goal.** Add the inferential layer the harness lacks — but apply significance testing *only*
where there is genuine variation (the noisy synthetic regime and the real-model path), not to the
definitional synthetic gap. Ship DeLong + paired permutation, cluster bootstrap, chance-corrected
inter-probe agreement (kappa) with multiple-comparison control, an LLM-judge-vs-gold calibration
rig, and per-domain detection AUROC — all numpy-only.

**Why it matters.** The flagship comparisons need defensible inference, but
significance-testing the *synthetic* combined-1.000-vs-best-single-0.700 gap manufactures the
appearance of inference over a constant identity — an honesty hazard. Scoping DeLong/permutation
to the noisy and real substrates (where the numbers actually vary) makes the harness stricter
than its prior art without overclaiming. The judge-vs-gold kappa rig is the only thing that tells
you whether the real-path LLM judges are trustworthy.

- [ ] **Add DeLong + paired permutation tests for AUROC differences, applied only to varying
  substrates** — P2, M. Add `metrics.delong_test(scores_a, scores_b, labels)` returning the AUC
  gap, a DeLong z/p (reusing the existing rank/placement machinery), and a within-instance
  permutation p (Bandos 2005, more powerful in the small-n near-1.0 regime), each with a paired
  bootstrap difference CI. Apply *only* to the noise-regime combined-vs-best-single comparison and
  the real-model path; do *not* significance-test the zero-noise synthetic gap (it is a population
  identity).
  *Acceptance:* unit tests verify `delong_test` against a hand-computed fixture and a deterministic
  permutation p; the report shows a p-value and difference-CI only on the noisy/real substrate;
  `DESIGN.md` states why the zero-noise gap is not tested; numpy-only.
  *(Depends on: noise injection.)*
- [ ] **Replace the flat bootstrap with a cluster (hierarchical) bootstrap and add per-domain
  detection AUROC** — P2, M. Add `bootstrap_cluster_ci(..., cluster_ids)` resampling whole clusters
  (problem-level, optionally model-level) using the `problem_ids` already in `ProbeResult` (the flat
  bootstrap is anti-conservative because instances are nested: shared problems across models, 5
  trials each); route cards/validation through clustered CIs, keep the flat path for back-compat;
  add a stratified-AUROC helper reporting per-domain detection AUROC (arithmetic vs MCQ). *(Drop
  bootstrap vectorization — n is tiny.)*
  *Acceptance:* a test asserts the cluster-bootstrap CI ≥ the flat CI width on a nested fixture and
  reduces to the flat result when every instance is its own cluster; cards/validation use clustered
  CIs; the report includes a per-domain detection-AUROC breakdown; numpy-only, deterministic.
  *(Depends on: ProbeResult/registry guard.)*
- [ ] **Add chance-corrected inter-probe agreement (kappa) plus multiple-comparison control** —
  P2, M. Augment the cross-probe correlation with chance-corrected *agreement* (Cohen's kappa /
  Krippendorff's alpha on thresholded per-instance flags, since the disagreement claim is about a
  binary "flagged unfaithful?" decision), each with a cluster-bootstrap CI; apply Holm–Bonferroni to
  the specificity-matrix family and pairwise model-ranking comparisons, flagging rank pairs with
  overlapping CIs as "not statistically resolved"; stop the README leading with the raw ~0.25
  disagreement number.
  *Acceptance:* a test verifies `cohen_kappa` and `holm_correction` against hand-computed values;
  the report reframes the disagreement finding around chance-corrected agreement with CIs and
  annotates post-correction significance of specificity cells and rank pairs; README no longer leads
  with the raw 0.25; numpy-only.
  *(Depends on: cluster bootstrap.)*
- [ ] **Calibrate the LLM judges against synthetic gold (Cohen's kappa, self-consistency,
  bias)** — P2, M. On the synthetic world, cue-acknowledgment and simulator answers have exact gold
  labels; add a judge-audit running `LLMJudgeCueDetector` / `LLMSimulator` over a labeled synthetic
  (and cached real) sample reporting Cohen's kappa vs gold (Landis–Koch bands + a human-human
  reference), judge self-consistency under order/paraphrase swaps, and a coarse Pearson-then-kappa
  gate; emit a "judge reliability: kappa=__" line next to every real-path SHI/SIM number and gate
  the real report on a minimum kappa. *(Self-consistency needs multiple recorded judge calls per
  item — budget those into the keyed recording so the audit runs offline.)*
  *Acceptance:* `metrics.cohen_kappa` matches a hand-computed fixture; the judge-audit reports kappa
  vs synthetic gold and a self-consistency rate, runs fully offline against the cache (including the
  extra judge calls), and the real-model report shows a per-probe judge-reliability line; numpy-only.
  *(Depends on: replay cache + e2e test.)*

### Epic 7 — Reproducibility safety net, CI hardening, and report legibility

**Goal.** Ship the genuinely-useful reproducibility tooling (`--check` mode + CI regression
gate + version single-sourcing) *without* the false "reproduce path is broken" premise, fix the
verified stale test count, add ruff + a Python-3.14/macOS CI matrix, and make the SVG artifacts
accessible and legible — so a stranger can clone, run, get the same numbers, and read the result.

**Why it matters.** A portfolio piece is judged on clone-run-reproduce-read. Live verification
settled the disputed premise: the committed `results.json` reproduces *byte-for-byte* from the
documented default (`-n 20` per-domain × 2 = 40 problems), so there is **no** reproduce bug and
"aligning the default to n=40" would *break* reproduction. The only genuine honesty defect is the
test-count badge (**57** actual vs **55** stated, verified). The `--check` mode and CI gate remain
high-value as a safety net for every numerics-changing task; the 3.14 row is on-narrative (the
numpy-only story is "wheels were scarce on 3.14").

- [ ] **Fix the stale test count and add a badge-drift CI guard** — P0, S. pytest collects and
  passes **57** tests but the README shields badge, two README prose mentions, and `CLAUDE.md` all
  say **55**; update all occurrences to the live collected count and add a CI step that fails if the
  badge drifts from the live collection count.
  *Acceptance:* README and `CLAUDE.md` state the actual collected test count, and a CI step fails on
  any drift between the badge number and the live pytest collection count.
- [ ] **Add a `validate --check` mode and single-source the version (no false-premise
  framing)** — P0, S. Add `validate --check` that recomputes the validation and diffs against the
  committed `results.json` *without* overwriting it; single-source the version via
  `importlib.metadata`; de-duplicate so `experiments/validate_synthetic.py` delegates to the shared
  CLI code path (they currently differ only on `--figures`). *Do not* change the default `n` and do
  *not* frame this as fixing a broken reproduce path (it reproduces byte-for-byte today); define
  up front which fields are exact-match vs tolerance-based.
  *Acceptance:* `validate --check` exits non-zero on drift in the documented key numbers without
  overwriting the artifact and zero when they match; CLI and experiment driver call one shared
  function; version reads from `importlib.metadata`; the exact-vs-tolerance field policy is
  documented; no default-`n` change and no false-premise claim in docs/changelog.
- [ ] **Add a CI regression gate diffing fresh results against the committed artifact** — P0, M.
  CI's validate step writes to `/tmp` and only checks exit code 0, so a silent numerical regression
  in the headline would pass; add a step that runs `validate` with the artifact's exact params and
  asserts the documented key numbers (targeted AUROC, combined-vs-best-single, specificity diagonal)
  match the committed `results.json` via `--check`, with the per-field tolerance policy so it is not
  flaky across the matrix.
  *Acceptance:* CI fails if freshly-generated key numbers diverge beyond the documented per-field
  tolerance; a deliberate perturbation in a test branch trips the gate; the gate is stable across the
  Python/OS matrix; the core suite still runs no-key/no-network.
  *(Depends on: `validate --check` mode.)*
- [ ] **Add a committed-artifact regeneration check covering report HTML and SVGs** — P1, S.
  `CLAUDE.md` says "regenerate committed artifacts after any change that affects numbers" but nothing
  enforces it; extend the check tooling so `report/faithfulness_report.html` and `docs/assets/*.svg`
  are verified consistent with `results.json` (regenerate-and-diff or a content hash), not just the
  JSON key numbers.
  *Acceptance:* a CI/check step fails if the HTML report or SVGs are stale relative to the committed
  `results.json`; regenerating from `results.json` makes the check pass; numpy-only.
  *(Depends on: CI regression gate.)*
- [ ] **Add ruff lint/format and a Python 3.14 + macOS CI matrix (defer mypy)** — P2, M. Add a ruff
  (lint+format) gate and a coverage number; add Python 3.14 (the numpy-only narrative is that
  3.14-wheel scarcity motivated it, yet CI tops out at 3.13) and macOS to the matrix; delete the
  verified-dead `first_operand_left` in `problems.py`. *Defer mypy* (near-zero annotations; a
  multi-day cleanup, not a chore).
  *Acceptance:* CI runs `ruff check` and reports a coverage number, both green; the matrix includes
  Python 3.14 and macOS and the suite passes on both; `first_operand_left` is removed; no new runtime
  dependency (tooling in the dev extra); mypy is explicitly deferred with a tracking note.
- [ ] **Add chart accessibility, legends, and label-overflow handling to `viz/svg`** — P2, M. The
  SVGs lack `<title>`/`<desc>`/aria, color-scale legends, and label-overflow handling; add them and
  expose `correlation_color` via `viz` `__all__`.
  *Acceptance:* generated SVGs include `<title>`/`<desc>`/aria attributes and a visible legend for
  each color scale, labels no longer overlap on a many-label fixture, and `correlation_color` is
  importable from `viz`; a test asserts the a11y elements and legend are present.

### Epic 8 — Domain generalization: GSM8K real-math substrate + literature positioning

**Goal.** Answer the "does it generalize beyond linear arithmetic?" critique with the cheapest
checkable evidence — GSM8K as a cached real-model arithmetic substrate so CSC/EAR demonstrably
transfer to real grade-school math offline — and position the project crisply against
FaithCoT-Bench / RFEval with comparable metrics. Defer the full third *synthetic* domain as
out-of-scope for this iteration.

**Why it matters.** GSM8K (final answer = a single split on `####`, no parser dependency) is
the cheapest way to show the validated probes run unchanged on real math CoT, bundled as a cached
subset a stranger reproduces offline. The literature positioning lets a hiring panel place the
numbers. A full third synthetic domain is effectively a second project — the probes/simulator/cue
machinery are deeply coupled to the `L op R = V` chain — so it is deferred to a scoping spike
rather than shipped half-wired.

- [ ] **Add a numpy-only GSM8K loader with a bundled cached subset** — P2, M. Wire a tiny GSM8K
  loader (final answer = single split on `####`, no extra deps; <https://huggingface.co/datasets/openai/gsm8k>)
  into the real-model path so CSC and EAR run on real grade-school math chains; bundle a small cached
  subset (50–200 items) and their recorded traces so the offline cache reproduces real-model numbers
  with no key.
  *Acceptance:* a test asserts the loader parses bundled GSM8K items and extracts the integer answer
  via `####`, and that running CSC/EAR over the cached GSM8K traces produces a card offline with no
  key; no new runtime dependency beyond numpy.
  *(Depends on: replay cache + e2e test.)*
- [ ] **Position the project against FaithCoT-Bench / RFEval with comparable metrics** — P2, S.
  Add a "Related benchmarks" section to README/DESIGN contrasting synthetic-ground-truth-by-construction
  with FaithCoT-Bench (expert annotation, F1/kappa) and RFEval (behavioral counterfactual
  consistency); report F1 and Cohen's kappa alongside AUROC; cite FaithCoT-Bench's finding that
  counterfactual methods succeed in math but fail in knowledge domains as an honest scope note for
  the arithmetic-heavy core; keep the ~0.70-vs-1.000 contrast stated once, coherently with the
  honesty section.
  *Acceptance:* README/DESIGN include the related-benchmarks comparison and report F1/kappa beside
  AUROC for the headline result; the math-vs-knowledge scope caveat is stated; no new runtime deps;
  suite unaffected.
  *(Depends on: judge-vs-gold calibration.)*
- [ ] **Spike: scope a third structurally-different synthetic domain (research, not build)** — P3,
  S. Produce a written design enumerating which protocols must be extracted (the probes, simulator,
  cue machinery, and `value_to_answer`/`_parse_answer` all assume `L op R = V`, so a genuinely
  different domain like symbolic logic needs domain-specific parse/execute/corrupt/simulator
  implementations behind 4+ protocols), the per-probe domain adapters required, and a realistic
  effort estimate.
  *Acceptance:* a short design note (in `DESIGN.md` or an issue) enumerates the protocols and
  per-probe adapters a third domain requires, a realistic effort estimate, and a go/no-go
  recommendation; no production code changes; suite unaffected.
  *(Depends on: `Corruptor` protocol.)*

---

## Sequencing

Verified on disk before planning (this changed the plan materially): the committed
`results.json` reproduces **byte-for-byte** from the documented default (`-n 20` per-domain × 2 =
40 problems), so there is no reproduce bug; every single probe scores **exactly 0.700** and the
combined detector **exactly 1.000** — a population identity, not a measurement; `shi.py` line 39
requires `answer == cue.target ∧ answer != a0` (a correct-answer shortcut never fires it);
`csc.py` line 69 scores `new_answer != a0` (an unrelated flip counts faithful); MCQ already shares
the `L op R = V` chain; the adapter sets `thinking.display='summarized'` for opus-4.x and a fixed
`max_tokens=8000`; the only test-count defect is **57 actual vs 55 stated**.

Recommended order:

- **Phase 0 — cheap, honest, table-stakes (Epic 7, P0):** test-count fix; `validate --check` +
  version single-sourcing; CI regression gate + artifact-regeneration check.
- **Phase 1 — foundations + honesty layer + real-model spine (parallel where independent):**
  the ProbeResult/registry guard and the orthogonality/seed-sweep tightening *before any new probe*;
  noise injection + shuffled-label control + frozen-population split + faithful-by-construction
  control + the by-construction honesty docs (Epic 1); and the real-model spine (Epic 5: API surface
  + caveat → SDK-shape verification → transport hardening incl. JSON escaping → SIM parse-failure fix
  → **keyed recording + e2e test**). The keyed recording is the only step needing a key/spend and
  gates GSM8K, judge-vs-gold kappa, the model-agnostic HTML report, and the thinking-vs-answer
  diagnostic, so it is a *gated do-it-now* item, not deferred.
- **Phase 2 — novelty payload (all on the noisy substrate + extended population):** FIL/PAR/IPR
  probes (Epic 3); monitor-grade metrics and ECE wiring (Epic 4); then OBF (monitor fixed first);
  then the SHI taxonomy and the reward-hack sub-signal last.
- **Cross-cutting (Epic 6):** the cluster bootstrap + per-domain AUROC precedes the
  kappa/multiple-comparison reframing; DeLong/permutation only after noise exists; GSM8K and the
  FaithCoT-Bench positioning after the cache and judge-kappa.

Re-run the CI regression gate after every numerics-affecting task to keep the committed artifact
honest.

## Invariants we will not break

1. **Runtime dependency is numpy only.** Metrics, AUROC/ECE/bootstrap, SVG, and HTML stay
   hand-rolled; no scipy/sklearn/matplotlib in runtime deps. `anthropic` and the dev tooling
   (`pytest`, `ruff`, coverage) are optional extras only.
2. **Deterministic, no-API-key, no-network core.** The synthetic validation runs with no key and
   no network, fully seeded via the existing `hashlib.sha256(seed, problem.id, keys, trial)` scheme
   — never unseeded `random`/`np.random`. Anything touching the network degrades to the committed
   replay cache and **must never break the core suite**.
3. **Green, reproducible artifact.** The full test suite stays green; the committed `results.json`,
   HTML report, and SVGs reproduce numerically-identically from the documented seed/params, enforced
   by the CI regression and artifact-regeneration gates.
4. **Research honesty / no overclaiming.** A synthetic AUROC is never presented as evidence of
   real-model probe sensitivity; by-construction results are labeled as such; monitor-evasion axes
   are "offline known-label *analogues*", not reproductions; the monitor is fixed *before* any
   obfuscation is defined; real-model numbers are reported as descriptive statistics with the
   no-ground-truth caveat.
