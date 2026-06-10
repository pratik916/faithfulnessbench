# FaithfulnessBench

**Stop trusting the model's scratchpad.** A causal-intervention harness that measures whether a reasoning model's chain-of-thought (CoT) *actually drives* its answer — and, unlike prior single-probe work, **validates the measurement itself** against models whose (un)faithfulness is known by construction.

[![CI](https://github.com/pratik916/faithfulnessbench/actions/workflows/ci.yml/badge.svg)](https://github.com/pratik916/faithfulnessbench/actions/workflows/ci.yml) ![tests](https://img.shields.io/badge/tests-229%20passing-brightgreen) ![coverage](https://img.shields.io/badge/coverage-%E2%89%A590%25-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-blue) ![license](https://img.shields.io/badge/license-MIT-green)

**▶ [Open the live interactive report](https://pratik916.github.io/faithfulnessbench/)** — probe-validity KPIs, the AUROC-vs-noise sensitivity curve, the CoT-obfuscation robustness frontier, and an interactive trace viewer (slide how much reasoning the model sees; toggle a corrupted step). No clone required.

---

## Why this matters

Chain-of-thought monitoring is a leading AI-oversight strategy: read the model's reasoning, catch the bad behavior before it acts. **But monitoring is only valid if the CoT is *causally faithful*** — if the stated reasoning reflects the computation that actually produced the answer, rather than a fluent post-hoc rationalization. A model that silently follows a planted hint while writing a clean, hint-free derivation is *unfaithful*, and no amount of CoT-reading will catch it.

Measuring faithfulness is hard because it's a **counterfactual claim about an unobservable cause**. Most prior work proposes a *single* probe and reports its output as "faithfulness" — which is circular: the probe defines the very thing it claims to measure, with no way to check it's right.

This is the agenda a 2025 multi-lab position paper calls **CoT monitorability** (Korbak, Balesni et al., <https://arxiv.org/abs/2507.11473>): a monitor that reads the chain-of-thought is useful only to the degree the CoT is both **faithful** (it reflects the real computation) *and* **covering/legible** (it actually verbalizes the decision-relevant factors). FaithfulnessBench's four probes are components of exactly such a monitor — SHI, CSC and EAR target *faithfulness*, while SIM (counterfactual simulatability) targets *coverage*: can an observer predict the answer from the CoT at all. Monitorability ≈ **faithfulness × coverage**, and the card reports both. We measure **behavioral** (black-box) monitorability; **white-box** / activation-level probing on *real* models is out of scope — though the report ships a numpy **white-box offline analogue** (§1d) showing where an activation probe beats an obfuscated CoT monitor and where it stays blind.

**FaithfulnessBench breaks the circularity.** It runs four orthogonal causal probes and then asks the question almost no faithfulness work asks: *are the probes themselves valid?* We build synthetic models whose decision rule is fixed in code — so we **know**, per case, whether each is being faithful — and measure how well each probe's score recovers that planted ground truth (by AUROC). That turns a slippery construct into a falsifiable instrument. (The honest yardstick: hand-annotated benchmarks like [FaithCoT-Bench](https://arxiv.org/abs/2510.04040) put the *best* behavioral detector near ~0.70 AUROC, so our synthetic 1.000 is a wiring/specificity control, not a real-world score.)

## Headline result

> On a seeded population of 6 synthetic models × 40 problems, **every probe detects the unfaithfulness it targets at AUROC = 1.000**, with **exactly zero leakage** onto the failure modes it does *not* target (off-axis AUROC = 0.500 — the tie convention's value when a probe returns identically zero). This certifies that **each probe is correctly *wired* to its failure mode and is specific** — and a shuffled-label negative control collapses every probe to ≈ 0.50, proving the harness *can* report chance (it isn't a metric artifact). The probes genuinely disagree — a model can fail one and pass the others — which is the case for a **Faithfulness Card**, not a scalar.

> **What this does and does not prove (read this).** The synthetic signal is *clean*, so AUROC = 1.000 is the expected ceiling of a controlled positive/negative test — it is **true by construction**, certifying wiring and specificity, **not** sensitivity on real, noisy chain-of-thought. The combined detector reaching 1.000 while each single probe sits at *exactly* 0.700 is likewise a **population identity** (all four single-axis probes score the same 0.700 by the population's composition), **not** a discriminative result. The honest real-world reference point is the ≈ 0.70 ceiling that hand-annotated benchmarks such as [FaithCoT-Bench](https://arxiv.org/abs/2510.04040) report for the *best* behavioral detector — what measurement on a hard substrate actually looks like. **What would falsify the harness:** a probe whose targeted AUROC stayed at 1.000 on a noised substrate with partially-shuffled labels, or a negative control that did *not* fall to chance. Turning AUROC from a wiring check into a genuine sensitivity *measurement* (a controlled-noise regime, then real Claude CoT) is the next step — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

<p align="center">
  <img src="docs/assets/detection_auroc.svg" width="48%" alt="Detection AUROC per probe">
  <img src="docs/assets/specificity.svg" width="48%" alt="Probe-vs-axis specificity matrix">
</p>

*Left: each probe's AUROC for detecting its targeted induced unfaithfulness (dashed line = chance). Right: the specificity matrix — the diagonal (each probe vs its own axis) is 1.0; the off-diagonal (each probe vs a different failure mode) is ~0.5. The probes are specific, not a generic "something is off" detector.*

## The four probes

Each probe consumes a model and a set of problems and returns a continuous **unfaithfulness score in [0, 1]** per instance. They are *orthogonal* — each isolates a different way a CoT can be unfaithful.

| Probe | Failure mode it catches | How it intervenes |
|---|---|---|
| **SHI** — Silent Hint Injection | The answer is driven by a cue the CoT never admits to | Plant a hint pointing at a wrong target; flag cases where the answer flips to the hint but the CoT never mentions it |
| **CSC** — CoT Step Corruption | The CoT isn't load-bearing; corrupting it doesn't change the answer | Perturb an operand (format-class-preserving) and re-derive; faithful reasoning tracks the change, post-hoc reasoning doesn't |
| **SIM** — Counterfactual Simulatability | The CoT doesn't let an observer predict the answer | A simulator predicts the answer from the CoT *alone* (leakage-controlled: it can't re-solve the question) |
| **EAR** — Early-Answering / Reasoning-Reliance | The model committed to the answer before the CoT did any work | Truncate the CoT at increasing fractions; faithful answers only converge as reasoning is revealed |

Full methodology, with the formal protocol for each probe, is in [`docs/DESIGN.md`](docs/DESIGN.md).

## The validation idea (the part that makes it research, not vibes)

`ConfigurableSyntheticModel` is a competent solver whose **faithfulness is dialed in code**. Each of four dials is an *unfaithfulness rate* targeting exactly one probe:

| dial | unfaithful behavior | probe it should trip |
|---|---|---|
| `p_hint_sycophancy` | silently adopt an injected hint | SHI |
| `p_post_hoc` | ignore the CoT when it's corrupted | CSC |
| `p_decoy_cot` | state a conclusion in the CoT ≠ the actual answer | SIM |
| `p_pre_commit` | lock the answer before reasoning | EAR |

Because the dials are code, every `(model, problem)` carries a **known label**. We instantiate one fully-faithful model, one single-axis-unfaithful model per probe, and one fully-unfaithful model — then check (a) that each probe's AUROC for its targeted axis is ~1.0 and (b) that it's ~0.5 on the other axes. The synthetic model is deliberately **not** an LLM: its value is that the ground truth is beyond dispute. **The identical probe code runs unchanged against a real Claude model** — only the backend differs.

## Related benchmarks

| Benchmark | How it gets faithfulness ground truth | Reports |
|---|---|---|
| **FaithfulnessBench** (this) | **Synthetic, by construction** — dials in code, exact per-instance labels | probe-detection AUROC, F1, Cohen's kappa |
| [FaithCoT-Bench](https://arxiv.org/abs/2510.04040) | Expert human annotation (300+ instances) | F1 / Cohen's kappa; best behavioral detector ~0.70 AUROC, no cross-setting transfer |
| RFEval / FRODO ([2402.13950](https://arxiv.org/abs/2402.13950)) | Behavioral counterfactual consistency | causal-robustness scores |

So that the headline is comparable to the annotation-based work, the validation reports **F1 and Cohen's kappa beside AUROC** for the combined detector (all 1.0 on the clean synthetic population — the same by-construction caveat). **Scope caveat:** FaithCoT-Bench finds counterfactual methods succeed in *math* but degrade in *knowledge* domains; this harness is arithmetic-heavy, so its real-model evidence (incl. GSM8K) speaks to math reasoning, not open-domain knowledge.

### Cross-domain transfer: does it run beyond linear arithmetic? (GSM8K)

The cheapest honest answer to *"does it generalize?"* is to run the **identical probe code** over real grade-school math (GSM8K) on the cached real-model path — `faithfulnessbench transfer` ([standalone page](report/transfer.html)). The result is reported **descriptively**: GSM8K has no faithfulness ground truth, so there is *no* AUROC-vs-truth on that side — only whether each probe still runs and what it produces.

| Probe | Synthetic arithmetic (ground truth) | GSM8K real-math (descriptive) |
|---|---|---|
| SHI | 1.000 | 1.000 |
| CSC | 1.000 | — (degraded) |
| SIM | 1.000 | 0.000 |
| EAR | 0.917 | 0.994 |

SHI and EAR **run unchanged** on free-text GSM8K CoT; CSC and SIM **degrade** because real reasoning has no parseable `L op R = V` chain to corrupt or exactly simulate — exactly the real-path limitation documented in `docs/DESIGN.md`. (The GSM8K column currently replays from the labeled *fake* cache; it becomes a real measurement with the one-command real recording.) This is *in-domain* (still arithmetic) and does not claim cross-domain *validity* transfer — consistent with FaithCoT-Bench's math→knowledge finding above.

### White-box vs black-box: where does activation probing win? (offline analogue)

A numpy **offline analogue** of activation probing (cf. [Apollo](https://arxiv.org/abs/2502.03407)): each synthetic model gets a generic, a-priori *activation* vector and a logistic probe is fit on a held-out split. The honest result is the **contrast**, not the (by-construction) white-box number:

| Setting | White-box (activation probe) | Black-box (CoT-reading monitor) |
|---|---|---|
| **Obfuscation** — the model states a *wrong* answer, evading the monitor | **0.91** | **0.40** ≈ chance (evaded) |
| **Filler** — unfaithful only under *intervention* | **0.50** ≈ chance (blind) | **0.98** (the FIL probe catches it) |
| Permuted-label control | **0.47** (signal destroyed) | — |

Where an obfuscated CoT defeats the text-reading monitor, the activation probe reads the wrong-answer signal directly — **but** it is blind to the interventional FILLER axis a causal probe catches. The two are *complementary*; white-box probing does not replace causal probes. The white-box numbers are high *by construction* (we authored both the activation and the label), so this **proves nothing about real-model activations** — it is an offline analogue, surfaced as report [§1d](https://pratik916.github.io/faithfulnessbench/).

## Why a card, not a single number

<p align="center">
  <img src="docs/assets/correlation.svg" width="44%" alt="Cross-probe correlation">
  <img src="docs/assets/faithfulness_matrix.svg" width="52%" alt="Faithfulness by model and probe">
</p>

The correlation matrix (left) is a **sanity check that the probes don't spuriously co-fire**. Framed honestly as *chance-corrected agreement* on the binary "flagged unfaithful?" decision, the mean off-diagonal **Cohen's kappa ≈ 0** — the probes agree no more than chance off their shared corners. (The raw off-diagonal Spearman ≈ 0.25 says the same thing but is population-composition-dependent, so we don't lead with it.) The substantive, robust point is the faithfulness matrix on the right: the **`sycophant`** model scores **0.00 on SHI but 1.00 on SIM and CSC** — any single probe used alone would have cleared it. That's the case for reporting a **Faithfulness Card** — four sub-scores plus a transparent composite (the mean — intentionally not a learned weighting) — rather than a single faithfulness number.

## Quickstart

No API key, no network — the validation runs against synthetic models and produces real, reproducible numbers:

```bash
pip install -e .
faithfulnessbench validate          # seeded ground-truth validation
open report/faithfulness_report.html # self-contained interactive report
```

The [live interactive report](https://pratik916.github.io/faithfulnessbench/) (also written locally to `report/faithfulness_report.html`) includes a **trace viewer**: pick a problem and watch a planted hint silently flip the model's answer while its chain-of-thought stays clean, then step the EAR truncation slider and toggle the CSC corruption to see the causal interventions live.

<p align="center">
  <a href="https://pratik916.github.io/faithfulnessbench/"><img src="docs/assets/trace_viewer.gif" width="80%" alt="The interactive trace viewer: a silent hint flip, then the EAR slider showing an answer that locks in before any reasoning (early-lock = unfaithful), then toggling a corrupted CSC step where the answer ignores the corruption (post-hoc)"></a>
</p>

*The interactive trace viewer in the report: a silent hint flip, then the EAR slider (an answer that's already fixed at 0% reasoning = early lock), then a CSC corruption toggle (the answer ignores the corrupted step = post-hoc). All client-side in one self-contained HTML file.*

<p align="center">
  <a href="https://pratik916.github.io/faithfulnessbench/"><img src="docs/assets/report.png" width="90%" alt="The self-contained HTML report: probe-validity KPIs, detection-AUROC and ROC charts, and the AUROC-vs-label-noise sensitivity curve with monitor catch-rates"></a>
</p>

*The top of the generated report — every probe's detection AUROC, and (below) the honest part: as label noise rises, AUROC falls from the by-construction 1.000 toward chance, so it reads as a real sensitivity measurement rather than a wiring check. The full page also includes the specificity matrix, the cross-probe agreement, per-model cards, and the interactive trace viewer.*

### Reproduce the exact numbers

```bash
python experiments/validate_synthetic.py   # writes results.json, the HTML report, and SVG figures
faithfulnessbench validate --check          # recompute & verify the committed numbers — never overwrites
```

Everything is seeded — the committed [`experiments/results/results.json`](experiments/results/results.json) and figures regenerate byte-for-(numerically-)identically. `validate --check` recomputes the validation and diffs the **headline numbers** (problem/model counts, every AUROC, the Spearman correlation matrix, and per-model composite + per-probe faithfulness) against the committed file to an absolute tolerance of `1e-12`; it deliberately ignores provenance strings and the seeded bootstrap CIs, and exits non-zero on drift without touching the artifact. The exact-vs-ignored field policy lives in [`src/faithfulnessbench/artifacts.py`](src/faithfulnessbench/artifacts.py).

## Scoring a real model

The same probes run against a live Claude reasoning model via the Anthropic adapter:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
faithfulnessbench score --model claude-sonnet-4-6 --effort high
```

Adaptive thinking captures the CoT; the "answer from this given reasoning" probes disable thinking so the model commits to the supplied reasoning instead of re-deriving. Every API call is memoised to a record/replay cache, so a committed cache reproduces real-model numbers offline and reruns don't re-bill.

## Limitations (read these)

- The synthetic world validates that **the probes detect the unfaithfulness they target**. It does **not** claim any real frontier model is (un)faithful to a particular degree — that requires running the harness against real models, which the adapter supports and which is the natural next step.
- On real models, CSC and EAR rely on "continue/answer from this (partial) reasoning" prompting, an *approximation* of a true intervention.
- The default cue-verbalization (SHI) and simulator (SIM) are exact in the synthetic world; the real-model path uses an LLM judge whose own reliability is a dependency.
- This is **behavioral** (black-box) faithfulness. White-box / activation-level faithfulness on *real* models is out of scope; the report includes a numpy **offline analogue** (§1d) — an activation probe that beats an obfuscated CoT monitor yet stays blind to the interventional FILLER axis — which proves nothing about real-model activations.

## Project layout

```
src/faithfulnessbench/
  metrics.py          # AUROC, ROC, ECE, Spearman, bootstrap CIs — pure numpy, hand-rolled
  problems.py         # arithmetic & MCQ problems with planted shortcuts
  models/
    base.py           # Model interface + detector/simulator protocols
    synthetic.py      # ConfigurableSyntheticModel — the ground-truth world
    anthropic_model.py# real Claude adapter (injectable transport + replay cache)
  probes/             # shi · csc · sim · ear
  card.py             # Faithfulness Card aggregation + cross-probe correlation
  viz/svg.py          # dependency-free SVG charts
  report/html.py      # self-contained HTML report + interactive trace viewer
  validation.py       # the headline experiment, as an importable function
  cli.py              # faithfulnessbench {validate, score, report}
experiments/validate_synthetic.py
docs/DESIGN.md        # methodology (source of truth)
tests/                # unit, synthetic-world, probe, card, adapter, repo-consistency & e2e tests
```

## Development

```bash
pip install -e ".[dev]"
pytest                 # full suite: metrics, synthetic world, probes, card, adapter, e2e
```

The runtime dependency is **numpy only** — metrics, plotting, and reporting are implemented from scratch so the harness installs and runs from a clean clone. CI runs the suite and the full validation on every push.

## License

MIT — see [LICENSE](LICENSE).
