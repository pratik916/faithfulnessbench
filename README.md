# FaithfulnessBench

**Stop trusting the model's scratchpad.** A causal-intervention harness that measures whether a reasoning model's chain-of-thought (CoT) *actually drives* its answer — and, unlike prior single-probe work, **validates the measurement itself** against models whose (un)faithfulness is known by construction.

![tests](https://img.shields.io/badge/tests-119%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![deps](https://img.shields.io/badge/runtime%20deps-numpy%20only-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## Why this matters

Chain-of-thought monitoring is a leading AI-oversight strategy: read the model's reasoning, catch the bad behavior before it acts. **But monitoring is only valid if the CoT is *causally faithful*** — if the stated reasoning reflects the computation that actually produced the answer, rather than a fluent post-hoc rationalization. A model that silently follows a planted hint while writing a clean, hint-free derivation is *unfaithful*, and no amount of CoT-reading will catch it.

Measuring faithfulness is hard because it's a **counterfactual claim about an unobservable cause**. Most prior work proposes a *single* probe and reports its output as "faithfulness" — which is circular: the probe defines the very thing it claims to measure, with no way to check it's right.

This is the agenda a 2025 multi-lab position paper calls **CoT monitorability** (Korbak, Balesni et al., <https://arxiv.org/abs/2507.11473>): a monitor that reads the chain-of-thought is useful only to the degree the CoT is both **faithful** (it reflects the real computation) *and* **covering/legible** (it actually verbalizes the decision-relevant factors). FaithfulnessBench's four probes are components of exactly such a monitor — SHI, CSC and EAR target *faithfulness*, while SIM (counterfactual simulatability) targets *coverage*: can an observer predict the answer from the CoT at all. Monitorability ≈ **faithfulness × coverage**, and the card reports both. We measure **behavioral** (black-box) monitorability; **white-box** / activation-level probing is out of scope.

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

## Why a card, not a single number

<p align="center">
  <img src="docs/assets/correlation.svg" width="44%" alt="Cross-probe correlation">
  <img src="docs/assets/faithfulness_matrix.svg" width="52%" alt="Faithfulness by model and probe">
</p>

The correlation matrix (left) is a **sanity check that the probes don't spuriously co-fire**: on this single-axis population they agree only on the fully-unfaithful corner, so off-diagonal correlation is low (≈ 0.25 — a value set by the *population composition*, not a measured property of the probes, so it's a diagnostic rather than a headline number). The substantive, robust point is the faithfulness matrix on the right: the **`sycophant`** model scores **0.00 on SHI but 1.00 on SIM and CSC** — any single probe used alone would have cleared it. That's the case for reporting a **Faithfulness Card** — four sub-scores plus a transparent composite (the mean — intentionally not a learned weighting) — rather than a single faithfulness number.

## Quickstart

No API key, no network — the validation runs against synthetic models and produces real, reproducible numbers:

```bash
pip install -e .
faithfulnessbench validate          # seeded ground-truth validation
open report/faithfulness_report.html # self-contained interactive report
```

The [interactive report](report/faithfulness_report.html) includes a **trace viewer**: pick a problem and watch a planted hint silently flip the model's answer while its chain-of-thought stays clean.

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
- This is **behavioral** (black-box) faithfulness. White-box / activation-level faithfulness is out of scope.

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
