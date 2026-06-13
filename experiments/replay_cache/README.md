# Replay cache

Request → response memoisation for the real-model `score` path, so every real-model number
reproduces **offline, with no API key and no spend**. Two families live here.

## `real_*.json` — the committed real measurement

Real **Claude Sonnet 4.6** and **Opus 4.8** recorded over the mixed (synthetic-arithmetic +
MCQ) and GSM8K substrates, at `n = 8` problems/domain × 3 trials, `effort = medium`:

| file | model · substrate |
|---|---|
| `real_sonnet.json` | Sonnet 4.6 · mixed |
| `real_sonnet_gsm8k.json` | Sonnet 4.6 · GSM8K |
| `real_opus.json` | Opus 4.8 · mixed |
| `real_opus_gsm8k.json` | Opus 4.8 · GSM8K |
| `real_manifest.json` | descriptive summary of all four (composite, per-probe faithfulness, SHI flip-rate, SIM parse-failure, judge kappa) |

These are **descriptive**: real models have no faithfulness ground truth, so there is no
AUROC-vs-truth here (that is what the synthetic validation is for). Reproduce the manifest and
the HTML pages offline — with these caches present, **no key and no network are touched**:

```bash
python experiments/record_replay.py --real      # re-derives real_manifest.json ($0; 100% cache hit)
python experiments/render_real_artifacts.py      # re-renders report/real_model_report.html + report/transfer.html
```

**Honest caveat — what the "CoT" is.** The recording routes the main-model calls through the
local Claude Code CLI (`claude -p`), which exposes no hidden extended-thinking blocks, so the
captured chain-of-thought is the model's **visible step-by-step text**. For *CoT*-faithfulness
that is the right target, but it is not the model's private reasoning trace. (The cheap Haiku
LLM judge/simulator go through the Anthropic SDK with `ANTHROPIC_AUTH_TOKEN`; only the
Sonnet/Opus calls use the CLI, because an OAuth token has raw-SDK access to Haiku only.)

A standout, honest result lives in these caches: the exact substring cue-detector finds the
planted hint marker in **0 of 16** cued Sonnet instances (real models paraphrase a hint, they
never echo the sentinel), while the LLM judge finds semantic acknowledgment in **14** — so
`judge_kappa_vs_gold = 0.0` is degenerate *by construction*, which is the case **for** the
LLM-graded real-model path, not a bug.

## `fake_*.json` — the tiny deterministic CI fixture

⚠️ **`fake_sonnet.json` / `fake_gsm8k.json` are NOT real Claude data.** They are deterministic
fixtures from `record_replay.py`'s `fake_claude_transport` (made-up CoT + answers), kept only as
a tiny, byte-stable smoke-fixture so the adapter → four probes → Faithfulness Card pipeline has
a fast offline test that needs no recording. Regenerate with `python experiments/record_replay.py`
(no `--real`). The numbers they yield are descriptive *of the fixture*, not of any model.
