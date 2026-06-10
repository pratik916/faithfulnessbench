# Replay cache (FAKE fixture)

⚠️ **`fake_sonnet.json` is NOT real Claude data.** It is a deterministic fixture produced by
`experiments/record_replay.py`'s `fake_claude_transport` (made-up chain-of-thought + answers),
committed so the full real-model `score` pipeline — Anthropic adapter → four probes →
Faithfulness Card — runs end-to-end **offline in CI, with no API key and no spend**, and so the
cache *format* is exactly what a real recording produces.

The numbers it yields are **descriptive of the fixture, not of any real model**. In particular
it honestly surfaces the real-path limitations: SIM's `parse_failure_rate` is 1.0 (free-text CoT
doesn't parse as `L op R = V`) and CSC has no parseable steps to corrupt.

## Recording against the real model (costs money, needs a key)

```bash
ANTHROPIC_API_KEY=sk-... python experiments/record_replay.py --real
```

records the **identical** requests through the live SDK into this same file — a one-command swap.
Then `faithfulnessbench score --cache experiments/replay_cache/fake_sonnet.json -n 2 --trials 2`
replays the real numbers offline.
