"""Command-line interface: ``faithfulnessbench {validate, score, report}``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

_DEFAULT_REPORT = "report/faithfulness_report.html"
_DEFAULT_JSON = "experiments/results/results.json"


def _cmd_validate(args: argparse.Namespace) -> int:
    from .report import write_report
    from .validation import json_safe, run_validation, summarize

    report = run_validation(
        n_per_domain=args.n, seed=args.seed, n_trials=args.trials,
        reproduce_cmd="faithfulnessbench validate",
    )
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(json_safe(report), indent=2))
    write_report(report, args.report)
    print(summarize(report))
    print(f"\nWrote results -> {args.json}")
    print(f"Wrote report  -> {args.report}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .report import write_report

    report = json.loads(Path(args.json).read_text())
    write_report(report, args.report)
    print(f"Wrote report -> {args.report}")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    from .card import build_card
    from .models.anthropic_model import AnthropicModel
    from .problems import mixed_problems

    model = AnthropicModel(args.model, effort=args.effort, cache_path=args.cache)
    problems = mixed_problems(args.n, seed=args.seed)
    try:
        card = build_card(model, problems, n_trials=args.trials)
    except RuntimeError as exc:  # missing SDK / key
        print(f"error: {exc}", file=sys.stderr)
        print('hint: pip install "faithfulnessbench[anthropic]" and set ANTHROPIC_API_KEY', file=sys.stderr)
        return 2

    print(f"Faithfulness Card — {card.model_name}")
    print(f"  composite faithfulness: {card.composite_faithfulness:.3f}")
    for name, d in card.probe_scores.items():
        print(f"  {name}: faithfulness {d['faithfulness']:.3f}  (CI {d['ci_lo']:.3f}-{d['ci_hi']:.3f})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(card.to_dict(), indent=2))
        print(f"Wrote card -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="faithfulnessbench",
        description="Measure and validate chain-of-thought faithfulness in reasoning models.",
    )
    parser.add_argument("--version", action="version", version=f"faithfulnessbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="run the seeded ground-truth validation and write a report")
    v.add_argument("-n", type=int, default=20, help="problems per domain (default 20)")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--trials", type=int, default=5, help="trials per probe (default 5)")
    v.add_argument("--report", default=_DEFAULT_REPORT)
    v.add_argument("--json", default=_DEFAULT_JSON)
    v.set_defaults(func=_cmd_validate)

    r = sub.add_parser("report", help="render the HTML report from an existing results.json")
    r.add_argument("--json", default=_DEFAULT_JSON)
    r.add_argument("--report", default=_DEFAULT_REPORT)
    r.set_defaults(func=_cmd_report)

    s = sub.add_parser("score", help="score a real Claude model (needs ANTHROPIC_API_KEY)")
    s.add_argument("--model", default="claude-sonnet-4-6")
    s.add_argument("-n", type=int, default=15, help="problems per domain (default 15)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--trials", type=int, default=3)
    s.add_argument("--effort", default="medium", choices=["low", "medium", "high", "max"])
    s.add_argument("--cache", default=".fb_cache/score.json", help="record/replay cache path")
    s.add_argument("--out", default=None, help="optional path to write the card JSON")
    s.set_defaults(func=_cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
