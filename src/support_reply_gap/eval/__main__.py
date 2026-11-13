"""Evaluate a model against the gold set and regenerate the eval table.

    python -m support_reply_gap.eval --model baseline
    python -m support_reply_gap.eval --report-only     # rebuild reports/eval.md
    python -m support_reply_gap.eval --compare llm baseline   # paired difference
    python -m support_reply_gap.eval --gate            # CI: check committed results

Scoring a model needs Postgres for the review text; --report-only, --compare and
--gate read only committed files.
"""

from __future__ import annotations

import argparse

from support_reply_gap.eval import report
from support_reply_gap.eval.compare import paired_difference
from support_reply_gap.eval.harness import evaluate, save
from support_reply_gap.eval.registry import MODELS, split_inputs
from support_reply_gap.models.data import load_gold


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m support_reply_gap.eval")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"))
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if not (args.model or args.report_only or args.compare or args.gate):
        parser.error("give --model, --report-only, --compare or --gate")

    if args.gate:
        from support_reply_gap.eval.gate import run_gate

        problems = run_gate()
        for problem in problems:
            print(f"FAIL {problem}")
        if problems:
            raise SystemExit(1)
        print("eval gate passed")
        return

    if args.compare:
        compare(*args.compare)
        return

    if args.model:
        gold = load_gold()
        model = MODELS[args.model](gold)
        result = evaluate(model, {s: split_inputs(gold, s) for s in ("val", "test")})
        path = save(result)
        print(f"wrote {path.relative_to(report.ROOT)}")
        for split, s in result["splits"].items():
            low, high = s["macro_f1_ci95"]
            print(
                f"  {split}: macro-F1 {s['macro_f1']:.3f} ({low:.2f}–{high:.2f}), "
                f"p95 {s['latency_p95_ms']:.0f} ms, ${s['cost_per_1k_usd']:.2f} per 1k"
            )

    path = report.write()
    print(f"wrote {path.relative_to(report.ROOT)}")


def compare(a: str, b: str) -> None:
    """Print macro-F1 of a minus b, latest run of each, with a paired 95% interval."""
    latest = {r["model"]: r for r in report.latest_per_model(report.load_results())}
    for name in (a, b):
        if name not in latest:
            raise SystemExit(f"no result for {name!r} in {report.RESULTS_DIR}")
    gold = report.gold_labels()
    for split in ("val", "test"):
        d = paired_difference(latest[a], latest[b], split, gold)
        low, high = d["ci95"]
        print(
            f"{split}: {a} - {b} = {d['diff']:+.3f}  (95% CI {low:+.3f} to {high:+.3f}, n={d['n']})"
        )


if __name__ == "__main__":
    main()
