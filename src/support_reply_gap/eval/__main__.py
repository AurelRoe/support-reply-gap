"""Evaluate a model against the gold set and regenerate the eval table.

    python -m support_reply_gap.eval --model baseline
    python -m support_reply_gap.eval --model llm --limit 5    # paid smoke test, nothing saved
    python -m support_reply_gap.eval --model llm --max-usd 0.10
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
    parser.add_argument("--max-usd", type=float, default=0.10, help="hard spending cap per run")
    parser.add_argument(
        "--limit", type=int, help="smoke test: first N validation reviews only, nothing saved"
    )
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
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
        from support_reply_gap.models.llm import BudgetExceeded

        gold = load_gold()
        model = MODELS[args.model](gold, max_usd=args.max_usd)
        splits = {s: split_inputs(gold, s) for s in ("val", "test")}
        if args.limit:
            ids, texts, y = splits["val"]
            splits = {"val": (ids[: args.limit], texts[: args.limit], y.iloc[: args.limit])}
        if hasattr(model, "estimate_usd") and not confirm_spend(model, splits, args.yes):
            print("cancelled, nothing spent")
            return
        try:
            result = evaluate(model, splits)
        except BudgetExceeded as err:
            raise SystemExit(f"stopped: {err}. Answers so far are cached; re-run to continue.")
        print_run(result)
        if args.limit:
            print("smoke test: nothing saved")
            return
        path = save(result)
        print(f"wrote {path.relative_to(report.ROOT)}")

    path = report.write()
    print(f"wrote {path.relative_to(report.ROOT)}")


def confirm_spend(model, splits: dict, assume_yes: bool) -> bool:
    """Show what the run will cost before anything is spent, and ask."""
    texts = [text for _, split_texts, _ in splits.values() for text in split_texts]
    est = model.estimate_usd(texts)
    if est["uncached"] == 0:
        print(f"all {len(texts)} answers are cached: this run costs $0")
        return True
    print(
        f"{est['uncached']} of {len(texts)} reviews need an API call: expected "
        f"${est['expected_usd']:.3f}, worst case ${est['worst_case_usd']:.3f}, "
        f"hard cap ${model.max_usd:.2f}"
    )
    return assume_yes or input("continue? [y/N] ").strip().lower() == "y"


def print_run(result: dict) -> None:
    for split, s in result["splits"].items():
        low, high = s["macro_f1_ci95"]
        print(
            f"  {split}: macro-F1 {s['macro_f1']:.3f} ({low:.2f}–{high:.2f}), "
            f"p95 {s['latency_p95_ms']:.0f} ms, ${s['cost_per_1k_usd']:.2f} per 1k"
        )
    config = result["config"]
    if "new_spend_usd" in config:
        print(
            f"  API calls {config['api_calls']}, cache hits {config['cache_hits']}, "
            f"failures {config['failures']}, new spend ${config['new_spend_usd']:.4f}"
        )


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
