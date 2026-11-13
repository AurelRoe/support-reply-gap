"""Evaluate a model against the gold set and regenerate the eval table.

    python -m support_reply_gap.eval --model baseline
    python -m support_reply_gap.eval --report-only     # rebuild reports/eval.md

Needs Postgres for the review text.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import pandas as pd

from support_reply_gap.eval import report
from support_reply_gap.eval.harness import Classifier, evaluate, save
from support_reply_gap.models.data import load_gold, xy


def build_baseline(gold: pd.DataFrame) -> Classifier:
    from support_reply_gap.models.baseline import BaselineClassifier

    # Train only: validation stays unseen so its score means what it meant in C1.
    return BaselineClassifier(*xy(gold, "train"))


MODELS: dict[str, Callable[[pd.DataFrame], Classifier]] = {"baseline": build_baseline}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m support_reply_gap.eval")
    parser.add_argument("--model", choices=sorted(MODELS))
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.model and not args.report_only:
        parser.error("give --model or --report-only")

    if args.model:
        gold = load_gold()
        model = MODELS[args.model](gold)
        result = evaluate(model, {s: xy(gold, s) for s in ("val", "test")})
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


if __name__ == "__main__":
    main()
