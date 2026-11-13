"""Tests for rendering the eval table. Hand-made result dicts; no database."""

from support_reply_gap.eval.report import latest_per_model, render
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

HUMAN = {"n": 50, "macro_f1": 0.69, "exact_match": 0.64}


def split_result(f1: float) -> dict:
    return {
        "n": 60,
        "macro_f1": f1,
        "macro_f1_ci95": [f1 - 0.1, f1 + 0.1],
        "exact_match": 0.3,
        "cost_per_1k_usd": 0.0,
        "latency_p95_ms": 4.2,
        "latency_median_ms": 3.0,
        "per_class": [
            {"label": n, "f1": 0.5, "support": 3, "predicted": 3, "few_examples": True}
            for n in LABEL_NAMES
        ],
        "confusion": {
            n: {**{p: int(p == n) for p in LABEL_NAMES}, "missed": 0} for n in LABEL_NAMES
        },
    }


def result(model: str, run_at: str, f1: float, fingerprint: str = "abc") -> dict:
    return {
        "model": model,
        "config": {"C": 10},
        "run_at": run_at,
        "commit": "1234567",
        "gold_fingerprint": fingerprint,
        "splits": {"val": split_result(f1), "test": split_result(f1)},
    }


def test_only_the_latest_run_per_model_is_shown():
    runs = [result("baseline", "2026-11-01", 0.4), result("baseline", "2026-11-13", 0.5)]
    assert [r["macro_f1"] for r in (x["splits"]["val"] for x in latest_per_model(runs))] == [0.5]


def test_table_has_ceiling_row_and_model_row():
    text = render([result("baseline", "2026-11-13", 0.49)], HUMAN, "abc")
    assert "| Human self-agreement (50 reviews) | 0.69 |" in text
    assert "| baseline | 0.49 (0.39–0.59) | 0.49 (0.39–0.59) |" in text


def test_result_from_another_gold_set_is_flagged():
    text = render([result("baseline", "2026-11-13", 0.49, fingerprint="old")], HUMAN, "abc")
    assert "older gold set" in text


def test_small_classes_are_italic():
    text = render([result("baseline", "2026-11-13", 0.49)], HUMAN, "abc")
    assert "| `fees_pricing` | *0.50 (3)* | *0.50 (3)* |" in text
