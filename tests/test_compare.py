"""Tests for the paired comparison of two eval results. Hand-made results; no database."""

import pandas as pd
import pytest

from support_reply_gap.eval.compare import paired_difference
from support_reply_gap.eval.metrics import macro_f1
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

# Every label occurs once in the gold labels, so the shared label set is all eleven.
GOLD_SETS = [{name} for name in LABEL_NAMES] * 3
IDS = [f"r{i:02d}" for i in range(len(GOLD_SETS))]


def gold() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"review_id": rid, **{n: int(n in s) for n in LABEL_NAMES}}
            for rid, s in zip(IDS, GOLD_SETS)
        ]
    )


def result(model: str, pred_sets: list[set[str]], fingerprint: str = "abc") -> dict:
    preds = {rid: [n for n in LABEL_NAMES if n in s] for rid, s in zip(IDS, pred_sets)}
    return {
        "model": model,
        "gold_fingerprint": fingerprint,
        "splits": {"val": {"predictions": preds}},
    }


def frame(sets: list[set[str]]) -> pd.DataFrame:
    return pd.DataFrame([{n: int(n in s) for n in LABEL_NAMES} for s in sets])


def with_errors(every: int) -> list[set[str]]:
    """The gold labels, with every n-th review predicted as feature_gap instead."""
    return [{"feature_gap"} if i % every == 0 else s for i, s in enumerate(GOLD_SETS)]


def test_identical_models_differ_by_exactly_zero():
    d = paired_difference(result("a", GOLD_SETS), result("b", GOLD_SETS), "val", gold())
    assert d["diff"] == 0.0
    assert d["ci95"] == [0.0, 0.0]


def test_point_difference_matches_the_shared_metric():
    a, b = with_errors(4), with_errors(2)
    d = paired_difference(result("a", a), result("b", b), "val", gold())
    expected = macro_f1(frame(GOLD_SETS), frame(a)) - macro_f1(frame(GOLD_SETS), frame(b))
    assert d["diff"] == pytest.approx(expected)
    assert d["ci95"][0] <= d["diff"] <= d["ci95"][1]


def test_better_model_gets_a_positive_difference():
    d = paired_difference(result("a", with_errors(10)), result("b", with_errors(2)), "val", gold())
    assert d["diff"] > 0


def test_results_from_different_gold_sets_are_refused():
    with pytest.raises(ValueError, match="different gold sets"):
        paired_difference(result("a", GOLD_SETS), result("b", GOLD_SETS, "old"), "val", gold())


def test_results_on_different_reviews_are_refused():
    b = result("b", GOLD_SETS)
    b["splits"]["val"]["predictions"].pop("r00")
    with pytest.raises(ValueError, match="different val reviews"):
        paired_difference(result("a", GOLD_SETS), b, "val", gold())


def test_old_result_without_predictions_asks_for_a_rerun():
    old = {"model": "b", "gold_fingerprint": "abc", "splits": {"val": {"macro_f1": 0.5}}}
    with pytest.raises(ValueError, match="re-run"):
        paired_difference(result("a", GOLD_SETS), old, "val", gold())
