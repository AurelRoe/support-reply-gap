"""Tests for the C3 eval harness. A fake classifier with known answers; no database."""

import json

import numpy as np
import pandas as pd
import pytest

from support_reply_gap.eval.harness import (
    bootstrap_interval,
    confusion,
    evaluate,
    fast_macro_f1,
    gold_fingerprint,
    run_split,
    save,
    score,
)
from support_reply_gap.eval.metrics import macro_f1
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


class FakeClassifier:
    """Answers from a lookup table and charges a fixed amount per call."""

    name = "fake"

    def __init__(self, answers: dict[str, set[str]], cost: float = 0.001):
        self.config = {"kind": "lookup"}
        self.answers = answers
        self.cost = cost

    def classify(self, text: str) -> tuple[set[str], float]:
        return self.answers[text], self.cost


def frame(label_sets: list[set[str]]) -> pd.DataFrame:
    return pd.DataFrame([{n: int(n in s) for n in LABEL_NAMES} for s in label_sets])


TRUTH = [{"login_access"}, {"performance"}, {"feature_gap", "support_experience"}, {"no_complaint"}]
TEXTS = pd.Series([f"review {i}" for i in range(len(TRUTH))])
IDS = pd.Series([f"id{i}" for i in range(len(TRUTH))])


def test_perfect_model_scores_one():
    model = FakeClassifier(dict(zip(TEXTS, TRUTH)))
    s = score(run_split(model, IDS, TEXTS, frame(TRUTH)))
    assert s["macro_f1"] == 1.0
    assert s["macro_f1_ci95"] == [1.0, 1.0]
    assert s["exact_match"] == 1.0


def test_cost_is_scaled_to_1000_reviews():
    model = FakeClassifier(dict(zip(TEXTS, TRUTH)), cost=0.002)
    assert score(run_split(model, IDS, TEXTS, frame(TRUTH)))["cost_per_1k_usd"] == pytest.approx(
        2.0
    )


def test_latency_is_measured():
    model = FakeClassifier(dict(zip(TEXTS, TRUTH)))
    s = score(run_split(model, IDS, TEXTS, frame(TRUTH)))
    assert s["latency_p95_ms"] >= 0


def test_labels_outside_the_taxonomy_are_rejected():
    model = FakeClassifier({t: {"made_up_label"} for t in TEXTS})
    with pytest.raises(ValueError, match="outside the taxonomy"):
        run_split(model, IDS, TEXTS, frame(TRUTH))


def test_fast_macro_f1_matches_the_shared_metric():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t = pd.DataFrame(rng.integers(0, 2, (30, len(LABEL_NAMES))), columns=list(LABEL_NAMES))
        p = pd.DataFrame(rng.integers(0, 2, (30, len(LABEL_NAMES))), columns=list(LABEL_NAMES))
        t.iloc[:, 0] = 0  # a label absent from the truth, as happens in small splits
        assert fast_macro_f1(t.to_numpy().astype(bool), p.to_numpy().astype(bool)) == (
            pytest.approx(macro_f1(t, p))
        )


def test_bootstrap_interval_contains_the_score():
    truth = TRUTH * 10
    # A few scattered errors, so resamples differ in how many they catch.
    pred = [{"no_complaint"} if i in (0, 5, 13, 22, 31) else s for i, s in enumerate(truth)]
    y_true, y_pred = frame(truth), frame(pred)
    low, high = bootstrap_interval(y_true, y_pred)
    assert low <= macro_f1(y_true, y_pred) <= high
    assert low < high


def test_confusion_shows_where_a_true_label_went():
    y_true = frame([{"performance"}, {"performance"}])
    y_pred = frame([{"performance"}, {"crash_stability"}])
    m = confusion(y_true, y_pred)
    assert m.loc["performance", "performance"] == 1
    assert m.loc["performance", "crash_stability"] == 1
    assert m.loc["performance", "missed"] == 1


def test_fingerprint_changes_with_the_gold_files(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("a,1\n")
    before = gold_fingerprint((labels,))
    labels.write_text("a,0\n")
    assert gold_fingerprint((labels,)) != before


def test_saved_result_is_readable_json(tmp_path):
    model = FakeClassifier(dict(zip(TEXTS, TRUTH)))
    result = evaluate(model, {"val": (IDS, TEXTS, frame(TRUTH))})
    path = save(result, tmp_path)
    loaded = json.loads(path.read_text())
    assert loaded["model"] == "fake"
    assert loaded["splits"]["val"]["macro_f1"] == 1.0
    assert path.name.startswith("fake-")


def test_predictions_are_stored_per_review_id():
    model = FakeClassifier(dict(zip(TEXTS, TRUTH)))
    preds = score(run_split(model, IDS, TEXTS, frame(TRUTH)))["predictions"]
    assert preds["id0"] == ["login_access"]
    assert preds["id2"] == ["support_experience", "feature_gap"]  # taxonomy order
