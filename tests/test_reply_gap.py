"""Tests for the C2 reply-gap analysis. Hand-written reviews; no database."""

import pandas as pd

from support_reply_gap.analysis import reply_gap
from support_reply_gap.analysis.reply_gap import (
    ALL_APPS,
    ALL_REVIEWS,
    label_population,
    reply_table,
    select_population,
    within_star_band,
)
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

GERMAN = "Die Überweisung kommt seit drei Tagen nicht an und der Support antwortet nicht."


def reviews(*texts: str | None) -> pd.DataFrame:
    return pd.DataFrame({"review_id": [f"r{i}" for i in range(len(texts))], "content": list(texts)})


def kept_ids(*texts: str | None) -> list[str]:
    return list(select_population(reviews(*texts))["review_id"])


def test_german_review_is_kept():
    assert kept_ids(GERMAN) == ["r0"]


def test_english_review_is_dropped():
    assert kept_ids("The transfer has not arrived for three days and support never answers.") == []


def test_one_word_emoji_and_empty_reviews_are_dropped():
    assert kept_ids("Super", "👍👍", None) == []


def test_repeated_text_is_kept():
    # Unlike the gold sample: each repeat is a separate review with its own reply status.
    assert kept_ids(GERMAN, GERMAN) == ["r0", "r1"]


def test_star_band_keeps_one_to_three_stars():
    rated = pd.DataFrame({"review_id": [f"s{s}" for s in range(1, 6)], "score": range(1, 6)})
    assert list(within_star_band(rated)["review_id"]) == ["s1", "s2", "s3"]


def toy_gold() -> pd.DataFrame:
    rows = [("g1", "train"), ("g2", "val"), ("g3", "test")]
    return pd.DataFrame(
        [
            {
                "review_id": rid,
                "split": split,
                "content": f"Text {rid}",
                **dict.fromkeys(LABEL_NAMES, 0),
            }
            for rid, split in rows
        ]
    )


def fake_model(monkeypatch) -> dict:
    """Swap the real model for a stub that records its training texts and predicts feature_gap."""
    seen = {}

    def fit_chosen(texts, y):
        seen["texts"] = list(texts)
        return "model"

    def predict(model, texts, threshold):
        return pd.DataFrame([{n: int(n == "feature_gap") for n in LABEL_NAMES} for _ in texts])

    monkeypatch.setattr(reply_gap, "fit_chosen", fit_chosen)
    monkeypatch.setattr(reply_gap, "predict", predict)
    return seen


def test_model_is_trained_on_train_and_val_never_test(monkeypatch):
    seen = fake_model(monkeypatch)
    label_population(reviews("Text g1", "Text g3", "neu"), toy_gold())
    assert seen["texts"] == ["Text g1", "Text g2"]


def test_labels_align_with_population_rows(monkeypatch):
    fake_model(monkeypatch)
    population = reviews("a", "b", "c")
    population.index = [7, 8, 9]  # a filtered frame keeps gaps in its index
    labelled = label_population(population, toy_gold())
    assert list(labelled["review_id"]) == ["r0", "r1", "r2"]
    assert labelled["feature_gap"].tolist() == [1, 1, 1]
    assert labelled[list(LABEL_NAMES)].notna().all().all()


def test_training_reviews_are_flagged(monkeypatch):
    fake_model(monkeypatch)
    population = pd.DataFrame({"review_id": ["g1", "g3", "x"], "content": ["a", "b", "c"]})
    labelled = label_population(population, toy_gold())
    # g1 was trained on; g3 is test, x is outside the gold set
    assert labelled["in_training"].tolist() == [True, False, False]


def labelled_reviews() -> pd.DataFrame:
    """Four reviews with known reply figures. a2 carries two labels; a2's latency is unknown."""
    specs = [
        # id, app, labels, replied, latency_h, stars, in_training
        ("a1", "a", {"login_access"}, True, 2.0, 1, False),
        ("a2", "a", {"login_access", "support_experience"}, True, None, 2, False),
        ("a3", "a", {"feature_gap"}, False, None, 3, True),
        ("b1", "b", {"login_access"}, False, None, 1, False),
    ]
    return pd.DataFrame(
        [
            {
                "review_id": rid,
                "app_key": app,
                "has_reply": replied,
                "reply_latency_hours": latency,
                "score": stars,
                "in_training": training,
                **{name: int(name in labels) for name in LABEL_NAMES},
            }
            for rid, app, labels, replied, latency, stars, training in specs
        ]
    )


def cell(app: str, label: str) -> pd.Series:
    table = reply_table(labelled_reviews())
    return table[(table["app_key"] == app) & (table["label"] == label)].iloc[0]


def test_reply_figures_for_one_app_and_label():
    row = cell("a", "login_access")
    assert row["n"] == 2
    assert row["reply_rate"] == 1.0
    assert row["share"] == 2 / 3
    assert row["share_1_star"] == 0.5  # a1 is 1★, a2 is 2★


def test_latency_ignores_unknown_values():
    row = cell("a", "login_access")
    assert row["median_latency_h"] == 2.0
    assert row["n_latency"] == 1


def test_review_with_two_labels_counts_in_both():
    assert cell("a", "support_experience")["n"] == 1
    assert cell("a", "login_access")["n"] == 2


def test_all_apps_row_pools_every_app():
    row = cell(ALL_APPS, "login_access")
    assert row["n"] == 3
    assert row["reply_rate"] == 2 / 3


def test_all_reviews_row_is_the_app_reference():
    row = cell("a", ALL_REVIEWS)
    assert row["n"] == 3
    assert row["share"] == 1.0


def test_held_out_figures_leave_out_training_reviews():
    row = cell("a", "feature_gap")
    assert row["n"] == 1
    assert row["n_held_out"] == 0
    assert pd.isna(row["reply_rate_held_out"])


def test_empty_cell_has_no_rate_and_is_flagged_small():
    row = cell("a", "fees_pricing")
    assert row["n"] == 0
    assert pd.isna(row["reply_rate"])
    assert row["small"]
