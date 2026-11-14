"""Tests for the D1 additions: labelling with a classifier and the star adjustment.

Hand-made reviews and a fake classifier; no database, no API calls.
"""

import numpy as np
import pandas as pd
import pytest

from support_reply_gap.analysis.reply_gap import (
    SMALL_CELL,
    label_with_classifier,
    observed_and_expected,
    star_adjusted,
)
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


class FakeClassifier:
    """Labels a review `login_access` when it mentions a login, `feature_gap` otherwise."""

    def classify(self, text: str) -> tuple[set[str], float]:
        return ({"login_access"} if "Login" in text else {"feature_gap"}), 0.0


def test_classifier_labels_align_with_population_rows():
    population = pd.DataFrame({"review_id": ["a", "b"], "content": ["Login geht nicht", "Fehlt"]})
    population.index = [5, 9]  # a filtered frame keeps gaps in its index
    labelled = label_with_classifier(population, FakeClassifier())
    assert labelled["review_id"].tolist() == ["a", "b"]
    assert labelled["login_access"].tolist() == [1, 0]
    assert labelled["feature_gap"].tolist() == [0, 1]
    assert not labelled["in_training"].any()


def test_empty_population_still_has_label_columns():
    empty = pd.DataFrame({"review_id": [], "content": []})
    labelled = label_with_classifier(empty, FakeClassifier())
    assert set(LABEL_NAMES) <= set(labelled.columns)


def test_expected_rate_follows_the_cells_star_mix():
    # App: 1★ always answered, 3★ never. A cell made only of 1★ reviews is
    # expected at 100%, so its high reply rate is no difference at all.
    score = np.array([1, 1, 3, 3])
    replied = np.array([1.0, 1.0, 0.0, 0.0])
    observed, expected = observed_and_expected(score, replied, np.array([True, True, False, False]))
    assert (observed, expected) == (1.0, 1.0)


def test_expected_rate_for_a_mixed_cell():
    score = np.array([1, 1, 3, 3])
    replied = np.array([1.0, 0.0, 1.0, 1.0])  # app: 1★ at 50%, 3★ at 100%
    in_cell = np.array([True, False, True, True])  # one 1★, two 3★
    observed, expected = observed_and_expected(score, replied, in_cell)
    assert observed == 1.0
    assert expected == pytest.approx(1 / 3 * 0.5 + 2 / 3 * 1.0)


def test_empty_cell_has_no_rates():
    observed, expected = observed_and_expected(np.array([1]), np.array([1.0]), np.array([False]))
    assert np.isnan(observed) and np.isnan(expected)


def app_reviews(app: str, n_label: int, n_other: int) -> pd.DataFrame:
    """All 1★. Reviews labelled `login_access` are all answered, the others never."""
    n = n_label + n_other
    return pd.DataFrame(
        {
            "app_key": app,
            "score": 1,
            "has_reply": [True] * n_label + [False] * n_other,
            **{
                name: [int(name == "login_access")] * n_label + [0] * n_other
                for name in LABEL_NAMES
            },
        },
        index=range(n),
    )


def row(table: pd.DataFrame, app: str, label: str) -> pd.Series:
    return table[(table["app_key"] == app) & (table["label"] == label)].iloc[0]


def test_clear_difference_in_a_large_cell_is_a_finding():
    table = star_adjusted(app_reviews("c24", 40, 60), rounds=200)
    cell = row(table, "c24", "login_access")
    assert cell["n"] == 40
    assert cell["difference"] == pytest.approx(1.0 - 0.4)
    assert cell["low"] > 0
    assert cell["finding"]


def test_small_cell_is_never_a_finding():
    table = star_adjusted(app_reviews("c24", SMALL_CELL - 1, 60), rounds=200)
    cell = row(table, "c24", "login_access")
    assert cell["low"] > 0
    assert not cell["finding"]


def test_n26_is_never_a_finding_on_reply_rate():
    table = star_adjusted(app_reviews("n26", 40, 60), rounds=200)
    assert not row(table, "n26", "login_access")["finding"]


def test_apps_are_not_pooled():
    both = pd.concat([app_reviews("c24", 40, 60), app_reviews("revolut", 40, 60)])
    table = star_adjusted(both, rounds=50)
    assert set(table["app_key"]) == {"c24", "revolut"}
    assert len(table) == 2 * len(LABEL_NAMES)


def test_intervals_are_reproducible():
    reviews = app_reviews("c24", 40, 60)
    first = star_adjusted(reviews, rounds=100)
    second = star_adjusted(reviews, rounds=100)
    pd.testing.assert_frame_equal(first, second)
