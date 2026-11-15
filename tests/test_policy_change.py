"""Tests for the D1b series. Hand-made reviews; no database."""

import math

import pandas as pd

from support_reply_gap.analysis.policy_change import (
    ALL,
    WITHOUT_EDITED,
    monthly,
    period_table,
    spans,
    summarise,
    with_month,
)


def reviews(rows) -> pd.DataFrame:
    """rows: (app, posted_at as text, score, replied, edited)."""
    return pd.DataFrame(
        [
            {
                "app_key": app,
                "posted_at": pd.Timestamp(posted, tz="UTC"),
                "score": score,
                "has_reply": replied,
                "likely_edited": edited,
            }
            for app, posted, score, replied, edited in rows
        ]
    )


def test_months_outside_the_window_are_dropped():
    kept = with_month(
        reviews(
            [
                ("c24", "2024-10-31 23:00", 1, False, False),
                ("c24", "2024-11-03 10:00", 1, False, False),
                ("c24", "2026-10-31 10:00", 1, False, False),
                ("c24", "2026-11-02 10:00", 1, False, False),  # incomplete last month
            ]
        )
    )
    assert [str(m) for m in kept["month"]] == ["2024-11", "2026-10"]


def test_month_is_taken_in_utc():
    berlin = pd.DataFrame(
        {"posted_at": [pd.Timestamp("2025-03-01 00:30", tz="Europe/Berlin")]}
    )  # still February in UTC
    assert str(with_month(berlin)["month"].iloc[0]) == "2025-02"


def test_summary_figures():
    group = reviews(
        [
            ("c24", "2025-01-01", 1, True, False),
            ("c24", "2025-01-02", 5, False, False),
        ]
    )
    s = summarise(group)
    assert s["n"] == 2
    assert s["mean_score"] == 3.0
    assert s["share_1_star"] == 0.5
    assert s["reply_rate"] == 0.5
    assert s["low"] < 3.0 < s["high"]


def test_single_review_has_no_interval():
    s = summarise(reviews([("c24", "2025-01-01", 4, False, False)]))
    assert s["mean_score"] == 4.0
    assert math.isnan(s["low"])


def test_edited_reviews_are_left_out_only_in_the_second_version():
    frame = with_month(
        reviews(
            [
                ("c24", "2025-01-05", 1, True, False),
                ("c24", "2025-01-06", 5, True, True),  # edited after the reply
            ]
        )
    )
    table = monthly(frame).set_index("version")
    assert table.loc[ALL, "n"] == 2
    assert table.loc[WITHOUT_EDITED, "n"] == 1
    assert table.loc[WITHOUT_EDITED, "mean_score"] == 1.0


def test_n26_spans_follow_the_events():
    assert [(name, str(a), str(b)) for name, a, b in spans("n26")] == [
        ("before", "2024-11", "2024-11"),
        ("replying", "2024-12", "2025-01"),
        ("paused", "2025-02", "2025-04"),
        ("replying again", "2025-05", "2026-10"),
    ]


def test_app_without_events_is_one_span():
    assert [name for name, _, _ in spans("revolut")] == ["before"]


def test_period_table_puts_revolut_beside_each_span():
    frame = with_month(
        reviews(
            [
                ("c24", "2025-06-01", 1, False, False),
                ("c24", "2026-03-01", 5, True, False),
                ("revolut", "2025-06-01", 4, False, False),
                ("revolut", "2026-03-01", 2, False, False),
            ]
        )
    )
    table = period_table(frame)
    c24 = table[(table["app_key"] == "c24") & (table["version"] == ALL)].set_index("span")
    assert c24.loc["before", "mean_score"] == 1.0
    assert c24.loc["replying", "mean_score"] == 5.0
    assert c24.loc["before", "ref_mean_score"] == 4.0
    assert c24.loc["replying", "ref_mean_score"] == 2.0


def test_empty_span_has_no_figures():
    frame = with_month(reviews([("c24", "2026-03-01", 5, True, False)]))
    table = period_table(frame)
    row = table[(table["app_key"] == "c24") & (table["span"] == "before")].iloc[0]
    assert row["n"] == 0
    assert math.isnan(row["mean_score"])
