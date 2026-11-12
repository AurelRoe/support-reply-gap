"""Tests for the C2 chart. A hand-made reply table; no database."""

import pandas as pd

from support_reply_gap.analysis.reply_gap import ALL_APPS, ALL_REVIEWS
from support_reply_gap.analysis.reply_gap_chart import gap_frame, plot_reply_gap


def toy_table() -> pd.DataFrame:
    rows = [
        # app, label, n, reply_rate
        ("c24", ALL_REVIEWS, 100, 0.60),
        ("c24", "login_access", 40, 0.80),
        ("c24", "fees_pricing", 5, 0.40),
        ("c24", "no_complaint", 3, 0.00),
        ("c24", "crash_stability", 0, float("nan")),
        ("revolut", ALL_REVIEWS, 200, 0.75),
        ("revolut", "login_access", 50, 0.70),
        (ALL_APPS, ALL_REVIEWS, 300, 0.70),
        (ALL_APPS, "login_access", 90, 0.74),
        (ALL_APPS, "fees_pricing", 5, 0.40),
    ]
    table = pd.DataFrame(rows, columns=["app_key", "label", "n", "reply_rate"])
    table["small"] = table["n"] < 30
    return table


def gap(app: str, label: str) -> float:
    g = gap_frame(toy_table())
    return g[(g["app_key"] == app) & (g["label"] == label)]["gap_pp"].iloc[0]


def test_gap_is_measured_against_the_apps_own_rate():
    assert round(gap("c24", "login_access"), 6) == 20.0
    assert round(gap("revolut", "login_access"), 6) == -5.0


def test_pooled_reference_and_empty_rows_are_not_plotted():
    g = gap_frame(toy_table())
    assert ALL_APPS not in set(g["app_key"])
    assert ALL_REVIEWS not in set(g["label"])
    assert "no_complaint" not in set(g["label"])
    assert "crash_stability" not in set(g["label"])  # n = 0, nothing to plot


def test_chart_is_written(tmp_path):
    path = tmp_path / "chart.png"
    plot_reply_gap(toy_table(), path)
    assert path.stat().st_size > 0
