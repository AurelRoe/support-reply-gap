"""Tests for the C2 report formatting. No database."""

import pandas as pd

from support_reply_gap.analysis.c2_report import fmt_latency, fmt_rate, markdown_table


def test_rate_shows_count_and_marks_small_cells():
    assert fmt_rate(0.583, 219, small=False) == "58% (219)"
    assert fmt_rate(0.5, 8, small=True) == "*50% (8)*"


def test_empty_cell_is_a_dash():
    assert fmt_rate(float("nan"), 0, small=True) == "–"
    assert fmt_latency(float("nan"), 0) == "–"


def test_latency_switches_units():
    assert fmt_latency(0.38, 300) == "23 min (300)"
    assert fmt_latency(3.83, 300) == "3.8 h (300)"
    assert fmt_latency(74.0, 300) == "3.1 d (300)"


def test_markdown_table_has_header_separator_and_rows():
    frame = pd.DataFrame({"C24": ["58% (219)"]}, index=["feature_gap"])
    lines = markdown_table(frame).splitlines()
    assert lines[0] == "| Label | C24 |"
    assert lines[1] == "|---|---:|"
    assert lines[2] == "| `feature_gap` | 58% (219) |"
