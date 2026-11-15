"""Tests for the D1b report text and chart helpers. No database."""

import pandas as pd

from support_reply_gap.analysis.policy_change_chart import paused_spans
from support_reply_gap.analysis.policy_change_report import events_text, fmt_mean, fmt_share


def test_mean_with_interval():
    row = pd.Series({"n": 10, "mean_score": 3.416, "low": 3.2, "high": 3.63})
    assert fmt_mean(row) == "3.42 (3.20–3.63)"


def test_empty_group_is_a_dash():
    assert fmt_mean(pd.Series({"n": 0})) == "–"
    assert fmt_share(float("nan"), 0) == "–"


def test_events_text_lists_every_event_and_the_reference():
    text = events_text()
    assert "N26: replying from 2024-12, paused from 2025-02, replying again from 2025-05." in text
    assert "C24 Bank: replying from 2026-01." in text
    assert "Revolut: no change" in text


def test_n26_pause_is_shaded_up_to_the_restart():
    assert paused_spans("n26") == [(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-05-01"))]
    assert paused_spans("c24") == []
