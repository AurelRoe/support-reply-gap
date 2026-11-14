"""Tests for the D1 report text. Hand-made table rows; no database, no API calls."""

import pandas as pd

from support_reply_gap.analysis.d1_report import (
    co_mention_table,
    findings_text,
    fmt_difference,
    length_table,
)


def cell(app="c24", label="login_access", n=40, diff=0.12, low=0.03, high=0.20, finding=True):
    return {
        "app_key": app,
        "label": label,
        "n": n,
        "observed": 0.8,
        "expected": 0.8 - diff,
        "difference": diff,
        "low": low,
        "high": high,
        "finding": finding,
    }


def test_difference_shows_interval_in_points():
    assert fmt_difference(pd.Series(cell())) == "+12 (+3 to +20)"


def test_small_cell_difference_is_italic_without_interval():
    assert fmt_difference(pd.Series(cell(n=12))) == "*+12*"


def test_findings_list_the_finding_and_count_the_nulls():
    table = pd.DataFrame(
        [
            cell(),
            cell(label="feature_gap", diff=-0.02, low=-0.08, high=0.04, finding=False),
            cell(label="fees_pricing", n=10, finding=False),
        ]
    )
    text = findings_text(table)
    assert "C24 Bank · `login_access`: 80% replied against 68% expected, +12 points" in text
    assert "No difference detected in the other 1 of 2 cells" in text
    assert "1 cells have fewer than 30 reviews" in text


def test_no_finding_is_said_plainly():
    table = pd.DataFrame([cell(finding=False, low=-0.01)])
    assert findings_text(table).startswith("- No cell meets the rule.")


def test_n26_cells_are_left_out_of_the_findings():
    table = pd.DataFrame([cell(app="n26", finding=False)])
    assert "of 0 cells" in findings_text(table)


def reviews(rows):
    """rows: (app, chars, score, replied, support_experience, account_restricted)."""
    from support_reply_gap.labeling.taxonomy import LABEL_NAMES

    return pd.DataFrame(
        [
            {
                "app_key": app,
                "content": "x" * chars,
                "score": score,
                "has_reply": replied,
                **dict.fromkeys(LABEL_NAMES, 0),
                "support_experience": support,
                "account_restricted": restricted,
            }
            for app, chars, score, replied, support, restricted in rows
        ]
    )


def test_length_table_puts_reviews_in_their_band():
    table = length_table(reviews([("c24", 10, 1, False, 0, 0), ("c24", 300, 1, True, 0, 0)]))
    lines = table.splitlines()
    assert lines[2] == "| under 40 | *0% vs. 50% (1)* |"
    assert lines[5] == "| 250 and more | *100% vs. 50% (1)* |"


def test_co_mention_splits_a_label_by_support_experience():
    table = co_mention_table(
        reviews([("c24", 50, 1, True, 1, 1), ("c24", 50, 1, False, 0, 1)]), ["c24"]
    )
    row = next(line for line in table.splitlines() if "`account_restricted`" in line)
    assert row == "| `account_restricted` | 50% of 2 | *100% (1)* | *0% (1)* |"
