"""D1b: ratings around the reply-policy changes, as monthly series and per-period figures.

Descriptive only. The companies chose when to start replying, and other things
changed at the same time, so nothing here says what replying does to ratings.
Rules and event dates are fixed in docs/build-order.md (D1b).

Edited reviews: Google Play keeps only a review's latest version, so a review
edited after a reply carries the edit's date and score. Edits are detectable
only when a reply exists, so every figure is given with and without the
detectable ones, and the edited share itself is never an outcome.
"""

from __future__ import annotations

import math

import pandas as pd

from support_reply_gap.analysis.data import load_reviews
from support_reply_gap.apps import APPS

FIRST_MONTH = pd.Period("2024-11", "M")
# 2026-11 is left out: the scrape ended on 2026-11-03.
LAST_MONTH = pd.Period("2026-10", "M")

# Dated by the month replies were written, read from reply counts only (build-order D1b).
# Each entry starts a span; the span before the first entry is "before".
EVENTS: dict[str, list[tuple[str, str]]] = {
    "n26": [("2024-12", "replying"), ("2025-02", "paused"), ("2025-05", "replying again")],
    "c24": [("2026-01", "replying")],
}
REFERENCE_APP = "revolut"  # no change in the window; a reference series, not a control

ALL = "all reviews"
WITHOUT_EDITED = "without edited"


def with_month(reviews: pd.DataFrame) -> pd.DataFrame:
    """Add the review's month (UTC) and keep the months in the window."""
    # Periods carry no timezone; converting to UTC first makes the month boundary explicit.
    month = reviews["posted_at"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M")
    out = reviews.assign(month=month)
    return out[(out["month"] >= FIRST_MONTH) & (out["month"] <= LAST_MONTH)].reset_index(drop=True)


def load_monthly_input() -> pd.DataFrame:
    """All reviews of the three apps from Postgres, with their month, inside the window."""
    return with_month(load_reviews(apps=tuple(app.key for app in APPS)))


def versions(reviews: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The reviews as they are, and without those detectably edited after a reply."""
    return {ALL: reviews, WITHOUT_EDITED: reviews[~reviews["likely_edited"].astype(bool)]}


def summarise(group: pd.DataFrame) -> dict:
    """Count, mean score with a 95% interval, share of 1★ and reply rate."""
    n = len(group)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean_score": nan,
            "low": nan,
            "high": nan,
            "share_1_star": nan,
            "reply_rate": nan,
        }
    mean = group["score"].mean()
    # Normal approximation: with 50+ reviews a month it is close to a bootstrap, and simpler.
    half = 1.96 * group["score"].std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {
        "n": n,
        "mean_score": mean,
        "low": mean - half,
        "high": mean + half,
        "share_1_star": (group["score"] == 1).mean(),
        "reply_rate": group["has_reply"].mean(),
    }


def monthly(reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per version, app and month."""
    rows = []
    for version, frame in versions(reviews).items():
        for (app, month), group in frame.groupby(["app_key", "month"]):
            rows.append({"version": version, "app_key": app, "month": month, **summarise(group)})
    return pd.DataFrame(rows)


def spans(app: str) -> list[tuple[str, pd.Period, pd.Period]]:
    """(name, first month, last month) for each stretch between an app's events."""
    starts = [("before", FIRST_MONTH)] + [
        (name, pd.Period(month, "M")) for month, name in EVENTS.get(app, [])
    ]
    ends = [start - 1 for _, start in starts[1:]] + [LAST_MONTH]
    return [(name, start, end) for (name, start), end in zip(starts, ends, strict=True)]


def period_table(reviews: pd.DataFrame) -> pd.DataFrame:
    """Per app with events, version and span: the app's figures and Revolut's over the same months."""
    rows = []
    for version, frame in versions(reviews).items():
        for app in EVENTS:
            for name, start, end in spans(app):
                in_span = frame[(frame["month"] >= start) & (frame["month"] <= end)]
                own = summarise(in_span[in_span["app_key"] == app])
                ref = summarise(in_span[in_span["app_key"] == REFERENCE_APP])
                rows.append(
                    {
                        "version": version,
                        "app_key": app,
                        "span": name,
                        "start": start,
                        "end": end,
                        **own,
                        **{f"ref_{key}": value for key, value in ref.items()},
                    }
                )
    return pd.DataFrame(rows)
