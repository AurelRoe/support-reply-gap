"""D1b chart: monthly series per app, with a line at each reply-policy event.

Small multiples: one column per app, one row per series. Rows share their y-axis
across apps so levels can be compared, except review volume, where Revolut's
several hundred a month would flatten the other two.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Figure instead of pyplot: renders straight to a file with no display backend.
from matplotlib.dates import DateFormatter, MonthLocator
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from support_reply_gap.analysis.policy_change import ALL, EVENTS, WITHOUT_EDITED
from support_reply_gap.analysis.reply_gap_chart import APP_STYLE, GRID, MUTED, SURFACE, TEXT
from support_reply_gap.apps import by_key

APP_ORDER = ["n26", "c24", "revolut"]  # the two apps with events first, reference last
ROWS = [
    ("reply_rate", "Reply rate", True),
    ("mean_score", "Mean score", True),
    ("share_1_star", "Share of 1★", True),
    ("n", "Reviews per month", False),
]
PERCENT_ROWS = {"reply_rate", "share_1_star"}


def month_start(months: pd.Series) -> pd.Series:
    return months.dt.to_timestamp()


def paused_spans(app: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Start and end of each span named 'paused', for shading."""
    events = EVENTS.get(app, [])
    out = []
    for i, (month, name) in enumerate(events):
        if name == "paused" and i + 1 < len(events):
            out.append(
                (
                    pd.Period(month, "M").to_timestamp(),
                    pd.Period(events[i + 1][0], "M").to_timestamp(),
                )
            )
    return out


def plot_policy_change(series: pd.DataFrame, path: Path, subtitle: str = "") -> None:
    fig = Figure(figsize=(11, 9.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes = fig.subplots(len(ROWS), len(APP_ORDER), sharex=True, squeeze=False)

    for row, (column, title, shared) in enumerate(ROWS):
        for col, app in enumerate(APP_ORDER):
            ax = axes[row][col]
            colour = APP_STYLE[app][0]
            ax.set_facecolor(SURFACE)
            for start, end in paused_spans(app):
                ax.axvspan(start, end, color=GRID, alpha=0.6, linewidth=0, zorder=0)
            for i, (month, name) in enumerate(EVENTS.get(app, [])):
                ax.axvline(
                    pd.Period(month, "M").to_timestamp(),
                    color=MUTED,
                    linewidth=1,
                    linestyle=":",
                    zorder=1,
                )
                if row == 0:
                    ax.text(
                        pd.Period(month, "M").to_timestamp(),
                        # Staggered: N26's events are two to three months apart.
                        1.03 + 0.09 * i,
                        f" {name}",
                        transform=ax.get_xaxis_transform(),
                        rotation=0,
                        fontsize=7,
                        color=MUTED,
                        ha="left",
                        va="bottom",
                    )

            for version, style in ((ALL, "-"), (WITHOUT_EDITED, "--")):
                data = series[(series["app_key"] == app) & (series["version"] == version)]
                x = month_start(data["month"])
                ax.plot(
                    x,
                    data[column],
                    color=colour,
                    linewidth=2 if version == ALL else 1.4,
                    linestyle=style,
                    zorder=3,
                )
                if column == "mean_score" and version == ALL:
                    ax.fill_between(
                        x,
                        data["low"],
                        data["high"],
                        color=colour,
                        alpha=0.15,
                        linewidth=0,
                        zorder=2,
                    )

            if row == 0:
                ax.set_title(by_key(app).label, color=TEXT, fontsize=10, loc="left", pad=30)
            if col == 0:
                ax.set_ylabel(title, color=TEXT, fontsize=9)
            if column in PERCENT_ROWS:
                ax.set_ylim(0, 1.02)
                ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
            ax.tick_params(colors=MUTED, labelsize=7)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)

        if shared:
            # Same scale across a row, so levels compare between apps.
            low = min(a.get_ylim()[0] for a in axes[row])
            high = max(a.get_ylim()[1] for a in axes[row])
            for a in axes[row]:
                a.set_ylim(low, high)
        else:
            for a in axes[row]:
                a.set_ylim(bottom=0)

    for ax in axes[-1]:
        ax.xaxis.set_major_locator(MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(DateFormatter("%b %y"))

    handles = [
        Line2D([], [], color=MUTED, linewidth=2, label="all reviews"),
        Line2D(
            [],
            [],
            color=MUTED,
            linewidth=1.4,
            linestyle="--",
            label="without reviews detectably edited after a reply",
        ),
        Patch(facecolor=MUTED, alpha=0.15, label="95% interval of the mean score"),
        Line2D([], [], color=MUTED, linewidth=1, linestyle=":", label="reply-policy event"),
        Patch(facecolor=GRID, alpha=0.6, label="replies paused"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Ratings around the reply-policy changes", x=0.02, ha="left", fontsize=12, color=TEXT
    )
    if subtitle:
        fig.text(0.02, 0.945, subtitle, fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(path, facecolor=SURFACE)
