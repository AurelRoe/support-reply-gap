"""D1 chart: observed minus star-adjusted expected reply rate, per label, with 95% intervals.

One panel per app rather than one shared axis, because the comparison is within an
app. N26 is left out: it replies to 99% of 1–3★ reviews, so its rate separates nothing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Figure instead of pyplot: renders straight to a file with no display backend.
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from support_reply_gap.analysis.reply_gap import LATENCY_ONLY_APPS, SMALL_CELL
from support_reply_gap.analysis.reply_gap_chart import (
    APP_STYLE,
    CHART_EXCLUDE,
    GRID,
    MUTED,
    SURFACE,
    TEXT,
)
from support_reply_gap.apps import by_key


def chart_rows(adjusted: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Apps to draw, and labels to draw, most common across those apps on top."""
    apps = [a for a in sorted(adjusted["app_key"].unique()) if a not in LATENCY_ONLY_APPS]
    shown = adjusted[adjusted["app_key"].isin(apps) & ~adjusted["label"].isin(CHART_EXCLUDE)]
    totals = shown.groupby("label")["n"].sum()
    labels = list(totals[totals > 0].sort_values(ascending=False).index)
    return apps, labels


def plot_star_adjusted(adjusted: pd.DataFrame, path: Path, subtitle: str = "") -> None:
    apps, labels = chart_rows(adjusted)
    row_of = {label: len(labels) - 1 - i for i, label in enumerate(labels)}
    large = adjusted[adjusted["label"].isin(labels) & (adjusted["n"] >= SMALL_CELL)]
    # Axis set by the intervals that are drawn (small cells have none), one side at a
    # time, so a single far-out cell does not squash the others. Right side keeps room for n.
    left = min(-10.0, large["low"].min() * 100 - 4) if len(large) else -10.0
    right = max(10.0, large["high"].max() * 100 + 4) + 12 if len(large) else 22.0

    fig = Figure(figsize=(9, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    axes = fig.subplots(1, len(apps), sharey=True, squeeze=False)[0]

    for ax, app in zip(axes, apps, strict=True):
        colour, marker = APP_STYLE[app]
        ax.set_facecolor(SURFACE)
        cells = adjusted[(adjusted["app_key"] == app) & adjusted["label"].isin(labels)]
        for _, r in cells[cells["n"] > 0].iterrows():
            y = row_of[r["label"]]
            small = r["n"] < SMALL_CELL
            if not small:
                ax.plot(
                    [r["low"] * 100, r["high"] * 100],
                    [y, y],
                    color=colour,
                    linewidth=2,
                    solid_capstyle="round",
                    zorder=2,
                )
            ax.scatter(
                max(left, min(right, r["difference"] * 100)),
                y,
                s=48,
                marker=marker,
                # hollow, no interval = fewer than 30 reviews: shown, not interpreted
                facecolor=SURFACE if small else colour,
                edgecolor=colour,
                linewidth=1.5,
                zorder=3,
            )
            ax.text(
                right, y, f"n={r['n']}", color=MUTED, fontsize=7, ha="right", va="center", zorder=4
            )

        ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
        ax.set_xlim(left, right)
        ax.set_ylim(-0.6, len(labels) - 0.4)
        ax.set_title(by_key(app).label, color=TEXT, fontsize=10, loc="left")
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].set_yticks(range(len(labels)), list(reversed(labels)), fontsize=9, color=TEXT)
    fig.supxlabel(
        "Reply rate minus the rate expected from the label's star mix (percentage points)",
        color=MUTED,
        fontsize=9,
        y=0.09,
    )
    handles = [
        Line2D([], [], color=MUTED, linewidth=2, marker="o", markersize=6, label="95% interval"),
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=7,
            color=MUTED,
            markerfacecolor=SURFACE,
            label=f"fewer than {SMALL_CELL} reviews: no interval, not interpreted",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Reply rate by complaint label, adjusted for star rating",
        x=0.02,
        ha="left",
        fontsize=12,
        color=TEXT,
    )
    if subtitle:
        fig.text(0.02, 0.925, subtitle, fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.savefig(path, facecolor=SURFACE)
