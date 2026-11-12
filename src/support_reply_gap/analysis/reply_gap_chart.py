"""C2 chart: reply rate per complaint label, relative to each app's own reply rate.

Plotted as a gap to the app's average because the apps' overall reply rates differ
far more (63% to 99%) than labels differ within an app. Raw rates would mostly show
which app a dot belongs to.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Figure instead of pyplot: it renders straight to a file with no display backend,
# so the same code runs on a Mac and headless in CI.
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from support_reply_gap.analysis.reply_gap import ALL_APPS, ALL_REVIEWS
from support_reply_gap.apps import by_key

# Colour plus marker shape per app, so identity never rests on colour alone.
# Colours are the dataviz reference palette's first three slots, validated as a set.
APP_STYLE = {
    "c24": ("#2a78d6", "o"),
    "n26": ("#eb6834", "s"),
    "revolut": ("#1baf7a", "^"),
}
# Vertical offsets so the three apps' dots on one label row don't cover each other.
APP_OFFSET = {"c24": 0.22, "n26": 0.0, "revolut": -0.22}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# not a complaint type: it stays in the table, but has no place on a complaint chart
CHART_EXCLUDE = {"no_complaint"}


def gap_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Per app and label: reply rate minus the app's overall reply rate, in points."""
    apps = table[table["app_key"] != ALL_APPS]
    baseline = apps[apps["label"] == ALL_REVIEWS].set_index("app_key")["reply_rate"]
    rows = apps[
        (apps["label"] != ALL_REVIEWS) & ~apps["label"].isin(CHART_EXCLUDE) & (apps["n"] > 0)
    ].copy()
    rows["gap_pp"] = (rows["reply_rate"] - rows["app_key"].map(baseline)) * 100
    return rows.reset_index(drop=True)


def plot_reply_gap(table: pd.DataFrame, path: Path, subtitle: str = "") -> None:
    """Dot plot, one row per label (most common on top), one dot per app."""
    gaps = gap_frame(table)
    pooled_n = table[table["app_key"] == ALL_APPS].set_index("label")["n"]
    labels = [
        lbl for lbl in pooled_n.sort_values(ascending=False).index if lbl in set(gaps["label"])
    ]
    row_of = {label: len(labels) - 1 - i for i, label in enumerate(labels)}

    fig = Figure(figsize=(8, 6.5), dpi=150)
    ax = fig.subplots()
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for _, r in gaps.iterrows():
        colour, marker = APP_STYLE[r["app_key"]]
        ax.scatter(
            r["gap_pp"],
            row_of[r["label"]] + APP_OFFSET[r["app_key"]],
            s=46,
            marker=marker,
            # hollow = fewer than 30 reviews: shown, but not to be read as a finding
            facecolor=SURFACE if r["small"] else colour,
            edgecolor=colour,
            linewidth=1.5,
            zorder=3,
        )

    ax.axvline(0, color=MUTED, linewidth=1, zorder=2)
    ax.text(1, len(labels) - 0.45, "app's own reply rate", color=MUTED, fontsize=8, va="bottom")

    limit = max(10.0, gaps["gap_pp"].abs().max() + 5)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_yticks(range(len(labels)), [lbl for lbl in reversed(labels)], fontsize=9, color=TEXT)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}" if v else "0")
    ax.set_xlabel(
        "Reply rate minus the app's overall reply rate (percentage points)", color=MUTED, fontsize=9
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

    baseline = table[(table["label"] == ALL_REVIEWS) & (table["app_key"] != ALL_APPS)]
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=APP_STYLE[r["app_key"]][1],
            markersize=7,
            color=APP_STYLE[r["app_key"]][0],
            label=f"{by_key(r['app_key']).label}  ({r['reply_rate']:.0%} overall, n={r['n']})",
        )
        for _, r in baseline.iterrows()
    ]
    handles.append(
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=7,
            color=MUTED,
            markerfacecolor=SURFACE,
            label="fewer than 30 reviews",
        )
    )
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
        "Reply rate by complaint label, relative to each app's average",
        x=0.02,
        ha="left",
        fontsize=12,
        color=TEXT,
    )
    if subtitle:
        fig.text(0.02, 0.925, subtitle, fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.92))
    fig.savefig(path, facecolor=SURFACE)
