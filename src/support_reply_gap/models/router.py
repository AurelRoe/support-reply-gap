"""C6 router: the baseline labels every review; uncertain ones go to the LLM.

    python -m support_reply_gap.models.router   # cost/accuracy curve on validation, $0

Uncertainty is the baseline's margin: how close its closest-call label probability
sits to the decision threshold. A review with any label near the threshold is one
where a small change would flip the baseline's answer.

The curve is computed from the LLM's cached answers, so it costs nothing. The
operating point is picked by a rule fixed before the curve was first drawn.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from support_reply_gap.eval.metrics import macro_f1
from support_reply_gap.labeling.gold import ROOT
from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.baseline import CHOSEN, fit_chosen, predict_proba, to_labels

# Escalate a review when its margin is below this. Picked in C6 on validation by the
# rule below: 85% of validation reviews escalated, val macro-F1 0.662 (LLM 0.673).
ROUTER_MARGIN = 0.13982597909800396

# Fixed before the curve was drawn: the smallest share of reviews sent to the LLM
# that reaches at least this fraction of the LLM's gain over the baseline.
GAIN_TARGET = 0.90
# Escalation shares tried on validation; each becomes a margin cut-off.
SHARES = [i / 20 for i in range(21)]

CURVE_PATH = ROOT / "reports/router_curve.png"


def margins(proba: pd.DataFrame, threshold: float = CHOSEN["threshold"]) -> np.ndarray:
    """Per review: distance of the closest label probability to the threshold."""
    return (proba[list(LABEL_NAMES)] - threshold).abs().min(axis=1).to_numpy()


def combine(base: pd.DataFrame, llm: pd.DataFrame, escalate: np.ndarray) -> pd.DataFrame:
    """Baseline labels, with the escalated rows replaced by the LLM's."""
    out = base.copy()
    out.loc[escalate] = llm.loc[escalate].to_numpy()
    return out


def sweep(
    y_true: pd.DataFrame, base: pd.DataFrame, llm: pd.DataFrame, margin: np.ndarray
) -> pd.DataFrame:
    """Macro-F1 at each escalation share; the cut-off is the margin quantile."""
    rows = []
    for share in SHARES:
        cut = np.inf if share >= 1 else float(np.quantile(margin, share)) if share > 0 else -np.inf
        escalate = margin < cut
        rows.append(
            {
                "target_share": share,
                "margin_cut": cut,
                "share_escalated": float(escalate.mean()),
                "macro_f1": macro_f1(y_true, combine(base, llm, escalate)),
            }
        )
    return pd.DataFrame(rows)


def operating_point(curve: pd.DataFrame, base_f1: float, llm_f1: float) -> pd.Series:
    """The smallest escalation share reaching GAIN_TARGET of the LLM's gain."""
    needed = base_f1 + GAIN_TARGET * (llm_f1 - base_f1)
    reaching = curve[curve["macro_f1"] >= needed]
    return reaching.sort_values("share_escalated").iloc[0]


class RouterClassifier:
    """Baseline first; the LLM only for reviews whose margin is below the cut-off."""

    name = "router"

    def __init__(self, texts: pd.Series, y: pd.DataFrame, llm, margin_cut: float) -> None:
        self.model = fit_chosen(texts, y)
        self.llm = llm
        self.margin_cut = margin_cut
        self.escalated = 0
        self.total = 0
        self.last_latency_s: float | None = None

    @property
    def config(self) -> dict:
        return {
            **CHOSEN,
            "margin_cut": self.margin_cut,
            "llm": self.llm.config,
            "escalated": self.escalated,
            "reviews": self.total,
        }

    def classify(self, text: str) -> tuple[set[str], float]:
        start = time.perf_counter()
        proba = predict_proba(self.model, pd.Series([text]))
        elapsed = time.perf_counter() - start
        self.total += 1
        if margins(proba)[0] < self.margin_cut:
            self.escalated += 1
            labels, cost = self.llm.classify(text)
            # Baseline time plus the LLM call's own latency (the original one on a cache hit).
            self.last_latency_s = elapsed + (self.llm.last_latency_s or 0.0)
            return labels, cost
        pred = to_labels(proba, CHOSEN["threshold"])
        self.last_latency_s = elapsed
        return {n for n in LABEL_NAMES if pred.loc[0, n] == 1}, 0.0


def plot_curve(
    curve: pd.DataFrame, base_f1: float, llm_f1: float, chosen: pd.Series, path: Path
) -> None:
    """Validation macro-F1 against the share of reviews sent to the LLM."""
    from matplotlib.figure import Figure

    surface, text, muted, grid, series = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    fig = Figure(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(surface)
    ax = fig.subplots()
    ax.set_facecolor(surface)

    share = curve["share_escalated"] * 100
    ax.plot(share, curve["macro_f1"], color=series, linewidth=2, marker="o", markersize=4, zorder=3)
    for value, name in ((base_f1, "baseline alone"), (llm_f1, "LLM alone")):
        ax.axhline(value, color=muted, linewidth=1, linestyle="--", zorder=1)
        ax.text(101, value, f"{name} {value:.2f}", color=muted, fontsize=8, va="center")
    x, y = chosen["share_escalated"] * 100, chosen["macro_f1"]
    ax.scatter([x], [y], s=90, facecolor=surface, edgecolor=series, linewidth=2, zorder=4)
    ax.annotate(
        f"chosen: {x:.0f}% to the LLM, {y:.2f}",
        (x, y),
        xytext=(10, -22),
        textcoords="offset points",
        fontsize=8,
        color=text,
    )

    ax.set_xlim(-3, 100)
    ax.set_xlabel(
        "Share of reviews sent to the LLM (%), proportional to cost", color=muted, fontsize=9
    )
    ax.set_ylabel("Validation macro-F1", color=muted, fontsize=9)
    ax.tick_params(colors=muted, labelsize=8)
    ax.grid(color=grid, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    fig.suptitle("Router: accuracy against cost", x=0.02, ha="left", fontsize=12, color=text)
    fig.text(
        0.02,
        0.9,
        f"Validation, 60 reviews · reviews escalated in order of baseline uncertainty · "
        f"rule: smallest share reaching {GAIN_TARGET:.0%} of the LLM's gain",
        fontsize=7.5,
        color=muted,
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.9))
    fig.savefig(path, facecolor=surface)


def main() -> None:
    from support_reply_gap.models.data import load_gold, xy
    from support_reply_gap.models.llm import LLMClassifier

    gold = load_gold()
    model = fit_chosen(*xy(gold, "train"))
    texts, y_val = xy(gold, "val")
    proba = predict_proba(model, texts)
    base = to_labels(proba, CHOSEN["threshold"])
    # max_usd=0: answers must come from the cache; any API call is refused.
    llm = LLMClassifier(max_usd=0.0)
    llm_sets = [llm.classify(t)[0] for t in texts]
    llm_pred = pd.DataFrame([{n: int(n in s) for n in LABEL_NAMES} for s in llm_sets])

    curve = sweep(y_val, base, llm_pred, margins(proba))
    base_f1, llm_f1 = macro_f1(y_val, base), macro_f1(y_val, llm_pred)
    chosen = operating_point(curve, base_f1, llm_f1)
    print(curve.to_string(index=False, float_format="%.3f"))
    print(f"\nbaseline {base_f1:.3f}, LLM {llm_f1:.3f}; spent ${llm.spent_usd:.4f}")
    print(
        f"chosen: margin cut {chosen['margin_cut']:.4f}, {chosen['share_escalated']:.0%} "
        f"escalated, val macro-F1 {chosen['macro_f1']:.3f}"
    )
    plot_curve(curve, base_f1, llm_f1, chosen, CURVE_PATH)
    print(f"wrote {CURVE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
