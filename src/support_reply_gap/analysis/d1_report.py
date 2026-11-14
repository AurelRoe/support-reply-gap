"""D1 report: the C2 analysis redone with the LLM's labels and the star adjustment.

    python -m support_reply_gap.analysis.d1_report --estimate      # free: cost of labelling
    python -m support_reply_gap.analysis.d1_report                 # label if needed, write report

Writes reports/reply_gap_d1.md and reports/reply_gap_d1.png. Rules fixed before the
population was labelled are in docs/build-order.md (D1). LLM answers are cached
locally, so once the population is labelled a re-run costs nothing. The
interpretation lives in the README, where re-running cannot overwrite it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from support_reply_gap.analysis.c2_report import fmt_latency, fmt_rate, grid, markdown_table
from support_reply_gap.analysis.d1_chart import plot_star_adjusted
from support_reply_gap.analysis.reply_gap import (
    LATENCY_ONLY_APPS,
    MAX_STARS,
    SMALL_CELL,
    label_with_classifier,
    load_population,
    observed_and_expected,
    reply_table,
    star_adjusted,
)
from support_reply_gap.apps import by_key
from support_reply_gap.eval.report import latest_per_model, load_results
from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.llm import BudgetExceeded, LLMClassifier

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = ROOT / "reports/reply_gap_d1.md"
CHART_PATH = ROOT / "reports/reply_gap_d1.png"

# Exploratory section only, chosen after the findings were read: review length in
# characters. Under 40 is roughly one short sentence.
LENGTH_BINS = [0, 40, 100, 250, np.inf]
LENGTH_NAMES = ["under 40", "40–99", "100–249", "250 and more"]
# Labels that never co-occur with support_experience by definition, or are it.
NOT_CO_MENTIONED = {"support_experience", "complaint_unspecified", "no_complaint"}


def llm_validation() -> dict:
    """The LLM's committed validation scores, as shown in reports/eval.md."""
    llm = next(r for r in latest_per_model(load_results()) if r["model"] == "llm")
    return llm["splits"]["val"]


def fmt_pp(value: float) -> str:
    return f"{value * 100:+.0f}"


def fmt_difference(row: pd.Series) -> str:
    """'+12 (+3 to +20)'; italic and without interval for small cells."""
    if row["n"] == 0:
        return "–"
    if row["n"] < SMALL_CELL:
        return f"*{fmt_pp(row['difference'])}*"
    return f"{fmt_pp(row['difference'])} ({fmt_pp(row['low'])} to {fmt_pp(row['high'])})"


def app_table(adjusted: pd.DataFrame, app: str) -> str:
    """Observed, expected and their difference for one app, most common label first."""
    rows = adjusted[adjusted["app_key"] == app].sort_values("n", ascending=False)
    lines = [
        "| Label | Reviews | Observed | Expected | Difference, points (95% CI) | Finding |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for _, r in rows.iterrows():
        small = r["n"] < SMALL_CELL
        observed = "–" if r["n"] == 0 else f"{r['observed']:.0%}"
        expected = "–" if r["n"] == 0 else f"{r['expected']:.0%}"
        n = f"*{r['n']}*" if small else str(r["n"])
        mark = "yes" if r["finding"] else ""
        lines.append(
            f"| `{r['label']}` | {n} | {observed} | {expected} | {fmt_difference(r)} | {mark} |"
        )
    return "\n".join(lines)


def findings_text(adjusted: pd.DataFrame) -> str:
    """One bullet per cell meeting the rule, then a count of everything else, stated plainly."""
    rated = adjusted[~adjusted["app_key"].isin(LATENCY_ONLY_APPS)]
    found = rated[rated["finding"]].sort_values(["app_key", "difference"], ascending=[True, False])
    readable = rated[rated["n"] >= SMALL_CELL]
    bullets = [
        f"- {by_key(r['app_key']).label} · `{r['label']}`: {r['observed']:.0%} replied against "
        f"{r['expected']:.0%} expected, {fmt_pp(r['difference'])} points "
        f"({fmt_pp(r['low'])} to {fmt_pp(r['high'])}), {r['n']} reviews."
        for _, r in found.iterrows()
    ]
    if not bullets:
        bullets = ["- No cell meets the rule."]
    nulls = len(readable) - len(found)
    summary = (
        f"No difference detected in the other {nulls} of {len(readable)} cells with at "
        f"least {SMALL_CELL} reviews. {len(rated) - len(readable)} cells have fewer than "
        f"{SMALL_CELL} reviews and are not interpreted."
    )
    return "\n".join([*bullets, "", summary])


def quality_table(val: dict, labelled: pd.DataFrame) -> str:
    """Per label: how well the LLM predicts it, and how often it predicts it here."""
    lines = [
        "| Label | Val F1 | Val true / predicted | Labelled here |",
        "|---|---:|---:|---:|",
    ]
    for c in val["per_class"]:
        f1 = "–" if c["f1"] is None or math.isnan(c["f1"]) else f"{c['f1']:.2f}"
        f1 = f"*{f1}*" if c["few_examples"] else f1
        count = int(labelled[c["label"]].sum())
        lines.append(f"| `{c['label']}` | {f1} | {c['support']} / {c['predicted']} | {count} |")
    return "\n".join(lines)


def fmt_vs(observed: float, expected: float, n: int) -> str:
    """'31% vs. 73% (54)'; italic when the group is too small to read."""
    if n == 0:
        return "–"
    text = f"{observed:.0%} vs. {expected:.0%} ({n})"
    return f"*{text}*" if n < SMALL_CELL else text


def length_table(labelled: pd.DataFrame) -> str:
    """Per app and length band: reply rate against the rate expected from its star mix."""
    lengths = labelled["content"].str.len()
    band = pd.cut(lengths, LENGTH_BINS, labels=LENGTH_NAMES, right=False)
    apps = sorted(labelled["app_key"].unique())
    lines = [
        "| Length (characters) | " + " | ".join(by_key(a).label for a in apps) + " |",
        "|---|" + "---:|" * len(apps),
    ]
    for name in LENGTH_NAMES:
        cells = []
        for app in apps:
            in_app = (labelled["app_key"] == app).to_numpy()
            in_band = (band == name).to_numpy()[in_app]
            score = labelled["score"].to_numpy()[in_app]
            replied = labelled["has_reply"].to_numpy(dtype=float)[in_app]
            observed, expected = observed_and_expected(score, replied, in_band)
            cells.append(fmt_vs(observed, expected, int(in_band.sum())))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def co_mention_table(labelled: pd.DataFrame, apps: list[str]) -> str:
    """Per label: how often support is also complained about, and reply rates with and without."""
    header = ["Label", "Also `support_experience`"]
    for app in apps:
        header += [f"{by_key(app).label}: with", f"{by_key(app).label}: without"]
    lines = ["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)]
    for label in LABEL_NAMES:
        if label in NOT_CO_MENTIONED or not labelled[label].any():
            continue
        cell = labelled[labelled[label] == 1]
        row = [f"`{label}`", f"{cell['support_experience'].mean():.0%} of {len(cell)}"]
        for app in apps:
            in_app = cell[cell["app_key"] == app]
            for flag in (1, 0):
                group = in_app[in_app["support_experience"] == flag]
                n = len(group)
                row.append(
                    fmt_rate(group["has_reply"].mean() if n else float("nan"), n, n < SMALL_CELL)
                )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render(labelled: pd.DataFrame, table: pd.DataFrame, adjusted: pd.DataFrame, val: dict) -> str:
    first = labelled["posted_at"].min().date()
    last = labelled["posted_at"].max().date()
    low, high = val["macro_f1_ci95"]
    rates = grid(table, lambda r: fmt_rate(r["reply_rate"], r["n"], r["small"]))
    latency = grid(table, lambda r: fmt_latency(r["median_latency_h"], r["n_latency"]))
    by_star = labelled.groupby("score")["has_reply"].mean()
    rated_apps = [a for a in sorted(adjusted["app_key"].unique()) if a not in LATENCY_ONLY_APPS]
    n26 = table[(table["app_key"] == "n26") & (table["label"] == "(all reviews)")].iloc[0]
    app_sections = "\n\n".join(
        f"### {by_key(app).label}\n\n{app_table(adjusted, app)}" for app in rated_apps
    )

    return f"""# Reply gap: final analysis (stage D1)

Generated by `python -m support_reply_gap.analysis.d1_report`. Do not edit by hand.

Descriptive only. Which reviews receive a reply is each company's choice, and that
choice is unobserved. Nothing here says why a complaint type is or is not answered.

**Population.** {len(labelled):,} Google Play reviews rated 1–{MAX_STARS}★, posted
{first} to {last}, the same population as C2.

**Labels.** Predicted by the LLM path (prompt `classify_v1.md`), validation
macro-F1 {val["macro_f1"]:.2f} ({low:.2f}–{high:.2f}) against a human self-agreement
ceiling of 0.69. Not trained on any review. A review with several labels counts once
in each.

**Rules**, fixed in `docs/build-order.md` before the population was labelled:
expected reply rate = the app's reply rate per star rating, weighted by the label's own
mix of 1★, 2★ and 3★. A cell is a finding when it has at least {SMALL_CELL} reviews and
the 95% bootstrap interval of observed minus expected excludes 0. N26 is compared on
latency only. Apps are never pooled.

## Findings

{findings_text(adjusted)}

N26 replies to {n26["reply_rate"]:.0%} of its {n26["n"]} reviews, so its reply rate cannot
separate labels: no difference can be detected there by design.

![Reply rate by complaint label, adjusted for star rating]({CHART_PATH.name})

## Observed against expected reply rate

{app_sections}

## Reply rate

Share of reviews with a reply, number of reviews in brackets, not adjusted for star
rating. *Italic*: fewer than {SMALL_CELL} reviews. 1★ reviews receive a reply more often
({by_star[1]:.0%}) than 2★ ({by_star[2]:.0%}) and 3★ ({by_star[3]:.0%}), which is what
the adjustment above accounts for.

{markdown_table(rates)}

## Median reply latency

Among replied reviews whose latency is known (edited reviews have none), count in
brackets. *Italic*: fewer than {SMALL_CELL}.

{markdown_table(latency)}

## Label quality

How far each label's counts can be trusted. Validation F1 and true / predicted counts
from the LLM's validation run (60 reviews); *italic*: fewer than 5 true examples.
A label predicted much more or less often than it truly occurs on validation is
probably over- or under-counted here as well.

{quality_table(val, labelled)}

## Exploratory: outside the pre-committed rule

Both tables were computed after the findings above had been read, so the rule does
not cover them: no intervals, several looks at the same data. They are leads for
further work, not findings.

**Review length.** Reply rate against the rate expected from the band's star mix,
number of reviews in brackets. *Italic*: fewer than {SMALL_CELL}.

{length_table(labelled)}

**Co-occurrence with `support_experience`.** The share of each label's reviews (all
apps) that also complain about support, then the reply rate of the label's reviews
with and without that second label. *Italic*: fewer than {SMALL_CELL}.

{co_mention_table(labelled, rated_apps)}

## Caveats

- Labels are model predictions at validation macro-F1 {val["macro_f1"]:.2f}, not human
  labels. Validation holds 60 reviews, so per-label F1 is itself uncertain.
- A label's errors move reviews between cells. A misread review lands in a cell with
  its own reply status, which pulls that cell's rate toward the app's.
- The star adjustment covers star rating only. Anything else that differs between
  labels and goes with replying, such as review length or timing, is not adjusted for.
- The expected rate includes the label's own reviews, which pulls it slightly toward
  the observed rate and makes a finding slightly harder to reach.
- 4–5★ reviews are left out: the gold set holds 21 5★ reviews, too few to know how
  accurate labels on them are.
"""


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m support_reply_gap.analysis.d1_report")
    parser.add_argument("--estimate", action="store_true", help="print the cost, spend nothing")
    parser.add_argument("--max-usd", type=float, default=0.20, help="hard spending cap per run")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()

    llm = LLMClassifier(max_usd=args.max_usd)
    population = load_population()
    est = llm.estimate_usd(population["content"])
    if est["uncached"] == 0:
        print(f"all {len(population)} answers are cached: this run costs $0")
    else:
        print(
            f"{est['uncached']} of {len(population)} reviews need an API call: expected "
            f"${est['expected_usd']:.3f}, worst case ${est['worst_case_usd']:.3f}, "
            f"hard cap ${llm.max_usd:.2f}"
        )
    if args.estimate:
        return
    if est["uncached"] and not args.yes and input("continue? [y/N] ").strip().lower() != "y":
        print("cancelled, nothing spent")
        return

    try:
        labelled = label_with_classifier(population, llm)
    except BudgetExceeded as err:
        raise SystemExit(f"stopped: {err}. Answers so far are cached; re-run to continue.")
    if llm.failures:
        # Failed answers count as no labels and are not cached; a re-run retries them.
        raise SystemExit(f"{llm.failures} reviews got no answer; re-run before reporting.")

    val = llm_validation()
    adjusted = star_adjusted(labelled)
    subtitle = (
        f"1–{MAX_STARS}★ Google Play reviews · labels predicted by the LLM "
        f"(validation macro-F1 {val['macro_f1']:.2f}) · descriptive only"
    )
    plot_star_adjusted(adjusted, CHART_PATH, subtitle)
    REPORT_PATH.write_text(render(labelled, reply_table(labelled), adjusted, val))
    print(f"wrote {REPORT_PATH.relative_to(ROOT)} and {CHART_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
