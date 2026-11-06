"""Self-agreement between two labelling passes over the same reviews.

Pass 1 (the gold labels) is the reference and pass 2 is scored against it with
the same macro-F1 the models get. That number is the accuracy ceiling. Pass 1
stays the gold label even where the passes disagree: adjudicating would change
the thing being measured.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from support_reply_gap.eval.metrics import exact_match, macro_f1, per_class_f1
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

# Below this many pass-1 examples, a per-label score on ~50 reviews is close to noise.
FEW_EXAMPLES = 5


def align(pass1: pd.DataFrame, pass2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict pass 1 to the relabelled reviews and put both in the same row order."""
    ids2 = set(pass2["review_id"])
    missing = ids2 - set(pass1["review_id"])
    if missing:
        raise ValueError(f"{len(missing)} pass-2 reviews have no pass-1 label")
    p1 = pass1[pass1["review_id"].isin(ids2)].set_index("review_id").sort_index()
    p2 = pass2.set_index("review_id").sort_index()
    return p1[list(LABEL_NAMES)], p2[list(LABEL_NAMES)]


def per_label_agreement(p1: pd.DataFrame, p2: pd.DataFrame) -> pd.DataFrame:
    """Per-label F1 and support, plus Cohen's kappa, which corrects for chance agreement."""
    table = per_class_f1(p1, p2)
    kappas = []
    for label in LABEL_NAMES:
        # Kappa is undefined when both passes give one label the same constant value.
        if p1[label].nunique() == 1 and p2[label].nunique() == 1:
            kappas.append(float("nan"))
        else:
            kappas.append(cohen_kappa_score(p1[label], p2[label]))
    table["kappa"] = kappas
    table["few_examples"] = table["support"] < FEW_EXAMPLES
    return table


def render_report(p1: pd.DataFrame, p2: pd.DataFrame, today: date | None = None) -> str:
    table = per_label_agreement(p1, p2)
    today = today or date.today()

    def fmt(x: float) -> str:
        return "n/a" if pd.isna(x) else f"{x:.2f}"

    lines = [
        "# Self-agreement",
        "",
        f"Generated {today.isoformat()} by `python -m support_reply_gap.labeling.gold agreement`.",
        "",
        (
            f"{len(p1)} gold reviews were labelled a second time, blind to the first pass. "
            "Pass 1 is the reference; pass 2 is scored against it with the same macro-F1 "
            "used for the models. This is the ceiling model scores are read against."
        ),
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Macro-F1 (pass 2 vs pass 1) | {fmt(macro_f1(p1, p2))} |",
        f"| Exact match (whole label set) | {fmt(exact_match(p1, p2))} |",
        "",
        "## Per label",
        "",
        (
            f"Labels with fewer than {FEW_EXAMPLES} pass-1 examples are marked; their scores "
            "rest on a handful of reviews and should not be read as a ceiling. "
            "n/a: the label does not occur in either pass, or kappa is undefined."
        ),
        "",
        "| Label | Pass 1 | Pass 2 | F1 | Kappa | |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, row in table.iterrows():
        flag = "few examples" if row["few_examples"] else ""
        lines.append(
            f"| `{label}` | {row['support']} | {row['predicted']} | {fmt(row['f1'])} "
            f"| {fmt(row['kappa'])} | {flag} |"
        )
    return "\n".join(lines) + "\n"
