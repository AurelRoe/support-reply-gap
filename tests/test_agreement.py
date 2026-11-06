"""Tests for the shared metrics and the self-agreement report.

Inputs are dummy 0/1 frames with ids like "t1" and no text. They test the
arithmetic and never touch data/gold/.
"""

import math

import pandas as pd
import pytest

from support_reply_gap.eval.metrics import exact_match, macro_f1, per_class_f1
from support_reply_gap.labeling.agreement import align, per_label_agreement, render_report
from support_reply_gap.labeling.gold import pick_relabel_ids
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


def frame(rows: dict[str, list[str]]) -> pd.DataFrame:
    """review_id -> labels that are 1; every other label is 0."""
    return pd.DataFrame(
        [{"review_id": rid, **{n: int(n in on) for n in LABEL_NAMES}} for rid, on in rows.items()]
    )


ROWS = {
    "t1": ["crash_stability", "support_experience"],
    "t2": ["no_complaint"],
    "t3": ["fees_pricing"],
}


def test_identical_passes_score_one():
    p1, p2 = align(frame(ROWS), frame(ROWS))
    assert macro_f1(p1, p2) == 1.0
    assert exact_match(p1, p2) == 1.0


def test_labels_absent_from_both_passes_are_left_out_of_the_mean():
    p1, p2 = align(frame(ROWS), frame(ROWS))
    table = per_class_f1(p1, p2)
    assert math.isnan(table.loc["performance", "f1"])
    assert table["f1"].notna().sum() == 4


def test_one_disagreement():
    changed = {**ROWS, "t3": ["payments_transfers"]}
    p1, p2 = align(frame(ROWS), frame(changed))
    table = per_class_f1(p1, p2)
    assert table.loc["fees_pricing", "f1"] == 0.0
    assert table.loc["payments_transfers", "f1"] == 0.0
    # Five labels occur; three agree perfectly, two score 0.
    assert macro_f1(p1, p2) == pytest.approx(3 / 5)
    assert exact_match(p1, p2) == pytest.approx(2 / 3)


def test_align_uses_only_relabelled_reviews_in_matching_order():
    pass2 = frame({"t3": ["fees_pricing"], "t1": ROWS["t1"]})
    p1, p2 = align(frame(ROWS), pass2)
    assert list(p1.index) == list(p2.index) == ["t1", "t3"]


def test_align_rejects_reviews_missing_from_pass_one():
    with pytest.raises(ValueError, match="no pass-1 label"):
        align(frame(ROWS), frame({"t9": ["no_complaint"]}))


def test_kappa_is_nan_when_a_label_never_varies():
    p1, p2 = align(frame(ROWS), frame(ROWS))
    assert math.isnan(per_label_agreement(p1, p2).loc["performance", "kappa"])


def test_report_lists_every_label():
    p1, p2 = align(frame(ROWS), frame(ROWS))
    report = render_report(p1, p2)
    assert all(f"`{name}`" in report for name in LABEL_NAMES)


def test_relabel_pick_is_deterministic_and_order_independent():
    ids = [f"t{i}" for i in range(400)]
    picked = pick_relabel_ids(ids)
    assert len(set(picked)) == 50
    assert picked == pick_relabel_ids(list(reversed(ids)))
