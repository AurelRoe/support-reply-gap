"""Tests for the gold split. Dummy 0/1 frames with ids like "t1"; no text, no data/gold/."""

import pandas as pd

from support_reply_gap.labeling.split import assign_splits
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


def dummy_labels(n: int = 200) -> pd.DataFrame:
    # Cycle through the labels so each one appears; every 10th row gets one of the two
    # rarest labels, to check that rare labels still reach every split.
    rows = []
    for i in range(n):
        on = {LABEL_NAMES[i % 9]} if i % 10 else {LABEL_NAMES[9 + (i // 10) % 2]}
        rows.append({"review_id": f"t{i:03d}", **{name: int(name in on) for name in LABEL_NAMES}})
    return pd.DataFrame(rows)


def test_splits_cover_every_review_exactly_once():
    labels = dummy_labels()
    splits = assign_splits(labels)
    assert sorted(splits["review_id"]) == sorted(labels["review_id"])
    assert splits["review_id"].is_unique


def test_split_sizes_are_60_20_20():
    counts = assign_splits(dummy_labels())["split"].value_counts()
    assert (counts["train"], counts["val"], counts["test"]) == (120, 40, 40)


def test_split_is_deterministic_and_ignores_row_order():
    labels = dummy_labels()
    first = assign_splits(labels).set_index("review_id")["split"]
    shuffled = assign_splits(labels.sample(frac=1.0, random_state=1)).set_index("review_id")
    assert first.equals(shuffled["split"].loc[first.index])


def test_rare_labels_reach_every_split():
    labels = dummy_labels()
    merged = labels.merge(assign_splits(labels), on="review_id")
    per_split = merged.groupby("split")[["complaint_unspecified", "no_complaint"]].sum()
    assert (per_split > 0).all().all()


def test_proportional_subset_keeps_exact_size_and_group_shares():
    from support_reply_gap.labeling.sampling import proportional_subset

    df = pd.DataFrame(
        {
            "review_id": [f"t{i:03d}" for i in range(400)],
            "split": ["train"] * 243 + ["val"] * 81 + ["test"] * 76,
            "app_key": (["a", "b", "c"] * 134)[:400],
        }
    )
    kept = proportional_subset(df, ["split", "app_key"], 300)
    assert len(kept) == 300 and kept["review_id"].is_unique
    counts = kept["split"].value_counts()
    assert abs(counts["train"] - 243 * 0.75) <= 1.5
    assert abs(counts["test"] - 76 * 0.75) <= 1.5
    again = proportional_subset(df, ["split", "app_key"], 300)
    assert set(again["review_id"]) == set(kept["review_id"])
