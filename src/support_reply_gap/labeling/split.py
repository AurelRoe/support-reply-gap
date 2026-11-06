"""Train/val/test split of the gold set, stratified on the labels.

Iterative stratification (Sechidis et al., 2011) spreads every label across the
splits in proportion, including rare ones. A plain random split of 400 reviews
could leave a 9-example label with no test case at all.

60/20/20 rather than 70/15/15: the rare labels need the extra test support more
than the baseline needs the extra training rows.
"""

from __future__ import annotations

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from support_reply_gap.labeling.taxonomy import LABEL_NAMES

TEST_SHARE = 0.2
VAL_SHARE = 0.2
SPLIT_SEED = 20261106


def assign_splits(labels: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Return review_id and split (train / val / test) for every labelled review."""
    # Sorting first makes the result depend only on the labels, not the file's row order.
    labels = labels.sort_values("review_id").reset_index(drop=True)
    y = labels[list(LABEL_NAMES)].to_numpy()

    # Cut test first, then val out of what is left, so test never depends on val.
    outer = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=TEST_SHARE, random_state=seed)
    rest_idx, test_idx = next(outer.split(labels, y))

    inner = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SHARE / (1 - TEST_SHARE), random_state=seed
    )
    _train_pos, val_pos = next(inner.split(rest_idx, y[rest_idx]))

    split = pd.Series("train", index=labels.index)
    split.iloc[test_idx] = "test"
    split.iloc[rest_idx[val_pos]] = "val"
    return pd.DataFrame({"review_id": labels["review_id"], "split": split})


def split_summary(labels: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Label counts per split, so thin test support is visible rather than hidden."""
    merged = labels.merge(splits, on="review_id")
    table = merged.groupby("split")[list(LABEL_NAMES)].sum().T[["train", "val", "test"]]
    table.loc["(reviews)"] = merged["split"].value_counts()[["train", "val", "test"]]
    return table
