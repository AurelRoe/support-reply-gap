"""Tests for joining gold labels to review text. Dummy ids and text; no database."""

import pandas as pd
import pytest

from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.data import join_gold, xy


def frames(n: int = 4):
    labels = pd.DataFrame(
        [
            {"review_id": f"r{i}", **{name: int(j == i % 3) for j, name in enumerate(LABEL_NAMES)}}
            for i in range(n)
        ]
    )
    splits = pd.DataFrame(
        {"review_id": labels["review_id"], "split": ["train", "train", "val", "test"][:n]}
    )
    texts = pd.DataFrame(
        {"review_id": labels["review_id"], "content": [f"Text {i}" for i in range(n)]}
    )
    return labels, splits, texts


def test_join_keeps_every_review_with_text_and_labels():
    joined = join_gold(*frames())
    assert len(joined) == 4
    assert list(joined.columns) == ["review_id", "split", "content", *LABEL_NAMES]


def test_missing_text_raises_instead_of_dropping():
    labels, splits, texts = frames()
    with pytest.raises(ValueError, match="no text"):
        join_gold(labels, splits, texts.iloc[:3])


def test_empty_text_counts_as_missing():
    labels, splits, texts = frames()
    texts.loc[0, "content"] = "   "
    with pytest.raises(ValueError, match="no text"):
        join_gold(labels, splits, texts)


def test_split_ids_must_match_label_ids():
    labels, splits, texts = frames()
    with pytest.raises(ValueError, match="different review ids"):
        join_gold(labels, splits.iloc[:3], texts)


def test_xy_returns_one_split_aligned_by_position():
    texts, y = xy(join_gold(*frames()), "train")
    assert list(texts) == ["Text 0", "Text 1"]
    assert list(y.columns) == list(LABEL_NAMES)
    assert list(y.index) == [0, 1]
