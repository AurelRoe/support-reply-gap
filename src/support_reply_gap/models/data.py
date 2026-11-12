"""The gold set as model input: review text joined to labels and split.

    python -m support_reply_gap.models.data   # check the join, print counts per split

data/gold/ holds only ids and labels, so the text comes from Postgres. The join
is a pure function so it can be tested without a database.
"""

from __future__ import annotations

import pandas as pd

from support_reply_gap.db import get_connection
from support_reply_gap.labeling.gold import LABELS_PATH, SPLITS_PATH
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

SPLITS = ("train", "val", "test")

TEXT_SQL = "select review_id, content from reviews where review_id = any(%s)"


def join_gold(labels: pd.DataFrame, splits: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """One row per gold review: review_id, split, content, then the label columns.

    Raises instead of dropping rows: a gold review that silently goes missing
    would change the yardstick without anyone noticing.
    """
    ids = set(labels["review_id"])
    problems = []
    if not labels["review_id"].is_unique:
        problems.append("duplicate review_id in labels")
    if set(splits["review_id"]) != ids:
        problems.append("splits and labels cover different review ids")
    if not set(splits["split"]) <= set(SPLITS):
        problems.append(f"unknown split value: {set(splits['split']) - set(SPLITS)}")
    missing = ids - set(texts.loc[texts["content"].fillna("").str.strip() != "", "review_id"])
    if missing:
        problems.append(f"{len(missing)} gold reviews have no text, e.g. {sorted(missing)[:3]}")
    if problems:
        raise ValueError("; ".join(problems))

    joined = labels.merge(splits, on="review_id").merge(texts, on="review_id")
    return joined[["review_id", "split", "content", *LABEL_NAMES]]


def fetch_texts(review_ids: list[str]) -> pd.DataFrame:
    with get_connection() as conn:
        rows = conn.execute(TEXT_SQL, [review_ids]).fetchall()
    return pd.DataFrame(rows, columns=["review_id", "content"])


def load_gold() -> pd.DataFrame:
    """The full gold set with text. Needs a running Postgres."""
    labels = pd.read_csv(LABELS_PATH)
    splits = pd.read_csv(SPLITS_PATH)
    return join_gold(labels, splits, fetch_texts(labels["review_id"].tolist()))


def xy(gold: pd.DataFrame, split: str) -> tuple[pd.Series, pd.DataFrame]:
    """Texts and the 0/1 label frame for one split, index reset so rows align by position."""
    part = gold[gold["split"] == split].reset_index(drop=True)
    return part["content"], part[list(LABEL_NAMES)]


def main() -> None:
    gold = load_gold()
    print(f"{len(gold)} gold reviews with text")
    print(gold.groupby("split")[list(LABEL_NAMES)].sum().T[list(SPLITS)].to_string())


if __name__ == "__main__":
    main()
