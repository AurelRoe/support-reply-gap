"""Which models the harness can score, and how each is built and fed.

Kept apart from the command-line code because everything here can change a score,
and the gate re-checks results whenever this file changes. CLI edits should not.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from support_reply_gap.eval.harness import Classifier
from support_reply_gap.models.data import xy


def split_inputs(gold: pd.DataFrame, split: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Review ids, texts and gold labels for one split, in the same row order."""
    texts, y = xy(gold, split)
    ids = gold.loc[gold["split"] == split, "review_id"].reset_index(drop=True)
    return ids, texts, y


def build_baseline(gold: pd.DataFrame, **_options) -> Classifier:
    from support_reply_gap.models.baseline import BaselineClassifier

    # Train only: validation stays unseen so its score means what it meant in C1.
    return BaselineClassifier(*xy(gold, "train"))


def build_llm(gold: pd.DataFrame, max_usd: float = 0.10) -> Classifier:
    from support_reply_gap.models.llm import LLMClassifier

    # Nothing to train: the gold set is not used, so validation and test stay unseen.
    return LLMClassifier(max_usd=max_usd)


MODELS: dict[str, Callable[..., Classifier]] = {"baseline": build_baseline, "llm": build_llm}
