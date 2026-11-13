"""Paired comparison of two eval results on the same reviews.

Both models are scored on the same reviews, so instead of comparing two separate
intervals, the difference itself is bootstrapped: each resample draws one set of
reviews and scores both models on it. A review that is hard for both models moves
both scores the same way and cancels out of the difference. How much narrower that
makes the interval depends on how much the two models' errors overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from support_reply_gap.eval.harness import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED
from support_reply_gap.labeling.taxonomy import LABEL_NAMES


def shared_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> float:
    """Mean F1 over the given label columns.

    Both models are averaged over the same labels in each resample. Otherwise a rare
    label missing from a resample could drop out for one model and count as 0 for the
    other, and the difference would compare averages over different labels.
    """
    tp = (y_true & y_pred).sum(axis=0)
    fp = (~y_true & y_pred).sum(axis=0)
    fn = (y_true & ~y_pred).sum(axis=0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return float(f1[labels].mean())


def predictions_frame(result: dict, split: str, ids: list[str]) -> np.ndarray:
    preds = result["splits"][split]["predictions"]
    return np.array([[name in preds[rid] for name in LABEL_NAMES] for rid in ids])


def paired_difference(
    a: dict, b: dict, split: str, gold: pd.DataFrame, rounds: int = BOOTSTRAP_ROUNDS
) -> dict:
    """Macro-F1 of model a minus model b on one split, with a paired 95% interval.

    Labels are those present in the gold labels of the (re)sampled reviews. On a full
    split where every label occurs, that matches metrics.macro_f1 exactly.
    """
    for r in (a, b):
        if "predictions" not in r["splits"].get(split, {}):
            raise ValueError(f"{r['model']} has no per-review predictions for {split}; re-run it")
    if a["gold_fingerprint"] != b["gold_fingerprint"]:
        raise ValueError("the two results were measured against different gold sets")
    ids = sorted(a["splits"][split]["predictions"])
    if ids != sorted(b["splits"][split]["predictions"]):
        raise ValueError(f"the two results cover different {split} reviews")

    truth = gold.set_index("review_id").loc[ids, list(LABEL_NAMES)].to_numpy().astype(bool)
    pa, pb = predictions_frame(a, split, ids), predictions_frame(b, split, ids)

    def diff(idx: np.ndarray) -> float:
        t = truth[idx]
        labels = t.any(axis=0)
        return shared_macro_f1(t, pa[idx], labels) - shared_macro_f1(t, pb[idx], labels)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [diff(rng.integers(0, len(ids), len(ids))) for _ in range(rounds)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "a": a["model"],
        "b": b["model"],
        "split": split,
        "n": len(ids),
        "diff": diff(np.arange(len(ids))),
        "ci95": [float(low), float(high)],
    }
