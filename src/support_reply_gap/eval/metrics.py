"""Multi-label scores shared by the human ceiling (B5) and every model (C3 onward).

One function scores both, so "model vs. human ceiling" compares like with like.

Inputs are 0/1 frames with one column per label, rows aligned by position.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score


def scorable_labels(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> list[str]:
    """Labels that occur in at least one of the two frames.

    A label absent from both has no defined F1. Counting it as 0 would punish
    agreement, and counting it as 1 would reward a class that was never tested.
    So it is left out of the macro average and reported as n/a.
    """
    return [c for c in y_true.columns if y_true[c].any() or y_pred[c].any()]


def per_class_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """F1 and support per label. Support counts y_true, the reference side."""
    _check_aligned(y_true, y_pred)
    rows = []
    for label in y_true.columns:
        defined = y_true[label].any() or y_pred[label].any()
        rows.append(
            {
                "label": label,
                "f1": f1_score(y_true[label], y_pred[label], zero_division=0)
                if defined
                else float("nan"),
                "support": int(y_true[label].sum()),
                "predicted": int(y_pred[label].sum()),
            }
        )
    return pd.DataFrame(rows).set_index("label")


def macro_f1(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Unweighted mean of per-label F1 over the labels that occur at all."""
    return float(per_class_f1(y_true, y_pred)["f1"].mean(skipna=True))


def exact_match(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Share of rows where the whole label set agrees."""
    _check_aligned(y_true, y_pred)
    return float((y_true.to_numpy() == y_pred.to_numpy()).all(axis=1).mean())


def _check_aligned(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    if list(y_true.columns) != list(y_pred.columns) or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same labels and row count")
