"""C3 evaluation harness: run any classifier over gold reviews and score it.

The yardstick every model is measured by. It was written before the LLM path so the
LLM is judged by a test that existed before it did.

Validation scores are for decisions (thresholds, prompts); test scores are reported
and never used to choose anything.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from support_reply_gap.eval.metrics import exact_match, macro_f1, per_class_f1
from support_reply_gap.labeling.gold import LABELS_PATH, ROOT, SPLITS_PATH
from support_reply_gap.labeling.taxonomy import LABEL_NAMES

RESULTS_DIR = ROOT / "reports/eval"

BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 20261113
# Below this many examples in a split, a per-label F1 swings between 0 and 1 on one review.
FEW_EXAMPLES = 5


class Classifier(Protocol):
    """What a model must provide to be evaluated."""

    name: str
    config: dict

    def classify(self, text: str) -> tuple[set[str], float]:
        """Labels for one review, and the cost of the call in USD."""
        ...


def run_split(model: Classifier, ids: pd.Series, texts: pd.Series, y_true: pd.DataFrame) -> dict:
    """Classify each review on its own, timing every call. Returns raw predictions."""
    rows, latencies, costs = [], [], []
    for text in texts:
        start = time.perf_counter()
        labels, cost = model.classify(text)
        latencies.append(time.perf_counter() - start)
        costs.append(cost)
        unknown = labels - set(LABEL_NAMES)
        if unknown:
            raise ValueError(f"{model.name} returned labels outside the taxonomy: {unknown}")
        rows.append({name: int(name in labels) for name in LABEL_NAMES})
    y_pred = pd.DataFrame(rows, columns=list(LABEL_NAMES))
    return {
        "ids": list(ids),
        "y_true": y_true.reset_index(drop=True),
        "y_pred": y_pred,
        "latency_s": latencies,
        "cost_usd": costs,
    }


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Same definition as metrics.macro_f1, in numpy, so 1,000 bootstrap rounds stay quick."""
    tp = (y_true & y_pred).sum(axis=0)
    fp = (~y_true & y_pred).sum(axis=0)
    fn = (y_true & ~y_pred).sum(axis=0)
    defined = (y_true.any(axis=0)) | (y_pred.any(axis=0))
    f1 = np.where(defined, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return float(np.nanmean(f1))


def bootstrap_interval(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, rounds: int = BOOTSTRAP_ROUNDS
) -> tuple[float, float]:
    """95% interval for macro-F1 from resampling reviews with replacement.

    It shows how much the score would move on another sample of the same size, which
    is what decides whether two models actually differ.
    """
    t = y_true.to_numpy().astype(bool)
    p = y_pred.to_numpy().astype(bool)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores = []
    for _ in range(rounds):
        idx = rng.integers(0, len(t), len(t))
        scores.append(fast_macro_f1(t[idx], p[idx]))
    low, high = np.nanpercentile(scores, [2.5, 97.5])
    return float(low), float(high)


def confusion(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """Label x label counts: row = true label, column = predicted label.

    A cell counts reviews carrying both, so the diagonal holds correct hits and the
    rest of a row shows what a true label was predicted alongside or instead.
    `missed` counts reviews where the true label was not predicted at all.
    """
    matrix = y_true.T.dot(y_pred)
    matrix["missed"] = y_true.sum() - np.diag(matrix[list(LABEL_NAMES)])
    return matrix


def score(run: dict) -> dict:
    y_true, y_pred = run["y_true"], run["y_pred"]
    low, high = bootstrap_interval(y_true, y_pred)
    per_class = per_class_f1(y_true, y_pred)
    per_class["few_examples"] = per_class["support"] < FEW_EXAMPLES
    return {
        "n": len(y_true),
        "macro_f1": macro_f1(y_true, y_pred),
        "macro_f1_ci95": [low, high],
        "exact_match": exact_match(y_true, y_pred),
        "cost_per_1k_usd": 1000 * float(np.mean(run["cost_usd"])),
        "latency_p95_ms": 1000 * float(np.percentile(run["latency_s"], 95)),
        "latency_median_ms": 1000 * float(np.median(run["latency_s"])),
        "per_class": per_class.reset_index().to_dict(orient="records"),
        "confusion": confusion(y_true, y_pred).to_dict(orient="index"),
        # Per review, so two models can be compared on the same reviews (paired bootstrap).
        # Ids and labels only, no text, like data/gold/.
        "predictions": {
            rid: [name for name in LABEL_NAMES if row[name]]
            for rid, (_, row) in zip(run["ids"], y_pred.iterrows(), strict=True)
        },
    }


def gold_fingerprint(paths: tuple[Path, ...] = (LABELS_PATH, SPLITS_PATH)) -> str:
    """Short checksum of the gold labels and split. Results with different
    fingerprints were measured against different yardsticks and are not comparable."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def git_commit() -> str:
    """Current commit, with '-dirty' when uncommitted changes could have shaped the result."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        # reports/ holds the harness's own output, which cannot change a result.
        changes = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!reports")
        return commit + ("-dirty" if changes else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def evaluate(
    model: Classifier, splits: dict[str, tuple[pd.Series, pd.Series, pd.DataFrame]]
) -> dict:
    """Score one model on each named split: (review ids, texts, gold labels) per split."""
    return {
        "model": model.name,
        "config": model.config,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "gold_fingerprint": gold_fingerprint(),
        "splits": {name: score(run_split(model, *parts)) for name, parts in splits.items()},
    }


def save(result: dict, results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    day = result["run_at"][:10]
    path = results_dir / f"{result['model']}-{day}-{result['commit']}.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    return path


def _json_default(value):
    # numpy and pandas scalars are not plain Python numbers
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")
