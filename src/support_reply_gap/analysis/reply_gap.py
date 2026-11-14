"""Reply-gap analysis: complaint labels joined to whether and how fast the app replied.

C2 labels the population with the baseline; D1 with the LLM and adds the star
adjustment below. The D1 rules are fixed in docs/build-order.md.

Descriptive only. Which reviews get a reply is the company's choice and that choice is
unobserved, so nothing here says why a complaint type is or is not answered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from support_reply_gap.analysis.data import load_reviews
from support_reply_gap.apps import APPS
from support_reply_gap.config import ANALYSIS_WINDOW_START
from support_reply_gap.eval.harness import BOOTSTRAP_ROUNDS, BOOTSTRAP_SEED
from support_reply_gap.labeling.sampling import (
    add_language,
    add_text_flags,
    eligible_for_labelling,
)
from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.baseline import CHOSEN, fit_chosen, predict

# C2 covers 1–3★ only. The gold set holds 21 5★ reviews, and on the population the
# baseline gives complaint labels to about half of all 5★ reviews, which would drag
# label reply rates toward the low 4–5★ reply rate. Revisit in D1 with a better model.
MAX_STARS = 3

# Cells below this many reviews are flagged, not interpreted.
SMALL_CELL = 30
ALL_APPS = "all"
ALL_REVIEWS = "(all reviews)"

# N26 replies to 99% of 1–3★ reviews, so its reply rate cannot separate labels.
# Its cells are never findings on reply rate; it is compared on latency only.
LATENCY_ONLY_APPS = ("n26",)


def select_population(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep the reviews the analysis covers: the B3 cleaning rules, repeated texts kept.

    The same rules as the gold sample, so the model only labels the kind of review
    it was trained on. Repeats stay in: each is a real review that did or did not
    get a reply, whereas the gold sample dropped them to save labelling time.
    """
    flagged = add_language(add_text_flags(reviews))
    return eligible_for_labelling(flagged).reset_index(drop=True)


def within_star_band(reviews: pd.DataFrame, max_stars: int = MAX_STARS) -> pd.DataFrame:
    """Keep reviews rated at most `max_stars`."""
    return reviews[reviews["score"] <= max_stars].reset_index(drop=True)


def load_population() -> pd.DataFrame:
    """The analysis population from Postgres: the three apps, analysis window, 1–3★."""
    reviews = load_reviews(since=ANALYSIS_WINDOW_START, apps=tuple(app.key for app in APPS))
    return select_population(within_star_band(reviews))


def label_population(population: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """The population plus one 0/1 column per label, predicted by the C1 baseline.

    Trained on train + validation: the settings were already chosen on validation, so
    its reviews are now just more training data. Test stays held out for C3.
    `in_training` marks the reviews the model was fitted on, so their share is visible.
    """
    fit_on = gold[gold["split"] != "test"]
    model = fit_chosen(fit_on["content"], fit_on[list(LABEL_NAMES)])
    labels = predict(model, population["content"], CHOSEN["threshold"])

    out = population.reset_index(drop=True)
    out["in_training"] = out["review_id"].isin(fit_on["review_id"])
    return pd.concat([out, labels], axis=1)


def summarise(cell: pd.DataFrame) -> dict:
    """Reply figures for one group of reviews."""
    # reply_latency_hours is null both without a reply and for edited reviews,
    # whose original post time is unknown, so its count can be below the reply count.
    latency = cell["reply_latency_hours"].dropna()
    held_out = cell[~cell["in_training"]]
    return {
        "n": len(cell),
        "reply_rate": cell["has_reply"].mean() if len(cell) else float("nan"),
        "median_latency_h": latency.median() if len(latency) else float("nan"),
        "n_latency": len(latency),
        "share_1_star": (cell["score"] == 1).mean() if len(cell) else float("nan"),
        # The same rate without reviews the model was trained on, whose labels mostly
        # repeat the human ones. A large gap means the training reviews drive the cell.
        "n_held_out": len(held_out),
        "reply_rate_held_out": held_out["has_reply"].mean() if len(held_out) else float("nan"),
    }


def reply_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """One row per app (plus all apps pooled) and label (plus all reviews as reference).

    A review with several labels counts once in each. `share` is the label's share of
    the app's reviews, so shares within an app can add up to more than 1.
    """
    rows = []
    for app in [*sorted(labelled["app_key"].unique()), ALL_APPS]:
        in_app = labelled if app == ALL_APPS else labelled[labelled["app_key"] == app]
        for label in [ALL_REVIEWS, *LABEL_NAMES]:
            cell = in_app if label == ALL_REVIEWS else in_app[in_app[label] == 1]
            rows.append(
                {
                    "app_key": app,
                    "label": label,
                    "share": len(cell) / len(in_app),
                    **summarise(cell),
                }
            )
    table = pd.DataFrame(rows)
    table["small"] = table["n"] < SMALL_CELL
    return table


def label_with_classifier(population: pd.DataFrame, classifier) -> pd.DataFrame:
    """The population plus one 0/1 column per label, from any harness classifier.

    For D1's LLM, which was not trained on any review, so `in_training` is False
    throughout and C2's check without the training reviews has nothing to leave out.
    """
    predicted = [classifier.classify(text)[0] for text in population["content"]]
    labels = pd.DataFrame(
        [{name: int(name in found) for name in LABEL_NAMES} for found in predicted],
        columns=list(LABEL_NAMES),  # keeps the columns when the population is empty
    )
    out = population.reset_index(drop=True)
    out["in_training"] = False
    return pd.concat([out, labels], axis=1)


def observed_and_expected(
    score: np.ndarray, replied: np.ndarray, in_cell: np.ndarray
) -> tuple[float, float]:
    """A cell's reply rate, and the rate the app's per-star rates predict for its star mix.

    `score` and `replied` cover all of one app's reviews; `in_cell` marks the label's.
    """
    cell_scores = score[in_cell]
    if len(cell_scores) == 0:
        return float("nan"), float("nan")
    observed = replied[in_cell].mean()
    expected = sum(
        (cell_scores == star).mean() * replied[score == star].mean()
        for star in np.unique(cell_scores)
    )
    return float(observed), float(expected)


def star_adjusted(labelled: pd.DataFrame, rounds: int = BOOTSTRAP_ROUNDS) -> pd.DataFrame:
    """Per app and label: observed minus star-adjusted expected reply rate, with a 95% interval.

    Within apps only: pooling apps would mix their very different reply rates. The
    interval resamples the app's reviews, so the expected rate is re-estimated each
    round too. A cell is a finding only if it has at least SMALL_CELL reviews, its
    interval excludes 0, and the app is not in LATENCY_ONLY_APPS.
    """
    rows = []
    for app in sorted(labelled["app_key"].unique()):
        in_app = labelled[labelled["app_key"] == app]
        score = in_app["score"].to_numpy()
        replied = in_app["has_reply"].to_numpy(dtype=float)
        # Same seed per app, so adding or dropping an app leaves the others' intervals as they were.
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = [rng.integers(0, len(in_app), len(in_app)) for _ in range(rounds)]

        for label in LABEL_NAMES:
            in_cell = in_app[label].to_numpy() == 1
            n = int(in_cell.sum())
            observed, expected = observed_and_expected(score, replied, in_cell)
            low = high = float("nan")
            if n:
                diffs = []
                for idx in draws:
                    o, e = observed_and_expected(score[idx], replied[idx], in_cell[idx])
                    diffs.append(o - e)
                # A resample can miss a small cell entirely; those rounds are skipped.
                low, high = (float(v) for v in np.nanpercentile(diffs, [2.5, 97.5]))
            rows.append(
                {
                    "app_key": app,
                    "label": label,
                    "n": n,
                    "observed": observed,
                    "expected": expected,
                    "difference": observed - expected,
                    "low": low,
                    "high": high,
                    "finding": bool(
                        n >= SMALL_CELL and app not in LATENCY_ONLY_APPS and (low > 0 or high < 0)
                    ),
                }
            )
    return pd.DataFrame(rows)
