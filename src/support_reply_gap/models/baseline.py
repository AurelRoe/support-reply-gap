"""C1 baseline: TF-IDF features + one logistic regression per label.

    python -m support_reply_gap.models.baseline   # tune on validation, print scores

German text throughout: no English stopwords, umlauts and ß kept as they are.
"""

from __future__ import annotations

from itertools import product

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from support_reply_gap.eval.metrics import macro_f1, per_class_f1
from support_reply_gap.labeling.taxonomy import EXCLUSIVE_LABELS, LABEL_NAMES
from support_reply_gap.models.data import load_gold, xy


def build_features(
    word_ngrams: tuple[int, int] = (1, 2),
    char_ngrams: tuple[int, int] = (3, 5),
    stop_words: list[str] | None = None,
    min_df: int = 2,
) -> FeatureUnion:
    """Word and character TF-IDF side by side, as one feature matrix."""
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=word_ngrams,
                    lowercase=True,
                    # None keeps ä/ö/ü/ß; stripping them would merge e.g. "zahlen" and "zählen"
                    strip_accents=None,
                    stop_words=stop_words,
                    # log-scaled counts, so one word repeated in a rant doesn't dominate
                    sublinear_tf=True,
                    min_df=min_df,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    # char_wb: n-grams inside word boundaries, so parts of compounds
                    # ("Kontoeröffnung" / "Eröffnung") share features
                    analyzer="char_wb",
                    ngram_range=char_ngrams,
                    lowercase=True,
                    strip_accents=None,
                    sublinear_tf=True,
                    min_df=min_df,
                ),
            ),
        ]
    )


# Negations carry the complaint: "funktioniert nicht" without "nicht" reads as praise.
KEEP_NEGATIONS = frozenset(
    {"nicht", "nichts", "kein", "keine", "keinem", "keinen", "keiner", "keines", "ohne"}
)


def german_stopwords() -> list[str]:
    """NLTK's German stopword list minus negations. A list, because sklearn expects one."""
    return sorted(set(stopwords.words("german")) - KEEP_NEGATIONS)


def build_model(C: float = 1.0, **feature_options) -> Pipeline:
    """Features and classifier as one object, so fit() only ever sees training text."""
    return Pipeline(
        [
            ("features", build_features(**feature_options)),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=C,
                        # rare labels (fees_pricing: 7 train examples) get weighted up,
                        # otherwise "never predict it" is the cheapest answer
                        class_weight="balanced",
                        max_iter=1000,
                    )
                ),
            ),
        ]
    )


def fit(texts: pd.Series, y: pd.DataFrame, C: float = 1.0, **feature_options) -> Pipeline:
    model = build_model(C=C, **feature_options)
    model.fit(texts, y[list(LABEL_NAMES)].to_numpy())
    return model


def predict_proba(model: Pipeline, texts: pd.Series) -> pd.DataFrame:
    """Probability per label, one column per label in taxonomy order."""
    return pd.DataFrame(model.predict_proba(texts), columns=list(LABEL_NAMES))


def to_labels(proba: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """0/1 labels that follow the gold set's rules: at least one label, exclusive labels alone."""
    pred = (proba >= threshold).astype(int)

    # Every gold review has at least one label, so an empty row takes its most likely one.
    empty = pred.sum(axis=1) == 0
    top = proba.idxmax(axis=1)
    for i in pred.index[empty]:
        pred.loc[i, top[i]] = 1

    exclusive = sorted(EXCLUSIVE_LABELS)
    others = [c for c in pred.columns if c not in EXCLUSIVE_LABELS]

    # A named complaint wins over "no complaint" / "unspecified", as in the taxonomy.
    has_other = pred[others].sum(axis=1) > 0
    pred.loc[has_other, exclusive] = 0

    # Both exclusive labels and nothing else: keep the more likely one.
    both = pred[exclusive].sum(axis=1) == 2
    for i in pred.index[both]:
        pred.loc[i, proba.loc[i, exclusive].idxmin()] = 0

    return pred


def predict(model: Pipeline, texts: pd.Series, threshold: float = 0.5) -> pd.DataFrame:
    """0/1 label frame in taxonomy order, ready for eval/metrics.py."""
    return to_labels(predict_proba(model, texts), threshold)


# Kept small on purpose: 60 validation reviews can't tell apart many close settings,
# and every extra setting tried makes the best validation score more flattering.
GRID = {
    "C": [0.3, 1.0, 3.0, 10.0, 30.0],
    "stop_words": ["none", "german"],
    "word_ngrams": [(1, 1), (1, 2)],
}
# Thresholds reuse one fit's probabilities, so trying them costs no extra training.
THRESHOLDS = [0.2, 0.3, 0.4, 0.5]

# Picked in C1 from the grid (val macro-F1 0.486). C=30 / threshold 0.2 scored 0.489, a gap
# 60 reviews can't resolve, so the setting inside the grid is kept over the one on its edge.
CHOSEN = {"C": 10.0, "stop_words": "none", "word_ngrams": (1, 1), "threshold": 0.3}


def stopword_list(name: str) -> list[str] | None:
    return german_stopwords() if name == "german" else None


def tune(gold: pd.DataFrame) -> pd.DataFrame:
    """Validation macro-F1 for every grid setting, best first. Never touches test."""
    train_x, train_y = xy(gold, "train")
    val_x, val_y = xy(gold, "val")
    rows = []
    for C, stop, word in product(GRID["C"], GRID["stop_words"], GRID["word_ngrams"]):
        model = fit(train_x, train_y, C=C, stop_words=stopword_list(stop), word_ngrams=word)
        proba = predict_proba(model, val_x)
        for threshold in THRESHOLDS:
            score = macro_f1(val_y, to_labels(proba, threshold))
            rows.append(
                {
                    "C": C,
                    "stop_words": stop,
                    "word_ngrams": word,
                    "threshold": threshold,
                    "val_macro_f1": score,
                }
            )
    # Stable sort: on a tie the earlier, more regularised setting stays ahead.
    return pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False, kind="stable")


def fit_chosen(texts: pd.Series, y: pd.DataFrame) -> Pipeline:
    """The baseline as C3 and later stages use it: CHOSEN settings, fitted on the given data."""
    return fit(
        texts,
        y,
        C=CHOSEN["C"],
        stop_words=stopword_list(CHOSEN["stop_words"]),
        word_ngrams=CHOSEN["word_ngrams"],
    )


class BaselineClassifier:
    """The baseline behind the harness interface: one review in, labels and cost out."""

    name = "baseline"

    def __init__(self, texts: pd.Series, y: pd.DataFrame) -> None:
        self.model = fit_chosen(texts, y)
        self.config = {**CHOSEN, "trained_on": len(texts)}

    def classify(self, text: str) -> tuple[set[str], float]:
        pred = predict(self.model, pd.Series([text]), CHOSEN["threshold"])
        # Runs locally, so no per-call charge; compute time shows up in latency instead.
        return {name for name in LABEL_NAMES if pred.loc[0, name] == 1}, 0.0


def main() -> None:
    gold = load_gold()
    results = tune(gold)
    print(f"{len(results)} settings, validation macro-F1 (top 10):")
    print(results.head(10).to_string(index=False, float_format="%.3f"))

    model = fit_chosen(*xy(gold, "train"))
    val_x, val_y = xy(gold, "val")
    scores = per_class_f1(val_y, predict(model, val_x, CHOSEN["threshold"]))
    print(f"\nChosen setting {CHOSEN}, per class on validation:")
    print(scores.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
