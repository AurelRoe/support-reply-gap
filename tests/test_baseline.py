"""Tests for the C1 baseline. Short German toy sentences; no database."""

import pandas as pd

from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.baseline import (
    KEEP_NEGATIONS,
    build_features,
    fit,
    german_stopwords,
    predict_proba,
    to_labels,
)

TEXTS = [
    "Kontoeröffnung dauert ewig",
    "Die Eröffnung klappt nicht",
    "Überweisung kommt nicht an",
    "Überweisung dauert ewig",
]


def vocab(features, part: str) -> set[str]:
    return set(dict(features.transformer_list)[part].vocabulary_)


def test_umlauts_are_kept():
    features = build_features()
    features.fit(TEXTS)
    assert "überweisung" in vocab(features, "word")
    assert "uberweisung" not in vocab(features, "word")


def test_compound_parts_share_char_features():
    features = build_features()
    features.fit(TEXTS)
    # "röff" appears in both Kontoeröffnung and Eröffnung, so it passes min_df=2
    assert "röff" in vocab(features, "char")


def test_one_row_per_text():
    features = build_features()
    assert features.fit_transform(TEXTS).shape[0] == len(TEXTS)


def test_stopwords_are_german_and_keep_negations():
    words = german_stopwords()
    assert {"und", "der", "ist"} <= set(words)
    assert not KEEP_NEGATIONS & set(words)


def test_stopwords_are_removed_but_negations_survive():
    features = build_features(stop_words=german_stopwords(), min_df=1)
    features.fit(TEXTS)
    words = vocab(features, "word")
    assert "die" not in words
    assert "nicht" in words
    assert "klappt nicht" in words  # word pairs are built after removal, so this pair survives


def toy_train() -> tuple[pd.Series, pd.DataFrame]:
    # Two texts per label, each naming its label, so every label has positives to learn from.
    rows = [
        (f"Bewertung {name.replace('_', ' ')} Nummer {i}", name)
        for name in LABEL_NAMES
        for i in range(2)
    ]
    texts = pd.Series([text for text, _ in rows])
    y = pd.DataFrame([{n: int(n == label) for n in LABEL_NAMES} for _, label in rows])
    return texts, y


def test_proba_has_one_column_per_label_in_taxonomy_order():
    texts, y = toy_train()
    proba = predict_proba(fit(texts, y, min_df=1), texts)
    assert list(proba.columns) == list(LABEL_NAMES)
    assert proba.shape == (len(texts), len(LABEL_NAMES))
    assert ((proba >= 0) & (proba <= 1)).all().all()


def test_model_learns_the_toy_labels():
    texts, y = toy_train()
    proba = predict_proba(fit(texts, y, min_df=1), texts)
    # on its own training data the true label should come out on top for every row
    assert (proba.to_numpy().argmax(axis=1) == y.to_numpy().argmax(axis=1)).all()


def proba_row(**high: float) -> pd.DataFrame:
    """One review: the named labels get the given probability, the rest 0.1."""
    return pd.DataFrame([{name: high.get(name, 0.1) for name in LABEL_NAMES}])


def labels_of(pred: pd.DataFrame) -> set[str]:
    return {name for name in LABEL_NAMES if pred.loc[0, name] == 1}


def test_labels_above_threshold_are_kept():
    pred = to_labels(proba_row(login_access=0.8, support_experience=0.6))
    assert labels_of(pred) == {"login_access", "support_experience"}


def test_empty_row_takes_most_likely_label():
    pred = to_labels(proba_row(performance=0.3, feature_gap=0.4))
    assert labels_of(pred) == {"feature_gap"}


def test_named_complaint_wins_over_exclusive_label():
    pred = to_labels(proba_row(no_complaint=0.9, fees_pricing=0.6))
    assert labels_of(pred) == {"fees_pricing"}


def test_both_exclusive_labels_keep_the_more_likely():
    pred = to_labels(proba_row(no_complaint=0.7, complaint_unspecified=0.6))
    assert labels_of(pred) == {"no_complaint"}


def test_exclusive_label_alone_is_kept():
    pred = to_labels(proba_row(complaint_unspecified=0.3))
    assert labels_of(pred) == {"complaint_unspecified"}


def test_classifier_returns_taxonomy_labels_and_no_cost():
    from support_reply_gap.models.baseline import BaselineClassifier

    texts, y = toy_train()
    labels, cost = BaselineClassifier(texts, y).classify(texts[0])
    assert labels and labels <= set(LABEL_NAMES)
    assert cost == 0.0
