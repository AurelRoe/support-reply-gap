"""Tests for saving and loading the router's baseline. Toy data; no database."""

import pandas as pd
import pytest
import sklearn

from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models import artifact
from support_reply_gap.models.baseline import fit_chosen, predict_proba


def toy_model():
    texts = pd.Series(["Login geht nicht", "App stürzt ab", "Super Bank", "Gebühren zu hoch"] * 3)
    # Every label needs both a 0 and a 1 to be fitted, so alternate them.
    y = pd.DataFrame({name: [i % 2 for i in range(len(texts))] for name in LABEL_NAMES})
    return fit_chosen(texts, y)


def test_saved_model_predicts_the_same_after_loading(tmp_path):
    model = toy_model()
    path = tmp_path / "baseline.joblib"
    artifact.save(model, path, trained_on=12)
    bundle = artifact.load(path)
    texts = pd.Series(["Kann mich nicht einloggen"])
    pd.testing.assert_frame_equal(
        predict_proba(model, texts), predict_proba(bundle["model"], texts)
    )


def test_bundle_carries_the_settings_it_was_measured_with(tmp_path):
    path = tmp_path / "baseline.joblib"
    artifact.save(toy_model(), path, trained_on=12)
    bundle = artifact.load(path)
    assert bundle["trained_on"] == 12
    assert bundle["threshold"] == 0.3
    assert bundle["margin_cut"] == artifact.ROUTER_MARGIN


def test_loading_refuses_a_different_scikit_learn_version(tmp_path, monkeypatch):
    path = tmp_path / "baseline.joblib"
    artifact.save(toy_model(), path, trained_on=12)
    monkeypatch.setattr(sklearn, "__version__", "0.0.1")
    with pytest.raises(RuntimeError, match="saved with scikit-learn"):
        artifact.load(path)
