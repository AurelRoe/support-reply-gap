"""Tests for the C6 router. Toy data and a fake LLM; no database, no API."""

import numpy as np
import pandas as pd

from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.router import (
    RouterClassifier,
    combine,
    margins,
    operating_point,
    sweep,
)


def frame(sets: list[set[str]]) -> pd.DataFrame:
    return pd.DataFrame([{n: int(n in s) for n in LABEL_NAMES} for s in sets])


def test_margin_is_the_closest_call_to_the_threshold():
    proba = pd.DataFrame([{n: 0.0 for n in LABEL_NAMES}, {n: 0.0 for n in LABEL_NAMES}])
    proba.loc[0, "login_access"] = 0.29  # almost at the 0.3 threshold: uncertain
    proba.loc[1, "login_access"] = 0.90  # far from it on every label: certain
    m = margins(proba, threshold=0.3)
    assert m[0] < m[1]
    assert abs(m[0] - 0.01) < 1e-9


def test_combine_replaces_only_escalated_rows():
    base = frame([{"feature_gap"}, {"performance"}])
    llm = frame([{"login_access"}, {"crash_stability"}])
    out = combine(base, llm, np.array([False, True]))
    assert out.loc[0, "feature_gap"] == 1
    assert out.loc[1, "crash_stability"] == 1 and out.loc[1, "performance"] == 0


TRUTH = [{"login_access"}, {"performance"}, {"feature_gap"}, {"no_complaint"}]


def test_sweep_runs_from_baseline_alone_to_llm_alone():
    base = frame([{"feature_gap"}] * 4)
    llm = frame(TRUTH)  # a perfect LLM
    curve = sweep(frame(TRUTH), base, llm, np.array([0.01, 0.02, 0.03, 0.04]))
    assert curve.iloc[0]["share_escalated"] == 0.0
    assert curve.iloc[-1]["share_escalated"] == 1.0
    assert curve.iloc[-1]["macro_f1"] == 1.0


def test_operating_point_is_the_smallest_share_reaching_the_target():
    curve = pd.DataFrame(
        {"share_escalated": [0.0, 0.3, 0.6, 1.0], "macro_f1": [0.5, 0.6, 0.69, 0.7]}
    )
    # 90% of the gain from 0.5 to 0.7 needs 0.68: first reached at 60%
    assert operating_point(curve, 0.5, 0.7)["share_escalated"] == 0.6


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.last_latency_s = 2.0
        self.config = {"model": "fake"}

    def classify(self, text):
        self.calls += 1
        return {"fees_pricing"}, 0.001


def toy_train():
    rows = [
        (f"Bewertung {n.replace('_', ' ')} Nummer {i}", n) for n in LABEL_NAMES for i in range(2)
    ]
    texts = pd.Series([t for t, _ in rows])
    return texts, frame([{label} for _, label in rows])


def test_router_never_escalating_is_the_baseline_and_free():
    router = RouterClassifier(*toy_train(), FakeLLM(), margin_cut=-1.0)
    labels, cost = router.classify("Bewertung login access Nummer 0")
    assert cost == 0.0 and router.llm.calls == 0
    assert labels <= set(LABEL_NAMES) and labels


def test_router_always_escalating_uses_the_llm_and_its_latency():
    router = RouterClassifier(*toy_train(), FakeLLM(), margin_cut=np.inf)
    labels, cost = router.classify("irgendwas")
    assert labels == {"fees_pricing"} and cost == 0.001
    assert router.last_latency_s >= 2.0
    assert router.config["escalated"] == 1
