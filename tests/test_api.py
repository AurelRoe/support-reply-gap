"""Tests for the HTTP API. TestClient calls the app in memory: no server, no network.

The app loads the committed model file from artifacts/, so these need no database.
The LLM tier is always switched off or replaced by a fake: these tests never spend.
"""

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from support_reply_gap.api import main
from support_reply_gap.api.main import BUNDLE, MAX_TEXT_LENGTH, app, classify_text, make_llm
from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.llm import BudgetExceeded

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """The local .env holds a real API key; no test may reach the real LLM."""
    monkeypatch.setattr(main, "LLM", None)


class FakeLLM:
    """Stands in for LLMClassifier: answers with fixed labels, or raises a given error."""

    def __init__(self, labels=("payments_transfers", "fees_pricing"), error=None):
        self.labels = set(labels)
        self.error = error
        self.calls = 0

    def classify(self, text):
        self.calls += 1
        if self.error:
            raise self.error
        return self.labels, 0.0001


def always_uncertain(monkeypatch):
    # Every margin is below 1.0, so every review escalates.
    monkeypatch.setitem(BUNDLE, "margin_cut", 1.0)


def never_uncertain(monkeypatch):
    # No margin is below 0, so no review escalates.
    monkeypatch.setitem(BUNDLE, "margin_cut", 0.0)


REVIEW = "Seit drei Tagen kann ich mich nicht einloggen und der Support antwortet nicht."


def test_root_redirects_to_the_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_labels_from_the_taxonomy():
    response = client.post("/classify", json={"text": REVIEW})
    assert response.status_code == 200
    body = response.json()
    assert body["labels"], "the baseline always gives at least one label"
    assert set(body["labels"]) <= set(LABEL_NAMES)
    assert body["tier"] == "baseline"


def test_labels_come_in_taxonomy_order():
    labels = client.post("/classify", json={"text": REVIEW}).json()["labels"]
    assert labels == [name for name in LABEL_NAMES if name in labels]


def test_uncertain_follows_the_router_cut_off():
    body = client.post("/classify", json={"text": REVIEW}).json()
    assert body["margin"] >= 0
    assert body["uncertain"] == (body["margin"] < BUNDLE["margin_cut"])


def test_same_text_gives_the_same_answer():
    first = client.post("/classify", json={"text": REVIEW}).json()
    second = client.post("/classify", json={"text": REVIEW}).json()
    assert first == second


def test_empty_text_is_rejected():
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_overlong_text_is_rejected():
    response = client.post("/classify", json={"text": "a" * (MAX_TEXT_LENGTH + 1)})
    assert response.status_code == 422


def test_missing_text_is_rejected():
    assert client.post("/classify", json={"review": REVIEW}).status_code == 422


def test_get_is_not_allowed_on_classify():
    assert client.get("/classify").status_code == 405


def test_confident_review_stays_with_the_baseline(monkeypatch):
    never_uncertain(monkeypatch)
    fake = FakeLLM()
    monkeypatch.setattr(main, "LLM", fake)
    result = classify_text(REVIEW)
    assert result.tier == "baseline"
    assert result.fallback_reason is None
    assert fake.calls == 0


def test_uncertain_review_goes_to_the_llm(monkeypatch):
    always_uncertain(monkeypatch)
    monkeypatch.setattr(main, "LLM", FakeLLM())
    result = classify_text(REVIEW)
    assert result.tier == "llm"
    assert result.labels == ["payments_transfers", "fees_pricing"]  # taxonomy order
    assert result.uncertain
    assert result.fallback_reason is None


def test_without_a_key_the_baseline_answers_and_says_so(monkeypatch):
    always_uncertain(monkeypatch)
    result = classify_text(REVIEW)
    assert result.tier == "baseline"
    assert result.fallback_reason == "llm_not_configured"


def test_spent_budget_falls_back_to_the_baseline(monkeypatch):
    always_uncertain(monkeypatch)
    monkeypatch.setattr(main, "LLM", FakeLLM(error=BudgetExceeded("spent")))
    result = classify_text(REVIEW)
    assert result.tier == "baseline"
    assert result.fallback_reason == "budget_spent"


def test_failed_llm_call_falls_back_to_the_baseline(monkeypatch):
    always_uncertain(monkeypatch)
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.com"))
    monkeypatch.setattr(main, "LLM", FakeLLM(error=error))
    result = classify_text(REVIEW)
    assert result.tier == "baseline"
    assert result.fallback_reason == "llm_error"


def test_refused_llm_answer_falls_back_to_the_baseline(monkeypatch):
    always_uncertain(monkeypatch)
    monkeypatch.setattr(main, "LLM", FakeLLM(labels=()))
    result = classify_text(REVIEW)
    assert result.tier == "baseline"
    assert result.fallback_reason == "llm_error"


def test_escalation_shows_in_the_http_response(monkeypatch):
    always_uncertain(monkeypatch)
    monkeypatch.setattr(main, "LLM", FakeLLM())
    body = client.post("/classify", json={"text": REVIEW}).json()
    assert body["tier"] == "llm"
    assert body["fallback_reason"] is None


def test_no_llm_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_llm() is None


def test_budget_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")  # no call is made
    monkeypatch.setenv("LLM_BUDGET_USD", "0.2")
    llm = make_llm()
    assert llm.max_usd == 0.2
    assert llm.spent_usd == 0.0
