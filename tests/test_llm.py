"""Tests for the C5 LLM classifier. A fake API client: these tests never cost anything."""

import json
from types import SimpleNamespace

import pytest

from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.llm import (
    MAX_TOKENS,
    BudgetExceeded,
    LLMClassifier,
    apply_label_rules,
    cost_usd,
    system_prompt,
)


def usage(input_tokens=100, output_tokens=50, cache_write=0, cache_read=0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_write,
        cache_read_input_tokens=cache_read,
    )


class FakeClient:
    """Stands in for anthropic.Anthropic(); records calls, answers from a list."""

    def __init__(self, answers, stop_reason="end_turn"):
        self.answers = list(answers)
        self.stop_reason = stop_reason
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        labels = self.answers.pop(0)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=json.dumps({"labels": labels})),
            ],
            usage=usage(),
        )

    def count_tokens(self, **kwargs):
        return SimpleNamespace(input_tokens=3000)


def classifier(tmp_path, answers, **kwargs) -> LLMClassifier:
    stop = kwargs.pop("stop_reason", "end_turn")
    return LLMClassifier(client=FakeClient(answers, stop), cache_dir=tmp_path, **kwargs)


def test_prompt_contains_every_label():
    prompt = system_prompt()
    assert "{taxonomy}" not in prompt
    assert all(name in prompt for name in LABEL_NAMES)


def test_classify_returns_labels_and_cost(tmp_path):
    llm = classifier(tmp_path, [["login_access", "support_experience"]])
    labels, cost = llm.classify("Kann mich nicht einloggen, Support antwortet nicht")
    assert labels == {"login_access", "support_experience"}
    assert cost == pytest.approx(cost_usd(usage()))


def test_request_uses_structured_output_and_caches_the_system_prompt(tmp_path):
    llm = classifier(tmp_path, [["feature_gap"]])
    llm.classify("Es fehlt eine Funktion")
    call = llm.client.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["max_tokens"] == MAX_TOKENS


def test_second_run_answers_from_cache_without_an_api_call(tmp_path):
    first = classifier(tmp_path, [["performance"]])
    _, cost = first.classify("App ist langsam")
    second = classifier(tmp_path, [])  # no answers queued: an API call would fail
    labels, cached_cost = second.classify("App ist langsam")
    assert labels == {"performance"}
    assert cached_cost == cost  # reports the original cost, for the cost-per-1k column
    assert second.client.calls == []
    assert second.spent_usd == 0.0
    assert second.last_latency_s is not None


def test_budget_stops_before_the_call(tmp_path):
    llm = classifier(tmp_path, [["feature_gap"]], max_usd=0.0000001)
    with pytest.raises(BudgetExceeded):
        llm.classify("Es fehlt eine Funktion")
    assert llm.client.calls == []


def test_refusal_scores_as_no_labels_and_is_not_cached(tmp_path):
    llm = classifier(tmp_path, [["feature_gap"]], stop_reason="refusal")
    labels, _ = llm.classify("irgendwas")
    assert labels == set()
    assert llm.failures == 1
    assert list(tmp_path.iterdir()) == []


def test_exclusive_label_is_dropped_next_to_a_named_complaint():
    assert apply_label_rules(["no_complaint", "fees_pricing"]) == {"fees_pricing"}
    assert apply_label_rules(["complaint_unspecified", "no_complaint"]) == {"complaint_unspecified"}
    assert apply_label_rules(["no_complaint"]) == {"no_complaint"}


def test_cost_uses_the_cheaper_rate_for_cache_reads():
    full = cost_usd(usage(input_tokens=3000, output_tokens=0))
    cached = cost_usd(usage(input_tokens=0, output_tokens=0, cache_read=3000))
    assert cached == pytest.approx(full / 10)


def test_estimate_counts_only_uncached_reviews(tmp_path):
    llm = classifier(tmp_path, [["performance"]])
    llm.classify("App ist langsam")
    est = llm.estimate_usd(["App ist langsam", "Neue Bewertung"])
    assert est["uncached"] == 1
    assert 0 < est["expected_usd"] <= est["worst_case_usd"]


def test_config_reports_what_the_run_did(tmp_path):
    llm = classifier(tmp_path, [["performance"]])
    llm.classify("App ist langsam")
    llm.classify("App ist langsam")
    assert llm.config["api_calls"] == 1
    assert llm.config["cache_hits"] == 1
