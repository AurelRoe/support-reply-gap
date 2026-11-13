"""C5 LLM path: Claude Haiku 5.5 labels one review per call, with structured output.

Cost is the constraint, so three guards sit in front of every API call:
- a hard budget per run: a call that could push spending past it is not made;
- a local answer cache keyed by model, prompt and review text, so a re-run of the
  same prompt costs nothing and reports the original cost and latency;
- prompt caching on the system prompt (the taxonomy), which every call shares.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from support_reply_gap.labeling.gold import ROOT
from support_reply_gap.labeling.taxonomy import EXCLUSIVE_LABELS, LABEL_NAMES, render_markdown

MODEL = "claude-haiku-5-5"
# Low effort: a short classification gains little from long thinking, and thinking
# is billed as output. Thinking stays adaptive, so the model can still think briefly.
EFFORT = "low"
# Room for brief thinking plus the JSON answer; also the per-call worst case for the budget.
MAX_TOKENS = 1024
# Assumed output per call for the up-front estimate; the smoke test checks it.
TYPICAL_OUTPUT_TOKENS = 150

# USD per million tokens, Claude Haiku 5.5, prompts up to 100K tokens (Nov 2026 price list).
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01}

PROMPT_PATH = ROOT / "prompts/classify_v1.md"
CACHE_DIR = ROOT / "data/llm_cache"  # gitignored; labels, cost and latency only, no text

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": list(LABEL_NAMES)},
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


class BudgetExceeded(RuntimeError):
    """Raised before a call that could take spending past the run's budget."""


def system_prompt(path: Path = PROMPT_PATH) -> str:
    return path.read_text().replace("{taxonomy}", render_markdown())


def cost_usd(usage) -> float:
    """Price one response from its usage block."""
    tokens = {
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    return sum(tokens[k] * PRICE_PER_MTOK[k] for k in tokens) / 1_000_000


def apply_label_rules(labels: list[str]) -> set[str]:
    """The gold set's rules, as for the baseline: exclusive labels never with others."""
    chosen = set(labels)
    others = chosen - EXCLUSIVE_LABELS
    if others:
        return others
    # Both exclusive labels and nothing else: no probabilities to break the tie,
    # so keep the one the model listed first.
    return {labels[0]} if labels else set()


class LLMClassifier:
    """The LLM behind the harness interface: one review in, labels and cost out."""

    name = "llm"

    def __init__(
        self,
        client=None,
        max_usd: float = 0.10,
        prompt_path: Path = PROMPT_PATH,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.max_usd = max_usd
        self.system = system_prompt(prompt_path)
        self.prompt_sha = hashlib.sha256(self.system.encode()).hexdigest()[:12]
        self.cache_dir = cache_dir
        self.prompt_name = prompt_path.name
        self.spent_usd = 0.0  # new money this run; cache hits cost nothing
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        # Set on every call so the harness reports the original latency for cache hits.
        self.last_latency_s: float | None = None

    @property
    def config(self) -> dict:
        """Settings plus what this run did, so a result file shows how much was new spend."""
        return {
            "model": MODEL,
            "effort": EFFORT,
            "max_tokens": MAX_TOKENS,
            "prompt": self.prompt_name,
            "prompt_sha": self.prompt_sha,
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "new_spend_usd": round(self.spent_usd, 6),
        }

    def estimate_usd(self, texts) -> dict:
        """Expected and worst-case cost of the reviews not yet cached, before any spending.

        Token counting is free; it is used once, for the system prompt every call shares.
        """
        uncached = [t for t in texts if not self._cache_path(t).exists()]
        if not uncached:
            return {"uncached": 0, "expected_usd": 0.0, "worst_case_usd": 0.0}
        system_tokens = self.client.messages.count_tokens(
            model=MODEL, system=self.system, messages=[{"role": "user", "content": "."}]
        ).input_tokens
        review_tokens = sum(len(t) for t in uncached) / 3
        p = PRICE_PER_MTOK
        expected = (
            system_tokens * p["cache_write"]  # first call writes the prompt cache
            + system_tokens * (len(uncached) - 1) * p["cache_read"]
            + review_tokens * p["input"]
            + len(uncached) * TYPICAL_OUTPUT_TOKENS * p["output"]
        ) / 1e6
        return {
            "uncached": len(uncached),
            "system_tokens": system_tokens,
            "expected_usd": expected,
            "worst_case_usd": sum(self.worst_case_call_usd(t) for t in uncached),
        }

    def _cache_path(self, text: str) -> Path:
        key = f"{MODEL}|{EFFORT}|{self.prompt_sha}|{text}"
        return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"

    def worst_case_call_usd(self, text: str) -> float:
        """Upper bound for one call: whole prompt at the full input rate, maximal output."""
        approx_input = (len(self.system) + len(text)) / 3  # generous chars-per-token guess
        return (
            approx_input * PRICE_PER_MTOK["input"] + MAX_TOKENS * PRICE_PER_MTOK["output"]
        ) / 1e6

    def classify(self, text: str) -> tuple[set[str], float]:
        path = self._cache_path(text)
        if path.exists():
            hit = json.loads(path.read_text())
            self.cache_hits += 1
            self.last_latency_s = hit["latency_s"]
            return apply_label_rules(hit["labels"]), hit["cost_usd"]

        if self.spent_usd + self.worst_case_call_usd(text) > self.max_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} of ${self.max_usd:.2f}; the next call could exceed it"
            )

        start = time.perf_counter()
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # One cache breakpoint on the shared system prompt: later calls read it at 0.1x.
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text}],
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        )
        latency = time.perf_counter() - start
        cost = cost_usd(response.usage)
        self.spent_usd += cost
        self.calls += 1
        self.last_latency_s = latency

        labels = self._parse(response)
        if labels is None:
            # Refusal or truncated answer: score it as no labels, never as a guess,
            # and don't cache it, so a later run can try again.
            self.failures += 1
            return set(), cost

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"labels": labels, "cost_usd": cost, "latency_s": latency, "model": MODEL},
                indent=2,
            )
        )
        return apply_label_rules(labels), cost

    @staticmethod
    def _parse(response) -> list[str] | None:
        if response.stop_reason != "end_turn":
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            return None
        return [label for label in json.loads(text)["labels"] if label in LABEL_NAMES]
