"""HTTP API: classify one German app review into complaint labels.

    uvicorn support_reply_gap.api.main:app --reload   # local, then open /docs

The C6 router: the baseline labels every review, and reviews it is unsure about go
to the LLM. The baseline is loaded from the file written by
`python -m support_reply_gap.models.artifact`. Review text comes in the request;
the service has no database.

The LLM tier runs only when ANTHROPIC_API_KEY is set, and stops for good once the
service has spent LLM_BUDGET_USD (default 0.50) since it started. Without it, or
when a call fails, the baseline's answer is returned and the response says why.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import anthropic
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from support_reply_gap.labeling.taxonomy import LABEL_NAMES
from support_reply_gap.models.artifact import load
from support_reply_gap.models.baseline import predict_proba, to_labels
from support_reply_gap.models.llm import BudgetExceeded, LLMClassifier
from support_reply_gap.models.router import margins

logger = logging.getLogger(__name__)

# Loaded once, when the server starts, and shared by every request: loading takes
# about a second, classifying one review a few milliseconds. A missing or
# incompatible file stops the server at startup instead of failing the first request.
BUNDLE = load()

# Long enough for any real review (the longest of 21,197 collected has 680 characters),
# short enough that nobody can send a book.
MAX_TEXT_LENGTH = 5000

DEFAULT_BUDGET_USD = 0.50


def make_llm() -> LLMClassifier | None:
    """The LLM tier, or None when no API key is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    budget = float(os.environ.get("LLM_BUDGET_USD", DEFAULT_BUDGET_USD))
    # The answer cache goes to a temporary folder: in the container the app folder
    # belongs to root, not to the user the service runs as, and a cache that
    # disappears on restart is fine.
    cache_dir = Path(tempfile.gettempdir()) / "llm_cache"
    return LLMClassifier(max_usd=budget, cache_dir=cache_dir)


# Created once, like the model. Its spending counter lives as long as the process, so
# the budget resets whenever the service restarts. The Anthropic console's spend
# limit is the real cap; this one stops a single run of the service from draining it.
LLM = make_llm()

FallbackReason = Literal["llm_not_configured", "budget_spent", "llm_error"]

app = FastAPI(
    title="Support Reply Gap",
    description=(
        "Complaint classification for German fintech app reviews. Labels are model "
        "predictions; see the eval table in the repository for how accurate they are."
    ),
    version="0.3.0",
)


class ClassifyRequest(BaseModel):
    """What a client sends. FastAPI rejects anything else with status 422."""

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="One Google Play review, in German.",
        examples=["Seit drei Tagen kann ich mich nicht einloggen und der Support antwortet nicht."],
    )


class ClassifyResponse(BaseModel):
    """What the service answers."""

    labels: list[str] = Field(description="Predicted labels, in taxonomy order.")
    tier: Literal["baseline", "llm"] = Field(description="Which model produced the labels.")
    margin: float = Field(
        description=(
            "Distance of the baseline's closest call to its decision threshold. Small means unsure."
        )
    )
    uncertain: bool = Field(
        description="True when the margin is below the router's cut-off for escalation."
    )
    fallback_reason: FallbackReason | None = Field(
        default=None,
        description=(
            "Set when the baseline was unsure but the LLM was not used: no API key "
            "configured, the service's budget is spent, or the call failed."
        ),
    )


def classify_text(text: str) -> ClassifyResponse:
    """The router, kept apart from HTTP so it reads and tests on its own."""
    proba = predict_proba(BUNDLE["model"], pd.Series([text]))
    predicted = to_labels(proba, BUNDLE["threshold"]).iloc[0]
    margin = float(margins(proba, BUNDLE["threshold"])[0])
    baseline = ClassifyResponse(
        labels=[name for name in LABEL_NAMES if predicted[name] == 1],
        tier="baseline",
        margin=round(margin, 4),
        uncertain=margin < BUNDLE["margin_cut"],
    )
    if not baseline.uncertain:
        return baseline
    if LLM is None:
        return baseline.model_copy(update={"fallback_reason": "llm_not_configured"})

    try:
        labels, _cost = LLM.classify(text)
    except BudgetExceeded:
        return baseline.model_copy(update={"fallback_reason": "budget_spent"})
    except anthropic.APIError:
        # Network trouble, rate limit, outage: answer from the baseline rather than fail.
        logger.exception("LLM call failed; answering from the baseline")
        return baseline.model_copy(update={"fallback_reason": "llm_error"})
    if not labels:
        # LLMClassifier returns no labels when the answer was refused or cut off.
        return baseline.model_copy(update={"fallback_reason": "llm_error"})

    return baseline.model_copy(
        update={"labels": [name for name in LABEL_NAMES if name in labels], "tier": "llm"}
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the bare URL to the interactive docs instead of a 404."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Hosting platforms poll this to decide if the container is up."""
    return {"status": "ok"}


@app.post("/classify")
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify one review."""
    return classify_text(request.text)
