"""Fetches Google Play reviews for one app over a fixed time window.

Deliberately knows nothing about the database: it returns records, and the
loader writes them. That split is what makes this testable without Postgres
running, and what lets the loader be tested without hitting the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from google_play_scraper import Sort, reviews

LANG = "de"
COUNTRY = "de"

PAGE_SIZE = 200  # reviews per request
SLEEP_SECONDS = 1.0  # between requests, to stay polite
MAX_PAGES = 300  # safety stop: 300 * 200 = 60k reviews per app

# All three apps are pulled over the same window so their numbers are
# comparable. 730 days rather than "24 months" because month lengths vary and
# the exact boundary does not matter — only that it is identical per app.
DEFAULT_WINDOW_DAYS = 730


@dataclass(frozen=True)
class ReviewRecord:
    """One review, in the shape the `reviews` table expects."""

    review_id: str
    app_key: str
    score: int
    content: str | None
    posted_at: datetime
    reply_content: str | None
    replied_at: datetime | None
    thumbs_up: int | None
    app_version: str | None


def window_start(days: int = DEFAULT_WINDOW_DAYS) -> datetime:
    """The oldest review date to collect."""
    return datetime.now(UTC) - timedelta(days=days)


def as_utc(value: datetime | None) -> datetime | None:
    """Attach UTC to a naive timestamp, or convert an aware one.

    The scraper returns naive datetimes that are really UTC. Storing those
    without a timezone would make every reply-latency calculation silently
    wrong by whatever offset the reader assumes.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_record(raw: dict, app_key: str) -> ReviewRecord:
    """Convert one raw review dict into a ReviewRecord.

    `userName` is intentionally dropped: it serves no analytical purpose and
    this repo is public.
    """
    return ReviewRecord(
        review_id=raw["reviewId"],
        app_key=app_key,
        score=int(raw["score"]),
        content=raw.get("content"),
        posted_at=as_utc(raw["at"]),
        reply_content=raw.get("replyContent"),
        replied_at=as_utc(raw.get("repliedAt")),
        thumbs_up=raw.get("thumbsUpCount"),
        app_version=raw.get("reviewCreatedVersion"),
    )


def fetch_reviews_since(
    package_id: str,
    app_key: str,
    since: datetime,
    page_size: int = PAGE_SIZE,
    max_pages: int = MAX_PAGES,
    verbose: bool = True,
) -> list[ReviewRecord]:
    """Fetch reviews newer than `since`, newest first.

    Sorting by NEWEST is what lets this terminate: once a page contains a review
    older than the cutoff, every later page is older still.
    """
    since = as_utc(since)
    collected: list[ReviewRecord] = []
    seen: set[str] = set()
    token = None
    reached_cutoff = False

    for page in range(max_pages):
        if token is None and page == 0:
            batch, token = reviews(
                package_id,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=page_size,
            )
        else:
            # With a continuation token the other arguments are ignored —
            # the token already encodes them.
            batch, token = reviews(package_id, continuation_token=token)

        if not batch:
            break

        for raw in batch:
            record = to_record(raw, app_key)
            if record.posted_at < since:
                reached_cutoff = True
                continue
            # Pagination can overlap when new reviews arrive mid-run.
            if record.review_id in seen:
                continue
            seen.add(record.review_id)
            collected.append(record)

        if verbose:
            oldest = min((r.posted_at for r in collected), default=None)
            stamp = oldest.date() if oldest else "-"
            print(f"    page {page + 1}: {len(collected)} collected, oldest {stamp}")

        if reached_cutoff or token is None:
            break

        time.sleep(SLEEP_SECONDS)
    else:
        print(f"    WARNING: hit max_pages={max_pages} before reaching the cutoff")

    return collected