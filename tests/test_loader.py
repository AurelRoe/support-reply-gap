from datetime import UTC, datetime, timedelta

import pytest

from support_reply_gap.db import get_connection
from support_reply_gap.ingest.loader import apply_schema, seed_apps, upsert_reviews
from support_reply_gap.ingest.scraper import ReviewRecord

TEST_ID = "pytest-synthetic-review-1"


def make_record(reply: str | None = None) -> ReviewRecord:
    return ReviewRecord(
        review_id=TEST_ID,
        app_key="n26",
        score=2,
        content="Testinhalt",
        posted_at=datetime.now(UTC) - timedelta(days=30),
        reply_content=reply,
        replied_at=datetime.now(UTC) - timedelta(days=29) if reply else None,
        thumbs_up=0,
        app_version="1.0.0",
    )


@pytest.fixture
def clean_row():
    """Remove the synthetic row before and after, so the test is repeatable."""
    apply_schema()
    seed_apps()
    with get_connection() as conn:
        conn.execute("delete from reviews where review_id = %s", (TEST_ID,))
        conn.commit()
    yield
    with get_connection() as conn:
        conn.execute("delete from reviews where review_id = %s", (TEST_ID,))
        conn.commit()


@pytest.mark.db
def test_upsert_is_idempotent(clean_row):
    """Writing the same review twice must not create two rows."""
    upsert_reviews([make_record()])
    upsert_reviews([make_record()])

    with get_connection() as conn:
        count = conn.execute(
            "select count(*) from reviews where review_id = %s", (TEST_ID,)
        ).fetchone()[0]
    assert count == 1


@pytest.mark.db
def test_upsert_picks_up_a_later_reply(clean_row):
    """A reply that appears after the first scrape must land on the existing row."""
    upsert_reviews([make_record(reply=None)])
    upsert_reviews([make_record(reply="Danke für dein Feedback.")])

    with get_connection() as conn:
        reply = conn.execute(
            "select reply_content from reviews where review_id = %s", (TEST_ID,)
        ).fetchone()[0]
    assert reply == "Danke für dein Feedback."