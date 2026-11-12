"""Writes to Postgres: schema setup, app seeding, and review upserts."""

from __future__ import annotations

from importlib import resources

from support_reply_gap.apps import APPS
from support_reply_gap.db import get_connection
from support_reply_gap.ingest.scraper import ReviewRecord

SEED_APPS_SQL = """
insert into apps (app_key, package_id, label, category)
values (%s, %s, %s, %s)
on conflict (app_key) do update set
    package_id = excluded.package_id,
    label      = excluded.label,
    category   = excluded.category
"""

# Only the mutable columns are updated. posted_at and app_key are facts about
# the review that never change; reply_content and replied_at are the whole
# reason re-running matters, since a reply can arrive after the first scrape.
UPSERT_REVIEW_SQL = """
insert into reviews (
    review_id, app_key, score, content, posted_at,
    reply_content, replied_at, thumbs_up, app_version, scraped_at
)
values (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
on conflict (review_id) do update set
    score         = excluded.score,
    content       = excluded.content,
    reply_content = excluded.reply_content,
    replied_at    = excluded.replied_at,
    thumbs_up     = excluded.thumbs_up,
    app_version   = excluded.app_version,
    scraped_at    = now()
"""


def apply_schema() -> None:
    """Create the tables and indexes if they are not already there.

    Read from the packaged schema.sql rather than from a path on disk, so this
    also works inside the container, where the repo layout does not exist.
    """
    sql = resources.files("support_reply_gap.ingest").joinpath("schema.sql").read_text()
    with get_connection() as conn:
        conn.execute(sql)
        conn.commit()


def seed_apps() -> None:
    """Insert or refresh the rows in `apps`.

    Reviews reference this table by foreign key, so it has to be populated
    before any review can be inserted.
    """
    rows = [(a.key, a.package_id, a.label, a.category) for a in APPS]
    with get_connection() as conn:
        conn.cursor().executemany(SEED_APPS_SQL, rows)
        conn.commit()


def upsert_reviews(records: list[ReviewRecord]) -> int:
    """Insert reviews, updating any that already exist. Returns rows written."""
    if not records:
        return 0

    rows = [
        (
            r.review_id,
            r.app_key,
            r.score,
            r.content,
            r.posted_at,
            r.reply_content,
            r.replied_at,
            r.thumbs_up,
            r.app_version,
        )
        for r in records
    ]

    with get_connection() as conn:
        conn.cursor().executemany(UPSERT_REVIEW_SQL, rows)
        conn.commit()
    return len(rows)


def review_counts() -> dict[str, int]:
    """Rows per app, for reporting after an ingest."""
    with get_connection() as conn:
        result = conn.execute(
            "select app_key, count(*) from reviews group by app_key order by app_key"
        ).fetchall()
    return {key: count for key, count in result}
