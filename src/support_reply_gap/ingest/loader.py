"""Writes to Postgres: schema setup now, review upserts in the next step."""

from importlib import resources

from support_reply_gap.apps import APPS
from support_reply_gap.db import get_connection

SEED_APPS_SQL = """
insert into apps (app_key, package_id, label, category)
values (%s, %s, %s, %s)
on conflict (app_key) do update set
    package_id = excluded.package_id,
    label      = excluded.label,
    category   = excluded.category
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
    """Insert or refresh the three rows in `apps`.

    Reviews reference this table by foreign key, so it has to be populated
    before any review can be inserted.
    """
    rows = [(a.key, a.package_id, a.label, a.category) for a in APPS]
    with get_connection() as conn:
        conn.cursor().executemany(SEED_APPS_SQL, rows)
        conn.commit()