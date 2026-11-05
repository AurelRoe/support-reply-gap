"""Reading reviews out of Postgres into pandas.

Queries the `reviews_analysis` view rather than the `reviews` table, so every
caller gets `reply_latency_hours` already nulled out for edited reviews instead
of having to remember that rule.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from support_reply_gap.db import get_connection

# The ::float8 cast matters: Postgres returns numeric as Python Decimal, which
# pandas stores as slow object-dtype rather than as a float column.
BASE_SQL = """
select
    review_id,
    app_key,
    score,
    content,
    posted_at,
    reply_content,
    replied_at,
    thumbs_up,
    app_version,
    scraped_at,
    likely_edited,
    reply_latency_hours::float8 as reply_latency_hours,
    (reply_content is not null) as has_reply
from reviews_analysis
"""


def load_reviews(
    since: date | None = None,
    apps: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load reviews, optionally filtered by date and app."""
    sql = BASE_SQL
    clauses: list[str] = []
    params: list = []

    if since is not None:
        clauses.append("posted_at >= %s")
        params.append(since)
    if apps:
        clauses.append("app_key = any(%s)")
        params.append(list(apps))

    if clauses:
        sql += " where " + " and ".join(clauses)
    sql += " order by app_key, posted_at"

    with get_connection() as conn:
        cursor = conn.execute(sql, params or None)
        columns = [col.name for col in cursor.description]
        rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=columns)