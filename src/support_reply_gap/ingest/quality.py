"""Data quality checks, run after every ingest.

The database already rejects what must never be true — a score outside 1-5, a
duplicate review id, a review for an unknown app. These checks cover the next
layer: data that is structurally valid but tells you something is wrong, such
as a reply timestamped before the review it answers.

Some checks here duplicate a database constraint on purpose. They cost one
query each and they catch the case where a constraint gets dropped or data
arrives by another path.
"""

from __future__ import annotations

from dataclasses import dataclass

from support_reply_gap.db import get_connection

OK = "ok"
WARN = "warn"
FAIL = "fail"

# An app with fewer rows than this almost certainly means a failed or partial
# fetch rather than a genuinely quiet app.
MIN_REVIEWS_PER_APP = 500


@dataclass(frozen=True)
class CheckResult:
    name: str
    value: float
    status: str
    detail: str


def _scalar(conn, sql: str) -> int:
    return conn.execute(sql).fetchone()[0]


def run_checks() -> list[CheckResult]:
    """Run every check and return the results. Does not raise."""
    results: list[CheckResult] = []

    with get_connection() as conn:
        total = _scalar(conn, "select count(*) from reviews")
        results.append(
            CheckResult(
                "total_reviews",
                total,
                OK if total > 0 else FAIL,
                "rows in the reviews table",
            )
        )
        if total == 0:
            return results

        # --- invariants the database already enforces -----------------------

        nulls = _scalar(conn, "select count(*) from reviews where review_id is null")
        results.append(
            CheckResult("null_review_id", nulls, OK if nulls == 0 else FAIL, "must be zero")
        )

        dupes = _scalar(
            conn, "select count(*) - count(distinct review_id) from reviews"
        )
        results.append(
            CheckResult("duplicate_review_id", dupes, OK if dupes == 0 else FAIL, "must be zero")
        )

        bad_score = _scalar(conn, "select count(*) from reviews where score not between 1 and 5")
        results.append(
            CheckResult("score_out_of_range", bad_score, OK if bad_score == 0 else FAIL, "must be zero")
        )

        # --- checks that can genuinely fail ---------------------------------

        future = _scalar(conn, "select count(*) from reviews where posted_at > now()")
        results.append(
            CheckResult(
                "posted_in_future",
                future,
                OK if future == 0 else FAIL,
                "a review dated ahead of now means a timezone bug",
            )
        )

        # Beyond a minute: the user edited the review after the reply. A known
        # property of the source data, measured and reported, not a defect.
        edited = _scalar(
            conn,
            "select count(*) from reviews where replied_at < posted_at - interval '1 minute'",
        )
        results.append(
            CheckResult(
                "replies_before_review_edited",
                edited,
                OK if edited == 0 else WARN,
                f"{edited / total:.2%} — review edited after the reply; latency excluded",
            )
        )

        # Within a minute: would mean a timezone or rounding bug, not an edit.
        # Currently zero, which is what makes the split above trustworthy.
        skew = _scalar(
            conn,
            "select count(*) from reviews "
            "where replied_at < posted_at "
            "and replied_at >= posted_at - interval '1 minute'",
        )
        results.append(
            CheckResult(
                "replies_before_review_skew",
                skew,
                OK if skew == 0 else FAIL,
                "sub-minute negatives indicate a timezone bug, not an edit",
            )
        )

        reply_no_timestamp = _scalar(
            conn,
            "select count(*) from reviews "
            "where reply_content is not null and replied_at is null",
        )
        results.append(
            CheckResult(
                "reply_without_timestamp",
                reply_no_timestamp,
                OK if reply_no_timestamp == 0 else WARN,
                "these rows count toward reply RATE but not toward latency",
            )
        )

        empty = _scalar(
            conn, "select count(*) from reviews where content is null or btrim(content) = ''"
        )
        results.append(
            CheckResult(
                "empty_content",
                empty,
                OK,
                f"{empty / total:.1%} are rating-only, not classifiable — informational",
            )
        )

        # --- per-app coverage -----------------------------------------------

        per_app = conn.execute(
            "select app_key, count(*) from reviews group by app_key order by app_key"
        ).fetchall()
        for app_key, count in per_app:
            results.append(
                CheckResult(
                    f"rows_{app_key}",
                    count,
                    OK if count >= MIN_REVIEWS_PER_APP else FAIL,
                    f"expected at least {MIN_REVIEWS_PER_APP}",
                )
            )

    return results


def report(results: list[CheckResult]) -> bool:
    """Print the results. Returns True if nothing failed."""
    width = max(len(r.name) for r in results)
    print("\nData quality")
    print("-" * (width + 40))
    for r in results:
        marker = {OK: "  ", WARN: "! ", FAIL: "X "}[r.status]
        print(f"{marker}{r.name.ljust(width)}  {r.value:>8,}  {r.detail}")
    print("-" * (width + 40))

    failures = [r for r in results if r.status == FAIL]
    warnings = [r for r in results if r.status == WARN]
    print(f"{len(failures)} failed, {len(warnings)} warnings\n")
    return not failures