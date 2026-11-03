"""One command that runs the whole ingest.

    python -m support_reply_gap.ingest

Safe to re-run: the schema uses IF NOT EXISTS and reviews are upserted by their
Google review id, so a second run updates rows rather than duplicating them.
"""

from __future__ import annotations

import argparse
import sys

from support_reply_gap.apps import APPS, by_key
from support_reply_gap.ingest.loader import (
    apply_schema,
    review_counts,
    seed_apps,
    upsert_reviews,
)
from support_reply_gap.ingest.quality import report, run_checks
from support_reply_gap.ingest.scraper import (
    DEFAULT_WINDOW_DAYS,
    fetch_reviews_since,
    window_start,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest Google Play reviews into Postgres.")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"how far back to collect (default {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--apps",
        nargs="*",
        default=[a.key for a in APPS],
        help="app keys to ingest (default: all)",
    )
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="skip fetching, just re-run the data quality checks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any check fails (use this in CI)",
    )
    args = parser.parse_args()

    print("Applying schema ...")
    apply_schema()
    seed_apps()

    if not args.checks_only:
        since = window_start(args.days)
        print(f"Collecting reviews posted since {since.date()}\n")

        for key in args.apps:
            app = by_key(key)
            print(f"{app.label} ({app.package_id})")
            records = fetch_reviews_since(app.package_id, app.key, since)
            written = upsert_reviews(records)
            replied = sum(1 for r in records if r.reply_content)
            print(f"    wrote {written:,} rows, {replied:,} with a reply\n")

    print("Rows per app:")
    for key, count in review_counts().items():
        print(f"    {key}: {count:,}")

    passed = report(run_checks())

    if args.strict and not passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())