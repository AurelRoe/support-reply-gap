"""Stage B1: screen candidate apps before committing to three.

Measures what the project depends on and cannot assume: how often developers
reply, how fast, how templated the replies are, and whether the reviews are
German. Writes a summary table plus the raw sample so the numbers in the README
can be traced back.

Apps are addressed by package id rather than by search, because the library's
search endpoint breaks whenever Google changes its page structure. Each id is
verified against the app's real title on every run — see verify_package().

To find or correct a package id: open the app on play.google.com and read the
`id=` parameter out of the URL, e.g.
    https://play.google.com/store/apps/details?id=de.number26.android
                                                  ^^^^^^^^^^^^^^^^^^^

Run from the repo root:
    python scripts/probe_apps.py
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews
from google_play_scraper import app as gp_app

# Candidates to screen, grouped by product category. Both groups are screened,
# then ONE group is chosen — complaint base rates differ between neobanks and
# food delivery, so a mixed set would confound the analysis.
#
# These package ids are UNVERIFIED starting guesses. The script prints the real
# app title for each one; check that every title matches its label before
# trusting any row, and correct the id from the Play Store URL if it does not.
CANDIDATES: list[tuple[str, str, str]] = [
    ("N26", "de.number26.android", "neobank"),
    ("Revolut", "com.revolut.revolut", "neobank"),
    ("bunq", "com.bunq.android", "neobank"),
    ("Vivid Money", "vivid.money", "neobank"),
    ("C24 Bank", "de.c24.bankapp", "neobank"),
    ("Trade Republic", "de.traderepublic.app", "broker"),
    ("Scalable Capital", "com.scalablecapital.app", "broker"),
    ("flatex", "de.xcom.flatexde", "broker"),
    ("justTRADE", "de.justtrade.app", "broker"),
    ("Bitpanda", "com.bitpanda.bitpanda", "broker"),
    ("Lieferando", "com.yopeso.lieferando", "food delivery"),
    ("Wolt", "com.wolt.android", "food delivery"),
    ("Uber Eats", "com.ubercab.eats", "food delivery"),
    ("deliveroo", "com.deliveroo.orderapp", "food delivery"),
    ("doordash", "com.dd.doordash", "food delivery"),
    ("foodpanda", "com.global.foodpanda.android", "food delivery"),
]

SAMPLE_SIZE = 2000
LANG = "de"
COUNTRY = "de"

# Reviews newer than this may simply not have been replied to YET. Counting them
# against the reply rate biases every app downward, and worst for the apps that
# reply slowly — i.e. it distorts exactly the variable being measured.
REPLY_GRACE_DAYS = 14

SUMMARY_PATH = Path("reports/app_probe.csv")
RAW_PATH = Path("data/raw/probe_reviews.csv")

# Crude language check: common German function words plus umlauts. Good enough
# to spot an app whose reviews are mostly English, which would change the model.
GERMAN_MARKERS = re.compile(
    r"\b(und|nicht|ich|die|der|das|ist|mit|für|aber|auf|kann|sehr|mein|"
    r"wurde|keine|schon|immer|leider|funktioniert)\b|[äöüß]",
    re.IGNORECASE,
)


def looks_german(text: str) -> bool:
    """True if the text has at least two German markers."""
    return bool(text) and len(GERMAN_MARKERS.findall(text)) >= 2


def verify_package(package: str) -> tuple[str | None, int | None]:
    """Return the app's real title and review count, or (None, None) if unreachable.

    Printing the title is the guard against a wrong package id: a wrong-but-valid
    id returns another app's reviews with no error at all, which is far worse
    than a crash.
    """
    try:
        meta = gp_app(package, lang=LANG, country=COUNTRY)
    except Exception as exc:  # noqa: BLE001
        print(f"    lookup failed: {type(exc).__name__}: {exc}")
        return None, None
    return meta.get("title"), meta.get("reviews")


def probe(label: str, package: str, group: str) -> tuple[dict, pd.DataFrame]:
    print(f"\n{label} ({group})  ->  {package}")

    title, total_reviews = verify_package(package)
    if title is None:
        print("    SKIPPED — correct this package id from the Play Store URL")
        return {"app": label, "group": group, "status": "bad package id"}, pd.DataFrame()
    print(f"    store title: {title!r}")

    try:
        batch, _ = reviews(
            package,
            lang=LANG,
            country=COUNTRY,
            count=SAMPLE_SIZE,
            sort=Sort.NEWEST,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    review fetch failed: {type(exc).__name__}: {exc}")
        return {"app": label, "group": group, "status": "fetch error"}, pd.DataFrame()

    if not batch:
        return {"app": label, "group": group, "status": "no reviews"}, pd.DataFrame()

    df = pd.DataFrame(batch)
    df["app"] = label
    df["group"] = group
    df["package"] = package
    df["at"] = pd.to_datetime(df["at"])
    df["repliedAt"] = pd.to_datetime(df.get("repliedAt"))

    span_days = max((df["at"].max() - df["at"].min()).days, 1)

    # Only reviews old enough to have plausibly been answered count toward the
    # reply rate. Fall back to the full sample if too few are mature.
    cutoff = datetime.now() - timedelta(days=REPLY_GRACE_DAYS)
    mature = df[df["at"] < cutoff]
    base = mature if len(mature) >= 50 else df

    def has_reply(frame: pd.DataFrame) -> pd.Series:
        return frame["replyContent"].notna() & (frame["replyContent"].str.strip() != "")

    replied = base[has_reply(base)]
    low = base[base["score"] <= 2]
    low_replied = low[has_reply(low)]

    if len(replied) and replied["repliedAt"].notna().any():
        latency_h = (replied["repliedAt"] - replied["at"]).dt.total_seconds() / 3600
        latency_h = latency_h[latency_h >= 0]
        median_h = round(latency_h.median(), 1)
        p90_h = round(latency_h.quantile(0.90), 1)
    else:
        median_h = p90_h = None

    # Distinct first-80-characters over total replies. Near 0.1 means one canned
    # template; reply TEXT is then useless as a signal, though timing still works.
    distinct_ratio = (
        round(replied["replyContent"].str.strip().str[:80].nunique() / len(replied), 2)
        if len(replied)
        else None
    )

    row = {
        "app": label,
        "group": group,
        "status": "ok",
        "package": package,
        "store_title": title,
        "total_reviews": total_reviews,
        "sample_n": len(df),
        "span_days": span_days,
        "reviews_per_day": round(len(df) / span_days, 1),
        "mean_score": round(df["score"].mean(), 2),
        "pct_german": round(df["content"].fillna("").map(looks_german).mean() * 100, 1),
        "n_mature": len(base),
        "reply_rate_pct": round(len(replied) / len(base) * 100, 1),
        "n_low_star": len(low),
        "reply_rate_low_star_pct": (
            round(len(low_replied) / len(low) * 100, 1) if len(low) else None
        ),
        "reply_median_h": median_h,
        "reply_p90_h": p90_h,
        "reply_distinct_ratio": distinct_ratio,
    }

    print(
        f"    {len(df)} reviews over {span_days}d | "
        f"replies {row['reply_rate_pct']}% overall, "
        f"{row['reply_rate_low_star_pct']}% on 1-2 star (n={len(low)}) | "
        f"distinct {distinct_ratio} | german {row['pct_german']}%"
    )
    return row, df


def main() -> None:
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []

    for label, package, group in CANDIDATES:
        row, df = probe(label, package, group)
        rows.append(row)
        if len(df):
            frames.append(df)
        time.sleep(1.5)  # don't hammer the endpoint

    summary = pd.DataFrame(rows)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(RAW_PATH, index=False)

    display_cols = [
        "app",
        "group",
        "status",
        "store_title",
        "reviews_per_day",
        "pct_german",
        "reply_rate_pct",
        "reply_rate_low_star_pct",
        "n_low_star",
        "reply_median_h",
        "reply_distinct_ratio",
    ]
    print("\n" + "=" * 110)
    for group in ("neobank", "food delivery"):
        part = summary[summary["group"] == group]
        if len(part):
            print(f"\n{group.upper()}")
            print(part[[c for c in display_cols if c in part.columns]].to_string(index=False))
    print("\n" + "=" * 110)

    bad = summary[summary["status"] != "ok"]
    if len(bad):
        print("\nNeeds a corrected package id:")
        for _, r in bad.iterrows():
            print(f"  {r['app']}: {r['status']}")

    print(f"\nwrote {SUMMARY_PATH} and {RAW_PATH}")


if __name__ == "__main__":
    main()