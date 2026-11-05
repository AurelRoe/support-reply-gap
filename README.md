# support-reply-gap

![CI](https://github.com/AurelRoe/support-reply-gap/actions/workflows/ci.yml/badge.svg)

Which complaint types in German fintech app reviews get a developer reply,
and how fast? This repo classifies Google Play reviews into complaint
categories and joins them to reply data. The result is a descriptive table:
reply rate and median reply latency per complaint type.

**Status:** Phase A (skeleton) is done. Phase B is under way: the three apps
are chosen (see [App selection](#app-selection-stage-b1)) and their reviews
are ingested (see [Data](#data)). There are no models or findings yet.
The full plan is in [docs/build-order.md](docs/build-order.md).

## Approach

1. Scrape Google Play reviews for three German apps from one product category
   (neobanks *or* brokers, not a mix, because complaint base rates differ).
   The apps are chosen by measured reply rate, not picked up front.
2. Hand-label ~400 reviews into a multi-label complaint taxonomy. This gold
   set is the yardstick for every model.
3. Compare two classifiers against it: a cheap TF-IDF + logistic regression
   baseline and an LLM with structured output. A confidence router sends
   uncertain reviews from the baseline to the LLM.
4. Join the predicted labels to reply data and report reply rate and latency
   per complaint type.

Findings will be descriptive only. Companies choose which reviews to answer
and that choice is not observed, so the data cannot show *why* a complaint
type gets more or fewer replies.

## App selection (stage B1)

Probed on 2026-11-03 with [scripts/probe_apps.py](scripts/probe_apps.py):
the newest 2,000 reviews per app from the German Play Store. Reply rates leave
out reviews younger than 14 days, which may not have been answered yet.

| App | Reviews/day | Reply rate | Reply rate, 1–2★ | Median reply time | German reviews\* |
|---|---:|---:|---:|---:|---:|
| N26 | 4.6 | 97.9% | 99.7% | 16.5 h | 68% |
| Revolut | 17.4 | 12.5% | 72.3% | 0.4 h | 43% |
| C24 Bank | 3.3 | 14.9% | 41.9% | 75.4 h | 70% |
| bunq | 1.4 | 99.7% | 99.9% | 58.6 h | 62% |
| Vivid Money | 1.1 | 32.6% | 48.2% | 59.8 h | 73% |

\* Rough heuristic: share of reviews with at least two common German words or
umlauts.

Neobanks were chosen over brokers because broker review volume is concentrated
in Trade Republic, which replies to 2.4% of its 1–2 star reviews. Food delivery
was rejected because only one of six apps (Lieferando) replies to most of its
reviews. bunq and Vivid Money were excluded because they get fewer than 1.5
reviews per day. That leaves **N26, Revolut and C24 Bank**.

**To revisit:** Revolut's median reply time is 0.4 hours, against 16–75 hours
for the other neobanks, while it replies to only 12.5% of reviews overall. Which
complaint types make up that fast-answered subset is a question for the
reply-gap analysis, not a finding yet.

## Data

Collected on 2026-11-03 with `python -m support_reply_gap.ingest`: every
review posted in the previous two years (2024-11-03 to 2026-11-02), 21,197 in
total.

| Quarter | C24 reviews | C24 replied | N26 reviews | N26 replied | Revolut reviews | Revolut replied |
|---|---:|---:|---:|---:|---:|---:|
| 2024Q4\* | 386 | 1.6% | 346 | 35.3% | 1,374 | 9.7% |
| 2025Q1 | 754 | 2.5% | 319 | 33.5% | 2,166 | 8.1% |
| 2025Q2 | 353 | 2.3% | 342 | 80.1% | 2,106 | 7.4% |
| 2025Q3 | 293 | 1.4% | 449 | 93.1% | 2,060 | 13.1% |
| 2025Q4 | 274 | 2.6% | 639 | 100.0% | 1,529 | 14.2% |
| 2026Q1 | 230 | 14.8% | 359 | 100.0% | 2,228 | 7.8% |
| **2026Q2** | 251 | 29.5% | 360 | 93.1% | 1,404 | 11.5% |
| **2026Q3** | 222 | 49.1% | 355 | 96.3% | 1,599 | 12.3% |
| **2026Q4**\* | 118 | 39.0% | 120 | 95.0% | 561 | 13.9% |

\* Partial quarter. The newest reviews in 2026Q4 may not have been answered
yet, so its reply rates are likely understated.

**Analysis window: 2026-04-01 onward (bold rows).** Two of the three apps
started replying regularly during the collection period: N26 in 2025Q2 and C24
in 2026Q1. Before those points, a missing reply reflects the absence of a reply
programme rather than anything about the review. Starting the window in 2026Q2
means all three apps are actively replying throughout. That leaves 591 C24, 835
N26 and 3,564 Revolut reviews. C24's reply rate still varies within the window
(29.5% to 49.1%), so time is part of what any C24 figure reflects.

**Edited reviews.** 512 reviews (2.4% of all reviews, 11.2% of replied ones)
have a reply timestamped earlier than the review itself. The user edited the
review after the reply, and Google keeps only the edit time, so the original
post time is lost. These reviews count toward reply rate but are excluded from
latency.

**Text reviews only.** The Play Store review endpoint returns only reviews with
text (none of the 21,197 are empty). Counts here are therefore not comparable
to an app's total rating count on the store, which includes star-only ratings.

**Observed, not analysed: reply programmes switching on.** The quarterly table
contains a sharper question than the static comparison: *when an app starts
replying, which complaint types are answered first?* C24's reply rate rises
from 2.6% to 49.1% across four quarters, with the same app and the same
complaint taxonomy before and after. This is outside the current scope and stays
descriptive, like everything else here. It is the fallback if the main
finding (stage C2) comes out thin.

## What works now

- **Package:** `support_reply_gap` installs with `pip install -e`.
- **Database:** Postgres 16 in Docker Compose, on host port 5433.
- **CI:** GitHub Actions runs `ruff` and `pytest` on every pull request and
  push to `main`. Tests that need a live database are skipped there.
- **API:** a FastAPI app with one `/health` endpoint, containerised and
  deployed on Render at
  [support-reply-gap.onrender.com/health](https://support-reply-gap.onrender.com/health).
  It is a stub; the real `/classify` endpoint replaces it in Phase D.

## Running it locally

Needs Python 3.11+ and Docker.

```bash
# Install the package and dev tools into a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Start Postgres (fill in .env first)
cp .env.example .env
docker compose up -d

# Run all tests, including the database ones
pytest

# Run the API on http://localhost:8000/health
docker build -t support-reply-gap .
docker run -p 8000:8000 support-reply-gap
```

## Repo layout

```
src/support_reply_gap/   the package (api/, ingest/, db.py so far)
tests/                   pytest; database tests marked @pytest.mark.db
notebooks/               exploration only, never imported by src/
scripts/                 standalone utilities
docs/build-order.md      stage-by-stage plan with a definition of done each
```
