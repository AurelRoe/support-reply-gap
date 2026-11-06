# support-reply-gap

![CI](https://github.com/AurelRoe/support-reply-gap/actions/workflows/ci.yml/badge.svg)

Which complaint types in German fintech app reviews get a developer reply,
and how fast? This repo classifies Google Play reviews into complaint
categories and joins them to reply data. The result is a descriptive table:
reply rate and median reply latency per complaint type.

**Status:** Phase A (skeleton) is done. Phase B is nearly done: the three apps
are chosen (see [App selection](#app-selection-stage-b1)), their reviews are
ingested (see [Data](#data)), the label taxonomy is set (see
[Taxonomy](#taxonomy-stage-b4)) and 400 reviews are hand-labelled and split
(see [Gold set](#gold-set-stage-b5)). The self-agreement measurement is still
open. There are no models or findings yet. The full plan is in
[docs/build-order.md](docs/build-order.md).

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

## Taxonomy (stage B4)

Eleven labels, defined with real examples in [docs/taxonomy.md](docs/taxonomy.md).
That file is generated from
[taxonomy.py](src/support_reply_gap/labeling/taxonomy.py), so the definitions
and the code cannot drift apart.

The taxonomy is **multi-label**: a review can carry several labels, because
reviews often complain about two things at once. `no_complaint` and
`complaint_unspecified` are exclusive and never combined with another label.

I read a 150-review development sample against eight draft labels. That took
the set to eleven: `account_restricted`, `performance` and
`complaint_unspecified` were added, and `fees_pricing` was widened to money
the user believes they are owed (rewards, bonuses, referral payouts).
Trading/crypto and in-app advertising were considered as labels and rejected.
Both were too rare, and trading only came up for Revolut. Ads fold into
`feature_gap`; trading complaints into `payments_transfers` (orders not
executed) or `feature_gap` (missing or changed trading features). The 150 development reviews are excluded
from the gold set.

## Gold set (stage B5)

400 reviews, hand-labelled by me, are the yardstick for every model. Only
review IDs and labels are committed
([data/gold/](data/gold/)). Review text stays in a local, gitignored sheet.

**Sampling.** The sample is drawn from the analysis window after the B3
cleaning rules: German or uncertain language, no one-word or emoji-only
reviews, repeated text dropped, at most two reviews per app, day and star
rating. It is weighted toward complaints (1★ 170, 2★ 90, 3★ 80, 4★ 30, 5★ 30)
and spread evenly across the three apps (135 C24, 132 N26, 133 Revolut). A
random sample would mostly contain 5★ praise and few examples of the rarer
complaint types.

**Label counts.** 303 reviews carry one label, 80 two, 13 three and 4 four.

| Label | Reviews | Train | Val | Test |
|---|---:|---:|---:|---:|
| `feature_gap` | 152 | 91 | 31 | 30 |
| `support_experience` | 72 | 44 | 14 | 14 |
| `crash_stability` | 63 | 38 | 12 | 13 |
| `kyc_onboarding` | 44 | 26 | 9 | 9 |
| `no_complaint` | 42 | 26 | 8 | 8 |
| `account_restricted` | 35 | 21 | 7 | 7 |
| `login_access` | 34 | 20 | 7 | 7 |
| `payments_transfers` | 30 | 18 | 6 | 6 |
| `fees_pricing` | 20 | 12 | 4 | 4 |
| `performance` | 17 | 11 | 3 | 3 |
| `complaint_unspecified` | 9 | 5 | 2 | 2 |
| **Reviews** | **400** | **243** | **81** | **76** |

**Classes too small to judge.** `performance` (17) and
`complaint_unspecified` (9) have three and two test examples. F1 scores for
them will be close to noise. They are reported with that caveat rather than
merged into other labels. `feature_gap` covers 38% of reviews; it is the
broadest label, and findings about it are correspondingly vague.

**Split.** 60/20/20, using iterative stratification so that every label
appears in train, validation and test. The test set is frozen and stays unused
until the final evaluation.

**Self-agreement (pending).** 50 of the 400 will be labelled a second time,
blind and days after the first pass. Scoring the second pass against the first
with the same macro-F1 used for the models gives the realistic ceiling for any
model.

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
src/support_reply_gap/   the package: ingest/, labeling/, eval/, analysis/, api/
data/gold/               hand-labelled gold set: IDs, labels, split (committed)
tests/                   pytest; database tests marked @pytest.mark.db
notebooks/               exploration only, never imported by src/
scripts/                 standalone utilities
reports/                 generated tables
docs/build-order.md      stage-by-stage plan with a definition of done each
docs/taxonomy.md         label definitions with examples, generated
```
