# support-reply-gap

![CI](https://github.com/AurelRoe/support-reply-gap/actions/workflows/ci.yml/badge.svg)

Which complaint types in German fintech app reviews get a developer reply,
and how fast? This repo classifies Google Play reviews into complaint
categories and joins them to reply data. The result is a descriptive table:
reply rate and median reply latency per complaint type.

**Status:** Phase A (skeleton) is done. Phase B is under way: the three apps
are chosen (see [App selection](#app-selection-stage-b1)); ingestion is next.
There are no models or findings yet.
The full plan is in [docs/build-order.md](docs/build-order.md).

## Approach

1. Scrape Google Play reviews for three German apps from one product category
   (neobanks *or* brokers, not a mix, because complaint base rates differ).
   The apps are chosen by measured reply rate, not picked up front.
2. Hand-label ~300 reviews into a multi-label complaint taxonomy. This gold
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
src/support_reply_gap/   the package (api/, db.py so far)
tests/                   pytest; database tests marked @pytest.mark.db
notebooks/               exploration only, never imported by src/
scripts/                 standalone utilities
docs/build-order.md      stage-by-stage plan with a definition of done each
```
