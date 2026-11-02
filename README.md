# support-reply-gap

![CI](https://github.com/AurelRoe/support-reply-gap/actions/workflows/ci.yml/badge.svg)

Which complaint types in German fintech app reviews get a developer reply,
and how fast? This repo classifies Google Play reviews into complaint
categories and joins them to reply data. The result is a descriptive table:
reply rate and median reply latency per complaint type.

**Status:** Phase A (skeleton) is done. Next up is Phase B: choosing the
three apps and ingesting reviews. There are no data, models or findings yet.
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
