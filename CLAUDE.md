# Support Reply Gap

## What this project is

NLP analysis of Google Play reviews for German fintech apps (N26, Trade Republic,
Revolut, Scalable Capital). Each review is classified into complaint topics and
joined to whether and how fast the company replied. The output is a mismatch
analysis: which complaint types get answered versus which do not.

Portfolio project for ML/AI engineering job applications. One week of work.

## The deliverable

The evaluation table (`reports/eval.md`) and the README. Everything else serves
those two. If a change does not improve the eval table or the README, it is
probably out of scope.

## Hard rules

- **No causal claims.** Companies choose who to reply to. Findings are
  descriptive only: "complaint type X receives replies at rate Y with median
  latency Z". Never "company A ignores X". This applies to code comments,
  variable names, generated reports and the README.
- **Reviews are German.** Never default to English tokenisers, English stopword
  lists, or `*-base-uncased` English checkpoints. Use German or multilingual
  models (e.g. `deepset/gbert-base`). Flag it if you are unsure a component
  handles German.
- **The gold set is human-labelled only.** Never write, modify, extend or
  "fill in" labels in `data/gold/`. Never generate synthetic labelled examples.
  If the gold set looks too small, say so — do not expand it.
- **Never write a model and the test that judges it in the same step.** The
  eval harness and metrics are defined against the frozen gold set first.
- **Report null results honestly.** If a model path performs badly, or the
  reply-gap finding is weak, that goes in the README as-is.

## Label taxonomy

Multi-label: a review can carry several labels. `no_complaint` is exclusive.

| Label | Covers | Does NOT cover |
|---|---|---|
| `crash_stability` | Crashes, freezes, app won't start, white screen, bugs that block use | Slow but working; missing features |
| `login_access` | Can't log in, 2FA/TAN problems, locked out, password reset loops | First-time account opening → `kyc_onboarding` |
| `kyc_onboarding` | Identity verification, document upload, video ident, account opening rejected or stuck | Being locked out of an existing account → `login_access` |
| `payments_transfers` | Transfers failing or delayed, deposits not arriving, card declined, blocked payments, order execution | Complaints about the *cost* of a transfer → `fees_pricing` |
| `fees_pricing` | Fees, spreads, subscription cost, unexpected charges, interest rates | Transfer failing for a non-cost reason → `payments_transfers` |
| `support_experience` | The experience of contacting support: unreachable, slow, canned replies, no human | The underlying problem they contacted support about — label that too |
| `feature_gap` | Missing functionality, feature requests, UI/UX complaints about existing features | Features that exist but are broken → `crash_stability` |
| `no_complaint` | Praise, neutral, off-topic, unintelligible | Anything with a complaint in it |

Notes:
- `support_experience` is a meta-label and co-occurs often. That co-occurrence
  is the core of the analysis — do not collapse it into other labels.
- With ~400 labelled reviews some classes will have fewer than 20 examples.
  Report per-class support alongside F1 and flag any class too small to judge.

## Repo layout

```
data/raw/        scraped JSON, gitignored
data/gold/       hand-labelled set, COMMITTED — the asset
src/reply_gap/
  ingest/        scraper, Postgres loader, data quality checks
  labeling/      label schema, labelling CLI
  models/        baseline.py (tfidf+logreg), transformer.py, llm.py
  eval/          harness, metrics, cost and latency measurement
  analysis/      the reply-gap join
tests/
reports/         generated eval table, committed
```

## Stack and conventions

- Python 3.11+, venv at `.venv`, pip
- Postgres via Docker Compose
- Ruff for linting and formatting
- pytest for tests
- Type hints on public functions; no heavy abstraction layers
- Keep modules small and independently runnable

## Scope

**In:** ingestion + DQ checks, gold set, three model paths, eval harness,
reply-gap analysis, README, Docker Compose (Postgres + app), GitHub Actions.

**Deferred — do not build unless I explicitly ask:** pgvector retrieval,
Streamlit dashboard, Langfuse tracing, cloud deployment, the confidence router.

## Working with me

- I am new to Git, CI and deployment. Explain what a command does before
  suggesting I run it, and tell me when something is irreversible.
- Use plan mode for anything structural. Show the plan before writing files.
- Prefer small, reviewable changes over large ones. I need to be able to
  explain every file in this repo in a job interview.
- When you write something non-obvious, add a one-line comment saying why,
  not what.