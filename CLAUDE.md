# Support Reply Gap

## What this project is

NLP analysis of Google Play reviews for German fintech apps. Each review is
classified into complaint topics and joined to whether and how fast the company
replied. The output is a mismatch analysis: which complaint types get answered
versus which do not.

Three apps, all in **one** product category — neobanks *or* brokers, not a mix,
because complaint base rates differ between the two and a mixed set confounds
the dependent variable. The specific apps are chosen in stage B1 by measuring
reply rates, not by assumption. Do not assume or hardcode a set before then.

Portfolio project. The working plan and
stage sequence is `docs/build-order.md` — read it before proposing structural
work.

## The deliverable

The evaluation table (`reports/eval.md`) and the README. Everything else serves
those two. If a change does not improve the eval table or the README, it is
probably out of scope.

## Hard rules

- **No causal claims.** Companies choose who to reply to, and that choice is
  unobserved. Findings are descriptive only: "complaint type X receives replies
  at rate Y with median latency Z". Never "company A ignores X". This applies to
  code comments, variable names, generated reports and the README.
- **Reviews are German.** Never default to English tokenisers, English stopword
  lists, or `*-base-uncased` English checkpoints. Use German or multilingual
  models (e.g. `deepset/gbert-base`). Flag it if you are unsure a component
  handles German.
- **The gold set is human-labelled only.** Never write, modify, extend or
  "fill in" labels in `data/gold/`. Never generate synthetic labelled examples.
  If the gold set looks too small, say so — do not expand it.
- **Never write a model and the test that judges it in the same step.** The eval
  harness and metrics are defined against the frozen gold set first. The stage
  order enforces this: the harness (C3) is built before the LLM path (C5), so
  every model is measured by a yardstick that existed before it did.
- **Report null results honestly.** If a model path performs badly, or the
  reply-gap finding is weak, that goes in the README as-is.

## Label taxonomy

Multi-label: a review can carry several labels. `no_complaint` and
`complaint_unspecified` are exclusive. This decision is **settled** — it is
already baked into the schema below, the metric (macro-F1 over multi-label) and
the router's confidence logic. Do not reopen it; changing it means relabelling
the gold set.

The source of truth is `src/support_reply_gap/labeling/taxonomy.py`, with
examples, rendered to `docs/taxonomy.md`. This table is a summary of it; if
they disagree, the code wins. Set in B4 from a 150-review dev sample.

| Label | Covers | Does NOT cover |
|---|---|---|
| `crash_stability` | Crashes, freezes, app won't start, white screen, bugs that block use | Slow but working → `performance`; missing features → `feature_gap` |
| `performance` | Slow but working: load times, lag, app getting slower over time | Hangs or anything that blocks use → `crash_stability` |
| `login_access` | Can't log in, 2FA/TAN problems, locked out, password reset loops | First-time account opening → `kyc_onboarding`; bank restricting the account → `account_restricted` |
| `kyc_onboarding` | Identity verification, document upload, video ident, account opening rejected or stuck | Locked out of an existing account → `login_access`; re-ident to lift a restriction → `account_restricted` |
| `account_restricted` | Bank-initiated: account closed, frozen or blocked, funds unreachable, re-ident forced to regain access | User failing to log in → `login_access`; single card or payment blocked while the account works → `payments_transfers` |
| `payments_transfers` | Transfers failing or delayed, deposits not arriving, card declined or blocked, blocked payments, order execution | Complaints about the *cost* of a transfer → `fees_pricing` |
| `fees_pricing` | Fees, spreads, subscription cost, unexpected charges, interest rates, and money the user believes they are owed (rewards, bonuses, referral payouts, interest not passed on) | Transfer failing for a non-cost reason → `payments_transfers` |
| `support_experience` | The experience of contacting support: unreachable, slow, canned replies, no human | The underlying problem they contacted support about — label that too |
| `feature_gap` | Missing functionality, feature requests, UI/UX complaints about existing features, ads and upselling in the app | Broken features → `crash_stability`; slowness → `performance` |
| `complaint_unspecified` | Clearly negative with no nameable subject ("Abzocke", "Nie wieder") | Any complaint with a subject, however vague — label the subject |
| `no_complaint` | Praise, neutral, off-topic, unintelligible | Anything with a complaint in it |

Notes:
- `support_experience` is a meta-label and co-occurs often. That co-occurrence
  is the core of the analysis — do not collapse it into other labels.
- `feature_gap` is the broadest class. Findings about it are correspondingly
  vague; say so rather than splitting it further.
- Trading/crypto and ads were considered as labels in B4 and rejected: too
  small, and trading is Revolut-specific. Both fold into existing labels and
  get a line in the README limitations.
- In the 300-review gold set two classes have fewer than 20 examples:
  `complaint_unspecified` (12) and `fees_pricing` (10), one test review each.
  Report per-class support alongside F1 and flag any class too small to judge.
  Do not quietly merge small classes to make the numbers look better.
- Human self-agreement on a doubly-labelled subset is the accuracy ceiling.
  Report model scores against that ceiling, not against 1.0.

## Repo layout

```
data/raw/        scraped JSON, gitignored
data/gold/       hand-labelled set, COMMITTED — the asset (ids + labels, no text)
data/labelling/  labelling sheets with review text, gitignored
notebooks/       exploration only; never imported by src/
scripts/         standalone utilities (probe_apps.py)
src/support_reply_gap/
  ingest/        scraper, Postgres loader, data quality checks
  labeling/      label schema, labelling CLI
  models/        baseline.py (tfidf+logreg), llm.py, router.py
  eval/          harness, metrics, cost and latency measurement
  analysis/      the reply-gap join
  api/           FastAPI service
tests/
reports/         generated eval table, committed
docs/            build-order.md
```

Dependencies point one direction: `notebooks/` imports from `src/`, never the
reverse. If a notebook grows a function another notebook needs, it moves to
`src/`.

## Stack and conventions

- Python 3.11+, venv at `.venv`, pip, package installed editable (`pip install -e`)
- Postgres via Docker Compose, host port 5433
- Plain SQL via `psycopg`. No ORM — I know SQL well, an ORM is a layer between
  me and something I can already read.
- Ruff for linting and formatting
- pytest for tests; tests needing a live database are marked `@pytest.mark.db`
  so CI can run `pytest -m "not db"`
- Type hints on public functions; no heavy abstraction layers
- Keep modules small and independently runnable

## Scope

**In:** ingestion + DQ checks, gold set, two model paths (TF-IDF baseline and
LLM with structured output), confidence router between them, eval harness wired
into CI as a gate, reply-gap analysis, FastAPI service, Docker Compose, GitHub
Actions, cloud deployment, README.

Two model paths, not three. Cheap plus expensive with a measured trade-off is
the entire comparison story, and the router needs exactly two tiers. A
fine-tuned German transformer is a *possible later addition* if time appears —
it would slot in as another eval row without disturbing anything, since the
harness already exists. Do not build it unasked.

**Deferred — do not build unless I explicitly ask:** pgvector retrieval,
Streamlit dashboard, Langfuse tracing.

Note: Compose uses the `pgvector/pgvector:pg16` image. That costs nothing over
plain Postgres and keeps the option open — it is **not** a signal that retrieval
is in scope. Do not swap the image for plain Postgres either; the volume is
built on it.

## Working with me

- I am new to Git, CI and deployment. Explain what a command does before
  suggesting I run it, and tell me when something is irreversible.
- Use plan mode for anything structural. Show the plan before writing files.
- Prefer small, reviewable changes over large ones. I need to be able to explain
  every file in this repo — if a suggestion is something I
  could not defend, say so rather than writing it.
- When you write something non-obvious, add a one-line comment saying why, not
  what.
- macOS terms and menu paths in English.