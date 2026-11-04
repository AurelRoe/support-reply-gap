# Build Order

Working plan for the project. Four phases. Each stage has a definition of done.

Ordering principles, since several stages sit earlier than feels natural:

1. **Infrastructure first, while stakes are low.** Docker and CI are easier to
   set up around an empty project than a finished one.
2. **The gold set gates everything.** Both model paths and the CI eval gate
   measure against it. It comes before any model.
3. **Get a rough answer early.** A bad version of the finding (C2) de-risks the
   project. Infrastructure with no findings is a failed project; findings with
   mediocre infrastructure is not.

---

## Phase A — Skeleton

No data, no models. A repo that runs, tests, and deploys *something*.

- [x] **A1. Repo and package layout.** `src/support_reply_gap/` as an
      installable package via `pyproject.toml`, plus `tests/`, `notebooks/`,
      `scripts/`.
      *Done:* `import support_reply_gap` works from a notebook and a test.

- [x] **A2. Postgres via Docker Compose.** One service, named volume,
      healthcheck. Image `pgvector/pgvector:pg16` — costs nothing over plain
      Postgres and leaves the option open. Host port 5433 to avoid collisions.
      *Done:* `docker compose up -d`, then `select 1` from Python.

- [x] **A3. CI that does almost nothing.** GitHub Actions: install, `ruff`, one
      trivial test. Run `pytest -m "not db"` so DB tests don't break it.
      *Done:* a green check on a pull request.

- [x] **A4. Walking skeleton deploy.** FastAPI with one `/health` endpoint,
      containerised, deployed. No models, no database.
      Deploying a stub is a different and much easier problem than deploying the
      finished system. Separating them is worth a lot given the deployment gap
      this project is meant to close.
      *Done:* a public URL returns `{"status": "ok"}`.

---

## Phase B — Data

- [x] **B1. Lock the app set.** Screened candidates with
      `scripts/probe_apps.py` and chose on reply rate for 1–2★ reviews, volume,
      German share and reply text diversity.
      **Three apps**, all in one product category (neobanks *or* brokers) so
      complaint base rates are comparable.
      *Done:* three apps chosen, probe numbers and rejected candidates written
      into the README as justification.

- [x] **B2. Ingest to Postgres.** `reviews` (review_id PK, app, author, score,
      content, at, reply_content, replied_at, scraped_at) and `apps`. Upsert on
      review_id so re-runs are safe.
      DQ checks as assertions: no null ids, no duplicates, scores 1–5,
      no future dates, per-app row counts in range. `replied_at >= at` is a
      warning rather than an assertion, because users edit reviews after a
      reply (2.4% of rows, see README).
      *Done:* one command populates the DB from empty, idempotently, checks
      passing, a few thousand reviews per app.

- [x] **B3. Explore (notebook).** Length and score distributions, language mix,
      reply rate by score and over time, duplicates and spam, share of
      contentless reviews.
      *Done:* I can describe the data in five sentences and know what cleaning
      is needed.

- [ ] **B4. Define the taxonomy.** Read 200–300 reviews and refine the five
      starting categories, which will probably turn out wrong.
      **Decided before labelling: multi-label.** Reviews routinely complain
      about two things at once. The choice shapes the metric, the prompt schema
      and the router's confidence logic, and reversing it later would mean
      relabelling. Multi-label is more honest here and slightly more work
      everywhere.
      An explicit `no_complaint` class keeps positive reviews from being
      forced into complaint categories.
      *Done:* written taxonomy, one paragraph and 2–3 real examples per class,
      committed.

- [ ] **B5. Hand-label the gold set.** ~300 reviews, stratified across apps and
      star ratings — not random, which over-samples 1★/5★ noise and
      under-samples rare classes.
      ~50 of them get labelled **twice**, days apart, to measure
      self-agreement. That is the realistic ceiling: at 85% self-agreement, a
      model at 0.85 macro-F1 is at human level and chasing 0.95 is chasing
      noise. The ceiling gets reported; it is the most distinctive thing in the
      project.
      The test split is held out at this point and stays unseen until the end.
      *Done:* labelled CSV committed, train/val/test split, self-agreement
      recorded.

---

## Phase C — Models and evaluation

Two model paths, not three: cheap and expensive, with a measured trade-off
between them. That is the whole comparison story, and the router needs exactly
two tiers.

- [ ] **C1. Baseline: TF-IDF + logistic regression.** German stopwords;
      character n-grams worth trying for compound words.
      *Done:* a macro-F1 on validation. This is the bar.

- [ ] **C2. First end-to-end answer** ← *milestone*
      Baseline over all reviews, joined to reply data. Mismatch table: complaint
      volume by category vs reply rate and median latency by category.
      Numbers will be mediocre. The point is that the finding now exists and
      everything after improves it rather than hoping to reach it.
      *Done:* one table, one chart, three sentences of honest interpretation.
      No causal language — companies choose who to reply to and that choice is
      unobserved.

- [ ] **C3. Evaluation harness.** A script, not a notebook. Model interface in,
      macro-F1 + per-class F1 + cost per 1k + p95 latency + confusion matrix
      out, written to a versioned file. Built *before* the LLM path so it is
      measured from the moment it exists.
      *Done:* `python -m support_reply_gap.eval --model baseline` produces the
      table.

- [ ] **C4. Eval as a CI gate.** Workflow fails if macro-F1 drops below a
      committed threshold. Cached fixtures so CI doesn't hit paid APIs.
      *Done:* a PR that degrades the model turns the check red.

- [ ] **C5. LLM path with structured output.** Constrained output against the
      taxonomy schema. Prompts versioned as files, not strings in code.
      *Done:* a second row in the eval table, with cost per 1k and p95 latency.

- [ ] **C6. Router.** Baseline first, escalate to the LLM below a confidence
      threshold. Calibrate on validation and plot the cost/accuracy curve — the
      curve is more interesting than any single operating point, and it's the
      part that reads as engineering judgement rather than model shopping.
      *Done:* a third row plus the trade-off curve.

---

## Phase D — Surface and findings

- [ ] **D1. Final analysis.** Redo C2 with the best configuration. Sample sizes
      per cell at minimum. State null results plainly.

- [ ] **D2. FastAPI service.** Replace the A4 stub: `/classify` over the router,
      reading from Postgres.

- [ ] **D3. README.** Written continuously, not at the end. The question, the
      data and why these three apps, the taxonomy, the eval table, findings
      including nulls, limitations, how to run it.

The eval table and the honest findings are the deliverable. Everything else
exists to make them credible.

---

## Risk notes

- **Reply rate is the single point of failure.** If the probe shows thin reply
  rates across all candidates, widen to incumbent banks (DKB, ING, Commerzbank)
  before committing to anything downstream.
- **Boilerplate replies** make reply *text* useless as a signal; reply timing
  still works. Which of the two applies needs checking before anything is
  designed around reply content.
- **Public repo hygiene:** code and aggregated tables get committed. A raw dump
  of review text with usernames attached does not.
- **Scope:** Phase C carries the weight. If something has to go, it goes from
  Phase D — and if time appears instead, a fine-tuned German transformer
  (`deepset/gbert-base`) slots in as another eval row without disturbing
  anything, since the harness already exists.