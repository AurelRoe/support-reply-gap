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

- [x] **B4. Define the taxonomy.** Read 200–300 reviews and refine the five
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
      *Outcome:* a 150-review dev sample (`notebooks/02_taxonomy_development`)
      took the eight draft labels to eleven: `account_restricted`,
      `performance` and `complaint_unspecified` added, `fees_pricing` widened
      to money owed (rewards, bonuses). Trading and ads were rejected as labels
      — too rare, and trading is Revolut-only. The 150 reviews are excluded
      from the gold set.

- [x] **B5. Hand-label the gold set.** ~400 reviews, stratified across apps and
      weighted toward 1–2★ — not random, which over-samples 1★/5★ noise and
      under-samples rare classes. 400 rather than 300 because eleven labels
      against 300 reviews leaves too many classes under 20 examples; the
      weighting follows the complaints, since most labels are complaint types.
      The B4 dev sample (`data/gold/dev_sample_ids.txt`) is excluded.
      ~50 of them get labelled **twice**, days apart, to measure
      self-agreement. That is the realistic ceiling: at 85% self-agreement, a
      model at 0.85 macro-F1 is at human level and chasing 0.95 is chasing
      noise. The ceiling gets reported; it is the most distinctive thing in the
      project.
      The test split is held out at this point and stays unseen until the end.
      *Done:* labelled CSV committed, train/val/test split, self-agreement
      recorded.
      *Outcome:* drawn at 400, then trimmed to 300 before labelling to
      keep hand-labelling time manageable, balanced within each split, app
      and star rating. This accepts more classes under 20 examples; they are
      flagged in the eval, not merged.
      All 300 then hand-labelled and `gold_labels.csv` committed.
      `complaint_unspecified` (12) and `fees_pricing` (10) are under 20, with
      one test review each; the frozen split is kept. Self-agreement on 50
      random reviews: 0.69 macro-F1, 64% exact match.

---

## Phase C — Models and evaluation

Two model paths, not three: cheap and expensive, with a measured trade-off
between them. That is the whole comparison story, and the router needs exactly
two tiers.

- [x] **C1. Baseline: TF-IDF + logistic regression.** German stopwords;
      character n-grams worth trying for compound words.
      *Done:* a macro-F1 on validation. This is the bar.
      *Outcome:* 0.486 validation macro-F1 (C = 10, no stopwords, single
      words plus character n-grams, threshold 0.3), about 70% of the 0.69
      self-agreement ceiling. German stopwords and word pairs did not help.

- [x] **C2. First end-to-end answer** ← *milestone*
      Baseline over all reviews, joined to reply data. Mismatch table: complaint
      volume by category vs reply rate and median latency by category.
      Numbers will be mediocre. The point is that the finding now exists and
      everything after improves it rather than hoping to reach it.
      *Done:* one table, one chart, three sentences of honest interpretation.
      No causal language — companies choose who to reply to and that choice is
      unobserved.
      *Outcome:* `reports/reply_gap_c2.md`, limited to 1–3★ (1,210 reviews)
      because the baseline puts complaint labels on about half of all 5★
      reviews. Within C24 and Revolut, `support_experience` and the account
      labels sit above the app's reply rate, `feature_gap` and `performance`
      slightly below; star rating lines up with part of that. N26 replies to
      nearly every review.

- [x] **C3. Evaluation harness.** A script, not a notebook. Model interface in,
      macro-F1 + per-class F1 + cost per 1k + p95 latency + confusion matrix
      out, written to a versioned file. Built *before* the LLM path so it is
      measured from the moment it exists.
      *Done:* `python -m support_reply_gap.eval --model baseline` produces the
      table.
      *Outcome:* `reports/eval.md`, regenerated from result files in
      `reports/eval/` that record the commit and a gold-set checksum. Added a
      bootstrap 95% interval and exact match. Validation and test both
      scored; only validation drives decisions. Baseline: 0.49 val
      (0.34–0.63), 0.45 test (0.34–0.59). Follow-up: paired bootstrap of the
      difference to the baseline, from per-review predictions in the result
      files. Measured on baseline variants, it is 1.7–4.5× narrower than
      comparing separate intervals; against a model with different errors,
      about a 10-point gap is resolvable. Open for C4: CI has no database,
      so the gate needs the review text from somewhere else.

- [x] **C4. Eval as a CI gate.** Workflow fails if macro-F1 drops below a
      committed threshold. Cached fixtures so CI doesn't hit paid APIs.
      *Done:* a PR that degrades the model turns the check red.
      *Outcome:* instead of cached fixtures, CI checks the committed result
      files: models are scored locally, where the review text is, and
      `--gate` fails if a result is stale (a score-relevant file changed
      after it), made from uncommitted code, inconsistent with its stored
      predictions, below the validation floor (baseline 0.45), or if
      `reports/eval.md` is out of date. No review text on GitHub, no API
      calls in CI. Shown on a throwaway PR with C = 0.3: the gate first
      flagged the stale result, then, after a re-run, val macro-F1 0.217
      below the floor; the check turned red. Requires merge commits, not
      squash merges.

- [x] **C5. LLM path with structured output.** Constrained output against the
      taxonomy schema. Prompts versioned as files, not strings in code.
      *Done:* a second row in the eval table, with cost per 1k and p95 latency.
      *Outcome:* Claude Haiku 5.5 at low effort, chosen for a budget of a few
      cents, with prompt `prompts/classify_v1.md` (taxonomy rendered from
      `taxonomy.py`; no gold review among its examples). Budget cap per run,
      free estimate before spending, local answer cache. Val 0.67
      (0.57–0.80), test 0.56 (0.45–0.67); paired difference to the baseline
      +0.19 val, +0.11 test, both intervals excluding 0. $0.09 per 1k
      reviews, p95 about 2.5 s. Total spend about one cent. `performance`
      remains the weakest label, matching low human self-agreement on it.

- [x] **C6. Router.** Baseline first, escalate to the LLM below a confidence
      threshold. Calibrate on validation and plot the cost/accuracy curve — the
      curve is more interesting than any single operating point, and it's the
      part that reads as engineering judgement rather than model shopping.
      *Done:* a third row plus the trade-off curve.
      *Outcome:* confidence is the baseline's smallest distance between a label
      probability and the threshold. Operating rule fixed before the sweep:
      smallest escalation share reaching 90% of the LLM's validation gain. It
      lands at 85% escalated (83% across val and test): val 0.66, test 0.58,
      about $0.075 per 1k, no latency gain. No worse than the LLM alone; most
      of the gain arrives by 50% escalated. At $0.09 per 1k for the LLM, the
      router is barely worth its complexity. Cost $0, all from cached answers.

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