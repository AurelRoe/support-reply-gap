# Self-agreement

Generated 2026-11-13 by `python -m support_reply_gap.labeling.gold agreement`.

50 gold reviews were labelled a second time, blind to the first pass. Pass 1 is the reference; pass 2 is scored against it with the same macro-F1 used for the models. This is the ceiling model scores are read against.

| Measure | Value |
|---|---|
| Macro-F1 (pass 2 vs pass 1) | 0.69 |
| Exact match (whole label set) | 0.64 |

## Per label

Labels with fewer than 5 pass-1 examples are marked; their scores rest on a handful of reviews and should not be read as a ceiling. n/a: the label does not occur in either pass, or kappa is undefined.

| Label | Pass 1 | Pass 2 | F1 | Kappa | |
|---|---:|---:|---:|---:|---|
| `crash_stability` | 6 | 6 | 0.50 | 0.43 |  |
| `performance` | 6 | 2 | 0.25 | 0.20 |  |
| `login_access` | 7 | 8 | 0.80 | 0.76 |  |
| `kyc_onboarding` | 3 | 4 | 0.86 | 0.85 | few examples |
| `account_restricted` | 2 | 1 | 0.67 | 0.66 | few examples |
| `payments_transfers` | 4 | 1 | 0.40 | 0.38 | few examples |
| `fees_pricing` | 0 | 0 | n/a | n/a | few examples |
| `support_experience` | 8 | 8 | 0.75 | 0.70 |  |
| `feature_gap` | 19 | 24 | 0.79 | 0.64 |  |
| `complaint_unspecified` | 3 | 4 | 0.86 | 0.85 | few examples |
| `no_complaint` | 4 | 4 | 1.00 | 1.00 | few examples |
