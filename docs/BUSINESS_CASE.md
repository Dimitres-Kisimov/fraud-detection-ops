# Business case: fraud review operations

This document restates the repository in business terms. Everything is measured on
**synthetic seeded data** and every dollar parameter is a labelled **assumption**; the value
of the exercise is the decision framework and the measured deltas between policies, not the
absolute dollar figures.

## Situation

A mid-size payments operation processes on the order of 60,000 card transactions per
~4 months (the synthetic horizon used here). About 1.45% of them are fraudulent. A fraud
model scores every transaction, alerts above a threshold go to a review queue, and a team of
four analysts works that queue. Two decisions dominate the economics:

1. **Where the alert threshold sits.** Too high and fraud sails through; too low and the
   queue drowns the team.
2. **Which alerts get reviewed** when the queue exceeds capacity - which it does every
   shift.

## Quantified problem (assumptions labelled)

On the held-out test window (9,000 transactions, ~18 days of the timeline, 126 fraud cases
worth $20,744):

- ASSUMPTION: one analyst review costs $8 of loaded time; a missed fraud costs its full
  transaction amount; a reviewed fraud is caught.
- With the industry-default alert threshold of 0.5, the measured cost is **$15,587** per
  window - only 25% better than having no fraud review at all ($20,744), because a
  calibrated model at 1.4% prevalence almost never outputs probabilities above 0.5.
- The team can review 100 transactions per shift window (ASSUMPTION: 4 analysts x 25
  reviews). The tuned threshold fires 608 alerts in the same window - a 6:1 overflow, so
  "review everything we flag" is not an option and the selection rule inside the queue
  carries real money.

## Solution

Three components, each measured against a baseline:

1. **Calibrated risk scores.** A from-scratch logistic regression (class-weighted /
   focal loss) ranks fraud 20x better than random (PR-AUC 0.270 vs 0.013) and 8x better
   than the best single rule ("flag amounts above the 99th percentile", PR-AUC 0.034).
   Platt calibration reduces expected calibration error from 0.37 to 0.003, which is what
   makes the next two steps trustworthy.
2. **Cost-based threshold.** Sweeping the empirical cost curve on validation and operating
   at t* = 0.047 cuts test-window cost from $15,587 to **$8,841 (-43.3%)** under the stated
   cost assumptions, catching 76 of 126 frauds instead of 16.
3. **Optimized review queue.** Allocating the 100 reviews with an LP/MILP (maximize
   calibrated expected recovered value under capacity + per-segment coverage floors)
   recovers an expected **$11,259** per window vs $9,793 for the intuitive
   top-100-by-probability queue (**+15%**) and $231 for random auditing - while keeping
   every merchant segment under watch (top-K leaves segments fully unreviewed, an
   invitation for fraud migration). Realized value on observed labels: $12,398, close to
   the expectation because the probabilities are calibrated.

## Return on investment (illustrative arithmetic, synthetic scale)

Per 18-day window, threshold tuning alone is worth $6,745 against the naive default; queue
optimization adds roughly $1,466 of expected recovered value against the intuitive queue.
Annualized at this synthetic volume that is on the order of $165k, against a build that is
pure analyst-time reallocation - no additional headcount, no vendor spend. These figures
inherit every assumption above and the synthetic data generation; they are decision-support
arithmetic, not a forecast.

## Stakeholders

- **Fraud operations lead** - owns the threshold and the queue policy; gets the expected-
  cost curve and the allocation comparison instead of a gut-feel threshold.
- **Analyst team** - works a queue that respects capacity and stops burying them in
  low-value alerts; coverage floors keep their segment knowledge alive.
- **Finance / risk** - gets loss-vs-spend framing with assumptions exposed as knobs they
  can re-price.
- **Data science** - gets a calibration-first template: every probability consumed by a
  decision is measured for calibration before it is trusted.

## Deliverable

`python -m fdo --deliverables` produces, deterministically:

- `deliverables/executive_report.pdf` - cover with disclaimer and headline numbers, PR
  curves vs baselines, reliability before/after calibration, the expected-cost threshold
  curve, and the queue-allocation comparison.
- `deliverables/fraud_ops_workbook.xlsx` - Metrics, Thresholds (full sweep), QueuePlan
  (strategy comparison, per-segment coverage, the 100 selected reviews), and an
  Assumptions sheet listing every assumption in one place.
