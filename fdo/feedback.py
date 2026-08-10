"""Analyst feedback-loop simulation: the selective-labels problem, measured.

The drift playbook in ``fdo/drift.py`` ends on a reassuring line: "the analyst
queue doubles as a continuous labelled sample". This module stress-tests that
claim honestly. In production, tomorrow's training labels come from *today's
review decisions*: analysts only confirm fraud on the transactions the model
flagged AND the queue had capacity to review. Everything else ages into the
training set with whatever label the labelling policy assumes. That closed
loop has a name - the SELECTIVE LABELS problem (Lakkaraju et al., KDD 2017;
"delayed feedback" in the ads/fraud literature) - and it is invisible on a
production dashboard precisely because the missing labels are missing.

Simulation design - one variable changes:

- A shared INITIAL model is trained on the first ``initial_frac`` of the
  timeline, which is treated as fully labelled history (chargebacks there have
  matured - a labelled ASSUMPTION). Its cost threshold t0* is swept once on
  the initial validation slice, then FROZEN, and the review rule (top
  ``reviews_per_round`` alerts by calibrated p x amount - the repo's
  unconstrained queue rule) is frozen too. Every arm starts from this exact
  model and operates this exact alert/review policy.
- The deployment period between the initial window and the held-out test
  window is split into ``n_rounds`` contiguous rounds. Each round, each arm
  scores the round's transactions, reviews what its policy allows, appends the
  round to its training pool under its LABELLING POLICY, and retrains
  (same family, loss, and hyperparameters - architecture never moves).
- Four labelling-policy arms, from least to most censored:

  1. ``full_labels``   - every row keeps its observed label. An oracle arm:
                         possible only in simulation, it is the ceiling that
                         makes the cost of the other arms measurable.
  2. ``chargeback``    - reviewed rows get true labels; unreviewed fraud
                         self-reports via chargeback with probability
                         ``chargeback_rate``; everything else is labelled
                         legitimate. The realistic production regime.
  3. ``assume_legit``  - reviewed rows get true labels; every unreviewed row
                         is labelled legitimate. The censoring regime: missed
                         fraud becomes a *wrong* negative label exactly where
                         the model is already blind.
  4. ``reviewed_only`` - only reviewed rows enter the pool at all. The purist
                         selective-labels regime: a tiny sample chosen by the
                         model's own opinions.

- After every retrain, each arm's model is evaluated on the SAME held-out test
  window the rest of the repo uses (verified index-for-index). The test window
  never feeds back into any arm - it is pure reporting, no selection.

The metric that makes the trap visible is ``observed_recall``: the share of
the fraud an arm KNOWS ABOUT that its reviews caught. Under ``assume_legit``
the only fraud the arm knows about is the fraud it reviewed, so its observed
recall is 100% *by construction* while its true label coverage is far lower -
the dashboard says "we catch everything" precisely because the misses were
relabelled as clean. That arithmetic identity is the selective-labels illusion
and the tests assert it.

HONESTY NOTES:

- Data is synthetic and seeded; the findings are properties of the generator
  (its scheduled day-60 concept drift makes the deployment rounds exactly the
  data a retrain most needs). Measured here, censoring barely moves the
  RANKING - the clean initial history anchors it - and instead poisons the
  PROBABILITIES, which is worse than it sounds: every decision layer in this
  repo (threshold, queue EV) consumes calibrated probabilities.
- ``reviews_per_round`` and ``chargeback_rate`` are labelled ASSUMPTIONS, not
  estimates. Real chargebacks land 30-90 days late; here they surface at the
  round boundary, so the chargeback arm is *optimistic* about label latency.
- The initial window being fully labelled is itself the "matured labels"
  assumption; the generator's 4% unreported-fraud noise is upstream of every
  arm (even ``full_labels`` never sees those - no arm is more than
  observed-label perfect).
- Frozen t0* and frozen capacity are a design choice, not a claim: they keep
  the labelling policy as the only difference between arms, so every
  divergence in the trajectory is attributable to labels alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fdo.evaluate import ece, pr_auc
from fdo.fairness import segment_fraud_coverage
from fdo.model import FraudModel, time_split
from fdo.threshold import choose_threshold, empirical_cost, sweep_thresholds

ARMS = ("full_labels", "chargeback", "assume_legit", "reviewed_only")

TRAJECTORY_COLUMNS = [
    "round",
    "arm",
    "model_updated",
    "pool_n",
    "pool_labelled_prevalence",
    "pool_true_prevalence",
    "n_alerts",
    "n_reviewed",
    "n_fraud_reviewed",
    "round_fraud",
    "round_known_fraud",
    "observed_recall",
    "label_coverage",
    "false_negative_labels_cum",
    "test_pr_auc",
    "test_ece",
    "test_recall_at_t0",
    "test_alerts_at_t0",
    "test_cost_at_t0",
]


@dataclass(frozen=True)
class FeedbackConfig:
    """Simulation shape and labelling knobs. All POLICY ASSUMPTIONS."""

    initial_frac: float = 0.40       # fully-labelled history the initial model trains on
    n_rounds: int = 3                # deployment rounds between history and the test window
    reviews_per_round: int = 100     # analyst capacity per round (same rate as the queue)
    chargeback_rate: float = 0.85    # P(unreviewed fraud self-reports by the round boundary)
    pool_val_frac: float = 0.15      # newest slice of each pool held out for early stop + Platt
    min_train_pos: int = 10          # below this many positive labels, skip the retrain

    def describe(self) -> list[str]:
        return [
            f"ASSUMPTION: the first {self.initial_frac:.0%} of the timeline is fully "
            "labelled history (chargebacks matured before deployment starts).",
            f"ASSUMPTION: analysts review at most {self.reviews_per_round} alerts per round, "
            "chosen by calibrated p x amount (the repo's unconstrained queue rule).",
            f"ASSUMPTION: an unreviewed fraud self-reports via chargeback with probability "
            f"{self.chargeback_rate:.0%}, and it lands by the round boundary - optimistic "
            "about label latency (real chargebacks take 30-90 days).",
            "DESIGN CHOICE: the alert threshold t0* and the review capacity are FROZEN at "
            "their initial values, so the labelling policy is the only variable between arms.",
        ]


def feedback_split(
    n: int, config: FeedbackConfig
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Time-ordered windows: initial labelled history, deployment rounds, test.

    The TEST window is taken from ``fdo.model.time_split`` verbatim, so every
    arm is judged on the identical rows the rest of the repo reports on. The
    deployment region between the initial window and the test window is split
    into ``n_rounds`` contiguous, near-equal, time-ordered rounds. No
    shuffling anywhere - same leakage discipline as the champion split.
    """
    if not 0.0 < config.initial_frac < 0.85:
        raise ValueError("initial_frac must be in (0, 0.85)")
    if config.n_rounds < 1:
        raise ValueError("n_rounds must be at least 1")
    if config.reviews_per_round < 0:
        raise ValueError("reviews_per_round must be non-negative")
    if not 0.0 <= config.chargeback_rate <= 1.0:
        raise ValueError("chargeback_rate must be in [0, 1]")
    _, _, idx_test = time_split(n)
    test_start = int(idx_test[0])
    initial_end = int(n * config.initial_frac)
    if initial_end < 1 or test_start - initial_end < config.n_rounds:
        raise ValueError("initial_frac leaves no room for the deployment rounds")
    idx = np.arange(n)
    rounds = [np.asarray(r) for r in np.array_split(idx[initial_end:test_start], config.n_rounds)]
    return idx[:initial_end], rounds, idx_test


def select_reviews(
    p: np.ndarray, amounts: np.ndarray, threshold: float, capacity: int
) -> np.ndarray:
    """Boolean mask of the alerts the analysts actually review this round.

    Alerts fire at ``p >= threshold``; when they exceed ``capacity``, the top
    ``capacity`` by expected value p x amount are reviewed (stable sort, so
    ties break toward the earlier transaction - deterministic). Analysts never
    review below the alert threshold: that is the shipped policy being
    simulated, and it is exactly what creates the censored region.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    amounts = np.asarray(amounts, dtype=np.float64).ravel()
    if p.shape != amounts.shape:
        raise ValueError("select_reviews: p and amounts must share one shape")
    if capacity < 0:
        raise ValueError("select_reviews: capacity must be non-negative")
    alerts = p >= threshold
    n_alerts = int(alerts.sum())
    if n_alerts <= capacity:
        return alerts
    idx_alert = np.flatnonzero(alerts)
    order = np.argsort(-(p[idx_alert] * amounts[idx_alert]), kind="stable")
    reviewed = np.zeros(p.shape[0], dtype=bool)
    reviewed[idx_alert[order[:capacity]]] = True
    return reviewed


def apply_label_policy(
    policy: str,
    y_observed: np.ndarray,
    reviewed: np.ndarray,
    surfaced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Labels one deployment round under a labelling policy.

    ``y_observed`` is the observed (noisy) label, ``reviewed`` marks the rows
    the analysts confirmed this round, ``surfaced`` marks the rows whose fraud
    would self-report via chargeback if left unreviewed. Returns
    ``(labels, labelled)``: the label each row would enter the pool with, and
    the mask of rows that enter the pool at all. Exact collapse cases (the
    tests assert all three):

    - ``surfaced`` everywhere fraud occurs  -> chargeback == full_labels,
    - ``surfaced`` nowhere                  -> chargeback == assume_legit,
    - ``reviewed`` everywhere               -> all four policies coincide.
    """
    y = np.asarray(y_observed, dtype=np.float64).ravel()
    rev = np.asarray(reviewed, dtype=bool).ravel()
    srf = np.asarray(surfaced, dtype=bool).ravel()
    if not (y.shape == rev.shape == srf.shape):
        raise ValueError("apply_label_policy: all inputs must share one shape")
    if policy == "full_labels":
        return y.copy(), np.ones(y.shape[0], dtype=bool)
    if policy == "reviewed_only":
        return y.copy(), rev.copy()
    if policy == "assume_legit":
        return np.where(rev, y, 0.0), np.ones(y.shape[0], dtype=bool)
    if policy == "chargeback":
        return np.where(rev, y, np.where(srf & (y == 1), 1.0, 0.0)), np.ones(
            y.shape[0], dtype=bool
        )
    raise ValueError(f"unknown labelling policy: {policy}")


def _fit_on_pool(
    df: pd.DataFrame,
    pool_idx: np.ndarray,
    pool_labels: np.ndarray,
    template: FraudModel,
    config: FeedbackConfig,
    prev_model: FraudModel,
) -> tuple[FraudModel, bool]:
    """Retrain on a labelled pool (labels as the arm's policy recorded them).

    The pool is time-ordered; the newest ``pool_val_frac`` is held out for
    early stopping and Platt calibration - so calibration inherits the label
    bias too, exactly as it would in production. If the training slice has
    fewer than ``min_train_pos`` positive labels or the validation slice has
    none, the retrain is skipped and the previous model kept (a shop cannot
    retrain a fraud model on a pool with no known fraud).
    """
    n_pool = int(pool_idx.size)
    n_val = max(int(n_pool * config.pool_val_frac), 1)
    idx_tr, idx_va = pool_idx[:-n_val], pool_idx[-n_val:]
    lab_tr, lab_va = pool_labels[:-n_val], pool_labels[-n_val:]
    if idx_tr.size == 0 or int(lab_tr.sum()) < config.min_train_pos or int(lab_va.sum()) < 1:
        return prev_model, False
    y_full = df["is_fraud"].to_numpy(dtype=np.int64).copy()
    y_full[pool_idx] = pool_labels.astype(np.int64)
    df_arm = df.assign(is_fraud=y_full)
    model = FraudModel(
        loss=template.loss, gamma=template.gamma, l2=template.l2,
        lr=template.lr, max_iter=template.max_iter,
    ).fit(df_arm, idx_tr, idx_va)
    return model, True


def run_feedback(results: dict, config: FeedbackConfig | None = None) -> dict:
    """Run the four-arm feedback-loop simulation. Deterministic given the
    pipeline result: training is zero-init full-batch gradient descent (no
    RNG) and the chargeback draws come from a dedicated seeded generator."""
    config = config or FeedbackConfig()
    df = results["df"]
    n = len(df)
    idx_init, rounds, idx_te = feedback_split(n, config)
    if not np.array_equal(idx_te, results["splits"]["test"]):
        raise RuntimeError("feedback test window diverged from the pipeline test window")

    y_obs = df["is_fraud"].to_numpy(dtype=np.float64)
    amounts = df["amount"].to_numpy(dtype=np.float64)
    segments = df["merchant_cat"].to_numpy()
    y_te, amounts_te, seg_te = y_obs[idx_te], amounts[idx_te], segments[idx_te]
    costs = results["thresholds"]["costs"]
    template = results["models"][results["primary_name"]]

    # One uniform draw per row, from a dedicated generator (prime offset so it
    # never aliases the data generator's stream). Only the chargeback arm
    # consumes it; the draws are fixed before any arm acts.
    draws = np.random.default_rng(int(results["seed"]) + 104_729).random(n)
    surfaced = (draws < config.chargeback_rate) & (y_obs == 1)

    # --- shared initial model + frozen operating point ---------------------
    n_val0 = max(int(idx_init.size * config.pool_val_frac), 1)
    idx_tr0, idx_va0 = idx_init[:-n_val0], idx_init[-n_val0:]
    model0 = FraudModel(
        loss=template.loss, gamma=template.gamma, l2=template.l2,
        lr=template.lr, max_iter=template.max_iter,
    ).fit(df, idx_tr0, idx_va0)
    p_va0 = model0.predict_calibrated(df.iloc[idx_va0])
    t0 = choose_threshold(sweep_thresholds(y_obs[idx_va0], amounts[idx_va0], p_va0, costs))

    def _test_eval(model: FraudModel) -> dict:
        p_te = model.predict_calibrated(df.iloc[idx_te])
        at_t0 = empirical_cost(y_te, amounts_te, p_te, t0, costs)
        cov = segment_fraud_coverage(y_te, amounts_te, seg_te, p_te >= t0)
        return {
            "pr_auc": float(pr_auc(y_te, p_te)),
            "ece": float(ece(y_te, p_te)),
            "recall": float(at_t0["recall"]),
            "n_flagged": int(at_t0["n_flagged"]),
            "total_cost": float(at_t0["total_cost"]),
            "zero_coverage_segments": int(
                ((cov["n_fraud"] > 0) & (cov["n_fraud_caught"] == 0)).sum()
            ),
        }

    initial_eval = _test_eval(model0)

    state = {
        arm: {
            "model": model0,
            "pool_idx": idx_init.copy(),
            "pool_labels": y_obs[idx_init].copy(),
            "fn_labels_cum": 0,
        }
        for arm in ARMS
    }

    rows: list[dict] = []
    for arm in ARMS:  # round 0: every arm starts from the identical model/pool
        rows.append(
            {
                "round": 0,
                "arm": arm,
                "model_updated": True,
                "pool_n": int(idx_init.size),
                "pool_labelled_prevalence": float(y_obs[idx_init].mean()),
                "pool_true_prevalence": float(y_obs[idx_init].mean()),
                "n_alerts": 0,
                "n_reviewed": 0,
                "n_fraud_reviewed": 0,
                "round_fraud": 0,
                "round_known_fraud": 0,
                "observed_recall": float("nan"),
                "label_coverage": float("nan"),
                "false_negative_labels_cum": 0,
                "test_pr_auc": initial_eval["pr_auc"],
                "test_ece": initial_eval["ece"],
                "test_recall_at_t0": initial_eval["recall"],
                "test_alerts_at_t0": initial_eval["n_flagged"],
                "test_cost_at_t0": initial_eval["total_cost"],
            }
        )

    for r, idx_round in enumerate(rounds, start=1):
        y_round = y_obs[idx_round]
        round_fraud = int(y_round.sum())
        for arm in ARMS:
            st = state[arm]
            p_round = st["model"].predict_calibrated(df.iloc[idx_round])
            reviewed = select_reviews(
                p_round, amounts[idx_round], t0, config.reviews_per_round
            )
            labels, labelled = apply_label_policy(arm, y_round, reviewed, surfaced[idx_round])

            st["pool_idx"] = np.concatenate([st["pool_idx"], idx_round[labelled]])
            st["pool_labels"] = np.concatenate([st["pool_labels"], labels[labelled]])
            fn_round = int((labelled & (labels == 0) & (y_round == 1)).sum())
            st["fn_labels_cum"] += fn_round

            st["model"], updated = _fit_on_pool(
                df, st["pool_idx"], st["pool_labels"], template, config, st["model"]
            )
            test = _test_eval(st["model"])

            n_fraud_reviewed = int(y_round[reviewed].sum())
            known = int((labelled & (labels == 1)).sum())
            covered = int((labelled & (labels == 1) & (y_round == 1)).sum())
            rows.append(
                {
                    "round": r,
                    "arm": arm,
                    "model_updated": bool(updated),
                    "pool_n": int(st["pool_idx"].size),
                    "pool_labelled_prevalence": float(st["pool_labels"].mean()),
                    "pool_true_prevalence": float(y_obs[st["pool_idx"]].mean()),
                    "n_alerts": int((p_round >= t0).sum()),
                    "n_reviewed": int(reviewed.sum()),
                    "n_fraud_reviewed": n_fraud_reviewed,
                    "round_fraud": round_fraud,
                    "round_known_fraud": known,
                    "observed_recall": (n_fraud_reviewed / known) if known else float("nan"),
                    "label_coverage": (covered / round_fraud) if round_fraud else float("nan"),
                    "false_negative_labels_cum": int(st["fn_labels_cum"]),
                    "test_pr_auc": test["pr_auc"],
                    "test_ece": test["ece"],
                    "test_recall_at_t0": test["recall"],
                    "test_alerts_at_t0": test["n_flagged"],
                    "test_cost_at_t0": test["total_cost"],
                }
            )

    def _days(idx: np.ndarray) -> tuple[float, float]:
        d = df["ts_days"].to_numpy()
        return float(d[idx[0]]), float(d[idx[-1]])

    return {
        "config": config,
        "t_star_frozen": float(t0),
        "windows": {
            "initial": _days(idx_init),
            "rounds": [_days(idx_round) for idx_round in rounds],
            "test": _days(idx_te),
            "n_initial": int(idx_init.size),
            "n_rounds": [int(idx_round.size) for idx_round in rounds],
            "n_test": int(idx_te.size),
        },
        "initial_eval": initial_eval,
        "trajectory": pd.DataFrame(rows, columns=TRAJECTORY_COLUMNS),
        "final_models": {arm: state[arm]["model"] for arm in ARMS},
        "final_eval": {arm: _test_eval(state[arm]["model"]) for arm in ARMS},
    }


def feedback_lines(analysis: dict) -> list[str]:
    """ASCII-only feedback-loop report for the CLI and the README."""
    w = analysis["windows"]
    cfg = analysis["config"]
    init = analysis["initial_eval"]
    traj = analysis["trajectory"]
    final_round = int(traj["round"].max())
    lines = [
        "FEEDBACK LOOP - what happens when reviewed alerts become the training labels?",
        "(synthetic data; the selective-labels findings are properties of the generator,",
        "and capacity/chargeback rates are labelled assumptions)",
        f"  Initial model: days {w['initial'][0]:.0f}-{w['initial'][1]:.0f} "
        f"(n={w['n_initial']:,}, fully labelled history), then {len(w['rounds'])} deployment "
        f"rounds of ~{w['n_rounds'][0]:,} rows to day {w['rounds'][-1][1]:.0f}.",
        f"  FROZEN policy: alert at calibrated p >= t0*={analysis['t_star_frozen']:.3f}, review "
        f"top {cfg.reviews_per_round} alerts per round by p x amount. Only the labelling",
        "  policy differs by arm; every arm retrains each round on its own labelled pool.",
        f"  Judged on the shared test window (days {w['test'][0]:.0f}-{w['test'][1]:.0f}, "
        f"n={w['n_test']:,}): initial model PR-AUC {init['pr_auc']:.3f}, "
        f"recall {init['recall']:.2f} at t0*.",
        "",
        f"  {'arm (labels for retraining)':<28} {'pool n':>8} {'lab prev':>9} "
        f"{'FN labels':>9} {'PR-AUC':>7} {'ECE':>7} {'recall':>7} {'alerts':>7} {'0-cov':>6}",
        f"  {'-' * 28} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 7} "
        f"{'-' * 7} {'-' * 6}",
    ]
    last = traj[traj["round"] == final_round].set_index("arm")
    for arm in ARMS:
        row = last.loc[arm]
        fin = analysis["final_eval"][arm]
        lines.append(
            f"  {arm:<28} {int(row['pool_n']):>8,} {row['pool_labelled_prevalence']:>9.2%} "
            f"{int(row['false_negative_labels_cum']):>9,} {fin['pr_auc']:>7.3f} "
            f"{fin['ece']:>7.4f} {fin['recall']:>7.2f} {fin['n_flagged']:>7,} "
            f"{fin['zero_coverage_segments']:>6}"
        )
    al = traj[(traj["arm"] == "assume_legit") & (traj["round"] > 0)]
    obs = al["observed_recall"].dropna()
    cov = al["label_coverage"].dropna()
    lines += [
        "",
        "  The selective-labels illusion, measured: the assume_legit arm's dashboard recall",
        f"  (fraud caught / fraud it KNOWS about) reads "
        f"{obs.mean():.0%} every round - by construction -",
        f"  while its true label coverage is {cov.mean():.0%} on average "
        f"({int(al['false_negative_labels_cum'].iloc[-1]):,} frauds entered its training pool",
        "  labelled LEGITIMATE). The misses are invisible precisely because they were never",
        "  reviewed: the model grades its own homework.",
        f"  Reviewed-only pool after round {final_round}: "
        f"{int(last.loc['reviewed_only', 'pool_n']):,} rows at "
        f"{last.loc['reviewed_only', 'pool_labelled_prevalence']:.2%} labelled prevalence vs "
        f"{last.loc['full_labels', 'pool_labelled_prevalence']:.2%} in the population pool -",
        "  a training sample chosen by the model's own opinions.",
        f"  Where the censoring lands: final-round PR-AUC spans only "
        f"{min(v['pr_auc'] for v in analysis['final_eval'].values()):.3f}-"
        f"{max(v['pr_auc'] for v in analysis['final_eval'].values()):.3f} across arms - the",
        "  RANKING largely survives, because the clean initial history anchors it. The damage",
        "  is in the PROBABILITIES: training on fraud relabelled as legitimate drags every",
        "  predicted p down, so at the frozen t0* the assume_legit arm fires "
        f"{analysis['final_eval']['assume_legit']['n_flagged']:,} alerts vs "
        f"{analysis['final_eval']['full_labels']['n_flagged']:,} (full_labels)",
        f"  and catches {analysis['final_eval']['assume_legit']['recall']:.0%} of test fraud vs "
        f"{analysis['final_eval']['full_labels']['recall']:.0%}. Every decision layer in this "
        "repo consumes calibrated",
        "  probabilities - label censoring poisons exactly the part the decisions trust.",
        "  (Round 1 reviews are identical across arms - all four start from the same model",
        "  and the same frozen policy. Every later divergence, including WHICH alerts get",
        "  reviewed, is the labelling loop feeding back into the model.)",
        "",
        "  READ HONESTLY: chargebacks are the escape valve - most missed fraud eventually",
        "  self-reports - but here they land instantly; real 30-90 day delays make every",
        "  censored arm look better than production would. The oracle arm exists only",
        "  because the data is synthetic. These are labelled assumptions, not measurements.",
    ]
    return lines


def save_feedback_csv(analysis: dict, path: str) -> str:
    """Write the per-round, per-arm trajectory to a tidy CSV. Returns the path."""
    analysis["trajectory"].to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path
