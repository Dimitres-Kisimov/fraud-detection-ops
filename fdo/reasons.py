"""Reason codes: why THIS alert fired, in the shape regulated explanations take.

Every other surface in this repo answers a population question - how well does
the model rank (``fdo/evaluate.py``), where should the threshold sit
(``fdo/threshold.py``), who gets reviewed (``fdo/queue_opt.py``), has the world
moved (``fdo/drift.py``). An analyst opening a single alert asks a different
one: *why is this transaction in my queue?* Regulated lending answers that with
a short list of PRINCIPAL REASONS; fraud shops have converged on the same
artifact because "the model said so" does not survive a QA review, a customer
call, or a model-risk audit.

This module computes that list from the model the repo already trained. It adds
no new model, no surrogate, no sampling, and no randomness.

METHOD - the decomposition is exact, not an approximation
---------------------------------------------------------
The champion is a logistic regression on standardized features, so its decision
score is a sum:

    z(x) = theta_0 + sum_j theta_j * xtilde_j

Fix a REFERENCE PROFILE r (here: the mean standardized feature vector of the
TRAINING window - "the average transaction the model was trained on"). Define

    contribution_j(x) = theta_j * (xtilde_j - r_j)
    z(x) = z(r) + sum_j contribution_j(x)                        (local accuracy)

Those contributions are not a heuristic: for a model that is linear in its
inputs, the Shapley value of feature j under an interventional reference IS
``theta_j * (x_j - r_j)`` exactly. ``shapley_bruteforce`` enumerates all 2^m
coalitions and the tests assert the closed form agrees to machine precision, so
the claim is verified in CI rather than cited.

Correlated columns are summed into REASON GROUPS before anything is ranked. The
design matrix encodes one clock three ways (``hour_sin``, ``hour_cos``,
``is_night``), account age twice (``log_tenure``, ``is_new_account``) and the
merchant category as seven dummies. Splitting credit across redundant encodings
is how explanation layers produce nonsense like "hour_cos" on a customer-facing
notice; group contributions are additive by construction (a coalition's Shapley
value in a linear model is the sum of its members'), so grouping costs nothing
and is the honest unit of explanation.

WHAT THE READ-OUTS ARE
----------------------
- Per alert: up to ``max_reasons`` principal reasons (largest positive group
  contributions), in logits, plus REASONS-TO-CROSS - the smallest number of
  those reasons whose removal drops the score back under the alert threshold.
- Per reason group, across every alert at t*: how often it is the principal
  reason, how often it appears at all, its mean contribution, and the confirmed
  fraud rate of the alerts it heads.
- Stability: the two models the pipeline already trains (weighted BCE and
  focal) rank alerts almost identically; this reports how often they still
  disagree about the principal reason. Ranking agreement is not explanation
  agreement.

HONESTY NOTES - what a contribution does NOT mean
-------------------------------------------------
- LOGITS, not probabilities and not dollars. The same +1.2 contribution moves
  the calibrated probability by wildly different amounts depending on where the
  rest of the score sits; there is no conversion to "$ of risk" here.
- RELATIVE TO A REFERENCE. Every number answers "compared with the average
  transaction in the training window". Change the reference and every
  contribution changes. The reference is reported next to the numbers.
- NOT CAUSAL, NOT A COUNTERFACTUAL ABOUT THE WORLD. REASONS-TO-CROSS is a
  statement about the model's own arithmetic (set these groups back to the
  reference profile and the score falls below the cut), not a claim that the
  transaction would have been legitimate.
- IT EXPLAINS THE MODEL, NOT THE FRAUD. This repo's generator deliberately
  withholds an amount x gift/digital interaction from the design matrix (see
  ``fdo/data.py`` and the oracle gap in the README). A reason code can be a
  faithful account of the model's arithmetic and still be an incomplete account
  of why the transaction was fraudulent.
- NOT AN ADVERSE-ACTION NOTICE. The four-reason format is borrowed from ECOA /
  Regulation B practice in credit decisioning because it is a good discipline;
  this is not legal advice, and a fraud alert is not a credit denial. Anyone
  shipping notices to customers needs counsel, not this file.
- SYNTHETIC DATA. Labels, amounts and patterns all come from the seeded
  generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from fdo.data import CATEGORIES, make_features
from fdo.palette import (
    ALERT_AMBER,
    BASELINE,
    CLEARED_GREEN,
    FRAUD_RED,
    GRID,
    INK,
    INK_2,
    MUTED,
    SURFACE,
)

# Raw design-matrix columns grouped into the concepts an analyst reads. Every
# column produced by ``fdo.data.make_features`` must land in exactly one group;
# ``group_contributions`` raises if the design matrix ever grows a column that
# is not mapped here.
REASON_GROUPS: dict[str, tuple[str, ...]] = {
    "transaction_amount": ("log_amount",),
    "time_of_day": ("hour_sin", "hour_cos", "is_night"),
    "account_age": ("log_tenure", "is_new_account"),
    "velocity_24h": ("velocity_24h",),
    "device_change": ("device_change",),
    "merchant_category": tuple(f"cat_{c}" for c in CATEGORIES if c != "grocery"),
}

# Plain-language phrasing for the CLI read-out (the notice line an analyst or a
# QA reviewer would read). Deliberately describes the DIRECTION of the model's
# score movement, never a verdict about the customer.
REASON_PHRASING: dict[str, str] = {
    "transaction_amount": "transaction amount vs the training-window average",
    "time_of_day": "time of day the transaction was made",
    "account_age": "how long the account has existed",
    "velocity_24h": "number of transactions in the last 24h",
    "device_change": "transaction came from a changed device",
    "merchant_category": "merchant category of the transaction",
}


@dataclass(frozen=True)
class ReasonConfig:
    """Explanation policy. All values are POLICY CHOICES, not measurements."""

    max_reasons: int = 4
    reference: str = "train_mean"

    def describe(self) -> list[str]:
        return [
            f"POLICY: at most {self.max_reasons} principal reasons per alert "
            "(the cap regulated adverse-action notices use; more reasons is not more "
            "explanation).",
            "POLICY: contributions are measured against the "
            f"{self.reference.replace('_', ' ')} reference profile - 'compared with the "
            "average transaction the model trained on'.",
            "POLICY: only risk-INCREASING groups can be principal reasons; protective "
            "groups are reported separately, never as a reason for the alert.",
        ]


def reference_profile(model, df_reference: pd.DataFrame) -> np.ndarray:
    """Mean STANDARDIZED feature vector of a reference window.

    With the model's train-fitted standardizer and the training window itself,
    this is the zero vector up to floating point (asserted in the tests) - which
    is the point: the reference is stated explicitly instead of being an
    accident of standardization.
    """
    X, _ = make_features(df_reference)
    return model.scaler.transform(X).mean(axis=0)


def baseline_logit(model, reference: np.ndarray) -> float:
    """Score of the reference profile itself: z(r) = theta_0 + theta[1:] @ r."""
    if model.theta is None:
        raise RuntimeError("model not fitted")
    return float(model.theta[0] + model.theta[1:] @ np.asarray(reference, dtype=np.float64))


def contributions(model, df: pd.DataFrame, reference: np.ndarray) -> np.ndarray:
    """Per-row, per-feature contributions ``theta_j * (xtilde_j - r_j)``.

    Returns an ``(n_rows, n_features)`` matrix whose rows sum to
    ``z(x) - z(r)`` exactly (local accuracy; asserted in the tests).
    """
    if model.theta is None:
        raise RuntimeError("model not fitted")
    X, names = make_features(df)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (len(names),):
        raise ValueError(f"reference must have shape ({len(names)},), got {reference.shape}")
    Xs = model.scaler.transform(X)
    return (Xs - reference) * model.theta[1:]


def shapley_bruteforce(theta: np.ndarray, x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Exact Shapley values of a LINEAR value function, by enumeration.

    Reference implementation used to verify the closed form, not to run in
    production: with ``v(S) = theta_0 + sum_{j in S} theta_j x_j + sum_{j not in
    S} theta_j r_j`` (an interventional reference: features outside the
    coalition are held at the reference profile), the Shapley value of feature
    j is the weighted average of its marginal contributions over all coalitions.
    For this v it must equal ``theta_j * (x_j - r_j)``.

    Costs O(2^m); capped at 12 features on purpose - it exists to be checked
    against, not to be scaled.
    """
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    m = x.size
    if not (theta.size == x.size == reference.size):
        raise ValueError("theta, x and reference must share one shape")
    if m > 12:
        raise ValueError("shapley_bruteforce is exponential; use it on <= 12 features")

    def value(subset: frozenset[int]) -> float:
        vals = np.where([j in subset for j in range(m)], x, reference)
        return float(theta @ vals)

    phi = np.zeros(m, dtype=np.float64)
    others = list(range(m))
    for j in range(m):
        rest = [k for k in others if k != j]
        for size in range(len(rest) + 1):
            weight = factorial(size) * factorial(m - size - 1) / factorial(m)
            for subset in combinations(rest, size):
                s = frozenset(subset)
                phi[j] += weight * (value(s | {j}) - value(s))
    return phi


def group_contributions(
    contrib: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Sum raw-column contributions into reason groups.

    Exact by additivity: in a linear model the Shapley value of a coalition of
    features equals the sum of its members', so grouping loses nothing and
    stops redundant encodings (three clock columns, seven category dummies)
    from splitting the credit for one concept.
    """
    contrib = np.asarray(contrib, dtype=np.float64)
    if contrib.shape[1] != len(feature_names):
        raise ValueError("contribution matrix and feature_names disagree on width")
    mapped = {c for cols in REASON_GROUPS.values() for c in cols}
    unmapped = [c for c in feature_names if c not in mapped]
    if unmapped:
        raise ValueError(f"design-matrix columns are not mapped to a reason group: {unmapped}")
    names = list(REASON_GROUPS)
    index = {c: i for i, c in enumerate(feature_names)}
    out = np.zeros((contrib.shape[0], len(names)), dtype=np.float64)
    for gi, g in enumerate(names):
        cols = [index[c] for c in REASON_GROUPS[g] if c in index]
        if cols:
            out[:, gi] = contrib[:, cols].sum(axis=1)
    return out, names


def threshold_logit(model, t_star: float) -> float:
    """Raw-score cut equivalent to the calibrated probability threshold t*.

    The calibrated probability is ``sigmoid(a*z + b)``, so ``p >= t*`` is
    ``z >= (logit(t*) - b) / a`` whenever a > 0 (the monotone case the repo
    already relies on for PR-AUC being calibration-invariant).
    """
    if not 0.0 < t_star < 1.0:
        raise ValueError("t_star must lie strictly between 0 and 1")
    if model.platt_a <= 0.0:
        raise ValueError("Platt slope <= 0: the calibrated score is not monotone in z")
    return float((np.log(t_star / (1.0 - t_star)) - model.platt_b) / model.platt_a)


def reasons_to_cross(group_contrib: np.ndarray, z: np.ndarray, z_cut: float) -> np.ndarray:
    """Smallest k such that removing the k largest positive group contributions
    (setting those groups back to the reference profile) drops the score below
    the alert cut. ``-1`` when even removing every positive group leaves the
    score above the cut - i.e. the reference profile itself would alert.

    A statement about the model's arithmetic, NOT a counterfactual about the
    transaction.
    """
    group_contrib = np.asarray(group_contrib, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if group_contrib.shape[0] != z.shape[0]:
        raise ValueError("group_contrib and z must have the same number of rows")
    positives = np.where(group_contrib > 0.0, group_contrib, 0.0)
    order = np.argsort(-positives, axis=1, kind="stable")
    ordered = np.take_along_axis(positives, order, axis=1)
    remaining = z[:, None] - np.cumsum(ordered, axis=1)
    below = remaining < z_cut
    k = np.where(below.any(axis=1), below.argmax(axis=1) + 1, -1)
    return k.astype(np.int64)


def principal_reasons(
    group_contrib: np.ndarray, group_names: list[str], max_reasons: int
) -> list[list[tuple[str, float]]]:
    """Up to ``max_reasons`` risk-INCREASING groups per row, largest first.

    Ties break on the fixed group order (stable sort), so the output is
    deterministic. Rows with fewer positive groups get a shorter list - padding
    a notice with reasons that did not push the score up would be a lie.
    """
    group_contrib = np.asarray(group_contrib, dtype=np.float64)
    order = np.argsort(-group_contrib, axis=1, kind="stable")
    out: list[list[tuple[str, float]]] = []
    for i in range(group_contrib.shape[0]):
        row: list[tuple[str, float]] = []
        for gi in order[i, :max_reasons]:
            value = float(group_contrib[i, gi])
            if value <= 0.0:
                break
            row.append((group_names[gi], value))
        out.append(row)
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (the Spearman convention), NumPy only."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # average ranks within tied groups
    sorted_vals = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on average ranks), NumPy only."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def reason_code_frame(
    df_alerts: pd.DataFrame,
    group_contrib: np.ndarray,
    group_names: list[str],
    z: np.ndarray,
    p: np.ndarray,
    z_baseline: float,
    z_cut: float,
    config: ReasonConfig,
) -> pd.DataFrame:
    """Per-alert reason-code table - the analyst-facing artifact."""
    reasons = principal_reasons(group_contrib, group_names, config.max_reasons)
    k_cross = reasons_to_cross(group_contrib, z, z_cut)
    rows = []
    for i in range(len(df_alerts)):
        row = {
            "timestamp": df_alerts["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),
            "amount": float(df_alerts["amount"].iloc[i]),
            "merchant_cat": str(df_alerts["merchant_cat"].iloc[i]),
            "p_fraud_calibrated": float(p[i]),
            "score_logit": float(z[i]),
            "baseline_logit": float(z_baseline),
            "reasons_to_cross_threshold": int(k_cross[i]),
        }
        for slot in range(config.max_reasons):
            name, value = (reasons[i][slot] if slot < len(reasons[i]) else ("", float("nan")))
            row[f"reason_{slot + 1}"] = name
            row[f"contribution_{slot + 1}"] = value
        row["is_fraud_observed"] = int(df_alerts["is_fraud"].iloc[i])
        rows.append(row)
    return pd.DataFrame(rows)


def reason_summary_frame(
    group_contrib: np.ndarray,
    group_names: list[str],
    y: np.ndarray,
    amounts: np.ndarray,
    config: ReasonConfig,
) -> pd.DataFrame:
    """Per-reason-group population read-out across every alert at t*."""
    reasons = principal_reasons(group_contrib, group_names, config.max_reasons)
    n = len(reasons)
    principal_of = np.array([r[0][0] if r else "" for r in reasons])
    in_top = {g: np.zeros(n, dtype=bool) for g in group_names}
    for i, row in enumerate(reasons):
        for name, _ in row:
            in_top[name][i] = True
    y = np.asarray(y, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    rows = []
    for gi, g in enumerate(group_names):
        is_principal = principal_of == g
        n_principal = int(is_principal.sum())
        col = group_contrib[:, gi]
        rows.append(
            {
                "reason_group": g,
                "share_principal_reason": float(n_principal / n) if n else float("nan"),
                "share_listed_in_top_reasons": float(in_top[g].mean()),
                "mean_contribution_all_alerts": float(col.mean()),
                "mean_contribution_when_principal": (
                    float(col[is_principal].mean()) if n_principal else float("nan")
                ),
                "alerts_headed": n_principal,
                "confirmed_fraud_rate_when_principal": (
                    float(y[is_principal].mean()) if n_principal else float("nan")
                ),
                "mean_amount_when_principal": (
                    float(amounts[is_principal].mean()) if n_principal else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "share_principal_reason", ascending=False, kind="stable"
    ).reset_index(drop=True)


def run_reasons(results: dict, config: ReasonConfig | None = None) -> dict:
    """Full reason-code analysis from one ``run_pipeline`` result.

    Explains the alerts the shipped policy actually raises: every test-window
    transaction at or above the cost threshold t*, plus the ``capacity`` rows
    the queue optimizer selects for review (the list an analyst opens).
    """
    config = config or ReasonConfig()
    df = results["df"]
    idx_tr, idx_te = results["splits"]["train"], results["splits"]["test"]
    primary = results["models"][results["primary_name"]]
    t_star = float(results["thresholds"]["threshold_star"])

    reference = reference_profile(primary, df.iloc[idx_tr])
    z_baseline = baseline_logit(primary, reference)
    z_cut = threshold_logit(primary, t_star)

    test = df.iloc[idx_te]
    p_test = results["p_test"]
    z_test = primary.decision_scores(test)
    alert_mask = p_test >= t_star
    alerts = test.loc[alert_mask]
    contrib_alerts = contributions(primary, alerts, reference)
    g_alerts, group_names = group_contributions(contrib_alerts, primary.feature_names)
    z_alerts = z_test[alert_mask]

    summary = reason_summary_frame(
        g_alerts,
        group_names,
        alerts["is_fraud"].to_numpy(dtype=np.float64),
        alerts["amount"].to_numpy(dtype=np.float64),
        config,
    )

    # The queue an analyst actually works: the optimizer's selection, ordered
    # by expected value so the CSV reads top-down like the worklist.
    sel = np.asarray(results["queue"]["selected_milp"], dtype=np.int64)
    ev = p_test[sel] * test["amount"].to_numpy(dtype=np.float64)[sel]
    sel = sel[np.argsort(-ev, kind="stable")]
    queue = test.iloc[sel]
    contrib_queue = contributions(primary, queue, reference)
    g_queue, _ = group_contributions(contrib_queue, primary.feature_names)
    queue_frame = reason_code_frame(
        queue, g_queue, group_names, z_test[sel], p_test[sel], z_baseline, z_cut, config
    )
    queue_frame.insert(0, "queue_rank", np.arange(1, len(queue_frame) + 1, dtype=np.int64))

    # Stability: the OTHER trained model, same data, same feature set.
    other_name = next(n for n in results["models"] if n != results["primary_name"])
    other = results["models"][other_name]
    other_reference = reference_profile(other, df.iloc[idx_tr])
    g_other, _ = group_contributions(
        contributions(other, alerts, other_reference), other.feature_names
    )
    top_primary = [r[0][0] if r else "" for r in principal_reasons(g_alerts, group_names, 1)]
    top_other = [r[0][0] if r else "" for r in principal_reasons(g_other, group_names, 1)]
    agree = float(np.mean([a == b for a, b in zip(top_primary, top_other, strict=True)]))
    p_other = other.predict_calibrated(test)
    stability = {
        "other_model": other_name,
        "score_spearman": spearman(p_test, p_other),
        "top_reason_agreement": agree,
        "n_alerts": int(alert_mask.sum()),
    }

    # What the analyst actually opens: the queue is chosen by p x amount, which
    # is a different mix of reasons than the alert population it is drawn from.
    # Ties fall to the fixed group order (argmax takes the first maximum).
    queue_counts = [int((queue_frame["reason_1"] == g).sum()) for g in group_names]
    top_i = int(np.argmax(queue_counts))
    queue_top = group_names[top_i]
    queue_view = {
        "top_reason": queue_top,
        "share_in_queue": float(queue_counts[top_i] / len(queue_frame)),
        "share_in_alerts": float(
            summary.loc[summary["reason_group"] == queue_top, "share_principal_reason"].iloc[0]
        ),
        "queue_fraud_rate": float(queue_frame["is_fraud_observed"].mean()),
        "n_reviews": int(len(queue_frame)),
    }

    k_cross = reasons_to_cross(g_alerts, z_alerts, z_cut)
    return {
        "config": config,
        "t_star": t_star,
        "z_cut": z_cut,
        "z_baseline": z_baseline,
        "reference_window": (float(df.iloc[idx_tr]["ts_days"].min()),
                             float(df.iloc[idx_tr]["ts_days"].max())),
        "group_names": group_names,
        "summary": summary,
        "queue_frame": queue_frame,
        "queue_view": queue_view,
        "n_alerts": int(alert_mask.sum()),
        "n_test": int(len(test)),
        "alert_fraud_rate": float(alerts["is_fraud"].mean()),
        "mean_reasons_listed": float(
            np.mean([len(r) for r in principal_reasons(g_alerts, group_names, config.max_reasons)])
        ),
        "k_cross_median": float(np.median(k_cross[k_cross > 0])),
        "k_cross_share_single": float(np.mean(k_cross == 1)),
        "k_cross_unexplained": int((k_cross < 0).sum()),
        "stability": stability,
        "primary_model": results["primary_name"],
    }


def reason_lines(analysis: dict) -> list[str]:
    """ASCII-only CLI read-out (no wall-clock, no RNG)."""
    cfg: ReasonConfig = analysis["config"]
    s = analysis["summary"]
    st = analysis["stability"]
    qv = analysis["queue_view"]
    d0, d1 = analysis["reference_window"]
    lines = [
        f"Reason codes: {analysis['n_alerts']:,} alerts at t* = {analysis['t_star']:.3f} on the "
        f"held-out test window ({analysis['n_test']:,} transactions, "
        f"{analysis['alert_fraud_rate']:.1%} of alerts confirmed fraud).",
        f"Model: {analysis['primary_model']} logistic regression. Contributions are LOGITS "
        f"against the training-window reference profile (days {d0:.0f}-{d1:.0f}); the reference "
        f"itself scores z = {analysis['z_baseline']:.3f}, the alert cut sits at "
        f"z = {analysis['z_cut']:.3f}.",
        "Decomposition is exact for this model family: contributions sum to z(x) - z(reference) "
        "and equal the Shapley values of the score (verified against brute-force enumeration "
        "in the tests).",
        "",
        f"  {'reason group':<20} {'principal':>10} {'listed':>8} {'mean contrib':>13} "
        f"{'fraud rate':>11} {'mean amount':>12}",
        f"  {'-' * 20} {'-' * 10} {'-' * 8} {'-' * 13} {'-' * 11} {'-' * 12}",
    ]
    for _, row in s.iterrows():
        fraud = row["confirmed_fraud_rate_when_principal"]
        amount = row["mean_amount_when_principal"]
        fraud_cell = f"{fraud:.1%}" if np.isfinite(fraud) else "n/a"
        amount_cell = f"${amount:,.0f}" if np.isfinite(amount) else "n/a"
        lines.append(
            f"  {row['reason_group']:<20} {row['share_principal_reason']:>9.1%} "
            f"{row['share_listed_in_top_reasons']:>7.1%} "
            f"{row['mean_contribution_all_alerts']:>+13.3f} "
            f"{fraud_cell:>11} {amount_cell:>12}"
        )
    top = s.iloc[0]
    lines += [
        "",
        f"Principal reason on {top['share_principal_reason']:.1%} of alerts: "
        f"{top['reason_group']} ({REASON_PHRASING[top['reason_group']]}); the alerts it heads "
        f"are confirmed fraud {top['confirmed_fraud_rate_when_principal']:.1%} of the time "
        f"against {analysis['alert_fraud_rate']:.1%} across all alerts.",
        f"Alerts list {analysis['mean_reasons_listed']:.2f} reasons on average (cap "
        f"{cfg.max_reasons}); the median alert needs {analysis['k_cross_median']:.0f} of them "
        f"removed to fall back under the threshold, and "
        f"{analysis['k_cross_share_single']:.1%} rest on a single reason.",
        f"QUEUE VIEW: the {qv['n_reviews']} reviews the optimizer actually selects are headed "
        f"by {qv['top_reason']} on {qv['share_in_queue']:.1%} of rows against "
        f"{qv['share_in_alerts']:.1%} across all alerts - ranking by p x amount rewrites the "
        f"reason mix an analyst sees (that worklist is {qv['queue_fraud_rate']:.1%} confirmed "
        "fraud).",
        f"STABILITY: {st['other_model']} ranks these transactions almost identically "
        f"(Spearman {st['score_spearman']:.4f} on calibrated probabilities) yet names a "
        f"different principal reason on {1.0 - st['top_reason_agreement']:.1%} of alerts. "
        "Ranking agreement is not explanation agreement.",
        "",
        "READ HONESTLY:",
        "  - Contributions are logits against a stated reference, not probabilities, not "
        "dollars, and not causal.",
        "  - 'Reasons to cross' is arithmetic inside the model (set these groups back to the "
        "reference and the score drops under the cut), not a claim the transaction would have "
        "been legitimate.",
        "  - Redundant encodings (three clock columns, seven category dummies) are summed into "
        "groups first; per-column contributions would split one concept across several lines.",
        "  - It explains the model, not the fraud: the model has known headroom against the "
        "oracle, so a faithful account of the score can still be an incomplete account of why "
        "the transaction was fraudulent.",
        "  - Format borrowed from ECOA / Regulation B adverse-action practice as a discipline. "
        "A fraud alert is not a credit denial and this is not legal advice.",
    ]
    lines += ["", "  " + " ".join(cfg.describe())]
    return lines


def save_reason_summary_csv(analysis: dict, path: str) -> str:
    """Write the per-reason-group summary table. Returns the path."""
    analysis["summary"].to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path


def save_reason_codes_csv(analysis: dict, path: str) -> str:
    """Write the per-alert reason codes for the review queue. Returns the path."""
    analysis["queue_frame"].to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path


def _nice_ceiling(value: float) -> float:
    """Round a positive number up to 1/2/5 x 10^k for a clean axis top."""
    if value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    base = 10.0**exp
    for mult in (1.0, 2.0, 5.0, 10.0):
        if value <= mult * base:
            return float(mult * base)
    return float(10.0 * base)


def save_reason_codes_svg(analysis: dict, path: str) -> str:
    """Hand-draw the reason-code read-out as a standalone SVG.

    Two panels on one row order, because they answer two different questions
    and share no scale: how OFTEN each reason heads an alert (one series, alert
    amber), and how those alerts RESOLVE - confirmed-fraud rate against the
    all-alert average, diverging into confirmed-fraud red above it and cleared
    green below. Every mark carries its value as a direct label, which is also
    what licenses the sub-3:1 green fill (see fdo/palette.py).

    Pure string construction - no matplotlib, no RNG, no timestamps - so the
    committed figure is byte-deterministic across machines.
    """
    s = analysis["summary"]
    base_rate = float(analysis["alert_fraud_rate"])
    groups = list(s["reason_group"])
    shares = s["share_principal_reason"].to_numpy(dtype=np.float64)
    rates = s["confirmed_fraud_rate_when_principal"].to_numpy(dtype=np.float64)
    headed = s["alerts_headed"].to_numpy(dtype=np.int64)

    W, H = 920, 452
    row_h = 40.0
    top_row = 122.0
    a_x, a_w = 214.0, 236.0          # panel A plot box
    b_x, b_w = 566.0, 306.0          # panel B plot box
    b_zero = b_x + b_w / 2.0
    a_max = _nice_ceiling(float(shares.max()))
    dev = np.where(np.isfinite(rates), rates - base_rate, 0.0)
    b_max = _nice_ceiling(float(np.abs(dev).max()) * 1.25)

    def row_y(i: int) -> float:
        return top_row + i * row_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="Principal reason codes for the alerts at the cost threshold">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="28" y="30" font-size="15" fill="{INK}" font-weight="700">'
        f'Why the alerts fired: principal reason codes at t* = {analysis["t_star"]:.3f}</text>',
        f'<text x="28" y="50" font-size="12" fill="{INK_2}">'
        f'{analysis["n_alerts"]:,} alerts on the held-out test window; contributions are logits '
        f'against the training-window average transaction - not probabilities, not dollars, '
        f'not causal.</text>',
        f'<text x="{a_x}" y="84" font-size="12.5" fill="{INK}" font-weight="600">'
        f'Share of alerts it heads as principal reason</text>',
        f'<text x="{b_x}" y="84" font-size="12.5" fill="{INK}" font-weight="600">'
        f'Confirmed-fraud rate of those alerts</text>',
        f'<text x="{b_x}" y="100" font-size="11.5" fill="{INK_2}">'
        f'deviation from {base_rate:.1%} across all alerts</text>',
    ]

    # panel A: gridlines at 0 / half / max
    for frac in (0.0, 0.5, 1.0):
        x = a_x + frac * a_w
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_row - 24:.1f}" x2="{x:.1f}" '
            f'y2="{row_y(len(groups) - 1) + 22:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{row_y(len(groups) - 1) + 38:.1f}" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}">{a_max * frac:.0%}</text>'
        )

    # panel B: the all-alert average is the reference the bars diverge from
    parts.append(
        f'<line x1="{b_zero:.1f}" y1="{top_row - 24:.1f}" x2="{b_zero:.1f}" '
        f'y2="{row_y(len(groups) - 1) + 22:.1f}" stroke="{BASELINE}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{b_zero:.1f}" y="{row_y(len(groups) - 1) + 38:.1f}" text-anchor="middle" '
        f'font-size="11" fill="{MUTED}">all alerts {base_rate:.1%}</text>'
    )
    parts.append(
        f'<text x="{b_x:.1f}" y="{row_y(len(groups) - 1) + 38:.1f}" font-size="11" '
        f'fill="{MUTED}">cleared more often</text>'
    )
    parts.append(
        f'<text x="{b_x + b_w:.1f}" y="{row_y(len(groups) - 1) + 38:.1f}" text-anchor="end" '
        f'font-size="11" fill="{MUTED}">fraud more often</text>'
    )

    bar_h = 17.0
    for i, group in enumerate(groups):
        y = row_y(i)
        parts.append(
            f'<text x="{a_x - 14:.1f}" y="{y + 5:.1f}" text-anchor="end" font-size="12.5" '
            f'fill="{INK}">{group.replace("_", " ")}</text>'
        )
        # panel A - one series, alert amber
        w = shares[i] / a_max * a_w
        parts.append(
            f'<rect x="{a_x:.1f}" y="{y - bar_h / 2:.1f}" width="{max(w, 1.0):.1f}" '
            f'height="{bar_h:.1f}" rx="3" fill="{ALERT_AMBER}"/>'
        )
        parts.append(
            f'<text x="{a_x + w + 8:.1f}" y="{y + 5:.1f}" font-size="12" fill="{INK}">'
            f'{shares[i]:.1%}</text>'
        )
        parts.append(
            f'<text x="{a_x + w + 52:.1f}" y="{y + 5:.1f}" font-size="11" fill="{MUTED}">'
            f'n={headed[i]:,}</text>'
        )
        # panel B - diverging around the all-alert fraud rate
        if not np.isfinite(rates[i]):
            continue
        d = dev[i]
        bw = abs(d) / b_max * (b_w / 2.0)
        x0 = b_zero if d >= 0 else b_zero - bw
        fill = FRAUD_RED if d >= 0 else CLEARED_GREEN
        parts.append(
            f'<rect x="{x0:.1f}" y="{y - bar_h / 2:.1f}" width="{max(bw, 1.0):.1f}" '
            f'height="{bar_h:.1f}" rx="3" fill="{fill}"/>'
        )
        if d >= 0:
            parts.append(
                f'<text x="{b_zero + bw + 8:.1f}" y="{y + 5:.1f}" font-size="12" fill="{INK}">'
                f'{rates[i]:.1%}</text>'
            )
        else:
            parts.append(
                f'<text x="{b_zero - bw - 8:.1f}" y="{y + 5:.1f}" text-anchor="end" '
                f'font-size="12" fill="{INK}">{rates[i]:.1%}</text>'
            )

    st = analysis["stability"]
    parts.append(
        f'<line x1="28" y1="{H - 62}" x2="{W - 28}" y2="{H - 62}" stroke="{GRID}" '
        f'stroke-width="1"/>'
    )
    parts.append(
        f'<text x="28" y="{H - 42}" font-size="11.5" fill="{INK_2}">'
        f'Reasons are groups of design-matrix columns (one clock, one account-age pair, seven '
        f'category dummies), summed - exact for a linear score. The median alert needs '
        f'{analysis["k_cross_median"]:.0f} reason(s) removed to fall back under the cut.</text>'
    )
    parts.append(
        f'<text x="28" y="{H - 24}" font-size="11.5" fill="{INK_2}">'
        f'{st["other_model"]} ranks these alerts almost identically (Spearman '
        f'{st["score_spearman"]:.4f}) yet names a different principal reason on '
        f'{1.0 - st["top_reason_agreement"]:.1%} of them. Synthetic data; explains the model, '
        f'not the fraud.</text>'
    )
    parts.append("</svg>\n")
    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    return path
