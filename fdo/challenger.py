"""Champion/challenger harness: is a retrain on the drifted regime worth promoting?

The drift monitor (``fdo/drift.py``) ends with a playbook line every fraud shop
knows: on confirmed concept drift, retrain on a window that includes the shifted
regime, re-fit calibration, re-sweep the threshold, and re-validate coverage.
This module *executes* that playbook and then asks the governance question that
actually gates a model swap in production: does the retrained CHALLENGER beat
the incumbent CHAMPION by enough, on pre-declared criteria, to promote?

Design - one variable changes:

- CHAMPION: the pipeline's primary model exactly as shipped. Trained on the
  first 70% of the timeline, calibrated and threshold-swept on the next 15%.
- CHALLENGER: the SAME model family, loss, and hyperparameters, retrained on a
  rolling window that drops the oldest ``drop_frac`` of the timeline and
  extends through most of the old validation window (so it includes more of
  the post-day-60 drifted regime). It early-stops, Platt-calibrates, and
  re-sweeps its own threshold t* on a FRESH validation slice carved from the
  last ``val_frac`` of the pre-test timeline. Architecture competition is
  deliberately out of scope (see README limitations); the only question is
  whether the *training window* should move.
- Both models are judged once, on the IDENTICAL held-out test window the
  champion pipeline already uses (verified index-for-index at runtime).

Read-outs, in the order a model-risk reviewer asks for them:

1. Ranking and calibration on test (PR-AUC / ROC-AUC / ECE / Brier).
2. Economics: each model at its OWN re-swept cost threshold (same labelled
   cost assumptions as ``fdo/threshold.py``).
3. SWAP-SET analysis - the classic champion/challenger read-out: which test
   alerts are flagged by both, champion-only (swapped out), challenger-only
   (swapped in), or neither, with fraud counts and dollar value per cell.
4. Fixed-capacity queue view: expected and realized recovered value of the
   top-``capacity`` reviews by p*amount under each model's calibrated p
   (the unconstrained queue rule, held fixed so only the scores differ).
5. A gated PROMOTION decision: five pre-declared pass/fail criteria
   (ranking non-inferiority, absolute calibration bound, economics
   non-worse, alert-volume bound, per-segment zero-coverage safety) and a
   PROMOTE/HOLD verdict.

HONESTY NOTES:

- Data is synthetic and seeded; both windows inherit every generator
  assumption. Dollar figures rest on the labelled cost ASSUMPTIONS
  ($8/review, missed fraud = its amount) - illustrative planning arithmetic,
  not measured savings.
- The gate thresholds in ``ChallengerConfig`` are declared POLICY KNOBS, not
  statistical guarantees; the harness demonstrates the governance mechanics
  (pre-declared criteria, judged once, out-of-sample), it does not claim
  these five checks are sufficient for a production model swap.
- The verdict is whatever the measurement says. A HOLD is as valid a result
  as a PROMOTE - the point of the gate is to stop reflexive retraining.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fdo.evaluate import summarize_scores
from fdo.fairness import segment_fraud_coverage
from fdo.model import FraudModel, time_split
from fdo.threshold import threshold_report

SWAP_CELLS = ("both_flag", "champion_only (swap-out)", "challenger_only (swap-in)", "neither")


@dataclass(frozen=True)
class ChallengerConfig:
    """Retrain-window shape and promotion-gate knobs. All POLICY ASSUMPTIONS."""

    drop_frac: float = 0.15          # oldest share of the timeline the challenger drops
    val_frac: float = 0.075          # fresh validation slice carved just before test
    pr_auc_epsilon: float = 0.010    # allowed test PR-AUC deficit vs champion
    ece_max: float = 0.020           # absolute post-Platt ECE bound on test
    cost_margin: float = 0.0         # allowed relative cost excess at own t* (0 = non-worse)
    alert_volume_max_ratio: float = 1.5  # challenger alerts <= ratio * champion alerts

    def describe(self) -> list[str]:
        return [
            f"POLICY ASSUMPTION: challenger drops the oldest {self.drop_frac:.0%} of the "
            f"timeline and re-validates on the freshest {self.val_frac:.1%} before test.",
            f"POLICY ASSUMPTION: promote only if test PR-AUC is within {self.pr_auc_epsilon:.3f} "
            f"of the champion, post-Platt ECE <= {self.ece_max:.3f}, cost at its own t* is "
            f"<= champion's x {1.0 + self.cost_margin:.2f}, alert volume <= "
            f"{self.alert_volume_max_ratio:.1f}x the champion's, and no new zero-coverage "
            "segment. Knobs, not statistical guarantees.",
        ]


def challenger_split(n: int, config: ChallengerConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Challenger windows (train, val, test) on the time-ordered table.

    The TEST window is taken from ``fdo.model.time_split`` verbatim, so the
    champion and challenger are judged on identical rows by construction. The
    challenger's validation slice is the last ``val_frac`` of the pre-test
    timeline; its training window runs from ``drop_frac`` up to that slice.
    All three windows stay contiguous and time-ordered - no shuffling, same
    leakage discipline as the champion split.
    """
    if not 0.0 <= config.drop_frac < 1.0:
        raise ValueError("drop_frac must be in [0, 1)")
    if config.val_frac <= 0.0:
        raise ValueError("val_frac must be positive")
    _, _, idx_test = time_split(n)
    test_start = int(idx_test[0])
    n_val = int(n * config.val_frac)
    val_start = test_start - n_val
    train_start = int(n * config.drop_frac)
    if n_val < 1 or val_start <= 0:
        raise ValueError("val_frac leaves no challenger validation rows")
    if train_start >= val_start:
        raise ValueError(
            "challenger training window is empty: drop_frac too large for val_frac/test split"
        )
    idx = np.arange(n)
    return idx[train_start:val_start], idx[val_start:test_start], idx_test


def swap_set(
    y: np.ndarray,
    amounts: np.ndarray,
    flag_champion: np.ndarray,
    flag_challenger: np.ndarray,
) -> pd.DataFrame:
    """Swap-set analysis of two alert decisions on the same labelled window.

    Partitions the window into the four champion/challenger flag cells and
    reports, per cell: transaction count, fraud count, fraud dollar value,
    and the cell's fraud rate (NaN for an empty cell). The four cells are a
    partition, so counts and values always sum to the window totals - the
    tests assert that conservation.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    amounts = np.asarray(amounts, dtype=np.float64).ravel()
    fc = np.asarray(flag_champion, dtype=bool).ravel()
    fh = np.asarray(flag_challenger, dtype=bool).ravel()
    if not (y.shape == amounts.shape == fc.shape == fh.shape):
        raise ValueError("swap_set: all inputs must share one shape")
    masks = (fc & fh, fc & ~fh, ~fc & fh, ~fc & ~fh)
    rows = []
    for cell, mask in zip(SWAP_CELLS, masks, strict=True):
        n_cell = int(mask.sum())
        fraud = mask & (y == 1)
        n_fraud = int(fraud.sum())
        rows.append(
            {
                "cell": cell,
                "n": n_cell,
                "n_fraud": n_fraud,
                "fraud_value": float(amounts[fraud].sum()),
                "fraud_rate": (n_fraud / n_cell) if n_cell else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def promotion_gates(measured: dict, config: ChallengerConfig) -> list[dict]:
    """Evaluate the five pre-declared promotion criteria on measured values.

    ``measured`` keys: pr_auc_champion, pr_auc_challenger, ece_challenger,
    cost_champion, cost_challenger, alerts_champion, alerts_challenger,
    zero_cov_champion, zero_cov_challenger. Pure function of its inputs so
    the gate logic is testable without a training run. Returns one dict per
    gate: name, requirement, measured (display strings), passed (bool).
    """
    m = measured
    gates = [
        {
            "gate": "ranking_non_inferior",
            "requirement": f"test PR-AUC >= champion - {config.pr_auc_epsilon:.3f}",
            "measured": f"{m['pr_auc_challenger']:.3f} vs champion {m['pr_auc_champion']:.3f}",
            "passed": bool(
                m["pr_auc_challenger"] >= m["pr_auc_champion"] - config.pr_auc_epsilon
            ),
        },
        {
            "gate": "calibration_sane",
            "requirement": f"test ECE (post-Platt) <= {config.ece_max:.3f}",
            "measured": f"{m['ece_challenger']:.4f}",
            "passed": bool(m["ece_challenger"] <= config.ece_max),
        },
        {
            "gate": "economics_non_worse",
            "requirement": (
                f"test cost at own t* <= champion's x {1.0 + config.cost_margin:.2f}"
            ),
            "measured": (
                f"${m['cost_challenger']:,.0f} vs champion ${m['cost_champion']:,.0f}"
            ),
            "passed": bool(
                m["cost_challenger"] <= m["cost_champion"] * (1.0 + config.cost_margin)
            ),
        },
        {
            "gate": "alert_volume_bounded",
            "requirement": f"alerts <= {config.alert_volume_max_ratio:.1f}x champion volume",
            "measured": f"{m['alerts_challenger']:,} vs champion {m['alerts_champion']:,}",
            "passed": bool(
                m["alerts_challenger"]
                <= config.alert_volume_max_ratio * max(m["alerts_champion"], 1)
            ),
        },
        {
            "gate": "segment_coverage_safe",
            "requirement": "zero-coverage segments (fraud present, none caught) do not increase",
            "measured": (
                f"{m['zero_cov_challenger']} vs champion {m['zero_cov_champion']}"
            ),
            "passed": bool(m["zero_cov_challenger"] <= m["zero_cov_champion"]),
        },
    ]
    return gates


def gate_verdict(gates: list[dict]) -> str:
    """PROMOTE only if every gate passed; otherwise HOLD."""
    return "PROMOTE" if all(g["passed"] for g in gates) else "HOLD"


def _capacity_review(p: np.ndarray, amounts: np.ndarray, y: np.ndarray, cap: int) -> dict:
    """Top-``cap`` reviews by expected value p*amount (stable sort): the
    unconstrained queue rule, held fixed so only the scores differ."""
    ev = p * amounts
    sel = np.argsort(-ev, kind="stable")[:cap]
    return {
        "expected_recovered": float(ev[sel].sum()),
        "realized_recovered": float((y[sel] * amounts[sel]).sum()),
    }


def _zero_coverage_segments(coverage: pd.DataFrame) -> int:
    """Segments with observed fraud where the decision caught none."""
    return int(((coverage["n_fraud"] > 0) & (coverage["n_fraud_caught"] == 0)).sum())


def run_challenger(results: dict, config: ChallengerConfig | None = None) -> dict:
    """Train the challenger, judge both models once on the shared test window,
    and evaluate the promotion gates. Deterministic given the pipeline result
    (training uses zero-init full-batch gradient descent - no RNG)."""
    config = config or ChallengerConfig()
    df = results["df"]
    n = len(df)
    idx_tr_h, idx_va_h, idx_te = challenger_split(n, config)
    if not np.array_equal(idx_te, results["splits"]["test"]):
        raise RuntimeError("challenger test window diverged from the pipeline test window")

    y = df["is_fraud"].to_numpy(dtype=np.float64)
    amounts = df["amount"].to_numpy(dtype=np.float64)
    segments = df["merchant_cat"].to_numpy()
    y_te, amounts_te, seg_te = y[idx_te], amounts[idx_te], segments[idx_te]

    champion = results["models"][results["primary_name"]]
    challenger = FraudModel(
        loss=champion.loss, gamma=champion.gamma, l2=champion.l2,
        lr=champion.lr, max_iter=champion.max_iter,
    ).fit(df, idx_tr_h, idx_va_h)

    p_test_c = np.asarray(results["p_test"], dtype=np.float64)
    p_val_h = challenger.predict_calibrated(df.iloc[idx_va_h])
    p_test_h = challenger.predict_calibrated(df.iloc[idx_te])

    costs = results["thresholds"]["costs"]
    thr_h = threshold_report(
        y[idx_va_h], amounts[idx_va_h], p_val_h, y_te, amounts_te, p_test_h, costs
    )
    thr_c = results["thresholds"]

    t_star_c = float(thr_c["threshold_star"])
    t_star_h = float(thr_h["threshold_star"])
    flag_c = p_test_c >= t_star_c
    flag_h = p_test_h >= t_star_h

    cov_c = segment_fraud_coverage(y_te, amounts_te, seg_te, flag_c)
    cov_h = segment_fraud_coverage(y_te, amounts_te, seg_te, flag_h)

    cap = int(results["queue"]["config"].capacity)
    test_c = summarize_scores(y_te, p_test_c)
    test_h = summarize_scores(y_te, p_test_h)

    measured = {
        "pr_auc_champion": test_c["pr_auc"],
        "pr_auc_challenger": test_h["pr_auc"],
        "ece_challenger": test_h["ece"],
        "cost_champion": thr_c["test_at_star"]["total_cost"],
        "cost_challenger": thr_h["test_at_star"]["total_cost"],
        "alerts_champion": int(thr_c["test_at_star"]["n_flagged"]),
        "alerts_challenger": int(thr_h["test_at_star"]["n_flagged"]),
        "zero_cov_champion": _zero_coverage_segments(cov_c),
        "zero_cov_challenger": _zero_coverage_segments(cov_h),
    }
    gates = promotion_gates(measured, config)

    def _days(idx: np.ndarray) -> tuple[float, float]:
        d = df["ts_days"].to_numpy()
        return float(d[idx[0]]), float(d[idx[-1]])

    return {
        "config": config,
        "windows": {
            "champion_train": _days(results["splits"]["train"]),
            "champion_val": _days(results["splits"]["val"]),
            "challenger_train": _days(idx_tr_h),
            "challenger_val": _days(idx_va_h),
            "test": _days(idx_te),
            "n_challenger_train": int(idx_tr_h.size),
            "n_challenger_val": int(idx_va_h.size),
            "n_champion_train": int(results["splits"]["train"].size),
            "n_champion_val": int(results["splits"]["val"].size),
            "n_test": int(idx_te.size),
        },
        "challenger_model": challenger,
        "test_champion": test_c,
        "test_challenger": test_h,
        "thresholds_champion": thr_c,
        "thresholds_challenger": thr_h,
        "swap": swap_set(y_te, amounts_te, flag_c, flag_h),
        "capacity": cap,
        "capacity_champion": _capacity_review(p_test_c, amounts_te, y_te, cap),
        "capacity_challenger": _capacity_review(p_test_h, amounts_te, y_te, cap),
        "coverage_champion": cov_c,
        "coverage_challenger": cov_h,
        "measured": measured,
        "gates": gates,
        "verdict": gate_verdict(gates),
    }


def challenger_lines(analysis: dict) -> list[str]:
    """ASCII-only champion/challenger report for the CLI and the README."""
    w = analysis["windows"]
    tc, th = analysis["test_champion"], analysis["test_challenger"]
    rc = analysis["thresholds_champion"]["test_at_star"]
    rh = analysis["thresholds_challenger"]["test_at_star"]
    cc, ch = analysis["capacity_champion"], analysis["capacity_challenger"]
    lines = [
        "CHAMPION/CHALLENGER - does retraining on the drifted regime pay? (synthetic data;",
        "dollar figures rest on the labelled cost assumptions - planning arithmetic, not savings)",
        f"  CHAMPION   trains days {w['champion_train'][0]:.0f}-{w['champion_train'][1]:.0f} "
        f"(n={w['n_champion_train']:,}), validates days {w['champion_val'][0]:.0f}-"
        f"{w['champion_val'][1]:.0f} (n={w['n_champion_val']:,}).",
        f"  CHALLENGER trains days {w['challenger_train'][0]:.0f}-{w['challenger_train'][1]:.0f} "
        f"(n={w['n_challenger_train']:,}), validates days {w['challenger_val'][0]:.0f}-"
        f"{w['challenger_val'][1]:.0f} (n={w['n_challenger_val']:,}). Same family, loss and",
        "  hyperparameters - ONLY the training window moves. Judged once on the identical",
        f"  test window (days {w['test'][0]:.0f}-{w['test'][1]:.0f}, n={w['n_test']:,}).",
        "",
        f"  {'test window':<28} {'champion':>12} {'challenger':>12}",
        f"  {'-' * 28} {'-' * 12} {'-' * 12}",
        f"  {'PR-AUC':<28} {tc['pr_auc']:>12.3f} {th['pr_auc']:>12.3f}",
        f"  {'ROC-AUC':<28} {tc['roc_auc']:>12.3f} {th['roc_auc']:>12.3f}",
        f"  {'ECE (post-Platt)':<28} {tc['ece']:>12.4f} {th['ece']:>12.4f}",
        f"  {'Brier':<28} {tc['brier']:>12.4f} {th['brier']:>12.4f}",
        f"  {'own t* (swept on own val)':<28} "
        f"{analysis['thresholds_champion']['threshold_star']:>12.3f} "
        f"{analysis['thresholds_challenger']['threshold_star']:>12.3f}",
        f"  {'alerts at own t*':<28} {rc['n_flagged']:>12,} {rh['n_flagged']:>12,}",
        f"  {'fraud caught at own t*':<28} "
        f"{rc['n_fraud_caught']:>12,} {rh['n_fraud_caught']:>12,}",
        f"  {'total cost at own t*':<28} "
        f"{'$' + format(rc['total_cost'], ',.0f'):>12} "
        f"{'$' + format(rh['total_cost'], ',.0f'):>12}",
        f"  {'expected $ @' + str(analysis['capacity']) + ' reviews':<28} "
        f"{'$' + format(cc['expected_recovered'], ',.0f'):>12} "
        f"{'$' + format(ch['expected_recovered'], ',.0f'):>12}",
        f"  {'realized $ @' + str(analysis['capacity']) + ' reviews':<28} "
        f"{'$' + format(cc['realized_recovered'], ',.0f'):>12} "
        f"{'$' + format(ch['realized_recovered'], ',.0f'):>12}",
        "",
        "  Swap-set on the test window (each model at its own t*):",
    ]
    for _, row in analysis["swap"].iterrows():
        rate = f"{row['fraud_rate']:.1%}" if np.isfinite(row["fraud_rate"]) else "n/a"
        lines.append(
            f"    {row['cell']:<28} n={row['n']:>6,}  fraud={row['n_fraud']:>4,} "
            f"(${row['fraud_value']:>9,.0f})  fraud rate {rate}"
        )
    lines += ["", "  Promotion gates (pre-declared policy knobs, judged once on test):"]
    for g in analysis["gates"]:
        status = "PASS" if g["passed"] else "FAIL"
        lines.append(
            f"    [{status}] {g['gate']:<22} {g['requirement']} -- measured {g['measured']}"
        )
    lines += [
        "",
        f"  VERDICT: {analysis['verdict']}"
        + (
            " - promote the challenger under the declared gate policy."
            if analysis["verdict"] == "PROMOTE"
            else " - keep the champion; the challenger does not clear the declared gates."
        ),
        "  READ HONESTLY: the gates are policy knobs, not statistical guarantees; a HOLD is",
        "  as valid a finding as a PROMOTE - the gate exists to stop reflexive retraining.",
    ]
    return lines


def save_challenger_csv(analysis: dict, path: str) -> str:
    """Write the per-model comparison table (one row per model) to CSV."""
    rows = []
    for name, test, thr, cap_view, zero_cov in (
        (
            "champion",
            analysis["test_champion"],
            analysis["thresholds_champion"],
            analysis["capacity_champion"],
            analysis["measured"]["zero_cov_champion"],
        ),
        (
            "challenger",
            analysis["test_challenger"],
            analysis["thresholds_challenger"],
            analysis["capacity_challenger"],
            analysis["measured"]["zero_cov_challenger"],
        ),
    ):
        w = analysis["windows"]
        d0, d1 = w[f"{name}_train"]
        v0, v1 = w[f"{name}_val"]
        star = thr["test_at_star"]
        rows.append(
            {
                "model": name,
                "train_day_start": d0,
                "train_day_end": d1,
                "n_train": w[f"n_{name}_train"],
                "val_day_start": v0,
                "val_day_end": v1,
                "n_val": w[f"n_{name}_val"],
                "pr_auc": test["pr_auc"],
                "roc_auc": test["roc_auc"],
                "ece": test["ece"],
                "brier": test["brier"],
                "t_star": thr["threshold_star"],
                "alerts_at_t_star": star["n_flagged"],
                "fraud_caught_at_t_star": star["n_fraud_caught"],
                "recall_at_t_star": star["recall"],
                "total_cost_at_t_star": star["total_cost"],
                "expected_recovered_at_capacity": cap_view["expected_recovered"],
                "realized_recovered_at_capacity": cap_view["realized_recovered"],
                "zero_coverage_segments": zero_cov,
                "verdict": analysis["verdict"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path


def save_swap_csv(analysis: dict, path: str) -> str:
    """Write the four-cell swap-set table to CSV. Returns the path."""
    analysis["swap"].to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path
