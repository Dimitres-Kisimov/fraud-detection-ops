"""Cost-sensitive alert threshold.

COST ASSUMPTIONS (labelled as such everywhere they surface):

- A missed fraud costs the full transaction amount (chargeback plus fees
  approximated as 1.0x amount). ASSUMPTION - the multiplier is a config knob,
  not an estimate from data.
- A review costs a flat ``review_cost`` (default $8) of analyst time whether
  or not the transaction turns out to be fraud. ASSUMPTION.
- A reviewed fraud is always caught (no analyst misses). Simplifying
  ASSUMPTION that flatters every strategy equally.

With those assumptions the empirical cost of operating at threshold t on a
labelled window is

    cost(t) = review_cost * #{p >= t}  +  sum(amount_i : fraud i with p_i < t)

We sweep t on the VALIDATION window, pick the minimizer, and then report the
chosen threshold on the TEST window against the naive t = 0.5 default. The
comparison is honest out-of-sample: the threshold is chosen without touching
test labels.

Decision-theory footnote: with per-transaction costs the optimal rule is
per-item (flag when p_i * amount_i > review_cost), not a single probability
threshold. The single threshold is kept because it is how alert rules ship in
practice; the per-item expected-value view is exactly what fdo/queue_opt.py
does with the analyst capacity constraint on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostAssumptions:
    """All values are ASSUMPTIONS, not measurements."""

    review_cost: float = 8.0
    missed_fraud_multiplier: float = 1.0

    def describe(self) -> list[str]:
        return [
            f"ASSUMPTION: each analyst review costs ${self.review_cost:.2f} of time, "
            "fraud or not.",
            "ASSUMPTION: a missed fraud costs "
            f"{self.missed_fraud_multiplier:.1f}x the transaction amount.",
            "ASSUMPTION: a reviewed fraud is always caught.",
        ]


def empirical_cost(
    y: np.ndarray,
    amounts: np.ndarray,
    p: np.ndarray,
    threshold: float,
    costs: CostAssumptions,
) -> dict[str, float]:
    """Empirical operating cost of flag-if-p>=t on one labelled window."""
    y = np.asarray(y, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    flagged = p >= threshold
    n_flagged = int(flagged.sum())
    review_total = costs.review_cost * n_flagged
    missed_mask = (y == 1) & ~flagged
    missed_loss = float(amounts[missed_mask].sum()) * costs.missed_fraud_multiplier
    caught_mask = (y == 1) & flagged
    return {
        "threshold": float(threshold),
        "n_flagged": n_flagged,
        "review_cost_total": float(review_total),
        "missed_fraud_loss": missed_loss,
        "total_cost": float(review_total + missed_loss),
        "n_fraud_caught": int(caught_mask.sum()),
        "n_fraud_missed": int(missed_mask.sum()),
        "fraud_value_caught": float(amounts[caught_mask].sum()),
        "recall": float(caught_mask.sum() / max(y.sum(), 1.0)),
    }


def sweep_thresholds(
    y: np.ndarray,
    amounts: np.ndarray,
    p: np.ndarray,
    costs: CostAssumptions,
    n_grid: int = 400,
) -> pd.DataFrame:
    """Empirical cost across a probability-threshold grid (quantiles of p,
    plus 0.5 and both endpoints so the naive default is always on the curve).
    """
    qs = np.quantile(np.asarray(p, dtype=np.float64), np.linspace(0.0, 1.0, n_grid))
    grid = np.unique(np.r_[qs, 0.5, 0.0, 1.0000001])
    rows = [empirical_cost(y, amounts, p, t, costs) for t in grid]
    return pd.DataFrame(rows)


def choose_threshold(sweep: pd.DataFrame) -> float:
    """Threshold with minimal total empirical cost (ties -> lowest threshold,
    i.e. the more fraud-averse choice)."""
    i = int(sweep["total_cost"].idxmin())
    return float(sweep.loc[i, "threshold"])


def threshold_report(
    y_val: np.ndarray,
    amounts_val: np.ndarray,
    p_val: np.ndarray,
    y_test: np.ndarray,
    amounts_test: np.ndarray,
    p_test: np.ndarray,
    costs: CostAssumptions,
) -> dict:
    """Choose t* on validation, then compare t* vs naive 0.5 on test."""
    sweep = sweep_thresholds(y_val, amounts_val, p_val, costs)
    t_star = choose_threshold(sweep)
    val_at_star = empirical_cost(y_val, amounts_val, p_val, t_star, costs)
    test_at_star = empirical_cost(y_test, amounts_test, p_test, t_star, costs)
    test_at_naive = empirical_cost(y_test, amounts_test, p_test, 0.5, costs)
    test_review_none = empirical_cost(y_test, amounts_test, p_test, 1.0000001, costs)
    test_review_all = empirical_cost(y_test, amounts_test, p_test, 0.0, costs)
    reduction = 1.0 - test_at_star["total_cost"] / test_at_naive["total_cost"]
    return {
        "costs": costs,
        "sweep_val": sweep,
        "threshold_star": t_star,
        "val_at_star": val_at_star,
        "test_at_star": test_at_star,
        "test_at_naive_05": test_at_naive,
        "test_review_none": test_review_none,
        "test_review_all": test_review_all,
        "test_cost_reduction_vs_05": float(reduction),
    }
