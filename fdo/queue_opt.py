"""Analyst review-queue allocation as an optimization problem.

The operational reality this models: far more transactions get flagged than a
small analyst team can review. Given calibrated fraud probabilities p_i and
amounts a_i, K analysts with capacity C reviews each per shift, choose WHICH
transactions to review to maximize expected recovered value

    maximize   sum_i x_i * p_i * a_i
    subject to sum_i x_i <= K*C                     (capacity)
               sum_{i in segment s} x_i >= m_s      (coverage floor, per
                                                     merchant segment)
               x_i in {0, 1}

CAPACITY / COVERAGE ASSUMPTIONS (config knobs, not data): K analysts, C
reviews per analyst per shift, and a minimum number of reviews per merchant
segment. The coverage floor exists for a real reason - a queue that chases
only today's highest expected value goes blind in the segments it stops
watching, and fraud migrates. It is also what makes this an actual
optimization problem:

HONESTY NOTE: WITHOUT the coverage constraint the LP relaxation is a
fractional knapsack with unit weights, so its optimum IS exactly "sort by
p_i * a_i, take the top K*C". We solve the unconstrained LP and verify that
equivalence numerically every run (see ``lp_equals_topk_ev`` in the result)
rather than pretending the solver adds magic. The solver earns its keep only
once the coverage constraints are on, which is the case we report.

Solved with scipy's HiGHS backend (``milp``; binary variables). The
constraint structure (a partition of columns plus one global row) is totally
unimodular, so the LP relaxation is already integral - milp terminates at the
root node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass(frozen=True)
class QueueConfig:
    """Capacity model. All values are ASSUMPTIONS about the ops team."""

    n_analysts: int = 4
    reviews_per_analyst: int = 25
    min_reviews_per_segment: int = 5

    @property
    def capacity(self) -> int:
        return self.n_analysts * self.reviews_per_analyst

    def describe(self) -> list[str]:
        return [
            f"ASSUMPTION: {self.n_analysts} analysts x {self.reviews_per_analyst} "
            f"reviews per shift = {self.capacity} reviews of capacity.",
            f"ASSUMPTION: at least {self.min_reviews_per_segment} reviews per merchant "
            "segment (anti-blind-spot coverage floor).",
        ]


def _expected_value(selected: np.ndarray, ev: np.ndarray) -> float:
    return float(ev[selected].sum())


def plan_queue(
    p: np.ndarray,
    amounts: np.ndarray,
    segments: np.ndarray,
    config: QueueConfig,
    y: np.ndarray | None = None,
) -> dict:
    """Solve the constrained allocation and compare against naive queues.

    Returns selection indices, the three-way comparison (constrained
    optimizer vs top-K-by-probability vs random), the unconstrained-LP
    equivalence check, and per-segment coverage of each strategy.
    ``y`` (observed labels) is optional and only used to report *realized*
    recovered value next to the expected one.
    """
    p = np.asarray(p, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    segments = np.asarray(segments)
    n = p.size
    ev = p * amounts
    cap = config.capacity
    if cap >= n:
        raise ValueError("capacity >= pool size; nothing to optimize")

    seg_names, seg_idx = np.unique(segments, return_inverse=True)
    n_seg = seg_names.size
    seg_counts = np.bincount(seg_idx, minlength=n_seg)
    floors = np.minimum(config.min_reviews_per_segment, seg_counts)
    if floors.sum() > cap:
        raise ValueError("coverage floors exceed total capacity; infeasible")

    # --- constrained problem ----------------------------------------------
    # The constraint matrix (one all-ones row + a partition of the columns)
    # is totally unimodular, so every LP vertex is integral and the HiGHS
    # simplex solution of the relaxation already solves the integer problem.
    # We exploit that for speed, verify integrality, and fall back to a true
    # MILP solve if the check ever fails.
    A = np.zeros((1 + n_seg, n))
    A[0, :] = 1.0
    for s in range(n_seg):
        A[1 + s, seg_idx == s] = 1.0
    lb = np.r_[0.0, floors.astype(np.float64)]
    ub = np.r_[float(cap), seg_counts.astype(np.float64)]
    res = linprog(
        c=-ev,
        A_ub=np.vstack([A[0:1], -A[1:]]),
        b_ub=np.r_[float(cap), -floors.astype(np.float64)],
        bounds=(0.0, 1.0),
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"linprog (constrained) failed: {res.message}")
    x = res.x
    if np.max(np.minimum(np.abs(x), np.abs(1.0 - x))) > 1e-6:  # fractional vertex?
        res_m = milp(
            c=-ev,
            constraints=LinearConstraint(A, lb, ub),
            integrality=np.ones(n),
            bounds=Bounds(0.0, 1.0),
        )
        if res_m.status != 0:
            raise RuntimeError(f"milp failed: {res_m.message}")
        x = res_m.x
    sel_milp = np.flatnonzero(x > 0.5)

    # --- unconstrained LP vs top-K by expected value (honesty check) ------
    res_lp = linprog(
        c=-ev,
        A_ub=np.ones((1, n)),
        b_ub=np.array([float(cap)]),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if res_lp.status != 0:
        raise RuntimeError(f"linprog failed: {res_lp.message}")
    sel_topk_ev = np.argsort(-ev, kind="stable")[:cap]
    lp_opt = float(-res_lp.fun)
    topk_ev_value = _expected_value(sel_topk_ev, ev)
    lp_equals_topk = bool(abs(lp_opt - topk_ev_value) <= 1e-6 * max(1.0, topk_ev_value))

    # --- naive comparators -------------------------------------------------
    sel_topk_prob = np.argsort(-p, kind="stable")[:cap]
    # Random baseline: expectation under uniform selection of `cap` rows,
    # reported analytically (no draw-to-draw luck): cap * mean(ev).
    random_expected = float(cap * ev.mean())

    def realized(selected: np.ndarray) -> float | None:
        if y is None:
            return None
        yy = np.asarray(y, dtype=np.float64)
        return float((yy[selected] * amounts[selected]).sum())

    random_realized = None
    if y is not None:
        yy = np.asarray(y, dtype=np.float64)
        random_realized = float(cap * (yy * amounts).mean())

    def coverage(selected: np.ndarray) -> np.ndarray:
        return np.bincount(seg_idx[selected], minlength=n_seg)

    comparison = pd.DataFrame(
        [
            {
                "strategy": "constrained_optimizer (MILP)",
                "expected_recovered": _expected_value(sel_milp, ev),
                "realized_recovered": realized(sel_milp),
                "n_reviews": int(sel_milp.size),
                "segments_covered": int((coverage(sel_milp) > 0).sum()),
            },
            {
                "strategy": "top_k_by_probability",
                "expected_recovered": _expected_value(sel_topk_prob, ev),
                "realized_recovered": realized(sel_topk_prob),
                "n_reviews": int(sel_topk_prob.size),
                "segments_covered": int((coverage(sel_topk_prob) > 0).sum()),
            },
            {
                "strategy": "top_k_by_expected_value (== unconstrained LP)",
                "expected_recovered": topk_ev_value,
                "realized_recovered": realized(sel_topk_ev),
                "n_reviews": int(sel_topk_ev.size),
                "segments_covered": int((coverage(sel_topk_ev) > 0).sum()),
            },
            {
                "strategy": "random_selection (analytic expectation)",
                "expected_recovered": random_expected,
                "realized_recovered": random_realized,
                "n_reviews": cap,
                "segments_covered": int((seg_counts > 0).sum()),
            },
        ]
    )

    coverage_table = pd.DataFrame(
        {
            "segment": seg_names,
            "pool_size": seg_counts,
            "floor": floors,
            "milp_reviews": coverage(sel_milp),
            "topk_prob_reviews": coverage(sel_topk_prob),
            "topk_ev_reviews": coverage(sel_topk_ev),
        }
    )

    return {
        "config": config,
        "selected_milp": sel_milp,
        "selected_topk_prob": sel_topk_prob,
        "selected_topk_ev": sel_topk_ev,
        "comparison": comparison,
        "coverage": coverage_table,
        "lp_equals_topk_ev": lp_equals_topk,
        "lp_unconstrained_optimum": lp_opt,
        "milp_expected": _expected_value(sel_milp, ev),
        "topk_prob_expected": _expected_value(sel_topk_prob, ev),
        "topk_ev_expected": topk_ev_value,
        "random_expected": random_expected,
    }
