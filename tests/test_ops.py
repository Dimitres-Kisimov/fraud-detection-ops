"""Integration tests on the shared pipeline run: baselines, calibration,
cost threshold, queue optimization, exports."""

from __future__ import annotations

import os

import numpy as np


def test_model_beats_both_baselines_on_test(results):
    m = results["metrics"]
    model_pr = m["test_primary"]["pr_auc"]
    random_pr = m["baseline_random"]["pr_auc"]
    heuristic_pr = m["baseline_p99"]["pr_auc"]
    prevalence = m["test_primary"]["prevalence"]
    assert model_pr > 2 * heuristic_pr
    assert model_pr > 5 * max(random_pr, prevalence)
    # and the model cannot beat the oracle built into the generator
    assert model_pr <= m["oracle_test_pr_auc"] + 0.02


def test_calibration_improves_ece_on_validation(results):
    cal = results["metrics"]["calibration"]
    assert cal["val_ece_cal"] < cal["val_ece_raw"] / 5
    assert cal["val_brier_cal"] < cal["val_brier_raw"]


def test_cost_threshold_beats_naive_05(results):
    t = results["thresholds"]
    # chosen on validation, judged on test - must still beat the 0.5 default
    assert t["test_at_star"]["total_cost"] < t["test_at_naive_05"]["total_cost"]
    # sanity: the sweep minimum on validation is exactly the chosen row
    sweep = t["sweep_val"]
    assert t["val_at_star"]["total_cost"] == sweep["total_cost"].min()


def test_queue_respects_capacity_and_coverage(results):
    q = results["queue"]
    cfg = q["config"]
    assert len(q["selected_milp"]) <= cfg.capacity
    cov = q["coverage"]
    assert (cov["milp_reviews"] >= cov["floor"]).all()
    # selection is a set of distinct transaction indices (integral solution)
    assert len(np.unique(q["selected_milp"])) == len(q["selected_milp"])


def test_queue_comparison_relations(results):
    q = results["queue"]
    # The constrained optimizer must crush random selection...
    assert q["milp_expected"] > 10 * q["random_expected"]
    # ...and on this run it also beats top-K-by-probability, because ranking
    # by p alone ignores amounts. (Measured relation for this seed; a
    # sufficiently harsh coverage floor could in principle reverse it, which
    # is why the assertion documents the >= that actually holds.)
    assert q["milp_expected"] >= q["topk_prob_expected"]
    # Honesty invariants: the coverage-constrained optimum can never exceed
    # the unconstrained one, and the unconstrained LP must equal top-K by
    # expected value exactly (fractional knapsack with unit weights).
    assert q["milp_expected"] <= q["topk_ev_expected"] + 1e-6
    assert q["lp_equals_topk_ev"] is True


def test_exports_nonempty(results, tmp_path):
    from fdo.exports import build_deliverables

    sizes = build_deliverables(results, str(tmp_path))
    assert len(sizes) == 2
    for path, size in sizes.items():
        assert os.path.exists(path)
        assert size > 10_240, f"{path} only {size} bytes"
