"""Tests for the cost-curve decision layer (fdo/cost_curve.py) and the
per-segment queue-fairness read-out (fdo/fairness.py). All read from the same
shared deterministic pipeline run (seed 7)."""

from __future__ import annotations

import numpy as np

from fdo.cost_curve import (
    cost_curve_frame,
    recommendation,
    save_cost_curve_csv,
    save_cost_curve_svg,
)
from fdo.fairness import (
    fairness_lines,
    queue_fairness,
    save_fairness_csv,
    segment_fraud_coverage,
)


def test_cost_curve_frame_shape_and_bounds(results):
    frame = cost_curve_frame(results)
    assert (frame["precision"] >= 0).all() and (frame["precision"] <= 1).all()
    assert (frame["recall"] >= 0).all() and (frame["recall"] <= 1).all()
    # review-queue volume is non-increasing as the threshold rises (flag-if-p>=t)
    vol = frame.sort_values("threshold")["review_queue_volume"].to_numpy()
    assert (np.diff(vol) <= 0).all()
    # total_cost = review_cost + missed_fraud_loss, exactly
    assert np.allclose(frame["total_cost"], frame["review_cost"] + frame["missed_fraud_loss"])


def test_cost_curve_minimum_matches_chosen_threshold(results):
    frame = cost_curve_frame(results)
    t_star = float(results["thresholds"]["threshold_star"])
    min_row = frame.loc[frame["total_cost"].idxmin()]
    assert np.isclose(min_row["threshold"], t_star)
    assert bool(frame.loc[frame["is_chosen_t_star"], "is_chosen_t_star"].any())
    assert int(frame["is_naive_default"].sum()) == 1


def test_recommendation_is_honest_and_names_t_star(results):
    lines = recommendation(results)
    text = "\n".join(lines)
    t_star = float(results["thresholds"]["threshold_star"])
    assert f"{t_star:.3f}" in text
    assert "ASSUMPTION" in text  # cost rates flagged as assumptions
    assert "VALIDATION" in text and "TEST" in text  # method disclosed
    assert "synthetic" in text


def test_cost_curve_artifacts_written_and_deterministic(results, tmp_path):
    svg = save_cost_curve_svg(results, str(tmp_path / "c.svg"))
    csv = save_cost_curve_csv(results, str(tmp_path / "c.csv"))
    svg_text = open(svg, encoding="utf-8").read()
    assert svg_text.startswith("<svg") or svg_text.lstrip().startswith("<svg")
    assert "t* =" in svg_text
    assert len(open(csv, encoding="utf-8").read()) > 0
    # byte-deterministic (no wall-clock / RNG in the committed output)
    again = save_cost_curve_svg(results, str(tmp_path / "c2.svg"))
    assert open(again, encoding="utf-8").read() == svg_text


def test_segment_coverage_conserves_fraud_counts(results):
    df = results["df"]
    te = results["splits"]["test"]
    sub = df.iloc[te].reset_index(drop=True)
    y = sub["is_fraud"].to_numpy(dtype=np.float64)
    amounts = sub["amount"].to_numpy(dtype=np.float64)
    seg = sub["merchant_cat"].to_numpy()
    p = np.asarray(results["p_test"])
    t_star = float(results["thresholds"]["threshold_star"])
    selected = p >= t_star

    cov = segment_fraud_coverage(y, amounts, seg, selected)
    # per-segment fraud counts and caught counts sum to the window totals
    assert int(cov["n_fraud"].sum()) == int((y == 1).sum())
    total_caught = int(((y == 1) & selected).sum())
    assert int(cov["n_fraud_caught"].sum()) == total_caught
    # catch_rate is a share where defined
    defined = cov[cov["n_fraud"] > 0]
    assert (defined["catch_rate"] >= 0).all() and (defined["catch_rate"] <= 1).all()


def test_queue_fairness_matches_recall_and_shows_gap(results):
    f = queue_fairness(results)
    # the threshold read-out's overall recall must equal the threshold report's
    assert np.isclose(
        f["threshold_summary"]["overall_recall"],
        results["thresholds"]["test_at_star"]["recall"],
    )
    for key in ("threshold_summary", "queue_summary"):
        s = f[key]
        assert 0.0 <= s["worst_catch_rate"] <= s["best_catch_rate"] <= 1.0
        assert s["coverage_gap"] >= 0.0
        # this synthetic data has real cross-segment disparity to surface
        assert s["coverage_gap"] > 0.0


def test_fairness_lines_and_csv(results, tmp_path):
    lines = fairness_lines(results)
    text = "\n".join(lines)
    assert "FAIRNESS" in text.upper()
    assert "synthetic" in text
    path = save_fairness_csv(results, str(tmp_path / "fair.csv"))
    body = open(path, encoding="utf-8").read()
    assert "cost_threshold" in body and "review_queue" in body
    assert "catch_rate" in body
