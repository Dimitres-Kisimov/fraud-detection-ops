"""Tests for the reason-code / adverse-action layer (fdo/reasons.py).

The load-bearing claims are mathematical, so they are tested as mathematics:
the decomposition is exact (contributions sum to the score minus the reference
score), it IS the Shapley value of the linear score (checked against brute-force
enumeration over all coalitions), grouping is exactly additive, and the
logit-space alert cut is the same set of alerts as the probability threshold the
rest of the repo operates on. The rest of the file guards the artifacts:
determinism, queue agreement, and the honesty framing in the CLI read-out.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from fdo.data import make_features
from fdo.model import sigmoid
from fdo.reasons import (
    REASON_GROUPS,
    REASON_PHRASING,
    ReasonConfig,
    baseline_logit,
    contributions,
    group_contributions,
    principal_reasons,
    reason_lines,
    reasons_to_cross,
    reference_profile,
    run_reasons,
    save_reason_codes_csv,
    save_reason_codes_svg,
    save_reason_summary_csv,
    shapley_bruteforce,
    spearman,
    threshold_logit,
)


@pytest.fixture(scope="module")
def reasons(results) -> dict:
    return run_reasons(results)


# --------------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------------


def test_reference_profile_is_the_training_mean(results):
    """With a train-fitted standardizer the train-window reference IS the zero
    vector, so the baseline score is exactly the intercept. The module states
    the reference explicitly rather than relying on that accident."""
    model = results["models"][results["primary_name"]]
    train = results["df"].iloc[results["splits"]["train"]]
    ref = reference_profile(model, train)
    assert ref.shape == (len(model.feature_names),)
    assert np.allclose(ref, 0.0, atol=1e-9)
    assert baseline_logit(model, ref) == pytest.approx(float(model.theta[0]), abs=1e-12)


def test_contributions_sum_to_score_minus_baseline(results):
    """Local accuracy: z(x) = z(reference) + sum_j contribution_j(x), exactly."""
    model = results["models"][results["primary_name"]]
    df = results["df"]
    train = df.iloc[results["splits"]["train"]]
    test = df.iloc[results["splits"]["test"]]
    ref = reference_profile(model, train)
    contrib = contributions(model, test, ref)
    z = model.decision_scores(test)
    assert contrib.shape == (len(test), len(model.feature_names))
    assert np.allclose(contrib.sum(axis=1) + baseline_logit(model, ref), z, atol=1e-9)


def test_contributions_are_the_shapley_values_of_the_score():
    """The closed form theta_j * (x_j - r_j) is not an approximation: brute-force
    Shapley over all 2^m coalitions of a linear value function returns it to
    machine precision."""
    rng = np.random.default_rng(11)
    m = 7
    theta = rng.normal(0.0, 1.5, size=m)
    x = rng.normal(0.0, 1.0, size=m)
    ref = rng.normal(0.0, 1.0, size=m)
    phi = shapley_bruteforce(theta, x, ref)
    assert np.allclose(phi, theta * (x - ref), atol=1e-12)
    # efficiency: the values split the whole gap between x and the reference
    assert phi.sum() == pytest.approx(float(theta @ x - theta @ ref), abs=1e-12)


def test_shapley_bruteforce_on_the_real_model_slice(results):
    """Same check, but with the trained model's own coefficients on a real
    alert (a subset of columns, because enumeration is exponential)."""
    model = results["models"][results["primary_name"]]
    df = results["df"]
    train = df.iloc[results["splits"]["train"]]
    test = df.iloc[results["splits"]["test"]]
    ref = reference_profile(model, train)
    X, _ = make_features(test)
    xs = model.scaler.transform(X)[0]
    cols = np.arange(8)
    phi = shapley_bruteforce(model.theta[1:][cols], xs[cols], ref[cols])
    expected = contributions(model, test.iloc[:1], ref)[0, cols]
    assert np.allclose(phi, expected, atol=1e-12)


def test_shapley_bruteforce_guards():
    with pytest.raises(ValueError):
        shapley_bruteforce(np.ones(3), np.ones(4), np.ones(3))
    with pytest.raises(ValueError):
        shapley_bruteforce(np.ones(13), np.ones(13), np.ones(13))


def test_contributions_reject_a_mismatched_reference(results):
    model = results["models"][results["primary_name"]]
    test = results["df"].iloc[results["splits"]["test"]]
    with pytest.raises(ValueError):
        contributions(model, test, np.zeros(3))


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------


def test_reason_groups_partition_the_design_matrix(results):
    """Every design-matrix column belongs to exactly one reason group, and every
    group has a plain-language phrasing."""
    _, names = make_features(results["df"].head(10))
    mapped = [c for cols in REASON_GROUPS.values() for c in cols]
    assert sorted(mapped) == sorted(names)
    assert len(mapped) == len(set(mapped))
    assert set(REASON_PHRASING) == set(REASON_GROUPS)


def test_group_contributions_are_exactly_additive(results, reasons):
    model = results["models"][results["primary_name"]]
    df = results["df"]
    ref = reference_profile(model, df.iloc[results["splits"]["train"]])
    test = df.iloc[results["splits"]["test"]]
    contrib = contributions(model, test, ref)
    grouped, names = group_contributions(contrib, model.feature_names)
    assert names == reasons["group_names"]
    assert np.allclose(grouped.sum(axis=1), contrib.sum(axis=1), atol=1e-12)
    index = {c: i for i, c in enumerate(model.feature_names)}
    for gi, g in enumerate(names):
        cols = [index[c] for c in REASON_GROUPS[g]]
        assert np.allclose(grouped[:, gi], contrib[:, cols].sum(axis=1), atol=1e-12)


def test_group_contributions_reject_an_unmapped_column():
    contrib = np.zeros((3, 2))
    with pytest.raises(ValueError, match="not mapped to a reason group"):
        group_contributions(contrib, ["log_amount", "brand_new_feature"])
    with pytest.raises(ValueError, match="disagree on width"):
        group_contributions(contrib, ["log_amount"])


# --------------------------------------------------------------------------
# the alert cut, in logit space
# --------------------------------------------------------------------------


def test_threshold_logit_is_the_same_alert_set(results):
    """Flagging on z >= z_cut selects exactly the transactions the calibrated
    probability threshold flags - the explanation layer never invents its own
    operating point."""
    model = results["models"][results["primary_name"]]
    t_star = float(results["thresholds"]["threshold_star"])
    test = results["df"].iloc[results["splits"]["test"]]
    z_cut = threshold_logit(model, t_star)
    assert sigmoid(model.platt_a * z_cut + model.platt_b) == pytest.approx(t_star, rel=1e-12)
    z = model.decision_scores(test)
    assert np.array_equal(z >= z_cut, results["p_test"] >= t_star)


def test_threshold_logit_guards(results):
    model = results["models"][results["primary_name"]]
    with pytest.raises(ValueError):
        threshold_logit(model, 0.0)
    with pytest.raises(ValueError):
        threshold_logit(model, 1.0)

    class Flipped:
        platt_a, platt_b = -1.0, 0.0

    with pytest.raises(ValueError, match="not monotone"):
        threshold_logit(Flipped(), 0.5)


# --------------------------------------------------------------------------
# the reason list itself
# --------------------------------------------------------------------------


def test_principal_reasons_are_positive_ranked_and_capped():
    grouped = np.array([[0.9, -0.4, 0.3, 0.5], [-0.2, -0.1, -0.5, -0.3]])
    names = ["a", "b", "c", "d"]
    out = principal_reasons(grouped, names, max_reasons=3)
    assert [n for n, _ in out[0]] == ["a", "d", "c"]
    assert all(v > 0 for _, v in out[0])
    assert out[1] == []  # nothing pushed the score up: list nothing, never pad


def test_reasons_to_cross_is_minimal(reasons, results):
    """Removing the reported k reasons drops the alert under the cut; removing
    any k-1 of them does not. Checked exhaustively against every subset."""
    model = results["models"][results["primary_name"]]
    df = results["df"]
    ref = reference_profile(model, df.iloc[results["splits"]["train"]])
    test = df.iloc[results["splits"]["test"]]
    alerts = test.loc[results["p_test"] >= reasons["t_star"]]
    grouped, _ = group_contributions(contributions(model, alerts, ref), model.feature_names)
    z = model.decision_scores(alerts)
    z_cut = reasons["z_cut"]
    k = reasons_to_cross(grouped, z, z_cut)
    assert (k >= 1).all()  # every alert is explainable from the reference profile
    for i in range(0, len(alerts), 97):  # deterministic stride, exhaustive per row
        row, ki = grouped[i], int(k[i])
        pos = np.flatnonzero(row > 0)
        assert z[i] - row[pos[np.argsort(-row[pos])][:ki]].sum() < z_cut
        for smaller in combinations(pos, ki - 1):
            assert z[i] - row[list(smaller)].sum() >= z_cut


def test_reasons_to_cross_shape_guard():
    with pytest.raises(ValueError):
        reasons_to_cross(np.zeros((3, 2)), np.zeros(2), 0.0)


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = a * 2.0 + rng.normal(scale=0.3, size=200)
    assert spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic), abs=1e-12)
    assert spearman(a, np.exp(a)) == pytest.approx(1.0, abs=1e-12)
    assert spearman(a, -a) == pytest.approx(-1.0, abs=1e-12)
    ties = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    assert spearman(ties, np.array([2.0, 2.0, 5.0, 9.0, 9.0])) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# the analysis and its artifacts
# --------------------------------------------------------------------------


def test_summary_accounts_for_every_alert(reasons):
    s = reasons["summary"]
    assert list(s["reason_group"]) == sorted(s["reason_group"], key=lambda g: -float(
        s.loc[s["reason_group"] == g, "share_principal_reason"].iloc[0]))
    assert int(s["alerts_headed"].sum()) == reasons["n_alerts"]
    assert s["share_principal_reason"].sum() == pytest.approx(1.0, abs=1e-9)
    assert (s["share_listed_in_top_reasons"] >= s["share_principal_reason"] - 1e-12).all()
    assert (s["confirmed_fraud_rate_when_principal"].dropna().between(0.0, 1.0)).all()


def test_queue_frame_is_the_optimizer_queue(reasons, results):
    q = reasons["queue_frame"]
    capacity = results["queue"]["config"].capacity
    assert len(q) == capacity
    assert list(q["queue_rank"]) == list(range(1, capacity + 1))
    ev = q["p_fraud_calibrated"].to_numpy() * q["amount"].to_numpy()
    assert (np.diff(ev) <= 1e-9).all()  # worklist reads top-down by expected value
    test = results["df"].iloc[results["splits"]["test"]]
    selected = test.iloc[results["queue"]["selected_milp"]]
    assert sorted(q["amount"].round(2)) == sorted(selected["amount"].round(2))
    # every listed reason is a real group, listed reasons are ranked and positive
    listed = [c for c in q.columns if c.startswith("reason_")]
    for col, val in zip(listed, [c for c in q.columns if c.startswith("contribution_")],
                        strict=True):
        named = q[col] != ""
        assert set(q.loc[named, col]) <= set(REASON_GROUPS)
        assert (q.loc[named, val] > 0).all()
    assert (q["contribution_1"] >= q["contribution_2"].fillna(0.0)).all()
    assert (q["is_fraud_observed"].isin([0, 1])).all()


def test_queue_reason_codes_reconstruct_the_score(reasons, results):
    """The published row is auditable: recomputing the decomposition from the
    model reproduces every listed contribution, and the FULL set of groups
    (listed or not) sums back to the score. A truncated notice is a subset of
    the arithmetic, never a restatement of it - the listed reasons alone
    overshoot the score by exactly the protective groups they omit."""
    model = results["models"][results["primary_name"]]
    df = results["df"]
    ref = reference_profile(model, df.iloc[results["splits"]["train"]])
    test = df.iloc[results["splits"]["test"]]
    q = reasons["queue_frame"]
    order = np.argsort(-(results["p_test"] * test["amount"].to_numpy()), kind="stable")
    sel = [i for i in order if i in set(results["queue"]["selected_milp"])]
    rows = test.iloc[sel]
    grouped, names = group_contributions(contributions(model, rows, ref), model.feature_names)
    z = model.decision_scores(rows)
    assert np.allclose(q["score_logit"].to_numpy(), z, atol=1e-9)
    assert np.allclose(grouped.sum(axis=1) + q["baseline_logit"].to_numpy(), z, atol=1e-9)
    expected = principal_reasons(grouped, names, reasons["config"].max_reasons)
    for i in range(len(q)):
        for slot, (name, value) in enumerate(expected[i], start=1):
            assert q[f"reason_{slot}"].iloc[i] == name
            assert q[f"contribution_{slot}"].iloc[i] == pytest.approx(value, abs=1e-9)
    listed = q[[c for c in q.columns if c.startswith("contribution_")]].fillna(0.0).sum(axis=1)
    positives = np.where(grouped > 0.0, grouped, 0.0).sum(axis=1)
    assert (listed <= positives + 1e-9).all()
    assert (q["reasons_to_cross_threshold"] >= 1).all()
    assert (q["reasons_to_cross_threshold"] <= len(REASON_GROUPS)).all()


def test_queue_view_compares_the_worklist_with_the_alert_population(reasons, results):
    """The queue is picked by p x amount, so the reason mix an analyst opens is
    not the alert population's - the read-out states both numbers."""
    qv = reasons["queue_view"]
    q = reasons["queue_frame"]
    s = reasons["summary"]
    assert qv["n_reviews"] == results["queue"]["config"].capacity == len(q)
    counts = q["reason_1"].value_counts()
    assert qv["top_reason"] == counts.index[0]
    assert qv["share_in_queue"] == pytest.approx(counts.iloc[0] / len(q), abs=1e-12)
    assert qv["share_in_alerts"] == pytest.approx(
        float(s.loc[s["reason_group"] == qv["top_reason"], "share_principal_reason"].iloc[0]),
        abs=1e-12,
    )
    assert qv["queue_fraud_rate"] == pytest.approx(q["is_fraud_observed"].mean(), abs=1e-12)
    assert 0.0 <= qv["queue_fraud_rate"] <= 1.0


def test_stability_read_out_is_honest(reasons):
    st = reasons["stability"]
    assert st["other_model"] != reasons["primary_model"]
    assert 0.0 <= st["top_reason_agreement"] <= 1.0
    assert st["score_spearman"] > 0.9  # the two models rank alike ...
    # ... and the read-out still reports where the explanations disagree
    assert st["n_alerts"] == reasons["n_alerts"]


def test_analysis_is_deterministic(results):
    a, b = run_reasons(results), run_reasons(results)
    pd.testing.assert_frame_equal(a["summary"], b["summary"])
    pd.testing.assert_frame_equal(a["queue_frame"], b["queue_frame"])
    assert a["k_cross_median"] == b["k_cross_median"]
    assert a["stability"] == b["stability"]


def test_config_is_respected(results):
    analysis = run_reasons(results, ReasonConfig(max_reasons=2))
    q = analysis["queue_frame"]
    assert "reason_2" in q.columns and "reason_3" not in q.columns
    assert analysis["mean_reasons_listed"] <= 2.0


def test_artifacts_are_written_and_byte_stable(reasons, tmp_path):
    summary = save_reason_summary_csv(reasons, str(tmp_path / "summary.csv"))
    codes = save_reason_codes_csv(reasons, str(tmp_path / "codes.csv"))
    svg = save_reason_codes_svg(reasons, str(tmp_path / "reasons.svg"))
    first = [open(p, "rb").read() for p in (summary, codes, svg)]
    save_reason_summary_csv(reasons, summary)
    save_reason_codes_csv(reasons, codes)
    save_reason_codes_svg(reasons, svg)
    assert [open(p, "rb").read() for p in (summary, codes, svg)] == first

    frame = pd.read_csv(summary)
    assert list(frame.columns)[:2] == ["reason_group", "share_principal_reason"]
    assert len(frame) == len(REASON_GROUPS)
    text = open(svg, encoding="utf-8").read()
    assert text.startswith("<svg") and text.rstrip().endswith("</svg>")
    # fraud-ops palette, and the honesty framing travels with the picture
    assert "#a06a00" in text and "#9c1c1c" in text and "#3fae63" in text
    assert "not probabilities, not dollars, not causal" in text
    assert "Synthetic data" in text


def test_cli_read_out_is_ascii_and_states_the_limits(reasons):
    lines = reason_lines(reasons)
    assert all(line.isascii() for line in lines)
    blob = "\n".join(lines).lower()
    for phrase in (
        "logits",
        "not causal",
        "not legal advice",
        "ranking agreement is not explanation agreement",
        "explains the model",
        "reference profile",
    ):
        assert phrase in blob or phrase in blob.replace("-", " ")
