"""Feedback-loop simulation tests: split discipline (shared test window, no
leakage), hand-checked review selection and labelling policies, exact collapse
cases (chargeback_rate 1 -> full labels, 0 -> assume-legit, capacity 0 freezes
the reviewed-only arm), the selective-labels illusion as an arithmetic
identity, bias directions, determinism, and the CLI."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from fdo.data import generate_transactions
from fdo.feedback import (
    ARMS,
    TRAJECTORY_COLUMNS,
    FeedbackConfig,
    apply_label_policy,
    feedback_lines,
    feedback_split,
    run_feedback,
    save_feedback_csv,
    select_reviews,
)
from fdo.model import FraudModel, time_split
from fdo.threshold import CostAssumptions

SMALL_N = 6_000


def _small_results(n: int = SMALL_N, seed: int = 11) -> dict:
    """Minimal pipeline-result stub: run_feedback only reads the frame, the
    primary model's hyperparameters, the cost assumptions, the test split, and
    the seed - so the small tests skip the full pipeline entirely."""
    df = generate_transactions(n=n, seed=seed)
    return {
        "seed": seed,
        "df": df,
        "splits": {"test": time_split(len(df))[2]},
        "models": {"stub": FraudModel(loss="weighted_bce")},
        "primary_name": "stub",
        "thresholds": {"costs": CostAssumptions()},
    }


SMALL_CFG = FeedbackConfig(reviews_per_round=40)


@pytest.fixture(scope="module")
def small_results() -> dict:
    return _small_results()


@pytest.fixture(scope="module")
def small_sim(small_results) -> dict:
    return run_feedback(small_results, SMALL_CFG)


@pytest.fixture(scope="module")
def analysis(results) -> dict:
    return run_feedback(results)


def _arm(traj: pd.DataFrame, arm: str) -> pd.DataFrame:
    return traj[traj["arm"] == arm].drop(columns="arm").reset_index(drop=True)


def test_split_preserves_test_window_and_time_order():
    n = 60_000
    cfg = FeedbackConfig()
    idx_init, rounds, idx_te = feedback_split(n, cfg)
    _, _, te_champ = time_split(n)
    # judged on the IDENTICAL test rows the rest of the repo reports on
    assert np.array_equal(idx_te, te_champ)
    assert idx_init[0] == 0 and idx_init[-1] == int(n * cfg.initial_frac) - 1
    # rounds partition [initial_end, test_start) contiguously, in time order
    joined = np.concatenate(rounds)
    assert np.array_equal(joined, np.arange(idx_init[-1] + 1, idx_te[0]))
    assert len(rounds) == cfg.n_rounds and all(r.size > 0 for r in rounds)
    assert idx_init[-1] < rounds[0][0] and rounds[-1][-1] < idx_te[0]


def test_split_rejects_degenerate_configs():
    for bad in (
        FeedbackConfig(initial_frac=0.9),       # runs into the test window
        FeedbackConfig(initial_frac=0.0),
        FeedbackConfig(n_rounds=0),
        FeedbackConfig(reviews_per_round=-1),
        FeedbackConfig(chargeback_rate=1.5),
    ):
        with pytest.raises(ValueError):
            feedback_split(60_000, bad)


def test_select_reviews_hand_checked():
    p = np.array([0.9, 0.5, 0.2, 0.8, 0.05])
    amounts = np.array([10.0, 100.0, 1000.0, 50.0, 10000.0])
    # threshold 0.3: alerts are rows 0, 1, 3 with EV 9, 50, 40
    reviewed = select_reviews(p, amounts, 0.3, 2)
    assert reviewed.tolist() == [False, True, False, True, False]  # top-2 EV, not top-p
    # row 4 has the highest EV of all (500) but sits below the alert threshold:
    # analysts never see it - that is the censored region by construction
    assert not reviewed[4]
    # capacity >= alert volume -> exactly the alert mask
    assert select_reviews(p, amounts, 0.3, 3).tolist() == [True, True, False, True, False]
    assert select_reviews(p, amounts, 0.3, 99).sum() == 3
    # capacity 0 -> nobody reviewed
    assert select_reviews(p, amounts, 0.3, 0).sum() == 0
    # deterministic tie-break: equal EV resolves to the earlier transactions
    tie = select_reviews(np.array([0.5, 0.5, 0.5]), np.array([10.0, 10.0, 10.0]), 0.1, 2)
    assert tie.tolist() == [True, True, False]
    with pytest.raises(ValueError):
        select_reviews(p, amounts, 0.3, -1)
    with pytest.raises(ValueError):
        select_reviews(p[:3], amounts, 0.3, 2)


def test_label_policy_hand_checked():
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    reviewed = np.array([True, True, False, False, False, False])
    surfaced = np.array([False, False, True, False, False, False])
    labels, labelled = apply_label_policy("full_labels", y, reviewed, surfaced)
    assert labels.tolist() == [1, 0, 1, 0, 1, 0] and labelled.all()
    labels, labelled = apply_label_policy("reviewed_only", y, reviewed, surfaced)
    assert labelled.tolist() == reviewed.tolist()
    assert labels[labelled].tolist() == [1, 0]  # reviewed rows keep the truth
    labels, labelled = apply_label_policy("assume_legit", y, reviewed, surfaced)
    # rows 2 and 4 are real fraud relabelled LEGITIMATE - the poison, by hand
    assert labels.tolist() == [1, 0, 0, 0, 0, 0] and labelled.all()
    labels, labelled = apply_label_policy("chargeback", y, reviewed, surfaced)
    # row 2 self-reports (surfaced), row 4 does not: one fraud recovered, one lost
    assert labels.tolist() == [1, 0, 1, 0, 0, 0] and labelled.all()
    with pytest.raises(ValueError):
        apply_label_policy("no_such_policy", y, reviewed, surfaced)
    with pytest.raises(ValueError):
        apply_label_policy("full_labels", y[:3], reviewed, surfaced)


def test_label_policy_collapse_cases():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.1).astype(np.float64)
    reviewed = rng.random(300) < 0.2
    all_srf, no_srf = np.ones(300, dtype=bool), np.zeros(300, dtype=bool)
    # every unreviewed fraud self-reports -> chargeback IS full labels
    lab_cb, _ = apply_label_policy("chargeback", y, reviewed, all_srf)
    lab_full, _ = apply_label_policy("full_labels", y, reviewed, all_srf)
    assert np.array_equal(lab_cb, lab_full)
    # no chargeback ever lands -> chargeback IS assume-legit
    lab_cb0, _ = apply_label_policy("chargeback", y, reviewed, no_srf)
    lab_al, _ = apply_label_policy("assume_legit", y, reviewed, no_srf)
    assert np.array_equal(lab_cb0, lab_al)
    # infinite review capacity -> all four policies coincide exactly
    everyone = np.ones(300, dtype=bool)
    reference, _ = apply_label_policy("full_labels", y, everyone, no_srf)
    for policy in ARMS:
        labels, labelled = apply_label_policy(policy, y, everyone, no_srf)
        assert labelled.all() and np.array_equal(labels, reference), policy


def test_chargeback_rate_collapses_to_neighbouring_arms(small_results):
    """End-to-end collapse: rate 1.0 makes the chargeback ARM reproduce the
    full-labels arm row for row (identical pools -> identical retrains ->
    identical test metrics); rate 0.0 reproduces assume_legit."""
    traj1 = run_feedback(
        small_results, FeedbackConfig(reviews_per_round=40, chargeback_rate=1.0)
    )["trajectory"]
    pd.testing.assert_frame_equal(_arm(traj1, "chargeback"), _arm(traj1, "full_labels"))
    traj0 = run_feedback(
        small_results, FeedbackConfig(reviews_per_round=40, chargeback_rate=0.0)
    )["trajectory"]
    pd.testing.assert_frame_equal(_arm(traj0, "chargeback"), _arm(traj0, "assume_legit"))


def test_capacity_zero_freezes_reviewed_only_and_maximises_poison(small_results):
    """With no analysts at all: the reviewed-only arm never gains a label, so
    its pool and model are frozen at round 0; the assume-legit arm mislabels
    every single fraud in every round."""
    traj = run_feedback(small_results, FeedbackConfig(reviews_per_round=0))["trajectory"]
    ro = traj[traj["arm"] == "reviewed_only"]
    assert ro["pool_n"].nunique() == 1
    assert ro["test_pr_auc"].nunique() == 1  # identical pool -> identical retrain
    assert (ro["n_reviewed"] == 0).all()
    al = traj[(traj["arm"] == "assume_legit") & (traj["round"] > 0)]
    assert al["false_negative_labels_cum"].tolist() == al["round_fraud"].cumsum().tolist()
    assert al["observed_recall"].isna().all()  # it knows about no fraud at all
    fl = traj[(traj["arm"] == "full_labels") & (traj["round"] > 0)]
    assert (fl["observed_recall"] == 0.0).all()  # knows all fraud, reviewed none
    assert (fl["label_coverage"] == 1.0).all()   # labels need no reviews


def test_full_labels_arm_is_the_uncensored_ceiling(small_sim):
    traj = small_sim["trajectory"]
    fl = traj[traj["arm"] == "full_labels"]
    assert (fl["false_negative_labels_cum"] == 0).all()
    assert (fl[fl["round"] > 0]["label_coverage"] == 1.0).all()
    # its pool is the whole pre-test timeline: labelled == true prevalence
    assert np.allclose(fl["pool_labelled_prevalence"], fl["pool_true_prevalence"])
    n_initial = small_sim["windows"]["n_initial"]
    assert fl["pool_n"].iloc[-1] == n_initial + sum(small_sim["windows"]["n_rounds"])


def test_observed_recall_illusion_is_an_identity(small_sim):
    """The dashboard trap: arms that only label what they review report 100%
    recall of the fraud they know about, whenever they reviewed any fraud at
    all - while their true label coverage sits strictly below 1."""
    traj = small_sim["trajectory"]
    for arm in ("assume_legit", "reviewed_only"):
        rows = traj[(traj["arm"] == arm) & (traj["n_fraud_reviewed"] > 0)]
        assert not rows.empty
        assert (rows["observed_recall"] == 1.0).all(), arm
        assert (rows["label_coverage"] < 1.0).any(), arm
    # the oracle arm reports the honest number: below 1 when fraud went unreviewed
    fl = traj[(traj["arm"] == "full_labels") & (traj["round"] > 0)]
    partial = fl[fl["n_fraud_reviewed"] < fl["round_fraud"]]
    assert (partial["observed_recall"] < 1.0).all()


def test_selective_label_bias_directions(small_sim):
    traj = small_sim["trajectory"]
    last = traj[traj["round"] == traj["round"].max()].set_index("arm")
    # reviewed-only pool is fraud-enriched: chosen by the model's own opinions
    assert (
        last.loc["reviewed_only", "pool_labelled_prevalence"]
        > last.loc["full_labels", "pool_labelled_prevalence"]
    )
    # assume-legit pool under-reports its own fraud: labels below the truth
    assert (
        last.loc["assume_legit", "pool_labelled_prevalence"]
        < last.loc["assume_legit", "pool_true_prevalence"]
    )
    # chargebacks recover labels: strictly less poison than assuming legitimate
    assert (
        last.loc["chargeback", "false_negative_labels_cum"]
        < last.loc["assume_legit", "false_negative_labels_cum"]
    )
    # poison only accumulates
    for arm in ARMS:
        fn = traj[traj["arm"] == arm]["false_negative_labels_cum"]
        assert (fn.diff().dropna() >= 0).all(), arm


def test_full_run_structure_and_measured_mechanism(results, analysis):
    cfg = analysis["config"]
    traj = analysis["trajectory"]
    assert list(traj.columns) == TRAJECTORY_COLUMNS
    assert len(traj) == (cfg.n_rounds + 1) * len(ARMS)
    assert set(traj["arm"]) == set(ARMS)
    assert analysis["windows"]["n_test"] == int(results["splits"]["test"].size)
    assert analysis["t_star_frozen"] > 0.0
    # round 0 is genuinely shared: identical rows across arms except the name
    r0 = traj[traj["round"] == 0].drop(columns="arm")
    assert all(r0.iloc[0].equals(r0.iloc[i]) for i in range(1, len(r0)))
    # pool conservation: full arm labels the whole pre-test timeline,
    # reviewed_only labels exactly initial + what its analysts reviewed
    last = traj[traj["round"] == cfg.n_rounds].set_index("arm")
    n_initial = analysis["windows"]["n_initial"]
    assert int(last.loc["full_labels", "pool_n"]) == n_initial + sum(
        analysis["windows"]["n_rounds"]
    )
    ro = traj[traj["arm"] == "reviewed_only"]
    assert int(last.loc["reviewed_only", "pool_n"]) == n_initial + int(ro["n_reviewed"].sum())
    # the measured mechanism (deterministic on seed 7): censored labels leave
    # the RANKING within noise of the oracle arm but poison the PROBABILITIES,
    # so the frozen operating point starves - fewer alerts, less fraud caught
    fin = analysis["final_eval"]
    assert abs(fin["assume_legit"]["pr_auc"] - fin["full_labels"]["pr_auc"]) < 0.02
    assert fin["assume_legit"]["ece"] > 2.0 * fin["full_labels"]["ece"]
    assert fin["assume_legit"]["n_flagged"] < 0.5 * fin["full_labels"]["n_flagged"]
    assert fin["assume_legit"]["recall"] < fin["full_labels"]["recall"]
    assert fin["chargeback"]["recall"] >= 0.95 * fin["full_labels"]["recall"]
    # report: ASCII with the honesty framing stated
    text = "\n".join(feedback_lines(analysis))
    assert text.isascii()
    assert "selective-labels" in text and "assumptions" in text
    assert "grades its own homework" in text


def test_small_rerun_is_deterministic(small_results, small_sim, tmp_path):
    """No RNG outside the seeded chargeback draws, zero-init training: a
    second run must reproduce every number and byte."""
    again = run_feedback(small_results, SMALL_CFG)
    pd.testing.assert_frame_equal(again["trajectory"], small_sim["trajectory"])
    assert feedback_lines(again) == feedback_lines(small_sim)
    a = save_feedback_csv(small_sim, str(tmp_path / "a.csv"))
    b = save_feedback_csv(again, str(tmp_path / "b.csv"))
    assert open(a, "rb").read() == open(b, "rb").read()


def test_cli_feedback_runs_and_writes_csv(tmp_path, capsys):
    from fdo.__main__ import main

    figdir = str(tmp_path / "figs")
    assert main(["--feedback", "--figdir", figdir]) == 0
    out = capsys.readouterr().out
    assert "FEEDBACK LOOP" in out and "selective-labels" in out
    assert out.isascii()
    csv_path = os.path.join(figdir, "feedback_loop.csv")
    assert os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    body = open(csv_path, encoding="utf-8").read()
    assert body.splitlines()[0] == ",".join(TRAJECTORY_COLUMNS)
    for arm in ARMS:
        assert arm in body
