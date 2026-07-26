"""Drift-monitor tests: PSI math (hand-checked), report coverage, CLI, and
the honest detection story for the generator's known day-60 drift."""

from __future__ import annotations

import math
import os

import numpy as np

from fdo.data import generate_transactions, make_features
from fdo.drift import (
    STABLE_MAX,
    drift_from_results,
    drift_report,
    psi,
    psi_noise_floor,
)
from fdo.model import time_split


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=4000)
    assert psi(x, x) == 0.0
    flags = (rng.random(1000) < 0.3).astype(float)  # categorical path
    assert psi(flags, flags) == 0.0


def test_psi_hand_checked_two_bin_shift():
    """Binary 50/50 -> 90/10: PSI = 0.4*ln(0.9/0.5) + (-0.4)*ln(0.1/0.5),
    checkable by hand."""
    expected = np.array([0.0] * 50 + [1.0] * 50)
    actual = np.array([0.0] * 90 + [1.0] * 10)
    by_hand = 0.4 * math.log(0.9 / 0.5) + (-0.4) * math.log(0.1 / 0.5)
    assert abs(psi(expected, actual) - by_hand) < 1e-12
    assert abs(by_hand - 0.8789) < 1e-4  # the arithmetic itself


def test_psi_grows_with_shift_and_noise_floor_is_sane():
    rng = np.random.default_rng(1)
    base = rng.normal(size=5000)
    vals = [psi(base, rng.normal(loc=mu, size=5000)) for mu in (0.0, 0.3, 1.0)]
    assert vals[0] < vals[1] < vals[2]
    floor = psi_noise_floor(5000, 5000, 10)
    assert 0.0 < floor["expected_if_no_change"] < floor["p95_if_no_change"]
    # two draws from the SAME distribution: positive but inside the noise band
    assert 0.0 < vals[0] < floor["p95_if_no_change"]
    assert vals[2] > STABLE_MAX  # a full-sigma mean shift is a major shift


def test_report_covers_all_features_plus_score():
    df = generate_transactions(n=12000, seed=3)
    idx_tr, _, idx_te = time_split(len(df))
    train, test = df.iloc[idx_tr], df.iloc[idx_te]
    _, names = make_features(df)
    rng = np.random.default_rng(2)
    rep = drift_report(train, test, names,
                       scores_train=rng.random(len(train)),
                       scores_recent=rng.random(len(test)))
    assert list(rep["feature"]) == names + ["model_score"]
    assert (rep["psi"] >= 0.0).all()
    assert set(rep.columns) == {"feature", "psi", "verdict"}
    assert rep["verdict"].str.contains("stable|moderate|major", regex=True).all()


def test_day60_drift_detected_where_it_actually_lives(results):
    """The generator's drift is CONCEPT drift: covariate PSI must stay in the
    stable band (features are time-stationary by construction), while the
    label-aware fraud-mix monitor must cross 0.10 and clear its own
    small-sample noise floor."""
    analysis = drift_from_results(results)
    rep = analysis["report"]
    assert (rep["psi"] < STABLE_MAX).all()  # honest: no covariate drift exists
    fm = analysis["fraud_mix"]
    assert fm["category_psi"] > STABLE_MAX  # above the stable band
    assert fm["category_psi"] > fm["category_noise"]["expected_if_no_change"]
    # the constructed shift: gift-card fraud share rises, night share falls
    mix = fm["mix"].set_index("category")
    assert mix.loc["gift_cards", "delta_pp"] > 5.0
    assert fm["night_share_recent"] < fm["night_share_train"] - 0.05


def test_cli_drift_runs_and_writes_figure(tmp_path, capsys):
    from fdo.__main__ import main

    figdir = str(tmp_path / "figs")
    assert main(["--drift", "--figdir", figdir]) == 0
    out = capsys.readouterr().out
    assert "Drift monitor" in out and "PSI" in out
    assert out.isascii()
    fig = os.path.join(figdir, "drift_psi.png")
    assert os.path.exists(fig) and os.path.getsize(fig) > 10_240
