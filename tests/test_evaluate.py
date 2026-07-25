from __future__ import annotations

import numpy as np
import pytest

from fdo.evaluate import (
    brier,
    ece,
    pr_auc,
    precision_at_k,
    reliability_curve,
    roc_auc,
)


def test_pr_auc_hand_cases():
    # groups: t=.9 -> P=1, R=.5 (+.5*1); t=.8 -> R unchanged; t=.7 -> P=2/3, R=1 (+.5*2/3)
    assert pr_auc(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])) == pytest.approx(5 / 6)
    # perfect ranking
    assert pr_auc(np.array([0, 1]), np.array([0.1, 0.9])) == pytest.approx(1.0)
    # all-tied scores collapse to one group: precision = prevalence at R=1
    assert pr_auc(np.array([1, 0, 0, 0]), np.array([0.5] * 4)) == pytest.approx(0.25)


def test_roc_auc_hand_cases():
    assert roc_auc(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])) == pytest.approx(0.5)
    assert roc_auc(np.array([0, 1]), np.array([0.1, 0.9])) == pytest.approx(1.0)
    assert roc_auc(np.array([1, 0]), np.array([0.1, 0.9])) == pytest.approx(0.0)
    # constant scores: ties everywhere -> 0.5 by average ranks
    assert roc_auc(np.array([1, 0, 1, 0]), np.array([0.3] * 4)) == pytest.approx(0.5)


def test_brier_and_precision_at_k_hand_cases():
    assert brier(np.array([1, 0]), np.array([0.8, 0.3])) == pytest.approx(0.065)
    assert precision_at_k(np.array([0, 1, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6]), 2) == 0.5
    assert precision_at_k(np.array([0, 1, 1, 0]), np.array([0.9, 0.8, 0.7, 0.6]), 3) == pytest.approx(2 / 3)


def test_ece_and_reliability_hand_cases():
    # perfectly calibrated single bin
    assert ece(np.array([1, 0, 0, 0]), np.array([0.25] * 4)) == pytest.approx(0.0)
    # maximally overconfident: predicted 0.9, observed rate 0
    assert ece(np.zeros(5), np.full(5, 0.9)) == pytest.approx(0.9)
    rel = reliability_curve(np.array([1, 0, 0, 0]), np.array([0.25] * 4), n_bins=10)
    assert rel["count"].sum() == 4
    assert rel["confidence"][0] == pytest.approx(0.25)
    assert rel["accuracy"][0] == pytest.approx(0.25)
    # quantile strategy partitions all samples too
    relq = reliability_curve(np.array([1, 0, 1, 0]), np.array([0.1, 0.2, 0.8, 0.9]),
                             n_bins=2, strategy="quantile")
    assert relq["count"].sum() == 4
