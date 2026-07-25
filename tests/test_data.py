from __future__ import annotations

import numpy as np

from fdo.data import generate_transactions, make_features


def test_generator_deterministic_ordered_and_seed_sensitive():
    a = generate_transactions(n=8000, seed=7)
    b = generate_transactions(n=8000, seed=7)
    c = generate_transactions(n=8000, seed=8)
    assert a.equals(b)
    assert not a["is_fraud"].equals(c["is_fraud"])
    assert np.all(np.diff(a["ts_days"].to_numpy()) > 0)  # strictly time-ordered


def test_prevalence_in_range():
    df = generate_transactions()  # default n=60000, target 1.5%
    assert 0.012 <= df["is_fraud"].mean() <= 0.018
    assert 0.012 <= df["is_fraud_true"].mean() <= 0.018


def test_features_never_touch_ground_truth():
    """Zeroing every ground-truth column must not change the design matrix."""
    df = generate_transactions(n=5000, seed=11)
    X1, names = make_features(df)
    df2 = df.copy()
    df2["p_fraud_true"] = 0.0
    df2["is_fraud_true"] = 0
    df2["is_fraud"] = 0
    X2, names2 = make_features(df2)
    assert names == names2
    np.testing.assert_array_equal(X1, X2)
