from __future__ import annotations

import numpy as np

from fdo.data import generate_transactions
from fdo.evaluate import pr_auc
from fdo.model import (
    Standardizer,
    add_intercept,
    balanced_class_weights,
    focal_loss_grad,
    time_split,
    weighted_bce_loss_grad,
)


def _toy_problem(seed: int = 0, n: int = 60, d: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.25).astype(np.float64)
    theta = rng.normal(scale=0.5, size=d + 1)
    sw = balanced_class_weights(y)
    return add_intercept(X), y, theta, sw


def test_gradients_match_finite_differences():
    Xb, y, theta, sw = _toy_problem()
    cases = [
        (weighted_bce_loss_grad, {}),
        (focal_loss_grad, {"gamma": 2.0}),
        (focal_loss_grad, {"gamma": 0.5}),
    ]
    for fn, kw in cases:
        _, grad = fn(theta, Xb, y, sw, 1e-3, **kw)
        num = np.zeros_like(theta)
        eps = 1e-6
        for j in range(theta.size):
            e = np.zeros_like(theta)
            e[j] = eps
            lp, _ = fn(theta + e, Xb, y, sw, 1e-3, **kw)
            lm, _ = fn(theta - e, Xb, y, sw, 1e-3, **kw)
            num[j] = (lp - lm) / (2 * eps)
        rel = np.max(np.abs(grad - num) / (np.abs(num) + 1e-9))
        assert rel < 1e-5, f"{fn.__name__} {kw}: rel grad error {rel}"


def test_focal_at_gamma_zero_equals_weighted_bce():
    for seed in (0, 1, 2):
        Xb, y, theta, sw = _toy_problem(seed=seed)
        l_b, g_b = weighted_bce_loss_grad(theta, Xb, y, sw, 1e-3)
        l_f, g_f = focal_loss_grad(theta, Xb, y, sw, 1e-3, gamma=0.0)
        assert abs(l_b - l_f) < 1e-10
        np.testing.assert_allclose(g_b, g_f, atol=1e-10)


def test_time_split_never_leaks():
    df = generate_transactions(n=10000, seed=5)
    tr, va, te = time_split(len(df))
    ts = df["ts_days"].to_numpy()
    assert ts[tr].max() < ts[va].min() < ts[va].max() < ts[te].min()
    assert len(tr) + len(va) + len(te) == len(df)
    # contiguous and ordered: no index from the future inside train
    assert tr.max() < va.min() and va.max() < te.min()


def test_standardizer_uses_train_stats_only():
    rng = np.random.default_rng(2)
    X_train = rng.normal(loc=2.0, scale=3.0, size=(500, 4))
    X_val = rng.normal(loc=-1.0, scale=0.5, size=(200, 4))
    sc = Standardizer().fit(X_train)
    np.testing.assert_allclose(sc.mu, X_train.mean(axis=0))
    np.testing.assert_allclose(sc.sd, X_train.std(axis=0))
    t = sc.transform(X_train)
    np.testing.assert_allclose(t.mean(axis=0), 0.0, atol=1e-12)
    # validation transformed with TRAIN stats keeps its distribution shift
    assert abs(sc.transform(X_val).mean()) > 0.5


def test_platt_is_monotone_and_preserves_ranking(results):
    df = results["df"]
    va = results["splits"]["val"]
    primary = results["models"][results["primary_name"]]
    assert primary.platt_a > 0  # monotone rescaling
    y = df["is_fraud"].to_numpy(dtype=float)[va]
    raw = primary.predict_raw(df.iloc[va])
    cal = primary.predict_calibrated(df.iloc[va])
    assert abs(pr_auc(y, raw) - pr_auc(y, cal)) < 1e-12
