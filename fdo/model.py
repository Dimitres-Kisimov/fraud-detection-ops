"""Logistic regression from scratch (NumPy only).

- Stable sigmoid / binary cross-entropy with L2 penalty and analytic gradient
  (checked against finite differences in the tests).
- Two imbalance-aware losses: class-weighted BCE and focal loss
  (Lin et al. 2017). The focal implementation is generic in gamma and reduces
  *exactly* to weighted BCE at gamma = 0 (asserted in tests).
- Full-batch gradient descent with early stopping on validation loss.
- Strict TIME-BASED split: first ~70% train, next ~15% validation, final ~15%
  test. No shuffling, no leakage; the split is on the already time-ordered
  table and tests assert max(train time) < min(val time) < ... .
- Features standardized with TRAIN-ONLY statistics.
- Platt calibration (sigmoid(a*z + b), Platt 1999 target smoothing) fitted on
  VALIDATION logits after the class-weighted training, because class weights
  deliberately distort probabilities and downstream cost decisions need
  honest ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fdo.data import make_features

EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X])


def balanced_class_weights(y: np.ndarray, pos_weight_scale: float = 1.0) -> np.ndarray:
    """Per-sample weights: n / (2 * n_class), 'balanced' convention.

    ``pos_weight_scale`` multiplies the positive-class weight on top.
    """
    n = y.shape[0]
    n_pos = max(int(y.sum()), 1)
    n_neg = max(n - int(y.sum()), 1)
    w_pos = n / (2.0 * n_pos) * pos_weight_scale
    w_neg = n / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(np.float64)


def weighted_bce_loss_grad(
    theta: np.ndarray,
    Xb: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    l2: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Weighted BCE + (l2/2)*||theta[1:]||^2, normalized by sum of weights.

    Returns (loss, gradient). Gradient is analytic: X^T (w * (p - y)) / sum(w).
    """
    z = Xb @ theta
    # stable log(1 + exp(-|z|)) formulation
    ll = np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))
    wsum = sample_weight.sum()
    loss = float((sample_weight * ll).sum() / wsum)
    p = sigmoid(z)
    grad = Xb.T @ (sample_weight * (p - y)) / wsum
    if l2 > 0.0:
        reg = np.zeros_like(theta)
        reg[1:] = theta[1:]
        loss += 0.5 * l2 * float(reg @ reg)
        grad = grad + l2 * reg
    return loss, grad


def focal_loss_grad(
    theta: np.ndarray,
    Xb: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    l2: float = 0.0,
    gamma: float = 2.0,
) -> tuple[float, np.ndarray]:
    """Focal loss FL = -w * (1 - p_t)^gamma * log(p_t), normalized like BCE.

    Generic in gamma; at gamma = 0 it equals weighted BCE to machine
    precision (numpy evaluates 0**0 as 1). Analytic per-sample gradient in z:

        y=1: w * (gamma * p_t * (1-p_t)^gamma * log(p_t) - (1-p_t)^(gamma+1))
        y=0: same with p_t = 1 - p and the sign flipped,

    both verified against central finite differences in the tests.
    """
    z = Xb @ theta
    p = sigmoid(z)
    pt = np.where(y == 1, p, 1.0 - p)
    pt = np.clip(pt, EPS, 1.0 - EPS)
    log_pt = np.log(pt)
    one_m = 1.0 - pt
    wsum = sample_weight.sum()
    loss = float((sample_weight * (-(one_m**gamma) * log_pt)).sum() / wsum)
    sign = np.where(y == 1, 1.0, -1.0)
    dl_dz = sign * (gamma * pt * (one_m**gamma) * log_pt - one_m ** (gamma + 1.0))
    grad = Xb.T @ (sample_weight * dl_dz) / wsum
    if l2 > 0.0:
        reg = np.zeros_like(theta)
        reg[1:] = theta[1:]
        loss += 0.5 * l2 * float(reg @ reg)
        grad = grad + l2 * reg
    return loss, grad


class Standardizer:
    """Column-wise (x - mu) / sd with statistics from the TRAIN split only."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> Standardizer:
        self.mu = X_train.mean(axis=0)
        self.sd = np.maximum(X_train.std(axis=0), 1e-9)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sd is None:
            raise RuntimeError("Standardizer.transform called before fit")
        return (X - self.mu) / self.sd


def time_split(
    n: int, fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous time-ordered split indices: train, validation, test.

    Assumes rows 0..n-1 are already sorted by time (the generator guarantees
    strictly increasing timestamps). No shuffling - shuffling would leak
    future information into training.
    """
    if abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    idx = np.arange(n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    loss: str = "weighted_bce",
    gamma: float = 2.0,
    l2: float = 1e-4,
    lr: float = 2.0,
    max_iter: int = 3000,
    eval_every: int = 10,
    patience: int = 30,
    pos_weight_scale: float = 1.0,
) -> dict:
    """Full-batch gradient descent with early stopping on validation loss.

    Validation loss uses the same objective with class weights derived from
    the TRAIN labels. Keeps the best-validation weights. A simple divergence
    guard halves the learning rate and restores the best weights if the
    training loss blows up.
    """
    if loss not in ("weighted_bce", "focal"):
        raise ValueError(f"unknown loss: {loss}")
    Xb_tr = add_intercept(X_train)
    Xb_va = add_intercept(X_val)
    sw_tr = balanced_class_weights(y_train, pos_weight_scale)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    w_pos = len(y_train) / (2.0 * n_pos) * pos_weight_scale
    w_neg = len(y_train) / (2.0 * n_neg)
    sw_va = np.where(y_val == 1, w_pos, w_neg).astype(np.float64)

    def objective(theta: np.ndarray, Xb: np.ndarray, y: np.ndarray, sw: np.ndarray, pen: float):
        if loss == "focal":
            return focal_loss_grad(theta, Xb, y, sw, pen, gamma)
        return weighted_bce_loss_grad(theta, Xb, y, sw, pen)

    theta = np.zeros(Xb_tr.shape[1])
    best_theta = theta.copy()
    best_val = np.inf
    bad_evals = 0
    prev_train = np.inf
    history: list[tuple[int, float, float]] = []
    it = 0
    while it < max_iter:
        train_loss, grad = objective(theta, Xb_tr, y_train, sw_tr, l2)
        if not np.isfinite(train_loss) or train_loss > prev_train * 1.5 + 1e-9:
            lr *= 0.5  # divergence guard
            theta = best_theta.copy()
            prev_train = np.inf
            it += 1
            continue
        prev_train = train_loss
        theta = theta - lr * grad
        if it % eval_every == 0:
            val_loss, _ = objective(theta, Xb_va, y_val, sw_va, 0.0)
            history.append((it, train_loss, val_loss))
            if val_loss < best_val - 1e-9:
                best_val = val_loss
                best_theta = theta.copy()
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    break
        it += 1
    return {
        "theta": best_theta,
        "best_val_loss": best_val,
        "history": history,
        "n_iter": it,
        "loss": loss,
        "gamma": gamma if loss == "focal" else 0.0,
    }


def fit_platt(z_val: np.ndarray, y_val: np.ndarray, max_iter: int = 100) -> tuple[float, float]:
    """Platt scaling: fit (a, b) of sigmoid(a*z + b) on validation logits.

    Newton's method on the 2-parameter BCE with Platt's (1999) smoothed
    targets t+ = (N+ + 1)/(N+ + 2), t- = 1/(N- + 2). Unweighted on purpose:
    calibration must match the *actual* validation prevalence, not the
    reweighted training objective.
    """
    n_pos = float(y_val.sum())
    n_neg = float(len(y_val) - n_pos)
    t = np.where(y_val == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    a, b = 1.0, 0.0

    def nll(a_: float, b_: float) -> float:
        q = sigmoid(a_ * z_val + b_)
        q = np.clip(q, EPS, 1.0 - EPS)
        return float(-(t * np.log(q) + (1.0 - t) * np.log(1.0 - q)).sum())

    cur = nll(a, b)
    for _ in range(max_iter):
        q = sigmoid(a * z_val + b)
        r = q - t
        g = np.array([(r * z_val).sum(), r.sum()])
        w = np.maximum(q * (1.0 - q), 1e-12)
        h11 = (w * z_val * z_val).sum()
        h12 = (w * z_val).sum()
        h22 = w.sum()
        H = np.array([[h11, h12], [h12, h22]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        # damped Newton
        scale = 1.0
        for _ in range(30):
            na, nb = a - scale * step[0], b - scale * step[1]
            new = nll(na, nb)
            if new <= cur + 1e-12:
                break
            scale *= 0.5
        if abs(cur - new) < 1e-10:
            a, b, cur = na, nb, new
            break
        a, b, cur = na, nb, new
    return float(a), float(b)


@dataclass
class FraudModel:
    """End-to-end from-scratch pipeline: features -> train-only standardization
    -> weighted/focal logistic regression -> Platt calibration on validation.
    """

    loss: str = "weighted_bce"
    gamma: float = 2.0
    l2: float = 1e-4
    lr: float = 2.0
    max_iter: int = 3000
    # fitted state
    scaler: Standardizer = field(default_factory=Standardizer)
    theta: np.ndarray | None = None
    platt_a: float = 1.0
    platt_b: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    fit_info: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, idx_train: np.ndarray, idx_val: np.ndarray) -> FraudModel:
        X_all, names = make_features(df)
        self.feature_names = names
        y = df["is_fraud"].to_numpy(dtype=np.float64)
        self.scaler.fit(X_all[idx_train])
        X_tr = self.scaler.transform(X_all[idx_train])
        X_va = self.scaler.transform(X_all[idx_val])
        result = train_logistic(
            X_tr, y[idx_train], X_va, y[idx_val],
            loss=self.loss, gamma=self.gamma, l2=self.l2, lr=self.lr, max_iter=self.max_iter,
        )
        self.theta = result["theta"]
        self.fit_info = {k: v for k, v in result.items() if k != "theta"}
        z_val = self.decision_scores(df.iloc[idx_val])
        self.platt_a, self.platt_b = fit_platt(z_val, y[idx_val])
        return self

    def decision_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Raw logits z = [1, x_std] @ theta."""
        if self.theta is None:
            raise RuntimeError("model not fitted")
        X, _ = make_features(df)
        return add_intercept(self.scaler.transform(X)) @ self.theta

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilities BEFORE calibration (distorted by class weighting)."""
        return sigmoid(self.decision_scores(df))

    def predict_calibrated(self, df: pd.DataFrame) -> np.ndarray:
        """Platt-calibrated probabilities. Monotone in the raw score when
        a > 0, so ranking metrics (PR-AUC / ROC-AUC) are unchanged."""
        return sigmoid(self.platt_a * self.decision_scores(df) + self.platt_b)
