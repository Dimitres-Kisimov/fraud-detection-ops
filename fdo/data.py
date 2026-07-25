"""Seeded synthetic transaction generator.

HONESTY NOTE: every row here is synthetic. No real customer, merchant or
transaction data is used anywhere in this project. The fraud patterns below
are *constructed*, which means a model can learn them by design; the point of
the project is the operational machinery around the model (metrics that
survive 1.5% prevalence, calibration, cost-based thresholds, review-queue
allocation), not a claim that these exact patterns exist in production data.

Construction (all deterministic given ``seed``):

- ~``n`` transactions over ``horizon_days`` days, timestamps strictly
  increasing (exponential inter-arrival times).
- Features: amount (log-normal), hour of day, merchant category (8 values),
  customer tenure in days, 24h velocity count, device-change flag.
- A latent risk score combines interpretable patterns: night-time activity,
  device changes, high velocity, risky merchant categories (gift cards,
  digital goods, electronics), new accounts, and large amounts (with an
  amount x gift/digital interaction the linear model does NOT get as a
  feature, so the model has honest headroom below the oracle).
- The intercept is solved by bisection so the *average* fraud probability is
  ``target_prevalence`` (default 1.5%). Labels are Bernoulli draws.
- Label noise: 4% of true frauds are recorded as legitimate (unreported
  fraud) and 0.08% of legitimate rows are recorded as fraud (wrongly upheld
  disputes). ``is_fraud`` is the noisy *observed* label used for training and
  evaluation; the pre-noise truth is exposed as ``is_fraud_true`` and the
  generator probability as ``p_fraud_true``.
- Mild drift: from day 60 onward the pattern mix shifts (gift-card and
  electronics fraud rise, night-time fraud falls), so the later validation
  and test windows are not identically distributed to training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = [
    "grocery",
    "restaurants",
    "fuel",
    "fashion",
    "electronics",
    "travel",
    "digital_goods",
    "gift_cards",
]

CATEGORY_PROBS = np.array([0.22, 0.18, 0.12, 0.12, 0.10, 0.08, 0.10, 0.08])

# Latent risk contribution per category (constructed pattern, not an estimate).
CATEGORY_RISK = {
    "grocery": -0.60,
    "restaurants": 0.00,
    "fuel": -0.40,
    "fashion": 0.10,
    "electronics": 1.30,
    "travel": 0.40,
    "digital_goods": 1.60,
    "gift_cards": 2.20,
}

START = pd.Timestamp("2025-09-01 00:00:00")


def _solve_intercept(z: np.ndarray, target: float) -> float:
    """Bisection for b so that mean(sigmoid(z + b)) == target. Deterministic."""
    lo, hi = -15.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(z + mid)))
        if p.mean() > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def generate_transactions(
    n: int = 60_000,
    seed: int = 7,
    horizon_days: float = 120.0,
    target_prevalence: float = 0.015,
) -> pd.DataFrame:
    """Generate the synthetic, time-ordered transaction table.

    Returns a DataFrame sorted by time with strictly increasing timestamps.
    Ground truth is exposed (``p_fraud_true``, ``is_fraud_true``) next to the
    noisy observed label ``is_fraud``.
    """
    rng = np.random.default_rng(seed)

    # --- timestamps: exponential inter-arrivals, strictly increasing -------
    gaps = rng.exponential(scale=horizon_days / n, size=n)
    ts_days = np.cumsum(gaps)
    ts_days = ts_days / ts_days[-1] * horizon_days  # normalize to the horizon
    ts_days = ts_days + np.arange(n) * 1e-9  # guarantee strict monotonicity
    hour = (ts_days % 1.0) * 24.0
    hour_int = np.floor(hour).astype(np.int64)

    # --- features ----------------------------------------------------------
    amount = rng.lognormal(mean=3.65, sigma=1.0, size=n)
    amount = np.round(amount, 2)
    cat_idx = rng.choice(len(CATEGORIES), size=n, p=CATEGORY_PROBS)
    tenure_days = np.minimum(rng.exponential(scale=400.0, size=n), 3000.0)
    tenure_days = np.round(tenure_days).astype(np.int64)
    lam = rng.gamma(shape=1.6, scale=0.9, size=n)
    velocity_24h = rng.poisson(lam=lam).astype(np.int64)
    device_change = (rng.random(n) < 0.05).astype(np.int64)

    # --- latent risk score (interpretable, constructed) --------------------
    is_night = (hour_int < 6).astype(np.float64)
    is_new_account = (tenure_days < 30).astype(np.float64)
    log_amount = np.log(amount)
    amount_z = (log_amount - 3.65) / 1.0
    cat_risk = np.array([CATEGORY_RISK[CATEGORIES[i]] for i in cat_idx])
    is_gift_or_digital = np.isin(cat_idx, [CATEGORIES.index("gift_cards"),
                                           CATEGORIES.index("digital_goods")]).astype(np.float64)
    is_electronics = (cat_idx == CATEGORIES.index("electronics")).astype(np.float64)
    is_gift = (cat_idx == CATEGORIES.index("gift_cards")).astype(np.float64)

    z = (
        1.60 * is_night
        + 2.10 * device_change
        + 0.50 * np.clip(velocity_24h - 2, 0, 6)
        + cat_risk
        + 1.70 * is_new_account
        + 0.80 * amount_z
        + 0.90 * amount_z * is_gift_or_digital  # interaction NOT given to the model
        + rng.normal(0.0, 0.55, size=n)  # unmodeled heterogeneity
    )

    # Mild drift: ramps 0 -> 1 between day 60 and the end of the horizon.
    drift = np.clip((ts_days - 60.0) / (horizon_days - 60.0), 0.0, 1.0)
    z = z + drift * (0.60 * is_gift + 0.50 * is_electronics - 0.45 * is_night)

    b = _solve_intercept(z, target_prevalence)
    p_true = 1.0 / (1.0 + np.exp(-(z + b)))

    is_fraud_true = (rng.random(n) < p_true).astype(np.int64)

    # --- label noise (observed labels) -------------------------------------
    flip_pos = rng.random(n) < 0.04    # unreported fraud
    flip_neg = rng.random(n) < 0.0008  # false dispute upheld
    is_fraud = np.where(is_fraud_true == 1, np.where(flip_pos, 0, 1),
                        np.where(flip_neg, 1, 0)).astype(np.int64)

    df = pd.DataFrame(
        {
            "ts_days": ts_days,
            "timestamp": START + pd.to_timedelta(ts_days, unit="days"),
            "amount": amount,
            "hour": hour_int,
            "merchant_cat": [CATEGORIES[i] for i in cat_idx],
            "tenure_days": tenure_days,
            "velocity_24h": velocity_24h,
            "device_change": device_change,
            "p_fraud_true": p_true,
            "is_fraud_true": is_fraud_true,
            "is_fraud": is_fraud,
        }
    )
    return df


def make_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Design matrix for the linear model (no intercept column; the trainer
    adds one). Uses only operationally observable columns - never the ground
    truth. One-hot merchant category drops 'grocery' as the reference level.

    Note the generator's amount x gift/digital interaction is deliberately
    absent here, so the linear model cannot fully recover the oracle.
    """
    hour = df["hour"].to_numpy(dtype=np.float64)
    tenure = df["tenure_days"].to_numpy(dtype=np.float64)
    cols: dict[str, np.ndarray] = {
        "log_amount": np.log(df["amount"].to_numpy(dtype=np.float64)),
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
        "is_night": (hour < 6).astype(np.float64),
        "log_tenure": np.log1p(tenure),
        "is_new_account": (tenure < 30).astype(np.float64),
        "velocity_24h": df["velocity_24h"].to_numpy(dtype=np.float64),
        "device_change": df["device_change"].to_numpy(dtype=np.float64),
    }
    for cat in CATEGORIES:
        if cat == "grocery":  # reference level
            continue
        cols[f"cat_{cat}"] = (df["merchant_cat"] == cat).to_numpy().astype(np.float64)
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return X, names
