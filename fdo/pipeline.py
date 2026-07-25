"""End-to-end orchestration: one deterministic run that every consumer
(CLI summary, PDF/Excel exports, tests) shares, so all reported numbers come
from the same place.
"""

from __future__ import annotations

import numpy as np

from fdo.data import generate_transactions
from fdo.evaluate import (
    amount_p99_heuristic,
    brier,
    ece,
    pr_auc,
    pr_curve,
    prevalence_random_baseline,
    reliability_curve,
    summarize_scores,
)
from fdo.model import FraudModel, time_split
from fdo.queue_opt import QueueConfig, plan_queue
from fdo.threshold import CostAssumptions, threshold_report


def run_pipeline(
    seed: int = 7,
    n: int = 60_000,
    costs: CostAssumptions | None = None,
    queue_config: QueueConfig | None = None,
) -> dict:
    """Generate data, train both losses, calibrate, evaluate against
    baselines, pick the operating threshold, and plan the review queue.

    Model selection (weighted BCE vs focal) uses VALIDATION PR-AUC only;
    the test window is touched exactly once per model for final reporting.
    """
    costs = costs or CostAssumptions()
    queue_config = queue_config or QueueConfig()

    df = generate_transactions(n=n, seed=seed)
    idx_tr, idx_va, idx_te = time_split(len(df))
    y = df["is_fraud"].to_numpy(dtype=np.float64)
    amounts = df["amount"].to_numpy(dtype=np.float64)

    models = {
        "weighted_bce": FraudModel(loss="weighted_bce").fit(df, idx_tr, idx_va),
        "focal_gamma2": FraudModel(loss="focal", gamma=2.0).fit(df, idx_tr, idx_va),
    }
    val_pr = {
        name: pr_auc(y[idx_va], m.predict_calibrated(df.iloc[idx_va]))
        for name, m in models.items()
    }
    primary_name = max(val_pr, key=lambda k: val_pr[k])
    primary = models[primary_name]

    p_val_raw = primary.predict_raw(df.iloc[idx_va])
    p_val = primary.predict_calibrated(df.iloc[idx_va])
    p_test_raw = primary.predict_raw(df.iloc[idx_te])
    p_test = primary.predict_calibrated(df.iloc[idx_te])

    metrics = {
        "primary_model": primary_name,
        "val_pr_auc_by_model": val_pr,
        "test_by_model": {
            name: summarize_scores(y[idx_te], m.predict_calibrated(df.iloc[idx_te]))
            for name, m in models.items()
        },
        "val_primary": summarize_scores(y[idx_va], p_val),
        "test_primary": summarize_scores(y[idx_te], p_test),
        "calibration": {
            "val_ece_raw": ece(y[idx_va], p_val_raw),
            "val_ece_cal": ece(y[idx_va], p_val),
            "val_brier_raw": brier(y[idx_va], p_val_raw),
            "val_brier_cal": brier(y[idx_va], p_val),
            "test_ece_raw": ece(y[idx_te], p_test_raw),
            "test_ece_cal": ece(y[idx_te], p_test),
            "test_brier_raw": brier(y[idx_te], p_test_raw),
            "test_brier_cal": brier(y[idx_te], p_test),
            "platt_a": primary.platt_a,
            "platt_b": primary.platt_b,
        },
        "baseline_random": prevalence_random_baseline(y[idx_te]),
        "baseline_p99": amount_p99_heuristic(y[idx_te], amounts[idx_te], amounts[idx_tr]),
        # Oracle = PR-AUC of the generator's own probability. Only possible
        # because the data is synthetic; it is the honest ceiling.
        "oracle_test_pr_auc": pr_auc(y[idx_te], df["p_fraud_true"].to_numpy()[idx_te]),
    }

    thresholds = threshold_report(
        y[idx_va], amounts[idx_va], p_val, y[idx_te], amounts[idx_te], p_test, costs
    )

    queue = plan_queue(
        p_test,
        amounts[idx_te],
        df["merchant_cat"].to_numpy()[idx_te],
        queue_config,
        y=y[idx_te],
    )

    # Curve data for the PDF.
    prec_m, rec_m, _ = pr_curve(y[idx_te], p_test)
    plots = {
        "pr_model_test": (prec_m, rec_m),
        # quantile bins for PLOTTING only; the ECE numbers use the
        # equal-width definition (see evaluate.reliability_curve docstring)
        "reliability_val_raw": reliability_curve(y[idx_va], p_val_raw, strategy="quantile"),
        "reliability_val_cal": reliability_curve(y[idx_va], p_val, strategy="quantile"),
    }

    return {
        "seed": seed,
        "df": df,
        "splits": {"train": idx_tr, "val": idx_va, "test": idx_te},
        "models": models,
        "primary_name": primary_name,
        "p_val": p_val,
        "p_test": p_test,
        "metrics": metrics,
        "thresholds": thresholds,
        "queue": queue,
        "plots": plots,
    }


def headline(results: dict) -> list[str]:
    """ASCII-only summary lines shared by the CLI and the PDF cover page."""
    m = results["metrics"]
    t = results["thresholds"]
    q = results["queue"]
    test = m["test_primary"]
    cal = m["calibration"]
    comp = q["comparison"].set_index("strategy")["expected_recovered"]
    milp = comp["constrained_optimizer (MILP)"]
    topk = comp["top_k_by_probability"]
    rand = comp["random_selection (analytic expectation)"]
    return [
        f"Primary model: logistic regression ({m['primary_model']}), from-scratch NumPy.",
        f"Test PR-AUC {test['pr_auc']:.3f} vs prevalence-random "
        f"{m['baseline_random']['pr_auc']:.3f} and amount>p99 rule "
        f"{m['baseline_p99']['pr_auc']:.3f} (oracle ceiling {m['oracle_test_pr_auc']:.3f}).",
        f"Test ROC-AUC {test['roc_auc']:.3f}; precision-at-100 {test.get('precision_at_100', float('nan')):.2f} "
        f"at {test['prevalence']:.2%} prevalence.",
        f"Calibration (val): ECE {cal['val_ece_raw']:.3f} -> {cal['val_ece_cal']:.3f}, "
        f"Brier {cal['val_brier_raw']:.4f} -> {cal['val_brier_cal']:.4f} after Platt scaling.",
        f"Cost threshold t*={t['threshold_star']:.3f} chosen on validation; test cost "
        f"${t['test_at_star']['total_cost']:,.0f} vs ${t['test_at_naive_05']['total_cost']:,.0f} "
        f"at naive 0.5 ({t['test_cost_reduction_vs_05']:.1%} lower, under stated cost assumptions).",
        f"Review queue ({q['config'].capacity} reviews): expected recovered "
        f"${milp:,.0f} constrained-optimizer vs ${topk:,.0f} top-K-by-probability "
        f"vs ${rand:,.0f} random.",
    ]
