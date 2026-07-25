"""Evaluation metrics from scratch (NumPy only) plus honest baselines.

Why PR-AUC is the primary metric here: at ~1.5% fraud prevalence, accuracy is
meaningless (predicting "legitimate" for everything scores ~98.5%) and even
ROC-AUC is dominated by the overwhelming negative class. Precision-recall
directly answers the operational question: "of the transactions we flag, how
many are fraud, at every possible review depth?" A random scorer's PR-AUC
equals the prevalence, which is the honest floor every model must be compared
against.

Calibration metrics (Brier, ECE) are reported because the cost-based
threshold and the review-queue optimizer consume *probabilities*, not ranks;
a miscalibrated model can rank well and still produce garbage expected-value
decisions.

Ties in scores are handled by grouping distinct score values (PR) and by
average ranks (ROC), so binary/constant scorers get well-defined values.
"""

from __future__ import annotations

import numpy as np


def _check(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError("y and scores must have the same shape")
    if y.size == 0:
        raise ValueError("empty input")
    return y, s


def pr_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at each distinct score threshold (descending).

    Returns (precision, recall, thresholds); one point per distinct score
    value, computed after including the whole tie group (the standard
    threshold-based construction).
    """
    y, s = _check(y, scores)
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    s_sorted = s[order]
    group_end = np.nonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])[0]
    tp = np.cumsum(y_sorted)[group_end]
    fp = np.cumsum(1.0 - y_sorted)[group_end]
    n_pos = y.sum()
    if n_pos == 0:
        raise ValueError("PR curve undefined without positives")
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return precision, recall, s_sorted[group_end]


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision: sum over thresholds of (R_k - R_{k-1}) * P_k.

    Step-wise integral of the PR curve; no trapezoidal interpolation, which
    is known to be optimistic for PR curves (Davis & Goadrich 2006).
    """
    precision, recall, _ = pr_curve(y, scores)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _average_ranks(s: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (from scratch, no scipy)."""
    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    is_new = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    group_id = np.cumsum(is_new) - 1
    counts = np.bincount(group_id)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    avg_rank = starts + (counts + 1.0) / 2.0  # 1-based average rank per group
    ranks = np.empty_like(s)
    ranks[order] = avg_rank[group_id]
    return ranks


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U formulation with tie-averaged ranks."""
    y, s = _check(y, scores)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC undefined with a single class")
    ranks = _average_ranks(s)
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of fraud among the k highest-scored transactions."""
    y, s = _check(y, scores)
    if not 0 < k <= y.size:
        raise ValueError("k out of range")
    top = np.argsort(-s, kind="stable")[:k]
    return float(y[top].mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probability forecast."""
    y, p = _check(y, p)
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    y: np.ndarray, p: np.ndarray, n_bins: int = 15, strategy: str = "width"
) -> dict[str, np.ndarray]:
    """Reliability bins for calibration plots and ECE.

    ``strategy="width"``: equal-width bins on [0, 1] - the ECE definition.
    ``strategy="quantile"``: equal-count bins - used for *plots*, because at
    1.5% prevalence almost all calibrated probabilities are tiny and
    equal-width bins high on the axis hold a handful of samples whose
    empirical rates are pure noise.

    Returns per-NON-EMPTY-bin arrays: mean predicted probability
    (``confidence``), empirical fraud rate (``accuracy``), and ``count``.
    """
    y, p = _check(y, p)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 3:  # degenerate (near-constant p): fall back
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        n_eff = edges.size - 1
        bin_idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_eff - 1)
    elif strategy == "width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        n_eff = n_bins
        bin_idx = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    else:
        raise ValueError(f"unknown binning strategy: {strategy}")
    counts = np.bincount(bin_idx, minlength=n_eff).astype(np.float64)
    sum_p = np.bincount(bin_idx, weights=p, minlength=n_eff)
    sum_y = np.bincount(bin_idx, weights=y, minlength=n_eff)
    nonempty = counts > 0
    return {
        "confidence": sum_p[nonempty] / counts[nonempty],
        "accuracy": sum_y[nonempty] / counts[nonempty],
        "count": counts[nonempty],
        "bin_edges": edges,
    }


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: count-weighted mean |accuracy - confidence|."""
    rel = reliability_curve(y, p, n_bins)
    w = rel["count"] / rel["count"].sum()
    return float(np.sum(w * np.abs(rel["accuracy"] - rel["confidence"])))


# ---------------------------------------------------------------------------
# Baselines (both are actually measured, not assumed)
# ---------------------------------------------------------------------------

def prevalence_random_baseline(y: np.ndarray, seed: int = 123) -> dict[str, float]:
    """Seeded uniform-random scores. In expectation PR-AUC equals the
    prevalence; we report the measured value alongside that theoretical floor.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    scores = np.random.default_rng(seed).random(y.size)
    return {
        "pr_auc": pr_auc(y, scores),
        "roc_auc": roc_auc(y, scores),
        "theoretical_pr_auc": float(y.mean()),
    }


def amount_p99_heuristic(
    y: np.ndarray, amounts: np.ndarray, train_amounts: np.ndarray
) -> dict[str, float]:
    """Named single-rule baseline: flag any transaction whose amount exceeds
    the 99th percentile of TRAIN amounts (the cutoff is fit on train only,
    so this rule is evaluated out-of-sample like the model).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    amounts = np.asarray(amounts, dtype=np.float64).ravel()
    cutoff = float(np.quantile(np.asarray(train_amounts, dtype=np.float64), 0.99))
    flag = (amounts >= cutoff).astype(np.float64)
    n_flagged = int(flag.sum())
    flag_precision = float(y[flag == 1].mean()) if n_flagged else 0.0
    flag_recall = float(y[flag == 1].sum() / max(y.sum(), 1.0))
    return {
        "pr_auc": pr_auc(y, flag),
        "roc_auc": roc_auc(y, flag),
        "cutoff": cutoff,
        "n_flagged": n_flagged,
        "flag_precision": flag_precision,
        "flag_recall": flag_recall,
    }


def summarize_scores(y: np.ndarray, p: np.ndarray, ks: tuple[int, ...] = (100, 500)) -> dict:
    """Bundle of all headline metrics for one scored dataset."""
    out = {
        "pr_auc": pr_auc(y, p),
        "roc_auc": roc_auc(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
        "prevalence": float(np.asarray(y, dtype=np.float64).mean()),
        "n": int(np.asarray(y).size),
    }
    for k in ks:
        if k <= np.asarray(y).size:
            out[f"precision_at_{k}"] = precision_at_k(y, p, k)
    return out
