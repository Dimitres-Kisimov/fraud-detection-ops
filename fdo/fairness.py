"""Queue fairness: per-segment fraud-catch coverage.

The threshold and queue layers optimise for money (expected cost, expected
recovered value). This module asks the orthogonal, easy-to-forget question a
fraud-ops lead has to answer to an auditor: *of the fraud that actually
happened in each merchant segment, what share did we catch?* Equal dollars is
not equal protection - a cost-weighted policy rationally concentrates on
high-value segments and can leave a low-value segment almost unreviewed.

Two operational decisions are read out per segment on the held-out TEST
window:

- the cost-sensitive alert threshold t* (flag if calibrated p >= t*), and
- the capacity+coverage-constrained review queue (who the analysts actually
  review, from ``fdo/queue_opt.py``).

For each we report, per segment: the fraud that occurred, the fraud caught,
and the catch rate (segment recall). A ``summary`` collapses that into the
best/worst segment and the coverage gap between them - the fairness headline.

HONESTY: labels are the synthetic observed labels; a "caught" fraud here means
its transaction was flagged / put in the queue (the same optimistic
"reviewed => caught" assumption the cost model already makes). Segments with
zero observed fraud have an undefined catch rate and are excluded from the
gap; they are still listed so the read-out never hides a segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def segment_fraud_coverage(
    y: np.ndarray,
    amounts: np.ndarray,
    segments: np.ndarray,
    selected: np.ndarray,
) -> pd.DataFrame:
    """Per-segment fraud-catch coverage for one selection.

    ``selected`` may be a boolean mask or an array of integer indices into the
    window. Returns one row per segment: pool size, fraud count and value,
    fraud caught (count and value), and catch_rate = caught / fraud (NaN when a
    segment has no observed fraud).
    """
    y = np.asarray(y, dtype=np.float64)
    amounts = np.asarray(amounts, dtype=np.float64)
    segments = np.asarray(segments)
    sel_mask = np.zeros(y.shape[0], dtype=bool)
    selected = np.asarray(selected)
    if selected.dtype == bool:
        if selected.shape[0] != y.shape[0]:
            raise ValueError("boolean selection mask has wrong length")
        sel_mask = selected
    else:
        sel_mask[selected.astype(int)] = True

    is_fraud = y == 1
    rows = []
    for seg in sorted(np.unique(segments).tolist()):
        in_seg = segments == seg
        fraud_in = in_seg & is_fraud
        caught = fraud_in & sel_mask
        n_fraud = int(fraud_in.sum())
        n_caught = int(caught.sum())
        rows.append(
            {
                "segment": seg,
                "pool_size": int(in_seg.sum()),
                "n_fraud": n_fraud,
                "fraud_value": float(amounts[fraud_in].sum()),
                "n_fraud_caught": n_caught,
                "fraud_value_caught": float(amounts[caught].sum()),
                "catch_rate": (n_caught / n_fraud) if n_fraud else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _summary(coverage: pd.DataFrame, overall_recall: float) -> dict:
    have = coverage[coverage["n_fraud"] > 0]
    rates = have["catch_rate"]
    best_i = int(rates.idxmax())
    worst_i = int(rates.idxmin())
    zero = have[have["n_fraud_caught"] == 0]["segment"].tolist()
    return {
        "overall_recall": float(overall_recall),
        "segments_with_fraud": int(len(have)),
        "best_segment": str(coverage.loc[best_i, "segment"]),
        "best_catch_rate": float(coverage.loc[best_i, "catch_rate"]),
        "worst_segment": str(coverage.loc[worst_i, "segment"]),
        "worst_catch_rate": float(coverage.loc[worst_i, "catch_rate"]),
        "coverage_gap": float(rates.max() - rates.min()),
        "segments_zero_coverage": zero,
    }


def queue_fairness(results: dict) -> dict:
    """Build the per-segment fairness read-out for both operating decisions on
    the test window from a ``run_pipeline`` result dict."""
    df = results["df"]
    te = results["splits"]["test"]
    sub = df.iloc[te].reset_index(drop=True)
    y = sub["is_fraud"].to_numpy(dtype=np.float64)
    amounts = sub["amount"].to_numpy(dtype=np.float64)
    segments = sub["merchant_cat"].to_numpy()
    p = np.asarray(results["p_test"], dtype=np.float64)

    t_star = float(results["thresholds"]["threshold_star"])
    thr_selected = p >= t_star
    queue_selected = results["queue"]["selected_milp"]

    thr_cov = segment_fraud_coverage(y, amounts, segments, thr_selected)
    queue_cov = segment_fraud_coverage(y, amounts, segments, queue_selected)

    thr_recall = float(results["thresholds"]["test_at_star"]["recall"])
    n_fraud_total = float((y == 1).sum())
    queue_caught = int(((y == 1) & _mask(queue_selected, y.shape[0])).sum())
    queue_recall = queue_caught / n_fraud_total if n_fraud_total else float("nan")

    return {
        "threshold_star": t_star,
        "threshold_coverage": thr_cov,
        "threshold_summary": _summary(thr_cov, thr_recall),
        "queue_coverage": queue_cov,
        "queue_summary": _summary(queue_cov, queue_recall),
        "queue_capacity": int(results["queue"]["config"].capacity),
    }


def _mask(selected: np.ndarray, n: int) -> np.ndarray:
    selected = np.asarray(selected)
    if selected.dtype == bool:
        return selected
    m = np.zeros(n, dtype=bool)
    m[selected.astype(int)] = True
    return m


def fairness_lines(results: dict) -> list[str]:
    """ASCII summary lines for the CLI and the deliverables."""
    f = queue_fairness(results)
    ts = f["threshold_summary"]
    qs = f["queue_summary"]

    def _zero(names: list[str]) -> str:
        return ", ".join(names) if names else "none"

    return [
        "QUEUE FAIRNESS - per-segment fraud-catch coverage (test window, synthetic labels):",
        f"  Cost threshold t* = {f['threshold_star']:.3f}: overall recall "
        f"{ts['overall_recall']:.2f}; best segment {ts['best_segment']} "
        f"({ts['best_catch_rate']:.0%}) vs worst {ts['worst_segment']} "
        f"({ts['worst_catch_rate']:.0%}) - a {ts['coverage_gap']:.0%} coverage gap; "
        f"segments with zero fraud caught: {_zero(ts['segments_zero_coverage'])}.",
        f"  Review queue ({f['queue_capacity']} reviews): overall recall "
        f"{qs['overall_recall']:.2f}; best {qs['best_segment']} "
        f"({qs['best_catch_rate']:.0%}) vs worst {qs['worst_segment']} "
        f"({qs['worst_catch_rate']:.0%}); zero fraud caught in: "
        f"{_zero(qs['segments_zero_coverage'])}.",
        "  READ HONESTLY: a cost-weighted policy concentrates on high-value segments by "
        "design; the coverage floor guards review *headcount* per segment, not fraud-catch "
        "*rate*. Equal dollars is not equal protection.",
    ]


def save_fairness_csv(results: dict, path: str) -> str:
    """Write both operating-point coverage tables (tagged by ``decision``) to
    one tidy CSV. Returns the path."""
    f = queue_fairness(results)
    thr = f["threshold_coverage"].copy()
    thr.insert(0, "decision", f"cost_threshold_t*={f['threshold_star']:.3f}")
    q = f["queue_coverage"].copy()
    q.insert(0, "decision", f"review_queue_{f['queue_capacity']}_reviews")
    out = pd.concat([thr, q], ignore_index=True)
    out.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path
