"""Cost-vs-threshold curve: a referenceable, plain-language decision layer on
top of the threshold sweep in ``fdo/threshold.py``.

``fdo/threshold.py`` already *computes* the cost-minimising alert threshold
(sweep on validation, judged on test). This module does not recompute any of
that; it turns the existing sweep into artifacts a fraud-ops reader can act on:

- a tidy per-threshold table (review-queue volume, precision, recall, and the
  cost breakdown) written to CSV,
- a hand-drawn SVG of total cost vs threshold (no plotting dependency, fully
  deterministic), and
- a short plain-language recommendation.

HONESTY / METHOD (unchanged from ``fdo/threshold.py``):

- The curve is the sweep on the VALIDATION window; the operating threshold t*
  is chosen there and then judged once on the held-out TEST window. Sweeping
  and choosing on test would leak the test labels into the decision, which the
  repo deliberately avoids - so the curve you see is validation, and the
  recommendation reports t* re-measured on test next to it.
- Every dollar figure rests on the labelled COST ASSUMPTIONS in
  ``fdo.threshold.CostAssumptions`` ($8/review, missed fraud = its amount).
  Change the assumptions and t* moves; nothing here is a guarantee.
- "precision" at a threshold is #fraud-caught / #flagged on that window; at a
  1.4% base rate it is low by construction, which is the whole reason a
  cost-weighted threshold beats a naive 0.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Palette from fdo/palette.py, imported without pulling in matplotlib (this
# module hand-draws its SVG on purpose). One series = one hue: the cost of
# running the alert queue is drawn in alert amber, the chosen operating point
# gets the sparing cleared-green accent, and all text stays in ink tokens.
from fdo.palette import ALERT_AMBER as _SERIES
from fdo.palette import BASELINE as _AXIS
from fdo.palette import CLEARED_GREEN as _CHOSEN
from fdo.palette import GRID as _GRID
from fdo.palette import INK as _INK
from fdo.palette import INK_2 as _INK_2
from fdo.palette import MUTED as _MUTED
from fdo.palette import SURFACE as _SURFACE


def cost_curve_frame(results: dict) -> pd.DataFrame:
    """Tidy per-threshold decision table from the validation sweep.

    Columns: threshold, review_queue_volume (alerts sent to analysts),
    precision, recall, review_cost, missed_fraud_loss, total_cost, and two
    marker flags (is_chosen_t_star, is_naive_default). Sorted by threshold.
    """
    sweep = results["thresholds"]["sweep_val"]
    t_star = float(results["thresholds"]["threshold_star"])
    df = sweep.sort_values("threshold").reset_index(drop=True)
    flagged = df["n_flagged"].to_numpy(dtype=np.float64)
    caught = df["n_fraud_caught"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, caught / flagged, 0.0)
    out = pd.DataFrame(
        {
            "threshold": df["threshold"],
            "review_queue_volume": df["n_flagged"].astype(int),
            "precision": precision,
            "recall": df["recall"],
            "review_cost": df["review_cost_total"],
            "missed_fraud_loss": df["missed_fraud_loss"],
            "total_cost": df["total_cost"],
        }
    )
    out["is_chosen_t_star"] = np.isclose(out["threshold"], t_star)
    # nearest grid row to 0.5 is the naive default drawn on the curve
    naive_i = int((out["threshold"] - 0.5).abs().idxmin())
    out["is_naive_default"] = False
    out.loc[naive_i, "is_naive_default"] = True
    return out


def recommendation(results: dict) -> list[str]:
    """Plain-language operating recommendation (ASCII, no wall-clock/RNG)."""
    t = results["thresholds"]
    costs = t["costs"]
    star = t["test_at_star"]
    naive = t["test_at_naive_05"]
    none = t["test_review_none"]
    t_star = float(t["threshold_star"])
    prec = star["n_fraud_caught"] / star["n_flagged"] if star["n_flagged"] else float("nan")
    return [
        f"RECOMMENDATION: operate the alert at t* = {t_star:.3f} on the calibrated "
        "fraud probability.",
        f"  Why: it minimises expected cost on validation and, re-checked on the held-out "
        f"test window, costs ${star['total_cost']:,.0f} vs ${naive['total_cost']:,.0f} at "
        f"the naive 0.5 default ({t['test_cost_reduction_vs_05']:.1%} lower) and "
        f"${none['total_cost']:,.0f} if nothing is reviewed.",
        f"  What it means operationally: about {star['n_flagged']:,} alerts reach the queue "
        f"(precision {prec:.2f}, recall {star['recall']:.2f}) - {star['n_fraud_caught']} of "
        f"{star['n_fraud_caught'] + star['n_fraud_missed']} test-window frauds caught.",
        "  ASSUMPTIONS (not guarantees): " + " ".join(costs.describe()),
        "  Method: threshold chosen on VALIDATION, judged once on TEST; synthetic data.",
    ]


def save_cost_curve_csv(results: dict, path: str) -> str:
    """Write the per-threshold decision table to CSV. Returns the path."""
    frame = cost_curve_frame(results)
    frame.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return path


def _nice_ceiling(value: float) -> float:
    """Round a positive number up to 1/2/5 x 10^k for a clean axis top."""
    if value <= 0:
        return 1.0
    exp = np.floor(np.log10(value))
    base = 10.0**exp
    for mult in (1.0, 2.0, 5.0, 10.0):
        if value <= mult * base:
            return mult * base
    return 10.0 * base


def save_cost_curve_svg(results: dict, path: str) -> str:
    """Hand-draw the cost-vs-threshold curve as a standalone SVG.

    Pure string construction - no matplotlib, no RNG, no timestamps - so the
    committed figure is byte-deterministic across machines. Log x-axis (the
    thresholds span several decades), linear cost axis anchored at $0.
    """
    frame = cost_curve_frame(results)
    t = results["thresholds"]
    t_star = float(t["threshold_star"])
    # positive thresholds only (log axis); drop the >1 sentinel endpoint
    m = (frame["threshold"] > 0) & (frame["threshold"] <= 1.0)
    thr = frame.loc[m, "threshold"].to_numpy(dtype=np.float64)
    cost = frame.loc[m, "total_cost"].to_numpy(dtype=np.float64)

    W, H = 760, 460
    left, right, top, bottom = 84, 28, 56, 64
    plot_w = W - left - right
    plot_h = H - top - bottom

    lx = np.log10(thr)
    x_lo, x_hi = np.floor(lx.min()), 0.0  # decades from min..10^0
    y_hi = _nice_ceiling(float(cost.max()))

    def px(logt: float) -> float:
        return left + (logt - x_lo) / (x_hi - x_lo) * plot_w

    def py(c: float) -> float:
        return top + plot_h - (c / y_hi) * plot_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif" '
        f'role="img" aria-label="Expected cost versus alert threshold">'
    )
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="{_SURFACE}"/>')

    # y gridlines + labels ($)
    n_y = 4
    for k in range(n_y + 1):
        c = y_hi * k / n_y
        y = py(c)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" '
            f'fill="{_MUTED}">${c:,.0f}</text>'
        )

    # x gridlines + labels (decade ticks)
    dec = int(x_lo)
    while dec <= int(x_hi):
        x = px(float(dec))
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
            f'stroke="{_GRID}" stroke-width="1"/>'
        )
        label = f"{10.0**dec:g}"
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" '
            f'font-size="12" fill="{_MUTED}">{label}</text>'
        )
        dec += 1

    # axis lines
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        f'stroke="{_AXIS}" stroke-width="1.2"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        f'stroke="{_AXIS}" stroke-width="1.2"/>'
    )

    # the cost curve (single 2px series)
    pts = " ".join(f"{px(lx[i]):.1f},{py(cost[i]):.1f}" for i in range(len(thr)))
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="{_SERIES}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    # marker: chosen t*
    xs = px(float(np.log10(t_star)))
    ys = py(float(t["val_at_star"]["total_cost"]))
    parts.append(
        f'<line x1="{xs:.1f}" y1="{top}" x2="{xs:.1f}" y2="{top + plot_h}" '
        f'stroke="{_INK}" stroke-width="1.2" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<circle cx="{xs:.1f}" cy="{ys:.1f}" r="4.5" fill="{_CHOSEN}" '
        f'stroke="{_SURFACE}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{xs + 8:.1f}" y="{ys - 10:.1f}" font-size="12.5" fill="{_INK}" '
        f'font-weight="600">t* = {t_star:.3f}</text>'
    )
    parts.append(
        f'<text x="{xs + 8:.1f}" y="{ys + 6:.1f}" font-size="11.5" fill="{_INK_2}">'
        f'val cost ${t["val_at_star"]["total_cost"]:,.0f}</text>'
    )

    # marker: naive 0.5 default
    xn = px(float(np.log10(0.5)))
    naive_row = frame.iloc[(frame["threshold"] - 0.5).abs().argmin()]
    yn = py(float(naive_row["total_cost"]))
    parts.append(
        f'<line x1="{xn:.1f}" y1="{top}" x2="{xn:.1f}" y2="{top + plot_h}" '
        f'stroke="{_MUTED}" stroke-width="1.2" stroke-dasharray="5 4"/>'
    )
    parts.append(
        f'<circle cx="{xn:.1f}" cy="{yn:.1f}" r="4" fill="{_MUTED}" '
        f'stroke="{_SURFACE}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{xn - 8:.1f}" y="{yn - 8:.1f}" font-size="11.5" fill="{_INK_2}" '
        f'text-anchor="end">naive 0.5</text>'
    )

    # titles / axis labels (ink tokens, not the series colour)
    parts.append(
        f'<text x="{left}" y="26" font-size="15" fill="{_INK}" font-weight="700">'
        f'Expected cost vs alert threshold (validation sweep)</text>'
    )
    parts.append(
        f'<text x="{left}" y="44" font-size="12" fill="{_INK_2}">'
        f'Cost-minimising t* = {t_star:.3f}; assumptions: ${t["costs"].review_cost:.0f}/review, '
        f'missed fraud = its amount</text>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{H - 22}" text-anchor="middle" '
        f'font-size="12.5" fill="{_INK_2}">Alert threshold on calibrated fraud '
        f'probability (log scale)</text>'
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="12.5" '
        f'fill="{_INK_2}" transform="rotate(-90 18 {top + plot_h / 2:.1f})">'
        f'Total expected cost ($, illustrative assumptions)</text>'
    )
    parts.append("</svg>\n")
    svg = "\n".join(parts)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg)
    return path
