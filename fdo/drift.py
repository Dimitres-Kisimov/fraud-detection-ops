"""Population Stability Index (PSI) drift monitor, from scratch.

PSI compares a "recent" sample against an "expected" (training) sample,
bin by bin:

    PSI = sum_b (a_b - e_b) * ln(a_b / e_b)

where e_b and a_b are the expected and actual proportion of observations in
bin b. Bins are deciles (quantiles) of the EXPECTED sample, the standard
scorecard-monitoring formulation, so each bin holds ~10% of the training
data by construction and the statistic asks "where did the recent data move?".

Conventional thresholds (an industry rule of thumb from credit-scorecard
practice, NOT a statistical law - the sampling distribution depends on n and
the number of bins):

    PSI < 0.10        stable - no action
    0.10 <= PSI <= 0.25  moderate shift - investigate
    PSI > 0.25        major shift - model review / retrain trigger

Because the rule of thumb ignores sample size, this module also computes the
finite-sample noise floor of PSI under "no change": asymptotically
PSI ~ (1/n_e + 1/n_a) * chi-squared with (B-1) degrees of freedom
(Yurdakul, 2018), which matters when monitoring small sub-populations such
as confirmed-fraud rows.

HONESTY NOTE on what PSI can and cannot see: PSI on inputs and scores
detects COVARIATE drift (the feature mix changing). It is blind to pure
CONCEPT drift, where P(fraud | x) changes while the feature mix stays put -
exactly the kind of drift this project's synthetic generator injects after
day 60 (fdo/data.py shifts the latent risk of gift-card / electronics /
night patterns, not the feature draws themselves). The label-aware monitor
in ``fraud_mix_shift`` exists for that reason: with (delayed) confirmed
labels, the composition of the fraud population itself is monitorable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from fdo.data import make_features

STABLE_MAX = 0.10
MODERATE_MAX = 0.25
_PROP_FLOOR = 1e-4  # smoothing floor so empty bins never produce log(0)

SCORE_ROW = "model_score"


def verdict(value: float) -> str:
    """Label a PSI value with the conventional rule-of-thumb band."""
    if value < STABLE_MAX:
        return "stable (<0.10)"
    if value <= MODERATE_MAX:
        return "moderate (0.10-0.25)"
    return "major (>0.25)"


def _psi_from_counts(cnt_e: np.ndarray, cnt_a: np.ndarray) -> float:
    """PSI from aligned bin counts, with proportion smoothing.

    Proportions are floored at 1e-4 (a common convention) so an empty bin
    contributes a large-but-finite term instead of +inf. Identical count
    vectors give exactly 0.0.
    """
    e = cnt_e / max(cnt_e.sum(), 1)
    a = cnt_a / max(cnt_a.sum(), 1)
    e = np.maximum(e, _PROP_FLOOR)
    a = np.maximum(a, _PROP_FLOOR)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Quantile-binned PSI of ``actual`` against ``expected``.

    - Bin edges are the ``bins``-quantiles of EXPECTED, deduplicated (heavy
      ties collapse bins), with open outer edges so recent values outside
      the training range still land in the extreme bins.
    - If EXPECTED has <= ``bins`` distinct values the variable is treated as
      categorical: one bin per distinct value over the union of both
      samples. Without this, a rare binary flag (e.g. device_change at 5%)
      would collapse to a single all-covering bin and every shift in it
      would be invisible.
    """
    expected = np.asarray(expected, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    if expected.size == 0 or actual.size == 0:
        raise ValueError("psi: expected and actual must be non-empty")

    uniq = np.unique(expected)
    if uniq.size <= bins:  # categorical / low-cardinality path
        values = np.unique(np.concatenate([uniq, np.unique(actual)]))
        cnt_e = np.array([(expected == v).sum() for v in values], dtype=np.float64)
        cnt_a = np.array([(actual == v).sum() for v in values], dtype=np.float64)
        return _psi_from_counts(cnt_e, cnt_a)

    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(expected, qs))
    interior = edges[1:-1]  # outer edges replaced by (-inf, +inf)
    cnt_e = np.bincount(np.searchsorted(interior, expected, side="right"),
                        minlength=interior.size + 1).astype(np.float64)
    cnt_a = np.bincount(np.searchsorted(interior, actual, side="right"),
                        minlength=interior.size + 1).astype(np.float64)
    return _psi_from_counts(cnt_e, cnt_a)


def psi_categorical(expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray) -> float:
    """PSI over the value proportions of a categorical variable (one bin per
    distinct value, union of both samples)."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    values = np.unique(np.concatenate([np.unique(expected), np.unique(actual)]))
    cnt_e = np.array([(expected == v).sum() for v in values], dtype=np.float64)
    cnt_a = np.array([(actual == v).sum() for v in values], dtype=np.float64)
    return _psi_from_counts(cnt_e, cnt_a)


def psi_noise_floor(n_expected: int, n_actual: int, n_bins: int) -> dict[str, float]:
    """Finite-sample behaviour of PSI when NOTHING has changed.

    Asymptotically PSI ~ (1/n_e + 1/n_a) * chi2_{B-1} (Yurdakul, 2018), so
    even two samples from the same distribution give a positive PSI. Returns
    the expected value and the 95th percentile under "no change"; a measured
    PSI below the 95th percentile is indistinguishable from sampling noise.
    """
    scale = 1.0 / n_expected + 1.0 / n_actual
    dof = max(n_bins - 1, 1)
    return {
        "expected_if_no_change": scale * dof,
        "p95_if_no_change": scale * float(chi2.ppf(0.95, dof)),
    }


def drift_report(
    train_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    features: list[str],
    scores_train: np.ndarray | None = None,
    scores_recent: np.ndarray | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI of ``recent_df`` against ``train_df``.

    ``features`` are names from ``make_features`` (model input columns).
    If both score arrays are given, a ``model_score`` row is appended - PSI
    on the model OUTPUT distribution, the single most-watched production
    drift metric because it summarizes every input shift the model responds
    to. Returns a DataFrame with columns: feature, psi, verdict.
    """
    X_tr, names = make_features(train_df)
    X_re, _ = make_features(recent_df)
    unknown = [f for f in features if f not in names]
    if unknown:
        raise ValueError(f"unknown feature(s): {unknown}; available: {names}")

    rows = [
        {"feature": f,
         "psi": psi(X_tr[:, names.index(f)], X_re[:, names.index(f)], bins=bins)}
        for f in features
    ]
    if scores_train is not None and scores_recent is not None:
        rows.append({"feature": SCORE_ROW,
                     "psi": psi(np.asarray(scores_train), np.asarray(scores_recent), bins=bins)})
    out = pd.DataFrame(rows)
    out["verdict"] = out["psi"].map(verdict)
    return out


def fraud_mix_shift(train_df: pd.DataFrame, recent_df: pd.DataFrame) -> dict:
    """Label-aware monitor: how the CONFIRMED-FRAUD population is composed.

    The generator's day-60 drift changes which patterns produce fraud, not
    the overall feature mix - so it is invisible to input PSI but visible
    here. In production this is the monitor that runs on delayed confirmed
    labels (chargebacks land weeks later).

    Returns the merchant-category mix among fraud rows (train vs recent),
    its categorical PSI, the same for the night-time share, and the
    finite-sample noise floor for the small fraud counts.
    """
    fr_tr = train_df[train_df["is_fraud"] == 1]
    fr_re = recent_df[recent_df["is_fraud"] == 1]
    n_tr, n_re = len(fr_tr), len(fr_re)
    if n_tr == 0 or n_re == 0:
        raise ValueError("fraud_mix_shift: no fraud rows in one of the windows")

    cats = sorted(set(train_df["merchant_cat"]) | set(recent_df["merchant_cat"]))
    mix = pd.DataFrame({
        "category": cats,
        "train_share": [float((fr_tr["merchant_cat"] == c).mean()) for c in cats],
        "recent_share": [float((fr_re["merchant_cat"] == c).mean()) for c in cats],
    })
    mix["delta_pp"] = (mix["recent_share"] - mix["train_share"]) * 100.0

    cat_psi = psi_categorical(fr_tr["merchant_cat"], fr_re["merchant_cat"])
    night_tr = (fr_tr["hour"] < 6).to_numpy().astype(np.float64)
    night_re = (fr_re["hour"] < 6).to_numpy().astype(np.float64)
    night_psi = psi(night_tr, night_re)

    return {
        "n_fraud_train": n_tr,
        "n_fraud_recent": n_re,
        "mix": mix,
        "category_psi": cat_psi,
        "category_noise": psi_noise_floor(n_tr, n_re, len(cats)),
        "night_share_train": float(night_tr.mean()),
        "night_share_recent": float(night_re.mean()),
        "night_psi": night_psi,
        "night_noise": psi_noise_floor(n_tr, n_re, 2),
    }


def draw_drift_chart(analysis: dict, ax) -> None:
    """Draw the PSI bar chart onto an existing axes: covariate monitors
    (inputs + model score) next to the label-aware fraud-population
    monitors, with the 0.10 / 0.25 rule-of-thumb lines. Shared by the
    standalone PNG and the executive-PDF drift page."""
    from matplotlib.patches import Rectangle

    from fdo.exports import _style_axes
    from fdo.palette import (
        ALERT_AMBER,
        CASE_GRAPHITE,
        CLEARED_GREEN,
        FRAUD_RED,
        GRID,
        INK,
        INK_2,
        MUTED,
        SURFACE,
    )

    rep = analysis["report"]
    fm = analysis["fraud_mix"]
    labels = list(rep["feature"]) + ["fraud mix: merchant category", "fraud mix: night share"]
    values = list(rep["psi"]) + [fm["category_psi"], fm["night_psi"]]
    # flat covariate monitors stay in paper-trail graphite; the label-aware
    # pair - the channel that actually sees this drift - carries the colour.
    colors = [CASE_GRAPHITE] * len(rep) + [CLEARED_GREEN, CLEARED_GREEN]

    _style_axes(ax)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=colors, edgecolor=SURFACE, linewidth=1.0)
    for yi, v in zip(y, values, strict=True):
        ax.annotate(f"{v:.3f}", (v, yi), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8.5, color=INK_2)
    ax.axvline(STABLE_MAX, color=ALERT_AMBER, linewidth=1.2, linestyle="--")
    ax.axvline(MODERATE_MAX, color=FRAUD_RED, linewidth=1.2, linestyle="--")
    top = y.max() + 0.45
    ax.annotate("0.10 investigate", (STABLE_MAX, top), fontsize=9, color=MUTED,
                textcoords="offset points", xytext=(4, 0))
    ax.annotate("0.25 major shift", (MODERATE_MAX, top), fontsize=9, color=INK_2,
                textcoords="offset points", xytext=(4, 0))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK_2)
    ax.set_xlabel("PSI, training window vs final test window (thresholds are rule-of-thumb)")
    ax.set_xlim(0, max(MODERATE_MAX * 1.3, max(values) * 1.25))
    ax.set_ylim(-0.6, top + 0.6)
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Drift monitor: inputs and model score are flat - the injected day-60 drift is\n"
        "concept drift, visible only in the label-aware fraud-population mix (green)",
        fontsize=11, color=INK,
    )
    handles = [Rectangle((0, 0), 1, 1, color=CASE_GRAPHITE),
               Rectangle((0, 0), 1, 1, color=CLEARED_GREEN)]
    leg = ax.legend(handles, ["covariate PSI (inputs + score)",
                              "label-aware PSI (confirmed-fraud rows)"],
                    loc="lower right", frameon=True, fontsize=9)
    leg.get_frame().set_edgecolor(GRID)


def save_drift_figure(analysis: dict, path: str) -> int:
    """Write the PSI bar chart as a standalone PNG. Returns the byte size."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fdo.palette import SURFACE

    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    fig.patch.set_facecolor(SURFACE)
    draw_drift_chart(analysis, ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return os.path.getsize(path)


def drift_from_results(results: dict, bins: int = 10) -> dict:
    """Full drift analysis from one ``run_pipeline`` result: training window
    vs the final test window, on inputs, on the model score, and on the
    label-aware fraud mix."""
    df = results["df"]
    idx_tr = results["splits"]["train"]
    idx_te = results["splits"]["test"]
    train, test = df.iloc[idx_tr], df.iloc[idx_te]
    primary = results["models"][results["primary_name"]]
    _, names = make_features(train)
    report = drift_report(
        train, test, names,
        scores_train=primary.predict_calibrated(train),
        scores_recent=results["p_test"],
        bins=bins,
    )
    return {
        "report": report,
        "fraud_mix": fraud_mix_shift(train, test),
        "train_days": (float(train["ts_days"].min()), float(train["ts_days"].max())),
        "test_days": (float(test["ts_days"].min()), float(test["ts_days"].max())),
        "n_train": len(train),
        "n_test": len(test),
        "bins": bins,
    }


def drift_lines(analysis: dict) -> list[str]:
    """ASCII-only report for the CLI. States the honest finding explicitly:
    if no covariate PSI crosses 0.10, say so and explain why the label-aware
    monitor is the one that sees this generator's drift."""
    rep = analysis["report"]
    fm = analysis["fraud_mix"]
    d0, d1 = analysis["train_days"]
    t0, t1 = analysis["test_days"]
    lines = [
        f"Drift monitor: training window (days {d0:.0f}-{d1:.0f}, n={analysis['n_train']:,}) "
        f"vs final test window (days {t0:.0f}-{t1:.0f}, n={analysis['n_test']:,}).",
        f"PSI, quantile bins={analysis['bins']}; rule-of-thumb bands (industry convention, "
        "not a law): <0.10 stable, 0.10-0.25 moderate, >0.25 major.",
        "",
        f"  {'feature':<20} {'PSI':>8}  verdict",
        f"  {'-' * 20} {'-' * 8}  {'-' * 20}",
    ]
    for _, row in rep.iterrows():
        lines.append(f"  {row['feature']:<20} {row['psi']:>8.4f}  {row['verdict']}")

    crossed = rep[rep["psi"] >= STABLE_MAX]
    if crossed.empty:
        lines += [
            "",
            "HONEST FINDING: no input feature and not the model score crosses 0.10 - the",
            "covariate mix did NOT shift. That is correct, not a monitoring failure: the",
            "generator's day-60 drift changes P(fraud | x) (which patterns produce fraud),",
            "while the feature draws themselves are time-stationary. PSI on inputs/scores",
            "detects covariate drift only; pure concept drift is its known blind spot.",
            "The label-aware monitor below is the channel that sees it:",
        ]
    else:
        names = ", ".join(f"{r['feature']} ({r['psi']:.3f})" for _, r in crossed.iterrows())
        lines += ["", f"Covariate monitors above 0.10: {names}."]

    cn = fm["category_noise"]
    nn = fm["night_noise"]
    lines += [
        "",
        f"Label-aware monitor (confirmed-fraud rows only: {fm['n_fraud_train']} train, "
        f"{fm['n_fraud_recent']} test):",
        f"  fraud merchant-category mix PSI = {fm['category_psi']:.4f} "
        f"[{verdict(fm['category_psi'])}]",
        f"    (small-sample noise floor if nothing changed: mean {cn['expected_if_no_change']:.3f}, "
        f"95th pct {cn['p95_if_no_change']:.3f})",
        f"  fraud night-time share {fm['night_share_train']:.1%} -> "
        f"{fm['night_share_recent']:.1%}, PSI = {fm['night_psi']:.4f} "
        f"[{verdict(fm['night_psi'])}; 95th pct of noise {nn['p95_if_no_change']:.3f}]",
        "  largest composition moves (share of fraud, percentage points):",
    ]
    mix = fm["mix"].reindex(fm["mix"]["delta_pp"].abs().sort_values(ascending=False).index)
    for _, row in mix.head(3).iterrows():
        lines.append(f"    {row['category']:<15} {row['train_share']:.1%} -> "
                     f"{row['recent_share']:.1%} ({row['delta_pp']:+.1f} pp)")
    return lines
