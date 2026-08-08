"""Champion/challenger harness tests: split discipline (identical test window,
no leakage), hand-checked swap-set math, conservation and base-case collapse,
gate logic driven as a pure function, cross-module consistency, determinism of
a full re-run, and the CLI."""

from __future__ import annotations

import os

import numpy as np
import pytest

from fdo.challenger import (
    SWAP_CELLS,
    ChallengerConfig,
    challenger_lines,
    challenger_split,
    gate_verdict,
    promotion_gates,
    run_challenger,
    save_challenger_csv,
    save_swap_csv,
    swap_set,
)
from fdo.model import time_split

CELL_BOTH, CELL_OUT, CELL_IN, CELL_NEITHER = SWAP_CELLS


@pytest.fixture(scope="module")
def analysis(results) -> dict:
    return run_challenger(results)


def test_split_preserves_test_window_and_time_order():
    n = 60_000
    cfg = ChallengerConfig()
    tr, va, te = challenger_split(n, cfg)
    _, _, te_champ = time_split(n)
    # judged on the IDENTICAL test rows the champion pipeline uses
    assert np.array_equal(te, te_champ)
    # contiguous, time-ordered, leakage-free
    assert tr.max() < va.min() <= va.max() < te.min()
    assert tr.min() == int(n * cfg.drop_frac)
    assert va.size == int(n * cfg.val_frac)
    assert np.intersect1d(tr, te).size == 0 and np.intersect1d(va, te).size == 0


def test_split_rejects_degenerate_configs():
    with pytest.raises(ValueError):
        challenger_split(60_000, ChallengerConfig(drop_frac=0.9))  # train window empty
    with pytest.raises(ValueError):
        challenger_split(60_000, ChallengerConfig(val_frac=0.0))
    with pytest.raises(ValueError):
        challenger_split(60_000, ChallengerConfig(drop_frac=-0.1))


def test_swap_set_hand_checked():
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    amounts = np.array([100.0, 50.0, 200.0, 25.0, 10.0, 5.0])
    flag_c = np.array([True, True, False, False, True, False])
    flag_h = np.array([True, False, True, False, False, False])
    sw = swap_set(y, amounts, flag_c, flag_h).set_index("cell")
    # cell by cell, checkable by hand
    assert int(sw.loc[CELL_BOTH, "n"]) == 1
    assert int(sw.loc[CELL_BOTH, "n_fraud"]) == 1
    assert sw.loc[CELL_BOTH, "fraud_value"] == 100.0
    assert sw.loc[CELL_BOTH, "fraud_rate"] == 1.0
    assert int(sw.loc[CELL_OUT, "n"]) == 2  # rows 1 and 4
    assert int(sw.loc[CELL_OUT, "n_fraud"]) == 1
    assert sw.loc[CELL_OUT, "fraud_value"] == 10.0
    assert sw.loc[CELL_OUT, "fraud_rate"] == 0.5
    assert int(sw.loc[CELL_IN, "n"]) == 1  # row 2
    assert int(sw.loc[CELL_IN, "n_fraud"]) == 1
    assert sw.loc[CELL_IN, "fraud_value"] == 200.0
    assert int(sw.loc[CELL_NEITHER, "n"]) == 2
    assert int(sw.loc[CELL_NEITHER, "n_fraud"]) == 0
    with pytest.raises(ValueError):
        swap_set(y[:3], amounts, flag_c, flag_h)  # shape mismatch


def test_swap_set_collapses_when_policies_agree():
    """Base case: identical decisions -> both swap cells are exactly empty."""
    rng = np.random.default_rng(5)
    y = (rng.random(500) < 0.1).astype(np.float64)
    amounts = np.round(rng.lognormal(3.0, 1.0, 500), 2)
    flags = rng.random(500) < 0.2
    sw = swap_set(y, amounts, flags, flags).set_index("cell")
    assert int(sw.loc[CELL_OUT, "n"]) == 0 and int(sw.loc[CELL_IN, "n"]) == 0
    assert np.isnan(sw.loc[CELL_OUT, "fraud_rate"])  # empty cell -> undefined rate
    assert int(sw.loc[CELL_BOTH, "n"]) == int(flags.sum())
    assert int(sw.loc[CELL_NEITHER, "n"]) == int((~flags).sum())


def test_swap_set_partitions_the_test_window(results, analysis):
    sw = analysis["swap"].set_index("cell")
    te = results["splits"]["test"]
    df = results["df"]
    y_te = df["is_fraud"].to_numpy(dtype=np.float64)[te]
    amounts_te = df["amount"].to_numpy(dtype=np.float64)[te]
    # the four cells are a partition of the window: counts, frauds, and value conserve
    assert int(sw["n"].sum()) == int(te.size)
    assert int(sw["n_fraud"].sum()) == int(y_te.sum())
    assert np.isclose(sw["fraud_value"].sum(), amounts_te[y_te == 1].sum())
    # marginal totals reproduce each model's alert volume
    m = analysis["measured"]
    assert int(sw.loc[CELL_BOTH, "n"] + sw.loc[CELL_OUT, "n"]) == m["alerts_champion"]
    assert int(sw.loc[CELL_BOTH, "n"] + sw.loc[CELL_IN, "n"]) == m["alerts_challenger"]


def _measured(**over) -> dict:
    base = {
        "pr_auc_champion": 0.270,
        "pr_auc_challenger": 0.270,
        "ece_challenger": 0.003,
        "cost_champion": 8841.0,
        "cost_challenger": 8841.0,
        "alerts_champion": 608,
        "alerts_challenger": 608,
        "zero_cov_champion": 1,
        "zero_cov_challenger": 1,
    }
    base.update(over)
    return base


def test_gates_pass_when_challenger_equals_champion():
    """Base case: a challenger identical to the champion clears every
    non-worse gate (equality is non-inferior)."""
    gates = promotion_gates(_measured(), ChallengerConfig())
    assert len(gates) == 5
    assert all(g["passed"] for g in gates)
    assert gate_verdict(gates) == "PROMOTE"
    for g in gates:
        assert set(g) == {"gate", "requirement", "measured", "passed"}


def test_each_gate_fails_on_exactly_its_own_violation():
    cfg = ChallengerConfig()
    cases = {
        "ranking_non_inferior": _measured(pr_auc_challenger=0.270 - 0.011),
        "calibration_sane": _measured(ece_challenger=0.021),
        "economics_non_worse": _measured(cost_challenger=8842.0),
        "alert_volume_bounded": _measured(alerts_challenger=913),  # 1.5 * 608 = 912
        "segment_coverage_safe": _measured(zero_cov_challenger=2),
    }
    for gate_name, m in cases.items():
        gates = promotion_gates(m, cfg)
        by_name = {g["gate"]: g["passed"] for g in gates}
        assert by_name[gate_name] is False, gate_name
        others = [p for name, p in by_name.items() if name != gate_name]
        assert all(others), gate_name  # one violation trips exactly one gate
        assert gate_verdict(gates) == "HOLD"


def test_gate_boundaries_are_inclusive():
    # exact-binary floats so <=/>= boundary semantics are testable without FP noise
    cfg = ChallengerConfig(pr_auc_epsilon=0.25, ece_max=0.020)
    m = _measured(
        pr_auc_champion=0.75, pr_auc_challenger=0.5,  # exactly champion - epsilon
        ece_challenger=0.020,                          # exactly the bound
        alerts_challenger=912,                         # exactly 1.5x champion
    )
    assert all(g["passed"] for g in promotion_gates(m, cfg))


def test_full_run_structure_and_cross_module_consistency(results, analysis):
    w = analysis["windows"]
    # the retrain window actually rolls forward: starts later AND ends later
    assert w["challenger_train"][0] > w["champion_train"][0]
    assert w["challenger_train"][1] > w["champion_train"][1]
    assert w["challenger_val"][1] < w["test"][0]
    # the challenger is a real model, not noise: well above the prevalence floor
    th = analysis["test_challenger"]
    assert th["pr_auc"] > 5 * th["prevalence"]
    # each model's threshold block is a genuine sweep result
    for key in ("thresholds_champion", "thresholds_challenger"):
        assert analysis[key]["threshold_star"] > 0.0
        assert analysis[key]["test_at_star"]["n_flagged"] > 0
    # fixed-capacity view must agree with queue_opt's independent top-K-by-EV
    # computation for the champion (same rule, same scores, same window)
    q = results["queue"]
    assert np.isclose(
        analysis["capacity_champion"]["expected_recovered"], q["topk_ev_expected"]
    )
    comp = q["comparison"].set_index("strategy")
    assert np.isclose(
        analysis["capacity_champion"]["realized_recovered"],
        comp.loc["top_k_by_expected_value (== unconstrained LP)", "realized_recovered"],
    )
    # verdict is exactly what the gates say
    assert analysis["verdict"] == gate_verdict(analysis["gates"])
    assert analysis["verdict"] in ("PROMOTE", "HOLD")
    # report is ASCII and states the honesty framing
    text = "\n".join(challenger_lines(analysis))
    assert text.isascii()
    assert "synthetic" in text and "policy knobs" in text
    assert "VERDICT" in text


def test_full_rerun_is_deterministic(results, analysis, tmp_path):
    """Retraining is zero-init full-batch gradient descent with no RNG: a
    second run must reproduce every number and byte."""
    again = run_challenger(results)
    assert again["test_challenger"]["pr_auc"] == analysis["test_challenger"]["pr_auc"]
    assert (
        again["thresholds_challenger"]["threshold_star"]
        == analysis["thresholds_challenger"]["threshold_star"]
    )
    assert challenger_lines(again) == challenger_lines(analysis)
    a = save_challenger_csv(analysis, str(tmp_path / "a.csv"))
    b = save_challenger_csv(again, str(tmp_path / "b.csv"))
    assert open(a, "rb").read() == open(b, "rb").read()
    sa = save_swap_csv(analysis, str(tmp_path / "sa.csv"))
    sb = save_swap_csv(again, str(tmp_path / "sb.csv"))
    assert open(sa, "rb").read() == open(sb, "rb").read()


def test_cli_challenger_runs_and_writes_csvs(tmp_path, capsys):
    from fdo.__main__ import main

    figdir = str(tmp_path / "figs")
    assert main(["--challenger", "--figdir", figdir]) == 0
    out = capsys.readouterr().out
    assert "CHAMPION/CHALLENGER" in out and "VERDICT" in out
    assert "swap-out" in out and "swap-in" in out
    assert out.isascii()
    cmp_path = os.path.join(figdir, "champion_challenger.csv")
    swap_path = os.path.join(figdir, "challenger_swap.csv")
    for path in (cmp_path, swap_path):
        assert os.path.exists(path) and os.path.getsize(path) > 0
    body = open(cmp_path, encoding="utf-8").read()
    assert "champion" in body and "challenger" in body and "verdict" in body
