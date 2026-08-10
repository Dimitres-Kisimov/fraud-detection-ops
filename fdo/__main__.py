"""CLI entry point.

    python -m fdo                  print the measured summary
    python -m fdo --deliverables   also write deliverables/ (PDF + Excel)
    python -m fdo --drift          drift monitor: training window vs final
                                   test window (PSI) + figures/drift_psi.png
    python -m fdo --decision       cost-curve recommendation + queue-fairness
                                   read-out; writes figures/cost_curve.svg,
                                   figures/cost_curve.csv, figures/queue_fairness.csv
    python -m fdo --challenger     champion/challenger retrain harness: swap-set
                                   analysis + gated PROMOTE/HOLD verdict; writes
                                   figures/champion_challenger.csv + challenger_swap.csv
    python -m fdo --feedback       analyst feedback-loop simulation (selective-labels
                                   problem): four labelling-policy arms retrained per
                                   round; writes figures/feedback_loop.csv

Console output is ASCII-only with UTF-8 reconfiguration guarded for older
Windows consoles.
"""

from __future__ import annotations

import argparse
import sys
import time


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(prog="fdo", description=__doc__)
    parser.add_argument("--deliverables", action="store_true",
                        help="write the executive PDF and Excel workbook")
    parser.add_argument("--drift", action="store_true",
                        help="run the PSI drift monitor (train window vs final test window) "
                             "and write the drift figure")
    parser.add_argument("--decision", action="store_true",
                        help="print the cost-curve recommendation and per-segment queue-fairness "
                             "read-out, and write the cost-curve SVG/CSV + fairness CSV")
    parser.add_argument("--challenger", action="store_true",
                        help="run the champion/challenger retrain harness (swap-set analysis, "
                             "promotion gates) and write the comparison + swap-set CSVs")
    parser.add_argument("--feedback", action="store_true",
                        help="run the analyst feedback-loop simulation (selective-labels "
                             "problem, four labelling-policy arms) and write the trajectory CSV")
    parser.add_argument("--out", default="deliverables", help="output directory")
    parser.add_argument("--figdir", default="figures", help="figure output directory (--drift)")
    parser.add_argument("--seed", type=int, default=7, help="generator seed")
    args = parser.parse_args(argv)

    from fdo.pipeline import headline, run_pipeline

    t0 = time.time()
    print("[INFO] running pipeline (synthetic seeded data, from-scratch NumPy) ...")
    results = run_pipeline(seed=args.seed)
    print(f"[OK] pipeline finished in {time.time() - t0:.1f}s")
    for line in headline(results):
        print("[RESULT] " + line)

    if args.drift:
        import os

        from fdo.drift import drift_from_results, drift_lines, save_drift_figure

        analysis = drift_from_results(results)
        print()
        for line in drift_lines(analysis):
            print(line)
        fig_path = os.path.join(args.figdir, "drift_psi.png")
        size = save_drift_figure(analysis, fig_path)
        print()
        print(f"[OK] wrote {fig_path} ({size:,} bytes)")

    if args.decision:
        import os

        from fdo.cost_curve import (
            recommendation,
            save_cost_curve_csv,
            save_cost_curve_svg,
        )
        from fdo.fairness import fairness_lines, save_fairness_csv

        print()
        for line in recommendation(results):
            print(line)
        print()
        for line in fairness_lines(results):
            print(line)
        os.makedirs(args.figdir, exist_ok=True)
        svg_path = save_cost_curve_svg(results, os.path.join(args.figdir, "cost_curve.svg"))
        csv_path = save_cost_curve_csv(results, os.path.join(args.figdir, "cost_curve.csv"))
        fair_path = save_fairness_csv(results, os.path.join(args.figdir, "queue_fairness.csv"))
        print()
        for path in (svg_path, csv_path, fair_path):
            print(f"[OK] wrote {path} ({os.path.getsize(path):,} bytes)")

    if args.challenger:
        import os

        from fdo.challenger import (
            challenger_lines,
            run_challenger,
            save_challenger_csv,
            save_swap_csv,
        )

        print()
        print("[INFO] training the challenger (rolling window; same family/loss/hyperparameters) ...")
        analysis = run_challenger(results)
        for line in challenger_lines(analysis):
            print(line)
        os.makedirs(args.figdir, exist_ok=True)
        cmp_path = save_challenger_csv(analysis, os.path.join(args.figdir, "champion_challenger.csv"))
        swap_path = save_swap_csv(analysis, os.path.join(args.figdir, "challenger_swap.csv"))
        print()
        for path in (cmp_path, swap_path):
            print(f"[OK] wrote {path} ({os.path.getsize(path):,} bytes)")

    if args.feedback:
        import os

        from fdo.feedback import feedback_lines, run_feedback, save_feedback_csv

        print()
        print("[INFO] running the feedback-loop simulation (4 labelling-policy arms, "
              "retrained per round) ...")
        analysis = run_feedback(results)
        for line in feedback_lines(analysis):
            print(line)
        os.makedirs(args.figdir, exist_ok=True)
        csv_path = save_feedback_csv(analysis, os.path.join(args.figdir, "feedback_loop.csv"))
        print()
        print(f"[OK] wrote {csv_path} ({os.path.getsize(csv_path):,} bytes)")

    if args.deliverables:
        from fdo.exports import build_deliverables

        sizes = build_deliverables(results, args.out)
        for path, size in sizes.items():
            print(f"[OK] wrote {path} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
