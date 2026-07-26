"""CLI entry point.

    python -m fdo                  print the measured summary
    python -m fdo --deliverables   also write deliverables/ (PDF + Excel)
    python -m fdo --drift          drift monitor: training window vs final
                                   test window (PSI) + figures/drift_psi.png

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

    if args.deliverables:
        from fdo.exports import build_deliverables

        sizes = build_deliverables(results, args.out)
        for path, size in sizes.items():
            print(f"[OK] wrote {path} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
