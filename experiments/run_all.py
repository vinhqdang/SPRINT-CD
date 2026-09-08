"""Run every experiment and collect the results.

``python experiments/run_all.py --quick`` gives a fast smoke run; the default
reproduces the figures and tables reported in ``results/``.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

SCRIPTS = [
    ("exp1_type1_calibration.py", "Continuous monitoring of a null CI query"),
    ("exp2_graph_fwer.py", "Whole-graph error over a monitored run"),
    ("exp3_sample_efficiency.py", "Adaptive stopping and the equivalence region"),
    ("exp4_latent_confounders.py", "Recovery under latent confounding"),
    ("exp5_eicp.py", "Invariant prediction under optional continuation"),
    ("exp6_certcd.py", "Certificate-only discovery vs delete-on-non-rejection"),
    ("exp7_orientation_vs_lingam.py", "Certified orientation against DirectLiNGAM"),
    ("exp8_primitive_calibration.py", "The per-set primitive under six noise laws"),
    ("exp9_assumption_violations.py", "Where the certificates are void"),
    ("exp10_sachs.py", "Real data: Sachs et al. (2005) protein signalling"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fast, low-precision run; writes to results/quick/")
    ap.add_argument("--only", type=str, default=None, help="substring filter on script name")
    args = ap.parse_args()

    failures = []
    for script, title in SCRIPTS:
        if args.only and args.only not in script:
            continue
        print("\n" + "=" * 78)
        print(f"  {title}")
        print("=" * 78)
        cmd = [sys.executable, str(HERE / script)]
        if args.quick:
            cmd.append("--quick")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=HERE)
        if proc.returncode != 0:
            failures.append(script)
            print(f"  !! {script} exited with code {proc.returncode}")
        print(f"  ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 78)
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    where = "results/quick/" if args.quick else "results/"
    print(f"All experiments completed.  Figures and JSON are in {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
