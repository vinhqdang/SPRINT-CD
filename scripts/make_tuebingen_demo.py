"""A worked example from the Tuebingen benchmark, chosen for concreteness.

Section 4.11 reports aggregate coverage and accuracy over 102 pairs. This
script draws one specific pair -- Tuebingen pair 1, altitude versus mean
temperature across weather stations, ground truth altitude -> temperature --
so a reader can see an actual scatter of real data next to what the
certificate concluded, rather than only the aggregate numbers.  This is the
pair the direction certificate got wrong (it certifies temperature ->
altitude here), which is the honest example to show: Section 4.11's finding
is that the certificate is barely better than chance on the pairs where it
commits, and this is a concrete instance of that, not a flattering outlier.

Reads the cached data/tuebingen_cep/pair0001.txt (fetched by
experiments/exp12_tuebingen_cep.py) and reruns only the certificate on this
one pair; does not repeat the full 102-pair sweep.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from common import PALETTE, setup_matplotlib  # noqa: E402
from exp12_tuebingen_cep import certify_direction  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "tuebingen_cep" / "pair0001.txt"
OUT = pathlib.Path(__file__).resolve().parent.parent / "results" / "tuebingen_demo_pair0001.png"


def main():
    arr = np.loadtxt(DATA)
    x, y = arr[:, 0], arr[:, 1]  # x = altitude (m), y = mean temperature (degC)

    verdict = certify_direction(x, y, alpha=0.1, batch=50, min_fit=60)
    verdict_label = {
        "x->y": "altitude $\\to$ temperature (correct)",
        "y->x": "temperature $\\to$ altitude (wrong)",
        None: "undecided",
    }[verdict]
    label_color = {"x->y": PALETTE["oracle"], "y->x": PALETTE["naive"],
                   None: PALETTE["fixed"]}[verdict]

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.scatter(x, y, s=10, alpha=0.5, color=PALETTE["fixed"])
    ax.set_xlabel("altitude (m)")
    ax.set_ylabel("mean temperature ($^\\circ$C)")
    ax.set_title(f"Tuebingen pair 1, $n={x.size}$")
    ax.text(0.03, 0.06,
            f"ground truth: altitude $\\to$ temperature\ncertificate: {verdict_label}",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            color=label_color,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"verdict: {verdict}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
