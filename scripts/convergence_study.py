"""Convergence study for the integrators: error vs dt on a log-log plot.

Runs the reference problems in thaad_sim.verification (a dragged ballistic
threat, and a vertical interceptor launch through burnout) at dt, dt/2, dt/4
and dt/8 with each integrator, measures the final-position error against a
fine RK4 run, and plots it against slope-1 and slope-4 reference lines.

    pip install -e ".[plot]"
    python scripts/convergence_study.py            # writes docs/figures/convergence.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim import load_config  # noqa: E402
from thaad_sim.verification import (  # noqa: E402
    INTERCEPTOR_DTS, THREAT_DTS, interceptor_convergence, observed_orders, threat_convergence,
)

DEFAULT_OUT = ROOT / "docs" / "figures" / "convergence.png"
DEFAULT_CONFIG = ROOT / "configs" / "demo.yaml"

# Categorical slots 1 and 2 of the reference palette, with text and grid inks
COLORS = {"euler": "#2a78d6", "rk4": "#eb6834"}
LABELS = {"euler": "Forward Euler", "rk4": "RK4"}
SLOPES = {"euler": 1, "rk4": 4}
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
REF_OFFSET = 0.2


def _panel(ax, dts, errors, title):
    dts = np.asarray(dts)
    for method in ("euler", "rk4"):
        e = errors[method]
        order = observed_orders(e, dts)[-1]
        ax.loglog(dts, e, color=COLORS[method], lw=2, marker="o", ms=7,
                  mec=SURFACE, mew=1.5, label=f"{LABELS[method]} (observed order {order:.2f})",
                  zorder=3)
        # Reference slope through the coarsest point, shifted down so it does not hide the data
        ref = REF_OFFSET * e[0] * (dts / dts[0]) ** SLOPES[method]
        ax.loglog(dts, ref, color=INK_3, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"slope {SLOPES[method]}", (dts[-1], ref[-1]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9, color=INK_2)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("time step  dt  [s]", color=INK_2)
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.minorticks_off()
    ax.set_xlim(dts[-1] / 1.4, dts[0] * 1.4)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="center")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output PNG path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="config whose interceptor flies the burnout problem")
    args = parser.parse_args(argv)

    spec = load_config(args.config).interceptor
    studies = [
        (THREAT_DTS, threat_convergence(), "Ballistic threat with drag, 40 s"),
        (INTERCEPTOR_DTS, interceptor_convergence(spec),
         f"Interceptor boost through burnout ({spec.burn_time:.2f} s), 12.8 s"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, (dts, errors, title) in zip(axes, studies):
        ax.set_facecolor(SURFACE)
        _panel(ax, dts, errors, title)
        for method in ("euler", "rk4"):
            orders = ", ".join(f"{p:.2f}" for p in observed_orders(errors[method], dts))
            print(f"{title}: {method:5s} errors {np.array2string(errors[method], precision=3)}"
                  f"  orders [{orders}]")
    axes[0].set_ylabel("final position error  [m]", color=INK_2)
    fig.suptitle("Integrator convergence: error vs dt (reference: RK4 at dt/128)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
