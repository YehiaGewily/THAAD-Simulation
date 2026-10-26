"""Model mismatch: impact-prediction bias from ignoring drag.

The truth threat flies with quadratic drag, but the Kalman filter and the
impact predictor assume drag-free ballistic flight. This script tracks N
seeded threats twice, with drag on (as configured) and off (the control),
and plots the predicted-impact-point (PIP) error against the true time to go.
It also counts how often the PIP gate at the first launch opportunity makes a
different inside/outside call than the true impact point would.

    pip install -e ".[plot]"
    python scripts/drag_mismatch.py          # writes docs/figures/pip_bias.png
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim import load_config  # noqa: E402
from thaad_sim.consistency import monte_carlo, pip_bias, without_drag  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "figures" / "pip_bias.png"
DEFAULT_CONFIG = ROOT / "configs" / "engineering.yaml"
TGO_BINS = np.arange(0.0, 152.0, 2.0)   # s

# Categorical slots 1 and 2 of the reference palette, with text and grid inks
COLORS = {"drag": "#eb6834", "nodrag": "#2a78d6"}
LABELS = {"drag": "Truth with drag (as configured)", "nodrag": "Truth without drag (control)"}
INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def _summary(values, min_runs):
    """Median and 10th/90th percentiles across runs, per bin; NaN where too few runs."""
    n = np.sum(np.isfinite(values), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bins
        pct = np.nanpercentile(values, [10, 50, 90], axis=1)
    pct[:, n < min_runs] = np.nan
    return pct


def _panel(ax, tgo, stats, title, ylabel, legend_loc, symlog=None):
    for key in ("nodrag", "drag"):
        lo, med, hi = stats[key]
        ax.fill_between(tgo, lo, hi, color=COLORS[key], alpha=0.15, lw=0, zorder=2)
        ax.plot(tgo, med, color=COLORS[key], lw=2, zorder=3, label=f"{LABELS[key]}: median, 10–90 %")
    ax.axhline(0, color=INK_3, lw=0.8, zorder=1)
    if symlog:
        ax.set_yscale("symlog", linthresh=symlog, linscale=0.6)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("true time to impact  [s]", color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_xlim(tgo[np.isfinite(stats["drag"][1])].max() + 1, 0)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc=legend_loc)


def gate_calls(runs, cfg):
    """PIP-gate calls at the first launch opportunity, checked against the true impact point.

    Returns (false engages, false holds, PIP miss distances, times to go). A false
    engage is a PIP inside the defended area for a threat that lands outside it;
    a false hold is the reverse. The engagement re-evaluates the gate every step,
    so a false hold only delays the launch until the PIP drifts inside.
    """
    eng = cfg.engagement
    false_engage = false_hold = 0
    miss, tgo = [], []
    for r in runs:
        ok = np.flatnonzero((r.t >= eng.min_track_time) & np.isfinite(r.pip[:, 0]))
        if len(ok) == 0:
            continue
        i = ok[0]
        predicted_in = np.linalg.norm(r.pip[i]) <= eng.defended_radius
        truly_in = np.linalg.norm(r.impact) <= eng.defended_radius
        false_engage += predicted_in and not truly_in
        false_hold += truly_in and not predicted_in
        miss.append(np.linalg.norm(r.pip[i] - r.impact))
        tgo.append(r.impact_time - r.t[i])
    return false_engage, false_hold, np.array(miss), np.array(tgo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=100, help="number of seeded runs (seeds 0..N-1)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seeds = range(args.runs)
    runs = {"drag": monte_carlo(cfg, seeds), "nodrag": monte_carlo(without_drag(cfg), seeds)}
    bias = {k: pip_bias(r, TGO_BINS) for k, r in runs.items()}
    min_runs = max(5, args.runs // 10)
    tgo = bias["drag"].tgo

    for key in ("drag", "nodrag"):
        b = bias[key]
        print(f"{LABELS[key]}:")
        for target in (100, 80, 60, 40, 20, 10, 5):
            j = np.argmin(np.abs(tgo - target))
            along, cross, dt_ = (np.nanmedian(a[j]) for a in (b.along, b.cross, b.tgo_err))
            print(f"  t_go ~ {tgo[j]:4.0f} s: median PIP error along-track {along:8.0f} m, "
                  f"cross-track {cross:6.0f} m, time-to-go error {dt_:+6.2f} s")
        fe, fh, miss, tg = gate_calls(runs[key], cfg)
        print(f"  PIP gate at the first launch opportunity (median t_go {np.median(tg):.0f} s): "
              f"median PIP miss {np.median(miss):.0f} m, {fe} false engages and {fh} false holds "
              f"in {len(miss)} runs")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    _panel(axes[0], tgo, {k: _summary(b.along, min_runs) for k, b in bias.items()},
           "Along-track PIP error (positive = predicted beyond true impact)", "PIP error  [m]",
           "upper right", symlog=100.0)
    _panel(axes[1], tgo, {k: _summary(b.tgo_err, min_runs) for k, b in bias.items()},
           "Time-to-impact prediction error", "predicted − true time to go  [s]", "lower left")
    fig.suptitle(f"Drag-free impact prediction against a dragged threat ({args.runs} seeded runs, "
                 f"{cfg.mode.lower()})", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
