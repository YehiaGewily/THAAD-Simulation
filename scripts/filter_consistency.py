"""Kalman filter consistency: average NEES and NIS against chi-square 95 % bounds.

Runs N seeded tracking runs (the engagement's sense-and-track loop without an
interceptor) in two settings and plots the per-step average of the normalized
estimation error squared (NEES) and the normalized innovation squared (NIS):

* Matched model: drag off, and the truth driven by the filter's own discretized
  white-noise acceleration Q. Here the filter's model is exactly right, so both
  averages must stay inside their chi-square bounds about 95 % of the time.
* As configured: the real threat with quadratic drag, which the filter does not
  model. This shows how far the configured q is from consistent.

    pip install -e ".[plot]"
    python scripts/filter_consistency.py     # writes docs/figures/nees.png and nis.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim import load_config  # noqa: E402
from thaad_sim.consistency import (  # noqa: E402
    MEAS_DIM, STATE_DIM, average_bounds, average_statistics, fraction_inside, monte_carlo,
    without_drag,
)

DEFAULT_OUT_DIR = ROOT / "docs" / "figures"
DEFAULT_CONFIG = ROOT / "configs" / "engineering.yaml"
DEFAULT_JSON = ROOT / "results" / "consistency.json"
JSON_POINTS = 600   # the series in the JSON are decimated to about this many points

# Categorical slot 1 of the reference palette, with text, band and grid inks
SERIES = "#2a78d6"
INK, INK_2, INK_3, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#ecebe7"


def _panel(ax, t, avg, bounds, dim, title):
    lo, hi = bounds
    inside = fraction_inside(avg, bounds)
    ax.axhspan(lo, hi, color=BAND, lw=0, zorder=1, label=f"95 % bounds [{lo:.2f}, {hi:.2f}]")
    for y in (lo, hi):
        ax.axhline(y, color=INK_3, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.axhline(dim, color=INK_3, lw=0.8, zorder=2)
    ax.plot(t, avg, color=SERIES, lw=0.8, zorder=3,
            label=f"average over runs ({100 * inside:.1f} % of steps inside)")
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel("time since first detection  [s]", color=INK_2)
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    return inside


def _figure(name, cases, dim, n_runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, sharey=True)
    for ax, (title, t, avg, bounds) in zip(axes, cases):
        ax.set_facecolor(SURFACE)
        _panel(ax, t, avg, bounds, dim, title)
    top = max(np.nanmax(avg) for _, _, avg, _ in cases)
    axes[0].set_ylim(0.5 * dim, 1.25 * top)
    axes[0].set_ylabel(f"average {name}  (dof {dim})", color=INK_2)
    fig.suptitle(f"Kalman filter consistency: average {name} over {n_runs} seeded runs "
                 f"against chi-square 95 % bounds",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out}")


def _series(t, avg, bounds, max_points=JSON_POINTS):
    """Summary and a decimated copy of one averaged statistic, for the JSON file."""
    step = max(1, int(np.ceil(len(t) / max_points)))
    return {
        "mean": round(float(np.mean(avg)), 4),
        "inside": round(fraction_inside(avg, bounds), 4),
        "steps": len(t),
        "t": np.round(t[::step], 3).tolist(),
        "value": np.round(avg[::step], 4).tolist(),
    }


def write_json(path, n_runs, q, nees_b, nis_b, results):
    """The averaged NEES and NIS behind the figures, read by python -m thaad_sim.dashboard."""
    data = {"runs": n_runs, "q": q, "state_dim": STATE_DIM, "meas_dim": MEAS_DIM,
            "nees_bounds": [round(b, 4) for b in nees_b], "nis_bounds": [round(b, 4) for b in nis_b]}
    for key, (t, nees, nis, t_nis) in results.items():
        data[key] = {"nees": _series(t, nees, nees_b), "nis": _series(t_nis, nis, nis_b)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=50, help="number of seeded runs (seeds 0..N-1)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON,
                        help="where to write the averaged series for the dashboard")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    seeds = range(args.runs)
    results = {}
    for key, c, truth_q in (("matched", without_drag(cfg), cfg.kalman.q), ("configured", cfg, None)):
        t, nees, nis = average_statistics(monte_carlo(c, seeds, truth_q=truth_q))
        results[key] = (t, nees, nis[1:], t[1:])   # NIS starts at the first real update

    nees_b = average_bounds(args.runs, STATE_DIM)
    nis_b = average_bounds(args.runs, MEAS_DIM)
    q = cfg.kalman.q
    for key, label in (("matched", "matched model"), ("configured", "as configured")):
        t, nees, nis, _ = results[key]
        print(f"{label:14s} q={q:g}: mean NEES {nees.mean():7.2f} "
              f"({100 * fraction_inside(nees, nees_b):5.1f} % inside {nees_b[0]:.2f}-{nees_b[1]:.2f}), "
              f"mean NIS {nis.mean():5.2f} ({100 * fraction_inside(nis, nis_b):5.1f} % inside "
              f"{nis_b[0]:.2f}-{nis_b[1]:.2f}), {len(t)} steps, {args.runs} runs")

    write_json(args.json, args.runs, q, nees_b, nis_b, results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    titles = (f"Matched model: drag off, truth noise = Q  (q = {q:g} m²/s³)",
              f"As configured: threat with drag  (q = {q:g} m²/s³)")
    m, c = results["matched"], results["configured"]
    _figure("NEES", [(titles[0], m[0], m[1], nees_b), (titles[1], c[0], c[1], nees_b)],
            STATE_DIM, args.runs, args.out_dir / "nees.png")
    _figure("NIS", [(titles[0], m[3], m[2], nis_b), (titles[1], c[3], c[2], nis_b)],
            MEAS_DIM, args.runs, args.out_dir / "nis.png")


if __name__ == "__main__":
    main()
