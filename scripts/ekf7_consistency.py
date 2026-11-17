"""EKF7 consistency: 7-state NEES, NIS and the drag estimate against the truth.

Runs N seeded tracking runs (the engagement's sense-and-track loop without an
interceptor) of the 7-state EKF with each radar model and each atmosphere, in
two settings:

* Matched model: the true drag is drawn from the filter's prior and follows
  its random walk, and the truth is driven by the filter's own white-noise
  acceleration Q. The filter's model is then right, so the average 7-state
  NEES and the NIS must stay inside their chi-square bounds about 95 % of the
  time. The prior is narrowed to MATCHED_DRAG_STD so that a negative drag,
  which the filter clips at zero, is practically never drawn.
* As configured: the nominal threat (fixed drag, no process noise) and the
  filter exactly as the ablation study runs it.

It also reports the drag estimate, its claimed 1-sigma and its normalized
error at the launch decision and at the median intercept time, which is what
the drag impact predictor and the drag feed-forward actually get to use.

    pip install -e ".[plot]"
    python scripts/ekf7_consistency.py     # writes results/consistency_ekf7.json
                                           # and docs/figures/ekf7_consistency.png
"""
import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim.consistency import (  # noqa: E402
    MEAS_DIM, STATE_DIM, STATE_DIM_EKF7, average_bounds, average_nees7, average_statistics,
    drag_snapshot, fraction_inside, track_run, with_drag_prior_std,
)

_spec = importlib.util.spec_from_file_location("ablation_study", ROOT / "scripts" / "ablation_study.py")
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)

OUT_JSON = ROOT / "results" / "consistency_ekf7.json"
OUT_FIG = ROOT / "docs" / "figures" / "ekf7_consistency.png"

MATCHED_DRAG_STD = 1.0e-5   # 1/m; the 3.5e-5 prior mean is then 3.5 sigma above zero
LAUNCH_TIME = 5.0           # s, the earliest launch decision (and the median one)
# Median kill time of the EKF7 + drag PIP + APN configurations in the 1000-seed
# ablation (results/ablation_1000.csv and ablation_1000_exp.csv), s
INTERCEPT_TIME = {"constant": 58.5, "exponential": 50.7}
RADARS = {"rae": "C9_adv_rae", "cartesian": "C7_ekf7_drag_apn"}
JSON_POINTS = 300

SERIES, TRUTH = "#2a78d6", "#e8622c"
INK, INK_2, INK_3, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb", "#ecebe7"


def _one(job):
    cfg, seed, matched = job
    kwargs = {"truth_q": cfg.kalman.q, "drag_from_prior": True} if matched else {}
    return track_run(cfg, np.random.default_rng(seed), with_pip=False, **kwargs)


def _window(avg, bounds, t, t_max=None):
    avg = avg if t_max is None else avg[t <= t_max]
    avg = avg[np.isfinite(avg)]
    return {"mean": round(float(np.mean(avg)), 3), "inside": round(fraction_inside(avg, bounds), 4)}


def summarize(runs, n_runs, t_int):
    """Averaged statistics of one case, over the whole track and up to the intercept."""
    t, nees6, nis = average_statistics(runs)
    nees7 = average_nees7(runs)
    b7, b6 = average_bounds(n_runs, STATE_DIM_EKF7), average_bounds(n_runs, STATE_DIM)
    n = len(t)
    est = np.array([r.est_drag[:n] for r in runs])
    sig = np.array([r.drag_sigma[:n] for r in runs])
    true = np.array([r.true_drag[:n] for r in runs])
    out = {"steps": n, "track_time_s": round(float(t[-1]), 2)}
    for key, t_max in (("to_intercept", t_int), ("full_track", None)):
        out[key] = {"nees7": _window(nees7, b7, t, t_max), "nees6": _window(nees6, b6, t, t_max),
                    "nis": _window(nis[1:], b6, t[1:], t_max)}
    out["drag_at_launch"] = drag_snapshot(runs, LAUNCH_TIME)
    out["drag_at_intercept"] = drag_snapshot(runs, t_int)
    series = {"t": t, "nees7": nees7, "nis": nis,
              "est_median": np.median(est, axis=0), "est_q25": np.percentile(est, 25, axis=0),
              "est_q75": np.percentile(est, 75, axis=0), "sigma_median": np.median(sig, axis=0),
              "truth_median": np.median(true, axis=0)}
    return out, series


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)


def _events(ax, t_int):
    for x, ls in ((LAUNCH_TIME, (0, (1, 2))), (t_int, (0, (5, 3)))):
        ax.axvline(x, color=INK_2, lw=1, ls=ls, zorder=2)


def plot(rows, n_runs, out_path):
    """One row per case: average 7-state NEES, average NIS and the drag estimate."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(13.5, 3.5 * len(rows)), facecolor=SURFACE,
                             squeeze=False)
    b7, b6 = average_bounds(n_runs, STATE_DIM_EKF7), average_bounds(n_runs, MEAS_DIM)
    for row, (label, t_int, s) in zip(axes, rows):
        t = s["t"]
        for ax, key, bounds, dim, name in ((row[0], "nees7", b7, STATE_DIM_EKF7, "7-state NEES"),
                                           (row[1], "nis", b6, MEAS_DIM, "NIS")):
            avg = s[key]
            _style(ax, f"{label}: average {name}")
            ax.axhspan(*bounds, color=BAND, lw=0, zorder=1)
            ax.axhline(dim, color=INK_3, lw=0.8, zorder=2)
            ax.plot(t, avg, color=SERIES, lw=0.8, zorder=3)
            _events(ax, t_int)
            ok = np.isfinite(avg)
            ax.set_ylim(min(0.5 * dim, 0.9 * np.percentile(avg[ok], 0.5)),
                        max(1.6 * dim, 1.15 * np.percentile(avg[ok], 99.5)))
            ax.set_xlim(t[0], t[-1])
            # Same window and same NIS start as the JSON's "to_intercept" summary
            first = 1 if key == "nis" else 0
            inside = _window(avg[first:], bounds, t[first:], t_int)["inside"]
            ax.text(0.98, 0.95, f"{100 * inside:.1f} % of steps inside "
                    f"[{bounds[0]:.2f}, {bounds[1]:.2f}] up to intercept", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5, color=INK_2, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax = row[2]
        _style(ax, f"{label}: drag estimate")
        scale = 1e5
        ax.fill_between(t, scale * (s["est_median"] - s["sigma_median"]),
                        scale * (s["est_median"] + s["sigma_median"]), color=SERIES, alpha=0.18, lw=0,
                        zorder=2, label="median estimate ± the filter's 1σ")
        ax.plot(t, scale * s["est_median"], color=SERIES, lw=1.4, zorder=4, label="median estimate")
        ax.plot(t, scale * s["truth_median"], color=TRUTH, lw=1.2, ls=(0, (4, 2)), zorder=3,
                label="truth (median)")
        _events(ax, t_int)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 7)
        ax.set_ylabel("k_T  [1e-5 1/m]", color=INK_2, fontsize=9)
        legend = ax.legend(frameon=True, fontsize=8, labelcolor=INK, loc="upper right",
                           facecolor=SURFACE, edgecolor="none", framealpha=0.92)
        legend.set_zorder(5)
    for ax in axes[-1]:
        ax.set_xlabel("time since first detection  [s]", color=INK_2, fontsize=9)
    fig.suptitle(f"EKF7 consistency over {n_runs} seeded tracking runs (RAE radar). "
                 f"Dotted line: launch decision at {LAUNCH_TIME:g} s; dashed: median intercept",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Wrote {out_path}")


def _decimate(series):
    step = max(1, int(np.ceil(len(series["t"]) / JSON_POINTS)))
    return {k: [float(f"{x:.5g}") for x in v[::step]] for k, v in series.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=200, help="number of seeded runs (seeds 0..N-1)")
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    parser.add_argument("--fig", type=Path, default=OUT_FIG)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)

    t0 = time.time()
    seeds = range(args.runs)
    data = {"runs": args.runs, "state_dim": STATE_DIM_EKF7, "meas_dim": MEAS_DIM,
            "nees7_bounds": [round(b, 4) for b in average_bounds(args.runs, STATE_DIM_EKF7)],
            "nees6_bounds": [round(b, 4) for b in average_bounds(args.runs, STATE_DIM)],
            "nis_bounds": [round(b, 4) for b in average_bounds(args.runs, MEAS_DIM)],
            "matched_drag_std": MATCHED_DRAG_STD, "launch_time_s": LAUNCH_TIME,
            "intercept_time_s": INTERCEPT_TIME, "cases": {}}
    figure_rows = []
    with multiprocessing.Pool(args.workers) as pool:
        for atmosphere in ("constant", "exponential"):
            cfgs = {cid: cfg for cid, _, _, cfg in ablation.build_configs(atmosphere)}
            t_int = INTERCEPT_TIME[atmosphere]
            for radar, cid in RADARS.items():
                for setting in ("matched", "configured"):
                    matched = setting == "matched"
                    cfg = with_drag_prior_std(cfgs[cid], MATCHED_DRAG_STD) if matched else cfgs[cid]
                    runs = pool.map(_one, [(cfg, s, matched) for s in seeds], chunksize=4)
                    summary, series = summarize(runs, args.runs, t_int)
                    key = f"{setting}_{atmosphere}_{radar}"
                    data["cases"][key] = {"setting": setting, "atmosphere": atmosphere, "radar": radar,
                                          "config_id": cid, **summary, "series": _decimate(series)}
                    w, d = summary["to_intercept"], summary["drag_at_intercept"]
                    print(f"{key:32s} to intercept: NEES7 {w['nees7']['mean']:6.2f} "
                          f"({100 * w['nees7']['inside']:5.1f} % in), NIS {w['nis']['mean']:5.2f} "
                          f"({100 * w['nis']['inside']:5.1f} % in) | drag at intercept "
                          f"{d['estimate_median']:.3g} +/- {d['sigma_median']:.2g}, "
                          f"z {d['normalized_mean']:+.2f} +/- {d['normalized_std']:.2f}, "
                          f"coverage {100 * d['coverage']:.0f} %   [{time.time() - t0:.0f} s]", flush=True)
                    if radar == "rae":
                        name = "Matched model" if matched else "As configured"
                        figure_rows.append((f"{name}, {atmosphere}", t_int, series))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_bytes((json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8"))
    print(f"Wrote {args.json}")
    plot(figure_rows, args.runs, args.fig)


if __name__ == "__main__":
    main()
