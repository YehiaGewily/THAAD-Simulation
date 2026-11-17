"""Paired Sensitivity Sweeps: KV Divert Budget and Threat Drag.

At the nominal 435 m/s divert budget every configuration that drops coast
gravity compensation protects all threatening scenarios, so the ablation
(``scripts/ablation_study.py``) cannot separate them on kills. These sweeps
move the operating point to find where each GNC component starts to matter:

  * divert: ``interceptor.kv_divert_dv`` from 100 to 550 m/s at the nominal drag
  * drag:   true threat drag ``threat.drag`` from 0 to 8e-5 1/m at the nominal budget
            (the EKF7 prior stays at 3.5e-5 +/- 2.5e-5 1/m, so this is a model-mismatch sweep)
  * noise:  every radar noise value (Cartesian position and velocity; RAE range,
            angle, Doppler and cross-range velocity) times a factor from 0.25 to 4,
            at divert budgets of 435, 150 and 100 m/s. The filter's measurement
            noise scales with it; its process noise and drag prior do not.
            Run with ``--sweeps noise``; it has its own summary and figure.

Four ablation configurations are swept over the same N paired seeds:
C1 (baseline), C5 (free-fall PN, KF6), C7 (EKF7 + drag PIP + APN, Cartesian)
and C9 (advanced preset, RAE). Metrics match the ablation: protection rate over
threatening scenarios (true impact inside the defended radius), false
engagements, and divert delta-v.

Writes:
  - results/sensitivity_divert.csv, results/sensitivity_drag.csv
  - results/sensitivity_summary.json
  - docs/figures/sensitivity_study.png
  - noise group: results/sensitivity_noise_<budget>.csv,
    results/sensitivity_noise_summary.json, docs/figures/sensitivity_noise.png

With ``--atmosphere exponential`` every run uses the exponential atmosphere and
the outputs get an ``_exp`` suffix; the swept drag is then the sea-level value.

Usage:
    py scripts/sensitivity_study.py --runs 1000
    py scripts/sensitivity_study.py --from-csv      # re-analyse the CSVs without re-running
    py scripts/sensitivity_study.py --sweeps noise --atmosphere exponential
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thaad_sim.batch import write_csv  # noqa: E402

_spec = importlib.util.spec_from_file_location("ablation_study", ROOT / "scripts" / "ablation_study.py")
ablation = importlib.util.module_from_spec(_spec)
sys.modules["ablation_study"] = ablation   # so worker processes can unpickle its functions
_spec.loader.exec_module(ablation)

OUT_JSON = ROOT / "results" / "sensitivity_summary.json"
OUT_FIG = ROOT / "docs" / "figures" / "sensitivity_study.png"
OUT_NOISE_JSON = ROOT / "results" / "sensitivity_noise_summary.json"
OUT_NOISE_FIG = ROOT / "docs" / "figures" / "sensitivity_noise.png"
NOISE_BUDGETS = [435.0, 150.0, 100.0]
CSV = {"divert": ROOT / "results" / "sensitivity_divert.csv",
       "drag": ROOT / "results" / "sensitivity_drag.csv",
       **{f"noise_{b:g}": ROOT / "results" / f"sensitivity_noise_{b:g}.csv" for b in NOISE_BUDGETS}}

SWEPT = ["C1_base_cart", "C5_kf6_vac_apn", "C7_ekf7_drag_apn", "C9_adv_rae"]
LABELS = {
    "C1_base_cart": "C1 baseline (KF6 + vac PIP + PN)",
    "C5_kf6_vac_apn": "C5 free-fall PN (KF6 + vac PIP)",
    "C7_ekf7_drag_apn": "C7 EKF7 + drag PIP + APN (Cart)",
    "C9_adv_rae": "C9 advanced preset (RAE)",
}
COLORS = {"C1_base_cart": "#eb6834", "C5_kf6_vac_apn": "#7b68ee",
          "C7_ekf7_drag_apn": "#2a78d6", "C9_adv_rae": "#184f95"}
MARKERS = {"C1_base_cart": "o", "C5_kf6_vac_apn": "s", "C7_ekf7_drag_apn": "D", "C9_adv_rae": "^"}

DIVERT_VALUES = [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 435.0, 550.0]
DRAG_VALUES = [0.0, 1.0e-5, 2.0e-5, 4.0e-5, 6.0e-5, 8.0e-5]
NOISE_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
SWEEP_COLUMNS = ("sweep", "sweep_value") + ablation.ABLATION_COLUMNS
PLOT_NOTE = ""   # extra title line, set for the exponential atmosphere
INK, INK_2 = ablation.INK, ablation.INK_2


def _with_divert(cfg, v):
    return replace(cfg, interceptor=replace(cfg.interceptor, kv_divert_dv=v))


def _with_drag(cfg, v):
    return replace(cfg, threat=replace(cfg.threat, drag=v))


def with_noise_factor(cfg, k):
    """Every radar noise value times ``k``; the filter's measurement noise follows the sensor."""
    sn = cfg.sensor
    return replace(cfg, sensor=replace(sn, pos_noise=k * sn.pos_noise, vel_noise=k * sn.vel_noise,
                                       range_noise=k * sn.range_noise, angle_noise=k * sn.angle_noise,
                                       range_rate_noise=k * sn.range_rate_noise))


def _with_noise_at_budget(budget, cfg, k):
    return with_noise_factor(_with_divert(cfg, budget), k)


SWEEPS = {"divert": (DIVERT_VALUES, _with_divert), "drag": (DRAG_VALUES, _with_drag),
          **{f"noise_{b:g}": (NOISE_FACTORS, partial(_with_noise_at_budget, b)) for b in NOISE_BUDGETS}}
GROUPS = {"noise": [f"noise_{b:g}" for b in NOISE_BUDGETS]}


def _job(job):
    sweep, value, cfg_id, cfg, seed = job
    return {"sweep": sweep, "sweep_value": value, **ablation.fast_run_metrics((cfg_id, cfg, seed))}


def run_sweep(name, base_cfgs, seeds, pool):
    values, apply = SWEEPS[name]
    jobs = [(name, v, cid, apply(base_cfgs[cid], v), s) for v in values for cid in SWEPT for s in seeds]
    t0 = time.time()
    print(f"[{name}] {len(jobs)} engagements ({len(values)} values x {len(SWEPT)} configs x {len(seeds)} seeds)...",
          flush=True)
    rows = list(pool.imap(_job, jobs, chunksize=25))
    write_csv(CSV[name], rows, SWEEP_COLUMNS)
    print(f"[{name}] wrote {CSV[name]} in {time.time() - t0:.0f} s", flush=True)
    return rows


def load_sweep(name):
    rows = ablation.load_rows(CSV[name])
    for r in rows:
        r["sweep"] = name
    return rows


def analyse(name, rows, base_cfgs, pool):
    """Per-value summaries and paired contrasts for one sweep."""
    values, apply = SWEEPS[name]
    out = []
    for v in values:
        v_rows = [r for r in rows if abs(r["sweep_value"] - v) <= 1e-12 * max(1.0, abs(v))]
        seeds = sorted({r["seed"] for r in v_rows})
        # The drag sweep changes the threat, so its true impacts are recomputed per value
        truth_cfg = apply(base_cfgs["C1_base_cart"], v)
        true_impacts = ablation.compute_true_impacts(seeds, cfg=truth_cfg, pool=pool)
        def_rad = truth_cfg.engagement.defended_radius
        by_cfg = {cid: [r for r in v_rows if r["config_id"] == cid] for cid in SWEPT}
        entry = {"value": v, "threatening": sum(1 for x in true_impacts.values() if x <= def_rad),
                 "configs": {}, "contrasts": {}}
        for cid in SWEPT:
            s = ablation.summarize_config(cid, LABELS[cid], LABELS[cid], apply(base_cfgs[cid], v),
                                          by_cfg[cid], true_impacts)
            entry["configs"][cid] = {k: s[k] for k in (
                "engaged", "kills", "threatening_total", "threatening_kills", "kill_rate_threatening",
                "ci95_threatening", "kill_rate_engaged", "false_engages", "false_holds",
                "median_dv_mps", "iqr_dv_mps", "misses_dv_exhausted", "misses")}
        for ref, cmp_ in (("C1_base_cart", "C5_kf6_vac_apn"), ("C5_kf6_vac_apn", "C7_ekf7_drag_apn"),
                          ("C7_ekf7_drag_apn", "C9_adv_rae")):
            entry["contrasts"][f"{ref}->{cmp_}"] = ablation.paired_contrast(
                by_cfg[ref], by_cfg[cmp_], true_impacts, def_rad)
        out.append(entry)
        cs = entry["configs"]
        print(f"[{name}] {v:>9.3g}: thr {entry['threatening']:4d}  " + "  ".join(
            f"{cid.split('_')[0]} {cs[cid]['kill_rate_threatening']:5.1f}% fe {cs[cid]['false_engages']:3d} "
            f"dv {cs[cid]['median_dv_mps']:5.0f}" for cid in SWEPT), flush=True)
    return out


def _style(ax, title, xlabel, ylabel):
    ablation._style_ax(ax, title, xlabel, ylabel)


def plot(summary, n_runs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), facecolor=ablation.SURFACE)

    div = summary["divert"]
    drg = summary["drag"]
    x_div = [e["value"] for e in div]
    x_drg = [e["value"] * 1e5 for e in drg]

    def series(entries, cid, key):
        return np.array([e["configs"][cid][key] for e in entries], dtype=float)

    def ci(entries, cid):
        lo = np.array([e["configs"][cid]["ci95_threatening"][0] for e in entries])
        hi = np.array([e["configs"][cid]["ci95_threatening"][1] for e in entries])
        return lo, hi

    ax = axes[0, 0]
    _style(ax, "a) Protection rate vs. KV divert budget (nominal drag)",
           "kill-vehicle divert budget  [m/s]", "threatening threats killed  [%]")
    for cid in SWEPT:
        y = series(div, cid, "kill_rate_threatening")
        lo, hi = ci(div, cid)
        ax.plot(x_div, y, color=COLORS[cid], marker=MARKERS[cid], ms=5, lw=1.8, label=LABELS[cid])
        ax.fill_between(x_div, lo, hi, color=COLORS[cid], alpha=0.12, lw=0)
    ax.axvline(435.0, color=INK_2, ls="--", lw=1.0, label="nominal budget (435 m/s)")
    ax.set_ylim(0, 104)
    ax.legend(frameon=False, fontsize=8.3, labelcolor=INK, loc="lower right")

    ax = axes[0, 1]
    _style(ax, "b) Protection rate vs. true threat drag (435 m/s budget)",
           "true threat drag k_T  [1e-5 1/m]   (EKF7 prior 3.5 ± 2.5)", "threatening threats killed  [%]")
    for cid in SWEPT:
        y = series(drg, cid, "kill_rate_threatening")
        lo, hi = ci(drg, cid)
        ax.plot(x_drg, y, color=COLORS[cid], marker=MARKERS[cid], ms=5, lw=1.8, label=LABELS[cid])
        ax.fill_between(x_drg, lo, hi, color=COLORS[cid], alpha=0.12, lw=0)
    ax.axvline(4.0, color=INK_2, ls="--", lw=1.0)
    ax.set_ylim(0, 104)

    ax = axes[1, 0]
    _style(ax, "c) False engagements vs. true threat drag",
           "true threat drag k_T  [1e-5 1/m]  (n = non-threatening seeds)", "non-threatening threats engaged  [%]")
    n_safe = np.array([n_runs - e["threatening"] for e in drg], dtype=float)
    for cid in SWEPT:
        rate = 100.0 * series(drg, cid, "false_engages") / np.maximum(n_safe, 1.0)
        ax.plot(x_drg, rate, color=COLORS[cid], marker=MARKERS[cid], ms=5, lw=1.8, label=LABELS[cid])
    ax.axvline(4.0, color=INK_2, ls="--", lw=1.0)
    ax.set_ylim(0, 100)
    ax.text(0.02, 0.97, "C1 lies under C5 (same vacuum PIP)", transform=ax.transAxes,
            ha="left", va="top", color=INK_2, fontsize=8)
    ax.set_xticks(x_drg)
    ax.set_xticklabels([f"{x:g}\nn={int(n)}" for x, n in zip(x_drg, n_safe)], fontsize=8)

    ax = axes[1, 1]
    _style(ax, "d) Median KV divert Δv vs. true threat drag",
           "true threat drag k_T  [1e-5 1/m]", "median divert Δv, engaged  [m/s]")
    for cid in SWEPT:
        ax.plot(x_drg, series(drg, cid, "median_dv_mps"), color=COLORS[cid], marker=MARKERS[cid],
                ms=5, lw=1.8, label=LABELS[cid])
    ax.axhline(435.0, color="#b2182b", ls="--", lw=1.1)
    ax.axvline(4.0, color=INK_2, ls="--", lw=1.0)
    ax.set_ylim(0, 470)

    fig.suptitle(f"Sensitivity of the GNC ablation to divert budget and drag mismatch "
                 f"({n_runs} paired seeds per point){PLOT_NOTE}",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=ablation.SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_noise(summary, n_runs, out_path):
    """Protection vs. noise factor at each budget, and median divert at the nominal budget."""
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), facecolor=ablation.SURFACE)
    panels = [(axes[0, 0], "a", 435.0), (axes[0, 1], "b", 150.0), (axes[1, 0], "c", 100.0)]
    ticks = [f"{k:g}×\n{25.0 * k:g} m" for k in NOISE_FACTORS]

    def series(entries, cid, key):
        return np.array([e["configs"][cid][key] for e in entries], dtype=float)

    def log_axis(ax, x):
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.minorticks_off()
        ax.axvline(1.0, color=INK_2, ls="--", lw=1.0)

    for ax, letter, budget in panels:
        entries = summary[f"noise_{budget:g}"]
        x = [e["value"] for e in entries]
        _style(ax, f"{letter}) Protection rate vs. radar noise ({budget:g} m/s divert budget)",
               "radar noise factor  (Cartesian position 1σ)", "threatening threats killed  [%]")
        for cid in SWEPT:
            lo = np.array([e["configs"][cid]["ci95_threatening"][0] for e in entries])
            hi = np.array([e["configs"][cid]["ci95_threatening"][1] for e in entries])
            ax.plot(x, series(entries, cid, "kill_rate_threatening"), color=COLORS[cid], marker=MARKERS[cid],
                    ms=5, lw=1.8, label=LABELS[cid])
            ax.fill_between(x, lo, hi, color=COLORS[cid], alpha=0.12, lw=0)
        log_axis(ax, x)
        ax.set_ylim(0, 104)

    ax = axes[1, 1]
    entries = summary["noise_435"]
    x = [e["value"] for e in entries]
    _style(ax, "d) Median KV divert Δv vs. radar noise (435 m/s budget)",
           "radar noise factor  (Cartesian position 1σ)", "median divert Δv, engaged  [m/s]")
    for cid in SWEPT:
        ax.plot(x, series(entries, cid, "median_dv_mps"), color=COLORS[cid], marker=MARKERS[cid],
                ms=5, lw=1.8, label=LABELS[cid])
    ax.axhline(435.0, color="#b2182b", ls="--", lw=1.1)
    log_axis(ax, x)
    ax.set_ylim(0, 470)
    ax.legend(frameon=True, facecolor=ablation.SURFACE, edgecolor="none", fontsize=8.3,
              labelcolor=INK, loc="center left")

    fig.suptitle(f"Sensitivity of the GNC ablation to radar noise ({n_runs} paired seeds per point; "
                 f"1× = nominal, filter process noise not retuned){PLOT_NOTE}",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=ablation.SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=1000, help="paired seeds 0..N-1 (default: 1000)")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: CPU count)")
    parser.add_argument("--sweeps", default="divert,drag",
                        help="comma list of sweeps or groups (default: divert,drag; also: noise)")
    parser.add_argument("--from-csv", action="store_true", help="re-analyse the sweep CSVs instead of re-running")
    parser.add_argument("--smoke", type=Path, default=None, metavar="DIR",
                        help="quick check: two values per sweep, all outputs written to DIR")
    parser.add_argument("--atmosphere", choices=("constant", "exponential"), default="constant",
                        help="density model for every configuration; exponential writes *_exp outputs")
    args = parser.parse_args(argv)

    global OUT_JSON, OUT_FIG, OUT_NOISE_JSON, OUT_NOISE_FIG, PLOT_NOTE
    if args.atmosphere == "exponential":
        OUT_JSON = OUT_JSON.with_name("sensitivity_summary_exp.json")
        OUT_FIG = OUT_FIG.with_name("sensitivity_study_exp.png")
        OUT_NOISE_JSON = OUT_NOISE_JSON.with_name("sensitivity_noise_summary_exp.json")
        OUT_NOISE_FIG = OUT_NOISE_FIG.with_name("sensitivity_noise_exp.png")
        for k in CSV:
            CSV[k] = CSV[k].with_name(CSV[k].stem + "_exp.csv")
        PLOT_NOTE = "\nExponential atmosphere; drag values are sea-level coefficients"
    if args.smoke:
        args.smoke.mkdir(parents=True, exist_ok=True)
        OUT_JSON, OUT_FIG = args.smoke / OUT_JSON.name, args.smoke / OUT_FIG.name
        OUT_NOISE_JSON, OUT_NOISE_FIG = args.smoke / OUT_NOISE_JSON.name, args.smoke / OUT_NOISE_FIG.name
        for k in CSV:
            CSV[k] = args.smoke / CSV[k].name
        DIVERT_VALUES[:] = [150.0, 435.0]
        DRAG_VALUES[:] = [0.0, 4.0e-5]
        NOISE_FACTORS[:] = [0.5, 2.0]

    names = []
    for n in (n.strip() for n in args.sweeps.split(",") if n.strip()):
        names.extend(GROUPS.get(n, [n]))
    for n in names:
        if n not in SWEEPS:
            parser.error(f"unknown sweep {n!r}; choose from {', '.join(list(SWEEPS) + list(GROUPS))}")

    base_cfgs = {cid: cfg for cid, _, _, cfg in ablation.build_configs(args.atmosphere) if cid in SWEPT}
    seeds = list(range(args.runs))
    workers = args.workers or os.cpu_count() or 1

    # The noise group keeps its own summary, so the divert/drag summary is never rewritten by it
    paths = {"main": OUT_JSON, "noise": OUT_NOISE_JSON}
    summaries = {g: json.loads(p.read_text(encoding="utf-8")) if p.exists() else {} for g, p in paths.items()}
    touched = set()
    with multiprocessing.Pool(workers) as pool:
        for name in names:
            group = "noise" if name in GROUPS["noise"] else "main"
            rows = load_sweep(name) if args.from_csv else run_sweep(name, base_cfgs, seeds, pool)
            summaries[group][name] = analyse(name, rows, base_cfgs, pool)
            summaries[group]["runs_per_point"] = len({r["seed"] for r in rows})
            touched.add(group)

    for group in sorted(touched):
        paths[group].write_bytes((json.dumps(summaries[group], indent=2) + "\n").encode("utf-8"))
        print(f"Wrote {paths[group]}")
    main_s, noise_s = summaries["main"], summaries["noise"]
    if "main" in touched and "divert" in main_s and "drag" in main_s:
        plot(main_s, main_s["runs_per_point"], OUT_FIG)
    if "noise" in touched and all(n in noise_s for n in GROUPS["noise"]):
        plot_noise(noise_s, noise_s["runs_per_point"], OUT_NOISE_FIG)


if __name__ == "__main__":
    main()
