"""1000-Seed Paired GNC Ablation Study.

Decomposes the transition from the Baseline GNC stack (configs/engineering.yaml:
6-state linear KF, vacuum PIP, classical PN, Cartesian radar) to the Advanced
GNC stack (configs/advanced.yaml: 7-state drag-estimating EKF, RK4 drag PIP,
Augmented PN / Optimal ZEM, Cartesian and polar RAE radar) one component at a
time across N=1000 paired scenario seeds on the identical physical interceptor
(150 kN boost for 5.0 s, 30 g lateral limit, 435 m/s KV divert budget, 10 m
kill radius).

Writes:
  - results/ablation_1000.csv
  - results/ablation_summary.json
  - docs/figures/ablation_study.png
  - results/gnc_trade_study.json & docs/figures/gnc_trade_study.png (updated to N seeds)

The headline metric is the protection rate: the share of *threatening*
scenarios (the unintercepted threat lands inside the defended radius) that end
in a kill. Its denominator is the same for every configuration, unlike the
kill rate among engaged threats, whose denominator depends on each
configuration's own launch decisions. One-factor contrasts are tested with an
exact paired McNemar test on the threatening seeds, and divert use with a
bootstrap interval on the paired per-seed difference.

With ``--atmosphere exponential`` every configuration flies in an exponential
atmosphere (sim.atmosphere), and the outputs get an ``_exp`` suffix so the
constant-density results are kept. The trade-study files are not rewritten.

Usage:
    py scripts/ablation_study.py --runs 1000
    py scripts/ablation_study.py --from-csv      # re-analyse results/ablation_1000.csv without re-running
    py scripts/ablation_study.py --atmosphere exponential
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from dataclasses import replace
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

from thaad_sim import load_config  # noqa: E402
from thaad_sim.batch import COLUMNS, write_csv  # noqa: E402
from thaad_sim.consistency import monte_carlo, pip_bias  # noqa: E402
from thaad_sim.dynamics import Interceptor  # noqa: E402
from thaad_sim.engagement import get_cpa, make_rng  # noqa: E402
from thaad_sim.estimation import make_filter, measure_radar  # noqa: E402
from thaad_sim.scenario import spawn_threat  # noqa: E402
from thaad_sim.stats import mcnemar_exact, wilson_interval  # noqa: E402

OUT_CSV = ROOT / "results" / "ablation_1000.csv"
OUT_JSON = ROOT / "results" / "ablation_summary.json"
OUT_FIG = ROOT / "docs" / "figures" / "ablation_study.png"
TRADE_JSON = ROOT / "results" / "gnc_trade_study.json"
TRADE_FIG = ROOT / "docs" / "figures" / "gnc_trade_study.png"

INK, INK_2, INK_3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
C_BASE, C_MID, C_ADV, C_ZEM = "#eb6834", "#d99b26", "#2a78d6", "#1b7837"

ABLATION_COLUMNS = (
    "config_id",
    "filter_type",
    "pip_mode",
    "guidance_law",
    "sensor_model",
    *COLUMNS,
    "dv_exhausted",
    "est_drag_final",
)


def with_atmosphere(cfg, atmosphere):
    """The same config flying in the given atmosphere model."""
    return replace(cfg, sim=replace(cfg.sim, atmosphere=atmosphere))


def build_configs(atmosphere="constant"):
    """Return ordered list of (config_id, short_label, full_label, SimConfig)."""
    base = with_atmosphere(load_config(ROOT / "configs" / "engineering.yaml"), atmosphere)
    adv = with_atmosphere(load_config(ROOT / "configs" / "advanced.yaml"), atmosphere)

    kf6_cfg = base.kalman
    ekf7_vac = replace(adv.kalman, pip_mode="vacuum")
    ekf7_drag = adv.kalman

    cart_sensor = base.sensor
    rae_sensor = adv.sensor

    int_pn = replace(base.interceptor, guidance_law="pn")
    int_apn = replace(base.interceptor, guidance_law="apn")
    int_zem = replace(base.interceptor, guidance_law="zem")

    return [
        (
            "C1_base_cart",
            "C1: KF6 + Vac + PN\n(Cartesian, Baseline)",
            "C1: KF6 + Vacuum PIP + PN (Cartesian, Baseline)",
            replace(base, sensor=cart_sensor, kalman=kf6_cfg, interceptor=int_pn),
        ),
        (
            "C2_base_rae",
            "C2: KF6 + Vac + PN\n(RAE Polar Radar)",
            "C2: KF6 + Vacuum PIP + PN (RAE Polar Radar)",
            replace(base, sensor=rae_sensor, kalman=kf6_cfg, interceptor=int_pn),
        ),
        (
            "C3_ekf7_vac_pn",
            "C3: EKF7 + Vac + PN\n(Filter Only)",
            "C3: EKF7 + Vacuum PIP + PN (Cartesian)",
            replace(base, sensor=cart_sensor, kalman=ekf7_vac, interceptor=int_pn),
        ),
        (
            "C4_ekf7_drag_pn",
            "C4: EKF7 + Drag + PN\n(Filter + Drag PIP)",
            "C4: EKF7 + Drag PIP + PN (Cartesian)",
            replace(base, sensor=cart_sensor, kalman=ekf7_drag, interceptor=int_pn),
        ),
        (
            "C5_kf6_vac_apn",
            "C5: KF6 + Vac + APN\n(Free-Fall PN, a_T=0)",
            "C5: KF6 + Vacuum PIP + APN (Cartesian, a_T=0)",
            replace(base, sensor=cart_sensor, kalman=kf6_cfg, interceptor=int_apn),
        ),
        (
            "C6_ekf7_vac_apn",
            "C6: EKF7 + Vac + APN\n(Filter + Drag APN)",
            "C6: EKF7 + Vacuum PIP + APN (Cartesian)",
            replace(base, sensor=cart_sensor, kalman=ekf7_vac, interceptor=int_apn),
        ),
        (
            "C7_ekf7_drag_apn",
            "C7: EKF7 + Drag + APN\n(Full GNC, Cartesian)",
            "C7: EKF7 + Drag PIP + APN (Cartesian)",
            replace(base, sensor=cart_sensor, kalman=ekf7_drag, interceptor=int_apn),
        ),
        (
            "C8_ekf7_drag_zem",
            "C8: EKF7 + Drag + ZEM\n(Optimal ZEM, Cart)",
            "C8: EKF7 + Drag PIP + ZEM (Cartesian)",
            replace(base, sensor=cart_sensor, kalman=ekf7_drag, interceptor=int_zem),
        ),
        (
            "C9_adv_rae",
            "C9: EKF7 + Drag + APN\n(RAE, Advanced Preset)",
            "C9: EKF7 + Drag PIP + APN (RAE, Advanced Preset)",
            adv,
        ),
    ]


def fast_run_metrics(job):
    """Run one seeded engagement without per-step history copies or Console formatting.

    Produces the exact same numerical results as thaad_sim.batch.run_metrics.
    """
    cfg_id, cfg, seed = job
    rng = make_rng(seed)
    dt, g = cfg.sim.dt, cfg.sim.g
    eng = cfg.engagement

    sc = spawn_threat(cfg, rng)
    threat = sc.threat
    interceptor = Interceptor([0, 0, 0], cfg.interceptor, g, cfg.sim.integrator,
                              scale_height=cfg.sim.density_scale_height)
    kf = make_filter(cfg)

    t_max = max(cfg.sim.t_max, sc.impact_time * 2.0 + 30.0)
    t = 0.0
    outcome = "UNKNOWN"
    kill_info = {}
    launch_pip = None
    min_sep = None

    pos_err_sq = 0.0
    vel_err_sq = 0.0
    n_steps = 0
    end_t = 0.0

    while t < t_max:
        mp, mv, R_meas = measure_radar(threat.pos, threat.vel, cfg.sensor, rng)
        ep, ev = kf.update(mp, mv, R_meas=R_meas)
        dp = ep - threat.pos
        dv_err = ev - threat.vel
        pos_err_sq += float(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2])
        vel_err_sq += float(dv_err[0] * dv_err[0] + dv_err[1] * dv_err[1] + dv_err[2] * dv_err[2])
        n_steps += 1

        if (t >= eng.min_track_time) and (not interceptor.launched) and (ev[2] < 0):
            pip = kf.predict_impact(g)
            if cfg.pip_gated:
                engage = pip is not None and np.linalg.norm(pip[:2]) <= eng.defended_radius
            else:
                engage = True
            if engage:
                direction = ep - interceptor.pos
                direction = direction / np.linalg.norm(direction)
                interceptor.launch(t, direction)
                launch_pip = pip

        t_next = t + dt
        i_p0 = interceptor.pos.copy()
        t_p0 = threat.pos.copy()
        interceptor.update(t, ep, ev, dt, target_a=kf.est_target_accel)
        t_ground = threat.update(dt)
        window = dt if t_ground is None else t_ground

        i_v = (interceptor.pos - i_p0) / dt
        t_v = (threat.pos - t_p0) / window
        if interceptor.launched and not interceptor.exploded:
            t_off, d_min = get_cpa(i_p0, i_v, t_p0, t_v, window)
            if min_sep is None or d_min < min_sep:
                min_sep = d_min
            if d_min < eng.kill_radius:
                interceptor.exploded = True
                k_time = t + t_off
                k_pos_t = t_p0 + t_v * t_off
                closing = float(np.linalg.norm(i_v - t_v))
                end_t = k_time
                outcome = "KILL"
                kill_info = {"time": k_time, "alt": float(k_pos_t[2]), "dist": float(d_min), "closing": closing}
                break

        if not threat.active:
            end_t = t + t_ground
            impact_range = float(np.linalg.norm(threat.pos[:2]))
            if cfg.pip_gated and not interceptor.launched and impact_range > eng.defended_radius:
                outcome = "NO ENGAGE"
            else:
                outcome = "GROUND"
            break

        end_t = t_next
        t = t_next

    if outcome == "UNKNOWN":
        outcome = "TIMEOUT"

    launched = interceptor.launched
    dv_used = float(interceptor.divert_dv_used) if launched else None
    budget = cfg.interceptor.kv_divert_dv
    dv_exhausted = int(launched and budget is not None and dv_used >= budget - 1e-6)

    return {
        "config_id": cfg_id,
        "filter_type": cfg.kalman.filter_type,
        "pip_mode": cfg.kalman.pip_mode,
        "guidance_law": cfg.interceptor.guidance_law,
        "sensor_model": cfg.sensor.model,
        "seed": seed,
        "outcome": outcome,
        "launched": int(launched),
        "launch_time_s": interceptor.launch_time if launched else None,
        "launch_pip_range_m": None if launch_pip is None else float(np.linalg.norm(launch_pip[:2])),
        "kill_time_s": kill_info.get("time"),
        "kill_alt_m": kill_info.get("alt"),
        "closing_velocity_mps": kill_info.get("closing"),
        "miss_distance_m": float(min_sep) if min_sep is not None else None,
        "max_g": max(interceptor.g_hist) if interceptor.g_hist else None,
        "divert_dv_used_mps": dv_used,
        "impact_range_m": float(np.linalg.norm(threat.pos[:2])) if outcome in ("GROUND", "NO ENGAGE") else None,
        "end_time_s": float(end_t),
        "kf_rms_pos_m": float(np.sqrt(pos_err_sq / max(n_steps, 1))),
        "kf_rms_vel_mps": float(np.sqrt(vel_err_sq / max(n_steps, 1))),
        "pos_noise_m": cfg.sensor.pos_noise,
        "vel_noise_mps": cfg.sensor.vel_noise,
        "dv_exhausted": dv_exhausted,
        "est_drag_final": float(kf.est_drag),
    }


def true_impact_range(job):
    """Ground range of the unintercepted threat's impact for one (cfg, seed)."""
    cfg, seed = job
    threat = spawn_threat(cfg, make_rng(seed)).threat
    while threat.active:
        threat.update(cfg.sim.dt)
    return float(np.linalg.norm(threat.pos[:2]))


def compute_true_impacts(seeds, cfg=None, pool=None):
    """Integrate the unintercepted threat to ground for each seed to get its true impact range.

    The threat depends only on the scenario, threat and sim sections, so one
    config serves every GNC variant that shares them.
    """
    cfg = cfg or load_config(ROOT / "configs" / "engineering.yaml")
    jobs = [(cfg, s) for s in seeds]
    ranges = pool.map(true_impact_range, jobs, chunksize=20) if pool else map(true_impact_range, jobs)
    return dict(zip(seeds, ranges))


# One-factor contrasts: (reference, comparison, what changes)
CONTRASTS = [
    ("C1_base_cart", "C2_base_rae", "Cartesian -> RAE radar (KF6, vacuum PIP, PN)"),
    ("C1_base_cart", "C3_ekf7_vac_pn", "KF6 -> EKF7 filter (vacuum PIP, PN)"),
    ("C3_ekf7_vac_pn", "C4_ekf7_drag_pn", "vacuum -> drag PIP (EKF7, PN)"),
    ("C1_base_cart", "C5_kf6_vac_apn", "PN -> free-fall PN, no coast gravity comp (KF6)"),
    ("C5_kf6_vac_apn", "C6_ekf7_vac_apn", "KF6 -> EKF7 + drag feed-forward (vac PIP, APN)"),
    ("C6_ekf7_vac_apn", "C7_ekf7_drag_apn", "vacuum -> drag PIP (EKF7, APN)"),
    ("C7_ekf7_drag_apn", "C8_ekf7_drag_zem", "APN -> ZEM (EKF7, drag PIP)"),
    ("C7_ekf7_drag_apn", "C9_adv_rae", "Cartesian -> RAE radar (full advanced stack)"),
    ("C1_base_cart", "C9_adv_rae", "baseline -> full advanced stack"),
]


def _bootstrap_median_ci(x, n_boot=4000, seed=0):
    """Percentile bootstrap 95 % interval for the median of ``x`` (fixed RNG seed, reproducible)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def paired_contrast(ref_rows, cmp_rows, true_impacts, def_rad):
    """Paired statistics for one contrast over the same seeds."""
    ref = {r["seed"]: r for r in ref_rows}
    cmp_ = {r["seed"]: r for r in cmp_rows}
    seeds = sorted(set(ref) & set(cmp_))
    threat = [s for s in seeds if true_impacts[s] <= def_rad]
    safe = [s for s in seeds if true_impacts[s] > def_rad]

    # Protection: kill of a threatening threat. b = only ref kills, c = only cmp kills.
    b = sum(1 for s in threat if ref[s]["outcome"] == "KILL" and cmp_[s]["outcome"] != "KILL")
    c = sum(1 for s in threat if ref[s]["outcome"] != "KILL" and cmp_[s]["outcome"] == "KILL")
    k_ref = sum(1 for s in threat if ref[s]["outcome"] == "KILL")
    k_cmp = sum(1 for s in threat if cmp_[s]["outcome"] == "KILL")

    # False engagement: a launch at a threat that would have landed outside the defended area.
    fb = sum(1 for s in safe if ref[s]["launched"] and not cmp_[s]["launched"])
    fc = sum(1 for s in safe if not ref[s]["launched"] and cmp_[s]["launched"])
    f_ref = sum(1 for s in safe if ref[s]["launched"])
    f_cmp = sum(1 for s in safe if cmp_[s]["launched"])

    # Divert delta-v on seeds both configurations engaged
    both = [s for s in seeds if ref[s]["launched"] and cmp_[s]["launched"]]
    d_dv = [cmp_[s]["divert_dv_used_mps"] - ref[s]["divert_dv_used_mps"] for s in both]
    lo, hi = _bootstrap_median_ci(d_dv)

    n_t, n_s = len(threat), len(safe)
    return {
        "threatening": n_t,
        "protection_ref_pct": round(100.0 * k_ref / n_t, 2) if n_t else None,
        "protection_cmp_pct": round(100.0 * k_cmp / n_t, 2) if n_t else None,
        "protection_diff_pts": round(100.0 * (k_cmp - k_ref) / n_t, 2) if n_t else None,
        "discordant_only_ref": b,
        "discordant_only_cmp": c,
        "mcnemar_p": mcnemar_exact(b, c),
        "non_threatening": n_s,
        "false_engage_ref": f_ref,
        "false_engage_cmp": f_cmp,
        "false_engage_mcnemar_p": mcnemar_exact(fb, fc),
        "paired_engaged": len(both),
        "median_dv_diff_mps": round(float(np.median(d_dv)), 1) if d_dv else None,
        "median_dv_diff_ci95": [round(lo, 1), round(hi, 1)],
    }


def load_rows(path):
    """Read an ablation CSV back into typed rows (empty cells -> None)."""
    ints = {"seed", "launched", "dv_exhausted"}
    text = {"config_id", "filter_type", "pip_mode", "guidance_law", "sensor_model", "outcome", "sweep"}
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if k in text:
                    row[k] = v
                elif v == "":
                    row[k] = None
                elif k in ints:
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            rows.append(row)
    return rows


def summarize_config(cfg_id, short_label, full_label, cfg, rows, true_impacts):
    def_rad = cfg.engagement.defended_radius
    n_total = len(rows)
    eng = [r for r in rows if r["launched"] == 1]
    kills = [r for r in eng if r["outcome"] == "KILL"]
    misses = [r for r in eng if r["outcome"] != "KILL"]
    no_eng = [r for r in rows if r["outcome"] == "NO ENGAGE"]
    uneng_ground = [r for r in rows if r["launched"] == 0 and r["outcome"] == "GROUND"]

    # False engage: launched, but true unintercepted threat lands outside defended_radius
    false_engages = sum(1 for r in eng if true_impacts[r["seed"]] > def_rad)
    # Threatening scenarios: true unintercepted threat lands inside defended_radius
    threatening = [r for r in rows if true_impacts[r["seed"]] <= def_rad]
    threatening_kills = sum(1 for r in threatening if r["outcome"] == "KILL")
    false_holds = sum(1 for r in threatening if r["launched"] == 0)

    k, n_eng = len(kills), len(eng)
    p_kill = k / n_eng if n_eng else 0.0
    ci_lo, ci_hi = wilson_interval(k, n_eng) if n_eng else (0.0, 0.0)

    kt, nt = threatening_kills, len(threatening)
    p_th = kt / nt if nt else 0.0
    th_lo, th_hi = wilson_interval(kt, nt) if nt else (0.0, 0.0)

    dvs = np.array([r["divert_dv_used_mps"] for r in eng if r["divert_dv_used_mps"] is not None], dtype=float)
    dv_q1, dv_med, dv_q3 = np.percentile(dvs, [25, 50, 75]) if dvs.size else (0.0, 0.0, 0.0)
    dv_mean = float(np.mean(dvs)) if dvs.size else 0.0

    miss_dvs_exh = sum(r["dv_exhausted"] for r in misses)
    miss_dists = np.array([r["miss_distance_m"] for r in misses if r["miss_distance_m"] is not None], dtype=float)
    miss_med = float(np.median(miss_dists)) if miss_dists.size else 0.0

    cpa_all = np.array([r["miss_distance_m"] for r in eng if r["miss_distance_m"] is not None], dtype=float)
    cpa_med = float(np.median(cpa_all)) if cpa_all.size else 0.0

    pos_rms = np.array([r["kf_rms_pos_m"] for r in rows], dtype=float)
    vel_rms = np.array([r["kf_rms_vel_mps"] for r in rows], dtype=float)

    return {
        "config_id": cfg_id,
        "short_label": short_label.replace("\n", " "),
        "label": full_label,
        "filter_type": cfg.kalman.filter_type,
        "pip_mode": cfg.kalman.pip_mode,
        "guidance_law": cfg.interceptor.guidance_law,
        "sensor_model": cfg.sensor.model,
        "runs": n_total,
        "engaged": n_eng,
        "kills": k,
        "misses": len(misses),
        "no_engage": len(no_eng),
        "unengaged_ground": len(uneng_ground),
        "kill_rate_engaged": round(p_kill * 100.0, 2),
        "ci95_engaged": [round(ci_lo * 100.0, 2), round(ci_hi * 100.0, 2)],
        "threatening_total": nt,
        "threatening_kills": kt,
        "kill_rate_threatening": round(p_th * 100.0, 2),
        "ci95_threatening": [round(th_lo * 100.0, 2), round(th_hi * 100.0, 2)],
        "false_engages": false_engages,
        "false_holds": false_holds,
        "misses_dv_exhausted": miss_dvs_exh,
        "median_dv_mps": round(float(dv_med), 1),
        "iqr_dv_mps": [round(float(dv_q1), 1), round(float(dv_q3), 1)],
        "mean_dv_mps": round(dv_mean, 1),
        "median_cpa_engaged_m": round(cpa_med, 2),
        "median_miss_distance_m": round(miss_med, 2),
        "median_kf_rms_pos_m": round(float(np.median(pos_rms)), 2),
        "median_kf_rms_vel_mps": round(float(np.median(vel_rms)), 2),
    }


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=9, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_3)


def plot_ablation(summaries, by_cfg, n_runs, out_path, note=""):
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1.0, 0.8]})

    labels = [
        "C1: KF6 + Vac + PN (Cart, Base)",
        "C2: KF6 + Vac + PN (RAE Sensor)",
        "C3: EKF7 + Vac + PN (Filter Only)",
        "C4: EKF7 + Drag + PN (+Drag PIP)",
        "C5: KF6 + Vac + APN (Free-Fall PN)",
        "C6: EKF7 + Vac + APN (+Drag APN)",
        "C7: EKF7 + Drag + APN (Full Cart)",
        "C8: EKF7 + Drag + ZEM (Full ZEM)",
        "C9: EKF7 + Drag + APN (Full RAE)",
    ]
    colors = [
        C_BASE, C_BASE,
        C_MID, C_MID,
        "#7b68ee",
        C_ADV, C_ADV, C_ZEM, "#184f95",
    ]
    y = np.arange(len(summaries))[::-1]
    n_thr = summaries[0]["threatening_total"]
    n_safe = summaries[0]["runs"] - n_thr

    # Panel A: protection rate over the same threatening scenarios for every configuration
    ax = axes[0]
    _style_ax(ax, f"a) Protection rate: kills of the {n_thr} threatening threats",
              "threatening threats killed  [%]  (95% Wilson CI)", "")
    rates = np.array([s["kill_rate_threatening"] for s in summaries])
    lo = np.array([s["ci95_threatening"][0] for s in summaries])
    hi = np.array([s["ci95_threatening"][1] for s in summaries])
    ax.barh(y, rates, height=0.62, color=colors, alpha=0.88, zorder=3)
    ax.errorbar(rates, y, xerr=[rates - lo, hi - rates], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=3.5, zorder=4)
    ax.set_yticks(y, labels, fontsize=8.8, color=INK)
    ax.set_xlim(0, 118)
    for yi, s, r, h in zip(y, summaries, rates, hi):
        ax.text(h + 1.5, yi, f"{r:.1f}% ({s['threatening_kills']}/{s['threatening_total']})",
                va="center", ha="left", fontsize=8.3, color=INK, fontweight="semibold")

    # Panel B: divert delta-v median + IQR across engaged runs
    ax = axes[1]
    _style_ax(ax, "b) Kill-vehicle divert Δv (median & IQR, engaged)",
              "KV divert Δv used  [m/s]", "")
    meds = np.array([s["median_dv_mps"] for s in summaries])
    q1 = np.array([s["iqr_dv_mps"][0] for s in summaries])
    q3 = np.array([s["iqr_dv_mps"][1] for s in summaries])
    ax.barh(y, meds, height=0.62, color=colors, alpha=0.88, zorder=3)
    ax.errorbar(meds, y, xerr=[meds - q1, q3 - meds], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=3.5, zorder=4)
    ax.axvline(435.0, color="#b2182b", ls="--", lw=1.2, zorder=2, label="435 m/s divert budget")
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlim(0, 520)
    for yi, m, hi_q in zip(y, meds, q3):
        ax.text(max(m, hi_q) + 6.0, yi, f"{m:.0f}",
                va="center", ha="left", fontsize=8.3, color=INK, fontweight="semibold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")

    # Panel C: launches at threats that would have landed outside the defended area
    ax = axes[2]
    _style_ax(ax, f"c) False engagements (of {n_safe} safe)",
              "interceptors launched  [count]", "")
    fe = np.array([s["false_engages"] for s in summaries])
    ax.barh(y, fe, height=0.62, color=colors, alpha=0.88, zorder=3)
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlim(0, max(n_safe, 1) * 1.18)
    for yi, v in zip(y, fe):
        ax.text(v + 2.0, yi, f"{v}", va="center", ha="left", fontsize=8.3, color=INK, fontweight="semibold")

    fig.suptitle(
        f"Component-by-component GNC ablation ({n_runs} paired seeds, identical interceptor, "
        f"435 m/s KV divert budget){note}",
        x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def update_trade_study(summaries, by_cfg, n_runs):
    """Update results/gnc_trade_study.json and docs/figures/gnc_trade_study.png with the N-seed paired run."""
    s_base = next(s for s in summaries if s["config_id"] == "C1_base_cart")
    s_adv = next(s for s in summaries if s["config_id"] == "C9_adv_rae")

    trade_summary = {
        "base": {
            "label": "Baseline (KF6 + Vacuum PIP + PN)",
            "runs": n_runs,
            "engaged": s_base["engaged"],
            "kills": s_base["kills"],
            "kill_rate": s_base["kill_rate_engaged"],
            "ci95": s_base["ci95_engaged"],
            "median_dv_mps": s_base["median_dv_mps"],
            "mean_dv_mps": s_base["mean_dv_mps"],
            "threatening": s_base["threatening_total"],
            "protected": s_base["threatening_kills"],
            "protection_rate": s_base["kill_rate_threatening"],
            "ci95_protection": s_base["ci95_threatening"],
            "false_engages": s_base["false_engages"],
        },
        "adv": {
            "label": "Advanced (EKF7 + Drag PIP + APN)",
            "runs": n_runs,
            "engaged": s_adv["engaged"],
            "kills": s_adv["kills"],
            "kill_rate": s_adv["kill_rate_engaged"],
            "ci95": s_adv["ci95_engaged"],
            "median_dv_mps": s_adv["median_dv_mps"],
            "mean_dv_mps": s_adv["mean_dv_mps"],
            "threatening": s_adv["threatening_total"],
            "protected": s_adv["threatening_kills"],
            "protection_rate": s_adv["kill_rate_threatening"],
            "ci95_protection": s_adv["ci95_threatening"],
            "false_engages": s_adv["false_engages"],
        },
    }
    TRADE_JSON.write_bytes((json.dumps(trade_summary, indent=2) + "\n").encode("utf-8"))

    cfg_base = load_config(ROOT / "configs" / "engineering.yaml")
    cfg_adv = load_config(ROOT / "configs" / "advanced.yaml")
    tgo_bins = np.arange(0.0, 140.0, 5.0)
    tr_base = pip_bias(monte_carlo(cfg_base, range(min(n_runs, 15))), tgo_bins)
    tr_adv = pip_bias(monte_carlo(cfg_adv, range(min(n_runs, 15))), tgo_bins)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), facecolor=SURFACE)

    ax = axes[0]
    _style_ax(ax, f"a) Protection rate ({trade_summary['base']['threatening']} threatening threats)",
              "GNC architecture", "threatening threats killed  [%]")
    x = [0, 1]
    tb, ta = trade_summary["base"], trade_summary["adv"]
    rates = [tb["protection_rate"], ta["protection_rate"]]
    yerr = [
        [rates[0] - tb["ci95_protection"][0], rates[1] - ta["ci95_protection"][0]],
        [tb["ci95_protection"][1] - rates[0], ta["ci95_protection"][1] - rates[1]],
    ]
    bars = ax.bar(x, rates, width=0.48, color=[C_BASE, C_ADV], alpha=0.88, zorder=3)
    ax.errorbar(x, rates, yerr=yerr, fmt="none", ecolor=INK, elinewidth=1.5, capsize=5, zorder=4)
    ax.set_xticks(x, ["Baseline\n(KF6 + Vac PIP + PN)", "Advanced\n(EKF7 + Drag PIP + APN)"])
    ax.set_ylim(0, 108)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 4.5, f"{r:.1f}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)

    ax = axes[1]
    _style_ax(ax, "b) Kill-vehicle divert Δv consumption", "divert Δv spent  [m/s]", "engaged runs")
    dv_base = [r["divert_dv_used_mps"] for r in by_cfg["C1_base_cart"] if r["launched"] == 1]
    dv_adv = [r["divert_dv_used_mps"] for r in by_cfg["C9_adv_rae"] if r["launched"] == 1]
    bins = np.linspace(0, 450, 24)
    ax.hist(dv_base, bins=bins, color=C_BASE, alpha=0.55,
            label=f"Baseline (med {trade_summary['base']['median_dv_mps']:.0f} m/s)", zorder=2)
    ax.hist(dv_adv, bins=bins, color=C_ADV, alpha=0.65,
            label=f"Advanced (med {trade_summary['adv']['median_dv_mps']:.0f} m/s)", zorder=3)
    ax.axvline(435.0, color=INK, ls="--", lw=1.1, label="435 m/s divert budget")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")

    ax = axes[2]
    _style_ax(ax, "c) Predicted impact point (PIP) bias", "true time to impact  [s]", "median |PIP error|  [m]")
    tgo = tr_base.tgo
    med_b = np.nanmedian(np.hypot(tr_base.along, tr_base.cross), axis=1)
    med_a = np.nanmedian(np.hypot(tr_adv.along, tr_adv.cross), axis=1)
    valid = np.isfinite(med_b) & np.isfinite(med_a)
    ax.plot(tgo[valid], med_b[valid], color=C_BASE, lw=2.0, label="Vacuum PIP (KF6)")
    ax.plot(tgo[valid], med_a[valid], color=C_ADV, lw=2.0, label="Drag-integrated PIP (EKF7)")
    ax.set_yscale("log")
    ax.set_xlim(tgo[valid].max() + 2, 0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")

    fig.suptitle(f"Closing the GNC model-mismatch loop ({n_runs} paired seeds, identical 435 m/s KV divert budget)",
                 x=0.01, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(TRADE_FIG, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Updated {TRADE_FIG} and {TRADE_JSON}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=1000, help="paired seeds 0..N-1 (default: 1000)")
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: CPU count)")
    parser.add_argument("--from-csv", action="store_true",
                        help="re-analyse results/ablation_1000.csv instead of re-running the engagements")
    parser.add_argument("--atmosphere", choices=("constant", "exponential"), default="constant",
                        help="density model for every configuration (default: constant)")
    args = parser.parse_args(argv)

    global OUT_CSV, OUT_JSON, OUT_FIG
    exponential = args.atmosphere == "exponential"
    if exponential:
        OUT_CSV = OUT_CSV.with_name("ablation_1000_exp.csv")
        OUT_JSON = OUT_JSON.with_name("ablation_summary_exp.json")
        OUT_FIG = OUT_FIG.with_name("ablation_study_exp.png")

    t0 = time.time()
    configs = build_configs(args.atmosphere)
    workers = args.workers or os.cpu_count() or 1

    with multiprocessing.Pool(workers) as pool:
        if args.from_csv:
            rows = load_rows(OUT_CSV)
            seeds = sorted({r["seed"] for r in rows})
            print(f"Loaded {len(rows)} rows ({len(seeds)} seeds) from {OUT_CSV}")
        else:
            seeds = list(range(args.runs))
            jobs = [(cfg_id, cfg, s) for cfg_id, _, _, cfg in configs for s in seeds]
            print(f"Running {len(jobs)} engagements ({len(configs)} configs x {len(seeds)} seeds) "
                  f"across {workers} workers...")
            rows = list(pool.imap(fast_run_metrics, jobs, chunksize=25))
            write_csv(OUT_CSV, rows, ABLATION_COLUMNS)
            print(f"Wrote {OUT_CSV} in {time.time() - t0:.1f} s")

        print(f"Computing true unintercepted impact points for {len(seeds)} seeds...")
        true_impacts = compute_true_impacts(seeds, cfg=configs[0][3], pool=pool)

    def_rad = configs[0][3].engagement.defended_radius
    n_threatening = sum(1 for r in true_impacts.values() if r <= def_rad)
    print(f"  Threatening scenarios (true impact <= {def_rad/1000:.0f} km): {n_threatening}/{len(seeds)}")

    by_cfg = {cfg_id: [] for cfg_id, _, _, _ in configs}
    for r in rows:
        by_cfg[r["config_id"]].append(r)

    summaries = []
    print("\n" + "=" * 118)
    print(f"{'Configuration':48s} {'Protect (95% CI)':>24s} {'Kill|Eng':>9s} {'Med dv (IQR)':>18s} {'FalseEng':>8s}")
    print("-" * 118)
    for cfg_id, short_label, full_label, cfg in configs:
        s = summarize_config(cfg_id, short_label, full_label, cfg, by_cfg[cfg_id], true_impacts)
        summaries.append(s)
        pr = f"{s['kill_rate_threatening']:5.1f}% [{s['ci95_threatening'][0]:4.1f}, {s['ci95_threatening'][1]:5.1f}]"
        dv_str = f"{s['median_dv_mps']:5.1f} ({s['iqr_dv_mps'][0]:.0f}-{s['iqr_dv_mps'][1]:.0f})"
        print(f"{full_label:48s} {pr:>24s} {s['kill_rate_engaged']:8.1f}% {dv_str:>18s} {s['false_engages']:8d}")
    print("=" * 118 + "\n")

    contrasts = []
    print(f"{'One-factor contrast':52s} {'dProtect':>9s} {'b/c':>9s} {'McNemar p':>10s} {'dFalseEng':>10s} {'med d(dv) [95% CI]':>26s}")
    for ref_id, cmp_id, what in CONTRASTS:
        c = paired_contrast(by_cfg[ref_id], by_cfg[cmp_id], true_impacts, def_rad)
        contrasts.append({"ref": ref_id, "cmp": cmp_id, "change": what, **c})
        ci = c["median_dv_diff_ci95"]
        print(f"{what:52s} {c['protection_diff_pts']:+8.1f} {c['discordant_only_ref']:4d}/{c['discordant_only_cmp']:<4d} "
              f"{c['mcnemar_p']:10.2e} {c['false_engage_cmp'] - c['false_engage_ref']:+10d} "
              f"{c['median_dv_diff_mps']:+9.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")
    print()

    out_data = {
        **({"atmosphere": "exponential", "scale_height_m": configs[0][3].sim.scale_height} if exponential else {}),
        "runs_per_config": len(seeds),
        "threatening_scenarios": n_threatening,
        "non_threatening_scenarios": len(seeds) - n_threatening,
        "metric_note": ("kill_rate_threatening (protection rate) shares one denominator across configs; "
                        "kill_rate_engaged does not, because each config makes its own launch decisions"),
        "configs": summaries,
        "contrasts": contrasts,
    }
    OUT_JSON.write_bytes((json.dumps(out_data, indent=2) + "\n").encode("utf-8"))
    print(f"Wrote {OUT_JSON}")

    if exponential:
        h_km = configs[0][3].sim.scale_height / 1000.0
        plot_ablation(summaries, by_cfg, len(seeds), OUT_FIG,
                      note=f"\nExponential atmosphere, scale height {h_km:g} km; drag coefficients are sea-level values")
    else:
        plot_ablation(summaries, by_cfg, len(seeds), OUT_FIG)
        update_trade_study(summaries, by_cfg, len(seeds))


if __name__ == "__main__":
    main()
