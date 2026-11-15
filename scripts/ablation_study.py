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

Usage:
    py scripts/ablation_study.py --runs 1000
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
from thaad_sim.stats import wilson_interval  # noqa: E402

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


def build_configs():
    """Return ordered list of (config_id, short_label, full_label, SimConfig)."""
    base = load_config(ROOT / "configs" / "engineering.yaml")
    adv = load_config(ROOT / "configs" / "advanced.yaml")

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
    interceptor = Interceptor([0, 0, 0], cfg.interceptor, g, cfg.sim.integrator)
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


def compute_true_impacts(seeds):
    """Integrate the unintercepted threat to ground for each seed to get its true impact range."""
    base = load_config(ROOT / "configs" / "engineering.yaml")
    dt = base.sim.dt
    impacts = {}
    for s in seeds:
        rng = make_rng(s)
        threat = spawn_threat(base, rng).threat
        while threat.active:
            threat.update(dt)
        impacts[s] = float(np.linalg.norm(threat.pos[:2]))
    return impacts


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


def plot_ablation(summaries, by_cfg, n_runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), facecolor=SURFACE)

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

    # Panel A: Kill Rate Among Engaged Threats (with 95% Wilson CI)
    ax = axes[0]
    _style_ax(ax, "a) Intercept rate among engaged threats (95% Wilson CI)",
              "kill probability | engaged  [%]", "")
    rates = np.array([s["kill_rate_engaged"] for s in summaries])
    lo = np.array([s["ci95_engaged"][0] for s in summaries])
    hi = np.array([s["ci95_engaged"][1] for s in summaries])
    bars = ax.barh(y, rates, height=0.62, color=colors, alpha=0.88, zorder=3)
    ax.errorbar(rates, y, xerr=[rates - lo, hi - rates], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=3.5, zorder=4)
    ax.set_yticks(y, labels, fontsize=8.8, color=INK)
    ax.set_xlim(0, 114)
    for yi, s, r, h in zip(y, summaries, rates, hi):
        ax.text(h + 1.5, yi, f"{r:.1f}% ({s['kills']}/{s['engaged']})",
                va="center", ha="left", fontsize=8.3, color=INK, fontweight="semibold")

    # Panel B: Divert delta-v median + IQR across engaged runs
    ax = axes[1]
    _style_ax(ax, "b) Kill-vehicle divert Δv consumption (median & IQR)",
              "KV divert Δv used  [m/s]", "")
    meds = np.array([s["median_dv_mps"] for s in summaries])
    q1 = np.array([s["iqr_dv_mps"][0] for s in summaries])
    q3 = np.array([s["iqr_dv_mps"][1] for s in summaries])
    ax.barh(y, meds, height=0.62, color=colors, alpha=0.88, zorder=3)
    ax.errorbar(meds, y, xerr=[meds - q1, q3 - meds], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=3.5, zorder=4)
    ax.axvline(435.0, color="#b2182b", ls="--", lw=1.2, zorder=2, label="435 m/s divert budget")
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlim(0, 495)
    for yi, m, hi_q in zip(y, meds, q3):
        ax.text(max(m, hi_q) + 6.0, yi, f"{m:.1f} m/s",
                va="center", ha="left", fontsize=8.3, color=INK, fontweight="semibold")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")

    fig.suptitle(
        f"Component-by-component GNC ablation study ({n_runs} paired seeds, identical 435 m/s KV divert budget)",
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
        },
    }
    TRADE_JSON.write_text(json.dumps(trade_summary, indent=2) + "\n", encoding="utf-8")

    cfg_base = load_config(ROOT / "configs" / "engineering.yaml")
    cfg_adv = load_config(ROOT / "configs" / "advanced.yaml")
    tgo_bins = np.arange(0.0, 140.0, 5.0)
    tr_base = pip_bias(monte_carlo(cfg_base, range(min(n_runs, 15))), tgo_bins)
    tr_adv = pip_bias(monte_carlo(cfg_adv, range(min(n_runs, 15))), tgo_bins)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), facecolor=SURFACE)

    ax = axes[0]
    _style_ax(ax, "a) Intercept rate among engaged threats", "GNC architecture", "kill probability  [%]")
    x = [0, 1]
    rates = [trade_summary["base"]["kill_rate"], trade_summary["adv"]["kill_rate"]]
    yerr = [
        [rates[0] - trade_summary["base"]["ci95"][0], rates[1] - trade_summary["adv"]["ci95"][0]],
        [trade_summary["base"]["ci95"][1] - rates[0], trade_summary["adv"]["ci95"][1] - rates[1]],
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
    args = parser.parse_args(argv)

    t0 = time.time()
    configs = build_configs()
    seeds = list(range(args.runs))
    print(f"Computing true unintercepted impact points for {args.runs} seeds...")
    true_impacts = compute_true_impacts(seeds)
    n_threatening = sum(1 for r in true_impacts.values() if r <= 30000.0)
    print(f"  Threatening scenarios (true impact <= 30 km): {n_threatening}/{args.runs}")

    jobs = [(cfg_id, cfg, s) for cfg_id, _, _, cfg in configs for s in seeds]
    workers = min(args.workers or os.cpu_count() or 1, len(jobs))
    print(f"Running {len(jobs)} engagements ({len(configs)} configs x {args.runs} seeds) across {workers} workers...")

    with multiprocessing.Pool(workers) as pool:
        rows = list(pool.imap(fast_run_metrics, jobs, chunksize=25))

    write_csv(OUT_CSV, rows, ABLATION_COLUMNS)
    print(f"Wrote {OUT_CSV} in {time.time() - t0:.1f} s")

    by_cfg = {cfg_id: [] for cfg_id, _, _, _ in configs}
    for r in rows:
        by_cfg[r["config_id"]].append(r)

    summaries = []
    print("\n" + "=" * 112)
    print(f"{'Configuration':48s} {'Engaged':>7s} {'Kills':>7s} {'Kill|Eng (95% CI)':>22s} {'Med Δv (IQR)':>18s} {'FalseEng':>8s}")
    print("-" * 112)
    for cfg_id, short_label, full_label, cfg in configs:
        s = summarize_config(cfg_id, short_label, full_label, cfg, by_cfg[cfg_id], true_impacts)
        summaries.append(s)
        ci_str = f"{s['kill_rate_engaged']:5.1f}% [{s['ci95_engaged'][0]:4.1f}, {s['ci95_engaged'][1]:4.1f}]"
        dv_str = f"{s['median_dv_mps']:5.1f} ({s['iqr_dv_mps'][0]:.0f}-{s['iqr_dv_mps'][1]:.0f})"
        print(f"{full_label:48s} {s['engaged']:7d} {s['kills']:7d} {ci_str:>22s} {dv_str:>18s} {s['false_engages']:8d}")
    print("=" * 112 + "\n")

    out_data = {
        "runs_per_config": args.runs,
        "threatening_scenarios": n_threatening,
        "non_threatening_scenarios": args.runs - n_threatening,
        "configs": summaries,
    }
    OUT_JSON.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")

    plot_ablation(summaries, by_cfg, args.runs, OUT_FIG)
    update_trade_study(summaries, by_cfg, args.runs)


if __name__ == "__main__":
    main()
