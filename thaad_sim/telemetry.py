"""Complete GNC telemetry computation in Python (single source of truth).

Takes the raw simulation history ``hist`` from ``thaad_sim.engagement.run()`` and
computes all derived kinematic, guidance, Kalman filter, PIP, and trajectory
series in vectorized NumPy so the web workstation needs zero duplicated physics.
"""
from __future__ import annotations

import numpy as np

from .estimation import predict_impact_drag
from .guidance import G0, PN_MIN_RANGE, guidance_command, limit_g
from .replay_viewer import build_payload


def _integrate_true_impact(pos0, vel0, drag, g, dt=0.05, max_steps=4000):
    """Integrate the threat's final state to z = 0 under quadratic drag using RK4."""
    x, y, z = float(pos0[0]), float(pos0[1]), float(pos0[2])
    vx, vy, vz = float(vel0[0]), float(vel0[1]), float(vel0[2])
    if z <= 0.0:
        return [round(x, 1), round(y, 1), 0.0]
    pip, _ = predict_impact_drag(np.array([x, y, z]), np.array([vx, vy, vz]), drag, g, dt_step=0.25, max_steps=max_steps)
    if pip is not None:
        return [round(float(pip[0]), 1), round(float(pip[1]), 1), 0.0]
    return [round(x, 1), round(y, 1), 0.0]


def _rlist(arr, decimals=2):
    """Convert a 1D numpy array to a JSON-friendly list with NaNs mapped to None."""
    a = np.asarray(arr, dtype=float)
    rounded = np.round(a, decimals)
    return [None if not np.isfinite(v) else float(v) for v in rounded]


def build_workstation_telemetry(hist):
    """Build the full enriched telemetry dictionary for the THAAD Workstation."""
    base = build_payload(hist)
    meta = hist["meta"]
    prm = base.get("params", {})

    t = np.asarray(hist["t"], dtype=float)
    N = len(t)
    dt_sim = float(prm.get("dt", 0.02))
    g_acc = float(prm.get("g", 9.81))
    threat_drag = float(prm.get("threat_drag", 0.00004))
    pn_gain = float(prm.get("pn_gain", 4.0))
    max_g = float(base.get("maxG", prm.get("max_g", 30.0)))
    divert_budget = prm.get("kv_divert_dv")
    mass_wet = float(prm.get("mass_wet", 900.0))
    mass_dry = float(prm.get("mass_dry", 450.0))
    kv_mass = float(prm.get("kv_mass", 200.0))
    burn_time = float(prm.get("burn_time", 5.0))
    mass_flow = float(prm.get("mass_flow", (mass_wet - mass_dry) / max(burn_time, 1e-6)))
    guidance_law = base.get("guidanceLaw", "pn")
    use_drag_pip = base.get("pipMode") == "drag"

    t_pos = np.asarray(hist["t_pos"], dtype=float)       # (N, 3) ENU m
    i_pos = np.asarray(hist["i_pos"], dtype=float)       # (N, 3) ENU m
    k_pos = np.asarray(hist["k_pos"], dtype=float)       # (N, 3) ENU m
    raw_pos = np.asarray(hist["raw"], dtype=float)       # (N, 3) ENU m
    t_vel = np.asarray(hist["t_vel"], dtype=float)       # (N, 3) ENU m/s
    i_vel = np.asarray(hist["i_vel"], dtype=float)       # (N, 3) ENU m/s
    kf_err = np.asarray(hist["kf_err"], dtype=float)     # (N, 6)
    kf_sig = np.asarray(hist["kf_sig"], dtype=float)     # (N, 6)
    status = np.asarray(base["status"], dtype=int)       # 0=SEARCH, 1=FLYING, 2=HIT
    stage = np.asarray(base["stage"], dtype=int)         # 0=IDLE, 1=BOOST, 2=KV_COAST
    g_cmd = np.asarray(base["g"], dtype=float)           # (N,) achieved g-load

    est_drag_arr = (
        np.asarray(meta.get("est_drag", [0.0] * N), dtype=float)
        if meta.get("est_drag") is not None and len(meta.get("est_drag", [])) == N
        else np.zeros(N, dtype=float)
    )

    # Key event indices and timestamps
    launch_indices = np.where(status > 0)[0]
    launch_idx = int(launch_indices[0]) if len(launch_indices) > 0 else -1
    launch_t = float(t[launch_idx]) if launch_idx >= 0 else None

    burn_indices = np.where(stage == 2)[0]
    burn_idx = int(burn_indices[0]) if len(burn_indices) > 0 else -1
    burn_t = float(t[burn_idx]) if burn_idx >= 0 else None

    kill_t = float(base["kill"]["time"]) if base.get("kill") else None
    impact_t = float(t[-1]) if base["outcome"] in ("GROUND", "NO ENGAGE") else None
    end_event_t = kill_t if kill_t is not None else impact_t

    true_impact = _integrate_true_impact(t_pos[-1], t_vel[-1], threat_drag, g_acc)

    # Time-series arrays
    alt_t = t_pos[:, 2] / 1000.0
    spd_t = np.linalg.norm(t_vel, axis=1)
    rng_t = np.hypot(t_pos[:, 0], t_pos[:, 1]) / 1000.0

    alt_i = np.full(N, np.nan)
    spd_i = np.full(N, np.nan)
    rng_i = np.full(N, np.nan)
    sep = np.full(N, np.nan)
    vc = np.full(N, np.nan)
    ttg = np.full(N, np.nan)
    los_true = np.full(N, np.nan)
    los_est = np.full(N, np.nan)
    zem_true = np.full(N, np.nan)
    zem_est = np.full(N, np.nan)
    zem_vec = np.zeros((N, 3), dtype=float)
    ac_vec = np.zeros((N, 3), dtype=float)
    dv_used = np.zeros(N, dtype=float)
    mass = np.full(N, mass_wet, dtype=float)

    pip_track = np.zeros((N, 3), dtype=float)
    pip_valid = np.zeros(N, dtype=int)
    pip_sig_e = np.zeros(N, dtype=float)
    pip_sig_n = np.zeros(N, dtype=float)
    pip_bias = np.full(N, np.nan)
    pip_vac_bias = np.full(N, np.nan)

    epos = np.linalg.norm(kf_err[:, :3], axis=1)
    evel = np.linalg.norm(kf_err[:, 3:6], axis=1)
    spos3 = 3.0 * np.linalg.norm(kf_sig[:, :3], axis=1)
    svel3 = 3.0 * np.linalg.norm(kf_sig[:, 3:6], axis=1)

    # Raw radar error (logged truth is 1 step after measurement)
    prev_pos = t_pos - t_vel * dt_sim
    eraw = np.linalg.norm(raw_pos - prev_pos, axis=1)
    if N > 0 and status[-1] == 2:
        eraw[-1] = np.nan

    terminal_t = None
    dv_acc = 0.0
    cached_drag_pip = None

    for i in range(N):
        tx, ty, tz = t_pos[i]
        tvx, tvy, tvz = t_vel[i]
        ex, ey, ez, evx, evy, evz = kf_err[i]
        sx, sy, sz, svx, svy, svz = kf_sig[i]

        epx, epy, epz = tx + ex, ty + ey, tz + ez
        epvx, epvy, epvz = tvx + evx, tvy + evy, tvz + evz

        # Instantaneous PIP (vacuum + optional drag-integrated)
        if epvz < 0.0 and epz > 0.0:
            disc = epvz * epvz + 2.0 * g_acc * epz
            root = float(np.sqrt(max(0.0, disc)))
            t_imp = (epvz + root) / g_acc
            if t_imp > 0.0:
                px_vac = epx + epvx * t_imp
                py_vac = epy + epvy * t_imp
                pip_vac_bias[i] = float(np.hypot(px_vac - true_impact[0], py_vac - true_impact[1]))
                px, py = px_vac, py_vac
                if use_drag_pip:
                    if cached_drag_pip is None or i % 10 == 0:
                        kd = float(est_drag_arr[i]) if base.get("filterType") == "ekf7" else threat_drag
                        d_pip, _ = predict_impact_drag(
                            np.array([epx, epy, epz]),
                            np.array([epvx, epvy, epvz]),
                            kd,
                            g_acc,
                        )
                        if d_pip is not None:
                            cached_drag_pip = (float(d_pip[0]), float(d_pip[1]))
                    if cached_drag_pip is not None:
                        px, py = cached_drag_pip
                pip_track[i] = [round(px, 1), round(py, 1), 0.0]
                pip_valid[i] = 1
                pip_bias[i] = float(np.hypot(px - true_impact[0], py - true_impact[1]))

                dtdz = 1.0 / max(root, 1.0)
                dtdvz = t_imp / max(root, 1.0)
                sig_x = float(np.hypot(np.hypot(sx, t_imp * svx), np.hypot(epvx * dtdz * sz, epvx * dtdvz * svz)))
                sig_y = float(np.hypot(np.hypot(sy, t_imp * svy), np.hypot(epvy * dtdz * sz, epvy * dtdvz * svz)))
                pip_sig_e[i] = round(3.0 * sig_x, 1)
                pip_sig_n[i] = round(3.0 * sig_y, 1)

        if not pip_valid[i] and i > 0:
            pip_track[i] = pip_track[i - 1]
            pip_sig_e[i] = pip_sig_e[i - 1]
            pip_sig_n[i] = pip_sig_n[i - 1]

        # Mass schedule
        stg = stage[i]
        if status[i] == 0 or launch_t is None:
            mass[i] = mass_wet
        elif stg == 1:
            mass[i] = max(mass_dry, mass_wet - mass_flow * max(0.0, t[i] - launch_t))
        else:
            mass[i] = kv_mass

        if status[i] > 0:
            step_dt = float(t[i] - t[i - 1]) if i > 0 else dt_sim
            if stg == 2:
                dv_acc += g_cmd[i] * G0 * step_dt
            dv_used[i] = dv_acc

            ix, iy, iz = i_pos[i]
            ivx, ivy, ivz = i_vel[i]
            alt_i[i] = iz / 1000.0
            spd_i[i] = float(np.hypot(ivx, np.hypot(ivy, ivz)))
            rng_i[i] = float(np.hypot(ix, iy)) / 1000.0

            # True relative kinematics
            r_vec = t_pos[i] - i_pos[i]
            v_vec = t_vel[i] - i_vel[i]
            r_mag = max(float(np.linalg.norm(r_vec)), 1e-6)
            sep[i] = r_mag / 1000.0
            if terminal_t is None and stg == 2 and sep[i] < 15.0:
                terminal_t = float(t[i])

            rv = float(np.dot(r_vec, v_vec))
            vv = float(np.dot(v_vec, v_vec))
            vc[i] = -rv / r_mag
            if vc[i] > 0:
                ttg[i] = r_mag / vc[i]

            omega_true = np.cross(r_vec, v_vec) / (r_mag * r_mag)
            los_true[i] = 1000.0 * float(np.linalg.norm(omega_true))
            tgo_true = max(0.0, -rv / vv) if vv > 1e-9 else 0.0
            z_true = r_vec + v_vec * tgo_true
            zem_true[i] = float(np.linalg.norm(z_true))
            zem_vec[i] = np.round(z_true, 1)

            # Filter-perceived relative state
            er_vec = np.array([epx - ix, epy - iy, epz - iz])
            ev_vec = np.array([epvx - ivx, epvy - ivy, epvz - ivz])
            er_mag = max(float(np.linalg.norm(er_vec)), 1e-6)
            erv = float(np.dot(er_vec, ev_vec))
            evv = float(np.dot(ev_vec, ev_vec))
            omega_est = np.cross(er_vec, ev_vec) / (er_mag * er_mag + 1e-6)
            los_est[i] = 1000.0 * float(np.linalg.norm(omega_est))
            tgo_est = max(0.0, -erv / evv) if evv > 1e-9 else 0.0
            zem_est[i] = float(np.linalg.norm(er_vec + ev_vec * tgo_est))

            if er_mag >= PN_MIN_RANGE:
                ep_vel = np.array([epvx, epvy, epvz])
                est_t_acc = -float(est_drag_arr[i]) * float(np.linalg.norm(ep_vel)) * ep_vel
                raw_cmd = guidance_command(
                    guidance_law, er_vec, ev_vec, pn_gain, g_acc,
                    target_accel=est_t_acc if base.get("filterType") == "ekf7" else None,
                    boost=(stg == 1),
                )
                cmd_lim, cmd_mag = limit_g(raw_cmd, max_g)
                actual_mag = g_cmd[i] * G0
                scale = (actual_mag / cmd_mag) if cmd_mag > 1e-6 else 0.0
                ac_vec[i] = np.round(cmd_lim * scale, 2)

    # Filter post-transient statistics (t >= t[0] + 2.0 s)
    post = t >= (t[0] + 2.0)
    kf_stats = None
    if np.any(post):
        ep_post = epos[post]
        ev_post = evel[post]
        er_post = eraw[post & np.isfinite(eraw)]
        ex_p, ey_p, ez_p = np.abs(kf_err[post, 0]), np.abs(kf_err[post, 1]), np.abs(kf_err[post, 2])
        sx_p, sy_p, sz_p = 3.0 * kf_sig[post, 0], 3.0 * kf_sig[post, 1], 3.0 * kf_sig[post, 2]
        inside = np.mean((ex_p <= sx_p) & (ey_p <= sy_p) & (ez_p <= sz_p)) * 100.0
        kf_stats = {
            "rmsPos": round(float(np.sqrt(np.mean(ep_post ** 2))), 2),
            "rmsVel": round(float(np.sqrt(np.mean(ev_post ** 2))), 2),
            "rmsRaw": round(float(np.sqrt(np.mean(er_post ** 2))), 2) if len(er_post) else None,
            "inside3SigmaPct": round(float(inside), 1),
            "samples": int(np.sum(post)),
        }

    # Build timeline events & phases
    events = [{"t": round(float(t[0]), 2), "label": "Radar track", "detail": "Threat acquired", "type": "info"}]
    if base["outcome"] == "NO ENGAGE":
        hold_t = float(base.get("minTrackTime") or t[0])
        events.append({"t": round(hold_t, 2), "label": "Weapons hold", "detail": "PIP outside defended perimeter", "type": "info"})
    if launch_t is not None:
        events.append({
            "t": round(launch_t, 2),
            "label": "Launch",
            "detail": "PIP inside defended area" if base["engageRule"] == "PIP-GATED LAUNCH" else "Track confirmed — weapons free",
            "type": "warn",
        })
    if burn_t is not None:
        events.append({"t": round(burn_t, 2), "label": "Burnout, KV separation", "detail": "Kill vehicle enters midcourse coast", "type": "kv"})
    if terminal_t is not None:
        events.append({"t": round(terminal_t, 2), "label": "Terminal homing", "detail": "Kill vehicle within 15 km of the threat", "type": "kv"})
    if divert_budget is not None:
        dv_ex_idx = np.where(dv_used >= float(divert_budget) - 0.5)[0]
        if len(dv_ex_idx) > 0:
            dv_ex_t = float(t[dv_ex_idx[0]])
            if kill_t is None or dv_ex_t < kill_t:
                events.append({"t": round(dv_ex_t, 2), "label": "Divert Δv exhausted", "detail": f"Budget {divert_budget:.0f} m/s spent", "type": "bad"})
    cpa_info = base.get("cpa")
    if kill_t is None and cpa_info is not None:
        events.append({
            "t": round(float(cpa_info["time"]), 2),
            "label": "Closest approach",
            "detail": f"Miss distance {cpa_info['dist']:.2f} m at {cpa_info['alt']/1000:.1f} km alt",
            "type": "bad",
        })
    if kill_t is not None:
        events.append({
            "t": round(kill_t, 2),
            "label": "Hit-to-kill",
            "detail": f"Miss {base['kill']['dist']:.2f} m · Closing {base['kill']['closing']:.0f} m/s · Alt {base['kill']['alt']/1000:.1f} km",
            "type": "ok",
        })
    if base["outcome"] == "GROUND" and impact_t is not None:
        events.append({"t": round(impact_t, 2), "label": "Ground impact", "detail": "Intercept failed — threat reached ground", "type": "bad"})
    elif base["outcome"] == "NO ENGAGE" and impact_t is not None:
        events.append({"t": round(impact_t, 2), "label": "Ground impact, outside defended area", "detail": "Threat landed outside defended area", "type": "info"})
    events.sort(key=lambda e: e["t"])

    base["derived"] = {
        "launchIdx": launch_idx,
        "launchT": launch_t,
        "burnIdx": burn_idx,
        "burnT": burn_t,
        "terminalT": terminal_t,
        "killT": kill_t,
        "impactT": impact_t,
        "endEventT": end_event_t,
        "trueImpact": true_impact,
        "altT": _rlist(alt_t, 3),
        "spdT": _rlist(spd_t, 1),
        "rngT": _rlist(rng_t, 3),
        "altI": _rlist(alt_i, 3),
        "spdI": _rlist(spd_i, 1),
        "rngI": _rlist(rng_i, 3),
        "sep": _rlist(sep, 4),
        "vc": _rlist(vc, 1),
        "ttg": _rlist(ttg, 2),
        "losTrue": _rlist(los_true, 3),
        "losEst": _rlist(los_est, 3),
        "zemTrue": _rlist(zem_true, 2),
        "zemEst": _rlist(zem_est, 2),
        "zemVec": zem_vec.ravel().tolist(),
        "acVec": ac_vec.ravel().tolist(),
        "dvUsed": _rlist(dv_used, 1),
        "mass": _rlist(mass, 1),
        "pipTrack": pip_track.ravel().tolist(),
        "pipValid": pip_valid.tolist(),
        "pipSigE": _rlist(pip_sig_e, 1),
        "pipSigN": _rlist(pip_sig_n, 1),
        "pipBias": _rlist(pip_bias, 1),
        "pipVacBias": _rlist(pip_vac_bias, 1),
        "epos": _rlist(epos, 2),
        "evel": _rlist(evel, 2),
        "spos3": _rlist(spos3, 2),
        "svel3": _rlist(svel3, 2),
        "eraw": _rlist(eraw, 2),
        "ex": _rlist(kf_err[:, 0], 2),
        "ey": _rlist(kf_err[:, 1], 2),
        "ez": _rlist(kf_err[:, 2], 2),
        "sx3": _rlist(3.0 * kf_sig[:, 0], 2),
        "sy3": _rlist(3.0 * kf_sig[:, 1], 2),
        "sz3": _rlist(3.0 * kf_sig[:, 2], 2),
        "kfStats": kf_stats,
        "events": events,
    }
    return base

