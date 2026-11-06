"""Tests for the 5-pillar GNC upgrades:
- 7-state Extended Kalman Filter (EKF) analytical Jacobian & drag convergence
- Drag-integrated Predicted Impact Point (PIP)
- Range-Azimuth-Elevation (RAE) polar radar model & anisotropic 3D covariance
- Guidance law suite (PN, Augmented PN, Optimal ZEM) and 1st-order actuator lag
- End-to-end rescue of Seed 13 (divert-exhaustion miss in engineering.yaml -> kill in advanced.yaml)
"""
from dataclasses import replace

import numpy as np
import pytest
from rich.console import Console

from thaad_sim import load_config
from thaad_sim.dynamics import BallisticThreat, Interceptor
from thaad_sim.engagement import simulate
from thaad_sim.estimation import (
    ExtendedKalmanFilter,
    ekf7_derivative,
    ekf7_jacobian,
    measure_radar,
    predict_impact_drag,
    rae_basis,
)
from thaad_sim.guidance import apn_command, guidance_command, pn_command, zem_command

from conftest import ROOT


def test_ekf7_analytical_jacobian_matches_finite_differences():
    x7 = np.array([25000.0, -18000.0, 52000.0, -420.0, 310.0, -680.0, 4.2e-5])
    g = 9.81
    F_ana = ekf7_jacobian(x7)
    F_num = np.zeros((7, 7))
    for j in range(7):
        eps = 1e-6 if j < 6 else 1e-10
        dx = np.zeros(7)
        dx[j] = eps
        f_hi = ekf7_derivative(x7 + dx, g)
        f_lo = ekf7_derivative(x7 - dx, g)
        F_num[:, j] = (f_hi - f_lo) / (2 * eps)
    assert np.allclose(F_ana, F_num, rtol=1e-6, atol=1e-8)


def test_ekf7_converges_on_true_ballistic_drag_coefficient():
    dt, g = 0.02, 9.81
    true_drag = 4.0e-5
    threat = BallisticThreat([35000.0, 10000.0, 65000.0], [-500.0, -150.0, -600.0], true_drag, g)
    ekf = ExtendedKalmanFilter(dt, pos_std=20.0, vel_std=6.0, q=5.0, g=g,
                               drag_init=2.0e-5, drag_std=3.0e-5, q_drag=1e-12)
    rng = np.random.default_rng(42)
    for _ in range(1500):  # 30 s of tracking
        mp = threat.pos + rng.normal(0, 20.0, 3)
        mv = threat.vel + rng.normal(0, 6.0, 3)
        ekf.update(mp, mv)
        threat.update(dt)
        np.linalg.cholesky(ekf.P7)  # 7x7 covariance remains SPD throughout
    # Initial guess was 2.0e-5 (50% low); after 30 s it converges within 15% of 4.0e-5
    assert abs(ekf.est_drag - true_drag) / true_drag < 0.15


def test_drag_integrated_pip_eliminates_vacuum_overshoot():
    g, drag = 9.81, 4.0e-5
    pos0 = np.array([40000.0, -15000.0, 60000.0])
    vel0 = np.array([-480.0, 180.0, -520.0])
    # Propagate true dragged threat to ground at fine dt = 0.02 s
    threat = BallisticThreat(pos0.copy(), vel0.copy(), drag, g)
    t_true = 0.0
    while threat.active:
        tg = threat.update(0.02)
        t_true += 0.02 if tg is None else tg
    true_impact = threat.pos[:2].copy()

    pip_drag, tgo_drag = predict_impact_drag(pos0, vel0, drag, g)
    pip_vac, _ = predict_impact_drag(pos0, vel0, 0.0, g)

    err_drag = np.linalg.norm(pip_drag[:2] - true_impact)
    err_vac = np.linalg.norm(pip_vac[:2] - true_impact)
    assert err_vac > 1000.0          # vacuum PIP overshoots by > 1 km
    assert err_drag < 0.5            # RK4 drag PIP matches true impact to < 0.5 m
    assert abs(tgo_drag - t_true) < 0.01


def test_rae_polar_radar_produces_oriented_anisotropic_covariance():
    cfg = load_config(ROOT / "configs" / "advanced.yaml")
    pos = np.array([30000.0, 40000.0, 50000.0])   # r = 70.71 km
    vel = np.array([-400.0, -500.0, -600.0])
    rng = np.random.default_rng(7)
    mp, mv, R_meas = measure_radar(pos, vel, cfg.sensor, rng)

    assert R_meas.shape == (6, 6)
    assert np.allclose(R_meas, R_meas.T)
    np.linalg.cholesky(R_meas)

    R_p = R_meas[:3, :3]
    _, _, _, u_r, u_az, u_el = rae_basis(mp)
    var_radial = float(u_r @ R_p @ u_r)
    var_cross = float(u_el @ R_p @ u_el)
    assert np.sqrt(var_radial) == pytest.approx(cfg.sensor.range_noise, rel=1e-6)
    assert np.sqrt(var_cross) > 5.0 * np.sqrt(var_radial)   # pancake ellipsoid wide across beam


def test_guidance_laws_agree_in_vacuum_and_differ_under_target_deceleration():
    r = np.array([12000.0, -4000.0, 18000.0])
    v = np.array([-600.0, 190.0, -920.0])
    g = 9.81
    a_pn = pn_command(r, v, 4.0, g)
    a_apn0 = apn_command(r, v, 4.0, g, target_accel=np.zeros(3), boost=True)
    a_zem0 = zem_command(r, v, 4.0, g, target_accel=np.zeros(3), boost=True)
    assert np.allclose(a_pn, a_apn0)
    # ZEM and classical PN commands point in nearly the same direction in drag-free closing flight
    cos_sim = np.dot(a_pn - [0, 0, g], a_zem0 - [0, 0, g]) / (
        np.linalg.norm(a_pn - [0, 0, g]) * np.linalg.norm(a_zem0 - [0, 0, g])
    )
    assert cos_sim > 0.96

    # With non-collinear target drag acceleration, APN adds a feed-forward term perpendicular to LOS
    a_targ = np.array([25.0, 15.0, -10.0])
    a_apn_drag = guidance_command("apn", r, v, 4.0, g, target_accel=a_targ, boost=True)
    assert not np.allclose(a_pn, a_apn_drag)


def test_actuator_lag_smooths_step_command():
    cfg = load_config(ROOT / "configs" / "engineering.yaml")
    spec_lag = replace(cfg.interceptor, tau=0.1)
    icp = Interceptor([0.0, 0.0, 0.0], spec_lag, 9.81)
    icp.launch(0.0, np.array([0.0, 0.0, 1.0]))
    icp.update(0.0, np.array([10000.0, 5000.0, 20000.0]), np.array([-300.0, -200.0, -500.0]), 0.02)
    # After one step dt = 0.02 s with tau = 0.1 s, actuator reaches (1 - exp(-0.2)) ~ 18.1 % of command
    assert 0 < icp.g_hist[0] < cfg.interceptor.max_g


def test_advanced_gnc_rescues_seed_13_and_saves_divert_dv():
    quiet = Console(quiet=True)
    eng_cfg = load_config(ROOT / "configs" / "engineering.yaml")
    adv_cfg = load_config(ROOT / "configs" / "advanced.yaml")

    # Seed 13 in baseline engineering mode exhausts all 435 m/s of divert and misses (16.7 m CPA)
    h_eng = simulate(eng_cfg, 13, console=quiet)
    assert h_eng['meta']['outcome'] == "GROUND"
    assert h_eng['meta']['divert_dv_used'] == pytest.approx(435.0, abs=1.0)

    # With 7-state EKF + drag PIP + RAE radar + APN guidance, Seed 13 is intercepted with divert to spare
    h_adv = simulate(adv_cfg, 13, console=quiet)
    assert h_adv['meta']['outcome'] == "KILL"
    assert h_adv['meta']['divert_dv_used'] < 400.0
