"""Exponential atmosphere (sim.atmosphere: exponential).

The drag coefficients become sea-level values scaled by exp(-z / H). These
tests check the density function, that a very large scale height reproduces
the constant-density model, the EKF's extra Jacobian column, the drag impact
predictor against a fine-step truth integration, and that the filter, the
truth and the ablation script's fast loop all use the same atmosphere.
"""
import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from thaad_sim import load_config
from thaad_sim.atmosphere import density_ratio
from thaad_sim.batch import run_metrics
from thaad_sim.config import config_from_dict
from thaad_sim.dynamics import BallisticThreat, Interceptor
from thaad_sim.estimation import (
    ExtendedKalmanFilter,
    ekf7_derivative,
    ekf7_jacobian,
    make_filter,
    predict_impact_drag,
)
from thaad_sim.scenario import spawn_threat

from conftest import ROOT

G = 9.81
H = 7000.0
POS0 = np.array([40000.0, -15000.0, 60000.0])
VEL0 = np.array([-480.0, 180.0, -520.0])


def exponential(cfg, scale_height=H):
    return replace(cfg, sim=replace(cfg.sim, atmosphere="exponential", scale_height=scale_height))


def fly_to_ground(threat, dt=0.02):
    t = 0.0
    while threat.active:
        tg = threat.update(dt)
        t += dt if tg is None else tg
    return t


def test_density_ratio():
    assert density_ratio(12345.0) == 1.0                      # constant density
    assert density_ratio(0.0, H) == 1.0
    assert density_ratio(H, H) == pytest.approx(np.exp(-1.0))
    assert density_ratio(70000.0, H) == pytest.approx(np.exp(-10.0))
    assert density_ratio(-50.0, H) == 1.0                     # below ground counts as z = 0


def test_config_default_is_constant_density():
    cfg = load_config(ROOT / "configs" / "engineering.yaml")
    assert cfg.sim.atmosphere == "constant"
    assert cfg.sim.density_scale_height is None
    assert "scale_height" not in cfg.params()
    exp_cfg = exponential(cfg)
    assert exp_cfg.sim.density_scale_height == H
    assert exp_cfg.params()["scale_height"] == H


def test_config_rejects_bad_atmosphere():
    import yaml
    data = yaml.safe_load((ROOT / "configs" / "engineering.yaml").read_text(encoding="utf-8"))
    data["sim"]["atmosphere"] = "isothermal"
    with pytest.raises(ValueError, match="sim.atmosphere"):
        config_from_dict(data)
    data["sim"]["atmosphere"] = "exponential"
    data["sim"]["scale_height"] = 0.0
    with pytest.raises(ValueError, match="scale_height"):
        config_from_dict(data)


def test_huge_scale_height_reproduces_constant_density():
    drag = 4.0e-5
    const = BallisticThreat(POS0.copy(), VEL0.copy(), drag, G)
    huge = BallisticThreat(POS0.copy(), VEL0.copy(), drag, G, scale_height=1e15)
    fly_to_ground(const)
    fly_to_ground(huge)
    assert np.allclose(huge.state, const.state, rtol=0, atol=1e-6)

    pip_c, tgo_c = predict_impact_drag(POS0, VEL0, drag, G)
    pip_h, tgo_h = predict_impact_drag(POS0, VEL0, drag, G, scale_height=1e15)
    assert np.allclose(pip_h, pip_c, rtol=0, atol=1e-6)
    assert tgo_h == pytest.approx(tgo_c, abs=1e-6)


def test_exponential_drag_is_weak_at_altitude_and_full_at_sea_level():
    drag = 4.0e-5
    threat = BallisticThreat(POS0.copy(), VEL0.copy(), drag, G, scale_height=H)
    speed = np.linalg.norm(VEL0)
    high = threat.derivative(np.hstack((POS0, VEL0)))[3:] + [0.0, 0.0, G]
    low = threat.derivative(np.hstack(([0.0, 0.0, 0.0], VEL0)))[3:] + [0.0, 0.0, G]
    assert np.allclose(low, -drag * speed * VEL0)
    assert np.allclose(high, low * np.exp(-POS0[2] / H))


def test_interceptor_drag_uses_the_same_atmosphere():
    spec = load_config(ROOT / "configs" / "engineering.yaml").interceptor
    icp = Interceptor([0.0, 0.0, 0.0], spec, G, scale_height=H)
    vel = np.array([300.0, 0.0, 900.0])
    zero = np.zeros(3)
    deriv = icp._derivative(zero, zero, spec.kv_drag, 0.0)
    for z in (0.0, 20000.0):
        acc = deriv(np.hstack(([0.0, 0.0, z], vel, [spec.kv_mass])))[3:6] + [0.0, 0.0, G]
        expected = -spec.kv_drag * np.exp(-z / H) * np.linalg.norm(vel) * vel / spec.kv_mass
        assert np.allclose(acc, expected)


@pytest.mark.parametrize("z", [4000.0, 20000.0, 52000.0])
def test_ekf7_jacobian_matches_finite_differences_in_exponential_atmosphere(z):
    x7 = np.array([25000.0, -18000.0, z, -420.0, 310.0, -680.0, 4.2e-5])
    F_ana = ekf7_jacobian(x7, H)
    F_num = np.zeros((7, 7))
    for j in range(7):
        eps = 1e-3 if j < 6 else 1e-10
        dx = np.zeros(7)
        dx[j] = eps
        F_num[:, j] = (ekf7_derivative(x7 + dx, G, H) - ekf7_derivative(x7 - dx, G, H)) / (2 * eps)
    assert np.allclose(F_ana, F_num, rtol=1e-6, atol=1e-9)
    # The altitude column exists only because density depends on z
    assert np.any(F_ana[3:6, 2] != 0.0)
    assert np.all(ekf7_jacobian(x7)[3:6, 2] == 0.0)


@pytest.mark.parametrize("drag", [4.0e-5, 2.0e-4])
def test_drag_predictor_matches_truth_in_exponential_atmosphere(drag):
    threat = BallisticThreat(POS0.copy(), VEL0.copy(), drag, G, scale_height=H)
    t_true = fly_to_ground(threat)
    pip, tgo = predict_impact_drag(POS0, VEL0, drag, G, scale_height=H)
    # The predictor's default 2 s step has to stay accurate through the dense lower air
    assert np.linalg.norm(pip[:2] - threat.pos[:2]) < 2.0
    assert tgo == pytest.approx(t_true, abs=0.01)
    # And the vacuum prediction is no longer kilometres off, as the air is thin for most of the fall
    pip_vac, _ = predict_impact_drag(POS0, VEL0, 0.0, G)
    pip_const, _ = predict_impact_drag(POS0, VEL0, drag, G)
    assert (np.linalg.norm(pip_vac[:2] - threat.pos[:2])
            < np.linalg.norm(pip_const[:2] - threat.pos[:2]))


def test_ekf7_feed_forward_matches_truth_drag():
    drag = 4.0e-5
    ekf = ExtendedKalmanFilter(0.02, 20.0, 6.0, 5.0, G, drag_init=drag, scale_height=H)
    ekf.update(POS0, VEL0)   # the first measurement initialises the state
    truth = BallisticThreat(POS0.copy(), VEL0.copy(), drag, G, scale_height=H)
    true_drag_acc = truth.derivative(truth.state)[3:] + [0.0, 0.0, G]
    assert np.allclose(ekf.est_target_accel, true_drag_acc)


def test_ekf7_tracks_through_reentry_and_learns_sea_level_drag():
    dt, true_drag = 0.02, 4.0e-5
    threat = BallisticThreat([30000.0, 8000.0, 40000.0], [-600.0, -160.0, -900.0], true_drag, G,
                             scale_height=H)
    ekf = ExtendedKalmanFilter(dt, pos_std=20.0, vel_std=6.0, q=5.0, g=G,
                               drag_init=2.0e-5, drag_std=3.0e-5, q_drag=1e-12, scale_height=H)
    rng = np.random.default_rng(42)
    high_altitude_estimate = None
    while threat.active:
        ekf.update(threat.pos + rng.normal(0, 20.0, 3), threat.vel + rng.normal(0, 6.0, 3))
        np.linalg.cholesky(ekf.P7)
        if high_altitude_estimate is None and threat.pos[2] < 30000.0:
            high_altitude_estimate = ekf.est_drag
        threat.update(dt)
    # Above 30 km there is too little drag to observe, so the estimate has barely left its prior
    assert abs(high_altitude_estimate - 2.0e-5) < 0.5e-5
    # By the ground the dense air has made it observable
    assert abs(ekf.est_drag - true_drag) / true_drag < 0.15


def test_config_threads_the_atmosphere_to_truth_and_filter():
    cfg = exponential(load_config(ROOT / "configs" / "advanced.yaml"))
    threat = spawn_threat(cfg, np.random.default_rng(0)).threat
    assert threat.scale_height == H
    assert make_filter(cfg).scale_height == H
    base = load_config(ROOT / "configs" / "advanced.yaml")
    assert spawn_threat(base, np.random.default_rng(0)).threat.scale_height is None
    assert make_filter(base).scale_height is None


_spec = importlib.util.spec_from_file_location("ablation_study", ROOT / "scripts" / "ablation_study.py")
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)

COMPARED = ("outcome", "launched", "launch_time_s", "launch_pip_range_m", "kill_time_s",
            "miss_distance_m", "divert_dv_used_mps", "impact_range_m")


@pytest.mark.parametrize("cfg_id", ["C1_base_cart", "C9_adv_rae"])
def test_fast_loop_matches_batch_run_metrics_in_exponential_atmosphere(cfg_id):
    cfg = {cid: c for cid, _, _, c in ablation.build_configs("exponential")}[cfg_id]
    assert cfg.sim.atmosphere == "exponential"
    fast = ablation.fast_run_metrics((cfg_id, cfg, 13))
    ref = run_metrics(cfg, 13)
    for col in COMPARED:
        if isinstance(ref[col], float):
            assert fast[col] == pytest.approx(ref[col], rel=1e-12, abs=1e-9), col
        else:
            assert fast[col] == ref[col], col
    # The atmosphere changes the engagement: the constant-density run differs
    const = ablation.fast_run_metrics((cfg_id, ablation.with_atmosphere(cfg, "constant"), 13))
    assert const["miss_distance_m"] != fast["miss_distance_m"]
