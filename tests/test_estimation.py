"""The Kalman filter: model structure, numerics and statistical consistency."""
import numpy as np
import pytest

from thaad_sim import load_config
from thaad_sim.consistency import (
    MEAS_DIM, STATE_DIM, average_bounds, average_statistics, chi2_quantile, fraction_inside,
    monte_carlo, pip_bias, track_run, with_filter_q, without_drag,
)
from thaad_sim.estimation import KalmanFilter, cwna_process_noise
from thaad_sim.scenario import spawn_threat
from thaad_sim.verification import ballistic_closed_form

from conftest import CONFIGS

DT, G = 0.02, 9.81


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIGS["engineering"])


def _filter(q=1.0, pos_std=25.0, vel_std=8.0):
    return KalmanFilter(DT, pos_std, vel_std, q, G)


# ---- model structure ----

def test_cwna_process_noise_matches_the_discretized_integral():
    q, dt = 3.0, 0.5
    Q = cwna_process_noise(q, dt)
    # Q = int_0^dt Phi(s) G q G^T Phi(s)^T ds with G = [0, I]: one 2x2 block per axis
    for i in range(3):
        block = Q[np.ix_([i, i + 3], [i, i + 3])]
        assert np.allclose(block, q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]))
    # Axes are independent
    assert np.count_nonzero(Q) == 12
    assert np.all(np.linalg.eigvalsh(Q) > 0)


def test_gravity_control_input_propagates_a_ballistic_state_exactly():
    kf = _filter()
    p0, v0 = np.array([1000.0, -2000.0, 60000.0]), np.array([300.0, 100.0, -500.0])
    x = np.hstack((p0, v0))
    n = 500
    for _ in range(n):
        x = kf.F @ x + kf.B @ kf.u
    p, v = ballistic_closed_form(p0, v0, G, [n * DT])
    assert np.allclose(x[:3], p[0], rtol=0, atol=1e-6)
    assert np.allclose(x[3:], v[0], rtol=0, atol=1e-9)


def test_gain_and_joseph_update_match_the_textbook_forms():
    kf = _filter()
    rng = np.random.default_rng(0)
    kf.update(rng.normal(size=3), rng.normal(size=3))
    P_prior = kf.F @ kf.P @ kf.F.T + kf.Q
    S = P_prior + kf.R
    K_inv = P_prior @ np.linalg.inv(S)
    kf.update(rng.normal(size=3), rng.normal(size=3))
    # With the optimal gain, the Joseph form equals the short form (I - K H) P
    assert np.allclose(kf.P, (np.eye(6) - K_inv) @ P_prior, rtol=1e-10, atol=1e-10)


def test_first_measurement_initializes_with_covariance_r():
    kf = _filter()
    kf.update(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert np.array_equal(kf.x, [1, 2, 3, 4, 5, 6])
    assert np.array_equal(kf.P, kf.R)
    assert kf.nis is None


# ---- covariance health ----

@pytest.mark.parametrize("q", [0.0, 1.0, 1e4])
def test_covariance_stays_symmetric_positive_definite(cfg, q):
    rng = np.random.default_rng(7)
    kf = KalmanFilter(cfg.sim.dt, cfg.sensor.pos_noise, cfg.sensor.vel_noise, q, cfg.sim.g)
    threat = spawn_threat(cfg, rng).threat     # with drag: the filter is mismatched
    steps = 0
    while threat.active:
        kf.update(threat.pos + rng.normal(0, cfg.sensor.pos_noise, 3),
                  threat.vel + rng.normal(0, cfg.sensor.vel_noise, 3))
        P = kf.P
        assert np.allclose(P, P.T, rtol=0, atol=1e-12 * np.abs(P).max())
        np.linalg.cholesky(P)                  # raises if P is not positive definite
        threat.update(cfg.sim.dt)
        steps += 1
    assert steps > 1000


# ---- statistical consistency ----

def test_chi2_quantile_matches_reference_values():
    # scipy.stats.chi2.ppf
    assert chi2_quantile(0.025, 120) == pytest.approx(91.5726, rel=1e-4)
    assert chi2_quantile(0.975, 120) == pytest.approx(152.2114, rel=1e-4)
    assert chi2_quantile(0.025, 300) == pytest.approx(253.9123, rel=1e-4)
    assert chi2_quantile(0.975, 300) == pytest.approx(349.8745, rel=1e-4)


N_RUNS = 20
TRACK_TIME = 30.0   # s per run
TRUTH_Q = 1.0       # m^2/s^3

# Consecutive steps are strongly correlated, so the share of steps whose
# N-run average falls inside the 95 % bounds scatters around 95 % from one
# Monte Carlo set to the next (0.88-0.97 over three sets of 20 seeds). The
# criteria below accept that scatter but reject a q off by a factor of 5,
# which gives a mean NEES near 14 (q/5) or 5 (5q) and 2-78 % inside.
MIN_INSIDE = 0.85
MEAN_TOL = 0.10


def _consistent(avg, dim):
    avg = avg[np.isfinite(avg)]
    inside = fraction_inside(avg, average_bounds(N_RUNS, dim))
    return inside >= MIN_INSIDE and abs(np.mean(avg) / dim - 1) <= MEAN_TOL


def _matched_runs(cfg, filter_q):
    # Drag off and the truth driven by white-noise acceleration of density TRUTH_Q
    return monte_carlo(with_filter_q(without_drag(cfg), filter_q), range(N_RUNS),
                       truth_q=TRUTH_Q, max_time=TRACK_TIME)


@pytest.fixture(scope="module")
def matched_runs(cfg):
    return _matched_runs(cfg, TRUTH_Q)


def test_nees_is_consistent_with_drag_off_and_the_correct_q(matched_runs):
    _, nees, _ = average_statistics(matched_runs)
    assert _consistent(nees, STATE_DIM)


def test_nis_is_consistent_with_drag_off_and_the_correct_q(matched_runs):
    _, _, nis = average_statistics(matched_runs)
    assert _consistent(nis[1:], MEAS_DIM)


@pytest.mark.parametrize("factor", [0.2, 5.0])
def test_nees_check_rejects_a_wrong_q(cfg, factor):
    _, nees, _ = average_statistics(_matched_runs(cfg, factor * TRUTH_Q))
    assert not _consistent(nees, STATE_DIM)


def test_unmodeled_drag_breaks_the_consistent_filter(cfg):
    # Documented model mismatch: the filter that is consistent with drag off
    # (q = TRUTH_Q) is grossly overconfident once the threat has drag
    runs = monte_carlo(with_filter_q(cfg, TRUTH_Q), range(N_RUNS), max_time=TRACK_TIME)
    _, nees, _ = average_statistics(runs)
    assert not _consistent(nees, STATE_DIM)
    assert np.mean(nees) > 10 * STATE_DIM


def test_drag_free_predictor_overshoots_a_dragged_threat(cfg):
    # Documented model mismatch: the PIP falls beyond the true impact point,
    # and the bias shrinks to the noise level as the threat nears the ground
    bins = [0.0, 10.0, 40.0, 60.0]
    drag = pip_bias(monte_carlo(cfg, range(3)), bins)
    nodrag = pip_bias(monte_carlo(without_drag(cfg), range(3)), bins)
    along = np.nanmedian(drag.along, axis=1)
    assert along[2] > 300.0 and along[1] > 50.0
    assert abs(along[0]) < 20.0
    assert np.all(np.abs(np.nanmedian(nodrag.along, axis=1)) < 50.0)


def test_track_run_finds_the_ground_crossing(cfg):
    run = track_run(without_drag(cfg), np.random.default_rng(3))
    assert np.isfinite(run.impact).all()
    assert run.t[-1] <= run.impact_time <= run.t[-1] + cfg.sim.dt
