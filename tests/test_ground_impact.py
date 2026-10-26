"""Ground impact: the threat stops at the z = 0 crossing inside the step, not below it."""
import numpy as np
import pytest

from thaad_sim import load_config, simulate
from thaad_sim.dynamics import BallisticThreat
from thaad_sim.verification import G, ballistic_closed_form

from conftest import CONFIGS

P0 = np.array([2000.0, -1000.0, 1000.0])
V0 = np.array([300.0, 150.0, -30.0])


def _drag_free_impact():
    """Closed-form time and ground point where the drag-free arc from P0, V0 reaches z = 0."""
    a, b, c = 0.5 * G, -V0[2], -P0[2]
    t = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    p, _ = ballistic_closed_form(P0, V0, G, [t])
    return t, p[0]


def _fly_to_ground(dt, method):
    threat = BallisticThreat(P0, V0, 0.0, G, method)
    t = 0.0
    while True:
        t_ground = threat.update(dt)
        if t_ground is not None:
            return threat, t, t_ground
        t += dt


@pytest.mark.parametrize("dt", [0.02, 0.07, 0.5])
def test_threat_stops_on_the_ground_at_the_crossing(dt):
    t_true, p_true = _drag_free_impact()
    threat, t_step, t_ground = _fly_to_ground(dt, "rk4")
    assert not threat.active
    assert 0 < t_ground <= dt
    assert threat.pos[2] == 0.0
    # The chord departs from the arc by at most g dt^2 / 8, which bounds the
    # point error; the time error is that over the descent speed.
    chord = G * dt**2 / 8
    assert t_step + t_ground == pytest.approx(t_true, abs=2 * chord / abs(V0[2]) + 1e-9)
    assert np.linalg.norm(threat.pos - p_true) < 2 * chord * (1 + np.linalg.norm(V0[:2]) / abs(V0[2])) + 1e-9


def test_airborne_steps_return_none_and_a_landed_threat_stays_put():
    threat = BallisticThreat(P0, V0, 0.0, G, "rk4")
    assert threat.update(0.02) is None
    threat, _, _ = _fly_to_ground(0.02, "rk4")
    landed = threat.state.copy()
    assert threat.update(0.02) is None
    assert np.array_equal(threat.state, landed)


def test_euler_threat_also_stops_on_the_ground():
    threat, _, t_ground = _fly_to_ground(0.5, "euler")
    assert threat.pos[2] == 0.0 and 0 < t_ground <= 0.5


# Engineering seeds (legacy RNG) whose threats reach the ground: 3 is GROUND
# after a failed intercept, 7 lands outside the defended area untouched
@pytest.mark.parametrize("seed,outcome", [(3, "GROUND"), (7, "NO ENGAGE")])
def test_engagement_records_the_impact_on_the_ground(seed, outcome, quiet):
    cfg = load_config(CONFIGS["engineering"])
    hist = simulate(cfg, seed, legacy_rng=True, console=quiet)
    meta = hist['meta']
    assert meta['outcome'] == outcome

    # The last row is the impact, at the interpolated time and point
    assert hist['t_pos'][-1][2] == 0.0
    assert np.array_equal(meta['impact'], hist['t_pos'][-1])
    assert all(p[2] > 0 for p in hist['t_pos'][:-1])
    assert hist['t'][-2] < hist['t'][-1] <= hist['t'][-2] + cfg.sim.dt + 1e-9
    assert len(meta['g_hist']) == sum(s != "SEARCH" for s in hist['status'])
