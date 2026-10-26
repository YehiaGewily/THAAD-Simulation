"""Verification of the integrators and the interceptor mass model."""
from dataclasses import replace

import numpy as np
import pytest

from thaad_sim import load_config
from thaad_sim.config import STANDARD_GRAVITY
from thaad_sim.dynamics import LAUNCH_SPEED, Interceptor
from thaad_sim.integrators import integrate
from thaad_sim.verification import (
    EZ, G, INTERCEPTOR_DTS, INTERCEPTOR_TARGET, THREAT_DRAG, THREAT_DTS, THREAT_P0, THREAT_V0,
    ballistic_closed_form, interceptor_convergence, observed_orders, propagate_threat, simpson,
    specific_energy, threat_convergence,
)

from conftest import CONFIGS

METHODS = ("euler", "rk4")


@pytest.fixture(scope="module")
def demo_spec():
    return load_config(CONFIGS["demo"]).interceptor


# ---- integrate() itself ----

def test_rk4_is_exact_for_a_quartic():
    # y' = 4 t^3 written autonomously as [t, y]. RK4 reduces to Simpson's rule
    # here, which is exact for a cubic integrand, so y = t^4 comes out exact.
    deriv = lambda s: np.array([1.0, 4.0 * s[0]**3])
    s = np.array([0.0, 0.0])
    for _ in range(10):
        s = integrate(s, deriv, 0.1, "rk4")
    assert s[1] == pytest.approx(1.0, rel=1e-13)


def test_euler_is_one_forward_step():
    s = integrate(np.array([1.0, 2.0]), lambda s: np.array([s[1], -s[0]]), 0.5, "euler")
    assert np.allclose(s, [2.0, 1.5])


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown integrator"):
        integrate(np.zeros(2), lambda s: s, 0.1, "midpoint")


# ---- drag-free threat against the closed form ----

def test_rk4_matches_closed_form_ballistic():
    t, states = propagate_threat("rk4", 0.1, drag=0.0)
    p, v = ballistic_closed_form(THREAT_P0, THREAT_V0, G, t)
    # The solution is quadratic in t, which RK4 reproduces up to rounding
    assert np.max(np.abs(states[:, :3] - p)) < 1e-6
    assert np.max(np.abs(states[:, 3:] - v)) < 1e-9


def test_euler_error_matches_its_truncation_error():
    dt = 0.1
    t, states = propagate_threat("euler", dt, drag=0.0)
    p, v = ballistic_closed_form(THREAT_P0, THREAT_V0, G, t)
    # Velocity is linear in t, so Euler gets it exactly; position then lags by g t dt / 2
    assert np.max(np.abs(states[:, 3:] - v)) < 1e-9
    assert np.allclose(states[:, :3] - p, 0.5 * G * dt * t[:, None] * EZ, rtol=0, atol=1e-6)


# ---- energy ----

def test_rk4_conserves_energy_without_drag():
    _, states = propagate_threat("rk4", 0.1, drag=0.0)
    e = specific_energy(states)
    assert np.max(np.abs(e - e[0])) < 1e-9 * abs(e[0])


def test_euler_energy_drift_without_drag_is_first_order():
    dt = 0.1
    t, states = propagate_threat("euler", dt, drag=0.0)
    e = specific_energy(states)
    # Only the position lag g t dt / 2 contributes, giving an energy error of g^2 t dt / 2
    assert np.allclose(e - e[0], 0.5 * G**2 * dt * t, rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("drag", [THREAT_DRAG, 4 * THREAT_DRAG])
def test_energy_loss_equals_drag_work(drag):
    dt = 0.01
    _, states = propagate_threat("rk4", dt, drag=drag)
    e = specific_energy(states)
    speed = np.linalg.norm(states[:, 3:], axis=1)
    # Drag acceleration -k|v|v does work -k|v|^3 per unit mass and time
    drag_work = simpson(drag * speed**3, dt)
    assert drag_work > 0.1 * e[0]  # the drag is strong enough to matter
    assert e[0] - e[-1] == pytest.approx(drag_work, rel=1e-8)


# ---- convergence order ----

def _assert_orders(errors, dts, expected, tol):
    for method in METHODS:
        orders = observed_orders(errors[method], dts)
        assert np.all(np.abs(orders - expected[method]) < tol[method]), (method, orders)


def test_threat_convergence_order():
    _assert_orders(threat_convergence(), THREAT_DTS,
                   expected={"euler": 1.0, "rk4": 4.0}, tol={"euler": 0.1, "rk4": 0.3})


def test_interceptor_convergence_order_through_burnout(demo_spec):
    # Burnout falls inside a step for every dt; splitting the step there keeps RK4 at 4th order
    for dt in INTERCEPTOR_DTS:
        frac = (demo_spec.burn_time / dt) % 1
        assert 0.05 < frac < 0.95
    _assert_orders(interceptor_convergence(demo_spec), INTERCEPTOR_DTS,
                   expected={"euler": 1.0, "rk4": 4.0}, tol={"euler": 0.1, "rk4": 0.3})


# ---- interceptor mass model ----

def _fly(spec, g, method, dt, t_end):
    interceptor = Interceptor([0.0, 0.0, 0.0], spec, g, method)
    interceptor.launch(0.0, EZ)
    for k in range(int(round(t_end / dt))):
        interceptor.update(k * dt, INTERCEPTOR_TARGET, np.zeros(3), dt)
    return interceptor


def test_mass_flow_follows_from_thrust_and_isp(demo_spec):
    s = demo_spec
    assert s.mass_flow == pytest.approx(s.boost_thrust / (s.isp * STANDARD_GRAVITY))
    assert s.mass_flow * s.burn_time == pytest.approx(s.mass_wet - s.mass_dry)


@pytest.mark.parametrize("method", METHODS)
def test_mass_is_integrated_during_boost(demo_spec, method):
    dt, t = 0.02, 4.0
    interceptor = _fly(demo_spec, G, method, dt, t)
    assert interceptor.stage == "BOOST"
    # Constant mass flow makes m(t) linear, which both methods integrate exactly
    assert interceptor.mass == pytest.approx(demo_spec.mass_wet - demo_spec.mass_flow * t, rel=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_booster_separates_at_burnout(demo_spec, method):
    dt = 0.02
    interceptor = _fly(demo_spec, G, method, dt, demo_spec.burn_time + dt)
    assert interceptor.stage == "KV_COAST"
    assert interceptor.mass == demo_spec.kv_mass


def test_burnout_speed_matches_rocket_equation(demo_spec):
    # No gravity and no drag: the burn adds exactly Isp g0 ln(m0 / m_b)
    spec = replace(demo_spec, booster_drag=0.0, kv_drag=0.0)
    interceptor = _fly(spec, 0.0, "rk4", 0.02, spec.burn_time + 1.0)
    dv = spec.isp * STANDARD_GRAVITY * np.log(spec.mass_wet / spec.mass_dry)
    assert interceptor.vel == pytest.approx(EZ * (LAUNCH_SPEED + dv), rel=1e-9, abs=1e-9)
