"""Reference problems for verifying the integrators.

Shared by tests/test_integrators.py and scripts/convergence_study.py. Each
problem is deterministic and smooth (no sensor noise, no closed guidance loop),
so its error against a reference shrinks at the integrator's formal order.
"""
from dataclasses import replace

import numpy as np

from .dynamics import BallisticThreat, Interceptor

G = 9.81
EZ = np.array([0.0, 0.0, 1.0])

# Threat problem: a descending ballistic arc that stays well above the ground
THREAT_P0 = np.array([-30000.0, 10000.0, 60000.0])
THREAT_V0 = np.array([900.0, -300.0, -400.0])
THREAT_DRAG = 4e-5        # 1/m, the configured threat drag
THREAT_T_END = 40.0       # s

# Interceptor problem: a vertical launch at a target straight overhead. The
# line of sight never rotates, so the PN command is exactly the gravity bias
# and does not depend on dt; burnout falls inside a step for every dt used.
INTERCEPTOR_T_END = 12.8  # s
INTERCEPTOR_TARGET = np.array([0.0, 0.0, 1.0e6])

# Step sizes for the convergence study: dt, dt/2, dt/4, dt/8. They are large
# enough that RK4's error at dt/8 stays well above floating-point rounding.
THREAT_DTS = (0.8, 0.4, 0.2, 0.1)
INTERCEPTOR_DTS = (0.64, 0.32, 0.16, 0.08)
REFERENCE_REFINEMENT = 16  # reference run: RK4 at the smallest dt / 16


def _steps(t_end, dt):
    n = int(round(t_end / dt))
    if not np.isclose(n * dt, t_end, rtol=0, atol=1e-9):
        raise ValueError(f"t_end={t_end} is not a whole number of steps of dt={dt}")
    return n


def ballistic_closed_form(p0, v0, g, t):
    """Drag-free ballistic position and velocity at times ``t`` (shape (n, 3) each)."""
    t = np.asarray(t, dtype=float)[:, None]
    return p0 + v0 * t - 0.5 * g * t**2 * EZ, v0 - g * t * EZ


def propagate_threat(method, dt, drag=THREAT_DRAG, t_end=THREAT_T_END,
                     p0=THREAT_P0, v0=THREAT_V0, g=G):
    """Integrate the threat problem; returns (times, states) with states of shape (n+1, 6)."""
    threat = BallisticThreat(p0, v0, drag, g, method)
    n = _steps(t_end, dt)
    states = [threat.state.copy()]
    for _ in range(n):
        threat.update(dt)
        states.append(threat.state.copy())
    return np.arange(n + 1) * dt, np.array(states)


def specific_energy(states, g=G):
    """Kinetic plus potential energy per unit mass, J/kg, for [p, v] state rows."""
    states = np.atleast_2d(states)
    return 0.5 * np.sum(states[:, 3:6]**2, axis=1) + g * states[:, 2]


def simpson(y, dt):
    """Composite Simpson integral of samples ``y`` spaced ``dt`` apart (odd sample count)."""
    if len(y) % 2 == 0:
        raise ValueError("Simpson's rule needs an odd number of samples")
    return dt / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))


def propagate_interceptor(spec, method, dt, t_end=INTERCEPTOR_T_END, g=G):
    """Fly the vertical-launch interceptor problem; returns (times, states) of shape (n+1, 7)."""
    interceptor = Interceptor([0.0, 0.0, 0.0], spec, g, method)
    interceptor.launch(0.0, EZ)
    n = _steps(t_end, dt)
    states = [interceptor.state.copy()]
    for k in range(n):
        interceptor.update(k * dt, INTERCEPTOR_TARGET, np.zeros(3), dt)
        states.append(interceptor.state.copy())
    return np.arange(n + 1) * dt, np.array(states)


def interceptor_problem_spec(spec):
    """The configured interceptor with an unlimited divert budget, so the command stays dt-independent."""
    return replace(spec, kv_divert_dv=None)


def final_position_errors(propagate, dts):
    """Final-position error of each method at each dt against an RK4 run at min(dts)/16.

    ``propagate(method, dt)`` returns (times, states). Returns {method: errors}.
    """
    _, ref = propagate("rk4", min(dts) / REFERENCE_REFINEMENT)
    p_ref = ref[-1, :3]
    return {method: np.array([np.linalg.norm(propagate(method, dt)[1][-1, :3] - p_ref)
                              for dt in dts])
            for method in ("euler", "rk4")}


def observed_orders(errors, dts):
    """Observed order between successive step sizes: log(e1/e2) / log(dt1/dt2)."""
    errors, dts = np.asarray(errors), np.asarray(dts)
    return np.log(errors[:-1] / errors[1:]) / np.log(dts[:-1] / dts[1:])


def threat_convergence():
    return final_position_errors(lambda m, dt: propagate_threat(m, dt), THREAT_DTS)


def interceptor_convergence(spec):
    spec = interceptor_problem_spec(spec)
    return final_position_errors(lambda m, dt: propagate_interceptor(spec, m, dt),
                                 INTERCEPTOR_DTS)
