"""Fixed-step ODE integration shared by every body in the simulation."""
INTEGRATORS = ("euler", "rk4")


def integrate(state, deriv_fn, dt, method="rk4"):
    """Advance ``state`` by ``dt`` under ``d(state)/dt = deriv_fn(state)``; returns a new array.

    ``"euler"`` is explicit (forward) Euler, first order. ``"rk4"`` is the
    classical fourth-order Runge-Kutta method. ``deriv_fn`` must not depend on
    time inside the step: callers hold inputs such as thrust and guidance
    commands constant over a step and split it at discontinuities.
    """
    if method == "rk4":
        k1 = deriv_fn(state)
        k2 = deriv_fn(state + 0.5 * dt * k1)
        k3 = deriv_fn(state + 0.5 * dt * k2)
        k4 = deriv_fn(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if method == "euler":
        return state + dt * deriv_fn(state)
    raise ValueError(f"unknown integrator {method!r}; expected one of {INTEGRATORS}")
