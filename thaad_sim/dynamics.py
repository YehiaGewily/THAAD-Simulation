"""Point-mass dynamics for the ballistic threat and the two-stage interceptor.

Each body keeps its state in one array advanced by ``integrators.integrate``:
``[x, y, z, vx, vy, vz]`` for the threat and ``[x, y, z, vx, vy, vz, m]`` for
the interceptor, so the interceptor's mass is integrated with its motion.
"""
import numpy as np

from .guidance import G0, limit_divert, limit_g, pn_command
from .integrators import integrate

LAUNCH_SPEED = 80.0          # m/s tube-exit kick along the launch direction
BOOST_STEER_FRACTION = 0.1   # share of the maneuver command applied during boost


class BallisticThreat:
    def __init__(self, pos, vel, drag, g, integrator="rk4"):
        self.state = np.hstack((pos, vel)).astype(float)
        self.drag = drag
        self.g = g
        self.integrator = integrator
        self.active = True

    @property
    def pos(self):
        return self.state[:3]

    @property
    def vel(self):
        return self.state[3:]

    def derivative(self, state):
        """d/dt [p, v] under gravity and quadratic drag."""
        vel = state[3:]
        acc = -self.drag * np.linalg.norm(vel) * vel
        acc[2] -= self.g
        return np.concatenate((vel, acc))

    def update(self, dt):
        if not self.active:
            return
        self.state = integrate(self.state, self.derivative, dt, self.integrator)
        if self.pos[2] <= 0:
            self.active = False


class Interceptor:
    def __init__(self, pos, spec, g, integrator="rk4"):
        """``spec`` is an InterceptorSettings; ``g`` is gravity in m/s^2."""
        self.spec = spec
        self.g = g
        self.integrator = integrator
        self.state = np.hstack((pos, [0., 0., 0.], [spec.mass_wet])).astype(float)
        self.stage = "IDLE"
        self.launch_time = 0.0
        self.launched = False
        self.exploded = False
        self.divert_dv_used = 0.0
        self.g_hist = []

    @property
    def pos(self):
        return self.state[:3]

    @property
    def vel(self):
        return self.state[3:6]

    @property
    def mass(self):
        return self.state[6]

    def launch(self, t, direction):
        self.launched = True
        self.launch_time = t
        self.stage = "BOOST"
        self.state[3:6] = direction * LAUNCH_SPEED

    def _derivative(self, thrust, maneuver, drag, mass_flow):
        """d/dt [p, v, m] with thrust force, maneuver acceleration and mass flow held constant."""
        g = self.g

        def deriv(state):
            vel, m = state[3:6], state[6]
            acc = (thrust - drag * np.linalg.norm(vel) * vel) / m + maneuver
            acc[2] -= g
            return np.concatenate((vel, acc, [-mass_flow]))
        return deriv

    def update(self, t, target_p, target_v, dt):
        if not self.launched or self.exploded:
            return
        s = self.spec

        # The motor burns until its propellant is spent. Burnout can fall inside
        # this step, so the step is split there: thrust cuts off and the booster
        # drops at the right instant instead of at the next step boundary.
        burn_left = s.burn_time - (t - self.launch_time)
        t_boost = min(max(burn_left, 0.0), dt)
        t_coast = dt - t_boost
        # Stage at the end of the step, which is when the engagement records it
        self.stage = "BOOST" if burn_left > dt else "KV_COAST"

        # Guidance runs once per step; its command is held over the step
        acc_cmd = pn_command(target_p - self.pos, target_v - self.vel, s.pn_gain, self.g)
        acc_cmd, acc_mag = limit_g(acc_cmd, s.max_g)
        g_load = acc_mag

        state = self.state
        if t_boost > 0:
            # Thrust along the command, or along the velocity when there is none
            if acc_mag > 1e-3:
                u = acc_cmd / acc_mag
            else:
                u = self.vel / (np.linalg.norm(self.vel) + 1e-9)
            deriv = self._derivative(u * s.boost_thrust, BOOST_STEER_FRACTION * acc_cmd,
                                     s.booster_drag, s.mass_flow)
            state = integrate(state, deriv, t_boost, self.integrator)
            if burn_left <= dt:
                state[6] = s.kv_mass          # burnout: the booster separates

        if t_coast > 0:
            # KV divert thrusters may have a finite delta-v budget; once spent, the KV coasts
            kv_cmd, kv_mag = acc_cmd, acc_mag
            if s.kv_divert_dv is not None:
                remaining = max(s.kv_divert_dv - self.divert_dv_used, 0.0)
                kv_cmd, kv_mag = limit_divert(kv_cmd, kv_mag, remaining, t_coast)
            self.divert_dv_used += kv_mag * t_coast
            if t_boost == 0:
                g_load = kv_mag
            deriv = self._derivative(np.zeros(3), kv_cmd, s.kv_drag, 0.0)
            state = integrate(state, deriv, t_coast, self.integrator)

        self.state = state
        self.g_hist.append(g_load / G0)
