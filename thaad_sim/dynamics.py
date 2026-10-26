"""Point-mass dynamics for the ballistic threat and the two-stage interceptor."""
import numpy as np

from .guidance import G0, limit_divert, limit_g, pn_command

LAUNCH_SPEED = 80.0          # m/s tube-exit kick along the launch direction
BOOST_STEER_FRACTION = 0.1   # share of the maneuver command applied during boost


class BallisticThreat:
    def __init__(self, pos, vel, drag, g):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.drag = drag
        self.g = g
        self.active = True

    def update(self, dt):
        if not self.active:
            return
        speed = np.linalg.norm(self.vel)
        drag = -self.drag * speed * self.vel
        self.vel += (np.array([0, 0, -self.g]) + drag) * dt
        self.pos += self.vel * dt
        if self.pos[2] <= 0:
            self.active = False


class Interceptor:
    def __init__(self, pos, spec, g):
        """``spec`` is an InterceptorSettings; ``g`` is gravity in m/s^2."""
        self.spec = spec
        self.g = g
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([0., 0., 0.], dtype=float)
        self.acc = np.array([0., 0., 0.])
        self.mass = spec.mass_wet
        self.stage = "IDLE"
        self.launch_time = 0.0
        self.launched = False
        self.exploded = False
        self.divert_dv_used = 0.0
        self.g_hist = []

    def launch(self, t, direction):
        self.launched = True
        self.launch_time = t
        self.stage = "BOOST"
        self.vel = direction * LAUNCH_SPEED

    def update(self, t, target_p, target_v, dt):
        if not self.launched or self.exploded:
            return
        s = self.spec

        # Staging
        t_flight = t - self.launch_time
        if t_flight < s.boost_time:
            self.stage = "BOOST"
            burn = (s.mass_wet - s.mass_dry) / s.boost_time
            self.mass = s.mass_wet - burn * t_flight
            thrust = s.boost_thrust
        else:
            self.stage = "KV_COAST"
            self.mass = s.kv_mass
            thrust = 0.0

        acc_cmd = pn_command(target_p - self.pos, target_v - self.vel, s.pn_gain, self.g)
        acc_cmd, acc_mag = limit_g(acc_cmd, s.max_g)

        # KV divert thrusters may have a finite delta-v budget; once spent, the KV coasts
        if self.stage == "KV_COAST" and s.kv_divert_dv is not None:
            remaining = max(s.kv_divert_dv - self.divert_dv_used, 0.0)
            acc_cmd, acc_mag = limit_divert(acc_cmd, acc_mag, remaining, dt)
            self.divert_dv_used += acc_mag * dt
        self.g_hist.append(acc_mag / G0)

        # Force direction for thrust
        if np.linalg.norm(acc_cmd) > 1e-3:
            u = acc_cmd / np.linalg.norm(acc_cmd)
        else:
            vmag = np.linalg.norm(self.vel)
            u = self.vel / (vmag + 1e-9)

        f_thrust   = u * thrust
        f_drag     = -(s.booster_drag if self.stage == "BOOST" else s.kv_drag) * np.linalg.norm(self.vel) * self.vel
        f_grav     = np.array([0, 0, -self.g]) * self.mass
        f_maneuver = acc_cmd * self.mass

        if self.stage == "BOOST":
            total_f = f_thrust + f_grav + f_drag + BOOST_STEER_FRACTION * f_maneuver
        else:
            total_f = f_maneuver + f_grav + f_drag

        self.acc = total_f / self.mass
        self.vel += self.acc * dt
        self.pos += self.vel * dt
