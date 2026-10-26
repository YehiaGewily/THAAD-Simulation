"""Interceptor guidance: proportional navigation with a g-limit and a divert budget."""
import numpy as np

G0 = 9.81             # m/s^2 per g, for g-load limits and reporting
PN_MIN_RANGE = 20.0   # m; below this the LOS rate is singular, so no command is issued


def pn_command(r, v, pn_gain, g):
    """Augmented PN acceleration from relative position r and velocity v (target - interceptor)."""
    r_mag = np.linalg.norm(r)
    if r_mag < PN_MIN_RANGE:
        return np.zeros(3)
    omega = np.cross(r, v) / (r_mag**2 + 1e-6)
    acc_cmd = pn_gain * np.cross(v, omega)
    acc_cmd += np.array([0, 0, g])  # gravity comp
    return acc_cmd


def limit_g(acc_cmd, max_g):
    """Cap the command at max_g; returns (command, magnitude)."""
    acc_mag = np.linalg.norm(acc_cmd)
    lim = max_g * G0
    if acc_mag > lim:
        acc_cmd = acc_cmd / acc_mag * lim
        acc_mag = lim
    return acc_cmd, acc_mag


def limit_divert(acc_cmd, acc_mag, remaining_dv, dt):
    """Cap the command so this step spends no more than the remaining divert delta-v."""
    if acc_mag * dt > remaining_dv:
        acc_cmd = acc_cmd / acc_mag * (remaining_dv / dt)
        acc_mag = remaining_dv / dt
    return acc_cmd, acc_mag
