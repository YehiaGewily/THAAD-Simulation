"""Interceptor guidance laws (PN, Augmented PN, Optimal ZEM) with g-limit and divert budget."""
import numpy as np

G0 = 9.81             # m/s^2 per g, for g-load limits and reporting
PN_MIN_RANGE = 20.0   # m; below this the LOS rate is singular, so no command is issued


def pn_command(r, v, pn_gain, g):
    """Classical PN acceleration with gravity compensation from relative r and v (target - interceptor)."""
    r_mag = np.linalg.norm(r)
    if r_mag < PN_MIN_RANGE:
        return np.zeros(3)
    omega = np.cross(r, v) / (r_mag**2 + 1e-6)
    acc_cmd = pn_gain * np.cross(v, omega)
    acc_cmd += np.array([0.0, 0.0, g])  # gravity comp
    return acc_cmd


def apn_command(r, v, pn_gain, g, target_accel=None, boost=False):
    """Augmented Proportional Navigation (APN) with target acceleration feed-forward.

    In free-fall coast between two bodies in uniform gravity, gravity cancels
    identically in the relative acceleration (r_ddot = a_{T,drag} - a_{I,drag} - a_divert).
    Adding +g only during boost pitches the booster loft trajectory while sparing
    the kill vehicle from burning 9.81 m/s of divert delta-v every second fighting
    free fall during midcourse coast.
    """
    r_mag = np.linalg.norm(r)
    if r_mag < PN_MIN_RANGE:
        return np.zeros(3)
    omega = np.cross(r, v) / (r_mag**2 + 1e-6)
    acc_cmd = pn_gain * np.cross(v, omega)
    if target_accel is not None:
        r_hat = r / r_mag
        a_t = np.asarray(target_accel, dtype=float)
        a_t_perp = a_t - np.dot(a_t, r_hat) * r_hat
        acc_cmd = acc_cmd + 0.5 * pn_gain * a_t_perp
    if boost:
        acc_cmd += np.array([0.0, 0.0, g])
    return acc_cmd


def zem_command(r, v, pn_gain, g, target_accel=None, boost=False):
    """Optimal Zero-Effort Miss (ZEM) guidance law:
        t_go = - (r . v) / ||v||^2
        ZEM  = r + v * t_go + 0.5 * a_T * t_go^2
        a_c  = (N / t_go^2) * ZEM_perp
    """
    r_mag = np.linalg.norm(r)
    if r_mag < PN_MIN_RANGE:
        return np.zeros(3)
    v_sq = float(np.dot(v, v))
    if v_sq < 1e-6:
        return np.array([0.0, 0.0, g if boost else 0.0])
    rv = float(np.dot(r, v))
    if rv < 0:
        t_go = max(-rv / v_sq, 0.05)
    else:
        t_go = max(r_mag / np.sqrt(v_sq), 0.05)

    zem = r + v * t_go
    if target_accel is not None:
        zem = zem + 0.5 * np.asarray(target_accel, dtype=float) * (t_go ** 2)

    r_hat = r / r_mag
    zem_perp = zem - np.dot(zem, r_hat) * r_hat
    acc_cmd = (pn_gain / (t_go ** 2)) * zem_perp
    if boost:
        acc_cmd += np.array([0.0, 0.0, g])
    return acc_cmd


def guidance_command(law, r, v, pn_gain, g, target_accel=None, boost=False):
    """Dispatch guidance command for law in ('pn', 'apn', 'zem')."""
    if law == "apn":
        return apn_command(r, v, pn_gain, g, target_accel=target_accel, boost=boost)
    if law == "zem":
        return zem_command(r, v, pn_gain, g, target_accel=target_accel, boost=boost)
    return pn_command(r, v, pn_gain, g)


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
