"""Statistical checks of the Kalman filter: NEES, NIS and impact-prediction bias.

Shared by tests/test_estimation.py, scripts/filter_consistency.py and
scripts/drag_mismatch.py. A tracking run is the engagement's sense-and-track
loop on its own (no interceptor): the same scenario draw, the same radar noise
and the same filter, run until the threat reaches the ground.
"""
from dataclasses import dataclass, replace

import numpy as np

from .estimation import KalmanFilter, cwna_process_noise
from .scenario import spawn_threat

STATE_DIM = 6   # [p, v]
MEAS_DIM = 6    # H = I: position and velocity are both measured

# Standard-normal quantiles for two-sided 95 % bounds
_Z = {0.025: -1.959963984540054, 0.975: 1.959963984540054}


def chi2_quantile(p, dof):
    """Chi-square quantile via the Wilson-Hilferty approximation.

    Within 0.01 % of the exact quantile at the 120 or more degrees of freedom
    of an average over 20 or more runs (about 2.5 % off at 6), which avoids a
    SciPy dependency for one function.
    """
    z = _Z[p]
    c = 2.0 / (9.0 * dof)
    return dof * (1.0 - c + z * np.sqrt(c)) ** 3


def average_bounds(n_runs, dim):
    """Two-sided 95 % bounds on a statistic averaged over ``n_runs`` independent runs.

    For a consistent filter, n_runs times the average NEES (or NIS) is
    chi-square with n_runs * dim degrees of freedom.
    """
    dof = n_runs * dim
    return chi2_quantile(0.025, dof) / n_runs, chi2_quantile(0.975, dof) / n_runs


def without_drag(cfg):
    """The same config with drag off, so the truth obeys the filter's motion model."""
    return replace(cfg, threat=replace(cfg.threat, drag=0.0))


def with_filter_q(cfg, q):
    return replace(cfg, kalman=replace(cfg.kalman, q=q))


@dataclass
class TrackRun:
    t: np.ndarray            # (n,) s
    nees: np.ndarray         # (n,) NEES after each update
    nis: np.ndarray          # (n,) NIS of each update; NaN for the initializing measurement
    pip: np.ndarray          # (n, 2) drag-free predicted impact point; NaN while undefined
    tgo_pred: np.ndarray     # (n,) drag-free predicted time to impact; NaN while undefined
    vel: np.ndarray          # (n, 3) true velocity
    impact: np.ndarray       # (2,) true impact point on z = 0
    impact_time: float       # s


def track_run(cfg, rng, truth_q=None, max_time=None):
    """Track one seeded threat to the ground.

    With ``truth_q`` (m^2/s^3) the truth also receives discretized white-noise
    acceleration, w_k ~ N(0, Q(truth_q)), after every step. With drag off and
    ``truth_q`` equal to the filter's q, the truth follows exactly the
    filter's model, the setting in which NEES and NIS must be chi-square.
    """
    dt, g = cfg.sim.dt, cfg.sim.g
    threat = spawn_threat(cfg, rng).threat
    kf = KalmanFilter(dt, cfg.sensor.pos_noise, cfg.sensor.vel_noise, cfg.kalman.q, g)
    if truth_q:
        L = np.linalg.cholesky(cwna_process_noise(truth_q, dt))

    t = 0.0
    rows = []
    prev = threat.state.copy()
    while threat.active and (max_time is None or t < max_time):
        mp = threat.pos + rng.normal(0, cfg.sensor.pos_noise, 3)
        mv = threat.vel + rng.normal(0, cfg.sensor.vel_noise, 3)
        kf.update(mp, mv)
        pip = kf.predict_impact(g)
        tgo = kf.time_to_impact(g)
        rows.append((t, kf.nees(threat.state), np.nan if kf.nis is None else kf.nis,
                     pip[:2] if pip is not None else (np.nan, np.nan),
                     np.nan if tgo is None else tgo, threat.vel.copy()))
        prev = threat.state.copy()
        threat.update(dt)
        if truth_q:
            threat.state = threat.state + L @ rng.standard_normal(STATE_DIM)
        t += dt

    # Ground crossing, interpolated linearly inside the last step
    if threat.active:
        impact, impact_time = np.full(2, np.nan), np.nan
    else:
        f = prev[2] / (prev[2] - threat.pos[2])
        impact = prev[:2] + f * (threat.pos[:2] - prev[:2])
        impact_time = t - dt + f * dt

    t, nees, nis, pip, tgo, vel = (np.array(c, dtype=float) for c in zip(*rows))
    return TrackRun(t, nees, nis, pip, tgo, vel, impact, impact_time)


def monte_carlo(cfg, seeds, truth_q=None, max_time=None):
    """One TrackRun per seed, each from its own ``np.random.default_rng(seed)``."""
    return [track_run(cfg, np.random.default_rng(s), truth_q, max_time) for s in seeds]


def average_statistics(runs):
    """Per-step averages of NEES and NIS over the runs, up to the shortest run."""
    n = min(len(r.t) for r in runs)
    t = runs[0].t[:n]
    nees = np.mean([r.nees[:n] for r in runs], axis=0)
    nis = np.mean([r.nis[:n] for r in runs], axis=0)
    return t, nees, nis


def fraction_inside(avg, bounds):
    """Share of finite samples of an averaged statistic inside its 95 % bounds."""
    avg = avg[np.isfinite(avg)]
    lo, hi = bounds
    return float(np.mean((avg >= lo) & (avg <= hi)))


@dataclass
class PipBias:
    tgo: np.ndarray          # (m,) bin centres of true time to go, s
    along: np.ndarray        # (m, k) along-track PIP error per run (NaN where undefined), m
    cross: np.ndarray        # (m, k) cross-track PIP error per run, m
    tgo_err: np.ndarray      # (m, k) predicted minus true time to go, s


def pip_bias(runs, tgo_bins):
    """PIP error against the true impact point, binned by true time to go.

    The error is split along and across the threat's true horizontal velocity
    at that instant, so a positive along-track error means the predictor puts
    the impact beyond where the threat actually lands.
    """
    tgo_bins = np.asarray(tgo_bins, dtype=float)
    m, k = len(tgo_bins) - 1, len(runs)
    along, cross, tgo_err = (np.full((m, k), np.nan) for _ in range(3))
    for j, r in enumerate(runs):
        tgo_true = r.impact_time - r.t
        err = r.pip - r.impact
        h = r.vel[:, :2] / np.linalg.norm(r.vel[:, :2], axis=1, keepdims=True)
        a = np.sum(err * h, axis=1)
        c = err[:, 0] * h[:, 1] - err[:, 1] * h[:, 0]
        idx = np.digitize(tgo_true, tgo_bins) - 1
        for b in range(m):
            sel = idx == b
            if np.any(sel):
                along[b, j] = np.nanmean(a[sel]) if np.any(np.isfinite(a[sel])) else np.nan
                cross[b, j] = np.nanmean(c[sel]) if np.any(np.isfinite(c[sel])) else np.nan
                d = r.tgo_pred[sel] - tgo_true[sel]
                tgo_err[b, j] = np.nanmean(d) if np.any(np.isfinite(d)) else np.nan
    centres = 0.5 * (tgo_bins[:-1] + tgo_bins[1:])
    return PipBias(centres, along, cross, tgo_err)
