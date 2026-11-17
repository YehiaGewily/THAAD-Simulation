"""Statistical checks of the Kalman filter: NEES, NIS and impact-prediction bias.

Shared by tests/test_estimation.py, scripts/filter_consistency.py and
scripts/drag_mismatch.py. A tracking run is the engagement's sense-and-track
loop on its own (no interceptor): the same scenario draw, the same radar noise
and the same filter, run until the threat reaches the ground.
"""
from dataclasses import dataclass, replace

import numpy as np

from .estimation import KalmanFilter, cwna_process_noise, make_filter, measure_radar
from .scenario import spawn_threat

STATE_DIM = 6   # [p, v]
STATE_DIM_EKF7 = 7   # [p, v, k_T]
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


def with_drag_prior_std(cfg, drag_std):
    """The same config with a different 1-sigma width of the EKF7's drag prior."""
    return replace(cfg, kalman=replace(cfg.kalman, drag_std=drag_std))


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
    # Drag state of the 7-state EKF; all NaN for the 6-state filter
    nees7: np.ndarray = None       # (n,) NEES on [p, v, k_T]
    est_drag: np.ndarray = None    # (n,) estimated k_T, 1/m
    drag_sigma: np.ndarray = None  # (n,) the filter's 1-sigma for k_T, 1/m
    true_drag: np.ndarray = None   # (n,) true k_T, 1/m


def track_run(cfg, rng, truth_q=None, max_time=None, drag_from_prior=False, with_pip=True):
    """Track one seeded threat to the ground.

    With ``truth_q`` (m^2/s^3) the truth also receives discretized white-noise
    acceleration, w_k ~ N(0, Q(truth_q)), after every step. With drag off and
    ``truth_q`` equal to the filter's q, the truth follows exactly the
    filter's model, the setting in which NEES and NIS must be chi-square.

    For the 7-state EKF the matched setting is ``drag_from_prior`` with
    ``truth_q``: the true drag is drawn from the filter's prior
    N(drag_init, drag_std^2) and then follows the filter's random walk
    (kalman.q_drag). ``with_pip=False`` skips the impact prediction, which
    the NEES and NIS do not need.
    """
    dt, g = cfg.sim.dt, cfg.sim.g
    threat = spawn_threat(cfg, rng).threat
    kf = make_filter(cfg)
    is_ekf7 = hasattr(kf, "P7")
    if drag_from_prior:
        threat.drag = float(rng.normal(cfg.kalman.drag_init, cfg.kalman.drag_std))
    if truth_q:
        L = np.linalg.cholesky(cwna_process_noise(truth_q, dt))
        drag_walk = np.sqrt(cfg.kalman.q_drag * dt) if is_ekf7 else 0.0

    t = 0.0
    rows = []
    impact, impact_time = np.full(2, np.nan), np.nan
    while threat.active and (max_time is None or t < max_time):
        mp, mv, R_meas = measure_radar(threat.pos, threat.vel, cfg.sensor, rng)
        kf.update(mp, mv, R_meas=R_meas)
        pip = kf.predict_impact(g) if with_pip else None
        tgo = kf.time_to_impact(g) if with_pip else None
        if is_ekf7:
            drag_cols = (kf.nees7(np.hstack((threat.state, threat.drag))), float(kf.x7[6]),
                         float(np.sqrt(kf.P7[6, 6])), threat.drag)
        else:
            drag_cols = (np.nan, np.nan, np.nan, threat.drag)
        rows.append((t, kf.nees(threat.state), np.nan if kf.nis is None else kf.nis,
                     pip[:2] if pip is not None else (np.nan, np.nan),
                     np.nan if tgo is None else tgo, threat.vel.copy()) + drag_cols)
        t_ground = threat.update(dt)
        if t_ground is not None:
            # The threat stops at the ground crossing, interpolated inside the step
            impact, impact_time = threat.pos[:2].copy(), t + t_ground
            break
        if truth_q:
            threat.state = threat.state + L @ rng.standard_normal(STATE_DIM)
            if drag_walk:
                threat.drag += drag_walk * rng.standard_normal()
        t += dt

    t, nees, nis, pip, tgo, vel, nees7, est_drag, drag_sigma, true_drag = (
        np.array(c, dtype=float) for c in zip(*rows))
    return TrackRun(t, nees, nis, pip, tgo, vel, impact, impact_time,
                    nees7, est_drag, drag_sigma, true_drag)


def monte_carlo(cfg, seeds, truth_q=None, max_time=None, **kwargs):
    """One TrackRun per seed, each from its own ``np.random.default_rng(seed)``."""
    return [track_run(cfg, np.random.default_rng(s), truth_q, max_time, **kwargs) for s in seeds]


def average_nees7(runs):
    """Per-step average of the 7-state NEES over the runs, up to the shortest run."""
    n = min(len(r.t) for r in runs)
    return np.mean([r.nees7[:n] for r in runs], axis=0)


def drag_snapshot(runs, t):
    """The EKF7's drag estimate across runs at time ``t`` (s since first detection).

    ``normalized`` is (estimate - truth) / sigma per run: mean 0 and standard
    deviation 1 for a consistent filter. ``coverage`` is the share of runs
    whose truth lies within 1.96 sigma of the estimate.
    """
    i = int(round(t / (runs[0].t[1] - runs[0].t[0])))
    sel = [r for r in runs if len(r.t) > i]
    est = np.array([r.est_drag[i] for r in sel])
    sig = np.array([r.drag_sigma[i] for r in sel])
    true = np.array([r.true_drag[i] for r in sel])
    z = (est - true) / sig
    return {"t": float(t), "runs": len(sel), "estimate_median": float(np.median(est)),
            "sigma_median": float(np.median(sig)), "truth_median": float(np.median(true)),
            "abs_error_median": float(np.median(np.abs(est - true))),
            "normalized_mean": float(np.mean(z)), "normalized_std": float(np.std(z, ddof=1)),
            "coverage": float(np.mean(np.abs(z) <= 1.959963984540054))}


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
