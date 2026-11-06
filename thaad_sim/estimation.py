"""Threat tracking and impact prediction:
- 6-state linear Kalman filter (KalmanFilter, 'kf6')
- 7-state Extended Kalman Filter (ExtendedKalmanFilter, 'ekf7') that jointly
  estimates the threat's quadratic drag coefficient k_T
- Cartesian and Range-Azimuth-Elevation (RAE) polar radar measurement models
- Closed-form vacuum and RK4 drag-integrated Predicted Impact Point (PIP)
"""
import numpy as np


def cwna_process_noise(q, dt, dim=3):
    """Discretized continuous white-noise acceleration Q for a [p, v] state.

    ``q`` is the acceleration noise power spectral density (m^2/s^3) on each
    axis. Integrating it over one step gives, per axis,
    q * [[dt^3/3, dt^2/2], [dt^2/2, dt]].
    """
    I = np.eye(dim)
    return q * np.block([[dt**3 / 3 * I, dt**2 / 2 * I],
                         [dt**2 / 2 * I, dt * I]])


def predict_impact_drag(pos, vel, drag, g, dt_step=2.0, max_steps=180):
    """Integrate [pos, vel] to z = 0 under gravity and quadratic drag using RK4.

    Returns (pip_3d, time_to_go) or (None, None) if not descending or no ground
    crossing is reached within max_steps.
    """
    pz, vz = float(pos[2]), float(vel[2])
    if vz >= 0.0 or pz <= 0.0:
        if pz <= 0.0 and vz < 0.0:
            return np.array([float(pos[0]), float(pos[1]), 0.0]), 0.0
        return None, None
    k_d = max(float(drag), 0.0)
    if k_d < 1e-11:
        # Exact closed-form fallback when drag is zero
        a = 0.5 * g
        b = -vz
        c = -pz
        d = b * b - 4.0 * a * c
        if d < 0.0:
            return None, None
        t_sol = (-b + np.sqrt(d)) / (2.0 * a)
        if t_sol <= 0.0:
            return None, None
        return np.array([pos[0] + vel[0] * t_sol, pos[1] + vel[1] * t_sol, 0.0]), float(t_sol)

    x, y, z = float(pos[0]), float(pos[1]), pz
    vx, vy = float(vel[0]), float(vel[1])
    h = float(dt_step)
    t_acc = 0.0

    for _ in range(max_steps):
        px, py, pz0, pvx, pvy, pvz = x, y, z, vx, vy, vz
        # Use a finer step near the ground so chord interpolation is < 1 cm accurate
        dt = 0.1 if (z + vz * h <= 0.0 and h > 0.1) else h
        half = 0.5 * dt

        sp1 = (vx * vx + vy * vy + vz * vz) ** 0.5
        ax1, ay1, az1 = -k_d * sp1 * vx, -k_d * sp1 * vy, -k_d * sp1 * vz - g

        vx2, vy2, vz2 = vx + half * ax1, vy + half * ay1, vz + half * az1
        sp2 = (vx2 * vx2 + vy2 * vy2 + vz2 * vz2) ** 0.5
        ax2, ay2, az2 = -k_d * sp2 * vx2, -k_d * sp2 * vy2, -k_d * sp2 * vz2 - g

        vx3, vy3, vz3 = vx + half * ax2, vy + half * ay2, vz + half * az2
        sp3 = (vx3 * vx3 + vy3 * vy3 + vz3 * vz3) ** 0.5
        ax3, ay3, az3 = -k_d * sp3 * vx3, -k_d * sp3 * vy3, -k_d * sp3 * vz3 - g

        vx4, vy4, vz4 = vx + dt * ax3, vy + dt * ay3, vz + dt * az3
        sp4 = (vx4 * vx4 + vy4 * vy4 + vz4 * vz4) ** 0.5
        ax4, ay4, az4 = -k_d * sp4 * vx4, -k_d * sp4 * vy4, -k_d * sp4 * vz4 - g

        c = dt / 6.0
        x += c * (vx + 2.0 * vx2 + 2.0 * vx3 + vx4)
        y += c * (vy + 2.0 * vy2 + 2.0 * vy3 + vy4)
        z += c * (vz + 2.0 * vz2 + 2.0 * vz3 + vz4)
        vx += c * (ax1 + 2.0 * ax2 + 2.0 * ax3 + ax4)
        vy += c * (ay1 + 2.0 * ay2 + 2.0 * ay3 + ay4)
        vz += c * (az1 + 2.0 * az2 + 2.0 * az3 + az4)

        if z <= 0.0:
            f = pz0 / max(pz0 - z, 1e-12)
            return np.array([px + f * (x - px), py + f * (y - py), 0.0]), float(t_acc + f * dt)
        t_acc += dt
        if vz >= 0.0:
            return None, None
    return None, None


def rae_basis(pos):
    """Orthonormal basis (u_r, u_az, u_el) and polar coordinates (r, az, el) at pos."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    rho2 = x * x + y * y
    rho = np.sqrt(rho2)
    r = max(np.sqrt(rho2 + z * z), 1.0)
    az = np.arctan2(y, x)
    el = np.arctan2(z, max(rho, 1e-9))
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    u_r = np.array([ce * ca, ce * sa, se])
    u_az = np.array([-sa, ca, 0.0])
    u_el = np.array([-se * ca, -se * sa, ce])
    return r, az, el, u_r, u_az, u_el


def measure_radar(pos, vel, sensor_cfg, rng):
    """Simulate a radar measurement of (pos, vel).

    Returns (meas_pos, meas_vel, R_meas):
    - In 'cartesian' mode, R_meas is None (filter uses its isotropic R) and the
      RNG draws are identical to the baseline implementation.
    - In 'rae' (Range-Azimuth-Elevation + Doppler) mode, polar noise is injected
      in (r, az, el) and converted to ENU Cartesian coordinates with the exact
      oriented 6x6 covariance matrix R_meas aligned with the radar beam.
    """
    if getattr(sensor_cfg, "model", "cartesian") == "cartesian":
        mp = pos + rng.normal(0, sensor_cfg.pos_noise, 3)
        mv = vel + rng.normal(0, sensor_cfg.vel_noise, 3)
        return mp, mv, None

    r, az, el, _, _, _ = rae_basis(pos)
    sig_r = sensor_cfg.range_noise
    sig_a = sensor_cfg.angle_noise
    sig_rr = sensor_cfg.range_rate_noise
    sig_vc = sensor_cfg.vel_noise

    r_m = max(r + rng.normal(0.0, sig_r), 1.0)
    az_m = az + rng.normal(0.0, sig_a)
    el_m = el + rng.normal(0.0, sig_a)

    ca, sa = np.cos(az_m), np.sin(az_m)
    ce, se = np.cos(el_m), np.sin(el_m)
    u_r = np.array([ce * ca, ce * sa, se])
    u_az = np.array([-sa, ca, 0.0])
    u_el = np.array([-se * ca, -se * sa, ce])

    mp = r_m * u_r

    # Anisotropic position covariance in ENU: tight along LOS (u_r), wide cross-range
    std_az = max(r_m * abs(ce) * sig_a, sig_r)
    std_el = max(r_m * sig_a, sig_r)
    R_p = (sig_r**2 * np.outer(u_r, u_r)
           + std_az**2 * np.outer(u_az, u_az)
           + std_el**2 * np.outer(u_el, u_el))

    # Velocity measurement: tight Doppler radial velocity along u_r, angular-derived cross-range
    dv = (rng.normal(0.0, sig_rr) * u_r
          + rng.normal(0.0, sig_vc) * u_az
          + rng.normal(0.0, sig_vc) * u_el)
    mv = vel + dv
    R_v = (sig_rr**2 * np.outer(u_r, u_r)
           + sig_vc**2 * (np.outer(u_az, u_az) + np.outer(u_el, u_el)))

    R_meas = np.block([[R_p, np.zeros((3, 3))],
                       [np.zeros((3, 3)), R_v]])
    return mp, mv, R_meas


class KalmanFilter:
    """Linear 6-state Kalman filter with gravity as a known control input."""
    def __init__(self, dt, pos_std, vel_std, q, g, pip_mode="vacuum"):
        self.dt = dt
        self.g = g
        self.pip_mode = pip_mode
        I3 = np.eye(3)
        self.F = np.block([[I3, dt * I3], [np.zeros((3, 3)), I3]])
        self.B = np.vstack((dt**2 / 2 * I3, dt * I3))
        self.u = np.array([0.0, 0.0, -g])     # known acceleration: gravity

        self.H = np.eye(6)
        self.Q = cwna_process_noise(q, dt)
        self.R = np.diag([pos_std**2] * 3 + [vel_std**2] * 3)

        self.P = self.R.copy()                # replaced at the first measurement
        self.x = np.zeros(6)
        self.initialized = False
        self.nis = None                       # innovation statistic of the last update

    @property
    def est_drag(self):
        """Estimated drag coefficient (0.0 for the unaugmented 6-state filter)."""
        return 0.0

    @property
    def est_target_accel(self):
        """Estimated non-gravitational target acceleration vector (m/s^2)."""
        return np.zeros(3)

    def update(self, meas_pos, meas_vel, R_meas=None):
        z = np.hstack((meas_pos, meas_vel))
        R = self.R if R_meas is None else R_meas
        if not self.initialized:
            # The first measurement is the estimate, so its covariance is R
            self.x = z
            self.P = R.copy()
            self.initialized = True
            return self.x[:3], self.x[3:]

        # Predict
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update. K = P H^T S^-1 comes from a linear solve, not an explicit inverse.
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.nis = float(y @ np.linalg.solve(S, y))
        self.x = self.x + K @ y
        # Joseph form: symmetric and positive definite for any gain K
        A = np.eye(6) - K @ self.H
        self.P = A @ self.P @ A.T + K @ R @ K.T

        return self.x[:3], self.x[3:]

    def nees(self, x_true):
        """Normalized estimation error squared, e^T P^-1 e with e = x_hat - x_true."""
        e = self.x - np.asarray(x_true)[:6]
        return float(e @ np.linalg.solve(self.P, e))

    def predict_impact(self, g, pip_mode=None, drag_override=None):
        """Predict impact point on z=0 from current state."""
        mode = pip_mode or self.pip_mode
        if mode == "drag":
            k_d = self.est_drag if drag_override is None else drag_override
            pip, _ = predict_impact_drag(self.x[:3], self.x[3:6], k_d, g)
            return pip
        t_sol = self.time_to_impact(g, pip_mode="vacuum")
        if t_sol is None:
            return None
        return np.array([
            self.x[0] + self.x[3]*t_sol,
            self.x[1] + self.x[4]*t_sol,
            0.0
        ])

    def time_to_impact(self, g, pip_mode=None, drag_override=None):
        """Time until z = 0, or None while the estimate is not descending."""
        mode = pip_mode or self.pip_mode
        if mode == "drag":
            k_d = self.est_drag if drag_override is None else drag_override
            _, t_sol = predict_impact_drag(self.x[:3], self.x[3:6], k_d, g)
            return t_sol
        if self.x[5] >= 0:
            return None
        p_z, v_z = self.x[2], self.x[5]
        a = 0.5 * g
        b = -v_z
        c = -p_z
        d = b*b - 4*a*c
        if d < 0:
            return None
        t_sol = (-b + np.sqrt(d)) / (2*a)
        if t_sol <= 0:
            return None
        return t_sol


def ekf7_derivative(x7, g):
    """Continuous-time derivative d/dt [p, v, k_T] for the 7-state EKF."""
    vx, vy, vz = float(x7[3]), float(x7[4]), float(x7[5])
    k_t = max(float(x7[6]), 0.0)
    spd = (vx * vx + vy * vy + vz * vz) ** 0.5
    c = -k_t * spd
    return np.array([vx, vy, vz, c * vx, c * vy, c * vz - g, 0.0])


_I3 = np.eye(3)
_I7 = np.eye(7)


def ekf7_jacobian(x7):
    """Analytical 7x7 continuous-time Jacobian F = df/dx of ekf7_derivative."""
    v = x7[3:6]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    k_t = max(float(x7[6]), 0.0)
    spd = (vx * vx + vy * vy + vz * vz) ** 0.5
    F = np.zeros((7, 7))
    F[0, 3] = F[1, 4] = F[2, 5] = 1.0
    if spd > 1e-9:
        F[3:6, 3:6] = -k_t * (spd * _I3 + np.outer(v, v) / spd)
    F[3:6, 6] = -spd * v
    return F


class ExtendedKalmanFilter:
    """7-state Extended Kalman Filter tracking [p, v, k_T] with quadratic drag.

    Exposes the same 6-state kinematic interface (``.x``, ``.P``, ``.update``,
    ``.nees``, ``.predict_impact``, ``.time_to_impact``) as ``KalmanFilter``
    while also maintaining the 7th ballistic drag state ``k_T``.
    """
    def __init__(self, dt, pos_std, vel_std, q, g,
                 drag_init=3e-5, drag_std=3e-5, q_drag=1e-12, pip_mode="drag"):
        self.dt = dt
        self.g = g
        self.pip_mode = pip_mode
        self.drag_init = float(drag_init)
        self.drag_std = float(drag_std)

        self.H7 = np.zeros((6, 7))
        self.H7[:6, :6] = np.eye(6)

        self.Q7 = np.zeros((7, 7))
        self.Q7[:6, :6] = cwna_process_noise(q, dt)
        self.Q7[6, 6] = float(q_drag) * dt

        self.R = np.diag([pos_std**2] * 3 + [vel_std**2] * 3)
        self.P7 = np.zeros((7, 7))
        self.P7[:6, :6] = self.R.copy()
        self.P7[6, 6] = self.drag_std**2

        self.x7 = np.zeros(7)
        self.x7[6] = self.drag_init
        self.initialized = False
        self.nis = None

    @property
    def x(self):
        return self.x7[:6]

    @property
    def P(self):
        return self.P7[:6, :6]

    @property
    def est_drag(self):
        return max(float(self.x7[6]), 0.0)

    @property
    def est_target_accel(self):
        """Estimated aerodynamic drag acceleration vector -k_T * ||v|| * v."""
        v = self.x7[3:6]
        return -self.est_drag * np.linalg.norm(v) * v

    def _predict_state_rk4(self, x7):
        dt, g = self.dt, self.g
        k1 = ekf7_derivative(x7, g)
        k2 = ekf7_derivative(x7 + 0.5 * dt * k1, g)
        k3 = ekf7_derivative(x7 + 0.5 * dt * k2, g)
        k4 = ekf7_derivative(x7 + dt * k3, g)
        nxt = x7 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        nxt[6] = np.clip(nxt[6], 0.0, 5e-4)
        return nxt

    def update(self, meas_pos, meas_vel, R_meas=None):
        z = np.hstack((meas_pos, meas_vel))
        R = self.R if R_meas is None else R_meas
        if not self.initialized:
            self.x7[:6] = z
            self.x7[6] = self.drag_init
            self.P7 = np.zeros((7, 7))
            self.P7[:6, :6] = R.copy()
            self.P7[6, 6] = self.drag_std**2
            self.initialized = True
            return self.x7[:3].copy(), self.x7[3:6].copy()

        # Nonlinear RK4 state propagation + 2nd-order Taylor state transition Phi
        F = ekf7_jacobian(self.x7)
        Fdt = F * self.dt
        Phi = _I7 + Fdt + 0.5 * (Fdt @ Fdt)
        self.x7 = self._predict_state_rk4(self.x7)
        self.P7 = Phi @ self.P7 @ Phi.T + self.Q7

        # Joseph-form measurement update
        y = z - self.x7[:6]
        S = self.P7[:6, :6] + R
        K = np.linalg.solve(S, self.P7[:6, :]).T   # (7, 6)
        self.nis = float(y @ np.linalg.solve(S, y))
        self.x7 = self.x7 + K @ y
        self.x7[6] = np.clip(self.x7[6], 0.0, 5e-4)

        A = _I7 - K @ self.H7
        self.P7 = A @ self.P7 @ A.T + K @ R @ K.T
        self.P7 = 0.5 * (self.P7 + self.P7.T)
        self._step_count = getattr(self, "_step_count", 0) + 1
        if self._step_count % 10 == 0:
            self._pip_cache = None

        return self.x7[:3].copy(), self.x7[3:6].copy()

    def nees(self, x_true):
        """6-DOF kinematic NEES on [p, v], directly comparable to KalmanFilter.nees."""
        e = self.x7[:6] - np.asarray(x_true)[:6]
        return float(e @ np.linalg.solve(self.P7[:6, :6], e))

    def nees7(self, x7_true):
        """Full 7-DOF NEES including the drag state k_T."""
        e = self.x7 - np.asarray(x7_true)
        return float(e @ np.linalg.solve(self.P7, e))

    def predict_impact(self, g, pip_mode=None, drag_override=None):
        mode = pip_mode or self.pip_mode
        if mode == "drag":
            if drag_override is None and getattr(self, "_pip_cache", None) is not None:
                return self._pip_cache[0]
            k_d = self.est_drag if drag_override is None else drag_override
            res = predict_impact_drag(self.x7[:3], self.x7[3:6], k_d, g)
            if drag_override is None:
                self._pip_cache = res
            return res[0]
        t_sol = self.time_to_impact(g, pip_mode="vacuum")
        if t_sol is None:
            return None
        return np.array([
            self.x7[0] + self.x7[3] * t_sol,
            self.x7[1] + self.x7[4] * t_sol,
            0.0,
        ])

    def time_to_impact(self, g, pip_mode=None, drag_override=None):
        mode = pip_mode or self.pip_mode
        if mode == "drag":
            if drag_override is None and getattr(self, "_pip_cache", None) is not None:
                return self._pip_cache[1]
            k_d = self.est_drag if drag_override is None else drag_override
            res = predict_impact_drag(self.x7[:3], self.x7[3:6], k_d, g)
            if drag_override is None:
                self._pip_cache = res
            return res[1]
        if self.x7[5] >= 0:
            return None
        p_z, v_z = self.x7[2], self.x7[5]
        a = 0.5 * g
        b = -v_z
        c = -p_z
        d = b * b - 4 * a * c
        if d < 0:
            return None
        t_sol = (-b + np.sqrt(d)) / (2 * a)
        if t_sol <= 0:
            return None
        return t_sol


def make_filter(cfg):
    """Instantiate the configured tracking filter ('kf6' or 'ekf7')."""
    k = cfg.kalman
    if k.filter_type == "ekf7":
        return ExtendedKalmanFilter(
            cfg.sim.dt, cfg.sensor.pos_noise, cfg.sensor.vel_noise,
            k.q, cfg.sim.g,
            drag_init=k.drag_init, drag_std=k.drag_std,
            q_drag=k.q_drag, pip_mode=k.pip_mode,
        )
    return KalmanFilter(
        cfg.sim.dt, cfg.sensor.pos_noise, cfg.sensor.vel_noise,
        k.q, cfg.sim.g, pip_mode=k.pip_mode,
    )
