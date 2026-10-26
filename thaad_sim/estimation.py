"""Threat tracking: a 6-state Kalman filter.

The filter models the threat as a constant-acceleration body under known
gravity, perturbed by continuous white-noise acceleration (CWNA):

    x_{k+1} = F x_k + B u + w_k,    u = [0, 0, -g],    w_k ~ N(0, Q)
    z_k     = H x_k + v_k,          H = I,             v_k ~ N(0, R)

with x = [p, v]. It knows nothing about drag; see the README's model-mismatch
section for what that costs.
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


class KalmanFilter:
    """Linear Kalman filter with gravity as a known control input."""
    def __init__(self, dt, pos_std, vel_std, q, g):
        self.dt = dt
        self.g = g
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

    def update(self, meas_pos, meas_vel):
        z = np.hstack((meas_pos, meas_vel))
        if not self.initialized:
            # The first measurement is the estimate, so its covariance is R
            self.x = z
            self.P = self.R.copy()
            self.initialized = True
            return self.x[:3], self.x[3:]

        # Predict
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update. K = P H^T S^-1 comes from a linear solve, not an explicit inverse.
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, self.H @ self.P).T
        self.nis = float(y @ np.linalg.solve(S, y))
        self.x = self.x + K @ y
        # Joseph form: symmetric and positive definite for any gain K
        A = np.eye(6) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T

        return self.x[:3], self.x[3:]

    def nees(self, x_true):
        """Normalized estimation error squared, e^T P^-1 e with e = x_hat - x_true."""
        e = self.x - np.asarray(x_true)
        return float(e @ np.linalg.solve(self.P, e))

    def predict_impact(self, g):
        """Predict impact point on z=0 from current state, assuming drag-free flight."""
        t_sol = self.time_to_impact(g)
        if t_sol is None:
            return None
        return np.array([
            self.x[0] + self.x[3]*t_sol,
            self.x[1] + self.x[4]*t_sol,
            0.0
        ])

    def time_to_impact(self, g):
        """Drag-free time until z = 0, or None while the estimate is not descending."""
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
