"""Threat tracking: a 6-state Kalman filter."""
import numpy as np


class KalmanFilter:
    """Standard 6-state Kalman filter with simple gravity compensation."""
    def __init__(self, dt, pos_std, vel_std, q, p0, g):
        self.dt = dt
        self.g = g
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt

        self.H = np.eye(6)

        self.Q = np.eye(6) * q
        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = pos_std**2
            self.R[i+3, i+3] = vel_std**2

        self.P = np.eye(6) * p0
        self.x = np.zeros(6)
        self.initialized = False

    def update(self, meas_pos, meas_vel):
        z = np.hstack((meas_pos, meas_vel))
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x[:3], self.x[3:]

        # Predict
        self.x = self.F @ self.x
        self.x[5] -= self.g * self.dt        # simple gravity model on vz
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3], self.x[3:]

    def predict_impact(self, g):
        """Predict impact point on z=0 from current state."""
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
        return np.array([
            self.x[0] + self.x[3]*t_sol,
            self.x[1] + self.x[4]*t_sol,
            0.0
        ])
