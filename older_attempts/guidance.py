# guidance.py
import numpy as np

class Guidance:
    def __init__(self, n_gain=4.0):
        self.N = n_gain  # Navigation constant (usually 3-5)

    def proportional_navigation(self, m_pos, m_vel, t_pos, t_vel):
        """
        Calculates acceleration command using ZEM/ZEV or pure Vector PN.
        """
        # Relative position and velocity
        r_tm = t_pos - m_pos  # Range vector
        v_tm = t_vel - m_vel  # Closing velocity vector
        
        r_mag = np.linalg.norm(r_tm)
        if r_mag == 0:
            return np.array([0,0,0])

        # Rotation vector of the Line of Sight (Omega)
        # Omega = (R x V) / (R . R)
        omega = np.cross(r_tm, v_tm) / (np.dot(r_tm, r_tm))
        
        # PN Acceleration Command: a = N * (V_closing x Omega)
        # We use negative velocity for pure closing speed logic relative to missile frame
        accel_cmd = self.N * np.cross(v_tm, omega)
        
        return accel_cmd