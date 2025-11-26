# missile_interceptor.py
import numpy as np
from guidance import Guidance

class Interceptor:
    def __init__(self, start_pos):
        self.pos = np.array(start_pos, dtype=float)
        self.vel = np.array([0.0, 0.0, 10.0]) # Small initial vertical velocity
        
        self.mass = 300.0
        self.thrust = 15000.0 # Newtons
        self.burn_time = 10.0 # Seconds of fuel
        self.time_elapsed = 0
        
        self.guidance = Guidance()
        self.is_launched = False
        self.history = []

    def launch(self):
        self.is_launched = True

    def update(self, dt, env, target_pos, target_vel):
        if not self.is_launched:
            return

        self.time_elapsed += dt

        # 1. Guidance Command (PN)
        accel_cmd = self.guidance.proportional_navigation(
            self.pos, self.vel, target_pos, target_vel
        )

        # 2. Limit Structural G-Force (e.g., max 30 Gs)
        max_g = 30 * 9.81
        cmd_mag = np.linalg.norm(accel_cmd)
        if cmd_mag > max_g:
            accel_cmd = accel_cmd / cmd_mag * max_g

        # 3. Physics Forces
        accel_g = env.get_gravity_vector()
        
        # Motor Thrust
        if self.time_elapsed < self.burn_time:
            # Thrust adds to velocity vector, guided by accel_cmd
            # We simplify here: Thrust assumes it can push in the guided direction
            # Ideally, we rotate the body, but for 3-DOF point mass, we add thrust to velocity dir
            # and lateral forces from fins (accel_cmd)
            v_norm = self.vel / np.linalg.norm(self.vel)
            accel_thrust = (self.thrust / self.mass) * v_norm
        else:
            accel_thrust = np.array([0.0, 0.0, 0.0])

        total_accel = accel_g + accel_thrust + accel_cmd

        # Integration
        self.vel += total_accel * dt
        self.pos += self.vel * dt
        
        self.history.append(self.pos.copy())