# missile_threat.py
import numpy as np

class BallisticMissile:
    def __init__(self):
        # Randomize launch parameters for variety
        # Launching from x=0, y=0
        self.pos = np.array([0.0, 0.0, 0.0])
        
        # Random speed between 800 m/s and 1200 m/s
        speed = np.random.uniform(800, 1200)
        
        # Random angle (loft)
        angle_elev = np.radians(np.random.uniform(30, 70))
        angle_azi = np.radians(np.random.uniform(0, 45))
        
        # Velocity components
        vx = speed * np.cos(angle_elev) * np.cos(angle_azi)
        vy = speed * np.cos(angle_elev) * np.sin(angle_azi)
        vz = speed * np.sin(angle_elev)
        
        self.vel = np.array([vx, vy, vz])
        
        # Physics params
        self.mass = 1000.0  # kg
        self.area = 0.5     # Cross sectional area
        self.cd = 0.3       # Drag coefficient
        self.is_active = True
        self.history = []   # For visualization

    def update(self, dt, env):
        if not self.is_active:
            return

        # 1. Gravity
        accel_g = env.get_gravity_vector()

        # 2. Drag (Aerodynamics)
        # F_drag = 0.5 * rho * v^2 * Cd * A
        h = self.pos[2]
        rho = env.get_air_density(h)
        v_mag = np.linalg.norm(self.vel)
        
        if v_mag > 0:
            drag_force_mag = 0.5 * rho * (v_mag**2) * self.cd * self.area
            drag_vec = -drag_force_mag * (self.vel / v_mag) # Opposite to velocity
            accel_drag = drag_vec / self.mass
        else:
            accel_drag = np.array([0,0,0])

        # Total acceleration
        total_accel = accel_g + accel_drag

        # Integration (Euler)
        self.vel += total_accel * dt
        self.pos += self.vel * dt
        
        # Ground impact check
        if self.pos[2] < 0:
            self.pos[2] = 0
            self.is_active = False
            
        self.history.append(self.pos.copy())