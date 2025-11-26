import numpy as np
import random
import config

class BallisticTarget:
    def __init__(self):
        # Random Start Position
        x = random.uniform(config.X_START_MIN, config.X_START_MAX)
        y = random.uniform(config.Y_START_MIN, config.Y_START_MAX)
        z = random.uniform(config.Z_START_MIN, config.Z_START_MAX)
        self.pos = np.array([x, y, z], dtype=float)
        
        # Random Velocity (Flying generally towards 0,0,0)
        speed = random.uniform(config.TARGET_SPEED_MIN, config.TARGET_SPEED_MAX)
        target_vec = -self.pos # Vector pointing to origin
        norm_vec = target_vec / np.linalg.norm(target_vec)
        self.vel = norm_vec * speed
        
        # Maneuver State
        self.is_maneuvering = False
        self.maneuver_duration = 0
        self.maneuver_acc = np.array([0.0, 0.0, 0.0])
        
        print(f"THREAT DETECTED AT: {self.pos.astype(int)}")

    def update(self, dt):
        """
        Updates physics by one time step.
        Can randomly decide to jink/maneuver.
        """
        # 1. Random Maneuver Logic
        if not self.is_maneuvering:
            if random.random() < config.MANEUVER_CHANCE * dt:
                # Start a random turn
                self.is_maneuvering = True
                self.maneuver_duration = random.uniform(3.0, 8.0)
                
                # Random acceleration vector (perpendicular to flight)
                side_vec = np.cross(self.vel, np.array([0,0,1]))
                if np.linalg.norm(side_vec) < 0.1: side_vec = np.array([0,1,0])
                side_vec = side_vec / np.linalg.norm(side_vec)
                
                g_pull = random.uniform(3, 8) * 9.81 # 3G to 8G turn
                direction = random.choice([-1, 1])
                self.maneuver_acc = side_vec * g_pull * direction
                print(f"Target Maneuver! {g_pull/9.81:.1f}G Turn")
        else:
            self.maneuver_duration -= dt
            if self.maneuver_duration <= 0:
                self.is_maneuvering = False
                self.maneuver_acc = np.array([0.0, 0.0, 0.0])

        # 2. Physics Update
        # Acceleration = Gravity + Maneuver
        acc = np.array([0, 0, -config.G]) + self.maneuver_acc
        
        self.vel += acc * dt
        self.pos += self.vel * dt
        
        # Ground check
        if self.pos[2] < 0:
            self.pos[2] = 0
            return False # Crashed
        return True