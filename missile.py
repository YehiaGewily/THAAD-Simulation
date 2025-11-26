import numpy as np
import config

class THAADInterceptor:
    def __init__(self):
        self.pos = config.MISSILE_START_POS.astype(float)
        self.vel = np.array([0.0, 0.0, 0.0])
        self.launched = False
        self.launch_timer = 0
        self.active = True
        
    def update(self, dt, radar_track):
        # 1. Launch Logic
        if not self.launched:
            self.launch_timer += dt
            if self.launch_timer >= config.LAUNCH_DELAY and radar_track is not None:
                self.fire(radar_track)
            return
            
        if not self.active: return

        # 2. Guidance Logic (Proportional Navigation)
        if radar_track:
            target_pos = radar_track['pos']
            target_vel = radar_track['vel']
            
            # Relative vectors
            r_tm = target_pos - self.pos
            range_dist = np.linalg.norm(r_tm)
            v_tm = target_vel - self.vel
            
            # Proportional Navigation
            vc = -np.dot(v_tm, r_tm) / range_dist # Closing velocity
            omega = np.cross(r_tm, v_tm) / (range_dist**2) # LOS Rate
            
            # Acc Cmd = N * Vc * Omega
            # Apply perpendicular to missile heading
            speed = np.linalg.norm(self.vel)
            heading = self.vel / speed if speed > 0 else np.array([0,0,1])
            
            acc_cmd = config.GUIDANCE_GAIN * vc * np.cross(omega, heading)
            
            # G-Limiter
            acc_mag = np.linalg.norm(acc_cmd)
            if acc_mag > config.MAX_G_LOAD:
                acc_cmd = (acc_cmd / acc_mag) * config.MAX_G_LOAD
                
            # 3. Physics Update
            self.vel += acc_cmd * dt
            # Normalize speed (Energy management simplified)
            self.vel = (self.vel / np.linalg.norm(self.vel)) * config.MISSILE_VELOCITY
            self.pos += self.vel * dt

    def fire(self, radar_track):
        print("THAAD LAUNCH DETECTED!")
        self.launched = True
        
        # Initial Guess: Aim at the target
        t_pos = radar_track['pos']
        aim_vec = t_pos - self.pos
        aim_dir = aim_vec / np.linalg.norm(aim_vec)
        
        self.vel = aim_dir * config.MISSILE_VELOCITY