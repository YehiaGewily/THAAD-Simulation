import numpy as np
import config

class RadarSystem:
    def __init__(self):
        self.tracked_target = None
        self.last_update_time = 0
        self.scan_count = 0
        
    def scan(self, sim_time, true_target_pos):
        """
        Simulates the AN/TPY-2 Radar.
        Returns 'None' if target is undetected, or a 'Track' object if detected.
        """
        # 1. Check Range
        dist = np.linalg.norm(true_target_pos)
        if dist > config.RADAR_RANGE:
            return None # Target out of range
            
        # 2. Update Rate Limit (Radar scans at specific intervals, not every physics tick)
        if sim_time - self.last_update_time < config.TRACK_UPDATE_RATE:
            return self.tracked_target # Return last known data
            
        # 3. Simulate Sensor Noise
        # Real radars have error. We add Gaussian noise to the position.
        noise = np.random.normal(0, config.RADAR_NOISE_STD, 3)
        measured_pos = true_target_pos + noise
        
        # 4. Calculate Velocity (Doppler / Delta-Position)
        measured_vel = np.array([0.0, 0.0, 0.0])
        if self.tracked_target is not None:
            dt = sim_time - self.last_update_time
            prev_pos = self.tracked_target['pos']
            measured_vel = (measured_pos - prev_pos) / dt
            
        # Update Track
        self.tracked_target = {
            'pos': measured_pos,
            'vel': measured_vel,
            'time': sim_time
        }
        self.last_update_time = sim_time
        
        return self.tracked_target