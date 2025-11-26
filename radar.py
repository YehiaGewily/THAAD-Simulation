# radar.py
import numpy as np

class Radar:
    def __init__(self, update_rate=0.1, noise_std=50.0):
        self.update_rate = update_rate
        self.noise_std = noise_std  # Standard deviation of noise in meters
        self.last_scan_time = 0
        self.tracked_target = None

    def scan(self, time, true_target_pos):
        # Simulate discrete radar pings
        if time - self.last_scan_time >= self.update_rate:
            self.last_scan_time = time
            
            # Add noise (measurement error)
            noise = np.random.normal(0, self.noise_std, 3)
            measured_pos = true_target_pos + noise
            
            return measured_pos
        return None