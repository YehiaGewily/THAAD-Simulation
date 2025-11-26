# fire_control.py
import numpy as np

class FireControl:
    def __init__(self, radar_system, interceptor):
        self.radar = radar_system
        self.interceptor = interceptor
        self.launched = False

    def update(self, detected_pos):
        # Simple Logic: Launch if target is detected and incoming
        # In a real system, we check if target is inside the "Kill Zone"
        if detected_pos is not None and not self.launched:
            # Check if target is roughly within range (e.g., < 100km away)
            dist = np.linalg.norm(detected_pos) 
            if dist < 100000:
                print(f"!!! THREAT DETECTED AT DISTANCE {dist:.0f}m - LAUNCHING INTERCEPTOR !!!")
                self.interceptor.launch()
                self.launched = True