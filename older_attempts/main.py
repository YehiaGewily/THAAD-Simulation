# main.py
import numpy as np
from environment import Environment
from missile_threat import BallisticMissile
from missile_interceptor import Interceptor
from radar import Radar
from fire_control import FireControl
from visualization import Visualizer

def run_simulation():
    # 1. Setup
    env = Environment()
    visualizer = Visualizer()
    
    # 2. Init Threat (Incoming from random pos)
    threat = BallisticMissile()
    
    # 3. Init Defense System
    # Place radar/interceptor at a location "down range" relative to threat launch
    # Let's say threat targets x=50000. We place defense near there.
    defense_pos = np.array([40000.0, 5000.0, 0.0]) 
    
    radar = Radar(update_rate=0.5, noise_std=10.0)
    interceptor = Interceptor(start_pos=defense_pos)
    fcs = FireControl(radar, interceptor)

    # 4. Sim Loop
    dt = 0.05 # 50ms time step
    max_time = 200 # seconds
    time = 0
    intercept_successful = False
    intercept_pos = None

    print("--- Simulation Started ---")
    
    while time < max_time:
        # A. Update Physics
        threat.update(dt, env)
        interceptor.update(dt, env, threat.pos, threat.vel)
        
        # B. Radar & Fire Control
        measured_pos = radar.scan(time, threat.pos)
        fcs.update(measured_pos)
        
        # C. Collision Check (Hit-to-Kill)
        if interceptor.is_launched:
            dist = np.linalg.norm(threat.pos - interceptor.pos)
            # Threshold for "Kill" (e.g., 10 meters)
            if dist < 15.0: 
                print(f"TARGET DESTROYED at T={time:.2f}s, Alt={threat.pos[2]:.0f}m")
                intercept_successful = True
                intercept_pos = threat.pos
                break
        
        # D. Ground Impact Check
        if not threat.is_active:
            print("Threat hit the ground!")
            break

        time += dt

    # 5. Visualization
    visualizer.plot_simulation(threat.history, interceptor.history, intercept_pos)

if __name__ == "__main__":
    run_simulation()