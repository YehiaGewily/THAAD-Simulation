import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import time

# ============================================================================
# 1. MATHEMATICAL UTILITIES (Collision Physics)
# ============================================================================
def get_cpa_time_and_dist(pos1, vel1, pos2, vel2, dt):
    """
    Calculates the Closest Point of Approach (CPA) to solve the 'Tunneling' 
    problem where fast missiles skip over targets between frames.
    Returns: (time_offset, min_distance)
    """
    dp = pos2 - pos1
    dv = vel2 - vel1
    dv2 = np.dot(dv, dv)
    
    # If relative velocity is near zero, return current distance
    if dv2 < 1e-6: 
        return 0, np.linalg.norm(dp)
    
    # Calculate time to closest point
    t_min = -np.dot(dp, dv) / dv2
    
    # Check if the collision happens strictly within this time step (0 to dt)
    if 0 <= t_min <= dt:
        closest_p = dp + dv * t_min
        return t_min, np.linalg.norm(closest_p)
    
    # Otherwise check endpoints
    dist_end = np.linalg.norm(dp + dv * dt)
    dist_start = np.linalg.norm(dp)
    if dist_start < dist_end:
        return 0, dist_start
    else:
        return dt, dist_end

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
class SimConfig:
    dt = 0.02              # Simulation Time Step (s)
    t_max = 60.0           # Max Duration (s)
    g = 9.81               # Gravity (m/s^2)
    
    # Radar "Fog of War"
    pos_noise_std = 35.0   # Radar Position Uncertainty (m)
    vel_noise_std = 12.0   # Radar Velocity Uncertainty (m/s)
    
    # Interceptor Performance
    pn_gain = 4.0          # Proportional Navigation Gain (N)
    max_g = 30.0           # Max G-Force Limit
    speed = 2800.0         # Interceptor Speed (m/s)
    kill_radius = 8.0      # Hit-to-Kill Radius (m)
    launch_delay = 4.0     # Radar Lock-on Delay (s)

# ============================================================================
# 3. PHYSICS OBJECTS
# ============================================================================
class Object3D:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.acc = np.array([0., 0., 0.])
        self.active = True

    def update_physics(self, dt):
        if not self.active: return
        self.pos += self.vel * dt
        self.vel += self.acc * dt

class BallisticThreat(Object3D):
    def update(self, dt):
        if self.pos[2] < 0: self.active = False
        
        # Physics: Gravity + Atmospheric Drag Approximation
        v_mag = np.linalg.norm(self.vel)
        # Drag formula: a = -C * v * v_vec
        drag = -0.00004 * v_mag * self.vel 
        
        self.acc = np.array([0, 0, -SimConfig.g]) + drag
        self.update_physics(dt)

class Interceptor(Object3D):
    def __init__(self, pos):
        super().__init__(pos, [0,0,0])
        self.launched = False
        self.exploded = False
        self.g_history = [] # Store G-forces for post-analysis

    def launch(self, target_pred_pos):
        self.launched = True
        # Initial Aim: Point towards predicted location
        los = target_pred_pos - self.pos
        los = los / np.linalg.norm(los)
        self.vel = los * SimConfig.speed

    def guide(self, meas_pos, meas_vel, dt):
        if not self.launched or self.exploded: return

        # 1. Relative Kinematics (from noisy sensor data)
        r_vec = meas_pos - self.pos
        v_vec = meas_vel - self.vel
        range_scalar = np.linalg.norm(r_vec) + 1e-3
        
        # 2. Proportional Navigation (PN) Law
        # Omega = (R x V) / R^2
        omega = np.cross(r_vec, v_vec) / (range_scalar**2)
        
        # Acceleration Command: a = N * V_rel x Omega
        acc_cmd = SimConfig.pn_gain * np.cross(v_vec, omega)
        
        # 3. Gravity Compensation (Augmented PN)
        acc_cmd += np.array([0, 0, SimConfig.g])
        
        # 4. G-Force Limiting (Structural Limits)
        acc_mag = np.linalg.norm(acc_cmd)
        limit = SimConfig.max_g * 9.81
        if acc_mag > limit:
            acc_cmd = acc_cmd / acc_mag * limit
            
        self.g_history.append(acc_mag / 9.81)
        self.acc = acc_cmd
        self.update_physics(dt)

# ============================================================================
# 4. MAIN SIMULATION LOOP (WITH TERMINAL OUTPUT)
# ============================================================================
def run_simulation():
    # Initialize Objects
    threat = BallisticThreat([45000, 35000, 55000], [-650, -450, -250])
    interceptor = Interceptor([0, 0, 0])
    
    t = 0
    history = {'t': [], 't_pos': [], 'i_pos': [], 'r_pos': [], 'status': []}
    
    # Variables for Post-Mission Report
    outcome = "MISS"
    impact_time = 0.0
    impact_vel = 0.0
    min_dist = 99999.0

    print("\n" + "="*70)
    print(f"THAAD SIMULATION STARTED | TARGET ALT: {threat.pos[2]/1000:.1f} KM")
    print("="*70)
    print(f"{'TIME (s)':<10} {'STATUS':<15} {'RANGE (m)':<12} {'CLOSING (m/s)':<15} {'G-LOAD':<10}")
    print("-" * 70)

    while t < SimConfig.t_max:
        # A. SENSOR MODEL (Add Noise)
        noise_pos = np.random.normal(0, SimConfig.pos_noise_std, 3)
        noise_vel = np.random.normal(0, SimConfig.vel_noise_std, 3)
        meas_pos = threat.pos + noise_pos
        meas_vel = threat.vel + noise_vel
        
        # B. FIRE CONTROL LOGIC
        if t > SimConfig.launch_delay and not interceptor.launched:
            interceptor.launch(meas_pos)
            print(f"{t:<10.2f} {'LAUNCH':<15} {'---':<12} {'---':<15} {'0.0':<10}")

        # C. PHYSICS UPDATE & GUIDANCE
        if interceptor.launched and not interceptor.exploded:
            interceptor.guide(meas_pos, meas_vel, SimConfig.dt)

        # D. COLLISION DETECTION (CPA)
        if interceptor.launched:
            t_offset, frame_min_dist = get_cpa_time_and_dist(
                interceptor.pos, interceptor.vel, 
                threat.pos, threat.vel, SimConfig.dt
            )
            
            # Check Hit
            if frame_min_dist < SimConfig.kill_radius:
                interceptor.exploded = True
                outcome = "KILL"
                impact_time = t + t_offset
                min_dist = frame_min_dist
                
                # Move objects to exact impact point for visual clarity
                interceptor.pos += interceptor.vel * t_offset
                threat.pos += threat.vel * t_offset
                
                # Calculate relative impact velocity
                impact_vel = np.linalg.norm(interceptor.vel - threat.vel)
                
                # Log final frame
                history['t'].append(impact_time)
                history['t_pos'].append(threat.pos.copy())
                history['i_pos'].append(interceptor.pos.copy())
                history['r_pos'].append(meas_pos.copy())
                history['status'].append("HIT")
                
                print(f"{impact_time:<10.2f} {'*** KILL ***':<15} {min_dist:<12.2f} {impact_vel:<15.0f} {'---':<10}")
                break # STOP SIMULATION

        threat.update(SimConfig.dt)
        
        # E. TERMINAL LOGGING (Every 1.0 second)
        current_dist = np.linalg.norm(threat.pos - interceptor.pos)
        if interceptor.launched and not interceptor.exploded:
            # Print every ~50 frames (1 second)
            if int(t / SimConfig.dt) % 50 == 0:
                closure = np.linalg.norm(threat.vel - interceptor.vel)
                g_load = interceptor.g_history[-1] if interceptor.g_history else 0.0
                print(f"{t:<10.1f} {'GUIDING':<15} {current_dist:<12.0f} {closure:<15.0f} {g_load:<10.1f}")
        
        # Save History
        history['t'].append(t)
        history['t_pos'].append(threat.pos.copy())
        history['i_pos'].append(interceptor.pos.copy())
        history['r_pos'].append(meas_pos.copy())
        history['status'].append("GUIDING" if interceptor.launched else "SEARCH")
        
        if not threat.active:
            outcome = "GROUND_IMPACT"
            print(f"{t:<10.2f} {'FAIL':<15} Target hit ground.")
            break
            
        t += SimConfig.dt

    # ========================================================================
    # 5. POST-MISSION REPORT
    # ========================================================================
    print("\n" + "="*70)
    print("           BATTLE DAMAGE ASSESSMENT (BDA)")
    print("="*70)
    print(f"MISSION OUTCOME   : {outcome}")
    print(f"INTERCEPT TIME    : {impact_time:.2f} s")
    print(f"INTERCEPT RANGE   : {np.linalg.norm(interceptor.pos):.0f} m (Slant Range)")
    print(f"INTERCEPT ALTITUDE: {threat.pos[2]/1000:.2f} km")
    print(f"CLOSING VELOCITY  : {impact_vel:.0f} m/s (Mach {impact_vel/343:.1f})")
    print(f"MISS DISTANCE     : {min_dist:.2f} m")
    if interceptor.g_history:
        print(f"MAX G-LOAD PULLED : {max(interceptor.g_history):.2f} G")
    print("="*70 + "\n")
    
    return history

# ============================================================================
# 6. VISUALIZATION (Animation)
# ============================================================================
if __name__ == "__main__":
    history = run_simulation()

    # Convert to numpy arrays for plotting
    t_data = np.array(history['t'])
    t_pos = np.array(history['t_pos'])
    i_pos = np.array(history['i_pos'])
    r_pos = np.array(history['r_pos'])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title("THAAD Engagement Replay\nKinetic Hit Verification", fontsize=14)
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_xlim(-10000, 60000)
    ax.set_ylim(-10000, 60000)
    ax.set_zlim(0, 60000)

    # Plot Elements
    l_threat, = ax.plot([], [], [], 'r--', alpha=0.5, label='Threat')
    l_inter, = ax.plot([], [], [], 'b-', linewidth=2, label='Interceptor')
    head_t, = ax.plot([], [], [], 'ro', markersize=6)
    head_i, = ax.plot([], [], [], 'b^', markersize=6)
    ghost, = ax.plot([], [], [], 'k.', markersize=2, alpha=0.2, label='Radar Noise')
    expl, = ax.plot([], [], [], 'y*', markersize=30, markeredgecolor='orange', visible=False, label='Kill')
    
    # Ground Battery
    ax.scatter([0], [0], [0], c='k', marker='s', s=100, label='Launcher')
    
    # HUD Elements
    txt_stat = ax.text2D(0.05, 0.95, "REPLAY READY", transform=ax.transAxes, fontweight='bold')
    txt_info = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)

    def update(frame):
        # Stop animation if we hit the end
        if frame >= len(t_data): 
            return l_threat, l_inter, head_t, head_i, ghost, expl, txt_stat, txt_info

        # If HIT, freeze frame and show explosion
        if history['status'][frame] == "HIT":
            txt_stat.set_text("STATUS: KINETIC KILL - STOPPED")
            txt_stat.set_color("red")
            expl.set_data([t_pos[frame,0]], [t_pos[frame,1]])
            expl.set_3d_properties([t_pos[frame,2]])
            expl.set_visible(True)
            head_t.set_visible(False)
            head_i.set_visible(False)
            return l_threat, l_inter, head_t, head_i, ghost, expl, txt_stat, txt_info

        # Update Paths
        l_threat.set_data(t_pos[:frame,0], t_pos[:frame,1])
        l_threat.set_3d_properties(t_pos[:frame,2])
        l_inter.set_data(i_pos[:frame,0], i_pos[:frame,1])
        l_inter.set_3d_properties(i_pos[:frame,2])
        
        # Update Heads
        head_t.set_data([t_pos[frame,0]], [t_pos[frame,1]])
        head_t.set_3d_properties([t_pos[frame,2]])
        head_i.set_data([i_pos[frame,0]], [i_pos[frame,1]])
        head_i.set_3d_properties([i_pos[frame,2]])
        
        # Update Radar Ghost
        s = max(0, frame-15)
        ghost.set_data(r_pos[s:frame,0], r_pos[s:frame,1])
        ghost.set_3d_properties(r_pos[s:frame,2])
        
        # Update HUD
        status = history['status'][frame]
        txt_stat.set_text(f"STATUS: {status}")
        txt_stat.set_color("blue" if status=="GUIDING" else "black")
        
        alt = t_pos[frame,2]
        txt_info.set_text(f"Threat Alt: {alt:.0f} m")
        
        return l_threat, l_inter, head_t, head_i, ghost, expl, txt_stat, txt_info

    anim = FuncAnimation(fig, update, frames=len(t_data)+10, interval=20, blit=False, repeat=False)
    plt.legend(loc='upper right')
    plt.show()