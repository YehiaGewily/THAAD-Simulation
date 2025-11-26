import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import config
from target import BallisticTarget
from missile import THAADInterceptor
from radar import RadarSystem

def main():
    # 1. Initialize Objects
    target = BallisticTarget()
    missile = THAADInterceptor()
    radar = RadarSystem()
    
    # Data recording
    t_log, m_log = [], []
    sim_time = 0
    intercepted = False
    
    print("--- COMPUTING INTERCEPT SOLUTION ---")
    
    # 2. Simulation Loop (Calculate everything first)
    while sim_time < config.TMAX:
        # Store PREVIOUS positions (Crucial for anti-tunneling math)
        t_pos_prev = target.pos.copy()
        m_pos_prev = missile.pos.copy()
        
        # A. Update Physics
        # We store the state BEFORE checking bounds to allow for "last second" intercepts
        target_alive = target.update(config.DT)
        
        track = radar.scan(sim_time, target.pos)
        missile.update(config.DT, track)
        
        # B. Check Intercept (Geometric CPA Method)
        if missile.launched:
            # Current positions
            t_pos_curr = target.pos
            m_pos_curr = missile.pos
            
            # Calculate geometric overlap
            # We treat movements as line segments P_prev -> P_curr
            # We solve for the time 'u' (0.0 to 1.0) where distance was minimized
            
            # Relative position at start of frame
            p_rel_prev = t_pos_prev - m_pos_prev
            
            # Relative movement vector during this frame
            v_rel_frame = (t_pos_curr - t_pos_prev) - (m_pos_curr - m_pos_prev)
            
            # Squared length of relative movement
            v_rel_sq = np.dot(v_rel_frame, v_rel_frame)
            
            if v_rel_sq > 0:
                # Calculate time fraction 'u' where distance is minimized (derivative = 0)
                # u = - (P_rel . V_rel) / |V_rel|^2
                u = -np.dot(p_rel_prev, v_rel_frame) / v_rel_sq
                
                # Clamp u to the current time step (between 0.0 and 1.0)
                u_clamped = max(0.0, min(1.0, u))
                
                # Find the exact point of closest approach
                p_closest = p_rel_prev + v_rel_frame * u_clamped
                min_dist = np.linalg.norm(p_closest)
                
                # Use strict kill distance (no need for dynamic expansion anymore)
                if min_dist < config.KILL_DISTANCE:
                    interpolated_time = sim_time - config.DT + (u_clamped * config.DT)
                    print(f"*** SPLASH *** Target Destroyed at T={interpolated_time:.2f}s | CPA Dist: {min_dist:.2f}m")
                    intercepted = True
                    
                    # Log the exact hit point for visualization
                    hit_t_pos = t_pos_prev + (t_pos_curr - t_pos_prev) * u_clamped
                    hit_m_pos = m_pos_prev + (m_pos_curr - m_pos_prev) * u_clamped
                    
                    t_log.append(hit_t_pos)
                    m_log.append(hit_m_pos)
                    break

        # C. Check Ground Impact (Delayed to allow intercept check first)
        if not target_alive:
            print("Target Impacted Ground.")
            t_log.append(target.pos.copy())
            m_log.append(missile.pos.copy())
            break

        # D. Log Data
        t_log.append(target.pos.copy())
        m_log.append(missile.pos.copy())
        sim_time += config.DT
        
    # 3. Playback Animation
    print("--- STARTING REPLAY ---")
    run_replay(np.array(t_log), np.array(m_log), intercepted)

def run_replay(t_data, m_data, success):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Setup Lines
    t_line, = ax.plot([], [], [], 'r--', label='Threat')
    m_line, = ax.plot([], [], [], 'b-', label='THAAD')
    t_dot, = ax.plot([], [], [], 'ro', markersize=5)
    m_dot, = ax.plot([], [], [], 'b^', markersize=5)
    
    # Setup Axis
    ax.set_xlim(-10000, 50000)
    ax.set_ylim(-25000, 25000)
    ax.set_zlim(0, 40000)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Cross Range (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title("THAAD Engagement Replay")
    ax.legend()
    
    # Decimation (Skip frames to make animation faster)
    skip = 5 
    frames = len(t_data) // skip

    def update(frame):
        idx = frame * skip
        if idx >= len(t_data): idx = len(t_data) - 1
        
        # Update Target
        ax.set_title(f"Time: {idx * config.DT:.2f}s")
        t_curr = t_data[idx]
        t_line.set_data(t_data[:idx, 0], t_data[:idx, 1])
        t_line.set_3d_properties(t_data[:idx, 2])
        t_dot.set_data([t_curr[0]], [t_curr[1]])
        t_dot.set_3d_properties([t_curr[2]])
        
        # Update Missile (only if launched)
        if idx < len(m_data):
            m_curr = m_data[idx]
            # Only plot if missile has moved from origin
            if np.linalg.norm(m_curr) > 10:
                m_line.set_data(m_data[:idx, 0], m_data[:idx, 1])
                m_line.set_3d_properties(m_data[:idx, 2])
                m_dot.set_data([m_curr[0]], [m_curr[1]])
                m_dot.set_3d_properties([m_curr[2]])

        # Explosion on last frame
        if success and frame == frames - 1:
            ax.scatter([t_curr[0]], [t_curr[1]], [t_curr[2]], s=500, c='orange', marker='*')
            
        return t_line, m_line, t_dot, m_dot

    anim = FuncAnimation(fig, update, frames=frames, interval=20, blit=False)
    plt.show()

if __name__ == "__main__":
    main()