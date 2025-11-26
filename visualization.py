import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import config

def run_animation(times, target_states, missile_states, intercept_idx):
    """
    Sets up and runs the matplotlib 3D animation.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Decimate Data (Plotting every single point is slow)
    skip = 10 
    
    # Truncate arrays if intercept happened to stop animation shortly after hit
    end_idx = len(times)
    if intercept_idx != -1:
        end_idx = min(len(times), intercept_idx + 50) # Show a bit after hit
        
    t_data = target_states[:end_idx:skip]
    m_data = missile_states[:end_idx:skip]
    
    # 2. Static Background Paths
    ax.plot(target_states[:,0], target_states[:,1], target_states[:,2], 
            'r--', alpha=0.3, label='Threat Corridor')
    ax.plot(missile_states[:end_idx,0], missile_states[:end_idx,1], missile_states[:end_idx,2], 
            'b-', alpha=0.3, label='Interceptor Flight Path')
            
    # 3. Dynamic Objects (The dots moving)
    t_dot, = ax.plot([], [], [], 'ro', markersize=8, label='Threat')
    m_dot, = ax.plot([], [], [], 'b^', markersize=8, label='THAAD')
    m_trail, = ax.plot([], [], [], 'b-', linewidth=2)
    
    # 4. Status Text
    status_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
    
    # 5. Axis Formatting
    # Calculate bounds to keep everything centered
    all_p = np.vstack([target_states, missile_states])
    center = np.mean(all_p, axis=0)
    radius = np.max(np.linalg.norm(all_p - center, axis=1))
    
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(0, 25000)
    ax.set_xlabel('Range X (m)')
    ax.set_ylabel('Cross Range Y (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title(f'THAAD Simulation | Proportional Nav (N={config.GUIDANCE_GAIN})')
    ax.legend()
    
    # 6. Update Function
    def update(frame):
        # Update Target
        t_curr = t_data[frame]
        t_dot.set_data([t_curr[0]], [t_curr[1]])
        t_dot.set_3d_properties([t_curr[2]])
        
        # Update Missile
        m_curr = m_data[frame]
        m_dot.set_data([m_curr[0]], [m_curr[1]])
        m_dot.set_3d_properties([m_curr[2]])
        
        # Update Trail (Last 20 frames)
        start_trail = max(0, frame-20)
        trail_slice = m_data[start_trail:frame+1]
        m_trail.set_data(trail_slice[:,0], trail_slice[:,1])
        m_trail.set_3d_properties(trail_slice[:,2])
        
        # Update Text
        sim_time = frame * config.DT * skip
        dist = np.linalg.norm(t_curr - m_curr)
        
        status_msg = f"Time: {sim_time:.1f}s\nSeparation: {dist:.0f}m"
        
        # Check for hit visualization
        real_idx = frame * skip
        if intercept_idx != -1 and real_idx >= intercept_idx:
             status_msg += "\n\n*** KILL CONFIRMED ***"
             ax.scatter([m_curr[0]], [m_curr[1]], [m_curr[2]], 
                       s=500, c='orange', marker='*', alpha=0.5)
        
        status_text.set_text(status_msg)
        return t_dot, m_dot, m_trail, status_text

    # 7. Start Animation
    anim = FuncAnimation(fig, update, frames=len(m_data), interval=20, blit=False)
    plt.show()