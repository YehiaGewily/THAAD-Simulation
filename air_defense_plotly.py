import numpy as np
import plotly.graph_objects as go

# ============================================================================
# 1. PHYSICS & MATH (CORRECTED)
# ============================================================================

class Environment:
    def __init__(self):
        self.g = 9.81
        self.rho_0 = 1.225
        self.scale_height = 8500

    def get_air_density(self, altitude):
        if altitude < 0: return self.rho_0
        return self.rho_0 * np.exp(-altitude / self.scale_height)

def augmented_proportional_navigation(m_pos, m_vel, t_pos, t_vel, N=5.0): # Increased Gain to 5
    """
    Augmented PN (APN).
    Adds a gravity compensation term so the missile doesn't 'sag' under the target.
    """
    r_tm = t_pos - m_pos # Range Vector
    v_tm = t_vel - m_vel # Closing Velocity
    
    r_mag = np.linalg.norm(r_tm)
    if r_mag < 1.0: return np.array([0,0,0])

    # 1. Standard PN Rotation Vector (Omega)
    # Omega = (Range x ClosingVel) / Range^2
    omega = np.cross(r_tm, v_tm) / np.dot(r_tm, r_tm)
    
    # 2. Base Acceleration Command
    # a = N * (V_closing x Omega)
    pn_accel = N * np.cross(v_tm, omega)
    
    # 3. Augmented Term (Gravity Compensation)
    # We add 1g upwards to the command so the fins fight gravity automatically
    # This assumes the missile can pull Gs in the vertical plane
    gravity_comp = np.array([0, 0, 9.81]) 
    
    return pn_accel + gravity_comp

# ============================================================================
# 2. SIMULATION GENERATOR
# ============================================================================

def generate_intercept_data():
    print("Generating Physics Data...")
    
    dt = 0.05
    max_time = 150
    env = Environment()
    
    # --- THREAT (Incoming Ballistic Missile) ---
    threat_pos = np.array([0.0, 0.0, 0.0])
    # Target 60km away
    speed = 1200.0
    angle_elev = np.radians(55) 
    angle_azi = np.radians(0) # Flying along X axis
    
    threat_vel = np.array([
        speed * np.cos(angle_elev) * np.cos(angle_azi),
        speed * np.cos(angle_elev) * np.sin(angle_azi),
        speed * np.sin(angle_elev)
    ])
    threat_mass = 1000.0
    threat_area = 0.5
    threat_cd = 0.3

    # --- INTERCEPTOR (The Fix) ---
    # Placed at X=60km (Target area), launching back towards X=0
    interceptor_pos = np.array([50000.0, 0.0, 0.0]) 
    
    # FIX 1: "Tip Over" Launch
    # Don't launch [0,0,10]. Launch slightly towards threat [-10, 0, 10]
    # This helps the main motor push us in the right direction early.
    interceptor_vel = np.array([-10.0, 0.0, 20.0]) 
    
    interceptor_mass = 350.0 
    interceptor_thrust = 40000.0 # FIX 2: More Power (was 25000)
    burn_time = 8.0 
    
    launched = False
    launch_delay = 5.0 # FIX 3: Launch sooner (was 15.0)
    
    t_hist = []
    threat_hist = []
    interceptor_hist = []
    events = []

    time = 0
    intercepted = False
    
    while time < max_time:
        # --- Update Threat ---
        h = threat_pos[2]
        if h < 0 and time > 1: break # Ground impact
            
        rho = env.get_air_density(h)
        v_mag = np.linalg.norm(threat_vel)
        drag_force = 0.5 * rho * v_mag**2 * threat_cd * threat_area
        drag_accel = -(drag_force / threat_mass) * (threat_vel / v_mag)
        grav_accel = np.array([0, 0, -env.g])
        
        threat_vel += (drag_accel + grav_accel) * dt
        threat_pos += threat_vel * dt

        # --- Update Interceptor ---
        if time >= launch_delay and not launched:
            launched = True
            events.append(f"LAUNCH at {time}s")

        if launched and not intercepted:
            # Guidance (Augmented)
            accel_cmd = augmented_proportional_navigation(interceptor_pos, interceptor_vel, threat_pos, threat_vel)
            
            # Thrust Logic
            thrust_accel = np.array([0.0,0.0,0.0])
            if time < (launch_delay + burn_time):
                # Thrust is aligned with current velocity (simplified body frame)
                v_norm = interceptor_vel / np.linalg.norm(interceptor_vel)
                thrust_accel = (interceptor_thrust / interceptor_mass) * v_norm

            # Total Forces
            total_accel = grav_accel + thrust_accel + accel_cmd
            
            interceptor_vel += total_accel * dt
            interceptor_pos += interceptor_vel * dt
            
            # Collision Check
            dist = np.linalg.norm(threat_pos - interceptor_pos)
            # 20m Kill Radius
            if dist < 20.0: 
                intercepted = True
                print(f"!!! KILL !!! Intercept at T={time:.2f}s, Alt={h:.0f}m, Miss Dist={dist:.2f}m")

        # --- Recording ---
        t_hist.append(time)
        threat_hist.append(threat_pos.copy())
        
        if not launched:
            interceptor_hist.append(interceptor_pos.copy())
        elif intercepted:
            # Stop recording interceptor (it exploded)
            interceptor_hist.append(interceptor_pos.copy())
            # Also stop threat (it exploded)
            break 
        else:
            interceptor_hist.append(interceptor_pos.copy())

        time += dt

    return np.array(t_hist), np.array(threat_hist), np.array(interceptor_hist)

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

def visualize_simulation():
    times, t_pos, i_pos = generate_intercept_data()
    
    # Decimate for speed
    step = 4
    times = times[::step]
    t_pos = t_pos[::step]
    i_pos = i_pos[::step]

    fig = go.Figure()

    # Full Static Paths (Background reference)
    fig.add_trace(go.Scatter3d(x=t_pos[:,0], y=t_pos[:,1], z=t_pos[:,2], mode='lines', line=dict(color='darkred', width=1, dash='dot'), name='Threat Predicted Path'))
    fig.add_trace(go.Scatter3d(x=i_pos[:,0], y=i_pos[:,1], z=i_pos[:,2], mode='lines', line=dict(color='darkblue', width=1, dash='dot'), name='Interceptor Predicted Path'))

    # Animated Markers
    fig.add_trace(go.Scatter3d(x=[t_pos[0,0]], y=[t_pos[0,1]], z=[t_pos[0,2]], mode='markers', marker=dict(color='red', size=6), name='Threat'))
    fig.add_trace(go.Scatter3d(x=[i_pos[0,0]], y=[i_pos[0,1]], z=[i_pos[0,2]], mode='markers', marker=dict(color='cyan', size=6), name='Interceptor'))

    frames = []
    for k in range(len(times)):
        frames.append(go.Frame(data=[
            go.Scatter3d(x=t_pos[:k+1,0], y=t_pos[:k+1,1], z=t_pos[:k+1,2]), # Static path placeholder
            go.Scatter3d(x=i_pos[:k+1,0], y=i_pos[:k+1,1], z=i_pos[:k+1,2]), # Static path placeholder
            go.Scatter3d(x=[t_pos[k,0]], y=[t_pos[k,1]], z=[t_pos[k,2]]),
            go.Scatter3d(x=[i_pos[k,0]], y=[i_pos[k,1]], z=[i_pos[k,2]])
        ], name=str(k)))

    fig.frames = frames

    fig.update_layout(
        title="Air Defense Simulation (Augmented PN)",
        scene=dict(
            xaxis=dict(range=[-10000, 60000], title="Range X"),
            zaxis=dict(range=[0, 40000], title="Altitude"),
            aspectmode='manual', aspectratio=dict(x=1, y=0.5, z=0.5)
        ),
        updatemenus=[{
            "buttons": [{"label": "Play", "method": "animate", "args": [None]}],
            "type": "buttons", "showactive": False
        }],
        template="plotly_dark"
    )

    fig.show()

if __name__ == "__main__":
    visualize_simulation()