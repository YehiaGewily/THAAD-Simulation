import numpy as np

# Try to force a windowed backend for interactive rotation (optional)
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider

from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================================
# 1. MATH & PHYSICS MODULES
# ============================================================================
class KalmanFilter:
    """Standard 6-State Kalman Filter with simple gravity compensation."""
    def __init__(self, dt, pos_std, vel_std):
        self.dt = dt
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt

        self.H = np.eye(6)

        self.Q = np.eye(6) * 0.1
        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = pos_std**2
            self.R[i+3, i+3] = vel_std**2

        self.P = np.eye(6) * 500.0
        self.x = np.zeros(6)
        self.initialized = False

    def update(self, meas_pos, meas_vel):
        z = np.hstack((meas_pos, meas_vel))
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x[:3], self.x[3:]

        # Predict
        self.x = self.F @ self.x
        self.x[5] -= 9.81 * self.dt          # simple gravity model on vz
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3], self.x[3:]

    def predict_impact(self, g):
        """Predict impact point on z=0 from current state."""
        if self.x[5] >= 0:
            return None
        p_z, v_z = self.x[2], self.x[5]
        a = 0.5 * g
        b = -v_z
        c = -p_z
        d = b*b - 4*a*c
        if d < 0:
            return None
        t_sol = (-b + np.sqrt(d)) / (2*a)
        if t_sol <= 0:
            return None
        return np.array([
            self.x[0] + self.x[3]*t_sol,
            self.x[1] + self.x[4]*t_sol,
            0.0
        ])


def get_cpa(pos1, vel1, pos2, vel2, dt):
    """Closest Point of Approach over [0, dt]."""
    dp = pos2 - pos1
    dv = vel2 - vel1
    dv2 = np.dot(dv, dv)
    if dv2 < 1e-6:
        return 0.0, np.linalg.norm(dp)
    t_min = -np.dot(dp, dv) / dv2
    if 0 <= t_min <= dt:
        return t_min, np.linalg.norm(dp + dv * t_min)
    d_start = np.linalg.norm(dp)
    d_end   = np.linalg.norm(dp + dv * dt)
    if d_start < d_end:
        return 0.0, d_start
    else:
        return dt, d_end

# ============================================================================
# 2. CONFIGURATION – tuned for “reasonable” interceptor, not god mode
# ============================================================================
class SimConfig:
    dt = 0.02
    t_max = 150.0
    g = 9.81

    pos_noise = 25.0
    vel_noise = 8.0

    # More moderate interceptor numbers
    boost_thrust = 180000.0   # N
    boost_time   = 6.0        # s
    mass_wet     = 900.0      # kg
    mass_dry     = 450.0      # kg
    kv_mass      = 200.0      # kg
    kv_drag      = 0.00008

    pn_gain      = 4.0
    max_g        = 40.0       # 40 g lateral limit
    kill_radius  = 25.0       # 25 m hit bubble

# ============================================================================
# 3. OBJECTS
# ============================================================================
class BallisticThreat:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.active = True

    def update(self, dt):
        if self.pos[2] <= 0:
            self.active = False
            return
        speed = np.linalg.norm(self.vel)
        drag = -0.00004 * speed * self.vel
        self.vel += (np.array([0, 0, -SimConfig.g]) + drag) * dt
        self.pos += self.vel * dt


class AdvancedInterceptor:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([0., 0., 0.], dtype=float)
        self.acc = np.array([0., 0., 0.])
        self.mass = SimConfig.mass_wet
        self.stage = "IDLE"
        self.launch_time = 0.0
        self.launched = False
        self.exploded = False
        self.g_hist = []

    def launch(self, t, direction):
        self.launched = True
        self.launch_time = t
        self.stage = "BOOST"
        self.vel = direction * 80.0  # tube-exit kick

    def update(self, t, target_p, target_v, dt):
        if not self.launched or self.exploded:
            return

        # Staging
        t_flight = t - self.launch_time
        if t_flight < SimConfig.boost_time:
            self.stage = "BOOST"
            burn = (SimConfig.mass_wet - SimConfig.mass_dry) / SimConfig.boost_time
            self.mass = SimConfig.mass_wet - burn * t_flight
            thrust = SimConfig.boost_thrust
        else:
            self.stage = "KV_COAST"
            self.mass = SimConfig.kv_mass
            thrust = 0.0

        # PN guidance
        r = target_p - self.pos
        v = target_v - self.vel
        r_mag = np.linalg.norm(r)

        if r_mag < 20.0:
            acc_cmd = np.zeros(3)
        else:
            omega = np.cross(r, v) / (r_mag**2 + 1e-6)
            acc_cmd = SimConfig.pn_gain * np.cross(v, omega)
            acc_cmd += np.array([0, 0, SimConfig.g])  # gravity comp

        # G-limit
        acc_mag = np.linalg.norm(acc_cmd)
        lim = SimConfig.max_g * 9.81
        if acc_mag > lim:
            acc_cmd = acc_cmd / acc_mag * lim
            acc_mag = lim
        self.g_hist.append(acc_mag / 9.81)

        # Force direction for thrust
        if np.linalg.norm(acc_cmd) > 1e-3:
            u = acc_cmd / np.linalg.norm(acc_cmd)
        else:
            vmag = np.linalg.norm(self.vel)
            u = self.vel / (vmag + 1e-9)

        f_thrust   = u * thrust
        f_drag     = -(0.0001 if self.stage == "BOOST" else SimConfig.kv_drag) * np.linalg.norm(self.vel) * self.vel
        f_grav     = np.array([0, 0, -SimConfig.g]) * self.mass
        f_maneuver = acc_cmd * self.mass

        if self.stage == "BOOST":
            total_f = f_thrust + f_grav + f_drag + 0.1 * f_maneuver
        else:
            total_f = f_maneuver + f_grav + f_drag

        self.acc = total_f / self.mass
        self.vel += self.acc * dt
        self.pos += self.vel * dt

# ============================================================================
# 4. SCENARIO GENERATION – easier geometry, mild drag compensation
# ============================================================================
def spawn_threat():
    g = SimConfig.g

    # Slightly lower / closer than before so intercept is feasible
    T_imp = np.random.uniform(80.0, 110.0)
    z0    = np.random.uniform(50000.0, 70000.0)   # 50–70 km
    r0    = np.random.uniform(30000.0, 45000.0)   # 30–45 km

    theta0 = np.random.uniform(0, 2*np.pi)
    x0, y0 = r0*np.cos(theta0), r0*np.sin(theta0)

    x_imp, y_imp = 0.0, 0.0

    vz0 = (0.5 * g * T_imp**2 - z0) / T_imp
    vx0 = (x_imp - x0) / T_imp
    vy0 = (y_imp - y0) / T_imp

    # Light drag compensation (3%)
    drag_comp = 1.03
    vel0 = np.array([vx0, vy0, vz0]) * drag_comp

    console.print(
        f"[dim]Scenario: {z0/1000:.1f} km alt, {r0/1000:.1f} km range, T_imp≈{T_imp:.1f} s[/dim]"
    )

    return BallisticThreat([x0, y0, z0], vel0), T_imp

# ============================================================================
# 5. EXECUTION
# ============================================================================
def run():
    threat, nom_T = spawn_threat()
    interceptor = AdvancedInterceptor([0, 0, 0])
    kf = KalmanFilter(SimConfig.dt, SimConfig.pos_noise, SimConfig.vel_noise)

    t_max = max(SimConfig.t_max, nom_T * 2.0 + 30.0)

    t = 0.0
    hist = {'t':[], 't_pos':[], 'i_pos':[], 'k_pos':[], 'status':[], 'stage':[], 'raw':[]}
    outcome = "UNKNOWN"
    kill_info = {}

    console.print("\n" + "="*60, style="bold blue")
    console.print("[bold green]THAAD v5.1 | INTERCEPT SIM (TUNED)[/bold green]")
    console.print(f"[dim]Dynamic simulation window: 0–{t_max:.1f} s[/dim]")
    console.print("="*60, style="bold blue")

    while t < t_max:
        # Sensor model
        mp = threat.pos + np.random.normal(0, SimConfig.pos_noise, 3)
        mv = threat.vel + np.random.normal(0, SimConfig.vel_noise, 3)
        ep, ev = kf.update(mp, mv)

        # Launch when track is stable & descending
        if (t > 3.0) and (not interceptor.launched) and (ev[2] < 0):
            direction = ep - interceptor.pos
            direction = direction / np.linalg.norm(direction)
            interceptor.launch(t, direction)
            console.print(f"[bold yellow][T+{t:05.2f}] LAUNCH:[/bold yellow] Track confirmed.")

        # Interceptor physics + kill check
        if interceptor.launched and not interceptor.exploded:
            interceptor.update(t, ep, ev, SimConfig.dt)
            t_off, d_min = get_cpa(interceptor.pos, interceptor.vel,
                                   threat.pos,      threat.vel,
                                   SimConfig.dt)
            if d_min < SimConfig.kill_radius:
                interceptor.exploded = True
                k_time  = t + t_off
                k_pos_t = threat.pos + threat.vel * t_off
                k_pos_i = interceptor.pos + interceptor.vel * t_off
                closing = np.linalg.norm(interceptor.vel - threat.vel)

                hist['t'].append(k_time)
                hist['t_pos'].append(k_pos_t)
                hist['i_pos'].append(k_pos_i)
                hist['k_pos'].append(ep)
                hist['raw'].append(mp)
                hist['status'].append("HIT")
                hist['stage'].append(interceptor.stage)

                outcome = "KILL"
                kill_info = {
                    'time': k_time,
                    'alt':  k_pos_t[2],
                    'dist': d_min,
                    'closing': closing
                }
                console.print(
                    f"[bold green][T+{k_time:05.2f}] *** TARGET DESTROYED ***[/bold green] "
                    f"Miss {d_min:.2f} m, Closing {closing:.1f} m/s"
                )
                break

        # Threat physics
        threat.update(SimConfig.dt)
        if not threat.active:
            outcome = "GROUND"
            console.print(
                f"[bold red][T+{t:05.2f}] THREAT HIT GROUND BEFORE INTERCEPT.[/bold red]"
            )
            break

        # Log
        hist['t'].append(t)
        hist['t_pos'].append(threat.pos.copy())
        hist['i_pos'].append(interceptor.pos.copy())
        hist['k_pos'].append(ep.copy())
        hist['raw'].append(mp.copy())
        hist['status'].append("FLYING" if interceptor.launched else "SEARCH")
        hist['stage'].append(interceptor.stage)

        t += SimConfig.dt

    if outcome == "UNKNOWN":
        outcome = "TIMEOUT"
        console.print(
            f"[bold red][T+{t:05.2f}] SIM TIMEOUT – target still airborne.[/bold red]"
        )

    # BDA
    table = Table(title="BATTLE DAMAGE ASSESSMENT")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Outcome", outcome)
    if outcome == "KILL":
        table.add_row("Kill Time", f"{kill_info['time']:.2f} s")
        table.add_row("Kill Altitude", f"{kill_info['alt']/1000:.1f} km")
        table.add_row("Miss Distance", f"{kill_info['dist']:.2f} m")
        table.add_row("Closing Velocity", f"{kill_info['closing']:.1f} m/s")
    if interceptor.g_hist:
        table.add_row("Max G-Load", f"{max(interceptor.g_hist):.1f} G")
    console.print(table)

    return hist

# ============================================================================
# 6. VISUALIZATION
# ============================================================================
if __name__ == "__main__":
    h = run()

    t_pos = np.array(h['t_pos'])
    i_pos = np.array(h['i_pos'])
    k_pos = np.array(h['k_pos'])

    PLAYBACK_SPEED = 5
    frame_indices = np.arange(0, len(h['t']), PLAYBACK_SPEED, dtype=int)
    if frame_indices[-1] != len(h['t'])-1:
        frame_indices = np.append(frame_indices, len(h['t'])-1)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.mouse_init()

    # Defended zone (visual)
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(30000*np.cos(theta), 30000*np.sin(theta), 0, 'lime', label='Defended Area')

    lt, = ax.plot([], [], [], 'r--', label='Threat')
    li, = ax.plot([], [], [], 'cyan', linewidth=2, label='Interceptor')
    lk, = ax.plot([], [], [], 'g-', alpha=0.5, label='Kalman Track')

    pt, = ax.plot([], [], [], 'ro')
    pi, = ax.plot([], [], [], 'b^')
    expl, = ax.plot([], [], [], 'y*', markersize=50, visible=False)

    txt_stat = ax.text2D(0.05, 0.95, "READY", transform=ax.transAxes, fontweight='bold')

    ax.set_xlim(-60000, 60000)
    ax.set_ylim(-60000, 60000)
    ax.set_zlim(0, 80000)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.legend()

    def draw_frame(idx):
        if idx < 0 or idx >= len(h['t']):
            return

        lt.set_data(t_pos[:idx,0], t_pos[:idx,1])
        lt.set_3d_properties(t_pos[:idx,2])

        li.set_data(i_pos[:idx,0], i_pos[:idx,1])
        li.set_3d_properties(i_pos[:idx,2])

        lk.set_data(k_pos[:idx,0], k_pos[:idx,1])
        lk.set_3d_properties(k_pos[:idx,2])

        pt.set_data([t_pos[idx,0]],[t_pos[idx,1]])
        pt.set_3d_properties([t_pos[idx,2]])

        pi.set_data([i_pos[idx,0]],[i_pos[idx,1]])
        pi.set_3d_properties([i_pos[idx,2]])

        stat = h['status'][idx]
        mode = h['stage'][idx]
        txt_stat.set_text(f"STATUS: {stat} | MODE: {mode}")

        if stat == "HIT":
            txt_stat.set_color("green")
            expl.set_data([t_pos[idx,0]],[t_pos[idx,1]])
            expl.set_3d_properties([t_pos[idx,2]])
            expl.set_visible(True)
            pt.set_visible(False)
            pi.set_visible(False)
        else:
            txt_stat.set_color("black")

        return lt, li, lk, pt, pi, expl, txt_stat

    def update(f_idx):
        if f_idx >= len(frame_indices):
            return
        idx = frame_indices[f_idx]
        return draw_frame(idx)

    anim = FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=20, blit=False, repeat=False
    )


    # --- OPTIONAL: save to MP4 (needs ffmpeg installed) ---
    from matplotlib.animation import FFMpegWriter

    writer = FFMpegWriter(
        fps=30,                          # video framerate
        metadata={'title': 'THAAD Sim'}, # optional
        bitrate=2400
    )
        # Time Slider
    ax_sl = plt.axes([0.2, 0.02, 0.6, 0.03])
    sl = Slider(ax_sl, "Time Frame", 0, len(frame_indices) - 1,
                valinit=0, valfmt="%0.0f")

    def on_change(val):
        f = int(sl.val)
        f = max(0, min(f, len(frame_indices) - 1))
        idx = frame_indices[f]
        draw_frame(idx)
        fig.canvas.draw_idle()

    sl.on_changed(on_change)
    plt.show()


    
