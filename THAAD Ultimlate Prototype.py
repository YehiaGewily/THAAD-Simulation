import numpy as np
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
    """Standard 6-State Kalman Filter for 3D Tracking"""
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
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3], self.x[3:]

    def predict_impact(self, g):
        """Predict ground impact point assuming ballistic motion (z → 0)."""
        # x = [px, py, pz, vx, vy, vz]
        if self.x[5] >= 0:  # vz >= 0 → not descending yet
            return None

        p_z, v_z = self.x[2], self.x[5]
        # z(t) = p_z + v_z t - 0.5 g t^2 = 0
        a = 0.5 * g
        b = -v_z
        c = -p_z
        disc = b*b - 4*a*c
        if disc < 0:
            return None

        t_imp = (-b - np.sqrt(disc)) / (2 * a)
        if t_imp < 0:
            return None

        x_imp = self.x[0] + self.x[3] * t_imp
        y_imp = self.x[1] + self.x[4] * t_imp
        return np.array([x_imp, y_imp, 0.0])


def get_cpa(pos1, vel1, pos2, vel2, dt):
    """Continuous Collision Detection using Closest Point of Approach."""
    dp = pos2 - pos1
    dv = vel2 - vel1
    dv2 = np.dot(dv, dv)
    if dv2 < 1e-6:
        return 0.0, np.linalg.norm(dp)

    t_min = -np.dot(dp, dv) / dv2
    if 0 <= t_min <= dt:
        return t_min, np.linalg.norm(dp + dv * t_min)

    d_end = np.linalg.norm(dp + dv * dt)
    d_start = np.linalg.norm(dp)
    if d_start < d_end:
        return 0.0, d_start
    else:
        return dt, d_end

# ============================================================================
# 2. CONFIGURATION
# ============================================================================
class SimConfig:
    dt = 0.02
    t_max = 120.0      # fallback; real t_max is dynamic per scenario
    g = 9.81

    # used only for drawing the circle now (no gating)
    defended_radius = 20000.0  # meters

    pos_noise = 30.0
    vel_noise = 10.0

    # Interceptor
    boost_thrust = 40000.0
    boost_time = 6.0
    mass_wet = 800.0
    mass_dry = 400.0
    kv_mass = 60.0
    kv_drag = 0.0001

    pn_gain = 4.0
    max_g = 40.0
    kill_radius = 15.0   # a bit larger to make hits more certain

# ============================================================================
# 3. OBJECTS
# ============================================================================
class BallisticThreat:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.active = True

    def update(self, dt):
        # Ground is at z = 0
        if self.pos[2] <= 0:
            self.active = False
            return

        drag = -0.00004 * np.linalg.norm(self.vel) * self.vel
        self.vel += (np.array([0, 0, -SimConfig.g]) + drag) * dt
        self.pos += self.vel * dt


class AdvancedInterceptor:
    def __init__(self, pos):
        # Defense system is on the ground at z = 0
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
        self.vel = direction * 50.0  # tube exit velocity

    def update(self, t, target_p, target_v, dt):
        if not self.launched or self.exploded:
            return

        # Stage logic
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
        r_norm2 = np.linalg.norm(r)**2 + 1e-6
        omega = np.cross(r, v) / r_norm2
        acc_cmd = SimConfig.pn_gain * np.cross(v, omega)
        acc_cmd += np.array([0, 0, SimConfig.g])  # gravity bias

        # G-limit
        acc_mag = np.linalg.norm(acc_cmd)
        lim = SimConfig.max_g * 9.81
        if acc_mag > lim:
            acc_cmd = acc_cmd / acc_mag * lim
            acc_mag = lim

        self.g_hist.append(acc_mag / 9.81)

        # Orientation for thrust
        if acc_mag > 1e-3:
            u = acc_cmd / acc_mag
        else:
            v_norm = np.linalg.norm(self.vel)
            u = np.array([1., 0., 0.]) if v_norm < 1e-3 else self.vel / v_norm

        f_thrust = u * thrust
        f_drag = -(0.0001 if self.stage == "BOOST" else SimConfig.kv_drag) * np.linalg.norm(self.vel) * self.vel
        f_grav = np.array([0, 0, -SimConfig.g]) * self.mass
        f_maneuver = acc_cmd * self.mass

        if self.stage == "BOOST":
            total_f = f_thrust + f_grav + f_drag + (f_maneuver * 0.1)
        else:
            total_f = f_maneuver + f_grav + f_drag

        self.acc = total_f / self.mass
        self.vel += self.acc * dt
        self.pos += self.vel * dt

# ============================================================================
# 4. RANDOM THREAT SPAWNER (TUNED FOR EASIER INTERCEPT)
# ============================================================================
def spawn_random_threat():
    """
    Random threat whose no-drag nominal impact is near the origin.
    Geometry is tuned so the interceptor can realistically catch it.
    Returns (threat_object, nominal_impact_time_no_drag).
    """
    g = SimConfig.g

    # Slightly narrower ranges to keep geometry "reasonable"
    T_imp = np.random.uniform(60.0, 90.0)         # seconds
    z0 = np.random.uniform(40000.0, 60000.0)      # 40–60 km

    vz0 = (0.5 * g * T_imp**2 - z0) / T_imp       # solves z(T_imp) ≈ 0

    r0 = np.random.uniform(30000.0, 60000.0)      # 30–60 km from origin
    theta0 = np.random.uniform(0, 2 * np.pi)
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)

    # Aim impact within ~15 km of origin
    offset_r = np.random.uniform(0.0, 15000.0)
    offset_theta = np.random.uniform(0, 2 * np.pi)
    x_imp = offset_r * np.cos(offset_theta)
    y_imp = offset_r * np.sin(offset_theta)

    vx0 = (x_imp - x0) / T_imp
    vy0 = (y_imp - y0) / T_imp

    console.print(
        f"[dim]Scenario: Threat from (~{x0/1000:.1f} km, ~{y0/1000:.1f} km, {z0/1000:.1f} km)"
        f" → nominal impact near ({x_imp/1000:.1f} km, {y_imp/1000:.1f} km) in {T_imp:.1f} s[/dim]"
    )

    return BallisticThreat([x0, y0, z0], [vx0, vy0, vz0]), T_imp

# ============================================================================
# 5. EXECUTION
# ============================================================================
def run():
    threat, nominal_T_imp = spawn_random_threat()
    interceptor = AdvancedInterceptor([0, 0, 0])  # launcher at ground
    kf = KalmanFilter(SimConfig.dt, SimConfig.pos_noise, SimConfig.vel_noise)

    # Dynamic sim time; drag slows descent, so give plenty of time
    t_max = max(SimConfig.t_max, nominal_T_imp * 2.0 + 20.0)

    t = 0.0
    hist = {
        't': [], 't_pos': [], 'i_pos': [],
        'k_pos': [], 'status': [], 'stage': [], 'raw': []
    }
    pip = None
    authorized = False

    outcome = "UNKNOWN"
    kill_time = None
    kill_dist = None
    closing_vel = None
    kill_pos_threat = None
    kill_pos_int = None

    console.print("\n" + "="*60, style="bold blue")
    console.print("[bold cyan]THAAD v4.3 | ALWAYS-LAUNCH DEMO[/bold cyan]")
    console.print(f"[dim]Dynamic simulation window: 0–{t_max:.1f} s[/dim]")
    console.print("="*60, style="bold blue")

    while t < t_max:
        # Sensor
        meas_p = threat.pos + np.random.normal(0, SimConfig.pos_noise, 3)
        meas_v = threat.vel + np.random.normal(0, SimConfig.vel_noise, 3)
        est_p, est_v = kf.update(meas_p, meas_v)

        # Predict impact point (for info only now)
        pip = kf.predict_impact(SimConfig.g)

        # ALWAYS-LAUNCH LOGIC:
        # As soon as PIP is meaningful and t > 4 s, we authorize and fire.
        if pip is not None and not authorized and t > 4.0:
            authorized = True
            l_dir = est_p - interceptor.pos
            l_dir = l_dir / np.linalg.norm(l_dir)
            interceptor.launch(t, l_dir)
            range_pip = np.linalg.norm(pip[:2])
            console.print(
                f"[bold yellow][T+{t:05.2f}] AUTHORIZED:[/bold yellow] "
                f"PIP range ≈ {range_pip:.0f} m → [bold red]LAUNCHING INTERCEPTOR[/bold red]"
            )

        # Optional monitoring print (for flavor)
        if pip is not None and int(t) % 10 == 0 and abs(t - int(t)) < 0.02:
            range_pip = np.linalg.norm(pip[:2])
            console.print(
                f"[dim][T+{t:05.2f}] Tracking: PIP range ≈ {range_pip:.0f} m[/dim]"
            )

        # Interceptor physics
        if interceptor.launched and not interceptor.exploded:
            interceptor.update(t, est_p, est_v, SimConfig.dt)

        # Collision check
        if interceptor.launched and not interceptor.exploded:
            t_off, d_min = get_cpa(
                interceptor.pos, interceptor.vel,
                threat.pos, threat.vel,
                SimConfig.dt
            )
            if d_min < SimConfig.kill_radius:
                interceptor.exploded = True

                kill_time = t + t_off
                kill_dist = d_min
                kill_pos_threat = threat.pos + threat.vel * t_off
                kill_pos_int = interceptor.pos + interceptor.vel * t_off
                closing_vel = np.linalg.norm(interceptor.vel - threat.vel)

                hist['t'].append(kill_time)
                hist['t_pos'].append(kill_pos_threat)
                hist['i_pos'].append(kill_pos_int)
                hist['k_pos'].append(est_p.copy())
                hist['raw'].append(meas_p.copy())
                hist['status'].append("HIT")
                hist['stage'].append(interceptor.stage)

                console.print(
                    f"[bold green][T+{kill_time:05.2f}] *** KILL CONFIRMED ***[/bold green]  "
                    f"Miss distance = {d_min:.2f} m, Closing ≈ {closing_vel:.1f} m/s"
                )
                console.print(
                    f"[bold cyan]Impact point (Threat):[/bold cyan] "
                    f"x={kill_pos_threat[0]:.1f} m, "
                    f"y={kill_pos_threat[1]:.1f} m, "
                    f"z={kill_pos_threat[2]:.1f} m"
                )
                console.print(
                    f"[bold cyan]Impact point (Interceptor):[/bold cyan] "
                    f"x={kill_pos_int[0]:.1f} m, "
                    f"y={kill_pos_int[1]:.1f} m, "
                    f"z={kill_pos_int[2]:.1f} m"
                )

                outcome = "KILL"
                break

        # Threat physics
        threat.update(SimConfig.dt)
        if not threat.active:
            # Only happens if interceptor somehow never kills it
            console.print(
                f"[bold red][T+{t:05.2f}] FAILED INTERCEPT.[/bold red] "
                f"Threat impacted the ground (z=0)."
            )
            outcome = "GROUND_IMPACT"
            break

        # Log state
        hist['t'].append(t)
        hist['t_pos'].append(threat.pos.copy())
        hist['i_pos'].append(interceptor.pos.copy())
        hist['k_pos'].append(est_p.copy())
        hist['raw'].append(meas_p.copy())
        hist['status'].append("FLYING" if interceptor.launched else "SEARCH")
        hist['stage'].append(interceptor.stage)

        t += SimConfig.dt

    if outcome == "UNKNOWN":
        outcome = "TIMEOUT"
        console.print(
            f"[bold red][T+{t:05.2f}] SIM TIMEOUT – No intercept and target still airborne.[/bold red]"
        )
        console.print(
            f"[dim]Final threat altitude ≈ {threat.pos[2]:.1f} m, "
            f"range ≈ {np.linalg.norm(threat.pos[:2]):.1f} m.[/dim]"
        )

    # BDA table
    table = Table(title="BATTLE DAMAGE ASSESSMENT", show_header=True, header_style="bold magenta")
    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="left")

    table.add_row("Outcome", outcome)
    if kill_time is not None:
        table.add_row("Intercept Time", f"{kill_time:.2f} s")
    if kill_dist is not None:
        table.add_row("Miss Distance", f"{kill_dist:.2f} m")
    if closing_vel is not None:
        table.add_row("Closing Velocity", f"{closing_vel:.1f} m/s")
    if interceptor.g_hist:
        table.add_row("Max G-Load", f"{max(interceptor.g_hist):.2f} G")

    console.print()
    console.print(table)
    console.print()

    return hist, pip

# ============================================================================
# 6. VISUALIZATION
# ============================================================================
if __name__ == "__main__":
    h, pip = run()

    t_pos = np.array(h['t_pos'])
    i_pos = np.array(h['i_pos'])
    k_pos = np.array(h['k_pos'])
    r_pos = np.array(h['raw'])

    PLAYBACK_SPEED = 5  # 1 = “real-time”, >1 = faster replay

    frame_indices = np.arange(0, len(h['t']), PLAYBACK_SPEED, dtype=int)
    if frame_indices[-1] != len(h['t']) - 1:
        frame_indices = np.append(frame_indices, len(h['t']) - 1)

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.mouse_init()

    # Ground defended-circle (now just a visual reference)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        SimConfig.defended_radius * np.cos(theta),
        SimConfig.defended_radius * np.sin(theta),
        0,
        'g-',
        label='Defended Zone'
    )
    if pip is not None:
        ax.scatter(pip[0], pip[1], 0, c='r', marker='x', s=100, label='Predicted Impact')

    # Paths & markers
    l_t, = ax.plot([], [], [], 'r--', label='Threat')
    l_i, = ax.plot([], [], [], 'b-', linewidth=2, label='Interceptor')
    l_k, = ax.plot([], [], [], 'g-', linewidth=1, alpha=0.7, label='Kalman Soln')

    p_t, = ax.plot([], [], [], 'ro')
    p_i, = ax.plot([], [], [], 'b^')
    sc_r = ax.scatter([], [], [], c='gray', alpha=0.1, s=10)
    expl, = ax.plot([], [], [], 'y*', markersize=60, visible=False, label='Collision Point')

    txt_stat = ax.text2D(0.05, 0.95, "SYSTEM READY", transform=ax.transAxes, fontweight='bold')
    txt_alt = ax.text2D(0.05, 0.90, "", transform=ax.transAxes)

    ax.set_xlim(-80000, 80000)
    ax.set_ylim(-80000, 80000)
    ax.set_zlim(0, 80000)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("Altitude (m)")
    ax.legend(loc='upper right')

    def draw_frame(idx):
        if idx <= 0 or idx >= len(h['t']):
            return

        l_t.set_data(t_pos[:idx, 0], t_pos[:idx, 1])
        l_t.set_3d_properties(t_pos[:idx, 2])

        l_i.set_data(i_pos[:idx, 0], i_pos[:idx, 1])
        l_i.set_3d_properties(i_pos[:idx, 2])

        l_k.set_data(k_pos[:idx, 0], k_pos[:idx, 1])
        l_k.set_3d_properties(k_pos[:idx, 2])

        p_t.set_data([t_pos[idx, 0]], [t_pos[idx, 1]])
        p_t.set_3d_properties([t_pos[idx, 2]])

        p_i.set_data([i_pos[idx, 0]], [i_pos[idx, 1]])
        p_i.set_3d_properties([i_pos[idx, 2]])

        s = max(0, idx - 20)
        sc_r._offsets3d = (r_pos[s:idx, 0], r_pos[s:idx, 1], r_pos[s:idx, 2])

        stat = h['status'][idx]
        mode = h['stage'][idx]
        sim_t = h['t'][idx]

        txt_stat.set_text(f"STATUS: {stat} | MODE: {mode} | Playback x{PLAYBACK_SPEED}")
        if stat == "HIT":
            txt_stat.set_color("red")
            expl.set_data([t_pos[idx, 0]], [t_pos[idx, 1]])
            expl.set_3d_properties([t_pos[idx, 2]])
            expl.set_visible(True)
            p_t.set_visible(False)
            p_i.set_visible(False)
        else:
            txt_stat.set_color("black")

        txt_alt.set_text(f"Sim t = {sim_t:.2f} s | Threat Alt: {t_pos[idx, 2]:.0f} m")

        return l_t, l_i, l_k, p_t, p_i, expl, txt_stat, txt_alt

    def update(frame_number):
        if frame_number >= len(frame_indices):
            return
        idx = frame_indices[frame_number]
        return draw_frame(idx)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=20,
        blit=False,
        repeat=False
    )

    ax_slider = plt.axes([0.25, 0.02, 0.50, 0.02])
    frame_slider = Slider(
        ax_slider,
        "Frame",
        0,
        len(frame_indices) - 1,
        valinit=0,
        valfmt='%0.0f'
    )

    def on_slider(val):
        f = int(frame_slider.val)
        f = max(0, min(f, len(frame_indices) - 1))
        idx = frame_indices[f]
        draw_frame(idx)
        fig.canvas.draw_idle()

    frame_slider.on_changed(on_slider)

    plt.show()
