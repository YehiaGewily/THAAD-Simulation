import numpy as np

from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================================
# 1. MATH & PHYSICS MODULES
# ============================================================================
class KalmanFilter:
    """Standard 6-State Kalman Filter with Gravity Compensation"""
    def __init__(self, dt, pos_std, vel_std):
        self.dt = dt
        # State Transition Matrix (F)
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i+3] = dt

        # Measurement Matrix (H)
        self.H = np.eye(6)

        # Process Noise (Q) and Measurement Noise (R)
        self.Q = np.eye(6) * SimConfig.kf_q
        self.R = np.eye(6)
        for i in range(3):
            self.R[i, i] = pos_std**2
            self.R[i+3, i+3] = vel_std**2

        self.P = np.eye(6) * SimConfig.kf_p0
        self.x = np.zeros(6)
        self.initialized = False

    def update(self, meas_pos, meas_vel):
        z = np.hstack((meas_pos, meas_vel))
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x[:3], self.x[3:]

        # Predict Step
        self.x = self.F @ self.x
        # Simple gravity model: pull vz down each step
        self.x[5] -= 9.81 * self.dt
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update Step
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        return self.x[:3], self.x[3:]

    def predict_impact(self, g):
        """Predicts ground impact (z=0) based on current state."""
        if self.x[5] >= 0:
            return None  # moving up or level

        p_z, v_z = self.x[2], self.x[5]

        # Solve 0.5*g*t^2 - v0*t - z0 = 0  (z(t)=0)
        a = 0.5 * g
        b = -v_z
        c = -p_z
        d = b * b - 4 * a * c

        if d < 0:
            return None

        # Use positive root
        t_sol = (-b + np.sqrt(d)) / (2 * a)
        if t_sol <= 0:
            return None

        return np.array([
            self.x[0] + self.x[3] * t_sol,
            self.x[1] + self.x[4] * t_sol,
            0.0
        ])


def get_cpa(pos1, vel1, pos2, vel2, dt):
    """Continuous Collision Detection (solves tunneling between frames)."""
    dp = pos2 - pos1
    dv = vel2 - vel1
    dv2 = np.dot(dv, dv)

    if dv2 < 1e-6:
        return 0.0, np.linalg.norm(dp)

    t_min = -np.dot(dp, dv) / dv2

    if 0 <= t_min <= dt:
        return t_min, np.linalg.norm(dp + dv * t_min)

    # Otherwise compare start/end
    d_start = np.linalg.norm(dp)
    d_end = np.linalg.norm(dp + dv * dt)
    if d_start < d_end:
        return 0.0, d_start
    else:
        return dt, d_end

# ============================================================================
# 2. CONFIGURATION (MAX-PERFORMANCE INTERCEPTOR, NOT CRAZY)
# ============================================================================
class SimConfig:
    dt = 0.02
    t_max = 150.0  # base; we’ll also expand dynamically
    g = 9.81

    pos_noise = 25.0
    vel_noise = 8.0

    # MAX-PERFORMANCE INTERCEPTOR SPECS (strong but reasonable)
    boost_thrust = 220000.0  # N
    boost_time   = 8.0       # s
    mass_wet     = 900.0
    mass_dry     = 450.0
    kv_mass      = 200.0
    kv_drag      = 0.00008

    pn_gain      = 4.0       # PN gain
    max_g        = 50.0      # 50G turn capability
    kill_radius  = 30.0      # 30 m kill volume
    booster_drag = 0.0001    # booster quadratic drag coefficient
    threat_drag  = 0.00004   # 1/m, threat quadratic drag (acceleration per speed^2)
    kf_q         = 0.1       # Kalman process-noise variance (diagonal)
    kf_p0        = 500.0     # Kalman initial covariance (diagonal)

    defended_radius = 30000.0  # m – visual only; this mode fires at every threat

# ============================================================================
# 3. OBJECTS
# ============================================================================
class BallisticThreat:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.active = True

    def update(self, dt):
        if not self.active:
            return

        # Physics with drag
        speed = np.linalg.norm(self.vel)
        drag = -SimConfig.threat_drag * speed * self.vel
        self.vel += (np.array([0, 0, -SimConfig.g]) + drag) * dt
        self.pos += self.vel * dt
        if self.pos[2] <= 0:
            self.active = False


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
        self.vel = direction * 80.0  # Initial kick (slightly lower)

    def update(self, t, target_p, target_v, dt):
        if not self.launched or self.exploded:
            return

        # Stage management
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

        # Guidance (Augmented PN)
        r = target_p - self.pos
        v = target_v - self.vel
        r_mag = np.linalg.norm(r)

        if r_mag < 20.0:
            # Avoid singularity at very close range
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

        # Force application
        if np.linalg.norm(acc_cmd) > 1e-3:
            u = acc_cmd / np.linalg.norm(acc_cmd)
        else:
            vmag = np.linalg.norm(self.vel)
            u = self.vel / (vmag + 1e-9)

        f_thrust   = u * thrust
        f_drag     = -(SimConfig.booster_drag if self.stage == "BOOST" else SimConfig.kv_drag) * np.linalg.norm(self.vel) * self.vel
        f_grav     = np.array([0, 0, -SimConfig.g]) * self.mass
        f_maneuver = acc_cmd * self.mass

        if self.stage == "BOOST":
            total_f = f_thrust + f_grav + f_drag + (f_maneuver * 0.1)
        else:
            total_f = f_maneuver + f_grav + f_drag

        self.acc = total_f / self.mass
        self.vel += self.acc * dt
        self.pos += self.vel * dt

# ============================================================================
# 4. SCENARIO GENERATION (slightly easier geometry, mild drag comp)
# ============================================================================
def spawn_threat():
    """Spawns a threat with velocity boosted slightly to compensate for drag."""
    g = SimConfig.g

    # Slightly lower / closer so intercept is feasible without insane performance
    T_imp = np.random.uniform(80.0, 110.0)        # nominal time to impact (no-drag)
    z0    = np.random.uniform(40000.0, 60000.0)   # 40–60 km

    r0     = np.random.uniform(25000.0, 40000.0)  # 25–40 km ground range
    theta0 = np.random.uniform(0, 2 * np.pi)
    x0, y0 = r0 * np.cos(theta0), r0 * np.sin(theta0)

    # Aim at origin (0,0)
    x_imp, y_imp = 0.0, 0.0

    # Ideal vacuum velocity
    vz0 = (0.5 * g * T_imp**2 - z0) / T_imp
    vx0 = (x_imp - x0) / T_imp
    vy0 = (y_imp - y0) / T_imp

    # Light COMPENSATION FOR DRAG: boost the full vector by ~3%
    drag_comp = 1.03
    vel0 = np.array([vx0, vy0, vz0]) * drag_comp

    console.print(
        f"[dim]Scenario: {z0/1000:.1f} km Alt, {r0/1000:.1f} km Range, "
        f"T_imp≈{T_imp:.1f} s, Drag Comp: {drag_comp:.2f}x[/dim]"
    )

    return BallisticThreat([x0, y0, z0], vel0), T_imp

# ============================================================================
# 5. EXECUTION
# ============================================================================
def run():
    threat, nom_T = spawn_threat()
    interceptor = AdvancedInterceptor([0, 0, 0])
    kf = KalmanFilter(SimConfig.dt, SimConfig.pos_noise, SimConfig.vel_noise)

    # Dynamic sim window: enough time for drag and engagement
    t_max = max(SimConfig.t_max, nom_T * 2.0 + 30.0)

    t = 0.0
    hist = {
        't': [], 't_pos': [], 'i_pos': [],
        'k_pos': [], 'status': [], 'stage': [], 'raw': [],
        't_vel': [], 'i_vel': [], 'kf_err': [], 'kf_sig': []
    }

    outcome    = "UNKNOWN"
    kill_info  = {}
    launch_pip = None  # for display only; this mode does not gate on it

    console.print("\n" + "=" * 60, style="bold blue")
    console.print("[bold green]THAAD v5.0 | MAX-PERFORMANCE INTERCEPT DEMO[/bold green]")
    console.print(f"[dim]Dynamic simulation window: 0–{t_max:.1f} s[/dim]")
    console.print("=" * 60, style="bold blue")

    while t < t_max:
        # 1. Sense & Track
        mp = threat.pos + np.random.normal(0, SimConfig.pos_noise, 3)
        mv = threat.vel + np.random.normal(0, SimConfig.vel_noise, 3)
        ep, ev = kf.update(mp, mv)
        # Filter error vs truth and its 1-sigma bounds, for the analysis view
        kf_err = np.hstack((ep - threat.pos, ev - threat.vel))
        kf_sig = np.sqrt(np.diag(kf.P))

        # 2. Launch Logic:
        #    If target is descending (ev[2]<0), and we have some tracking, then fire.
        if (t > 3.0) and (not interceptor.launched) and (ev[2] < 0):
            direction = ep - interceptor.pos
            direction = direction / np.linalg.norm(direction)
            interceptor.launch(t, direction)
            launch_pip = kf.predict_impact(SimConfig.g)
            console.print(
                f"[bold yellow][T+{t:05.2f}] LAUNCH:[/bold yellow] Track confirmed. WEAPONS FREE."
            )

        # 3. Physics: propagate both bodies from t to t+dt, keeping start states
        t_next = t + SimConfig.dt
        i_p0 = interceptor.pos.copy()
        t_p0 = threat.pos.copy()
        interceptor.update(t, ep, ev, SimConfig.dt)
        threat.update(SimConfig.dt)

        # 4. Kill Check via CPA between both segments over the same step
        if interceptor.launched and not interceptor.exploded:
            i_v = (interceptor.pos - i_p0) / SimConfig.dt
            t_v = (threat.pos - t_p0) / SimConfig.dt
            t_off, d_min = get_cpa(i_p0, i_v, t_p0, t_v, SimConfig.dt)
            if d_min < SimConfig.kill_radius:
                interceptor.exploded = True

                k_time  = t + t_off
                k_pos_t = t_p0 + t_v * t_off
                k_pos_i = i_p0 + i_v * t_off
                closing = np.linalg.norm(i_v - t_v)

                hist['t'].append(k_time)
                hist['t_pos'].append(k_pos_t)
                hist['i_pos'].append(k_pos_i)
                hist['k_pos'].append(ep)
                hist['raw'].append(mp)
                hist['status'].append("HIT")
                hist['stage'].append(interceptor.stage)
                hist['t_vel'].append(t_v)
                hist['i_vel'].append(i_v)
                hist['kf_err'].append(kf_err)
                hist['kf_sig'].append(kf_sig)

                outcome = "KILL"
                kill_info = {
                    'dist': d_min,
                    'alt': k_pos_t[2],
                    'time': k_time,
                    'closing': closing
                }

                console.print(
                    f"[bold green][T+{k_time:05.2f}] *** TARGET DESTROYED ***[/bold green] "
                    f"Miss distance: {d_min:.2f} m, Closing ≈ {closing:.1f} m/s"
                )
                break

        # 5. Ground Impact
        if not threat.active:
            outcome = "GROUND"
            console.print(
                f"[bold red][T+{t_next:05.2f}] FAILED.[/bold red] Target hit ground (z=0) before intercept."
            )
            break

        # Log state
        hist['t'].append(t_next)
        hist['t_pos'].append(threat.pos.copy())
        hist['i_pos'].append(interceptor.pos.copy())
        hist['k_pos'].append(ep.copy())
        hist['raw'].append(mp.copy())
        hist['status'].append("FLYING" if interceptor.launched else "SEARCH")
        hist['stage'].append(interceptor.stage)
        hist['t_vel'].append(threat.vel.copy())
        hist['i_vel'].append(interceptor.vel.copy())
        hist['kf_err'].append(kf_err)
        hist['kf_sig'].append(kf_sig)

        t = t_next

    # If neither kill nor ground, classify as TIMEOUT
    if outcome == "UNKNOWN":
        outcome = "TIMEOUT"
        console.print(
            f"[bold red][T+{t:05.2f}] SIM TIMEOUT – No intercept, target still airborne.[/bold red]"
        )
        console.print(
            f"[dim]Final threat altitude ≈ {threat.pos[2]:.1f} m, "
            f"range ≈ {np.linalg.norm(threat.pos[:2]):.1f} m.[/dim]"
        )

    # Report
    table = Table(title="BATTLE DAMAGE ASSESSMENT")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Outcome", outcome)

    if outcome == "KILL":
        table.add_row("Kill Time", f"{kill_info['time']:.2f} s")
        table.add_row("Kill Altitude", f"{kill_info['alt'] / 1000:.1f} km")
        table.add_row("Miss Distance", f"{kill_info['dist']:.2f} m")
        table.add_row("Closing Velocity", f"{kill_info['closing']:.1f} m/s")

    if interceptor.g_hist:
        table.add_row("Max G-Load", f"{max(interceptor.g_hist):.1f} G")

    console.print(table)

    hist['meta'] = {
        'title':           "THAAD v5.0 | MAX-PERFORMANCE INTERCEPT DEMO",
        'mode':            "MAX-PERFORMANCE MODE",
        'engage_rule':     "FIRE ON TRACK",
        'outcome':         outcome,
        'kill':            kill_info or None,
        'launch_pip':      launch_pip,
        'impact':          threat.pos.copy() if outcome in ("GROUND", "NO ENGAGE") else None,
        'g_hist':          interceptor.g_hist,
        'defended_radius': SimConfig.defended_radius,
        'kill_radius':     SimConfig.kill_radius,
        'max_g':           SimConfig.max_g,
        'params':          {k: v for k, v in vars(SimConfig).items() if not k.startswith('_')},
        'min_track_time':  None,
    }

    return hist

# ============================================================================
# 6. VISUALIZATION – 3D browser replay (see replay_viewer.py)
# ============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from replay_viewer import show_replay

    # A fixed seed reproduces a scenario exactly: python "<this file>" <seed>
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else int(np.random.SeedSequence().entropy % 2**31)
    np.random.seed(seed)
    console.print(f"[dim]Seed {seed} – rerun this scenario with: python \"{Path(__file__).name}\" {seed}[/dim]")

    hist = run()
    hist['meta']['seed'] = seed
    path = show_replay(hist)
    console.print(f"[dim]Replay opened in your browser: {path}[/dim]")
