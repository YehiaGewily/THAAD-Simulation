"""The engagement loop: sense, track, launch, guide, check for a kill, assess."""
import numpy as np
from rich.console import Console
from rich.table import Table

from .dynamics import Interceptor
from .estimation import KalmanFilter
from .scenario import spawn_threat

HIST_KEYS = ('t', 't_pos', 'i_pos', 'k_pos', 'status', 'stage', 'raw',
             't_vel', 'i_vel', 'kf_err', 'kf_sig')


def make_rng(seed, legacy=False):
    """Random source for a run.

    ``legacy=True`` returns a ``RandomState``, which draws the same stream as
    the old ``np.random.seed(seed)`` scripts, so their printed seeds replay.
    """
    return np.random.RandomState(seed) if legacy else np.random.default_rng(seed)


def get_cpa(pos1, vel1, pos2, vel2, dt):
    """Closest Point of Approach over [0, dt]; returns (time offset, distance)."""
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


def _record(hist, t, t_pos, i_pos, k_pos, raw, status, stage, t_vel, i_vel, kf_err, kf_sig):
    row = (t, t_pos, i_pos, k_pos, status, stage, raw, t_vel, i_vel, kf_err, kf_sig)
    for key, value in zip(HIST_KEYS, row):
        hist[key].append(value.copy() if isinstance(value, np.ndarray) else value)


def run(cfg, rng, console=None):
    """Run one engagement; returns the history dict consumed by the replay viewer.

    The result depends only on ``cfg`` and the state of ``rng`` (anything with
    ``uniform`` and ``normal``, normally an ``np.random.Generator``).
    """
    console = console or Console()
    dt, g = cfg.sim.dt, cfg.sim.g
    eng = cfg.engagement

    sc = spawn_threat(cfg, rng)
    threat = sc.threat
    console.print(
        f"[dim]Scenario: {sc.altitude/1000:.1f} km alt, {sc.range/1000:.1f} km range, "
        f"aim {sc.aim_radius/1000:.1f} km from origin, T_imp~{sc.impact_time:.1f} s[/dim]"
    )
    interceptor = Interceptor([0, 0, 0], cfg.interceptor, g, cfg.sim.integrator)
    kf = KalmanFilter(dt, cfg.sensor.pos_noise, cfg.sensor.vel_noise, cfg.kalman.q, g)

    t_max = max(cfg.sim.t_max, sc.impact_time * 2.0 + 30.0)

    t = 0.0
    hist = {k: [] for k in HIST_KEYS}
    outcome = "UNKNOWN"
    kill_info = {}
    launch_pip = None
    min_sep = None   # closest approach over all steps since launch, from the CPA check

    console.print("\n" + "="*60, style="bold blue")
    console.print(f"[bold green]{cfg.title}[/bold green]")
    console.print(f"[dim]Dynamic simulation window: 0–{t_max:.1f} s[/dim]")
    console.print("="*60, style="bold blue")

    while t < t_max:
        # Sensor model
        mp = threat.pos + rng.normal(0, cfg.sensor.pos_noise, 3)
        mv = threat.vel + rng.normal(0, cfg.sensor.vel_noise, 3)
        ep, ev = kf.update(mp, mv)
        # Filter error vs truth and its 1-sigma bounds, for the analysis view
        kf_err = np.hstack((ep - threat.pos, ev - threat.vel))
        kf_sig = np.sqrt(np.diag(kf.P))

        # Launch once the track is mature and the threat is descending. The
        # PIP-gated rule also requires the predicted impact point to fall
        # inside the defended area; fire-on-track engages every threat.
        if (t >= eng.min_track_time) and (not interceptor.launched) and (ev[2] < 0):
            pip = kf.predict_impact(g)
            if cfg.pip_gated:
                engage = pip is not None and np.linalg.norm(pip[:2]) <= eng.defended_radius
            else:
                engage = True
            if engage:
                direction = ep - interceptor.pos
                direction = direction / np.linalg.norm(direction)
                interceptor.launch(t, direction)
                launch_pip = pip
                if cfg.pip_gated:
                    console.print(
                        f"[bold yellow][T+{t:05.2f}] LAUNCH:[/bold yellow] PIP "
                        f"{np.linalg.norm(pip[:2])/1000:.1f} km from origin, inside defended area."
                    )
                else:
                    console.print(
                        f"[bold yellow][T+{t:05.2f}] LAUNCH:[/bold yellow] Track confirmed. WEAPONS FREE."
                    )

        # Propagate both bodies from t to t+dt, keeping the start states for the kill check
        t_next = t + dt
        i_p0 = interceptor.pos.copy()
        t_p0 = threat.pos.copy()
        interceptor.update(t, ep, ev, dt)
        threat.update(dt)

        # Kill check: CPA between both straight-line segments over the same step.
        # The segments are the chords between the integrator's step endpoints,
        # so this works the same for any integrator; the chord departs from the
        # curved path by about a*dt^2/8 (a few cm at dt = 0.02 s).
        if interceptor.launched and not interceptor.exploded:
            i_v = (interceptor.pos - i_p0) / dt
            t_v = (threat.pos - t_p0) / dt
            t_off, d_min = get_cpa(i_p0, i_v, t_p0, t_v, dt)
            min_sep = d_min if min_sep is None else min(min_sep, d_min)
            if d_min < eng.kill_radius:
                interceptor.exploded = True
                k_time  = t + t_off
                k_pos_t = t_p0 + t_v * t_off
                k_pos_i = i_p0 + i_v * t_off
                closing = np.linalg.norm(i_v - t_v)

                _record(hist, k_time, k_pos_t, k_pos_i, ep, mp, "HIT", interceptor.stage,
                        t_v, i_v, kf_err, kf_sig)

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

        if not threat.active:
            impact_range = np.linalg.norm(threat.pos[:2])
            if cfg.pip_gated and not interceptor.launched and impact_range > eng.defended_radius:
                outcome = "NO ENGAGE"
                console.print(
                    f"[bold cyan][T+{t_next:05.2f}] HELD FIRE:[/bold cyan] threat impacted "
                    f"{impact_range/1000:.1f} km from origin, outside the defended area."
                )
            else:
                outcome = "GROUND"
                console.print(
                    f"[bold red][T+{t_next:05.2f}] THREAT HIT GROUND BEFORE INTERCEPT.[/bold red]"
                )
            break

        _record(hist, t_next, threat.pos, interceptor.pos, ep, mp,
                "FLYING" if interceptor.launched else "SEARCH", interceptor.stage,
                threat.vel, interceptor.vel, kf_err, kf_sig)

        t = t_next

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
    if launch_pip is not None:
        table.add_row("PIP at Launch", f"{np.linalg.norm(launch_pip[:2])/1000:.1f} km from origin")
    if outcome in ("GROUND", "NO ENGAGE"):
        table.add_row("Impact Point", f"{np.linalg.norm(threat.pos[:2])/1000:.1f} km from origin")
    if outcome == "KILL":
        table.add_row("Kill Time", f"{kill_info['time']:.2f} s")
        table.add_row("Kill Altitude", f"{kill_info['alt']/1000:.1f} km")
        table.add_row("Miss Distance", f"{kill_info['dist']:.2f} m")
        table.add_row("Closing Velocity", f"{kill_info['closing']:.1f} m/s")
    if interceptor.g_hist:
        table.add_row("Max G-Load", f"{max(interceptor.g_hist):.1f} G")
    console.print(table)

    hist['meta'] = {
        'title':           cfg.title,
        'mode':            cfg.mode,
        'engage_rule':     cfg.engage_rule_label,
        'integrator':      cfg.sim.integrator,
        'outcome':         outcome,
        'kill':            kill_info or None,
        'launch_pip':      launch_pip,
        'launch_time':     interceptor.launch_time if interceptor.launched else None,
        'min_separation':  min_sep,
        'divert_dv_used':  interceptor.divert_dv_used,
        'impact':          threat.pos.copy() if outcome in ("GROUND", "NO ENGAGE") else None,
        'g_hist':          interceptor.g_hist,
        'defended_radius': eng.defended_radius,
        'kill_radius':     eng.kill_radius,
        'max_g':           cfg.interceptor.max_g,
        'params':          cfg.params(),
        'min_track_time':  eng.min_track_time if cfg.pip_gated else None,
    }

    return hist


def simulate(cfg, seed, legacy_rng=False, console=None):
    """Run one engagement as a pure function of (config, seed)."""
    return run(cfg, make_rng(seed, legacy_rng), console)
