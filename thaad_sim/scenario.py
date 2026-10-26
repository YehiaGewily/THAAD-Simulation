"""Random threat scenarios drawn from the ranges in the config."""
from dataclasses import dataclass

import numpy as np

from .dynamics import BallisticThreat


@dataclass
class Scenario:
    threat: BallisticThreat
    impact_time: float   # s, nominal (no-drag) time to impact
    altitude: float      # m
    range: float         # m
    aim_radius: float    # m, aim point distance from the origin


def spawn_threat(cfg, rng):
    """Draw a threat aimed at (or scattered around) the origin.

    ``rng`` only needs ``uniform``: an ``np.random.Generator`` or, to replay
    legacy seeds, an ``np.random.RandomState``. The draw order is part of the
    seed contract, so do not reorder it.
    """
    s = cfg.scenario
    g = cfg.sim.g

    T_imp = rng.uniform(*s.impact_time)
    z0    = rng.uniform(*s.altitude)
    r0    = rng.uniform(*s.range)

    theta0 = rng.uniform(0, 2*np.pi)
    x0, y0 = r0*np.cos(theta0), r0*np.sin(theta0)

    if s.aim_radius is None:
        r_imp = 0.0
        x_imp, y_imp = 0.0, 0.0
    else:
        r_imp     = rng.uniform(*s.aim_radius)
        theta_imp = rng.uniform(0, 2*np.pi)
        x_imp, y_imp = r_imp*np.cos(theta_imp), r_imp*np.sin(theta_imp)

    # Vacuum ballistic velocity reaching the aim point after T_imp
    vz0 = (0.5 * g * T_imp**2 - z0) / T_imp
    vx0 = (x_imp - x0) / T_imp
    vy0 = (y_imp - y0) / T_imp

    vel0 = np.array([vx0, vy0, vz0]) * s.drag_comp

    threat = BallisticThreat([x0, y0, z0], vel0, cfg.threat.drag, g)
    return Scenario(threat, T_imp, z0, r0, r_imp)
