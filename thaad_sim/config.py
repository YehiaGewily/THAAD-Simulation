"""Simulation configuration, loaded from a YAML file (see configs/)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import yaml

LAUNCH_RULES = ("pip_gated", "fire_on_track")

# Engagement-rule labels shown in the terminal and the replay viewer
ENGAGE_RULE_LABELS = {"pip_gated": "PIP-GATED LAUNCH", "fire_on_track": "FIRE ON TRACK"}

Range = Tuple[float, float]


@dataclass(frozen=True)
class SimSettings:
    dt: float          # s, integration step
    t_max: float       # s, base window (extended per scenario)
    g: float           # m/s^2


@dataclass(frozen=True)
class SensorSettings:
    pos_noise: float   # m, 1-sigma per axis
    vel_noise: float   # m/s, 1-sigma per axis


@dataclass(frozen=True)
class KalmanSettings:
    q: float           # process-noise variance (diagonal)
    p0: float          # initial covariance (diagonal)


@dataclass(frozen=True)
class ThreatSettings:
    drag: float        # 1/m, quadratic drag (acceleration per speed^2)


@dataclass(frozen=True)
class InterceptorSettings:
    boost_thrust: float            # N
    boost_time: float              # s
    mass_wet: float                # kg
    mass_dry: float                # kg
    kv_mass: float                 # kg
    kv_drag: float                 # kill-vehicle quadratic drag coefficient
    booster_drag: float            # booster quadratic drag coefficient
    pn_gain: float
    max_g: float                   # lateral acceleration limit, g
    kv_divert_dv: Optional[float]  # m/s divert budget; None = unlimited


@dataclass(frozen=True)
class EngagementSettings:
    launch_rule: str               # "pip_gated" or "fire_on_track"
    min_track_time: float          # s of tracking before a launch is allowed
    defended_radius: float         # m; with pip_gated, engage only if the PIP falls inside
    kill_radius: float             # m


@dataclass(frozen=True)
class ScenarioSettings:
    impact_time: Range             # s, nominal (no-drag) time to impact
    altitude: Range                # m, initial altitude
    range: Range                   # m, initial ground range
    aim_radius: Optional[Range]    # m, aim-point scatter around the origin; None = aim at origin
    drag_comp: float               # launch-velocity boost compensating for drag


@dataclass(frozen=True)
class SimConfig:
    title: str
    mode: str
    sim: SimSettings
    sensor: SensorSettings
    kalman: KalmanSettings
    threat: ThreatSettings
    interceptor: InterceptorSettings
    engagement: EngagementSettings
    scenario: ScenarioSettings

    def __post_init__(self):
        if self.engagement.launch_rule not in LAUNCH_RULES:
            raise ValueError(f"engagement.launch_rule must be one of {LAUNCH_RULES}, "
                             f"got {self.engagement.launch_rule!r}")

    @property
    def pip_gated(self):
        return self.engagement.launch_rule == "pip_gated"

    @property
    def engage_rule_label(self):
        return ENGAGE_RULE_LABELS[self.engagement.launch_rule]

    def params(self):
        """Flat parameter dict reported to the replay viewer's Methods table."""
        i, e = self.interceptor, self.engagement
        p = {
            'dt': self.sim.dt, 't_max': self.sim.t_max, 'g': self.sim.g,
            'pos_noise': self.sensor.pos_noise, 'vel_noise': self.sensor.vel_noise,
            'boost_thrust': i.boost_thrust, 'boost_time': i.boost_time,
            'mass_wet': i.mass_wet, 'mass_dry': i.mass_dry, 'kv_mass': i.kv_mass,
            'kv_drag': i.kv_drag, 'pn_gain': i.pn_gain, 'max_g': i.max_g,
            'kill_radius': e.kill_radius, 'kv_divert_dv': i.kv_divert_dv,
            'booster_drag': i.booster_drag, 'threat_drag': self.threat.drag,
            'kf_q': self.kalman.q, 'kf_p0': self.kalman.p0,
            'defended_radius': e.defended_radius,
            # Only the PIP-gated rule treats the track time as a launch gate
            'min_track_time': e.min_track_time if self.pip_gated else None,
        }
        return {k: v for k, v in p.items() if v is not None}


SECTIONS = {
    'sim': SimSettings,
    'sensor': SensorSettings,
    'kalman': KalmanSettings,
    'threat': ThreatSettings,
    'interceptor': InterceptorSettings,
    'engagement': EngagementSettings,
    'scenario': ScenarioSettings,
}


def _coerce(value):
    """YAML numbers (or number lists) -> float / tuple of floats; strings and None pass through."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"expected a [low, high] pair, got {value!r}")
        return tuple(float(v) for v in value)
    return float(value)


def _section(cls, name, data):
    if not isinstance(data, dict):
        raise ValueError(f"config section '{name}' must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    missing = known - set(data)
    if unknown:
        raise ValueError(f"unknown key(s) in '{name}': {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing key(s) in '{name}': {', '.join(sorted(missing))}")
    return cls(**{k: _coerce(v) for k, v in data.items()})


def config_from_dict(data):
    expected = {'title', 'mode', *SECTIONS}
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueError(f"unknown top-level key(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"missing top-level key(s): {', '.join(sorted(missing))}")
    sections = {name: _section(cls, name, data[name]) for name, cls in SECTIONS.items()}
    return SimConfig(title=str(data['title']), mode=str(data['mode']), **sections)


def load_config(path):
    """Load a SimConfig from a YAML file."""
    with Path(path).open(encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f))
