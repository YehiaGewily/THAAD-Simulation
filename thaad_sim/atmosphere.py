"""Air density relative to sea level, shared by the truth dynamics, the
7-state EKF, the drag feed-forward and the drag impact predictor.

Two models are selected by ``sim.atmosphere``:

* ``constant``: density does not change with altitude, so a quadratic drag
  coefficient k acts at full strength everywhere (the original model).
* ``exponential``: rho(z) / rho_0 = exp(-z / H) with scale height
  ``sim.scale_height``. A drag coefficient is then its sea-level value k_0, and
  the drag acceleration is -k_0 exp(-z / H) |v| v.

Functions take ``scale_height``: ``None`` means constant density.
"""
import math

ATMOSPHERES = ("constant", "exponential")
DEFAULT_SCALE_HEIGHT = 7000.0   # m


def density_ratio(z, scale_height=None):
    """rho(z) / rho_0. Altitudes below the ground are treated as z = 0."""
    if scale_height is None:
        return 1.0
    return math.exp(-max(float(z), 0.0) / scale_height)
