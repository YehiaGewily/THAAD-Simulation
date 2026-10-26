"""Interval estimates for outcome rates measured over batches of seeded runs."""
import math

Z95 = 1.959963984540054   # standard-normal 97.5 % quantile, for two-sided 95 % intervals


def wilson_interval(k, n, z=Z95):
    """Wilson score interval for a binomial proportion with ``k`` successes in ``n`` trials.

    Unlike the normal (Wald) interval p ± z·sqrt(p(1-p)/n), it stays inside
    [0, 1] and keeps close to its nominal coverage when k is near 0 or n,
    where rates like 1000 kills in 1000 runs sit. Returns (low, high).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must lie in [0, n], got k={k}, n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    # At k = 0 (k = n) the low (high) end is exactly 0 (1); set it rather than leave rounding
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high
