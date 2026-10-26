import math

import pytest

from thaad_sim.stats import Z95, wilson_interval


def _wald(k, n, z=Z95):
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


@pytest.mark.parametrize("k,n,expected", [
    # Closed forms: at k = 0 the upper end is z²/(n + z²); at k = n the lower end is n/(n + z²)
    (0, 10, (0.0, Z95**2 / (10 + Z95**2))),
    (10, 10, (10 / (10 + Z95**2), 1.0)),
    # Worked examples of Newcombe (1998), method 3, given to 4 decimals
    (81, 263, (0.2553, 0.3662)),
    (1, 29, (0.0061, 0.1718)),
    # p = 1/2: centre 1/2, half-width z sqrt(n/4 + z²/4) / (n + z²)
    (50, 100, (0.5 - Z95 * math.sqrt(25 + Z95**2 / 4) / (100 + Z95**2),
               0.5 + Z95 * math.sqrt(25 + Z95**2 / 4) / (100 + Z95**2))),
])
def test_wilson_matches_reference_values(k, n, expected):
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(expected[0], abs=5e-5)
    assert hi == pytest.approx(expected[1], abs=5e-5)


def test_wilson_is_the_score_interval():
    # The ends are the two proportions p0 whose score test |p - p0| / sqrt(p0 (1 - p0) / n) = z
    k, n = 37, 120
    p = k / n
    for p0 in wilson_interval(k, n):
        assert abs(p - p0) / math.sqrt(p0 * (1 - p0) / n) == pytest.approx(Z95, rel=1e-12)


@pytest.mark.parametrize("n", [1, 7, 50, 1000])
def test_wilson_contains_estimate_and_stays_in_unit_interval(n):
    for k in range(0, n + 1, max(1, n // 25)):
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0


def test_wilson_is_symmetric_and_narrows_with_n():
    lo, hi = wilson_interval(13, 40)
    lo2, hi2 = wilson_interval(27, 40)
    assert (lo, hi) == pytest.approx((1 - hi2, 1 - lo2), abs=1e-15)
    widths = [hi - lo for lo, hi in (wilson_interval(n // 4, n) for n in (40, 400, 4000))]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_differs_from_wald_where_wald_fails():
    # Wald collapses to a point at k = 0 and leaves [0, 1] near the edges; Wilson does neither
    assert _wald(0, 30) == (0.0, 0.0)
    assert wilson_interval(0, 30)[1] > 0.1
    assert _wald(1, 30)[0] < 0 <= wilson_interval(1, 30)[0]
    # Far from the edges with a large n the two agree closely
    assert wilson_interval(500, 1000) == pytest.approx(_wald(500, 1000), abs=1e-3)


@pytest.mark.parametrize("k,n", [(0, 0), (-1, 10), (11, 10)])
def test_wilson_rejects_invalid_counts(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)
