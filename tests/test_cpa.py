import numpy as np
import pytest

from thaad_sim import get_cpa

DT = 0.02


def v(*c):
    return np.array(c, dtype=float)


def test_head_on_collision_inside_step():
    # Closing at 2000 m/s from 20 m apart: they meet halfway through the step
    t, d = get_cpa(v(0, 0, 0), v(1000, 0, 0), v(20, 0, 0), v(-1000, 0, 0), DT)
    assert t == pytest.approx(0.01)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_lateral_offset_gives_miss_distance():
    t, d = get_cpa(v(0, 0, 0), v(1000, 0, 0), v(20, 5, 0), v(-1000, 0, 0), DT)
    assert t == pytest.approx(0.01)
    assert d == pytest.approx(5.0)


def test_parallel_motion_returns_start_distance():
    t, d = get_cpa(v(0, 0, 0), v(300, 0, 0), v(0, 3, 4), v(300, 0, 0), DT)
    assert t == 0.0
    assert d == pytest.approx(5.0)


def test_separating_bodies_closest_at_start():
    t, d = get_cpa(v(0, 0, 0), v(-500, 0, 0), v(10, 0, 0), v(500, 0, 0), DT)
    assert t == 0.0
    assert d == pytest.approx(10.0)


def test_closest_approach_after_step_uses_end():
    # Closing at 1000 m/s from 100 m: closest point is at 0.1 s, past the step
    t, d = get_cpa(v(0, 0, 0), v(500, 0, 0), v(100, 0, 0), v(-500, 0, 0), DT)
    assert t == DT
    assert d == pytest.approx(80.0)


def test_symmetric_in_body_order():
    a = (v(1, 2, 3), v(900, -50, 20), v(15, 4, 2), v(-1200, 30, -10))
    t1, d1 = get_cpa(*a, DT)
    t2, d2 = get_cpa(a[2], a[3], a[0], a[1], DT)
    assert t1 == pytest.approx(t2)
    assert d1 == pytest.approx(d2)
