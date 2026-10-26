import csv

import pytest

from thaad_sim import load_config
from thaad_sim.batch import COLUMNS, SWEEP_COLUMNS, main, parse_sweep, run_metrics, with_param

from conftest import CONFIGS

DEMO = str(CONFIGS["demo"])


def _batch(out_dir, *extra):
    main(["--config", DEMO, "--n", "3", "--out-dir", str(out_dir), *extra])
    return (out_dir / "demo.csv").read_bytes()


def _rows(data):
    return list(csv.DictReader(data.decode("utf-8").splitlines()))


def test_same_seed_gives_the_same_csv(tmp_path):
    first = _batch(tmp_path / "a", "--seed", "4", "--workers", "1")
    second = _batch(tmp_path / "b", "--seed", "4", "--workers", "1")
    assert first == second
    rows = _rows(first)
    assert [r['seed'] for r in rows] == ["4", "5", "6"]
    assert tuple(rows[0]) == COLUMNS


def test_csv_does_not_depend_on_worker_count(tmp_path):
    serial = _batch(tmp_path / "serial", "--seed", "4", "--workers", "1")
    parallel = _batch(tmp_path / "parallel", "--seed", "4", "--workers", "2")
    assert serial == parallel


def test_different_seed_gives_a_different_csv(tmp_path):
    assert _batch(tmp_path / "a", "--seed", "4", "--workers", "1") != \
        _batch(tmp_path / "b", "--seed", "40", "--workers", "1")


def test_row_matches_the_run():
    cfg = load_config(CONFIGS["demo"])
    row = run_metrics(cfg, 0)
    assert row['outcome'] == "KILL" and row['launched'] == 1
    assert row['miss_distance_m'] < cfg.engagement.kill_radius
    assert row['kill_time_s'] == row['end_time_s'] > row['launch_time_s']
    assert row['impact_range_m'] is None
    # The filter beats the raw radar error, sqrt(3) sigma for a 3D error
    assert row['kf_rms_pos_m'] < 3 ** 0.5 * cfg.sensor.pos_noise
    assert row['kf_rms_vel_mps'] < 3 ** 0.5 * cfg.sensor.vel_noise


def test_parse_sweep_and_with_param():
    assert parse_sweep("sensor.pos_noise=5,10, 25") == ("sensor.pos_noise", [5, 10, 25])
    assert parse_sweep("sim.integrator=euler,rk4") == ("sim.integrator", ["euler", "rk4"])
    cfg = load_config(CONFIGS["demo"])
    swept = with_param(cfg, "sensor.pos_noise", 5)
    assert swept.sensor.pos_noise == 5.0 and swept.sensor.vel_noise == cfg.sensor.vel_noise
    assert swept.kalman == cfg.kalman
    with pytest.raises(ValueError):
        with_param(cfg, "sensor.no_such_field", 1)
    with pytest.raises(ValueError):
        with_param(cfg, "sim.integrator", "midpoint")   # SimConfig validation still applies


def test_sweep_repeats_the_seeds_at_each_value(tmp_path):
    main(["--config", DEMO, "--n", "2", "--seed", "0", "--workers", "1",
          "--out-dir", str(tmp_path), "--sweep", "sensor.pos_noise=10,50"])
    rows = _rows((tmp_path / "demo_sweep_sensor_pos_noise.csv").read_bytes())
    assert tuple(rows[0]) == SWEEP_COLUMNS + COLUMNS
    assert [(r['sweep_value'], r['seed']) for r in rows] == \
        [("10", "0"), ("10", "1"), ("50", "0"), ("50", "1")]
    assert [r['pos_noise_m'] for r in rows] == ["10", "10", "50", "50"]
