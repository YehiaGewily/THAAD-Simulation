"""The ablation script's fast engagement loop must match the reference batch runner."""
import importlib.util

import pytest

from thaad_sim.batch import run_metrics

from conftest import ROOT

_spec = importlib.util.spec_from_file_location("ablation_study", ROOT / "scripts" / "ablation_study.py")
ablation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation)

CONFIGS = {cid: cfg for cid, _, _, cfg in ablation.build_configs()}
COMPARED = ("outcome", "launched", "launch_time_s", "launch_pip_range_m", "kill_time_s",
            "miss_distance_m", "divert_dv_used_mps", "impact_range_m")


@pytest.mark.parametrize("cfg_id", ["C1_base_cart", "C5_kf6_vac_apn", "C9_adv_rae"])
@pytest.mark.parametrize("seed", [4, 13])
def test_fast_loop_matches_batch_run_metrics(cfg_id, seed):
    cfg = CONFIGS[cfg_id]
    fast = ablation.fast_run_metrics((cfg_id, cfg, seed))
    ref = run_metrics(cfg, seed)
    for col in COMPARED:
        if isinstance(ref[col], float):
            assert fast[col] == pytest.approx(ref[col], rel=1e-12, abs=1e-9), col
        else:
            assert fast[col] == ref[col], col


_sens_spec = importlib.util.spec_from_file_location("sensitivity_study", ROOT / "scripts" / "sensitivity_study.py")
sensitivity = importlib.util.module_from_spec(_sens_spec)
_sens_spec.loader.exec_module(sensitivity)


def test_noise_factor_scales_every_radar_noise_and_nothing_else():
    cfgs = {cid: c for cid, _, _, c in ablation.build_configs()}
    for cid in ("C7_ekf7_drag_apn", "C9_adv_rae"):
        base = cfgs[cid]
        noisy = sensitivity.with_noise_factor(base, 2.0)
        for f in ("pos_noise", "vel_noise", "range_noise", "angle_noise", "range_rate_noise"):
            assert getattr(noisy.sensor, f) == pytest.approx(2.0 * getattr(base.sensor, f)), f
        assert noisy.sensor.model == base.sensor.model
        assert (noisy.threat, noisy.interceptor, noisy.kalman, noisy.scenario) == \
            (base.threat, base.interceptor, base.kalman, base.scenario)
    # Each budget's sweep sets that budget and the factor
    values, apply = sensitivity.SWEEPS["noise_150"]
    cfg = apply(cfgs["C5_kf6_vac_apn"], 4.0)
    assert cfg.interceptor.kv_divert_dv == 150.0
    assert cfg.sensor.pos_noise == pytest.approx(4.0 * cfgs["C5_kf6_vac_apn"].sensor.pos_noise)
    assert values == [0.25, 0.5, 1.0, 2.0, 4.0]


@pytest.mark.parametrize("cfg_id", ["C5_kf6_vac_apn", "C9_adv_rae"])
def test_fast_loop_matches_batch_run_metrics_with_scaled_radar_noise(cfg_id):
    cfg = sensitivity.with_noise_factor({cid: c for cid, _, _, c in ablation.build_configs()}[cfg_id], 2.0)
    fast = ablation.fast_run_metrics((cfg_id, cfg, 13))
    ref = run_metrics(cfg, 13)
    for col in ("outcome", "launched", "launch_time_s", "miss_distance_m", "divert_dv_used_mps"):
        if isinstance(ref[col], float):
            assert fast[col] == pytest.approx(ref[col], rel=1e-12, abs=1e-9), col
        else:
            assert fast[col] == ref[col], col
