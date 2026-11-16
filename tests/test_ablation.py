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
