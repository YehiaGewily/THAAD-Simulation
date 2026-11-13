"""Tests for the live workstation's FastAPI server and its telemetry payload."""
import json

import pytest

pytest.importorskip("fastapi", reason="needs the workstation extra: pip install -e \".[workstation]\"")

from thaad_sim import load_config
from thaad_sim.engagement import simulate
from thaad_sim.server import (
    CompareItem,
    CompareRequest,
    LiveBatchRequest,
    SimOverrides,
    SimRequest,
    api_batch,
    api_batch_run,
    api_compare,
    api_simulate,
    get_configs,
)
from thaad_sim.telemetry import build_workstation_telemetry
from conftest import ROOT


def test_build_workstation_telemetry_contains_all_derived_series():
    cfg = load_config(ROOT / "configs" / "advanced.yaml")
    hist = simulate(cfg, 13)
    payload = build_workstation_telemetry(hist)
    drv = payload["derived"]
    n = len(payload["t"])

    for key in (
        "altT", "spdT", "rngT", "altI", "spdI", "rngI",
        "sep", "vc", "ttg", "losTrue", "losEst", "zemTrue", "zemEst",
        "dvUsed", "mass", "pipValid", "pipSigE", "pipSigN", "pipBias", "pipVacBias",
        "epos", "evel", "spos3", "svel3", "eraw", "ex", "ey", "ez",
    ):
        assert len(drv[key]) == n, f"length mismatch for {key}"

    assert len(drv["zemVec"]) == 3 * n
    assert len(drv["acVec"]) == 3 * n
    assert len(drv["pipTrack"]) == 3 * n
    assert drv["kfStats"] is not None
    assert len(drv["events"]) >= 3


def test_fastapi_endpoints_simulate_compare_and_batch():
    cfgs = get_configs()
    assert len(cfgs["configs"]) >= 3
    assert len(cfgs["presets"]) >= 5

    res = api_simulate(
        SimRequest(
            config="configs/advanced.yaml",
            seed=13,
            overrides=SimOverrides(filter_type="ekf7", guidance_law="apn", pip_mode="drag"),
        )
    )
    assert res.status_code == 200
    body = json.loads(res.body)
    assert body["outcome"] == "KILL"
    assert "derived" in body

    cmp_res = api_compare(
        CompareRequest(
            runs=[
                CompareItem(label="EKF7+APN", config="configs/advanced.yaml", seed=13),
                CompareItem(label="KF6+PN", config="configs/engineering.yaml", seed=13),
            ]
        )
    )
    assert cmp_res.status_code == 200
    cmp_body = json.loads(cmp_res.body)
    assert len(cmp_body["runs"]) == 2
    assert cmp_body["runs"][0]["outcome"] == "KILL"
    assert cmp_body["runs"][1]["outcome"] == "GROUND"

    batch_info = api_batch("engineering")
    assert batch_info["data"]["runs"] == 1000

    live_batch = api_batch_run(LiveBatchRequest(config="configs/demo.yaml", seed_start=0, n=5))
    assert live_batch["runs"] == 5
    assert len(live_batch["seedSummaries"]) == 5
