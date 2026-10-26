"""The browser-side outputs: replay payload, built templates, dashboard data, palette."""
import importlib.util
import json
import re

import numpy as np
import pytest

from thaad_sim import load_config
from thaad_sim.dashboard import build_data, write_dashboard
from thaad_sim.dashboard import TEMPLATE as DASHBOARD_TEMPLATE
from thaad_sim.engagement import POS_COV_INDEX, simulate
from thaad_sim.estimation import KalmanFilter
from thaad_sim.replay_viewer import TEMPLATE as REPLAY_TEMPLATE
from thaad_sim.replay_viewer import build_payload

from conftest import CONFIGS, ROOT


@pytest.fixture(scope="module")
def demo_hist():
    from rich.console import Console
    return simulate(load_config(CONFIGS["demo"]), 3, console=Console(quiet=True))


def test_payload_carries_the_position_covariance(demo_hist):
    payload = build_payload(demo_hist)
    n = len(payload['t'])
    assert len(payload['kfCov']) == 6 * n
    cov = np.asarray(payload['kfCov']).reshape(n, 6)
    sig = np.asarray(payload['kfSig']).reshape(n, 6)
    # The diagonal entries are the variances whose roots the sigma column holds
    assert np.allclose(np.sqrt(cov[:, :3]), sig[:, :3], atol=0.02)


def test_position_covariance_index_picks_the_unique_entries():
    kf = KalmanFilter(0.02, 25.0, 5.0, 30.0, 9.81)
    A = np.arange(36.0).reshape(6, 6)
    kf.P = A + A.T
    xx, yy, zz, xy, xz, yz = kf.P[POS_COV_INDEX]
    P = kf.P
    assert (xx, yy, zz, xy, xz, yz) == (P[0, 0], P[1, 1], P[2, 2], P[0, 1], P[0, 2], P[1, 2])


@pytest.mark.parametrize("template, token", [(REPLAY_TEMPLATE, "__REPLAY_DATA__"),
                                             (DASHBOARD_TEMPLATE, "__DASHBOARD_DATA__")])
def test_built_templates_are_self_contained(template, token):
    html = template.read_text(encoding="utf-8")
    assert html.count(token) == 1
    # Scripts are inlined by the build: no import map and no external module scripts
    assert "importmap" not in html
    assert not re.search(r'<script[^>]*type="module"[^>]*\bsrc=', html)
    assert "/src/" not in html


def test_replay_template_has_every_layer_toggle():
    html = REPLAY_TEMPLATE.read_text(encoding="utf-8")
    for layer in ("radar", "track", "cov", "los", "labels", "slowmo"):
        assert f'data-layer="{layer}"' in html


def test_dashboard_data_from_the_committed_batch(tmp_path):
    data = build_data(ROOT / "results" / "engineering.csv",
                      ROOT / "results" / "consistency.json", ROOT / "results" / "convergence.json")
    rates = {r['label']: r for r in data['rates']}
    assert data['runs'] == 1000 and data['seeds'] == [0, 999]
    kill = rates['KILL | engaged']
    assert (kill['k'], kill['n']) == (575, 960)
    assert kill['lo'] < kill['p'] < kill['hi']
    for h in data['histograms']:
        assert len(h['edges']) == len(h['counts']) + 1
        assert sum(h['counts']) == h['n']
    assert data['consistency']['runs'] == 50
    assert {p['problem'] for p in data['convergence']['convergence']} == {"threat", "interceptor"}

    out = write_dashboard(data, tmp_path / "dash.html")
    html = out.read_text(encoding="utf-8")
    assert "__DASHBOARD_DATA__" not in html
    blob = re.search(r'<script id="dashboard-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(blob)['runs'] == 1000


def test_dashboard_rejects_sweep_csvs():
    with pytest.raises(SystemExit, match="sweep"):
        build_data(ROOT / "results" / "engineering_sweep_sensor_pos_noise.csv")


def test_palette_passes_its_colour_vision_and_contrast_checks(capsys):
    spec = importlib.util.spec_from_file_location("palette_check", ROOT / "scripts" / "palette_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0, capsys.readouterr().out
