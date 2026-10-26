import pytest
import yaml

from thaad_sim.config import config_from_dict, load_config

from conftest import CONFIGS


def test_engineering_config_values():
    cfg = load_config(CONFIGS["engineering"])
    assert cfg.engagement.launch_rule == "pip_gated"
    assert cfg.interceptor.kv_divert_dv == 435.0
    assert cfg.scenario.aim_radius == (0.0, 35000.0)
    assert cfg.params()['min_track_time'] == 5.0


def test_demo_config_values():
    cfg = load_config(CONFIGS["demo"])
    assert cfg.engagement.launch_rule == "fire_on_track"
    assert cfg.interceptor.kv_divert_dv is None
    assert cfg.scenario.aim_radius is None
    # The viewer reads a missing divert budget as "unlimited"
    assert 'kv_divert_dv' not in cfg.params()


def _raw(name="engineering"):
    return yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))


def test_unknown_key_is_rejected():
    data = _raw()
    data['interceptor']['thrust'] = 1.0
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict(data)


def test_missing_key_is_rejected():
    data = _raw()
    del data['sensor']['pos_noise']
    with pytest.raises(ValueError, match="missing key"):
        config_from_dict(data)


def test_bad_launch_rule_is_rejected():
    data = _raw()
    data['engagement']['launch_rule'] = "always"
    with pytest.raises(ValueError, match="launch_rule"):
        config_from_dict(data)
