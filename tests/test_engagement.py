import json
from pathlib import Path

import numpy as np
import pytest

from thaad_sim import load_config, simulate

from conftest import CONFIGS

BASELINES = json.loads((Path(__file__).parent / "data" / "legacy_baselines.json").read_text())


def _assert_same_run(a, b):
    for key in a:
        if key == 'meta':
            continue
        if key in ('status', 'stage'):
            assert a[key] == b[key], key
        else:
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert a['meta']['outcome'] == b['meta']['outcome']
    assert a['meta']['kill'] == b['meta']['kill']
    assert a['meta']['g_hist'] == b['meta']['g_hist']


@pytest.mark.parametrize("legacy_rng", [False, True])
def test_fixed_seed_is_deterministic(mode_config, quiet, legacy_rng):
    _, cfg = mode_config
    first = simulate(cfg, 123, legacy_rng=legacy_rng, console=quiet)
    second = simulate(cfg, 123, legacy_rng=legacy_rng, console=quiet)
    _assert_same_run(first, second)


def test_different_seeds_differ(mode_config, quiet):
    _, cfg = mode_config
    a = simulate(cfg, 1, console=quiet)
    b = simulate(cfg, 2, console=quiet)
    assert not np.array_equal(a['t_pos'][0], b['t_pos'][0])


LEGACY_CASES = [
    (mode, int(seed), run['outcome'], run['kill_time'])
    for mode in ("engineering", "demo")
    for seed, run in BASELINES[mode]['runs'].items()
]


@pytest.mark.parametrize("mode,seed,outcome,kill_time", LEGACY_CASES,
                         ids=[f"{m}-{s}" for m, s, _, _ in LEGACY_CASES])
def test_reproduces_legacy_scripts(mode, seed, outcome, kill_time, quiet):
    hist = simulate(load_config(CONFIGS[mode]), seed, legacy_rng=True, console=quiet)
    meta = hist['meta']
    assert meta['outcome'] == outcome
    if kill_time is None:
        assert meta['kill'] is None
    else:
        assert meta['kill']['time'] == pytest.approx(kill_time, rel=0, abs=1e-9)


def test_baselines_cover_every_outcome():
    outcomes = {r['outcome'] for r in BASELINES['engineering']['runs'].values()}
    assert {"KILL", "GROUND", "NO ENGAGE"} <= outcomes
