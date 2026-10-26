from pathlib import Path

import pytest
from rich.console import Console

from thaad_sim import load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {name: ROOT / "configs" / f"{name}.yaml" for name in ("engineering", "demo")}


@pytest.fixture
def quiet():
    return Console(quiet=True)


@pytest.fixture(params=sorted(CONFIGS))
def mode_config(request):
    return request.param, load_config(CONFIGS[request.param])
