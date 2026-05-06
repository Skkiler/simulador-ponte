from __future__ import annotations

from copy import deepcopy

import pytest

from src.services.config_service import ConfigService


@pytest.fixture
def base_cfg() -> dict:
    cfg = ConfigService().load("bridge_config.json")
    return deepcopy(cfg)

