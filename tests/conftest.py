from pathlib import Path

import pytest

from crib_monitor.config import Config, load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config() -> Config:
    return load_config(ROOT / "config.example.toml")
