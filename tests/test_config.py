from datetime import date, time
from pathlib import Path

import pytest

from crib_monitor.config import ConfigError, load_config, load_secrets

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = (ROOT / "config.example.toml").read_text()

FULL_ENV = {
    "TAPO_RTSP_URL": "rtsp://u:p@cam:554/stream1",
    "PUSHOVER_TOKEN": "tok",
    "PUSHOVER_USER": "usr",
    "HEALTHCHECKS_URL": "https://hc-ping.com/abc",
    "CONTROL_TOKEN": "secret",
    "OPENROUTER_API_KEY": "sk-or",
}


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_example_config_loads(config):
    assert config.schedule.cycle_anchor == date(2026, 9, 25)
    assert config.schedule.window_start == time(19, 0)
    assert config.schedule.timezone == "America/New_York"
    assert [m.name for m in config.classifier.models] == ["local", "cloud"]
    assert config.classifier.models[1].extra_body == {"provider": {"data_collection": "deny"}}
    assert config.alerts.shadow_mode is True


def test_unknown_key_rejected(tmp_path):
    path = write(tmp_path, EXAMPLE.replace("[motion]", "[motion]\ntreshold = 0.5"))
    with pytest.raises(ConfigError, match="treshold"):
        load_config(path)


def test_bad_timezone_rejected(tmp_path):
    path = write(tmp_path, EXAMPLE.replace("America/New_York", "Mars/Olympus"))
    with pytest.raises(ConfigError, match="timezone"):
        load_config(path)


def test_odd_crop_rejected(tmp_path):
    path = write(tmp_path, EXAMPLE.replace("w = 1280", "w = 1281"))
    with pytest.raises(ConfigError, match="even"):
        load_config(path)


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.toml")


def test_secrets_loaded(config):
    secrets = load_secrets(FULL_ENV, config)
    assert secrets.control_token == "secret"
    assert secrets.tapo_rtsp_url.startswith("rtsp://")


def test_missing_secrets_are_all_listed(config):
    env = {k: v for k, v in FULL_ENV.items() if k not in ("PUSHOVER_USER", "OPENROUTER_API_KEY")}
    with pytest.raises(ConfigError) as err:
        load_secrets(env, config)
    assert "PUSHOVER_USER" in str(err.value)
    assert "OPENROUTER_API_KEY" in str(err.value)


def test_cycle_days_not_multiple_of_7_rejected(tmp_path):
    path = write(tmp_path, EXAMPLE.replace("cycle_days = 14", "cycle_days = 10"))
    with pytest.raises(ConfigError, match="multiple of 7"):
        load_config(path)


def test_duplicate_model_names_rejected(tmp_path):
    path = write(tmp_path, EXAMPLE.replace('name = "cloud"', 'name = "local"'))
    with pytest.raises(ConfigError, match="duplicate model name"):
        load_config(path)
