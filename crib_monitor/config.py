"""Typed configuration (config.toml) and secrets (environment)."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class ConfigError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScheduleConfig(_Strict):
    cycle_anchor: date
    cycle_days: int = Field(14, ge=1)
    nights: list[Weekday]
    window_start: time
    window_end: time
    extra_dates: list[date] = []
    skip_dates: list[date] = []
    timezone: str

    @field_validator("cycle_days")
    @classmethod
    def _multiple_of_7(cls, value: int) -> int:
        if value % 7 != 0:
            raise ValueError("cycle_days must be a multiple of 7")
        return value

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value


class CribCrop(_Strict):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    @model_validator(mode="after")
    def _even(self) -> CribCrop:
        for name in ("x", "y", "w", "h"):
            if getattr(self, name) % 2:
                raise ValueError(f"camera.crop.{name} must be even")
        return self


class CameraConfig(_Strict):
    crop: CribCrop
    stream_timeout_s: float = 5
    max_backoff_s: float = 15


class MotionConfig(_Strict):
    threshold: float = 0.02
    pixel_delta: int = 25
    min_check_interval_s: float = 10
    max_check_interval_s: float = 90


class ModelConfig(_Strict):
    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 30
    extra_body: dict[str, Any] = {}


class ClassifierConfig(_Strict):
    max_side_px: int = 768
    models: list[ModelConfig] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _unique_names(cls, models: list[ModelConfig]) -> list[ModelConfig]:
        # Health tracks each model by name, so two models with one name would share a streak.
        names = [m.name for m in models]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError("duplicate model name(s): " + ", ".join(duplicates))
        return models


class AlertConfig(_Strict):
    shadow_mode: bool = True
    emergency_retry_s: int = 60
    emergency_expire_s: int = 1800
    receipt_poll_s: float = 15
    ack_suppress_s: float = 180
    health_repeat_s: float = 900


class ArmingConfig(_Strict):
    manual_max_hours: float = 4
    pause_minutes: float = 45


class StorageConfig(_Strict):
    data_dir: Path = Path("data")
    retention_days: int = 30


class WebConfig(_Strict):
    host: str
    port: int = 8080


class HealthConfig(_Strict):
    blind_after_s: float = 120
    model_down_after: int = 3
    heartbeat_interval_s: float = 60


class Config(_Strict):
    schedule: ScheduleConfig
    camera: CameraConfig
    motion: MotionConfig = MotionConfig()
    classifier: ClassifierConfig
    alerts: AlertConfig = AlertConfig()
    arming: ArmingConfig = ArmingConfig()
    storage: StorageConfig = StorageConfig()
    web: WebConfig
    health: HealthConfig = HealthConfig()


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(Path(path).read_text())
        return Config.model_validate(raw)
    except (OSError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


_SECRET_VARS = {
    "tapo_rtsp_url": "TAPO_RTSP_URL",
    "pushover_token": "PUSHOVER_TOKEN",
    "pushover_user": "PUSHOVER_USER",
    "healthchecks_url": "HEALTHCHECKS_URL",
    "control_token": "CONTROL_TOKEN",
}


@dataclass(frozen=True)
class Secrets:
    tapo_rtsp_url: str
    pushover_token: str
    pushover_user: str
    healthchecks_url: str
    control_token: str


def load_secrets(env: Mapping[str, str], cfg: Config) -> Secrets:
    missing = [var for var in _SECRET_VARS.values() if not env.get(var)]
    missing += [m.api_key_env for m in cfg.classifier.models if m.api_key_env and not env.get(m.api_key_env)]
    if missing:
        raise ConfigError("missing environment variables: " + ", ".join(missing))
    return Secrets(**{field: env[var] for field, var in _SECRET_VARS.items()})
