# Crib Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python service on an Ubuntu server that watches a Tapo C210 crib camera during scheduled or manual sessions, classifies the baby's position with a local and a cloud vision model, and sends Pushover alerts when he is on his stomach or when the monitor itself stops working.

**Architecture:** One asyncio process. A capture thread keeps a persistent ffmpeg RTSP connection and exposes the latest cropped frame. An orchestrator (`Monitor.tick`, once per second) gates on motion, calls both models concurrently through LiteLLM, feeds the combined label into a pure decision state machine, and turns its actions into Pushover calls. Arming (schedule, manual On/Off, Pause) and health tracking are separate, clock-injected units. A small FastAPI page on the LAN controls it.

**Tech Stack:** Python 3.12, uv, LiteLLM (library), OpenCV (headless), NumPy, httpx, FastAPI + uvicorn, pydantic v2, pytest + pytest-asyncio, ffmpeg (system), mediamtx (integration test only).

**Spec:** `docs/superpowers/specs/2026-09-24-crib-monitor-design.md`

## Global Constraints

- Python `>=3.12`; dependencies pinned to exact versions in `uv.lock`; LiteLLM pinned with `==` in `pyproject.toml`.
- LiteLLM is used as a library inside the service, never as a separate proxy.
- Timezone `America/New_York`.
- No camera stream and no model calls while disarmed.
- Pushover priorities: `2` for a confirmed stomach roll (`retry=60`, `expire=1800`); `1` for "monitor blind", "no detectors", "can't see him"; `0` for everything else.
- A classifier timeout, error, or malformed answer is `Unavailable` and never counts as `back`.
- `shadow_mode` defaults to `true`; in shadow mode roll/no-view alerts are priority 0 and prefixed `[TEST]`; health alerts are unchanged.
- Secrets only come from environment variables (`.env`, mode 600, never committed): `TAPO_RTSP_URL`, `PUSHOVER_TOKEN`, `PUSHOVER_USER`, `HEALTHCHECKS_URL`, `CONTROL_TOKEN`, plus each model's `api_key_env` (e.g. `OPENROUTER_API_KEY`).
- Control page is LAN-only and every route requires the `CONTROL_TOKEN` query parameter `t`.
- All async web endpoints are `async def` so they run on the event loop thread with the monitor (no locking needed).

## Review Focus

- Pressing **Pause** while he is still in the crib, then picking him up: pause must not end before he has left the view (Task 4, `test_pause_needs_him_to_leave_view_first`).
- The service restarts mid-nap (crash, reboot, deploy): a manual **On** session and an active pause must survive the restart, including whether he has already been picked up during the pause (Task 4, `test_state_survives_restart`, `test_pause_progress_survives_restart`).
- Wrong RTSP URL or crop larger than the frame makes ffmpeg exit instantly: restarts must back off instead of spinning, and the blind alert must fire (Task 10, `test_failing_command_backs_off`; Task 7, `test_blind_after_no_frames`).
- One detector returns garbage or times out on every call while the other sees stomach: the roll alert must still fire (Task 12, `test_one_model_down_still_alerts`).
- A bug that makes every `tick()` raise: the heartbeat must go to `/fail` instead of staying green (Task 12, `test_tick_exception_fails_heartbeat`).

## File Map

```
pyproject.toml, uv.lock, .python-version, config.example.toml, README.md
crib_monitor/
  __init__.py
  config.py        typed config + secrets
  schedule.py      custody-cycle windows
  labels.py        Position / Combined enums, combine()
  arming.py        schedule + manual On/Off + Pause, persisted
  decision.py      stomach/side/no-view state machine
  classifier.py    LiteLLM vision call, parsing
  health.py        health conditions + healthchecks.io heartbeat
  notifier.py      Pushover client + Alerter (message wording, shadow mode)
  imaging.py       resize/encode helpers
  motion.py        frame-difference motion score
  capture.py       ffmpeg RTSP capture thread
  storage.py       frames, JSONL log, retention, labeled eval set
  monitor.py       orchestration (tick, run_forever), controller for web
  web.py           FastAPI control + labeling page
  evaluate.py      model accuracy report over the labeled set
  probe.py         one-shot camera + model check
  main.py          wiring, uvicorn, CLI
deploy/crib-monitor.service
tests/ (one test file per module, conftest.py, test_integration.py)
```

---

### Task 1: Project scaffold and config

**Files:**
- Create: `pyproject.toml`, `.python-version`, `config.example.toml`, `crib_monitor/__init__.py`, `crib_monitor/config.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config` (fields `schedule`, `camera`, `motion`, `classifier`, `alerts`, `arming`, `storage`, `web`, `health`), `ScheduleConfig`, `CribCrop(x,y,w,h)`, `CameraConfig(crop, stream_timeout_s, max_backoff_s)`, `MotionConfig(threshold, pixel_delta, min_check_interval_s, max_check_interval_s)`, `ModelConfig(name, model, api_base, api_key_env, timeout_s, extra_body)`, `ClassifierConfig(max_side_px, models)`, `AlertConfig(shadow_mode, emergency_retry_s, emergency_expire_s, receipt_poll_s, ack_suppress_s, health_repeat_s)`, `ArmingConfig(manual_max_hours, pause_minutes)`, `StorageConfig(data_dir, retention_days)`, `WebConfig(host, port)`, `HealthConfig(blind_after_s, model_down_after, heartbeat_interval_s)`, `Secrets(tapo_rtsp_url, pushover_token, pushover_user, healthchecks_url, control_token)`, `ConfigError`, `load_config(path) -> Config`, `load_secrets(env, cfg) -> Secrets`. Fixture `config` in `tests/conftest.py`.

- [ ] **Step 1: Create the project and install dependencies**

```bash
cd ~/Projects/crib-monitor
uv python pin 3.12
```

Write `pyproject.toml`:

```toml
[project]
name = "crib-monitor"
version = "0.1.0"
description = "Crib camera roll-over monitor"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
crib-monitor = "crib_monitor.main:run"
crib-monitor-eval = "crib_monitor.evaluate:run"
crib-monitor-probe = "crib_monitor.probe:run"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["crib_monitor"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = ["integration: needs ffmpeg and mediamtx on PATH"]
```

Then:

```bash
mkdir -p crib_monitor tests && touch crib_monitor/__init__.py
uv add fastapi uvicorn httpx litellm opencv-python-headless numpy "pydantic>=2" python-dotenv python-multipart
uv add --dev pytest pytest-asyncio
uv pip show litellm | grep Version
```

Replace the `litellm` entry in `pyproject.toml` `dependencies` with an exact pin using the version just printed, e.g. `"litellm==1.xx.y"`, then run `uv lock`.

- [ ] **Step 2: Write the example config**

`config.example.toml`:

```toml
# Copy to config.toml and edit. Secrets go in .env, not here.

[schedule]
cycle_anchor = "2026-09-25"      # a Friday that starts one of his weekends
cycle_days = 14
nights = ["fri", "sat", "sun"]   # nights within the first 7 days of the cycle that are armed
window_start = "19:00"
window_end = "07:00"             # next morning; windows may cross midnight
extra_dates = []                 # extra nights to arm, "YYYY-MM-DD" (date the window starts)
skip_dates = []                  # cycle nights to skip
timezone = "America/New_York"

[camera]
stream_timeout_s = 5
max_backoff_s = 15

[camera.crop]                    # crib rectangle in stream1 pixels; all values must be even
x = 0
y = 0
w = 1280
h = 720

[motion]
threshold = 0.02                 # fraction of pixels that changed
pixel_delta = 25                 # grey-level change that counts as "changed"
min_check_interval_s = 10
max_check_interval_s = 300

[classifier]
max_side_px = 768

[[classifier.models]]
name = "local"
model = "openai/qwen3-vl:30b"                 # LiteLLM id; "openai/" = any OpenAI-compatible server
api_base = "http://alien.lan:11434/v1"        # Ollama on the laptop
timeout_s = 30

[[classifier.models]]
name = "cloud"
model = "openrouter/anthropic/claude-haiku-4.5"
api_key_env = "OPENROUTER_API_KEY"
timeout_s = 20
extra_body = { provider = { data_collection = "deny" } }

[alerts]
shadow_mode = true
emergency_retry_s = 60
emergency_expire_s = 1800
receipt_poll_s = 15
ack_suppress_s = 180
health_repeat_s = 900

[arming]
manual_max_hours = 4
pause_minutes = 45

[storage]
data_dir = "data"
retention_days = 30

[web]
host = "192.168.1.50"            # the server's LAN address
port = 8080

[health]
blind_after_s = 120
model_down_after = 3
heartbeat_interval_s = 60
```

- [ ] **Step 3: Write the failing tests**

`tests/conftest.py`:

```python
from pathlib import Path

import pytest

from crib_monitor.config import Config, load_config

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def config() -> Config:
    return load_config(ROOT / "config.example.toml")
```

`tests/test_config.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'crib_monitor.config'`

- [ ] **Step 5: Implement `crib_monitor/config.py`**

```python
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
    max_check_interval_s: float = 300


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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version config.example.toml crib_monitor tests
git commit -m "feat: project scaffold and typed config"
```

---

### Task 2: Custody-cycle schedule

**Files:**
- Create: `crib_monitor/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: `ScheduleConfig` (Task 1).
- Produces: `Schedule(cfg)` with `is_armed_night(d: date) -> bool`, `window_for(d: date) -> tuple[datetime, datetime]`, `active_window(now: datetime) -> tuple[datetime, datetime] | None`. `now` must be timezone-aware; returned datetimes are in the schedule timezone.

- [ ] **Step 1: Write the failing tests**

`tests/test_schedule.py`:

```python
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from crib_monitor.config import ScheduleConfig
from crib_monitor.schedule import Schedule

NY = ZoneInfo("America/New_York")


def make(**overrides) -> Schedule:
    fields = dict(
        cycle_anchor=date(2026, 9, 25),  # Friday
        cycle_days=14,
        nights=["fri", "sat", "sun"],
        window_start=time(19, 0),
        window_end=time(7, 0),
        timezone="America/New_York",
    )
    fields.update(overrides)
    return Schedule(ScheduleConfig(**fields))


def at(y, mo, d, h, mi=0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=NY)


def test_cycle_nights():
    s = make()
    armed = [date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27), date(2026, 10, 9), date(2026, 9, 11)]
    unarmed = [date(2026, 9, 24), date(2026, 9, 28), date(2026, 10, 2), date(2026, 10, 3)]
    assert all(s.is_armed_night(d) for d in armed)
    assert not any(s.is_armed_night(d) for d in unarmed)


def test_window_boundaries():
    s = make()
    assert s.active_window(at(2026, 9, 25, 18, 59)) is None
    start, end = s.active_window(at(2026, 9, 25, 19, 0))
    assert start == at(2026, 9, 25, 19, 0)
    assert end == at(2026, 9, 26, 7, 0)


def test_window_crosses_midnight():
    s = make()
    assert s.active_window(at(2026, 9, 26, 3, 0)) is not None      # Friday night
    assert s.active_window(at(2026, 9, 28, 6, 59)) is not None     # Sunday night into Monday
    assert s.active_window(at(2026, 9, 28, 7, 0)) is None
    assert s.active_window(at(2026, 9, 28, 20, 0)) is None         # Monday night not armed


def test_utc_input_is_converted():
    s = make()
    assert s.active_window(datetime(2026, 9, 26, 0, 0, tzinfo=UTC)) is not None  # 20:00 EDT Fri


def test_extra_and_skip_dates():
    s = make(extra_dates=[date(2026, 10, 1)], skip_dates=[date(2026, 9, 26)])
    assert s.active_window(at(2026, 10, 1, 21, 0)) is not None
    assert s.active_window(at(2026, 9, 26, 21, 0)) is None
    assert s.active_window(at(2026, 9, 25, 21, 0)) is not None


def test_dst_end_lengthens_window():
    s = make(cycle_anchor=date(2026, 10, 30))  # Fri; Sat Oct 31 night spans DST end on Nov 1
    start, end = s.active_window(at(2026, 10, 31, 20, 0))
    assert end == at(2026, 11, 1, 7, 0)
    assert end.astimezone(UTC) - start.astimezone(UTC) == timedelta(hours=13)


def test_same_day_window():
    s = make(window_start=time(13, 0), window_end=time(15, 0))
    assert s.active_window(at(2026, 9, 25, 14, 0)) is not None
    assert s.active_window(at(2026, 9, 25, 15, 0)) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'crib_monitor.schedule'`

- [ ] **Step 3: Implement `crib_monitor/schedule.py`**

```python
"""Which nights are armed, and the wall-clock window for each."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .config import ScheduleConfig

_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class Schedule:
    def __init__(self, cfg: ScheduleConfig) -> None:
        self._cfg = cfg
        self._tz = ZoneInfo(cfg.timezone)
        wanted = {_WEEKDAYS.index(n) for n in cfg.nights}
        self._offsets = {
            i
            for i in range(min(7, cfg.cycle_days))
            if (cfg.cycle_anchor + timedelta(days=i)).weekday() in wanted
        }
        self._extra = set(cfg.extra_dates)
        self._skip = set(cfg.skip_dates)

    def is_armed_night(self, d: date) -> bool:
        """True if the window starting on date `d` is armed."""
        if d in self._extra:
            return True
        if d in self._skip:
            return False
        return (d - self._cfg.cycle_anchor).days % self._cfg.cycle_days in self._offsets

    def window_for(self, d: date) -> tuple[datetime, datetime]:
        start = datetime.combine(d, self._cfg.window_start, tzinfo=self._tz)
        crosses_midnight = self._cfg.window_end <= self._cfg.window_start
        end_day = d + timedelta(days=1) if crosses_midnight else d
        end = datetime.combine(end_day, self._cfg.window_end, tzinfo=self._tz)
        return start, end

    def active_window(self, now: datetime) -> tuple[datetime, datetime] | None:
        today = now.astimezone(self._tz).date()
        for d in (today - timedelta(days=1), today):
            if self.is_armed_night(d):
                start, end = self.window_for(d)
                if start <= now < end:
                    return start, end
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/schedule.py tests/test_schedule.py
git commit -m "feat: custody-cycle schedule windows"
```

---

### Task 3: Position labels and combining two answers

**Files:**
- Create: `crib_monitor/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Produces: `Position` (StrEnum: `BACK="back"`, `STOMACH="stomach"`, `SIDE="side"`, `UNCLEAR="unclear"`, `NOT_VISIBLE="not_visible"`), `Combined` (StrEnum: `STOMACH`, `SIDE`, `BACK`, `NO_VIEW`, `FAILED`, values lowercase), `IN_CRIB: frozenset[Combined]` = {BACK, SIDE, STOMACH}, `combine(labels: Iterable[Position | None]) -> Combined` (`None` = model unavailable).

- [ ] **Step 1: Write the failing tests**

`tests/test_labels.py`:

```python
import pytest

from crib_monitor.labels import Combined, Position, combine

B, S, SD, U, NV = Position.BACK, Position.STOMACH, Position.SIDE, Position.UNCLEAR, Position.NOT_VISIBLE


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([B, S], Combined.STOMACH),
        ([S, None], Combined.STOMACH),
        ([B, SD], Combined.SIDE),
        ([SD, S], Combined.STOMACH),
        ([B, B], Combined.BACK),
        ([B, U], Combined.BACK),
        ([B, None], Combined.BACK),
        ([U, NV], Combined.NO_VIEW),
        ([NV, None], Combined.NO_VIEW),
        ([None, None], Combined.FAILED),
        ([], Combined.FAILED),
    ],
)
def test_combine(labels, expected):
    assert combine(labels) is expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/labels.py`**

```python
"""Per-model positions and the combined reading the decision logic consumes."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Position(StrEnum):
    BACK = "back"
    STOMACH = "stomach"
    SIDE = "side"
    UNCLEAR = "unclear"
    NOT_VISIBLE = "not_visible"


class Combined(StrEnum):
    STOMACH = "stomach"
    SIDE = "side"
    BACK = "back"
    NO_VIEW = "no_view"
    FAILED = "failed"


IN_CRIB = frozenset({Combined.BACK, Combined.SIDE, Combined.STOMACH})


def combine(labels: Iterable[Position | None]) -> Combined:
    """Most alarming answer wins. `None` means that model was unavailable."""
    available = [label for label in labels if label is not None]
    if not available:
        return Combined.FAILED
    if Position.STOMACH in available:
        return Combined.STOMACH
    if Position.SIDE in available:
        return Combined.SIDE
    if Position.BACK in available:
        return Combined.BACK
    return Combined.NO_VIEW
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/labels.py tests/test_labels.py
git commit -m "feat: position labels and combine rule"
```

---

### Task 4: Arming (schedule, On/Off, Pause), persisted

**Files:**
- Create: `crib_monitor/arming.py`
- Test: `tests/test_arming.py`

**Interfaces:**
- Consumes: `Schedule.active_window` (Task 2), `ArmingConfig` (Task 1), `Combined`, `IN_CRIB` (Task 3).
- Produces: `ArmingStatus(armed: bool, paused: bool, source: "manual"|"schedule"|None, manual_until: datetime|None, paused_until: datetime|None, window_end: datetime|None)`; `Arming(schedule, cfg, state_path)` with `status(now) -> ArmingStatus`, `turn_on(now)`, `turn_off(now)`, `pause(now) -> bool`, `resume()`, `observe(result: Combined, now) -> bool` (True when a pause just ended because he was seen back in the crib).

- [ ] **Step 1: Write the failing tests**

`tests/test_arming.py`:

```python
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from crib_monitor.arming import Arming
from crib_monitor.config import ArmingConfig, ScheduleConfig
from crib_monitor.labels import Combined
from crib_monitor.schedule import Schedule

NY = ZoneInfo("America/New_York")
TUE_1PM = datetime(2026, 9, 29, 13, 0, tzinfo=NY)   # outside the schedule
FRI_8PM = datetime(2026, 9, 25, 20, 0, tzinfo=NY)   # inside the schedule


@pytest.fixture
def schedule() -> Schedule:
    return Schedule(ScheduleConfig(
        cycle_anchor=date(2026, 9, 25), cycle_days=14, nights=["fri", "sat", "sun"],
        window_start=time(19, 0), window_end=time(7, 0), timezone="America/New_York",
    ))


@pytest.fixture
def arming(schedule, tmp_path) -> Arming:
    return Arming(schedule, ArmingConfig(), tmp_path / "state.json")


def test_schedule_arms(arming):
    st = arming.status(FRI_8PM)
    assert st.armed and st.source == "schedule" and not st.paused
    assert st.window_end == datetime(2026, 9, 26, 7, 0, tzinfo=NY)
    assert not arming.status(TUE_1PM).armed


def test_manual_on_until_off(arming):
    arming.turn_on(TUE_1PM)
    st = arming.status(TUE_1PM + timedelta(hours=1))
    assert st.armed and st.source == "manual"
    assert st.manual_until == TUE_1PM + timedelta(hours=4)
    arming.turn_off(TUE_1PM + timedelta(hours=1))
    assert not arming.status(TUE_1PM + timedelta(hours=1)).armed


def test_manual_on_expires_at_cap(arming):
    arming.turn_on(TUE_1PM)
    assert arming.status(TUE_1PM + timedelta(hours=3, minutes=59)).armed
    assert not arming.status(TUE_1PM + timedelta(hours=4)).armed


def test_off_during_window_lasts_until_window_end(arming):
    arming.turn_off(FRI_8PM)
    assert not arming.status(FRI_8PM + timedelta(hours=3)).armed
    assert not arming.status(datetime(2026, 9, 26, 6, 59, tzinfo=NY)).armed
    assert arming.status(datetime(2026, 9, 26, 19, 30, tzinfo=NY)).armed   # next night


def test_on_after_off_rearms(arming):
    arming.turn_off(FRI_8PM)
    arming.turn_on(FRI_8PM + timedelta(minutes=5))
    assert arming.status(FRI_8PM + timedelta(minutes=6)).armed


def test_pause_expires(arming):
    assert arming.pause(FRI_8PM)
    st = arming.status(FRI_8PM + timedelta(minutes=44))
    assert st.armed and st.paused
    assert st.paused_until == FRI_8PM + timedelta(minutes=45)
    assert not arming.status(FRI_8PM + timedelta(minutes=45)).paused


def test_pause_ignored_when_disarmed(arming):
    assert not arming.pause(TUE_1PM)
    arming.turn_on(TUE_1PM)
    assert not arming.status(TUE_1PM).paused


def test_pause_needs_him_to_leave_view_first(arming):
    arming.pause(FRI_8PM)
    t = FRI_8PM
    for _ in range(4):                      # still in the crib while parent walks over
        t += timedelta(seconds=30)
        assert not arming.observe(Combined.BACK, t)
    assert arming.status(t).paused
    assert not arming.observe(Combined.NO_VIEW, t + timedelta(seconds=30))   # picked up
    assert not arming.observe(Combined.BACK, t + timedelta(minutes=20))       # back in crib
    assert arming.observe(Combined.BACK, t + timedelta(minutes=20, seconds=30))
    assert not arming.status(t + timedelta(minutes=21)).paused


def test_no_view_between_sightings_resets_count(arming):
    arming.pause(FRI_8PM)
    t = FRI_8PM + timedelta(minutes=1)
    arming.observe(Combined.NO_VIEW, t)
    assert not arming.observe(Combined.BACK, t + timedelta(seconds=30))
    assert not arming.observe(Combined.NO_VIEW, t + timedelta(seconds=60))
    assert not arming.observe(Combined.SIDE, t + timedelta(seconds=90))
    assert arming.observe(Combined.STOMACH, t + timedelta(seconds=120))


def test_state_survives_restart(schedule, tmp_path):
    path = tmp_path / "state.json"
    first = Arming(schedule, ArmingConfig(), path)
    first.turn_on(TUE_1PM)
    first.pause(TUE_1PM)
    second = Arming(schedule, ArmingConfig(), path)
    st = second.status(TUE_1PM + timedelta(minutes=10))
    assert st.armed and st.source == "manual" and st.paused


def test_pause_progress_survives_restart(schedule, tmp_path):
    path = tmp_path / "state.json"
    first = Arming(schedule, ArmingConfig(), path)
    first.pause(FRI_8PM)
    first.observe(Combined.NO_VIEW, FRI_8PM + timedelta(minutes=1))     # picked up, then restart
    second = Arming(schedule, ArmingConfig(), path)
    assert not second.observe(Combined.BACK, FRI_8PM + timedelta(minutes=20))
    assert second.observe(Combined.BACK, FRI_8PM + timedelta(minutes=20, seconds=30))


def test_corrupt_state_file_ignored(schedule, tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json")
    arming = Arming(schedule, ArmingConfig(), path)
    assert arming.status(FRI_8PM).armed
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_arming.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/arming.py`**

```python
"""Whether the monitor should be watching right now, and why."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from .config import ArmingConfig
from .labels import IN_CRIB, Combined
from .schedule import Schedule

log = logging.getLogger(__name__)

SIGHTINGS_TO_END_PAUSE = 2


@dataclass(frozen=True)
class ArmingStatus:
    armed: bool
    paused: bool
    source: Literal["manual", "schedule"] | None
    manual_until: datetime | None = None
    paused_until: datetime | None = None
    window_end: datetime | None = None


class Arming:
    def __init__(self, schedule: Schedule, cfg: ArmingConfig, state_path: Path) -> None:
        self._schedule = schedule
        self._cfg = cfg
        self._path = Path(state_path)
        self._manual_since: datetime | None = None
        self._off_until: datetime | None = None
        self._paused_until: datetime | None = None
        self._gone = False
        self._seen = 0
        self._load()

    # --- queries ---------------------------------------------------------

    def status(self, now: datetime) -> ArmingStatus:
        self._expire(now)
        manual_until = self._manual_until()
        window_end = None
        if manual_until is not None:
            armed, source = True, "manual"
        else:
            window = self._schedule.active_window(now)
            if window is not None and self._off_until is None:
                armed, source, window_end = True, "schedule", window[1]
            else:
                armed, source = False, None
        if not armed and self._paused_until is not None:
            self._paused_until = None
            self._save()
        paused = armed and self._paused_until is not None
        return ArmingStatus(
            armed=armed,
            paused=paused,
            source=source,
            manual_until=manual_until,
            paused_until=self._paused_until if paused else None,
            window_end=window_end,
        )

    # --- commands --------------------------------------------------------

    def turn_on(self, now: datetime) -> None:
        self._manual_since = now
        self._off_until = None
        self._save()

    def turn_off(self, now: datetime) -> None:
        self._manual_since = None
        self._paused_until = None
        window = self._schedule.active_window(now)
        self._off_until = window[1] if window else None
        self._save()

    def pause(self, now: datetime) -> bool:
        if not self.status(now).armed:
            return False
        self._paused_until = now + timedelta(minutes=self._cfg.pause_minutes)
        self._gone = False
        self._seen = 0
        self._save()
        return True

    def resume(self) -> None:
        self._paused_until = None
        self._save()

    def observe(self, result: Combined, now: datetime) -> bool:
        """Feed a combined reading while paused. True if the pause just ended."""
        if self._paused_until is None or now >= self._paused_until:
            return False
        if result is Combined.NO_VIEW:
            self._seen = 0
            if not self._gone:
                self._gone = True
                self._save()
            return False
        if result in IN_CRIB and self._gone:
            self._seen += 1
            if self._seen >= SIGHTINGS_TO_END_PAUSE:
                self._paused_until = None
                self._gone = False
                self._seen = 0
                self._save()
                return True
        return False

    # --- internals -------------------------------------------------------

    def _manual_until(self) -> datetime | None:
        if self._manual_since is None:
            return None
        return self._manual_since + timedelta(hours=self._cfg.manual_max_hours)

    def _expire(self, now: datetime) -> None:
        changed = False
        manual_until = self._manual_until()
        if manual_until is not None and now >= manual_until:
            self._manual_since = None
            changed = True
        if self._off_until is not None and now >= self._off_until:
            self._off_until = None
            changed = True
        if self._paused_until is not None and now >= self._paused_until:
            self._paused_until = None
            changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        def iso(value: datetime | None) -> str | None:
            return value.isoformat() if value else None

        data = {
            "manual_since": iso(self._manual_since),
            "off_until": iso(self._off_until),
            "paused_until": iso(self._paused_until),
            "pause_gone": self._gone,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())

            def parse(key: str) -> datetime | None:
                value = data.get(key)
                return datetime.fromisoformat(value) if value else None

            self._manual_since = parse("manual_since")
            self._off_until = parse("off_until")
            self._paused_until = parse("paused_until")
            self._gone = bool(data.get("pause_gone", False))
        except (ValueError, TypeError, AttributeError) as exc:
            log.warning("ignoring unreadable arming state %s: %s", self._path, exc)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_arming.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/arming.py tests/test_arming.py
git commit -m "feat: arming with schedule, manual on/off, pause, persisted state"
```

---

### Task 5: Decision state machine

**Files:**
- Create: `crib_monitor/decision.py`
- Test: `tests/test_decision.py`

**Interfaces:**
- Consumes: `Combined` (Task 3).
- Produces: `State` (StrEnum `MONITORING="monitoring"`, `WATCH="watch"`, `CONFIRMING="confirming"`, `ALERTED="alerted"`, `NO_VIEW="no_view"`); action dataclasses `StomachAlert()`, `NoViewAlert()`, `BackOnBack()`, `FalsePositive()`; `Engine(ack_suppress_s: float = 180)` with `state`, `force_interval_s -> float | None`, `reset()`, `step(result: Combined, now: datetime) -> list[Action]`, `on_ack(now)`, `on_expired(now)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_decision.py`:

```python
from datetime import UTC, datetime, timedelta

from crib_monitor.decision import BackOnBack, Engine, FalsePositive, NoViewAlert, State, StomachAlert
from crib_monitor.labels import Combined

T0 = datetime(2026, 9, 26, 3, 0, tzinfo=UTC)
ST, SD, BK, NV, FL = Combined.STOMACH, Combined.SIDE, Combined.BACK, Combined.NO_VIEW, Combined.FAILED


def feed(engine, results, start=T0, step_s=10):
    actions, t = [], start
    for r in results:
        actions += engine.step(r, t)
        t += timedelta(seconds=step_s)
    return actions, t


def names(actions):
    return [type(a).__name__ for a in actions]


def test_stomach_confirmed_alerts_once():
    e = Engine()
    actions, _ = feed(e, [ST, ST, ST, ST])
    assert names(actions) == ["StomachAlert"]
    assert e.state is State.ALERTED


def test_stomach_then_back_is_false_positive():
    e = Engine()
    actions, _ = feed(e, [ST, BK])
    assert names(actions) == ["FalsePositive"]
    assert e.state is State.MONITORING


def test_unconfirmed_goes_to_watch():
    e = Engine()
    actions, _ = feed(e, [ST, NV, SD, FL])
    assert actions == []
    assert e.state is State.WATCH


def test_stomach_after_unclear_confirmation_still_alerts():
    e = Engine()
    actions, _ = feed(e, [ST, SD, ST])
    assert names(actions) == ["StomachAlert"]


def test_side_watch_needs_two_consecutive_backs():
    e = Engine()
    feed(e, [SD])
    assert e.state is State.WATCH
    feed(e, [BK, SD, BK])
    assert e.state is State.WATCH
    feed(e, [BK])
    assert e.state is State.MONITORING


def test_no_view_alert_needs_three_checks_over_a_minute():
    e = Engine()
    actions, _ = feed(e, [NV, NV, NV], step_s=5)
    assert actions == []
    e = Engine()
    actions, _ = feed(e, [NV, NV, NV, NV, NV, NV], step_s=20)
    assert names(actions) == ["NoViewAlert"]


def test_no_view_recovers():
    e = Engine()
    feed(e, [NV, NV])
    feed(e, [BK])
    assert e.state is State.MONITORING
    feed(e, [NV, SD])
    assert e.state is State.WATCH


def test_failed_is_ignored_while_monitoring():
    e = Engine()
    actions, _ = feed(e, [FL, FL, FL])
    assert actions == [] and e.state is State.MONITORING


def test_no_repeat_alert_until_ack_and_suppression():
    e = Engine(ack_suppress_s=180)
    _, t = feed(e, [ST, ST])
    actions, t = feed(e, [ST, ST, ST], start=t)
    assert actions == []                                  # waiting for ack; Pushover repeats itself
    e.on_ack(t)
    actions, _ = feed(e, [ST], start=t + timedelta(seconds=179))
    assert actions == []
    actions, _ = feed(e, [ST], start=t + timedelta(seconds=180))
    assert names(actions) == ["StomachAlert"]


def test_expired_alert_realerts_on_next_stomach():
    e = Engine()
    _, t = feed(e, [ST, ST])
    e.on_expired(t)
    actions, _ = feed(e, [ST], start=t)
    assert names(actions) == ["StomachAlert"]


def test_two_backs_after_alert_send_back_on_back():
    e = Engine()
    _, t = feed(e, [ST, ST])
    actions, _ = feed(e, [BK, FL, BK], start=t)
    assert names(actions) == ["BackOnBack"]
    assert e.state is State.MONITORING


def test_force_intervals():
    e = Engine()
    assert e.force_interval_s is None
    e.step(ST, T0)
    assert e.force_interval_s == 10
    e.reset()
    e.step(SD, T0)
    assert e.force_interval_s == 30
    e.reset()
    e.step(NV, T0)
    assert e.force_interval_s == 20
    e.reset()
    feed(e, [ST, ST])
    assert e.force_interval_s == 30
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_decision.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/decision.py`**

```python
"""Turns a stream of combined readings into alerts. Pure: time is an argument."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from .labels import Combined

CONFIRM_INTERVAL_S = 10
WATCH_INTERVAL_S = 30
NO_VIEW_INTERVAL_S = 20
ALERTED_INTERVAL_S = 30
CONFIRM_ATTEMPTS = 3
BACK_STREAK = 2
NO_VIEW_STREAK = 3
NO_VIEW_MIN_SPAN_S = 60


class State(StrEnum):
    MONITORING = "monitoring"
    WATCH = "watch"
    CONFIRMING = "confirming"
    ALERTED = "alerted"
    NO_VIEW = "no_view"


@dataclass(frozen=True)
class StomachAlert:
    pass


@dataclass(frozen=True)
class NoViewAlert:
    pass


@dataclass(frozen=True)
class BackOnBack:
    pass


@dataclass(frozen=True)
class FalsePositive:
    pass


Action = StomachAlert | NoViewAlert | BackOnBack | FalsePositive

_FORCED = {
    State.CONFIRMING: CONFIRM_INTERVAL_S,
    State.WATCH: WATCH_INTERVAL_S,
    State.NO_VIEW: NO_VIEW_INTERVAL_S,
    State.ALERTED: ALERTED_INTERVAL_S,
}


class Engine:
    def __init__(self, ack_suppress_s: float = 180) -> None:
        self._ack_suppress = timedelta(seconds=ack_suppress_s)
        self.reset()

    def reset(self) -> None:
        self.state = State.MONITORING
        self._back = 0
        self._confirms = 0
        self._nv_count = 0
        self._nv_first: datetime | None = None
        self._nv_alerted = False
        self._awaiting_ack = False
        self._suppress_until: datetime | None = None

    @property
    def force_interval_s(self) -> float | None:
        return _FORCED.get(self.state)

    def on_ack(self, now: datetime) -> None:
        if self.state is State.ALERTED:
            self._awaiting_ack = False
            self._suppress_until = now + self._ack_suppress

    def on_expired(self, now: datetime) -> None:
        if self.state is State.ALERTED:
            self._awaiting_ack = False
            self._suppress_until = now

    def step(self, result: Combined, now: datetime) -> list[Action]:
        if self.state is State.CONFIRMING:
            return self._confirming(result)
        if self.state is State.ALERTED:
            return self._alerted(result, now)
        if result is Combined.FAILED:
            return []
        if result is Combined.STOMACH:
            self._set(State.CONFIRMING)
            return []
        if result is Combined.NO_VIEW:
            return self._no_view(now)
        if result is Combined.SIDE:
            self._set(State.WATCH)
            return []
        # BACK
        if self.state is State.WATCH:
            self._back += 1
            if self._back >= BACK_STREAK:
                self._set(State.MONITORING)
            return []
        self._set(State.MONITORING)
        return []

    def _set(self, state: State) -> None:
        self.state = state
        self._back = 0
        self._confirms = 0
        self._nv_count = 0
        self._nv_first = None
        self._nv_alerted = False

    def _confirming(self, result: Combined) -> list[Action]:
        if result is Combined.STOMACH:
            self._set(State.ALERTED)
            self._awaiting_ack = True
            self._suppress_until = None
            return [StomachAlert()]
        if result is Combined.BACK:
            self._set(State.MONITORING)
            return [FalsePositive()]
        self._confirms += 1
        if self._confirms >= CONFIRM_ATTEMPTS:
            self._set(State.WATCH)
        return []

    def _alerted(self, result: Combined, now: datetime) -> list[Action]:
        if result is Combined.FAILED:
            return []
        if result is Combined.BACK:
            self._back += 1
            if self._back >= BACK_STREAK:
                self._set(State.MONITORING)
                self._awaiting_ack = False
                self._suppress_until = None
                return [BackOnBack()]
            return []
        self._back = 0
        if (
            result is Combined.STOMACH
            and not self._awaiting_ack
            and self._suppress_until is not None
            and now >= self._suppress_until
        ):
            self._awaiting_ack = True
            self._suppress_until = None
            return [StomachAlert()]
        return []

    def _no_view(self, now: datetime) -> list[Action]:
        if self.state is not State.NO_VIEW:
            self._set(State.NO_VIEW)
            self._nv_first = now
            self._nv_count = 1
            return []
        self._nv_count += 1
        assert self._nv_first is not None
        span = (now - self._nv_first).total_seconds()
        if not self._nv_alerted and self._nv_count >= NO_VIEW_STREAK and span >= NO_VIEW_MIN_SPAN_S:
            self._nv_alerted = True
            return [NoViewAlert()]
        return []
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_decision.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/decision.py tests/test_decision.py
git commit -m "feat: decision state machine"
```

---

### Task 6: Vision classifier through LiteLLM

**Files:**
- Create: `crib_monitor/classifier.py`
- Test: `tests/test_classifier.py`

**Interfaces:**
- Consumes: `ModelConfig`, `ClassifierConfig` (Task 1), `Position` (Task 3).
- Produces: `PROMPT: str`, `RESPONSE_FORMAT: dict`, `ClassifyResult(model_name: str, position: Position | None, latency_s: float, error: str | None)`, `parse_position(content: str | None) -> Position` (raises `ValueError`), `Classifier(cfg: ModelConfig, api_key: str | None, completion=None)` with `.name` and `async classify(jpeg: bytes) -> ClassifyResult` (never raises), `build_classifiers(cfg: ClassifierConfig, env: Mapping[str, str]) -> list[Classifier]`, `async classify_all(classifiers, jpeg) -> list[ClassifyResult]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_classifier.py`:

```python
import asyncio
import base64
from types import SimpleNamespace

import pytest

from crib_monitor.classifier import Classifier, build_classifiers, classify_all, parse_position
from crib_monitor.config import ModelConfig
from crib_monitor.labels import Position


def reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeCompletion:
    def __init__(self, content=None, exc=None, delay=0.0):
        self.content, self.exc, self.delay, self.calls = content, exc, delay, []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return reply(self.content)


CLOUD = ModelConfig(
    name="cloud", model="openrouter/vendor/model", api_key_env="OPENROUTER_API_KEY",
    timeout_s=1, extra_body={"provider": {"data_collection": "deny"}},
)
LOCAL = ModelConfig(name="local", model="openai/qwen", api_base="http://alien:11434/v1", timeout_s=1)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"position": "stomach"}', Position.STOMACH),
        ('```json\n{"position": "back"}\n```', Position.BACK),
        ('  {"position": "not_visible"} ', Position.NOT_VISIBLE),
    ],
)
def test_parse_position(content, expected):
    assert parse_position(content) is expected


@pytest.mark.parametrize("content", [None, "", "stomach", '{"pos": "back"}', '{"position": "upside_down"}', "[1]"])
def test_parse_position_rejects(content):
    with pytest.raises(ValueError):
        parse_position(content)


async def test_classify_sends_image_and_options():
    fake = FakeCompletion('{"position": "side"}')
    result = await Classifier(CLOUD, "sk-or", completion=fake).classify(b"\xff\xd8jpeg")
    assert result.position is Position.SIDE and result.error is None and result.model_name == "cloud"
    call = fake.calls[0]
    assert call["model"] == "openrouter/vendor/model"
    assert call["api_key"] == "sk-or"
    assert call["extra_body"] == {"provider": {"data_collection": "deny"}}
    assert call["response_format"]["type"] == "json_schema"
    image = call["messages"][0]["content"][1]["image_url"]["url"]
    assert image == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()


async def test_local_uses_api_base():
    fake = FakeCompletion('{"position": "back"}')
    await Classifier(LOCAL, None, completion=fake).classify(b"x")
    assert fake.calls[0]["api_base"] == "http://alien:11434/v1"
    assert fake.calls[0]["api_key"] == "none"
    assert "extra_body" not in fake.calls[0]


async def test_malformed_answer_is_unavailable():
    result = await Classifier(LOCAL, None, completion=FakeCompletion("I think back")).classify(b"x")
    assert result.position is None and result.error.startswith("JSONDecodeError")


async def test_exception_is_unavailable():
    result = await Classifier(LOCAL, None, completion=FakeCompletion(exc=RuntimeError("boom"))).classify(b"x")
    assert result.position is None and "boom" in result.error


async def test_timeout_is_unavailable():
    cfg = LOCAL.model_copy(update={"timeout_s": 0.05})
    result = await Classifier(cfg, None, completion=FakeCompletion('{"position": "back"}', delay=1)).classify(b"x")
    assert result.position is None and "Timeout" in result.error


async def test_classify_all_runs_concurrently():
    slow = [Classifier(m, None, completion=FakeCompletion('{"position": "back"}', delay=0.2)) for m in (LOCAL, CLOUD)]
    loop = asyncio.get_running_loop()
    start = loop.time()
    results = await classify_all(slow, b"x")
    assert loop.time() - start < 0.35
    assert [r.model_name for r in results] == ["local", "cloud"]


def test_build_classifiers_reads_keys(config):
    classifiers = build_classifiers(config.classifier, {"OPENROUTER_API_KEY": "sk-or"})
    assert [c.name for c in classifiers] == ["local", "cloud"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/classifier.py`**

```python
"""Ask a vision model how the baby is lying."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .config import ClassifierConfig, ModelConfig
from .labels import Position

PROMPT = """You are checking an image from a camera above a baby's crib to see how the baby is lying.
The image may be grayscale night vision (infrared). There are no blankets in the crib.

Classify the baby's position:
- "back": lying face up, chest and face toward the ceiling.
- "stomach": lying face down, chest against the mattress, back of the head or back toward the ceiling.
- "side": lying on either side.
- "unclear": a baby is in the crib but you cannot tell the position.
- "not_visible": no baby is visible in the crib.

If you are unsure between "stomach" and another position, answer "stomach".

Respond with JSON only: {"position": "<one of the values above>"}"""

RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "crib_position",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"position": {"type": "string", "enum": [p.value for p in Position]}},
            "required": ["position"],
            "additionalProperties": False,
        },
    },
}

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)

Completion = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ClassifyResult:
    model_name: str
    position: Position | None
    latency_s: float
    error: str | None = None


def parse_position(content: str | None) -> Position:
    if not content:
        raise ValueError("empty response")
    text = content.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    data = json.loads(text)
    if not isinstance(data, dict) or "position" not in data:
        raise ValueError(f"no position in {text[:80]!r}")
    return Position(data["position"])


async def _litellm_completion(**kwargs: Any) -> Any:
    import litellm  # imported lazily: slow to import, and tests inject a fake

    return await litellm.acompletion(**kwargs)


class Classifier:
    def __init__(self, cfg: ModelConfig, api_key: str | None, completion: Completion | None = None) -> None:
        self._cfg = cfg
        self._api_key = api_key
        self._completion = completion or _litellm_completion
        self.name = cfg.name

    async def classify(self, jpeg: bytes) -> ClassifyResult:
        start = time.monotonic()
        try:
            response = await asyncio.wait_for(self._completion(**self._request(jpeg)), timeout=self._cfg.timeout_s)
            position = parse_position(response.choices[0].message.content)
        except Exception as exc:  # any failure means "unavailable", never "back"
            return ClassifyResult(self.name, None, time.monotonic() - start, f"{type(exc).__name__}: {exc}")
        return ClassifyResult(self.name, position, time.monotonic() - start)

    def _request(self, jpeg: bytes) -> dict[str, Any]:
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "response_format": RESPONSE_FORMAT,
            "temperature": 0,
            "max_tokens": 300,
            "timeout": self._cfg.timeout_s,
            "drop_params": True,
            # OpenAI-compatible local servers still require a non-empty key.
            "api_key": self._api_key or "none",
        }
        if self._cfg.api_base:
            kwargs["api_base"] = self._cfg.api_base
        if self._cfg.extra_body:
            kwargs["extra_body"] = self._cfg.extra_body
        return kwargs


def build_classifiers(cfg: ClassifierConfig, env: Mapping[str, str]) -> list[Classifier]:
    return [Classifier(m, env.get(m.api_key_env) if m.api_key_env else None) for m in cfg.models]


async def classify_all(classifiers: Sequence[Any], jpeg: bytes) -> list[ClassifyResult]:
    return list(await asyncio.gather(*(c.classify(jpeg) for c in classifiers)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/classifier.py tests/test_classifier.py
git commit -m "feat: LiteLLM vision classifier"
```

---

### Task 7: Health conditions and heartbeat

**Files:**
- Create: `crib_monitor/health.py`
- Test: `tests/test_health.py`

**Interfaces:**
- Consumes: `HealthConfig` (Task 1), `ClassifyResult` (Task 6).
- Produces: `HealthEvent(key: str, priority: int, message: str, recovered: bool = False)`; `Health(cfg, model_names: list[str], repeat_s: float)` with `reset(now: datetime | None)`, `frame(now)`, `results(results: list[ClassifyResult], now) -> list[HealthEvent]`, `tick(now) -> list[HealthEvent]`, `active -> list[str]`; `Heartbeat(url: str, client: httpx.AsyncClient)` with `async ping(ok: bool = True) -> None` (never raises). Keys: `"blind"`, `"no_detectors"`, `"model_down:<name>"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_health.py`:

```python
from datetime import UTC, datetime, timedelta

import httpx

from crib_monitor.classifier import ClassifyResult
from crib_monitor.config import HealthConfig
from crib_monitor.health import Health, Heartbeat
from crib_monitor.labels import Position

T0 = datetime(2026, 9, 26, 3, 0, tzinfo=UTC)


def make() -> Health:
    h = Health(HealthConfig(), ["local", "cloud"], repeat_s=900)
    h.reset(T0)
    return h


def res(local_ok: bool, cloud_ok: bool):
    return [
        ClassifyResult("local", Position.BACK if local_ok else None, 0.1),
        ClassifyResult("cloud", Position.BACK if cloud_ok else None, 0.1),
    ]


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def test_blind_after_no_frames():
    h = make()
    assert h.tick(at(119)) == []
    events = h.tick(at(120))
    assert [(e.key, e.priority, e.recovered) for e in events] == [("blind", 1, False)]
    assert h.tick(at(500)) == []
    assert [e.key for e in h.tick(at(1020))] == ["blind"]          # repeats after 900 s
    h.frame(at(1030))
    events = h.tick(at(1030))
    assert [(e.key, e.priority, e.recovered) for e in events] == [("blind", 0, True)]
    assert h.active == []


def test_frames_keep_it_quiet():
    h = make()
    for s in range(0, 600, 30):
        h.frame(at(s))
        assert h.tick(at(s)) == []


def test_one_model_down_and_recovered():
    h = make()
    assert h.results(res(False, True), at(0)) == []
    assert h.results(res(False, True), at(10)) == []
    events = h.results(res(False, True), at(20))
    assert [(e.key, e.priority) for e in events] == [("model_down:local", 0)]
    assert h.results(res(False, True), at(30)) == []
    events = h.results(res(True, True), at(40))
    assert [(e.key, e.recovered) for e in events] == [("model_down:local", True)]


def test_both_down_is_high_priority_and_repeats():
    h = make()
    events = []
    for s in (0, 10, 20):
        events += h.results(res(False, False), at(s))
    assert [(e.key, e.priority) for e in events] == [("no_detectors", 1)]
    h.frame(at(900))
    assert [e.key for e in h.tick(at(920))] == ["no_detectors"]
    events = h.results(res(True, False), at(930))
    assert [(e.key, e.recovered) for e in events] == [("no_detectors", True), ("model_down:cloud", False)]


def test_escalation_from_one_to_both_does_not_claim_recovery():
    h = make()
    for s in (0, 10, 20):
        h.results(res(False, True), at(s))
    events = []
    for s in (30, 40, 50):
        events += h.results(res(False, False), at(s))
    assert [(e.key, e.recovered) for e in events] == [("no_detectors", False)]


def test_reset_clears_everything():
    h = make()
    h.tick(at(200))
    h.reset(at(300))
    assert h.active == []
    assert h.tick(at(310)) == []


async def test_heartbeat_pings_and_fails():
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: seen.append(str(r.url)) or httpx.Response(200)))
    hb = Heartbeat("https://hc-ping.com/abc", client)
    await hb.ping()
    await hb.ping(ok=False)
    assert seen == ["https://hc-ping.com/abc", "https://hc-ping.com/abc/fail"]


async def test_heartbeat_swallows_network_errors():
    def boom(request):
        raise httpx.ConnectError("down")

    hb = Heartbeat("https://hc-ping.com/abc", httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    await hb.ping()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/health.py`**

```python
"""Health conditions that mean the monitor is not really watching."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import httpx

from .classifier import ClassifyResult
from .config import HealthConfig

log = logging.getLogger(__name__)

_REPEATING = ("blind", "no_detectors")


@dataclass(frozen=True)
class HealthEvent:
    key: str
    priority: int
    message: str
    recovered: bool = False


class Health:
    def __init__(self, cfg: HealthConfig, model_names: Sequence[str], repeat_s: float) -> None:
        self._cfg = cfg
        self._names = list(model_names)
        self._repeat_s = repeat_s
        self.reset(None)

    def reset(self, now: datetime | None) -> None:
        self._since = now
        self._last_frame: datetime | None = None
        self._streak = {name: 0 for name in self._names}
        self._active: dict[str, datetime] = {}

    @property
    def active(self) -> list[str]:
        return sorted(self._active)

    def frame(self, now: datetime) -> None:
        self._last_frame = now

    def results(self, results: Sequence[ClassifyResult], now: datetime) -> list[HealthEvent]:
        for r in results:
            self._streak[r.model_name] = 0 if r.position is not None else self._streak.get(r.model_name, 0) + 1
        down = {name for name, n in self._streak.items() if n >= self._cfg.model_down_after}
        events: list[HealthEvent] = []
        if down and down == set(self._names):
            for key in [k for k in self._active if k.startswith("model_down:")]:
                del self._active[key]
            if "no_detectors" not in self._active:
                self._active["no_detectors"] = now
                events.append(self._event("no_detectors"))
            return events
        if "no_detectors" in self._active:
            del self._active["no_detectors"]
            events.append(self._event("no_detectors", recovered=True))
        for name in self._names:
            key = f"model_down:{name}"
            if name in down and key not in self._active:
                self._active[key] = now
                events.append(self._event(key))
            elif name not in down and key in self._active:
                del self._active[key]
                events.append(self._event(key, recovered=True))
        return events

    def tick(self, now: datetime) -> list[HealthEvent]:
        baseline = self._last_frame or self._since
        if baseline is None:
            return []
        events: list[HealthEvent] = []
        blind = (now - baseline).total_seconds() >= self._cfg.blind_after_s
        if blind and "blind" not in self._active:
            self._active["blind"] = now
            events.append(self._event("blind"))
        elif not blind and "blind" in self._active:
            del self._active["blind"]
            events.append(self._event("blind", recovered=True))
        for key in _REPEATING:
            last = self._active.get(key)
            if last is not None and (now - last).total_seconds() >= self._repeat_s:
                self._active[key] = now
                events.append(self._event(key))
        return events

    def _event(self, key: str, recovered: bool = False) -> HealthEvent:
        if key == "blind":
            minutes = round(self._cfg.blind_after_s / 60)
            message = "Camera recovered." if recovered else f"Monitor blind: no camera frames for {minutes} min."
        elif key == "no_detectors":
            message = "Detectors recovered." if recovered else "No detectors responding. The monitor cannot tell his position."
        else:
            name = key.split(":", 1)[1]
            message = f"Detector '{name}' recovered." if recovered else f"Running on one detector: '{name}' is not responding."
        priority = 0 if recovered or key.startswith("model_down:") else 1
        return HealthEvent(key, priority, message, recovered)


class Heartbeat:
    """healthchecks.io dead-man's switch."""

    def __init__(self, url: str, client: httpx.AsyncClient) -> None:
        self._url = url.rstrip("/")
        self._client = client

    async def ping(self, ok: bool = True) -> None:
        url = self._url if ok else self._url + "/fail"
        try:
            await self._client.get(url, timeout=10)
        except httpx.HTTPError as exc:
            log.warning("heartbeat ping failed: %s", exc)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_health.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/health.py tests/test_health.py
git commit -m "feat: health conditions and healthchecks heartbeat"
```

---

### Task 8: Pushover client and alert wording

**Files:**
- Create: `crib_monitor/notifier.py`
- Test: `tests/test_notifier.py`

**Interfaces:**
- Consumes: `AlertConfig` (Task 1), `HealthEvent` (Task 7).
- Produces: `NotifyError`, `ReceiptStatus(acknowledged: bool, expired: bool)`, `Pushover(token, user, client, backoff=(1, 2, 4), sleep=asyncio.sleep)` with `async send(message, *, title="Crib monitor", priority=0, image=None, retry=None, expire=None) -> str | None` (receipt for priority 2), `async receipt(receipt) -> ReceiptStatus`, `async cancel(receipt)`; `Alerter(pushover, cfg)` with async `stomach(image) -> str | None`, `no_view(image)`, `back_on_back()`, `health(event)`, `monitoring_started(camera_ok: bool, models: dict[str, bool | None], image: bytes | None)`, `manual_ended()`, `test() -> str | None`, `receipt(r)`, `cancel(r)`. All raise `NotifyError` on failure.

- [ ] **Step 1: Write the failing tests**

`tests/test_notifier.py`:

```python
import re
from urllib.parse import parse_qs

import httpx
import pytest

from crib_monitor.config import AlertConfig
from crib_monitor.health import HealthEvent
from crib_monitor.notifier import Alerter, NotifyError, Pushover


def fields(request: httpx.Request) -> dict[str, str]:
    body = request.read()
    if request.headers["content-type"].startswith("multipart/form-data"):
        return {
            m.group(1).decode(): m.group(2).decode()
            for m in re.finditer(rb'name="([^"]+)"\r\n\r\n(.*?)\r\n--', body, re.S)
        }
    return {k: v[0] for k, v in parse_qs(body.decode()).items()}


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else httpx.Response(200, json={"status": 1})
        if isinstance(response, Exception):
            raise response
        return response


def pushover(recorder) -> Pushover:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return Pushover("tok", "usr", client, backoff=(0, 0, 0))


async def test_send_normal():
    rec = Recorder()
    assert await pushover(rec).send("hello", priority=0) is None
    sent = fields(rec.requests[0])
    assert sent["token"] == "tok" and sent["user"] == "usr" and sent["message"] == "hello" and sent["priority"] == "0"
    assert "retry" not in sent


async def test_send_emergency_with_image_returns_receipt():
    rec = Recorder(httpx.Response(200, json={"status": 1, "receipt": "R1"}))
    receipt = await pushover(rec).send("roll", priority=2, image=b"\xff\xd8x", retry=60, expire=1800)
    assert receipt == "R1"
    sent = fields(rec.requests[0])
    assert sent["priority"] == "2" and sent["retry"] == "60" and sent["expire"] == "1800"
    assert b'filename="frame.jpg"' in rec.requests[0].content


async def test_send_retries_server_errors():
    rec = Recorder(httpx.Response(500), httpx.ConnectError("x"), httpx.Response(200, json={"status": 1}))
    await pushover(rec).send("hello")
    assert len(rec.requests) == 3


async def test_send_gives_up():
    rec = Recorder(*[httpx.Response(503)] * 4)
    with pytest.raises(NotifyError):
        await pushover(rec).send("hello")
    assert len(rec.requests) == 4


async def test_client_error_not_retried():
    rec = Recorder(httpx.Response(400, json={"status": 0, "errors": ["user invalid"]}))
    with pytest.raises(NotifyError, match="400"):
        await pushover(rec).send("hello")
    assert len(rec.requests) == 1


async def test_receipt_and_cancel():
    rec = Recorder(httpx.Response(200, json={"status": 1, "acknowledged": 1, "expired": 0}), httpx.Response(200, json={"status": 1}))
    p = pushover(rec)
    status = await p.receipt("R1")
    assert status.acknowledged and not status.expired
    assert "receipts/R1.json" in str(rec.requests[0].url)
    await p.cancel("R1")
    assert str(rec.requests[1].url).endswith("receipts/R1/cancel.json")


async def test_receipt_failure_raises():
    rec = Recorder(httpx.Response(500))
    with pytest.raises(NotifyError):
        await pushover(rec).receipt("R1")


async def test_alerter_live_mode():
    rec = Recorder(httpx.Response(200, json={"status": 1, "receipt": "R9"}))
    alerter = Alerter(pushover(rec), AlertConfig(shadow_mode=False))
    assert await alerter.stomach(b"img") == "R9"
    await alerter.no_view(b"img")
    sent = [fields(r) for r in rec.requests]
    assert sent[0]["priority"] == "2" and "stomach" in sent[0]["message"]
    assert sent[1]["priority"] == "1"


async def test_alerter_shadow_mode():
    rec = Recorder()
    alerter = Alerter(pushover(rec), AlertConfig(shadow_mode=True))
    assert await alerter.stomach(b"img") is None
    await alerter.no_view(b"img")
    await alerter.health(HealthEvent("blind", 1, "Monitor blind"))
    sent = [fields(r) for r in rec.requests]
    assert sent[0]["priority"] == "0" and sent[0]["message"].startswith("[TEST]")
    assert sent[1]["priority"] == "0" and sent[1]["message"].startswith("[TEST]")
    assert sent[2]["priority"] == "1" and sent[2]["message"] == "Monitor blind"


async def test_monitoring_started_marks():
    rec = Recorder()
    alerter = Alerter(pushover(rec), AlertConfig(shadow_mode=False))
    await alerter.monitoring_started(True, {"local": True, "cloud": False}, None)
    assert fields(rec.requests[0])["message"] == "Monitoring: camera ✓ local ✓ cloud ✗"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_notifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/notifier.py`**

```python
"""Pushover delivery and the wording of every message the monitor sends."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import httpx

from .config import AlertConfig
from .health import HealthEvent

log = logging.getLogger(__name__)

MESSAGES_URL = "https://api.pushover.net/1/messages.json"
RECEIPT_URL = "https://api.pushover.net/1/receipts/{receipt}.json"
CANCEL_URL = "https://api.pushover.net/1/receipts/{receipt}/cancel.json"


class NotifyError(Exception):
    pass


@dataclass(frozen=True)
class ReceiptStatus:
    acknowledged: bool
    expired: bool


class Pushover:
    def __init__(
        self,
        token: str,
        user: str,
        client: httpx.AsyncClient,
        backoff: Sequence[float] = (1, 2, 4),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._token = token
        self._user = user
        self._client = client
        self._backoff = list(backoff)
        self._sleep = sleep

    async def send(
        self,
        message: str,
        *,
        title: str = "Crib monitor",
        priority: int = 0,
        image: bytes | None = None,
        retry: int | None = None,
        expire: int | None = None,
    ) -> str | None:
        data = {"token": self._token, "user": self._user, "message": message, "title": title, "priority": str(priority)}
        if priority == 2:
            data["retry"] = str(retry)
            data["expire"] = str(expire)
        files = {"attachment": ("frame.jpg", image, "image/jpeg")} if image else None
        last_error = "not attempted"
        for delay in [None, *self._backoff]:
            if delay is not None:
                await self._sleep(delay)
            try:
                response = await self._client.post(MESSAGES_URL, data=data, files=files, timeout=15)
            except httpx.HTTPError as exc:
                last_error = repr(exc)
                continue
            if 400 <= response.status_code < 500:
                raise NotifyError(f"Pushover rejected message: {response.status_code} {response.text}")
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if body.get("status") == 1:
                    return body.get("receipt")
            last_error = f"{response.status_code} {response.text}"
        raise NotifyError(f"Pushover send failed: {last_error}")

    async def receipt(self, receipt: str) -> ReceiptStatus:
        try:
            response = await self._client.get(RECEIPT_URL.format(receipt=receipt), params={"token": self._token}, timeout=15)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise NotifyError(f"receipt poll failed: {exc!r}") from exc
        return ReceiptStatus(acknowledged=body.get("acknowledged") == 1, expired=body.get("expired") == 1)

    async def cancel(self, receipt: str) -> None:
        try:
            response = await self._client.post(CANCEL_URL.format(receipt=receipt), data={"token": self._token}, timeout=15)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise NotifyError(f"receipt cancel failed: {exc!r}") from exc


class Alerter:
    def __init__(self, pushover: Pushover, cfg: AlertConfig) -> None:
        self._p = pushover
        self._cfg = cfg

    def _tag(self, text: str) -> str:
        return f"[TEST] {text}" if self._cfg.shadow_mode else text

    async def stomach(self, image: bytes) -> str | None:
        text = "He may be on his stomach. Check the camera."
        if self._cfg.shadow_mode:
            await self._p.send(self._tag(text), priority=0, image=image)
            return None
        return await self._p.send(
            text, priority=2, image=image, retry=self._cfg.emergency_retry_s, expire=self._cfg.emergency_expire_s
        )

    async def no_view(self, image: bytes) -> None:
        await self._p.send(self._tag("Can't see him in the crib."), priority=0 if self._cfg.shadow_mode else 1, image=image)

    async def back_on_back(self) -> None:
        await self._p.send(self._tag("Back on his back."), priority=0)

    async def health(self, event: HealthEvent) -> None:
        await self._p.send(event.message, priority=event.priority)

    async def monitoring_started(self, camera_ok: bool, models: dict[str, bool | None], image: bytes | None) -> None:
        def mark(ok: bool | None) -> str:
            return "?" if ok is None else ("✓" if ok else "✗")

        parts = [f"camera {mark(camera_ok)}"] + [f"{name} {mark(ok)}" for name, ok in models.items()]
        text = "Monitoring: " + " ".join(parts) + (" (shadow mode)" if self._cfg.shadow_mode else "")
        await self._p.send(text, priority=0, image=image)

    async def manual_ended(self) -> None:
        await self._p.send("Nap monitoring ended at the time limit. Press On to keep watching.", priority=0)

    async def test(self) -> str | None:
        return await self._p.send("TEST alert from the crib monitor. Acknowledge to stop it.", priority=2, retry=60, expire=180)

    async def receipt(self, receipt: str) -> ReceiptStatus:
        return await self._p.receipt(receipt)

    async def cancel(self, receipt: str) -> None:
        await self._p.cancel(receipt)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_notifier.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/notifier.py tests/test_notifier.py
git commit -m "feat: Pushover client and alert wording"
```

---

### Task 9: Image helpers and motion score

**Files:**
- Create: `crib_monitor/imaging.py`, `crib_monitor/motion.py`
- Test: `tests/test_motion.py`

**Interfaces:**
- Produces: `resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray`, `encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes`; `MotionDetector(pixel_delta: int = 25, width: int = 320)` with `score(image_bgr: np.ndarray) -> float` (fraction 0–1 of pixels changed vs the previous call; first call returns 0.0).

- [ ] **Step 1: Write the failing tests**

`tests/test_motion.py`:

```python
import cv2
import numpy as np

from crib_monitor.imaging import encode_jpeg, resize_max_side
from crib_monitor.motion import MotionDetector


def frame(value=0):
    return np.full((360, 640, 3), value, dtype=np.uint8)


def test_resize_max_side():
    assert resize_max_side(frame(), 320).shape == (180, 320, 3)
    assert resize_max_side(frame(), 1000).shape == (360, 640, 3)


def test_encode_jpeg_roundtrip():
    data = encode_jpeg(frame(128))
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)


def test_first_frame_scores_zero():
    assert MotionDetector().score(frame(100)) == 0.0


def test_identical_frames_score_zero():
    m = MotionDetector()
    m.score(frame(100))
    assert m.score(frame(100)) == 0.0


def test_sensor_noise_is_ignored():
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (360, 640, 3)).astype(np.int16)
    noisy = np.clip(base + rng.integers(-5, 6, base.shape), 0, 255).astype(np.uint8)
    m = MotionDetector()
    m.score(base.astype(np.uint8))
    assert m.score(noisy) < 0.02


def test_moving_region_is_detected():
    m = MotionDetector()
    m.score(frame(0))
    moved = frame(0)
    moved[100:200, 100:200] = 200
    assert m.score(moved) > 0.02


def test_light_change_scores_high():
    m = MotionDetector()
    m.score(frame(20))
    assert m.score(frame(220)) > 0.9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_motion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/imaging.py` and `crib_monitor/motion.py`**

`crib_monitor/imaging.py`:

```python
"""Small OpenCV helpers shared by the monitor and tools."""

from __future__ import annotations

import cv2
import numpy as np


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
```

`crib_monitor/motion.py`:

```python
"""Cheap local motion score used to decide when to ask the models."""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, pixel_delta: int = 25, width: int = 320) -> None:
        self._delta = pixel_delta
        self._width = width
        self._prev: np.ndarray | None = None

    def score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self._width:
            gray = cv2.resize(gray, (self._width, max(1, round(h * self._width / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        diff = cv2.absdiff(prev, gray)
        return float(np.count_nonzero(diff > self._delta)) / diff.size
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_motion.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/imaging.py crib_monitor/motion.py tests/test_motion.py
git commit -m "feat: image helpers and motion score"
```

---

### Task 10: RTSP capture thread

**Files:**
- Create: `crib_monitor/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `CribCrop` (Task 1).
- Produces: `Frame(image: np.ndarray, seq: int)`; `ffmpeg_command(rtsp_url: str, crop: CribCrop, fps: int = 1) -> list[str]`; `Capture(cmd: list[str], width: int, height: int, *, stall_s=5.0, startup_s=15.0, initial_backoff_s=1.0, max_backoff_s=15.0)` with `start()`, `stop()`, `latest() -> Frame | None`, `restarts: int`. The command must write raw `bgr24` frames of exactly `width*height*3` bytes to stdout.

- [ ] **Step 1: Write the failing tests**

`tests/test_capture.py`:

```python
import sys
import time

from crib_monitor.capture import Capture, ffmpeg_command
from crib_monitor.config import CribCrop

PRODUCER = """
import sys, time
w, h, count, value = (int(a) for a in sys.argv[1:5])
sleep_after = float(sys.argv[5])
frame = bytes([value]) * (w * h * 3)
for _ in range(count):
    sys.stdout.buffer.write(frame)
    sys.stdout.buffer.flush()
    time.sleep(0.05)
time.sleep(sleep_after)
"""


def producer(count, sleep_after, value=7, w=8, h=4):
    return [sys.executable, "-c", PRODUCER, str(w), str(h), str(count), str(value), str(sleep_after)]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_ffmpeg_command():
    cmd = ffmpeg_command("rtsp://u:p@cam:554/stream1", CribCrop(x=10, y=20, w=640, h=360))
    assert cmd[0] == "ffmpeg"
    assert cmd.index("-rtsp_transport") < cmd.index("-i")
    assert "crop=640:360:10:20,fps=1" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"


def test_frames_are_decoded():
    cap = Capture(producer(3, 30), 8, 4)
    cap.start()
    try:
        assert cap.latest() is None or cap.latest().seq >= 1
        assert wait_for(lambda: cap.latest() is not None)
        frame = cap.latest()
        assert frame.image.shape == (4, 8, 3)
        assert int(frame.image[0, 0, 0]) == 7
    finally:
        cap.stop()


def test_stalled_stream_is_restarted():
    cap = Capture(producer(1, 30), 8, 4, stall_s=0.3, startup_s=2, initial_backoff_s=0.05)
    cap.start()
    try:
        assert wait_for(lambda: cap.latest() is not None and cap.latest().seq >= 3, timeout=8)
        assert cap.restarts >= 2
    finally:
        cap.stop()


def test_failing_command_backs_off():
    cap = Capture([sys.executable, "-c", "import sys; sys.exit(1)"], 8, 4, initial_backoff_s=0.1, max_backoff_s=0.4)
    cap.start()
    time.sleep(1.6)
    cap.stop()
    assert 2 <= cap.restarts <= 8
    assert cap.latest() is None


def test_stop_is_prompt():
    cap = Capture(producer(100000, 0), 8, 4)
    cap.start()
    assert wait_for(lambda: cap.latest() is not None)
    start = time.monotonic()
    cap.stop()
    assert time.monotonic() - start < 3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_capture.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/capture.py`**

```python
"""Persistent ffmpeg RTSP reader that always holds the latest cropped frame."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import CribCrop

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    seq: int


def ffmpeg_command(rtsp_url: str, crop: CribCrop, fps: int = 1) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vf", f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


class Capture:
    def __init__(
        self,
        cmd: list[str],
        width: int,
        height: int,
        *,
        stall_s: float = 5.0,
        startup_s: float = 15.0,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 15.0,
    ) -> None:
        self._cmd = list(cmd)
        self._w = width
        self._h = height
        self._size = width * height * 3
        self._stall_s = stall_s
        self._startup_s = startup_s
        self._initial_backoff = initial_backoff_s
        self._max_backoff = max_backoff_s
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self.restarts = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def _supervise(self) -> None:
        backoff = self._initial_backoff
        while not self._stop.is_set():
            got_frames = self._run_once()
            if self._stop.is_set():
                break
            self.restarts += 1
            if got_frames:
                backoff = self._initial_backoff
            log.warning("capture stopped; restarting in %.1fs", backoff)
            self._stop.wait(backoff)
            if not got_frames:
                backoff = min(backoff * 2, self._max_backoff)

    def _run_once(self) -> bool:
        try:
            proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        except OSError as exc:
            log.error("cannot start capture command: %s", exc)
            return False
        self._proc = proc
        got = threading.Event()
        last = [time.monotonic()]
        reader = threading.Thread(target=self._read, args=(proc, got, last), name="capture-reader", daemon=True)
        reader.start()
        try:
            while not self._stop.is_set() and proc.poll() is None:
                limit = self._stall_s if got.is_set() else self._startup_s
                if time.monotonic() - last[0] > limit:
                    log.warning("no frame for %.1fs; restarting capture", limit)
                    break
                self._stop.wait(0.1)
        finally:
            self._kill()
            reader.join(timeout=2)
            self._proc = None
        return got.is_set()

    def _read(self, proc: subprocess.Popen[bytes], got: threading.Event, last: list[float]) -> None:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(self._size)
            if not data or len(data) < self._size:
                return
            image = np.frombuffer(data, dtype=np.uint8).reshape(self._h, self._w, 3)
            with self._lock:
                self._seq += 1
                self._latest = Frame(image=image, seq=self._seq)
            last[0] = time.monotonic()
            got.set()

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.error("capture process did not exit after kill")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_capture.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/capture.py tests/test_capture.py
git commit -m "feat: persistent ffmpeg capture with stall detection and backoff"
```

---

### Task 11: Storage, retention, and labeled eval set

**Files:**
- Create: `crib_monitor/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: `LABELS = ("back", "stomach", "side", "not_visible", "skip")`; `Storage(data_dir: Path, retention_days: int, tz: ZoneInfo)` with `save_check(now, jpeg: bytes, record: dict) -> str` (returns relative frame path `YYYY-MM-DD/HHMMSS_mmm.jpg`), `cleanup(now)`, `frame_path(rel: str) -> Path` (raises `ValueError` outside `frames/` or missing), `unlabeled() -> list[str]`, `add_label(rel: str, label: str)`, `eval_items() -> list[tuple[Path, str]]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_storage.py`:

```python
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from crib_monitor.storage import Storage

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 26, 3, 4, 5, 678000, tzinfo=NY)


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(tmp_path, retention_days=30, tz=NY)


def test_save_check_writes_frame_and_log(storage, tmp_path):
    rel = storage.save_check(T0, b"jpeg", {"combined": "back"})
    assert rel == "2026-09-26/030405_678.jpg"
    assert (tmp_path / "frames" / rel).read_bytes() == b"jpeg"
    line = json.loads((tmp_path / "log" / "2026-09-26.jsonl").read_text().strip())
    assert line["frame"] == rel and line["combined"] == "back" and line["ts"] == T0.isoformat()


def test_cleanup_removes_old_days_only(storage, tmp_path):
    old = storage.save_check(T0 - timedelta(days=31), b"old", {})
    new = storage.save_check(T0, b"new", {})
    storage.add_label(old, "stomach")
    storage.cleanup(T0)
    assert not (tmp_path / "frames" / old).exists()
    assert not (tmp_path / "log" / f"{(T0 - timedelta(days=31)):%Y-%m-%d}.jsonl").exists()
    assert (tmp_path / "frames" / new).exists()
    assert len(storage.eval_items()) == 1          # labeled copy survives


def test_labeling_flow(storage):
    a = storage.save_check(T0, b"a", {})
    b = storage.save_check(T0 + timedelta(seconds=10), b"b", {})
    c = storage.save_check(T0 + timedelta(seconds=20), b"c", {})
    assert storage.unlabeled() == [a, b, c]
    storage.add_label(a, "stomach")
    storage.add_label(b, "skip")
    assert storage.unlabeled() == [c]
    items = storage.eval_items()
    assert len(items) == 1
    path, label = items[0]
    assert label == "stomach" and path.read_bytes() == b"a"


def test_invalid_label_rejected(storage):
    rel = storage.save_check(T0, b"a", {})
    with pytest.raises(ValueError):
        storage.add_label(rel, "upside_down")


@pytest.mark.parametrize("rel", ["../state.json", "/etc/passwd", "2026-09-26/missing.jpg"])
def test_frame_path_rejects_outside_or_missing(storage, rel):
    with pytest.raises(ValueError):
        storage.frame_path(rel)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/storage.py`**

```python
"""Checked frames, the per-check log, retention, and the labeled eval set."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LABELS = ("back", "stomach", "side", "not_visible", "skip")


class Storage:
    def __init__(self, data_dir: Path, retention_days: int, tz: ZoneInfo) -> None:
        self.root = Path(data_dir)
        self.frames = self.root / "frames"
        self.logs = self.root / "log"
        self.eval = self.root / "eval"
        self._labels_file = self.eval / "labels.jsonl"
        self._retention = retention_days
        self._tz = tz

    def save_check(self, now: datetime, jpeg: bytes, record: dict[str, Any]) -> str:
        local = now.astimezone(self._tz)
        day = f"{local:%Y-%m-%d}"
        rel = f"{day}/{local:%H%M%S}_{local.microsecond // 1000:03d}.jpg"
        path = self.frames / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.logs / f"{day}.jsonl", "a") as f:
            f.write(json.dumps({"ts": now.isoformat(), "frame": rel, **record}) + "\n")
        return rel

    def cleanup(self, now: datetime) -> None:
        cutoff = now.astimezone(self._tz).date() - timedelta(days=self._retention)
        if self.frames.exists():
            for day_dir in self.frames.iterdir():
                if _day(day_dir.name) is not None and _day(day_dir.name) < cutoff:
                    shutil.rmtree(day_dir)
        if self.logs.exists():
            for log_file in self.logs.glob("*.jsonl"):
                if _day(log_file.stem) is not None and _day(log_file.stem) < cutoff:
                    log_file.unlink()

    def frame_path(self, rel: str) -> Path:
        base = self.frames.resolve()
        path = (self.frames / rel).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"no such frame: {rel}")
        return path

    def unlabeled(self) -> list[str]:
        if not self.frames.exists():
            return []
        done = {entry["source"] for entry in self._entries()}
        all_frames = sorted(p.relative_to(self.frames).as_posix() for p in self.frames.glob("*/*.jpg"))
        return [rel for rel in all_frames if rel not in done]

    def add_label(self, rel: str, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        source = self.frame_path(rel)
        entry: dict[str, str] = {"source": rel, "label": label}
        if label != "skip":
            dest = self.eval / "images" / rel.replace("/", "_")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            entry["image"] = f"images/{dest.name}"
        self.eval.mkdir(parents=True, exist_ok=True)
        with open(self._labels_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def eval_items(self) -> list[tuple[Path, str]]:
        latest: dict[str, dict[str, str]] = {}
        for entry in self._entries():
            latest[entry["source"]] = entry
        return [(self.eval / e["image"], e["label"]) for e in latest.values() if "image" in e]

    def _entries(self) -> list[dict[str, str]]:
        if not self._labels_file.exists():
            return []
        return [json.loads(line) for line in self._labels_file.read_text().splitlines() if line.strip()]


def _day(name: str) -> date | None:
    try:
        return date.fromisoformat(name)
    except ValueError:
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_storage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/storage.py tests/test_storage.py
git commit -m "feat: frame storage, retention, labeled eval set"
```

---

### Task 12: Monitor orchestration

**Files:**
- Create: `crib_monitor/monitor.py`
- Test: `tests/test_monitor.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `Arming`, `ArmingStatus` (Task 4), `Engine`, action classes, `State` (Task 5), `ClassifyResult`, `classify_all` (Task 6), `Health`, `Heartbeat` (Task 7), `Alerter`, `NotifyError`, `ReceiptStatus` (Task 8), `resize_max_side`, `encode_jpeg`, `MotionDetector` (Task 9), `Frame` (Task 10), `Storage` (Task 11), `combine`, `Combined` (Task 3).
- Produces: `Snapshot(status, state: str, last_check, last_results, last_combined: str | None, health: list[str], shadow_mode: bool)`; `Monitor(*, cfg, arming, capture_factory, classifiers, alerter, health, heartbeat, storage, now)` with `async tick()`, `async run_forever(interval_s=1.0)`, `shutdown()`, controller methods `on()`, `off()`, `pause() -> bool`, `resume()`, `async test_alert() -> bool`, `snapshot() -> Snapshot`, `latest_jpeg() -> bytes | None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_monitor.py`:

```python
import asyncio
import contextlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from crib_monitor.arming import Arming
from crib_monitor.capture import Frame
from crib_monitor.classifier import ClassifyResult
from crib_monitor.health import Health
from crib_monitor.labels import Position
from crib_monitor.monitor import Monitor
from crib_monitor.notifier import NotifyError, ReceiptStatus
from crib_monitor.schedule import Schedule
from crib_monitor.storage import Storage

NY = ZoneInfo("America/New_York")
TUE_1PM = datetime(2026, 9, 29, 13, 0, tzinfo=NY)   # outside the schedule
B, S, NV = Position.BACK, Position.STOMACH, Position.NOT_VISIBLE


def img(value=0):
    return np.full((180, 320, 3), value, dtype=np.uint8)


class Clock:
    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class FakeCapture:
    def __init__(self):
        self.started = self.stopped = 0
        self.frame = None

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def latest(self):
        return self.frame

    def push(self, image=None):
        seq = self.frame.seq + 1 if self.frame else 1
        self.frame = Frame(image=img() if image is None else image, seq=seq)


class FakeClassifier:
    def __init__(self, name, script=(), default=B):
        self.name, self.script, self.default, self.calls = name, list(script), default, 0

    async def classify(self, jpeg):
        self.calls += 1
        position = self.script.pop(0) if self.script else self.default
        return ClassifyResult(self.name, position, 0.1, None if position else "fake failure")


class FakeAlerter:
    def __init__(self):
        self.calls, self.args, self.fail = [], {}, set()
        self.receipt_status = ReceiptStatus(acknowledged=False, expired=False)

    async def _call(self, name, *args, result=None):
        self.calls.append(name)
        self.args[name] = args
        if name in self.fail:
            raise NotifyError(name)
        return result

    async def stomach(self, image):
        return await self._call("stomach", image, result="R1")

    async def no_view(self, image):
        return await self._call("no_view", image)

    async def back_on_back(self):
        return await self._call("back_on_back")

    async def health(self, event):
        return await self._call("health", event)

    async def monitoring_started(self, camera_ok, models, image):
        return await self._call("monitoring_started", camera_ok, models, image)

    async def manual_ended(self):
        return await self._call("manual_ended")

    async def test(self):
        return await self._call("test", result="T1")

    async def receipt(self, receipt):
        await self._call("receipt", receipt)
        return self.receipt_status

    async def cancel(self, receipt):
        return await self._call("cancel", receipt)


class FakeHeartbeat:
    def __init__(self):
        self.pings = []

    async def ping(self, ok=True):
        self.pings.append(ok)


class Rig:
    def __init__(self, config, tmp_path, local=(), cloud=(), local_default=B, cloud_default=B):
        cfg = config.model_copy(update={
            "alerts": config.alerts.model_copy(update={"shadow_mode": False}),
            "storage": config.storage.model_copy(update={"data_dir": tmp_path}),
        })
        self.clock = Clock(TUE_1PM)
        self.capture = FakeCapture()
        self.local = FakeClassifier("local", local, local_default)
        self.cloud = FakeClassifier("cloud", cloud, cloud_default)
        self.alerter = FakeAlerter()
        self.heartbeat = FakeHeartbeat()
        self.monitor = Monitor(
            cfg=cfg,
            arming=Arming(Schedule(cfg.schedule), cfg.arming, tmp_path / "state.json"),
            capture_factory=lambda: self.capture,
            classifiers=[self.local, self.cloud],
            alerter=self.alerter,
            health=Health(cfg.health, ["local", "cloud"], cfg.alerts.health_repeat_s),
            heartbeat=self.heartbeat,
            storage=Storage(tmp_path, cfg.storage.retention_days, NY),
            now=self.clock,
        )

    async def step(self, seconds=0, image=None):
        self.clock.advance(seconds)
        self.capture.push(image)
        await self.monitor.tick()

    @property
    def checks(self):
        return self.cloud.calls


@pytest.fixture
def rig(config, tmp_path):
    return lambda **kw: Rig(config, tmp_path, **kw)


async def test_disarmed_does_nothing(rig):
    r = rig()
    await r.monitor.tick()
    assert r.capture.started == 0 and r.checks == 0
    assert r.heartbeat.pings == [True]


async def test_arming_runs_selftest(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    assert r.capture.started == 1 and r.checks == 1
    camera_ok, models, image = r.alerter.args["monitoring_started"]
    assert camera_ok is True and models == {"local": True, "cloud": True} and image[:2] == b"\xff\xd8"


async def test_no_frame_selftest_reports_camera_down(rig):
    r = rig()
    r.monitor.on()
    await r.monitor.tick()
    r.clock.advance(30)
    await r.monitor.tick()
    camera_ok, models, _ = r.alerter.args["monitoring_started"]
    assert camera_ok is False and models == {"local": None, "cloud": None}


async def test_stomach_confirmed_alerts_polls_and_clears(rig):
    r = rig(local=[S, S])
    r.monitor.on()
    await r.step()                        # self-test check: stomach -> confirming
    await r.step(10)                      # confirmation: stomach -> alert
    assert r.alerter.calls.count("stomach") == 1
    assert r.monitor.snapshot().state == "alerted"
    r.alerter.receipt_status = ReceiptStatus(acknowledged=True, expired=False)
    await r.step(15)                      # receipt poll
    assert "receipt" in r.alerter.calls
    await r.step(15)                      # back (1)
    await r.step(30)                      # back (2) -> back on back
    assert "back_on_back" in r.alerter.calls
    assert r.monitor.snapshot().state == "monitoring"


async def test_one_model_down_still_alerts(rig):
    r = rig(cloud=[S, S], local_default=None)
    r.monitor.on()
    await r.step()
    await r.step(10)
    assert r.alerter.calls.count("stomach") == 1


async def test_paused_suppresses_and_resumes_after_return(rig):
    script = [B, S, NV, B, B]
    r = rig(local=script, cloud=script)
    r.monitor.on()
    await r.step()                        # self-test: back
    assert r.monitor.pause()
    await r.step(30)                      # stomach while paused: no alert
    await r.step(30)                      # not visible (picked up)
    await r.step(30)                      # back
    assert r.monitor.snapshot().status.paused
    await r.step(30)                      # back again -> pause ends
    await r.step(1)
    assert "stomach" not in r.alerter.calls
    assert not r.monitor.snapshot().status.paused


async def test_motion_gating(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    for _ in range(20):
        await r.step(1)
    assert r.checks == 1
    await r.step(1, image=img(200))
    assert r.checks == 2


async def test_max_interval_check_without_motion(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    for _ in range(299):
        await r.step(1)
    assert r.checks == 1
    await r.step(1)
    assert r.checks == 2


async def test_manual_cap_stops_capture_and_notifies(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    r.clock.advance(4 * 3600)
    await r.monitor.tick()
    assert r.capture.stopped == 1
    assert "manual_ended" in r.alerter.calls


async def test_notification_failure_fails_heartbeat(rig):
    r = rig(local=[S, S])
    r.alerter.fail = {"stomach"}
    r.monitor.on()
    await r.step()
    await r.step(10)
    assert False in r.heartbeat.pings


async def test_tick_exception_fails_heartbeat(rig):
    r = rig()

    class Broken(FakeCapture):
        def latest(self):
            raise RuntimeError("bug")

    r.capture = Broken()
    r.monitor.on()
    task = asyncio.create_task(r.monitor.run_forever(interval_s=0.01))
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert r.heartbeat.pings and r.heartbeat.pings[0] is False
    assert True not in r.heartbeat.pings


async def test_logs_every_check(rig, tmp_path):
    r = rig()
    r.monitor.on()
    await r.step()
    logs = list((tmp_path / "log").glob("*.jsonl"))
    text = logs[0].read_text()
    assert len(logs) == 1 and '"combined": "back"' in text and '"motion":' in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/monitor.py`**

```python
"""Ties capture, models, decision logic, health, and alerts together."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, TypeVar

from .arming import Arming, ArmingStatus
from .capture import Frame
from .classifier import ClassifyResult, classify_all
from .config import Config
from .decision import BackOnBack, Engine, FalsePositive, NoViewAlert, StomachAlert
from .health import Health
from .imaging import encode_jpeg, resize_max_side
from .labels import Combined, combine
from .motion import MotionDetector
from .notifier import NotifyError
from .storage import Storage

log = logging.getLogger(__name__)

PAUSED_INTERVAL_S = 30
SELFTEST_FRAME_WAIT_S = 30
CLEANUP_INTERVAL_S = 3600

T = TypeVar("T")


class CaptureLike(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self) -> Frame | None: ...


class HeartbeatLike(Protocol):
    async def ping(self, ok: bool = True) -> None: ...


@dataclass
class Snapshot:
    status: ArmingStatus | None
    state: str
    last_check: datetime | None
    last_results: list[ClassifyResult] = field(default_factory=list)
    last_combined: str | None = None
    health: list[str] = field(default_factory=list)
    shadow_mode: bool = True


class Monitor:
    def __init__(
        self,
        *,
        cfg: Config,
        arming: Arming,
        capture_factory: Callable[[], CaptureLike],
        classifiers: Sequence[Any],
        alerter: Any,
        health: Health,
        heartbeat: HeartbeatLike,
        storage: Storage,
        now: Callable[[], datetime],
    ) -> None:
        self._cfg = cfg
        self._arming = arming
        self._capture_factory = capture_factory
        self._classifiers = list(classifiers)
        self._alerter = alerter
        self._health = health
        self._heartbeat = heartbeat
        self._storage = storage
        self._now = now
        self._engine = Engine(ack_suppress_s=cfg.alerts.ack_suppress_s)
        self._motion = MotionDetector(pixel_delta=cfg.motion.pixel_delta)
        self._status: ArmingStatus | None = None
        self._capture: CaptureLike | None = None
        self._frame: Frame | None = None
        self._last_seq: int | None = None
        self._last_checked_seq: int | None = None
        self._last_check: datetime | None = None
        self._motion_pending = False
        self._motion_peak = 0.0
        self._selftest_pending = False
        self._armed_at: datetime | None = None
        self._receipt: str | None = None
        self._next_receipt_poll: datetime | None = None
        self._notify_failing = False
        self._last_heartbeat: datetime | None = None
        self._last_fail_ping: datetime | None = None
        self._last_cleanup: datetime | None = None
        self._last_results: list[ClassifyResult] = []
        self._last_combined: Combined | None = None
        self._last_jpeg: bytes | None = None

    # --- controller (web page) --------------------------------------------

    def on(self) -> None:
        self._arming.turn_on(self._now())

    def off(self) -> None:
        self._arming.turn_off(self._now())

    def pause(self) -> bool:
        return self._arming.pause(self._now())

    def resume(self) -> None:
        self._arming.resume()

    async def test_alert(self) -> bool:
        return await self._send(self._alerter.test()) is not None

    def snapshot(self) -> Snapshot:
        return Snapshot(
            status=self._status,
            state=self._engine.state.value,
            last_check=self._last_check,
            last_results=list(self._last_results),
            last_combined=self._last_combined.value if self._last_combined else None,
            health=self._health.active,
            shadow_mode=self._cfg.alerts.shadow_mode,
        )

    def latest_jpeg(self) -> bytes | None:
        if self._frame is not None:
            return encode_jpeg(resize_max_side(self._frame.image, self._cfg.classifier.max_side_px))
        return self._last_jpeg

    # --- loop ----------------------------------------------------------------

    async def run_forever(self, interval_s: float = 1.0) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("monitor tick failed")
                now = self._now()
                if (
                    self._last_fail_ping is None
                    or (now - self._last_fail_ping).total_seconds() >= self._cfg.health.heartbeat_interval_s
                ):
                    self._last_fail_ping = now
                    await self._heartbeat.ping(ok=False)
            await asyncio.sleep(interval_s)

    def shutdown(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

    async def tick(self) -> None:
        now = self._now()
        status = self._arming.status(now)
        await self._handle_transitions(status, now)
        self._status = status
        if status.armed:
            await self._tick_armed(status, now)
        if self._last_cleanup is None or (now - self._last_cleanup).total_seconds() >= CLEANUP_INTERVAL_S:
            self._storage.cleanup(now)
            self._last_cleanup = now
        # Reached only when everything above succeeded, so a tick that keeps raising stops the green pings.
        if (
            self._last_heartbeat is None
            or (now - self._last_heartbeat).total_seconds() >= self._cfg.health.heartbeat_interval_s
        ):
            await self._heartbeat.ping(ok=not self._notify_failing)
            self._last_heartbeat = now

    # --- internals ---------------------------------------------------------

    async def _handle_transitions(self, status: ArmingStatus, now: datetime) -> None:
        prev = self._status
        was_armed = prev is not None and prev.armed
        if status.armed and not was_armed:
            log.info("armed (%s)", status.source)
            self._capture = self._capture_factory()
            self._capture.start()
            self._motion = MotionDetector(pixel_delta=self._cfg.motion.pixel_delta)
            self._engine.reset()
            self._health.reset(now)
            self._frame = None
            self._last_seq = None
            self._last_checked_seq = None
            self._last_check = None
            self._motion_pending = False
            self._motion_peak = 0.0
            self._selftest_pending = True
            self._armed_at = now
        elif was_armed and not status.armed:
            log.info("disarmed")
            self.shutdown()
            await self._cancel_receipt()
            self._frame = None
            assert prev is not None
            if prev.source == "manual" and prev.manual_until is not None and now >= prev.manual_until:
                await self._send(self._alerter.manual_ended())
        elif status.armed and prev is not None:
            if status.paused and not prev.paused:
                await self._cancel_receipt()
            if prev.paused and not status.paused:
                self._engine.reset()

    async def _tick_armed(self, status: ArmingStatus, now: datetime) -> None:
        frame = self._capture.latest() if self._capture else None
        if frame is not None and frame.seq != self._last_seq:
            self._last_seq = frame.seq
            self._frame = frame
            self._health.frame(now)
            score = self._motion.score(frame.image)
            self._motion_peak = max(self._motion_peak, score)
            if score > self._cfg.motion.threshold:
                self._motion_pending = True
        for event in self._health.tick(now):
            await self._send(self._alerter.health(event))
        await self._poll_receipt(now)
        if self._frame is None:
            assert self._armed_at is not None
            if self._selftest_pending and (now - self._armed_at).total_seconds() >= SELFTEST_FRAME_WAIT_S:
                self._selftest_pending = False
                models = {c.name: None for c in self._classifiers}
                await self._send(self._alerter.monitoring_started(False, models, None))
            return
        if self._should_check(status, now):
            await self._check(status, now)

    def _should_check(self, status: ArmingStatus, now: datetime) -> bool:
        assert self._frame is not None
        if self._frame.seq == self._last_checked_seq:
            return False
        if self._selftest_pending or self._last_check is None:
            return True
        since = (now - self._last_check).total_seconds()
        forced = PAUSED_INTERVAL_S if status.paused else self._engine.force_interval_s
        if forced is not None and since >= forced:
            return True
        if since >= self._cfg.motion.max_check_interval_s:
            return True
        return self._motion_pending and since >= self._cfg.motion.min_check_interval_s

    async def _check(self, status: ArmingStatus, now: datetime) -> None:
        frame = self._frame
        assert frame is not None
        jpeg = encode_jpeg(resize_max_side(frame.image, self._cfg.classifier.max_side_px))
        results = await classify_all(self._classifiers, jpeg)
        combined = combine(r.position for r in results)
        for event in self._health.results(results, now):
            await self._send(self._alerter.health(event))
        before = self._engine.state
        actions = []
        if status.paused:
            if self._arming.observe(combined, now):
                self._engine.reset()
        else:
            actions = self._engine.step(combined, now)
        self._storage.save_check(now, jpeg, {
            "motion": round(self._motion_peak, 4),
            "results": [
                {
                    "model": r.model_name,
                    "position": r.position.value if r.position else None,
                    "latency_s": round(r.latency_s, 2),
                    "error": r.error,
                }
                for r in results
            ],
            "combined": combined.value,
            "paused": status.paused,
            "state_before": before.value,
            "state_after": self._engine.state.value,
            "actions": [type(a).__name__ for a in actions],
            "shadow_mode": self._cfg.alerts.shadow_mode,
        })
        self._last_check = now
        self._last_checked_seq = frame.seq
        self._motion_pending = False
        self._motion_peak = 0.0
        self._last_results = results
        self._last_combined = combined
        self._last_jpeg = jpeg
        for action in actions:
            await self._act(action, jpeg, now)
        if self._selftest_pending:
            self._selftest_pending = False
            models = {r.model_name: r.position is not None for r in results}
            await self._send(self._alerter.monitoring_started(True, models, jpeg))

    async def _act(self, action: object, jpeg: bytes, now: datetime) -> None:
        if isinstance(action, StomachAlert):
            receipt = await self._send(self._alerter.stomach(jpeg))
            if receipt:
                self._receipt = receipt
                self._next_receipt_poll = now + timedelta(seconds=self._cfg.alerts.receipt_poll_s)
            else:
                # Shadow mode or a failed send: no receipt to wait on, so re-alert after the suppression window.
                self._engine.on_ack(now)
        elif isinstance(action, NoViewAlert):
            await self._send(self._alerter.no_view(jpeg))
        elif isinstance(action, BackOnBack):
            await self._cancel_receipt()
            await self._send(self._alerter.back_on_back())
        elif isinstance(action, FalsePositive):
            log.info("stomach reading not confirmed")

    async def _poll_receipt(self, now: datetime) -> None:
        if self._receipt is None or self._next_receipt_poll is None or now < self._next_receipt_poll:
            return
        self._next_receipt_poll = now + timedelta(seconds=self._cfg.alerts.receipt_poll_s)
        try:
            status = await self._alerter.receipt(self._receipt)
        except NotifyError as exc:
            log.warning("%s", exc)
            return
        if status.acknowledged:
            self._receipt = None
            self._engine.on_ack(now)
        elif status.expired:
            self._receipt = None
            self._engine.on_expired(now)

    async def _cancel_receipt(self) -> None:
        if self._receipt is None:
            return
        receipt, self._receipt = self._receipt, None
        try:
            await self._alerter.cancel(receipt)
        except NotifyError as exc:
            log.warning("%s", exc)

    async def _send(self, coro: Awaitable[T]) -> T | None:
        try:
            result = await coro
        except NotifyError as exc:
            log.error("notification failed: %s", exc)
            if not self._notify_failing:
                self._notify_failing = True
                await self._heartbeat.ping(ok=False)
            return None
        self._notify_failing = False
        return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: 12 passed

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add crib_monitor/monitor.py tests/test_monitor.py
git commit -m "feat: monitor orchestration loop"
```

---

### Task 13: LAN control and labeling page

**Files:**
- Create: `crib_monitor/web.py`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `Snapshot` (Task 12), `ArmingStatus` (Task 4), `ClassifyResult` (Task 6), `Storage`, `LABELS` (Task 11).
- Produces: `create_app(controller, storage: Storage, token: str, tz: ZoneInfo) -> FastAPI`; `describe(snapshot: Snapshot, tz) -> str`. `controller` needs `on()`, `off()`, `pause()`, `resume()`, `async test_alert()`, `snapshot()`, `latest_jpeg()` (the `Monitor` from Task 12).

- [ ] **Step 1: Write the failing tests**

`tests/test_web.py`:

```python
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from crib_monitor.arming import ArmingStatus
from crib_monitor.monitor import Snapshot
from crib_monitor.storage import Storage
from crib_monitor.web import create_app, describe

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 26, 3, 0, tzinfo=NY)


class FakeController:
    def __init__(self):
        self.calls = []
        self.snap = Snapshot(status=None, state="monitoring", last_check=None)
        self.jpeg = b"\xff\xd8frame"

    def on(self):
        self.calls.append("on")

    def off(self):
        self.calls.append("off")

    def pause(self):
        self.calls.append("pause")
        return True

    def resume(self):
        self.calls.append("resume")

    async def test_alert(self):
        self.calls.append("test")
        return True

    def snapshot(self):
        return self.snap

    def latest_jpeg(self):
        return self.jpeg


@pytest.fixture
def setup(tmp_path):
    ctrl = FakeController()
    storage = Storage(tmp_path, 30, NY)
    client = TestClient(create_app(ctrl, storage, "s3cret", NY))
    return ctrl, storage, client


def test_token_required(setup):
    _, _, client = setup
    assert client.get("/").status_code == 403
    assert client.get("/?t=wrong").status_code == 403
    assert client.post("/on").status_code == 403
    assert client.get("/?t=s3cret").status_code == 200


@pytest.mark.parametrize("action", ["on", "off", "pause", "resume", "test"])
def test_buttons_call_controller(setup, action):
    ctrl, _, client = setup
    response = client.post(f"/{action}?t=s3cret", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/?t=s3cret"
    assert ctrl.calls == [action]


def test_status_page_shows_state_and_shadow(setup):
    ctrl, _, client = setup
    ctrl.snap = Snapshot(
        status=ArmingStatus(armed=True, paused=False, source="schedule", window_end=datetime(2026, 9, 26, 7, 0, tzinfo=NY)),
        state="monitoring", last_check=T0, health=["model_down:local"], shadow_mode=True,
    )
    page = client.get("/?t=s3cret").text
    assert "On (schedule) until 07:00" in page
    assert "Shadow mode" in page
    assert "model_down:local" in page


def test_frame_endpoint(setup):
    ctrl, _, client = setup
    assert client.get("/frame.jpg?t=s3cret").content == b"\xff\xd8frame"
    ctrl.jpeg = None
    assert client.get("/frame.jpg?t=s3cret").status_code == 404


def test_label_flow(setup):
    _, storage, client = setup
    rel = storage.save_check(T0, b"\xff\xd8a", {})
    page = client.get("/label?t=s3cret").text
    assert rel in page and "1 left" in page
    assert client.get(f"/frames/{rel}?t=s3cret").content == b"\xff\xd8a"
    response = client.post("/label?t=s3cret", data={"image": rel, "label": "stomach"}, follow_redirects=False)
    assert response.status_code == 303
    assert storage.unlabeled() == []
    assert "Nothing to label" in client.get("/label?t=s3cret").text


def test_label_rejects_bad_input(setup):
    _, storage, client = setup
    rel = storage.save_check(T0, b"a", {})
    assert client.post("/label?t=s3cret", data={"image": rel, "label": "nope"}).status_code == 400
    assert client.post("/label?t=s3cret", data={"image": "../state.json", "label": "back"}).status_code == 400
    assert client.get("/frames/../state.json?t=s3cret").status_code == 404


def test_describe():
    off = Snapshot(status=ArmingStatus(armed=False, paused=False, source=None), state="monitoring", last_check=None)
    assert describe(off, NY) == "Off"
    paused = Snapshot(
        status=ArmingStatus(armed=True, paused=True, source="manual", paused_until=datetime(2026, 9, 26, 3, 45, tzinfo=NY)),
        state="monitoring", last_check=None,
    )
    assert describe(paused, NY) == "Paused until 03:45"
    nap = Snapshot(
        status=ArmingStatus(armed=True, paused=False, source="manual", manual_until=datetime(2026, 9, 26, 17, 0, tzinfo=NY)),
        state="monitoring", last_check=None,
    )
    assert describe(nap, NY) == "On (nap) until 17:00"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_web.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/web.py`**

```python
"""LAN-only control page: status, On/Off/Pause, test alert, frame labeling."""

from __future__ import annotations

import html
import secrets
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .monitor import Snapshot
from .storage import Storage

_CSS = (
    "body{font-family:system-ui,sans-serif;margin:0 auto;max-width:640px;padding:16px;background:#fff;color:#111}"
    "@media (prefers-color-scheme:dark){body{background:#111;color:#eee}a{color:#9cf}}"
    "img{width:100%;border-radius:8px;background:#333}"
    ".row{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}"
    ".row form{flex:1 1 40%}"
    "button{width:100%;font-size:1.1rem;padding:14px 18px;border-radius:10px;border:1px solid #888}"
    ".warn{background:#fde68a;color:#111;padding:8px;border-radius:8px}"
)

_BUTTONS = [("on", "On"), ("off", "Off"), ("pause", "Pause 45 min"), ("resume", "Resume"), ("test", "Test alert")]
_LABEL_BUTTONS = [("back", "Back"), ("stomach", "Stomach"), ("side", "Side"), ("not_visible", "Not visible"), ("skip", "Skip")]


def describe(snapshot: Snapshot, tz: ZoneInfo) -> str:
    st = snapshot.status
    if st is None or not st.armed:
        return "Off"
    if st.paused and st.paused_until:
        return f"Paused until {st.paused_until.astimezone(tz):%H:%M}"
    if st.source == "manual" and st.manual_until:
        return f"On (nap) until {st.manual_until.astimezone(tz):%H:%M}"
    if st.window_end:
        return f"On (schedule) until {st.window_end.astimezone(tz):%H:%M}"
    return "On"


def _html(title: str, body: str, refresh: int | None = None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{meta}<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )


def _status_page(s: Snapshot, q: str, tz: ZoneInfo) -> str:
    parts = [f"<h1>{html.escape(describe(s, tz))}</h1>"]
    if s.shadow_mode:
        parts.append('<p class="warn">Shadow mode: roll alerts are sent as [TEST] messages.</p>')
    for issue in s.health:
        parts.append(f'<p class="warn">Health: {html.escape(issue)}</p>')
    if s.last_check:
        labels = ", ".join(
            f"{html.escape(r.model_name)}={html.escape(r.position.value if r.position else 'unavailable')}"
            for r in s.last_results
        )
        parts.append(
            f"<p>Last check {s.last_check.astimezone(tz):%H:%M:%S}: "
            f"{html.escape(s.last_combined or '')} ({labels}) &middot; state {html.escape(s.state)}</p>"
        )
    parts.append(f'<img src="/frame.jpg{q}" alt="Latest frame">')
    buttons = "".join(
        f'<form method="post" action="/{action}{q}"><button>{label}</button></form>' for action, label in _BUTTONS
    )
    parts.append(f'<div class="row">{buttons}</div>')
    parts.append(f'<p><a href="/label{q}">Label frames</a></p>')
    return _html("Crib monitor", "".join(parts), refresh=15)


def _label_page(remaining: list[str], q: str) -> str:
    if not remaining:
        body = "<h1>Nothing to label</h1>"
    else:
        rel = remaining[0]
        buttons = "".join(
            f'<button name="label" value="{value}" style="flex:1 1 40%">{text}</button>' for value, text in _LABEL_BUTTONS
        )
        body = (
            f"<h1>Label ({len(remaining)} left)</h1><p>{html.escape(rel)}</p>"
            f'<img src="/frames/{quote(rel)}{q}" alt="Frame to label">'
            f'<form method="post" action="/label{q}"><input type="hidden" name="image" value="{html.escape(rel)}">'
            f'<div class="row">{buttons}</div></form>'
        )
    return _html("Label frames", body + f'<p><a href="/{q}">Back to status</a></p>')


def create_app(controller: Any, storage: Storage, token: str, tz: ZoneInfo) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    q = f"?t={quote(token)}"

    def auth(t: str = Query("")) -> None:
        if not secrets.compare_digest(t.encode(), token.encode()):
            raise HTTPException(status_code=403)

    def home() -> RedirectResponse:
        return RedirectResponse("/" + q, status_code=303)

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def index() -> HTMLResponse:
        return HTMLResponse(_status_page(controller.snapshot(), q, tz))

    @app.post("/on", dependencies=[Depends(auth)])
    async def on() -> RedirectResponse:
        controller.on()
        return home()

    @app.post("/off", dependencies=[Depends(auth)])
    async def off() -> RedirectResponse:
        controller.off()
        return home()

    @app.post("/pause", dependencies=[Depends(auth)])
    async def pause() -> RedirectResponse:
        controller.pause()
        return home()

    @app.post("/resume", dependencies=[Depends(auth)])
    async def resume() -> RedirectResponse:
        controller.resume()
        return home()

    @app.post("/test", dependencies=[Depends(auth)])
    async def test() -> RedirectResponse:
        await controller.test_alert()
        return home()

    @app.get("/frame.jpg", dependencies=[Depends(auth)])
    async def frame() -> Response:
        data = controller.latest_jpeg()
        if data is None:
            raise HTTPException(status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/label", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def label_page() -> HTMLResponse:
        return HTMLResponse(_label_page(storage.unlabeled(), q))

    @app.post("/label", dependencies=[Depends(auth)])
    async def label(image: str = Form(...), label: str = Form(...)) -> RedirectResponse:
        try:
            storage.add_label(image, label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/label" + q, status_code=303)

    @app.get("/frames/{rel:path}", dependencies=[Depends(auth)])
    async def frames(rel: str) -> FileResponse:
        try:
            path = storage.frame_path(rel)
        except ValueError as exc:
            raise HTTPException(status_code=404) from exc
        return FileResponse(path, media_type="image/jpeg")

    return app
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_web.py -v`
Expected: 11 passed

Note: `TestClient` normalizes `/frames/../state.json` before routing, so that request may resolve to `/state.json` and return 404 from the router rather than from `frame_path`; either way it must not be 200.

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/web.py tests/test_web.py
git commit -m "feat: LAN control and labeling page"
```

---

### Task 14: Evaluation report and probe tool

**Files:**
- Create: `crib_monitor/evaluate.py`, `crib_monitor/probe.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `load_config` (Task 1), `build_classifiers`, `classify_all`, `ClassifyResult` (Task 6), `Position` (Task 3), `Storage.eval_items` (Task 11), `Capture`, `ffmpeg_command` (Task 10), `encode_jpeg`, `resize_max_side` (Task 9).
- Produces: `Stats` (fields `tp`, `fn`, `fp`, `tn`, `unavailable`, `latencies`; properties `recall`, `fp_rate`), `score(items: Iterable[tuple[str, list[ClassifyResult]]]) -> dict[str, Stats]` (per model plus `"combined"`), `format_report(stats, checks_per_night: int) -> str`, CLI `crib-monitor-eval` (exit 0 only if combined stomach recall is 100%); CLI `crib-monitor-probe`.

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate.py`:

```python
from crib_monitor.classifier import ClassifyResult
from crib_monitor.evaluate import format_report, score
from crib_monitor.labels import Position

B, S, U = Position.BACK, Position.STOMACH, Position.UNCLEAR


def item(truth, local, cloud):
    return truth, [ClassifyResult("local", local, 2.0), ClassifyResult("cloud", cloud, 1.0)]


def test_score_per_model_and_combined():
    stats = score([
        item("stomach", S, B),       # local catches, cloud misses
        item("stomach", None, S),    # local unavailable, cloud catches
        item("back", B, S),          # cloud false alarm
        item("side", U, B),
        item("not_visible", None, None),
    ])
    assert (stats["local"].tp, stats["local"].fn, stats["local"].fp, stats["local"].unavailable) == (1, 1, 0, 2)
    assert (stats["cloud"].tp, stats["cloud"].fn, stats["cloud"].fp) == (1, 1, 1)
    combined = stats["combined"]
    assert combined.recall == 1.0
    assert combined.fp == 1 and combined.tn == 2
    assert combined.unavailable == 1


def test_report_mentions_every_model():
    report = format_report(score([item("stomach", S, B), item("back", B, B)]), checks_per_night=200)
    assert "local: stomach caught 1/1 (100%)" in report
    assert "cloud: stomach caught 0/1 (0%)" in report
    assert "combined:" in report
    assert "200 checks per night" in report
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `crib_monitor/evaluate.py` and `crib_monitor/probe.py`**

`crib_monitor/evaluate.py`:

```python
"""How well does each model (and the either-says-stomach combination) do on labeled crib frames?"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .classifier import ClassifyResult, build_classifiers, classify_all
from .config import load_config
from .labels import Position
from .storage import Storage


@dataclass
class Stats:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    unavailable: int = 0
    latencies: list[float] = field(default_factory=list)

    def add(self, truth_stomach: bool, predicted_stomach: bool, available: bool) -> None:
        if not available:
            self.unavailable += 1
        if truth_stomach:
            if predicted_stomach:
                self.tp += 1
            else:
                self.fn += 1
        elif predicted_stomach:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def fp_rate(self) -> float | None:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else None


def score(items: Iterable[tuple[str, list[ClassifyResult]]]) -> dict[str, Stats]:
    stats: dict[str, Stats] = {}
    for truth, results in items:
        is_stomach = truth == "stomach"
        for r in results:
            s = stats.setdefault(r.model_name, Stats())
            s.add(is_stomach, r.position is Position.STOMACH, r.position is not None)
            s.latencies.append(r.latency_s)
        stats.setdefault("combined", Stats()).add(
            is_stomach,
            any(r.position is Position.STOMACH for r in results),
            any(r.position is not None for r in results),
        )
    # keep "combined" last in the report
    combined = stats.pop("combined", None)
    if combined is not None:
        stats["combined"] = combined
    return stats


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_report(stats: dict[str, Stats], checks_per_night: int) -> str:
    lines = []
    for name, s in stats.items():
        latency = f"{statistics.median(s.latencies):.1f}s" if s.latencies else "-"
        estimate = "n/a" if s.fp_rate is None else f"{checks_per_night * s.fp_rate ** 2:.2f}"
        lines.append(
            f"{name}: stomach caught {s.tp}/{s.tp + s.fn} ({_pct(s.recall)}), "
            f"false stomach {s.fp}/{s.fp + s.tn} ({_pct(s.fp_rate)}), unavailable {s.unavailable}, "
            f"median latency {latency}, est. false alerts/night {estimate}"
        )
    lines.append(
        f"Estimate assumes {checks_per_night} checks per night and that a false alert needs two independent "
        "false stomach readings in a row. Real errors are correlated, so treat it as a lower bound."
    )
    return "\n".join(lines)


async def evaluate(classifiers: Sequence[Any], items: list[tuple[Path, str]]) -> list[tuple[str, list[ClassifyResult]]]:
    out = []
    for path, truth in items:
        out.append((truth, await classify_all(classifiers, path.read_bytes())))
    return out


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crib-monitor-eval", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--checks-per-night", type=int, default=200)
    args = parser.parse_args(argv)
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
    cfg = load_config(Path(args.config))
    storage = Storage(cfg.storage.data_dir, cfg.storage.retention_days, ZoneInfo(cfg.schedule.timezone))
    items = storage.eval_items()
    if not items:
        print("No labeled frames yet. Label some on the control page's /label view first.")
        raise SystemExit(1)
    results = asyncio.run(evaluate(build_classifiers(cfg.classifier, os.environ), items))
    stats = score(results)
    print(format_report(stats, args.checks_per_night))
    passed = stats["combined"].recall == 1.0
    print("PASS: every labeled stomach frame was caught." if passed else "FAIL: at least one stomach frame was missed.")
    raise SystemExit(0 if passed else 1)
```

`crib_monitor/probe.py`:

```python
"""Grab one frame from the camera and run each model on it once."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .capture import Capture, ffmpeg_command
from .classifier import build_classifiers, classify_all
from .config import ConfigError, load_config
from .imaging import encode_jpeg, resize_max_side


async def probe(config_path: Path, out: Path) -> int:
    cfg = load_config(config_path)
    url = os.environ.get("TAPO_RTSP_URL")
    if not url:
        raise ConfigError("TAPO_RTSP_URL is not set")
    crop = cfg.camera.crop
    capture = Capture(ffmpeg_command(url, crop), crop.w, crop.h)
    capture.start()
    deadline = time.monotonic() + 20
    frame = None
    while time.monotonic() < deadline and frame is None:
        await asyncio.sleep(0.2)
        frame = capture.latest()
    capture.stop()
    if frame is None:
        print("camera: no frame within 20 s (check TAPO_RTSP_URL, the camera account, and the crop)")
        return 1
    jpeg = encode_jpeg(resize_max_side(frame.image, cfg.classifier.max_side_px))
    out.write_bytes(jpeg)
    print(f"camera: ok, saved {out}")
    for r in await classify_all(build_classifiers(cfg.classifier, os.environ), jpeg):
        answer = r.position.value if r.position else f"unavailable ({r.error})"
        print(f"{r.model_name}: {answer} in {r.latency_s:.1f}s")
    return 0


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crib-monitor-probe", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--out", default="probe.jpg")
    args = parser.parse_args(argv)
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
    raise SystemExit(asyncio.run(probe(Path(args.config), Path(args.out))))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add crib_monitor/evaluate.py crib_monitor/probe.py tests/test_evaluate.py
git commit -m "feat: evaluation report and probe tool"
```

---

### Task 15: Entry point, deployment, docs, integration test

**Files:**
- Create: `crib_monitor/main.py`, `deploy/crib-monitor.service`, `README.md`, `tests/test_main.py`, `tests/test_integration.py`
- Modify: `.gitignore` (append `.uv/`, `probe.jpg`, `full.jpg`)

**Interfaces:**
- Consumes: everything above.
- Produces: `build(cfg, secrets, env, client) -> tuple[Monitor, FastAPI]`, `async amain(cfg, secrets, env)`, CLI `crib-monitor --config --env-file`.

- [ ] **Step 1: Write the failing tests**

`tests/test_main.py`:

```python
import httpx
from fastapi.testclient import TestClient

from crib_monitor.config import load_secrets
from crib_monitor.main import build

ENV = {
    "TAPO_RTSP_URL": "rtsp://u:p@cam:554/stream1",
    "PUSHOVER_TOKEN": "tok",
    "PUSHOVER_USER": "usr",
    "HEALTHCHECKS_URL": "https://hc-ping.com/abc",
    "CONTROL_TOKEN": "s3cret",
    "OPENROUTER_API_KEY": "sk-or",
}


def test_build_wires_web_to_monitor(config, tmp_path):
    cfg = config.model_copy(update={"storage": config.storage.model_copy(update={"data_dir": tmp_path})})
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": 1})))
    monitor, app = build(cfg, load_secrets(ENV, cfg), ENV, client)
    with TestClient(app) as web:
        assert web.get("/").status_code == 403
        assert "Off" in web.get("/?t=s3cret").text
        web.post("/on?t=s3cret")
    assert (tmp_path / "state.json").exists()
```

`tests/test_integration.py`:

```python
"""Real ffmpeg against a local RTSP server. Run on the Ubuntu server: uv run pytest -m integration"""

import shutil
import subprocess
import time

import pytest

from crib_monitor.capture import Capture, ffmpeg_command
from crib_monitor.config import CribCrop

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("mediamtx")), reason="needs ffmpeg and mediamtx"),
]

URL = "rtsp://127.0.0.1:8554/test"


def publish() -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=5",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-f", "rtsp", "-rtsp_transport", "tcp", URL,
    ])


def wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_capture_survives_stream_restart(tmp_path):
    config = tmp_path / "mediamtx.yml"
    config.write_text("paths:\n  all_others:\n")
    server = subprocess.Popen(["mediamtx", str(config)])
    publisher = None
    capture = None
    try:
        time.sleep(1)
        publisher = publish()
        capture = Capture(
            ffmpeg_command(URL, CribCrop(x=0, y=0, w=320, h=180)), 320, 180,
            stall_s=2, startup_s=10, initial_backoff_s=0.5, max_backoff_s=2,
        )
        capture.start()
        assert wait_for(lambda: capture.latest() is not None, 20)
        publisher.kill()
        publisher.wait()
        seq = capture.latest().seq
        time.sleep(4)
        publisher = publish()
        assert wait_for(lambda: capture.latest().seq > seq + 3, 30)
        assert capture.restarts >= 1
    finally:
        if capture:
            capture.stop()
        for proc in (publisher, server):
            if proc and proc.poll() is None:
                proc.kill()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_main.py tests/test_integration.py -v`
Expected: `test_main.py` FAILS with `ModuleNotFoundError: No module named 'crib_monitor.main'`; `test_integration.py` is SKIPPED on a machine without ffmpeg/mediamtx.

- [ ] **Step 3: Implement `crib_monitor/main.py`**

```python
"""Service entry point: wire everything together and run the web page and monitor loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from .arming import Arming
from .capture import Capture, ffmpeg_command
from .classifier import build_classifiers
from .config import Config, ConfigError, Secrets, load_config, load_secrets
from .health import Health, Heartbeat
from .monitor import Monitor
from .notifier import Alerter, Pushover
from .schedule import Schedule
from .storage import Storage
from .web import create_app

log = logging.getLogger("crib_monitor")


def build(cfg: Config, secrets: Secrets, env: Mapping[str, str], client: httpx.AsyncClient) -> tuple[Monitor, FastAPI]:
    tz = ZoneInfo(cfg.schedule.timezone)
    storage = Storage(cfg.storage.data_dir, cfg.storage.retention_days, tz)
    arming = Arming(Schedule(cfg.schedule), cfg.arming, Path(cfg.storage.data_dir) / "state.json")
    crop = cfg.camera.crop

    def capture_factory() -> Capture:
        return Capture(
            ffmpeg_command(secrets.tapo_rtsp_url, crop), crop.w, crop.h,
            stall_s=cfg.camera.stream_timeout_s, max_backoff_s=cfg.camera.max_backoff_s,
        )

    monitor = Monitor(
        cfg=cfg,
        arming=arming,
        capture_factory=capture_factory,
        classifiers=build_classifiers(cfg.classifier, env),
        alerter=Alerter(Pushover(secrets.pushover_token, secrets.pushover_user, client), cfg.alerts),
        health=Health(cfg.health, [m.name for m in cfg.classifier.models], cfg.alerts.health_repeat_s),
        heartbeat=Heartbeat(secrets.healthchecks_url, client),
        storage=storage,
        now=lambda: datetime.now(UTC),
    )
    return monitor, create_app(monitor, storage, secrets.control_token, tz)


async def amain(cfg: Config, secrets: Secrets, env: Mapping[str, str]) -> None:
    async with httpx.AsyncClient() as client:
        monitor, app = build(cfg, secrets, env, client)
        server = uvicorn.Server(
            uvicorn.Config(app, host=cfg.web.host, port=cfg.web.port, access_log=False, log_level="info")
        )
        loop_task = asyncio.create_task(monitor.run_forever())
        try:
            await server.serve()
        finally:
            loop_task.cancel()
            monitor.shutdown()


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crib-monitor")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
    try:
        cfg = load_config(Path(args.config))
        secrets = load_secrets(os.environ, cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    asyncio.run(amain(cfg, secrets, os.environ))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -v`
Expected: all tests pass; `test_integration.py` skipped unless ffmpeg and mediamtx are installed.

- [ ] **Step 5: Write `deploy/crib-monitor.service`**

```ini
[Unit]
Description=Crib roll-over monitor
After=network-online.target
Wants=network-online.target

[Service]
User=cribmon
Group=cribmon
WorkingDirectory=/opt/crib-monitor
EnvironmentFile=/opt/crib-monitor/.env
ExecStart=/opt/crib-monitor/.venv/bin/crib-monitor --config /opt/crib-monitor/config.toml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 6: Write `README.md`**

````markdown
# Crib monitor

Watches a Tapo C210 over the crib during scheduled nights or manual nap sessions and sends a Pushover
emergency alert if he rolls onto his stomach. It is a backup to the Owlet, not a medical device.
Design: `docs/superpowers/specs/2026-09-24-crib-monitor-design.md`.

## One-time setup

**Camera.** In the Tapo app: camera → Settings → Advanced Settings → Camera Account; create a username and
password. The stream is `rtsp://USER:PASS@CAMERA_IP:554/stream1`. Give the camera a DHCP reservation.

**Pushover.** Install the Android app, note your user key, and create an application at pushover.net to get
an API token. In the Android app's settings, allow emergency-priority alerts to override Do Not Disturb and
silent mode.

**healthchecks.io.** Create a check with period 1 minute and grace 2 minutes. Add a Pushover integration
(high priority) so a dead server reaches your phone. Copy the ping URL.

**OpenRouter.** Create an API key. In OpenRouter's privacy settings, disallow providers that train on or
retain prompts. The config also sends `provider.data_collection = "deny"` on every request.

**Laptop (local model).** Install Ollama, set `OLLAMA_HOST=0.0.0.0` so the server can reach it, and pull the
vision model named in `config.toml` (for example `ollama pull qwen3-vl:30b`; check the exact tag in the
Ollama library). Disable suspend on lid close (on Ubuntu: `HandleLidSwitch=ignore` in
`/etc/systemd/logind.conf`, then `sudo systemctl restart systemd-logind`). Give it a DHCP reservation. It only
needs to be on during armed sessions.

## Server install (Ubuntu)

uv is installed system-wide, and the Python it downloads lives inside `/opt/crib-monitor` so the
`cribmon` user can reach it.

```bash
sudo apt install ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
sudo useradd --system --home /opt/crib-monitor --shell /usr/sbin/nologin cribmon
sudo git clone <this repo> /opt/crib-monitor
sudo chown -R cribmon:cribmon /opt/crib-monitor
cd /opt/crib-monitor
sudo -u cribmon env UV_PYTHON_INSTALL_DIR=/opt/crib-monitor/.uv/python UV_CACHE_DIR=/opt/crib-monitor/.uv/cache uv sync --frozen --no-dev
sudo -u cribmon cp config.example.toml config.toml
sudo -u cribmon install -m 600 /dev/null .env
```

Put these in `.env` (no quotes; URL-encode any special characters in the camera password):

```
TAPO_RTSP_URL=rtsp://USER:PASS@CAMERA_IP:554/stream1
PUSHOVER_TOKEN=...
PUSHOVER_USER=...
HEALTHCHECKS_URL=https://hc-ping.com/...
CONTROL_TOKEN=<long random string, e.g. from: openssl rand -hex 16>
OPENROUTER_API_KEY=...
```

**Crib crop.** Grab a full frame and find the crib rectangle in pixel coordinates (any image viewer that
shows cursor position works). Put it in `[camera.crop]`; all four values must be even.

```bash
set -a; . ./.env; set +a
ffmpeg -rtsp_transport tcp -i "$TAPO_RTSP_URL" -frames:v 1 full.jpg
```

Set `[web] host` to the server's LAN address, then check everything end to end:

```bash
sudo -u cribmon .venv/bin/crib-monitor-probe
sudo cp deploy/crib-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now crib-monitor
journalctl -u crib-monitor -f
```

Bookmark `http://SERVER_IP:8080/?t=<CONTROL_TOKEN>` on your phone's home screen and press **Test alert** to
confirm Pushover gets through Do Not Disturb.

## Before trusting it

1. First weekend: leave `shadow_mode = true`. Roll alerts arrive as normal-priority `[TEST]` messages.
2. Collect stomach frames safely: press **On**, then **Pause**, and do supervised tummy time in the crib while
   he is awake, once with the lights on and once with them off (infrared). Pause keeps checking every 30 s
   without alerting.
3. Label frames on the `/label` page.
4. Run `sudo -u cribmon .venv/bin/crib-monitor-eval` from `/opt/crib-monitor`. It must print `PASS` (every labeled stomach frame caught) before you set
   `shadow_mode = false` and restart the service. Re-run it when you change models.

## Tests

```bash
uv run pytest                    # unit tests
uv run pytest -m integration     # on the server, needs ffmpeg and mediamtx on PATH
```
````

- [ ] **Step 7: Run the full suite and commit**

Run: `uv run pytest -v`
Expected: all unit tests pass; integration test skipped on the dev machine.

```bash
printf '.uv/\nprobe.jpg\nfull.jpg\n' >> .gitignore
git add .gitignore crib_monitor/main.py deploy README.md tests/test_main.py tests/test_integration.py
git commit -m "feat: service entry point, systemd unit, setup docs, integration test"
```

- [ ] **Step 8: On-server verification (done by the user, not the implementer)**

These check the spec's "Risks to verify early". Record the results in the PR or a note.

1. `crib-monitor-probe` prints `camera: ok` while the Tapo app is showing the live feed on the phone at the same time.
2. `crib-monitor-probe` shows the local model's latency under 30 s. If it is slower, choose a smaller model or lower `max_side_px`.
3. `crib-monitor-probe` gets a valid answer from the cloud model through OpenRouter (no `unavailable (… response_format …)` error). If the chosen model rejects `json_schema`, pick another model; parsing already tolerates plain JSON.
4. `uv run pytest -m integration` passes on the server.
5. Unplug the camera during a manual session: a "Monitor blind" alert arrives within about 2 minutes, and "Camera recovered" after plugging it back in.
6. `sudo systemctl stop crib-monitor`: healthchecks.io alerts within about 3 minutes.
