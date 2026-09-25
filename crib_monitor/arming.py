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
