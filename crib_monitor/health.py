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
