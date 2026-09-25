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
        if self.state is State.ALERTED and self._awaiting_ack:
            self._awaiting_ack = False
            self._suppress_until = now + self._ack_suppress

    def on_expired(self, now: datetime) -> None:
        if self.state is State.ALERTED and self._awaiting_ack:
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
            self._set(State.WATCH)
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
