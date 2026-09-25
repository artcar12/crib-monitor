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
