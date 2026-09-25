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
