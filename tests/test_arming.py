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
