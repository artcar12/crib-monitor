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
