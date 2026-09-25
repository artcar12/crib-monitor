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
    # Keep looking closely after a cancelled confirmation: a single misread BACK must not
    # drop the next look to motion-or-max-interval.
    assert e.state is State.WATCH
    assert e.force_interval_s == 30


def test_false_positive_relaxes_after_back_streak():
    e = Engine()
    feed(e, [ST, BK])
    feed(e, [BK])
    assert e.state is State.WATCH
    feed(e, [BK])
    assert e.state is State.MONITORING


def test_stomach_after_false_positive_confirms_again():
    e = Engine()
    actions, _ = feed(e, [ST, BK, ST, ST])
    assert names(actions) == ["FalsePositive", "StomachAlert"]
    assert e.state is State.ALERTED


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


def test_stomach_from_watch_confirms_then_alerts():
    e = Engine()
    feed(e, [SD])
    assert e.state is State.WATCH
    actions, _ = feed(e, [ST])
    assert actions == []
    assert e.state is State.CONFIRMING
    actions, _ = feed(e, [ST])
    assert names(actions) == ["StomachAlert"]
    assert e.state is State.ALERTED


def test_stomach_from_no_view_confirms_then_alerts():
    e = Engine()
    feed(e, [NV])
    assert e.state is State.NO_VIEW
    actions, _ = feed(e, [ST])
    assert actions == []
    assert e.state is State.CONFIRMING
    actions, _ = feed(e, [ST])
    assert names(actions) == ["StomachAlert"]
    assert e.state is State.ALERTED


def test_watch_then_no_view_enters_no_view():
    e = Engine()
    feed(e, [SD])
    assert e.state is State.WATCH
    feed(e, [NV])
    assert e.state is State.NO_VIEW


def test_confirming_boundary_alert_on_fourth_stomach():
    e = Engine()
    actions, _ = feed(e, [ST, SD, NV, ST])
    assert names(actions) == ["StomachAlert"]
    assert e.state is State.ALERTED


def test_confirming_boundary_falls_to_watch_without_alert():
    e = Engine()
    actions, _ = feed(e, [ST, SD, NV, SD])
    assert actions == []
    assert e.state is State.WATCH


def test_no_view_alerts_again_after_recovery():
    e = Engine()
    actions1, t = feed(e, [NV, NV, NV, NV], step_s=20)
    assert names(actions1) == ["NoViewAlert"]
    _, t = feed(e, [BK], start=t)
    assert e.state is State.MONITORING
    actions2, _ = feed(e, [NV, NV, NV, NV], start=t, step_s=20)
    assert names(actions2) == ["NoViewAlert"]


def test_alerted_non_consecutive_backs_no_back_on_back():
    e = Engine()
    _, t = feed(e, [ST, ST])
    actions, _ = feed(e, [BK, ST, BK], start=t)
    assert actions == []
    assert e.state is State.ALERTED


def test_no_third_alert_until_next_ack():
    e = Engine(ack_suppress_s=180)
    _, t = feed(e, [ST, ST])
    e.on_ack(t)
    actions, t = feed(e, [ST], start=t + timedelta(seconds=180))
    assert names(actions) == ["StomachAlert"]
    actions, _ = feed(e, [ST], start=t + timedelta(seconds=1000))
    assert actions == []


def test_watch_back_streak_with_failed_interleaved():
    e = Engine()
    feed(e, [SD])
    assert e.state is State.WATCH
    actions, _ = feed(e, [BK, FL, BK])
    assert actions == []
    assert e.state is State.MONITORING


def test_no_view_streak_with_failed_interleaved_still_alerts():
    e = Engine()
    actions, _ = feed(e, [NV, FL, NV, FL, NV, FL, NV], step_s=20)
    assert names(actions) == ["NoViewAlert"]


def test_alerted_failed_interleaved_still_realerts():
    e = Engine(ack_suppress_s=180)
    _, t = feed(e, [ST, ST])
    e.on_ack(t)
    actions, _ = feed(e, [FL, ST], start=t + timedelta(seconds=180), step_s=10)
    assert names(actions) == ["StomachAlert"]


def test_repeated_ack_does_not_push_suppression_out():
    e = Engine(ack_suppress_s=180)
    _, t = feed(e, [ST, ST])
    e.on_ack(t)
    e.on_ack(t + timedelta(seconds=170))  # spurious repeat ack must be a no-op
    actions, _ = feed(e, [ST], start=t + timedelta(seconds=180))
    assert names(actions) == ["StomachAlert"]
