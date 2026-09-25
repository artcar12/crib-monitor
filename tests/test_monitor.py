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
        self.health_events = []
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
        self.health_events.append(event)
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


class FailingStorage:
    """A Storage stand-in whose save_check always raises, to test that a persistence
    failure never swallows an already-decided alert."""

    def __init__(self):
        self.calls = 0

    def cleanup(self, now):
        pass

    def save_check(self, now, jpeg, record):
        self.calls += 1
        raise OSError("disk full")


class Rig:
    def __init__(self, config, tmp_path, local=(), cloud=(), local_default=B, cloud_default=B, storage=None):
        cfg = config.model_copy(update={
            "alerts": config.alerts.model_copy(update={"shadow_mode": False}),
            "storage": config.storage.model_copy(update={"data_dir": tmp_path}),
        })
        self.cfg = cfg
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
            storage=storage if storage is not None else Storage(tmp_path, cfg.storage.retention_days, NY),
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


async def test_receipt_deadline_forces_new_alert_when_polling_keeps_failing(rig):
    # alerter.receipt() always raises NotifyError; the models keep reporting stomach.
    # Before the receipt deadline, only one stomach alert should have gone out. Once the
    # deadline passes, on_expired must fire so a still-stomach baby gets a second alert.
    r = rig(local_default=S, cloud_default=S)
    r.alerter.fail = {"receipt"}
    r.monitor.on()
    await r.step()                        # self-test check: stomach -> confirming
    await r.step(10)                      # confirmation: stomach -> alert, receipt scheduled
    assert r.alerter.calls.count("stomach") == 1

    deadline_s = r.cfg.alerts.emergency_expire_s + 2 * r.cfg.alerts.receipt_poll_s
    poll_s = r.cfg.alerts.receipt_poll_s

    # Poll repeatedly (each poll raises NotifyError) but stay strictly short of the deadline.
    elapsed = 0.0
    while elapsed + poll_s < deadline_s:
        await r.step(poll_s)
        elapsed += poll_s
    assert "receipt" in r.alerter.calls
    assert r.alerter.calls.count("stomach") == 1     # no second alert just before the deadline

    # Cross the deadline by a few seconds only: on_expired should fire this tick, and the
    # following forced check (still reading stomach) should send exactly one more alert.
    await r.step(deadline_s - elapsed + 5)
    assert r.alerter.calls.count("stomach") == 2


async def test_save_failure_does_not_lose_the_alert(rig):
    # If persisting the check to disk fails (e.g. OSError, disk full), the decision already
    # made (and any alert it produced) must not be lost, and bookkeeping for the checked frame
    # must still update so the same frame isn't re-classified forever.
    storage = FailingStorage()
    r = rig(local=[S, S], storage=storage)
    r.monitor.on()
    await r.step()                        # self-test check: stomach -> confirming
    await r.step(10)                      # confirmation: stomach -> alert
    assert "stomach" in r.alerter.calls
    assert storage.calls == 2             # save_check was attempted (and raised) both times

    checks_before = r.checks
    await r.monitor.tick()                # same frame seq, no new frame pushed
    assert r.checks == checks_before      # must not re-run the models on the same frame


async def _run_until(r, predicate, limit_s):
    """Step one second at a time with still frames; return seconds taken, or None."""
    for t in range(1, limit_s + 1):
        await r.step(1)
        if predicate():
            return t
    return None


def _health_keys(r):
    return [c.key for c in r.alerter.health_events]


async def test_both_detectors_down_alerts_within_two_minutes_when_still(rig):
    r = rig()
    r.monitor.on()
    await r.step()                            # self-test ok (back)
    r.local.default = r.cloud.default = None
    # The monitor only learns the models are down when it next checks (max interval, no motion).
    assert await _run_until(r, lambda: r.checks == 2, 300) is not None
    took = await _run_until(r, lambda: "no_detectors" in _health_keys(r), 120)
    assert took is not None, "no 'no detectors' alert within 120 s of the first failed check"


async def test_both_detectors_down_from_arming_alerts_within_two_minutes(rig):
    r = rig(local_default=None, cloud_default=None)
    r.monitor.on()
    await r.step()
    assert await _run_until(r, lambda: "no_detectors" in _health_keys(r), 120) is not None


async def test_pause_ended_by_stomach_sighting_keeps_that_reading(rig):
    r = rig(local=[B, NV, S, S], cloud=[B, NV, S, S], local_default=S, cloud_default=S)
    r.monitor.on()
    await r.step()                            # self-test: back
    assert r.monitor.pause()
    await r.step(30)                          # not visible: he has left the view
    await r.step(30)                          # stomach sighting 1
    await r.step(30)                          # stomach sighting 2 -> pause ends
    assert r.monitor.snapshot().state == "confirming"
    assert not r.monitor.snapshot().status.paused
    assert await _run_until(r, lambda: "stomach" in r.alerter.calls, 60) is not None


async def test_pause_timeout_checks_on_the_next_tick(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    assert r.monitor.pause()
    pause_s = r.cfg.arming.pause_minutes * 60
    elapsed = 0
    while elapsed + 30 < pause_s:
        await r.step(30)
        elapsed += 30
    n = r.checks
    await r.step(pause_s - elapsed + 1)       # pause times out on this tick
    assert not r.monitor.snapshot().status.paused
    assert r.checks == n + 1


async def test_resume_checks_on_the_next_tick(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    assert r.monitor.pause()
    await r.step(30)
    n = r.checks
    r.monitor.resume()
    await r.step(1)
    assert r.checks == n + 1


async def test_schedule_end_leaves_unacked_emergency_running(rig):
    r = rig(local=[S, S])
    r.clock.now = datetime(2026, 9, 26, 6, 58, tzinfo=NY)    # Friday night's window ends at 07:00
    await r.step()                            # armed by the schedule; self-test: stomach -> confirming
    await r.step(10)                          # confirmed -> emergency alert
    assert r.alerter.calls.count("stomach") == 1
    await r.step(15)                          # receipt polled: not acknowledged
    r.clock.now = datetime(2026, 9, 26, 7, 0, 1, tzinfo=NY)
    await r.monitor.tick()                    # window over: disarmed automatically
    assert not r.monitor.snapshot().status.armed
    assert "cancel" not in r.alerter.calls
    r.clock.advance(60)
    await r.monitor.tick()
    assert "cancel" not in r.alerter.calls


async def test_manual_cap_leaves_unacked_emergency_running(rig):
    r = rig(local_default=S, cloud_default=S)
    r.monitor.on()
    await r.step()
    await r.step(10)
    assert r.alerter.calls.count("stomach") == 1
    r.clock.now = r.monitor.snapshot().status.manual_until
    await r.monitor.tick()                    # 4 h cap: disarmed automatically
    assert "manual_ended" in r.alerter.calls
    assert "cancel" not in r.alerter.calls


async def test_manual_off_cancels_unacked_emergency(rig):
    r = rig(local=[S, S])
    r.monitor.on()
    await r.step()
    await r.step(10)
    assert r.alerter.calls.count("stomach") == 1
    r.monitor.off()
    await r.step(1)
    assert r.alerter.args["cancel"] == ("R1",)


async def test_failed_send_fails_one_heartbeat_then_goes_green(rig):
    r = rig()
    r.monitor.on()
    await r.step()
    r.alerter.fail.add("manual_ended")
    r.clock.advance(4 * 3600)
    await r.monitor.tick()                    # manual cap: the "nap monitoring ended" send fails
    assert r.heartbeat.pings[-1] is False
    r.clock.advance(61)
    await r.monitor.tick()
    assert r.heartbeat.pings[-1] is True      # the failure was reported; don't latch /fail
    for _ in range(3):
        r.clock.advance(61)
        await r.monitor.tick()
    assert r.heartbeat.pings[-3:] == [True] * 3


async def test_each_failure_episode_fails_the_heartbeat_again(rig):
    r = rig(local_default=S, cloud_default=S)
    r.alerter.fail.add("stomach")
    r.monitor.on()
    await r.step()
    await r.step(10)                          # first emergency send fails
    assert r.heartbeat.pings[-1] is False
    await r.step(61)
    assert r.heartbeat.pings[-1] is True
    n = r.alerter.calls.count("stomach")
    assert await _run_until(r, lambda: r.alerter.calls.count("stomach") > n, 300) is not None
    assert r.heartbeat.pings[-1] is False     # the re-alert failed too: /fail again


async def test_unexpected_send_error_is_a_failed_send_and_re_alerts(rig, caplog):
    r = rig(local_default=S, cloud_default=S)
    sent = []

    async def stomach(image):
        sent.append(image)
        if len(sent) == 1:
            raise RuntimeError("client closed, token=hunter2")
        return "R2"

    r.alerter.stomach = stomach
    r.monitor.on()
    await r.step()
    await r.step(10)                          # first emergency send raises a non-NotifyError
    assert len(sent) == 1
    assert r.heartbeat.pings[-1] is False
    assert await _run_until(r, lambda: len(sent) == 2, 300) is not None
    assert "RuntimeError" in caplog.text
    assert "hunter2" not in caplog.text


async def test_emergency_is_sent_before_health_messages_from_the_same_check(rig):
    r = rig(local_default=None, cloud=[B, S, S])
    r.monitor.on()
    await r.step()                            # back; local failure 1
    await r.step(30)                          # stomach -> confirming; local failure 2
    await r.step(10)                          # confirmed; local failure 3 -> "one detector"
    assert "stomach" in r.alerter.calls and "health" in r.alerter.calls
    assert r.alerter.calls.index("stomach") < r.alerter.calls.index("health")
