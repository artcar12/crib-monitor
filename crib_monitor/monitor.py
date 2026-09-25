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
        self._receipt_deadline: datetime | None = None
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
        # Update bookkeeping and dispatch any actions (alerts) before touching storage: a
        # persistence failure below must never lose an already-decided alert, nor leave the
        # same frame re-classified on every subsequent tick.
        self._last_check = now
        self._last_checked_seq = frame.seq
        motion_peak = self._motion_peak
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
        try:
            self._storage.save_check(now, jpeg, {
                "motion": round(motion_peak, 4),
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
        except Exception as exc:
            log.warning("could not save check: %s", type(exc).__name__)

    async def _act(self, action: object, jpeg: bytes, now: datetime) -> None:
        if isinstance(action, StomachAlert):
            receipt = await self._send(self._alerter.stomach(jpeg))
            if receipt:
                self._receipt = receipt
                self._next_receipt_poll = now + timedelta(seconds=self._cfg.alerts.receipt_poll_s)
                self._receipt_deadline = now + timedelta(
                    seconds=self._cfg.alerts.emergency_expire_s + 2 * self._cfg.alerts.receipt_poll_s
                )
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
        if self._receipt is None:
            return
        if self._receipt_deadline is not None and now >= self._receipt_deadline:
            # The receipt has kept failing to poll (or never resolved) long enough that a baby
            # still on his stomach would otherwise never trigger a new alert. Force expiry.
            log.warning("receipt %s never resolved by its deadline; forcing expiry", self._receipt)
            self._receipt = None
            self._next_receipt_poll = None
            self._receipt_deadline = None
            self._engine.on_expired(now)
            return
        if self._next_receipt_poll is None or now < self._next_receipt_poll:
            return
        self._next_receipt_poll = now + timedelta(seconds=self._cfg.alerts.receipt_poll_s)
        try:
            status = await self._alerter.receipt(self._receipt)
        except NotifyError as exc:
            log.warning("%s", exc)
            return
        if status.acknowledged:
            self._receipt = None
            self._receipt_deadline = None
            self._engine.on_ack(now)
        elif status.expired:
            self._receipt = None
            self._receipt_deadline = None
            self._engine.on_expired(now)

    async def _cancel_receipt(self) -> None:
        if self._receipt is None:
            return
        receipt, self._receipt = self._receipt, None
        self._receipt_deadline = None
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
