"""Pushover delivery and the wording of every message the monitor sends."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import httpx

from .config import AlertConfig
from .health import HealthEvent

log = logging.getLogger(__name__)

MESSAGES_URL = "https://api.pushover.net/1/messages.json"
RECEIPT_URL = "https://api.pushover.net/1/receipts/{receipt}.json"
CANCEL_URL = "https://api.pushover.net/1/receipts/{receipt}/cancel.json"


class NotifyError(Exception):
    pass


@dataclass(frozen=True)
class ReceiptStatus:
    acknowledged: bool
    expired: bool


class Pushover:
    def __init__(
        self,
        token: str,
        user: str,
        client: httpx.AsyncClient,
        backoff: Sequence[float] = (1, 2, 4),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._token = token
        self._user = user
        self._client = client
        self._backoff = list(backoff)
        self._sleep = sleep

    async def send(
        self,
        message: str,
        *,
        title: str = "Crib monitor",
        priority: int = 0,
        image: bytes | None = None,
        retry: int | None = None,
        expire: int | None = None,
    ) -> str | None:
        data = {"token": self._token, "user": self._user, "message": message, "title": title, "priority": str(priority)}
        if priority == 2:
            data["retry"] = str(retry)
            data["expire"] = str(expire)
        files = {"attachment": ("frame.jpg", image, "image/jpeg")} if image else None
        last_error = "not attempted"
        for delay in [None, *self._backoff]:
            if delay is not None:
                await self._sleep(delay)
            try:
                response = await self._client.post(MESSAGES_URL, data=data, files=files, timeout=15)
            except httpx.HTTPError as exc:
                last_error = type(exc).__name__
                log.warning("Pushover send attempt failed: %s", last_error)
                continue
            if 400 <= response.status_code < 500 and response.status_code != 429:
                raise NotifyError(f"Pushover rejected message: {response.status_code} {response.text}")
            if response.status_code == 200:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if body.get("status") == 1:
                    return body.get("receipt")
            last_error = f"{response.status_code} {response.text}"
            log.warning("Pushover send attempt failed: %s", last_error)
        raise NotifyError(f"Pushover send failed: {last_error}")

    async def receipt(self, receipt: str) -> ReceiptStatus:
        try:
            response = await self._client.get(RECEIPT_URL.format(receipt=receipt), params={"token": self._token}, timeout=15)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise NotifyError(f"receipt poll failed: HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise NotifyError(f"receipt poll failed: {type(exc).__name__}") from exc
        return ReceiptStatus(acknowledged=body.get("acknowledged") == 1, expired=body.get("expired") == 1)

    async def cancel(self, receipt: str) -> None:
        try:
            response = await self._client.post(CANCEL_URL.format(receipt=receipt), data={"token": self._token}, timeout=15)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise NotifyError(f"receipt cancel failed: HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise NotifyError(f"receipt cancel failed: {type(exc).__name__}") from exc


class Alerter:
    def __init__(self, pushover: Pushover, cfg: AlertConfig) -> None:
        self._p = pushover
        self._cfg = cfg

    def _tag(self, text: str) -> str:
        return f"[TEST] {text}" if self._cfg.shadow_mode else text

    async def stomach(self, image: bytes) -> str | None:
        text = "He may be on his stomach. Check the camera."
        if self._cfg.shadow_mode:
            await self._p.send(self._tag(text), priority=0, image=image)
            return None
        return await self._p.send(
            text, priority=2, image=image, retry=self._cfg.emergency_retry_s, expire=self._cfg.emergency_expire_s
        )

    async def no_view(self, image: bytes) -> None:
        await self._p.send(self._tag("Can't see him in the crib."), priority=0 if self._cfg.shadow_mode else 1, image=image)

    async def back_on_back(self) -> None:
        await self._p.send(self._tag("Back on his back."), priority=0)

    async def health(self, event: HealthEvent) -> None:
        await self._p.send(event.message, priority=event.priority)

    async def monitoring_started(self, camera_ok: bool, models: dict[str, bool | None], image: bytes | None) -> None:
        def mark(ok: bool | None) -> str:
            return "?" if ok is None else ("✓" if ok else "✗")

        parts = [f"camera {mark(camera_ok)}"] + [f"{name} {mark(ok)}" for name, ok in models.items()]
        text = "Monitoring: " + " ".join(parts) + (" (shadow mode)" if self._cfg.shadow_mode else "")
        await self._p.send(text, priority=0, image=image)

    async def manual_ended(self) -> None:
        await self._p.send("Nap monitoring ended at the time limit. Press On to keep watching.", priority=0)

    async def test(self) -> str | None:
        return await self._p.send("TEST alert from the crib monitor. Acknowledge to stop it.", priority=2, retry=60, expire=180)

    async def receipt(self, receipt: str) -> ReceiptStatus:
        return await self._p.receipt(receipt)

    async def cancel(self, receipt: str) -> None:
        await self._p.cancel(receipt)
