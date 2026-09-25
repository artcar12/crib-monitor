import re
from urllib.parse import parse_qs

import httpx
import pytest

from crib_monitor.config import AlertConfig
from crib_monitor.health import HealthEvent
from crib_monitor.notifier import Alerter, NotifyError, Pushover


def fields(request: httpx.Request) -> dict[str, str]:
    body = request.read()
    if request.headers["content-type"].startswith("multipart/form-data"):
        return {
            m.group(1).decode(): m.group(2).decode()
            for m in re.finditer(rb'name="([^"]+)"\r\n\r\n(.*?)\r\n--', body, re.S)
        }
    return {k: v[0] for k, v in parse_qs(body.decode()).items()}


class Recorder:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else httpx.Response(200, json={"status": 1})
        if isinstance(response, Exception):
            raise response
        return response


def pushover(recorder) -> Pushover:
    client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return Pushover("tok", "usr", client, backoff=(0, 0, 0))


async def test_send_normal():
    rec = Recorder()
    assert await pushover(rec).send("hello", priority=0) is None
    sent = fields(rec.requests[0])
    assert sent["token"] == "tok" and sent["user"] == "usr" and sent["message"] == "hello" and sent["priority"] == "0"
    assert "retry" not in sent


async def test_send_emergency_with_image_returns_receipt():
    rec = Recorder(httpx.Response(200, json={"status": 1, "receipt": "R1"}))
    receipt = await pushover(rec).send("roll", priority=2, image=b"\xff\xd8x", retry=60, expire=1800)
    assert receipt == "R1"
    sent = fields(rec.requests[0])
    assert sent["priority"] == "2" and sent["retry"] == "60" and sent["expire"] == "1800"
    assert b'filename="frame.jpg"' in rec.requests[0].content


async def test_send_retries_server_errors():
    rec = Recorder(httpx.Response(500), httpx.ConnectError("x"), httpx.Response(200, json={"status": 1}))
    await pushover(rec).send("hello")
    assert len(rec.requests) == 3


async def test_send_gives_up():
    rec = Recorder(*[httpx.Response(503)] * 4)
    with pytest.raises(NotifyError):
        await pushover(rec).send("hello")
    assert len(rec.requests) == 4


async def test_client_error_not_retried():
    rec = Recorder(httpx.Response(400, json={"status": 0, "errors": ["user invalid"]}))
    with pytest.raises(NotifyError, match="400"):
        await pushover(rec).send("hello")
    assert len(rec.requests) == 1


async def test_receipt_and_cancel():
    rec = Recorder(httpx.Response(200, json={"status": 1, "acknowledged": 1, "expired": 0}), httpx.Response(200, json={"status": 1}))
    p = pushover(rec)
    status = await p.receipt("R1")
    assert status.acknowledged and not status.expired
    assert "receipts/R1.json" in str(rec.requests[0].url)
    await p.cancel("R1")
    assert str(rec.requests[1].url).endswith("receipts/R1/cancel.json")


async def test_receipt_failure_raises():
    rec = Recorder(httpx.Response(500))
    with pytest.raises(NotifyError):
        await pushover(rec).receipt("R1")


async def test_alerter_live_mode():
    rec = Recorder(httpx.Response(200, json={"status": 1, "receipt": "R9"}))
    alerter = Alerter(pushover(rec), AlertConfig(shadow_mode=False))
    assert await alerter.stomach(b"img") == "R9"
    await alerter.no_view(b"img")
    sent = [fields(r) for r in rec.requests]
    assert sent[0]["priority"] == "2" and "stomach" in sent[0]["message"]
    assert sent[1]["priority"] == "1"


async def test_alerter_shadow_mode():
    rec = Recorder()
    alerter = Alerter(pushover(rec), AlertConfig(shadow_mode=True))
    assert await alerter.stomach(b"img") is None
    await alerter.no_view(b"img")
    await alerter.health(HealthEvent("blind", 1, "Monitor blind"))
    sent = [fields(r) for r in rec.requests]
    assert sent[0]["priority"] == "0" and sent[0]["message"].startswith("[TEST]")
    assert sent[1]["priority"] == "0" and sent[1]["message"].startswith("[TEST]")
    assert sent[2]["priority"] == "1" and sent[2]["message"] == "Monitor blind"


async def test_monitoring_started_marks():
    rec = Recorder()
    alerter = Alerter(pushover(rec), AlertConfig(shadow_mode=False))
    await alerter.monitoring_started(True, {"local": True, "cloud": False}, None)
    assert fields(rec.requests[0])["message"] == "Monitoring: camera ✓ local ✓ cloud ✗"
