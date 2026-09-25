from datetime import UTC, datetime, timedelta

import httpx

from crib_monitor.classifier import ClassifyResult
from crib_monitor.config import HealthConfig
from crib_monitor.health import Health, Heartbeat
from crib_monitor.labels import Position

T0 = datetime(2026, 9, 26, 3, 0, tzinfo=UTC)


def make() -> Health:
    h = Health(HealthConfig(), ["local", "cloud"], repeat_s=900)
    h.reset(T0)
    return h


def res(local_ok: bool, cloud_ok: bool):
    return [
        ClassifyResult("local", Position.BACK if local_ok else None, 0.1),
        ClassifyResult("cloud", Position.BACK if cloud_ok else None, 0.1),
    ]


def at(s: float) -> datetime:
    return T0 + timedelta(seconds=s)


def test_blind_after_no_frames():
    h = make()
    assert h.tick(at(119)) == []
    events = h.tick(at(120))
    assert [(e.key, e.priority, e.recovered) for e in events] == [("blind", 1, False)]
    assert h.tick(at(500)) == []
    assert [e.key for e in h.tick(at(1020))] == ["blind"]          # repeats after 900 s
    h.frame(at(1030))
    events = h.tick(at(1030))
    assert [(e.key, e.priority, e.recovered) for e in events] == [("blind", 0, True)]
    assert h.active == []


def test_frames_keep_it_quiet():
    h = make()
    for s in range(0, 600, 30):
        h.frame(at(s))
        assert h.tick(at(s)) == []


def test_one_model_down_and_recovered():
    h = make()
    assert h.results(res(False, True), at(0)) == []
    assert h.results(res(False, True), at(10)) == []
    events = h.results(res(False, True), at(20))
    assert [(e.key, e.priority) for e in events] == [("model_down:local", 0)]
    assert h.results(res(False, True), at(30)) == []
    events = h.results(res(True, True), at(40))
    assert [(e.key, e.recovered) for e in events] == [("model_down:local", True)]


def test_both_down_is_high_priority_and_repeats():
    h = make()
    events = []
    for s in (0, 10, 20):
        events += h.results(res(False, False), at(s))
    assert [(e.key, e.priority) for e in events] == [("no_detectors", 1)]
    h.frame(at(900))
    assert [e.key for e in h.tick(at(920))] == ["no_detectors"]
    events = h.results(res(True, False), at(930))
    assert [(e.key, e.recovered) for e in events] == [("no_detectors", True), ("model_down:cloud", False)]


def test_escalation_from_one_to_both_does_not_claim_recovery():
    h = make()
    for s in (0, 10, 20):
        h.results(res(False, True), at(s))
    events = []
    for s in (30, 40, 50):
        events += h.results(res(False, False), at(s))
    assert [(e.key, e.recovered) for e in events] == [("no_detectors", False)]


def test_reset_clears_everything():
    h = make()
    h.tick(at(200))
    h.reset(at(300))
    assert h.active == []
    assert h.tick(at(310)) == []


async def test_heartbeat_pings_and_fails():
    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: seen.append(str(r.url)) or httpx.Response(200)))
    hb = Heartbeat("https://hc-ping.com/abc", client)
    await hb.ping()
    await hb.ping(ok=False)
    assert seen == ["https://hc-ping.com/abc", "https://hc-ping.com/abc/fail"]


async def test_heartbeat_swallows_network_errors():
    def boom(request):
        raise httpx.ConnectError("down")

    hb = Heartbeat("https://hc-ping.com/abc", httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    await hb.ping()


async def test_heartbeat_swallows_non_http_errors():
    def boom(request):
        raise httpx.InvalidURL("bad url")

    hb = Heartbeat("https://hc-ping.com/abc", httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    await hb.ping()
