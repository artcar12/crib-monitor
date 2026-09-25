from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from crib_monitor.arming import ArmingStatus
from crib_monitor.monitor import Snapshot
from crib_monitor.storage import Storage
from crib_monitor.web import create_app, describe

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 26, 3, 0, tzinfo=NY)


class FakeController:
    def __init__(self):
        self.calls = []
        self.snap = Snapshot(status=None, state="monitoring", last_check=None)
        self.jpeg = b"\xff\xd8frame"

    def on(self):
        self.calls.append("on")

    def off(self):
        self.calls.append("off")

    def pause(self):
        self.calls.append("pause")
        return True

    def resume(self):
        self.calls.append("resume")

    async def test_alert(self):
        self.calls.append("test")
        return True

    def snapshot(self):
        return self.snap

    def latest_jpeg(self):
        return self.jpeg


@pytest.fixture
def setup(tmp_path):
    ctrl = FakeController()
    storage = Storage(tmp_path, 30, NY)
    client = TestClient(create_app(ctrl, storage, "s3cret", NY))
    return ctrl, storage, client


def test_token_required(setup):
    _, _, client = setup
    assert client.get("/").status_code == 403
    assert client.get("/?t=wrong").status_code == 403
    assert client.post("/on").status_code == 403
    assert client.get("/?t=s3cret").status_code == 200


@pytest.mark.parametrize("action", ["on", "off", "pause", "resume", "test"])
def test_buttons_call_controller(setup, action):
    ctrl, _, client = setup
    response = client.post(f"/{action}?t=s3cret", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/?t=s3cret"
    assert ctrl.calls == [action]


def test_status_page_shows_state_and_shadow(setup):
    ctrl, _, client = setup
    ctrl.snap = Snapshot(
        status=ArmingStatus(armed=True, paused=False, source="schedule", window_end=datetime(2026, 9, 26, 7, 0, tzinfo=NY)),
        state="monitoring", last_check=T0, health=["model_down:local"], shadow_mode=True,
    )
    page = client.get("/?t=s3cret").text
    assert "On (schedule) until 07:00" in page
    assert "Shadow mode" in page
    assert "model_down:local" in page


def test_frame_endpoint(setup):
    ctrl, _, client = setup
    assert client.get("/frame.jpg?t=s3cret").content == b"\xff\xd8frame"
    ctrl.jpeg = None
    assert client.get("/frame.jpg?t=s3cret").status_code == 404


def test_label_flow(setup):
    _, storage, client = setup
    rel = storage.save_check(T0, b"\xff\xd8a", {})
    page = client.get("/label?t=s3cret").text
    assert rel in page and "1 left" in page
    assert client.get(f"/frames/{rel}?t=s3cret").content == b"\xff\xd8a"
    response = client.post("/label?t=s3cret", data={"image": rel, "label": "stomach"}, follow_redirects=False)
    assert response.status_code == 303
    assert storage.unlabeled() == []
    assert "Nothing to label" in client.get("/label?t=s3cret").text


def test_label_rejects_bad_input(setup):
    _, storage, client = setup
    rel = storage.save_check(T0, b"a", {})
    assert client.post("/label?t=s3cret", data={"image": rel, "label": "nope"}).status_code == 400
    assert client.post("/label?t=s3cret", data={"image": "../state.json", "label": "back"}).status_code == 400
    assert client.get("/frames/../state.json?t=s3cret").status_code == 404


def test_describe():
    off = Snapshot(status=ArmingStatus(armed=False, paused=False, source=None), state="monitoring", last_check=None)
    assert describe(off, NY) == "Off"
    paused = Snapshot(
        status=ArmingStatus(armed=True, paused=True, source="manual", paused_until=datetime(2026, 9, 26, 3, 45, tzinfo=NY)),
        state="monitoring", last_check=None,
    )
    assert describe(paused, NY) == "Paused until 03:45"
    nap = Snapshot(
        status=ArmingStatus(armed=True, paused=False, source="manual", manual_until=datetime(2026, 9, 26, 17, 0, tzinfo=NY)),
        state="monitoring", last_check=None,
    )
    assert describe(nap, NY) == "On (nap) until 17:00"
