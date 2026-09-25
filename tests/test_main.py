import asyncio
import contextlib
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient

from crib_monitor import main
from crib_monitor.config import load_secrets
from crib_monitor.main import build, loop_done, monitor_lifespan
from crib_monitor.monitor import Monitor
from crib_monitor.storage import Storage
from crib_monitor.web import create_app

NY = ZoneInfo("America/New_York")

ENV = {
    "TAPO_RTSP_URL": "rtsp://u:p@cam:554/stream1",
    "PUSHOVER_TOKEN": "tok",
    "PUSHOVER_USER": "usr",
    "HEALTHCHECKS_URL": "https://hc-ping.com/abc",
    "CONTROL_TOKEN": "s3cret",
    "OPENROUTER_API_KEY": "sk-or",
}


def test_build_wires_web_to_monitor(config, tmp_path):
    cfg = config.model_copy(update={"storage": config.storage.model_copy(update={"data_dir": tmp_path})})
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": 1})))
    monitor, app = build(cfg, load_secrets(ENV, cfg), ENV, client)
    with TestClient(app) as web:
        assert web.get("/").status_code == 403
        assert "Off" in web.get("/?t=s3cret").text
        web.post("/on?t=s3cret")
    assert (tmp_path / "state.json").exists()


class FakeMonitor:
    def __init__(self, crash=None):
        self.crash = crash
        self.started = self.cancelled = False
        self.shutdowns = 0

    async def run_forever(self):
        self.started = True
        if self.crash is not None:
            raise self.crash
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    def shutdown(self):
        self.shutdowns += 1


def _app(monitor, tmp_path, on_exit):
    return create_app(monitor, Storage(tmp_path, 30, NY), "s3cret", NY, lifespan=monitor_lifespan(monitor, on_exit))


def test_lifespan_runs_the_loop_and_shuts_down(tmp_path):
    monitor, exits = FakeMonitor(), []
    with TestClient(_app(monitor, tmp_path, lambda: exits.append(1))) as web:
        assert web.get("/").status_code == 403
        assert monitor.started and monitor.shutdowns == 0
    assert monitor.cancelled and monitor.shutdowns == 1
    assert exits == []                        # a cancelled loop is a normal shutdown


def test_lifespan_exits_when_the_loop_dies(tmp_path, caplog):
    monitor, exits = FakeMonitor(crash=RuntimeError("token=hunter2")), []
    with TestClient(_app(monitor, tmp_path, lambda: exits.append(1))) as web:
        web.get("/")
    assert exits == [1] and monitor.shutdowns == 1
    assert "RuntimeError" in caplog.text and "hunter2" not in caplog.text


async def test_loop_done_callback():
    exits = []

    async def boom():
        raise ValueError("x")

    async def forever():
        await asyncio.Event().wait()

    crashed = asyncio.create_task(boom())
    with contextlib.suppress(ValueError):
        await crashed
    loop_done(crashed, lambda: exits.append("crashed"))
    cancelled = asyncio.create_task(forever())
    await asyncio.sleep(0)
    cancelled.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await cancelled
    loop_done(cancelled, lambda: exits.append("cancelled"))
    returned = asyncio.create_task(asyncio.sleep(0))
    await returned
    loop_done(returned, lambda: exits.append("returned"))
    assert exits == ["crashed", "returned"]


def test_build_attaches_the_monitor_lifespan(config, tmp_path, monkeypatch):
    cfg = config.model_copy(update={"storage": config.storage.model_copy(update={"data_dir": tmp_path})})
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": 1})))
    monitor, app = build(cfg, load_secrets(ENV, cfg), ENV, client)
    fake = FakeMonitor()
    monkeypatch.setattr(monitor, "run_forever", fake.run_forever)
    monkeypatch.setattr(monitor, "shutdown", fake.shutdown)
    with TestClient(app):
        pass
    assert fake.started and fake.cancelled and fake.shutdowns == 1


async def test_amain_exits_nonzero_when_the_loop_dies(config, tmp_path, monkeypatch):
    cfg = config.model_copy(update={"storage": config.storage.model_copy(update={"data_dir": tmp_path})})

    async def boom(self, interval_s=1.0):
        raise RuntimeError("boom")

    class FakeServer:
        def __init__(self, config):
            self.app, self.should_exit = config.app, False

        async def serve(self):
            async with self.app.router.lifespan_context(self.app):
                for _ in range(200):
                    if self.should_exit:
                        return
                    await asyncio.sleep(0.01)

    monkeypatch.setattr(Monitor, "run_forever", boom)
    monkeypatch.setattr(main.uvicorn, "Server", FakeServer)
    assert await main.amain(cfg, load_secrets(ENV, cfg), ENV) == 1


ROOT = Path(__file__).resolve().parent.parent


class FakeLitellmFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves a stand-in `litellm` and records the environment at the moment it is imported."""

    def __init__(self):
        self.cost_map_at_import = "not imported"

    def find_spec(self, name, path, target=None):
        return importlib.util.spec_from_loader(name, self) if name == "litellm" else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.cost_map_at_import = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
        module.suppress_debug_info = False


@pytest.fixture
def run_main(tmp_path, monkeypatch):
    finder = FakeLitellmFinder()
    saved = sys.modules.pop("litellm", None)
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    seen = {}

    def run(code):
        async def fake_amain(cfg, secrets, env):
            seen["suppress_debug_info"] = sys.modules["litellm"].suppress_debug_info
            return code

        monkeypatch.setattr(main, "amain", fake_amain)
        main.run(["--config", str(ROOT / "config.example.toml"), "--env-file", str(tmp_path / "none")])
        return finder, seen

    yield run
    sys.modules.pop("litellm", None)
    if saved is not None:
        sys.modules["litellm"] = saved


def test_run_imports_litellm_at_startup_with_the_local_cost_map(run_main):
    finder, seen = run_main(0)
    assert finder.cost_map_at_import == "True"
    assert seen["suppress_debug_info"] is True


def test_run_exits_nonzero_when_amain_reports_a_dead_loop(run_main):
    with pytest.raises(SystemExit) as exit_:
        run_main(1)
    assert exit_.value.code == 1


def test_service_unit_uses_the_local_cost_map():
    unit = (ROOT / "deploy" / "crib-monitor.service").read_text().splitlines()
    assert "Environment=LITELLM_LOCAL_MODEL_COST_MAP=True" in unit
