import httpx
from fastapi.testclient import TestClient

from crib_monitor.config import load_secrets
from crib_monitor.main import build

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
