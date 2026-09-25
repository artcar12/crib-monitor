"""Real ffmpeg against a local RTSP server. Run on the Ubuntu server: uv run pytest -m integration"""

import shutil
import subprocess
import time

import pytest

from crib_monitor.capture import Capture, ffmpeg_command
from crib_monitor.config import CribCrop

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("mediamtx")), reason="needs ffmpeg and mediamtx"),
]

URL = "rtsp://127.0.0.1:8554/test"


def publish() -> subprocess.Popen:
    return subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-re",
        "-f", "lavfi", "-i", "testsrc=size=640x360:rate=5",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-f", "rtsp", "-rtsp_transport", "tcp", URL,
    ])


def wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def test_capture_survives_stream_restart(tmp_path):
    config = tmp_path / "mediamtx.yml"
    config.write_text("paths:\n  all_others:\n")
    server = subprocess.Popen(["mediamtx", str(config)])
    publisher = None
    capture = None
    try:
        time.sleep(1)
        publisher = publish()
        capture = Capture(
            ffmpeg_command(URL, CribCrop(x=0, y=0, w=320, h=180)), 320, 180,
            stall_s=2, startup_s=10, initial_backoff_s=0.5, max_backoff_s=2,
        )
        capture.start()
        assert wait_for(lambda: capture.latest() is not None, 20)
        publisher.kill()
        publisher.wait()
        seq = capture.latest().seq
        time.sleep(4)
        publisher = publish()
        assert wait_for(lambda: capture.latest().seq > seq + 3, 30)
        assert capture.restarts >= 1
    finally:
        if capture:
            capture.stop()
        for proc in (publisher, server):
            if proc and proc.poll() is None:
                proc.kill()
