import logging
import sys
import time

from crib_monitor.capture import Capture, ffmpeg_command
from crib_monitor.config import CribCrop

PRODUCER = """
import sys, time
w, h, count, value = (int(a) for a in sys.argv[1:5])
sleep_after = float(sys.argv[5])
frame = bytes([value]) * (w * h * 3)
for _ in range(count):
    sys.stdout.buffer.write(frame)
    sys.stdout.buffer.flush()
    time.sleep(0.05)
time.sleep(sleep_after)
"""


def producer(count, sleep_after, value=7, w=8, h=4):
    return [sys.executable, "-c", PRODUCER, str(w), str(h), str(count), str(value), str(sleep_after)]


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_ffmpeg_command():
    cmd = ffmpeg_command("rtsp://u:p@cam:554/stream1", CribCrop(x=10, y=20, w=640, h=360))
    assert cmd[0] == "ffmpeg"
    assert cmd.index("-rtsp_transport") < cmd.index("-i")
    assert "crop=640:360:10:20,fps=1" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "bgr24"


def test_frames_are_decoded():
    cap = Capture(producer(3, 30), 8, 4)
    cap.start()
    try:
        assert wait_for(lambda: cap.latest() is not None)
        frame = cap.latest()
        assert frame.seq == 1
        assert frame.image.shape == (4, 8, 3)
        assert int(frame.image[0, 0, 0]) == 7
    finally:
        cap.stop()


def test_stalled_stream_is_restarted():
    cap = Capture(producer(1, 30), 8, 4, stall_s=0.3, startup_s=2, initial_backoff_s=0.05)
    cap.start()
    try:
        assert wait_for(lambda: cap.latest() is not None and cap.latest().seq >= 3, timeout=8)
        assert cap.restarts >= 2
    finally:
        cap.stop()


def test_failing_command_backs_off():
    cap = Capture([sys.executable, "-c", "import sys; sys.exit(1)"], 8, 4, initial_backoff_s=0.1, max_backoff_s=0.4)
    cap.start()
    time.sleep(1.6)
    cap.stop()
    assert 2 <= cap.restarts <= 8
    assert cap.latest() is None


def test_stop_is_prompt():
    cap = Capture(producer(100000, 0), 8, 4)
    cap.start()
    assert wait_for(lambda: cap.latest() is not None)
    start = time.monotonic()
    cap.stop()
    assert time.monotonic() - start < 3


def test_stderr_credentials_are_redacted(caplog, capfd):
    cmd = [
        sys.executable, "-c",
        "import sys; sys.stderr.write('rtsp://u:secret@cam/x: 401 unauthorized\\n'); "
        "sys.stderr.flush(); sys.exit(1)",
    ]
    cap = Capture(cmd, 8, 4, initial_backoff_s=0.1, max_backoff_s=0.1)
    with caplog.at_level(logging.WARNING):
        cap.start()
        try:
            assert wait_for(lambda: cap.restarts >= 1)
            time.sleep(0.2)
        finally:
            cap.stop()
    assert "secret" not in caplog.text
    out, err = capfd.readouterr()
    assert "secret" not in out
    assert "secret" not in err
    assert "***@" in caplog.text


def test_stop_is_prompt_during_backoff():
    cap = Capture(
        [sys.executable, "-c", "import sys; sys.exit(1)"], 8, 4, initial_backoff_s=10, max_backoff_s=10
    )
    cap.start()
    try:
        assert wait_for(lambda: cap.restarts >= 1)
        start = time.monotonic()
    finally:
        cap.stop()
    assert time.monotonic() - start < 1


def test_stop_is_prompt_during_startup_wait():
    cap = Capture(producer(0, 30), 8, 4, startup_s=10)
    cap.start()
    try:
        assert wait_for(lambda: cap._proc is not None)
        time.sleep(0.2)
        start = time.monotonic()
    finally:
        cap.stop()
    assert time.monotonic() - start < 3


class _NeverExits:
    def join(self, timeout=None):
        pass

    def is_alive(self):
        return True


def test_stop_keeps_thread_ref_when_supervisor_does_not_exit():
    cap = Capture(producer(1, 0), 8, 4)
    cap._thread = _NeverExits()
    cap.stop()
    assert cap._thread is not None
    cap.start()
    assert isinstance(cap._thread, _NeverExits)
