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
        assert cap.latest() is None or cap.latest().seq >= 1
        assert wait_for(lambda: cap.latest() is not None)
        frame = cap.latest()
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
