import cv2
import numpy as np

from crib_monitor.imaging import encode_jpeg, resize_max_side
from crib_monitor.motion import MotionDetector


def frame(value=0):
    return np.full((360, 640, 3), value, dtype=np.uint8)


def test_resize_max_side():
    assert resize_max_side(frame(), 320).shape == (180, 320, 3)
    assert resize_max_side(frame(), 1000).shape == (360, 640, 3)


def test_encode_jpeg_roundtrip():
    data = encode_jpeg(frame(128))
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)


def test_first_frame_scores_zero():
    assert MotionDetector().score(frame(100)) == 0.0


def test_identical_frames_score_zero():
    m = MotionDetector()
    m.score(frame(100))
    assert m.score(frame(100)) == 0.0


def test_sensor_noise_is_ignored():
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (360, 640, 3)).astype(np.int16)
    noisy = np.clip(base + rng.integers(-5, 6, base.shape), 0, 255).astype(np.uint8)
    m = MotionDetector()
    m.score(base.astype(np.uint8))
    assert m.score(noisy) < 0.02


def test_moving_region_is_detected():
    m = MotionDetector()
    m.score(frame(0))
    moved = frame(0)
    moved[100:200, 100:200] = 200
    assert m.score(moved) > 0.02


def test_light_change_scores_high():
    m = MotionDetector()
    m.score(frame(20))
    assert m.score(frame(220)) > 0.9
