"""Cheap local motion score used to decide when to ask the models."""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, pixel_delta: int = 25, width: int = 320) -> None:
        self._delta = pixel_delta
        self._width = width
        self._prev: np.ndarray | None = None

    def score(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w > self._width:
            gray = cv2.resize(gray, (self._width, max(1, round(h * self._width / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        prev, self._prev = self._prev, gray
        if prev is None or prev.shape != gray.shape:
            return 0.0
        diff = cv2.absdiff(prev, gray)
        return float(np.count_nonzero(diff > self._delta)) / diff.size
