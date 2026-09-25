"""Persistent ffmpeg RTSP reader that always holds the latest cropped frame."""

from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import CribCrop

log = logging.getLogger(__name__)

_CREDENTIAL_RE = re.compile(r"(\w+://)[^/@\s]+@")


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    seq: int


def ffmpeg_command(rtsp_url: str, crop: CribCrop, fps: int = 1) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vf", f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y},fps={fps}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]


class Capture:
    def __init__(
        self,
        cmd: list[str],
        width: int,
        height: int,
        *,
        stall_s: float = 5.0,
        startup_s: float = 15.0,
        initial_backoff_s: float = 1.0,
        max_backoff_s: float = 15.0,
    ) -> None:
        self._cmd = list(cmd)
        self._w = width
        self._h = height
        self._size = width * height * 3
        self._stall_s = stall_s
        self._startup_s = startup_s
        self._initial_backoff = initial_backoff_s
        self._max_backoff = max_backoff_s
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self.restarts = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._supervise, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                log.warning("capture supervisor thread did not exit within timeout")
            else:
                self._thread = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def _supervise(self) -> None:
        backoff = self._initial_backoff
        while not self._stop.is_set():
            got_frames = self._run_once()
            if self._stop.is_set():
                break
            self.restarts += 1
            if got_frames:
                backoff = self._initial_backoff
            log.warning("capture stopped; restarting in %.1fs", backoff)
            self._stop.wait(backoff)
            if not got_frames:
                backoff = min(backoff * 2, self._max_backoff)

    def _run_once(self) -> bool:
        try:
            proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.error("cannot start capture command: %s", exc)
            return False
        self._proc = proc
        got = threading.Event()
        last = [time.monotonic()]
        reader = threading.Thread(target=self._read, args=(proc, got, last), name="capture-reader", daemon=True)
        stderr_reader = threading.Thread(
            target=self._drain_stderr, args=(proc,), name="capture-stderr", daemon=True
        )
        reader.start()
        stderr_reader.start()
        try:
            while not self._stop.is_set() and proc.poll() is None:
                limit = self._stall_s if got.is_set() else self._startup_s
                if time.monotonic() - last[0] > limit:
                    log.warning("no frame for %.1fs; restarting capture", limit)
                    break
                self._stop.wait(0.1)
        finally:
            self._kill()
            reader.join(timeout=2)
            stderr_reader.join(timeout=2)
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
            self._proc = None
        return got.is_set()

    def _read(self, proc: subprocess.Popen[bytes], got: threading.Event, last: list[float]) -> None:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(self._size)
            if not data or len(data) < self._size:
                return
            image = np.frombuffer(data, dtype=np.uint8).reshape(self._h, self._w, 3)
            with self._lock:
                self._seq += 1
                self._latest = Frame(image=image, seq=self._seq)
            last[0] = time.monotonic()
            got.set()

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        # ffmpeg logs the RTSP URL (which embeds the camera password) on failures such as a
        # bad crop or a 401. Never forward it verbatim; redact credentials before logging.
        assert proc.stderr is not None
        for raw_line in proc.stderr:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            if line:
                log.warning("ffmpeg: %s", _CREDENTIAL_RE.sub(r"\1***@", line))

    def _kill(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.error("capture process did not exit after kill")
