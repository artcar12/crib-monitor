"""Grab one frame from the camera and run each model on it once."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .capture import Capture, ffmpeg_command
from .classifier import build_classifiers, classify_all
from .config import ConfigError, load_config
from .imaging import encode_jpeg, resize_max_side


async def probe(config_path: Path, out: Path) -> int:
    cfg = load_config(config_path)
    url = os.environ.get("TAPO_RTSP_URL")
    if not url:
        raise ConfigError("TAPO_RTSP_URL is not set")
    crop = cfg.camera.crop
    capture = Capture(ffmpeg_command(url, crop), crop.w, crop.h)
    capture.start()
    deadline = time.monotonic() + 20
    frame = None
    while time.monotonic() < deadline and frame is None:
        await asyncio.sleep(0.2)
        frame = capture.latest()
    capture.stop()
    if frame is None:
        print("camera: no frame within 20 s (check TAPO_RTSP_URL, the camera account, and the crop)")
        return 1
    jpeg = encode_jpeg(resize_max_side(frame.image, cfg.classifier.max_side_px))
    out.write_bytes(jpeg)
    print(f"camera: ok, saved {out}")
    for r in await classify_all(build_classifiers(cfg.classifier, os.environ), jpeg):
        answer = r.position.value if r.position else f"unavailable ({r.error})"
        print(f"{r.model_name}: {answer} in {r.latency_s:.1f}s")
    return 0


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crib-monitor-probe", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--out", default="probe.jpg")
    args = parser.parse_args(argv)
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
    raise SystemExit(asyncio.run(probe(Path(args.config), Path(args.out))))
